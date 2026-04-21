"""
Executor: runs a validated Plan against a ToolRegistry.

Execution model:
  - Plan is processed group by group (from plan.parallel_groups).
  - Steps in the same group run concurrently via asyncio.gather.
  - Each step's arguments may reference prior steps' outputs via the
    `{{step_N.output}}` placeholder syntax — substituted before execution.
  - Tool failures are captured (not raised). The executor can retry once
    per step with a slightly relaxed arg (delegated to the synthesis LLM
    for complex recovery).
  - After all steps run, a final synthesis call to Haiku turns the
    observation trail into a user-facing answer.

Termination / loop prevention:
  - Plans have finite steps (no dynamic extension), so infinite loops
    are structurally impossible within a single plan.
  - The orchestrator enforces MAX_ITERATIONS across plan-reflect-replan
    cycles (see orchestrator.py). This executor handles one plan.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from agent.budget import BudgetTracker
from agent.config import ANTHROPIC_API_KEY, EXECUTION_MODEL, MAX_PARALLEL_TOOLS
from agent.planner import Plan, PlanStep
from agent.prompts import get_prompts
from tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class StepExecution:
    """Record of a single step's execution."""
    step: PlanStep
    result: ToolResult
    retry_count: int = 0


@dataclass
class ExecutionTrace:
    """Complete record of executing a Plan."""
    plan: Plan
    executions: list[StepExecution] = field(default_factory=list)
    final_answer: str = ""
    synthesis_input_tokens: int = 0
    synthesis_output_tokens: int = 0

    def success_count(self) -> int:
        return sum(1 for e in self.executions if e.result.success)

    def total_tool_latency_ms(self) -> float:
        return sum(e.result.latency_ms for e in self.executions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "executions": [
                {
                    "step_id": e.step.step_id,
                    "tool": e.step.tool,
                    "arguments": e.step.arguments,
                    "success": e.result.success,
                    "output": (
                        e.result.output
                        if e.result.success
                        else None
                    ),
                    "error": e.result.error if not e.result.success else "",
                    "latency_ms": round(e.result.latency_ms, 1),
                    "retry_count": e.retry_count,
                }
                for e in self.executions
            ],
            "final_answer": self.final_answer,
            "metrics": {
                "steps_total": len(self.executions),
                "steps_succeeded": self.success_count(),
                "total_tool_latency_ms": round(self.total_tool_latency_ms(), 1),
            },
        }


class ExecutorError(Exception):
    """Raised when execution fails in an unrecoverable way."""


# ==========================================================
# PLACEHOLDER SUBSTITUTION
# ==========================================================

_PLACEHOLDER_RE = re.compile(r"\{\{\s*step_(\d+)\.output\s*\}\}")


def _substitute_placeholders(
    arguments: dict[str, Any],
    completed: dict[int, ToolResult],
) -> dict[str, Any]:
    """
    Replace {{step_N.output}} placeholders inside string arg values
    with the stringified output of step N. Non-string args pass through.
    """
    def _sub_value(v: Any) -> Any:
        if isinstance(v, str):
            def _repl(match: re.Match) -> str:
                step_id = int(match.group(1))
                if step_id not in completed:
                    return match.group(0)  # leave unchanged — executor logs a warning
                prev = completed[step_id]
                if prev.success:
                    return str(prev.output)
                return f"[error from step_{step_id}: {prev.error}]"
            return _PLACEHOLDER_RE.sub(_repl, v)
        if isinstance(v, dict):
            return {k: _sub_value(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_sub_value(x) for x in v]
        return v

    return {k: _sub_value(v) for k, v in arguments.items()}


# ==========================================================
# EXECUTOR
# ==========================================================

class Executor:
    """Runs a Plan and synthesises a final answer."""

    def __init__(
        self,
        registry: ToolRegistry,
        budget: BudgetTracker,
        prompt_version: str = "v2",
        model: str = EXECUTION_MODEL,
        max_parallel: int = MAX_PARALLEL_TOOLS,
    ) -> None:
        self.registry = registry
        self.budget = budget
        self.model = model
        self.prompt_version = prompt_version
        self.max_parallel = max_parallel
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self._system_prompt = get_prompts(prompt_version)["executor"]

    async def execute(self, query: str, plan: Plan) -> ExecutionTrace:
        """
        Run the plan end-to-end and return the trace.
        Raises ExecutorError only on truly unrecoverable issues;
        individual tool failures are captured in the trace.
        """
        trace = ExecutionTrace(plan=plan)
        completed: dict[int, ToolResult] = {}

        # Build a step lookup for convenience
        step_lookup = {s.step_id: s for s in plan.steps}

        # Execute each parallel group in order
        for group_idx, group in enumerate(plan.parallel_groups):
            # Validate the group contains known step_ids
            steps_in_group = [step_lookup[sid] for sid in group if sid in step_lookup]
            if not steps_in_group:
                continue

            # Cap concurrency for safety — don't fire 20 tools at once
            semaphore = asyncio.Semaphore(self.max_parallel)

            async def _run_step(step: PlanStep) -> tuple[PlanStep, ToolResult]:
                async with semaphore:
                    resolved_args = _substitute_placeholders(step.arguments, completed)
                    tool = self.registry.get(step.tool)
                    logger.info(
                        f"[group={group_idx}] Step {step.step_id} -> {step.tool}"
                    )
                    result = await tool.execute(**resolved_args)
                    return step, result

            # Run all steps in this group concurrently
            group_results = await asyncio.gather(
                *(_run_step(s) for s in steps_in_group),
                return_exceptions=False,  # Tool.execute never raises
            )

            # Record results (+ one retry if failed, with same args)
            for step, result in group_results:
                retry_count = 0
                if not result.success:
                    logger.warning(
                        f"Step {step.step_id} ({step.tool}) failed: {result.error}. Retrying once."
                    )
                    retry_count = 1
                    resolved_args = _substitute_placeholders(step.arguments, completed)
                    tool = self.registry.get(step.tool)
                    result = await tool.execute(**resolved_args)

                trace.executions.append(StepExecution(
                    step=step,
                    result=result,
                    retry_count=retry_count,
                ))
                completed[step.step_id] = result

        # All steps done. Synthesise final answer.
        trace.final_answer = await self._synthesize(query, plan, trace.executions)
        return trace

    async def _synthesize(
        self,
        query: str,
        plan: Plan,
        executions: list[StepExecution],
    ) -> str:
        """Call Haiku with observations to produce the final user-facing answer."""
        observations = self._format_observations(executions)
        user_message = (
            f"User query:\n{query}\n\n"
            f"Plan reasoning (from planner):\n{plan.reasoning}\n\n"
            f"Tool observations:\n{observations}\n\n"
            f"Produce the final answer to the user query. "
            f"Be concise and direct. If any step failed and you cannot "
            f"recover the information, say so clearly."
        )

        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        self.budget.record(
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            label="executor-synthesis",
        )
        return text.strip()

    @staticmethod
    def _format_observations(executions: list[StepExecution]) -> str:
        if not executions:
            return "(no steps were executed)"
        lines = []
        for e in executions:
            header = (
                f"[step {e.step.step_id}] tool={e.step.tool} "
                f"args={json.dumps(e.step.arguments, ensure_ascii=False)} "
                f"success={e.result.success} retries={e.retry_count}"
            )
            body = e.result.output if e.result.success else f"ERROR: {e.result.error}"
            lines.append(header)
            lines.append(str(body))
            lines.append("")
        return "\n".join(lines).strip()