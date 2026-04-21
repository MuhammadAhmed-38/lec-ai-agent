"""
Orchestrator: top-level agent entry point.

Coordinates the plan → execute → synthesise flow, handles budget
lifecycle, enforces iteration caps, and produces a single structured
result that callers (CLI, eval harness) can consume.

Single-iteration model for now: plan once, execute once, synthesise once.
The MAX_ITERATIONS guard is in place for a future reflection loop where
the agent might re-plan after a failed execution; the hook is there but
the current implementation does not re-plan.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent.budget import BudgetExceededError, BudgetTracker, reset_global_spend
from agent.config import MAX_ITERATIONS
from agent.executor import ExecutionTrace, Executor
from agent.planner import Plan, Planner, PlannerError
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Everything a caller needs to know about a single agent run."""
    query: str
    success: bool
    final_answer: str
    plan: Plan | None = None
    trace: ExecutionTrace | None = None
    error: str = ""
    prompt_version: str = "v2"
    iterations_used: int = 0
    total_wall_time_s: float = 0.0
    budget_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "success": self.success,
            "final_answer": self.final_answer,
            "error": self.error,
            "prompt_version": self.prompt_version,
            "iterations_used": self.iterations_used,
            "total_wall_time_s": round(self.total_wall_time_s, 3),
            "budget_summary": self.budget_summary,
            "plan": self.plan.to_dict() if self.plan else None,
            "trace": self.trace.to_dict() if self.trace else None,
        }


class Orchestrator:
    """
    High-level agent entry point. Owns a ToolRegistry and produces
    AgentResults for queries.

    Usage:
        orch = Orchestrator(registry, prompt_version="v2")
        result = await orch.run("What is ...?")
        print(result.final_answer)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        prompt_version: str = "v2",
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.registry = registry
        self.prompt_version = prompt_version
        self.max_iterations = max_iterations

    async def run(self, query: str, reset_global: bool = False) -> AgentResult:
        """
        Run the agent end-to-end for a single query.

        Captures all exceptions and returns them as structured
        AgentResult(success=False, error=...) so eval harnesses don't crash.
        """
        if reset_global:
            reset_global_spend()

        budget = BudgetTracker()
        planner = Planner(
            registry=self.registry,
            budget=budget,
            prompt_version=self.prompt_version,
        )
        executor = Executor(
            registry=self.registry,
            budget=budget,
            prompt_version=self.prompt_version,
        )

        start = time.perf_counter()
        iterations = 0
        plan: Plan | None = None
        trace: ExecutionTrace | None = None

        try:
            # Iteration cap: currently we only plan+execute once, but the
            # structure supports re-planning on failure in future work.
            while iterations < self.max_iterations:
                iterations += 1
                logger.info(f"[iter {iterations}] planning...")
                plan = planner.plan(query)

                logger.info(f"[iter {iterations}] executing {len(plan.steps)} steps...")
                trace = await executor.execute(query, plan)

                # For now we always stop after one successful iteration.
                # A future reflection step could inspect the trace and
                # decide to re-plan.
                break

            elapsed = time.perf_counter() - start
            return AgentResult(
                query=query,
                success=True,
                final_answer=trace.final_answer if trace else "",
                plan=plan,
                trace=trace,
                prompt_version=self.prompt_version,
                iterations_used=iterations,
                total_wall_time_s=elapsed,
                budget_summary=budget.summary(),
            )

        except BudgetExceededError as e:
            elapsed = time.perf_counter() - start
            logger.warning(f"Budget exceeded: {e}")
            return AgentResult(
                query=query,
                success=False,
                final_answer="",
                plan=plan,
                trace=trace,
                error=f"BudgetExceeded: {e}",
                prompt_version=self.prompt_version,
                iterations_used=iterations,
                total_wall_time_s=elapsed,
                budget_summary=budget.summary(),
            )

        except PlannerError as e:
            elapsed = time.perf_counter() - start
            logger.error(f"Planner failed: {e}")
            return AgentResult(
                query=query,
                success=False,
                final_answer="",
                plan=plan,
                trace=trace,
                error=f"PlannerError: {e}",
                prompt_version=self.prompt_version,
                iterations_used=iterations,
                total_wall_time_s=elapsed,
                budget_summary=budget.summary(),
            )

        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.exception(f"Orchestrator caught unexpected error: {e}")
            return AgentResult(
                query=query,
                success=False,
                final_answer="",
                plan=plan,
                trace=trace,
                error=f"{type(e).__name__}: {e}",
                prompt_version=self.prompt_version,
                iterations_used=iterations,
                total_wall_time_s=elapsed,
                budget_summary=budget.summary(),
            )


def build_default_registry() -> ToolRegistry:
    """
    Factory: registers all 5 standard tools.
    Separated so eval harness can reuse (or swap for test-mode registry).
    """
    from tools.calculator import CalculatorTool
    from tools.code_executor import CodeExecutorTool
    from tools.document_qa import DocumentQATool
    from tools.knowledge_base import KnowledgeBaseTool
    from tools.web_search import WebSearchTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(KnowledgeBaseTool())
    registry.register(CodeExecutorTool())
    registry.register(DocumentQATool())
    return registry