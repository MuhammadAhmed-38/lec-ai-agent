"""
Planner: converts a user query into a structured execution plan.

Uses Sonnet (higher-quality model) because planning quality dominates
overall agent success. A bad plan cannot be recovered by a good executor.

The plan is a JSON document with:
  - reasoning: chain-of-thought from the planner
  - steps: list of tool calls with arguments and dependencies
  - parallel_groups: which steps can run concurrently

Robustness:
  - JSON repair for common LLM output issues (trailing commas, fences)
  - One retry with a stricter prompt on parse failure
  - Validation that referenced tools exist
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from agent.budget import BudgetTracker
from agent.config import ANTHROPIC_API_KEY, PLANNING_MODEL
from agent.prompts import get_prompts
from tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class PlanStep:
    step_id: int
    tool: str
    arguments: dict[str, Any]
    depends_on: list[int] = field(default_factory=list)
    rationale: str = ""


@dataclass
class Plan:
    reasoning: str
    steps: list[PlanStep]
    parallel_groups: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "steps": [
                {
                    "step_id": s.step_id,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "depends_on": s.depends_on,
                    "rationale": s.rationale,
                }
                for s in self.steps
            ],
            "parallel_groups": self.parallel_groups,
        }


class PlannerError(Exception):
    """Raised when planning fails unrecoverably."""


# ==========================================================
# JSON REPAIR (inspired by lessons from structured-output work)
# ==========================================================

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _repair_json(raw: str) -> str:
    """
    Best-effort repair of common LLM-JSON mishaps:
      - Markdown code fences
      - Leading/trailing commentary text
      - Trailing commas before ] or }
    """
    s = raw.strip()

    # Strip markdown fences
    s = _FENCE_RE.sub("", s).strip()

    # Extract the first JSON object if there's surrounding prose
    first_brace = s.find("{")
    last_brace = s.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        s = s[first_brace : last_brace + 1]

    # Remove trailing commas: ", ]" or ", }"
    s = re.sub(r",\s*([\]\}])", r"\1", s)

    return s


def _parse_plan_json(raw: str) -> dict[str, Any]:
    """Try parsing raw; if it fails, repair and try once more."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise PlannerError(
            f"Could not parse plan JSON even after repair. "
            f"Error: {e}. Raw output (first 300 chars): {raw[:300]!r}"
        )


# ==========================================================
# PLAN VALIDATION
# ==========================================================

def _validate_plan(data: dict[str, Any], registry: ToolRegistry) -> Plan:
    """Convert raw dict into a Plan, raising PlannerError on structural issues."""
    if not isinstance(data, dict):
        raise PlannerError(f"Plan must be an object, got {type(data).__name__}")

    reasoning = str(data.get("reasoning", "")).strip()
    raw_steps = data.get("steps", [])
    raw_groups = data.get("parallel_groups", [])

    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlannerError("Plan must contain a non-empty 'steps' array")

    steps: list[PlanStep] = []
    seen_ids: set[int] = set()
    for raw in raw_steps:
        if not isinstance(raw, dict):
            raise PlannerError(f"Step must be an object, got: {raw!r}")
        try:
            step_id = int(raw["step_id"])
            tool = str(raw["tool"]).strip()
            arguments = raw.get("arguments") or {}
            depends_on = [int(x) for x in raw.get("depends_on", [])]
            rationale = str(raw.get("rationale", "")).strip()
        except (KeyError, TypeError, ValueError) as e:
            raise PlannerError(f"Malformed step {raw!r}: {e}")

        if step_id in seen_ids:
            raise PlannerError(f"Duplicate step_id: {step_id}")
        seen_ids.add(step_id)

        if tool not in registry:
            raise PlannerError(
                f"Step {step_id} references unknown tool '{tool}'. "
                f"Available: {[t.name for t in registry.all()]}"
            )
        if not isinstance(arguments, dict):
            raise PlannerError(f"Step {step_id} 'arguments' must be an object")

        steps.append(PlanStep(
            step_id=step_id,
            tool=tool,
            arguments=arguments,
            depends_on=depends_on,
            rationale=rationale,
        ))

    # Validate parallel_groups or synthesize a trivial one
    if not isinstance(raw_groups, list):
        raw_groups = []

    # Ensure every step appears exactly once across the groups
    all_ids_in_groups: list[int] = []
    validated_groups: list[list[int]] = []
    for g in raw_groups:
        if not isinstance(g, list):
            continue
        validated_groups.append([int(x) for x in g])
        all_ids_in_groups.extend(int(x) for x in g)

    if sorted(all_ids_in_groups) != sorted(seen_ids):
        # Fall back to a safe sequential grouping
        logger.warning(
            "parallel_groups missing or malformed; defaulting to fully sequential."
        )
        validated_groups = [[s.step_id] for s in steps]

    return Plan(reasoning=reasoning, steps=steps, parallel_groups=validated_groups)


# ==========================================================
# PLANNER
# ==========================================================

class Planner:
    """
    Given a user query and a tool registry, produces a validated Plan.
    Uses Anthropic's Claude for generation.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        budget: BudgetTracker,
        prompt_version: str = "v2",
        model: str = PLANNING_MODEL,
    ) -> None:
        self.registry = registry
        self.budget = budget
        self.model = model
        self.prompt_version = prompt_version
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self._system_prompt = get_prompts(prompt_version)["planner"]

    def plan(self, query: str, retry_on_parse_fail: bool = True) -> Plan:
        """Generate a plan. Raises PlannerError if unrecoverable."""
        user_message = self._build_user_message(query)

        raw_output, usage = self._call_llm(user_message)
        self.budget.record(
            model=self.model,
            input_tokens=usage["input"],
            output_tokens=usage["output"],
            label="planner",
        )

        try:
            data = _parse_plan_json(raw_output)
            return _validate_plan(data, self.registry)
        except PlannerError as e:
            if not retry_on_parse_fail:
                raise
            logger.warning(f"First plan attempt failed ({e}); retrying with stricter prompt")
            stricter = (
                user_message
                + "\n\nYour previous response was not valid JSON matching the schema. "
                "Return ONLY the JSON object. No prose, no markdown fences, no commentary. "
                "Begin your response with '{' and end with '}'."
            )
            raw_output, usage = self._call_llm(stricter)
            self.budget.record(
                model=self.model,
                input_tokens=usage["input"],
                output_tokens=usage["output"],
                label="planner-retry",
            )
            data = _parse_plan_json(raw_output)
            return _validate_plan(data, self.registry)

    def _build_user_message(self, query: str) -> str:
        import json as _json
        tool_blocks = []
        for t in self.registry.all():
            schema_json = _json.dumps(t.input_schema, indent=2)
            tool_blocks.append(
                f"### Tool: {t.name}\n"
                f"Description: {t.description}\n"
                f"Input schema (USE THESE EXACT ARGUMENT NAMES):\n{schema_json}"
            )
        tool_section = "\n\n".join(tool_blocks)
        return (
            f"User query:\n{query}\n\n"
            f"Available tools:\n\n{tool_section}\n\n"
            f"IMPORTANT: In each step's 'arguments' field, use the EXACT "
            f"parameter names from the input_schema above. Do not invent "
            f"new parameter names.\n\n"
            f"Produce the plan as JSON."
        )

    def _call_llm(self, user_message: str) -> tuple[str, dict[str, int]]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.2,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
        return text, usage