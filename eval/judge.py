"""
LLM-as-judge for grading agent answers.

Why an LLM judge:
  - Many queries have non-deterministic correct answers (prices, natural
    language explanations, acceptable ranges).
  - Simple substring matching is brittle: "The capital of France is Paris"
    and "Paris is France's capital" both pass, but substring check for
    "is Paris" would fail one of them.
  - The judge can apply the nuanced rubric encoded in EvalQuery fields.

Why Haiku for the judge:
  - Grading is a classification task, not open-ended reasoning.
  - 10x cheaper than Sonnet, fast, deterministic at temperature 0.
  - Bias risk from self-grading is mitigated because the agent uses
    Sonnet for planning and Haiku for execution — judge uses a separate
    Haiku call with a narrowly-scoped system prompt.

Output is a structured Judgment dataclass with:
  - passed: bool              (binary pass/fail on the core assertion)
  - score: float              (0.0 - 1.0, allows partial credit)
  - reasoning: str            (why the judge decided this way)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from agent.budget import BudgetTracker
from agent.config import ANTHROPIC_API_KEY, JUDGE_MODEL, get_pricing
from eval.queries import EvalQuery

logger = logging.getLogger(__name__)


# ==========================================================
# DATA CLASSES
# ==========================================================

@dataclass
class Judgment:
    query_id: str
    passed: bool
    score: float            # 0.0 - 1.0
    reasoning: str
    raw_judge_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "passed": self.passed,
            "score": round(self.score, 3),
            "reasoning": self.reasoning,
        }


# ==========================================================
# JUDGE SYSTEM PROMPT
# ==========================================================

JUDGE_SYSTEM_PROMPT = """\
You are grading an AI agent's answer to a test query.

You will be given:
  1. The query the agent was asked
  2. The rubric (required content, forbidden content, capability notes)
  3. The agent's final answer

Apply the rubric strictly but fairly:
  - If the required content is present (even if phrased differently), pass.
  - If forbidden content appears (e.g., hallucinated facts), fail.
  - If the answer is partially correct (some required content present, \
some missing), use the score field to reflect partial credit (0.5-0.8).
  - If the agent honestly reports it couldn't find something, and the \
query's rubric allows for that (e.g., "agent should acknowledge failure"), \
pass that portion.

Return ONLY a JSON object matching this schema:
{
  "passed": true|false,
  "score": 0.0-1.0,
  "reasoning": "<1-3 sentence explanation of the grade>"
}

No markdown fences, no preamble, no trailing text. Begin with '{'.\
"""


# ==========================================================
# JUDGE
# ==========================================================

class Judge:
    """Grades agent answers using an LLM."""

    def __init__(
        self,
        budget: BudgetTracker | None = None,
        model: str = JUDGE_MODEL,
    ) -> None:
        self.model = model
        self.budget = budget  # optional — judge cost is tracked separately from agent budget
        self._client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def grade(
        self,
        query: EvalQuery,
        agent_answer: str,
    ) -> Judgment:
        """Grade a single (query, answer) pair."""
        user_message = self._build_user_message(query, agent_answer)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            temperature=0.0,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = "".join(
            b.text for b in response.content if getattr(b, "type", "") == "text"
        )

        # Track budget if a tracker was supplied
        if self.budget is not None:
            self.budget.record(
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                label="judge",
            )

        return self._parse_judgment(query.id, raw)

    def _build_user_message(self, query: EvalQuery, agent_answer: str) -> str:
        rubric_lines = [f"Capability under test: {query.capability}"]
        if query.expected_answer_contains:
            rubric_lines.append(
                "REQUIRED content (at least one substring-or-equivalent must appear):\n  - "
                + "\n  - ".join(query.expected_answer_contains)
            )
        if query.must_not_contain:
            rubric_lines.append(
                "FORBIDDEN content (fail if any of these appear):\n  - "
                + "\n  - ".join(query.must_not_contain)
            )
        if query.notes:
            rubric_lines.append(f"Judge notes:\n{query.notes}")

        rubric = "\n\n".join(rubric_lines)

        return (
            f"Query:\n{query.query}\n\n"
            f"Rubric:\n{rubric}\n\n"
            f"Agent's final answer:\n{agent_answer}\n\n"
            f"Grade the answer. Return JSON only."
        )

    def _parse_judgment(self, query_id: str, raw: str) -> Judgment:
        """Parse JSON judge output; repair if needed."""
        cleaned = raw.strip()
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        # Extract outermost JSON object
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last > first:
            cleaned = cleaned[first : last + 1]

        try:
            data = json.loads(cleaned)
            passed = bool(data.get("passed", False))
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", "")).strip()
            # Clamp score
            score = max(0.0, min(1.0, score))
            return Judgment(
                query_id=query_id,
                passed=passed,
                score=score,
                reasoning=reasoning,
                raw_judge_output=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Judge output unparseable for {query_id}: {e}. Raw: {raw[:200]!r}")
            return Judgment(
                query_id=query_id,
                passed=False,
                score=0.0,
                reasoning=f"[JUDGE PARSE ERROR] {e}",
                raw_judge_output=raw,
            )