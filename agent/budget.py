"""
Budget tracker: token + cost tracking with hard budget caps.

- Tracks per-query AND total spend across the session.
- Raises BudgetExceededError when either cap is exceeded.
- Thread-safe via threading.Lock for parallel tool execution.

Design choice: hard rejection (not graceful degradation) because
the assignment spec explicitly says "rejects when exceeded".
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from agent.config import (
    BUDGET_CAP_PER_QUERY,
    BUDGET_CAP_TOTAL,
    get_pricing,
)


class BudgetExceededError(Exception):
    """Raised when a call would exceed the per-query or total budget cap."""
    pass


@dataclass
class UsageRecord:
    """Single API-call usage record."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    label: str = ""  # e.g. "planner", "executor-iter-2", "judge"


@dataclass
class BudgetTracker:
    """
    Tracks spend for a single agent run (one query).

    Call `record(model, input_tokens, output_tokens, label)` after every
    Anthropic API call. It will raise BudgetExceededError if the cost
    pushes either cap over its limit.
    """
    per_query_cap_usd: float = BUDGET_CAP_PER_QUERY
    total_cap_usd: float = BUDGET_CAP_TOTAL

    # State
    query_spend_usd: float = 0.0
    total_spend_usd: float = 0.0  # class-level via _global_total below
    records: list[UsageRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = get_pricing(model)
        return (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

    def preflight(self, model: str, estimated_input_tokens: int, estimated_output_tokens: int) -> None:
        """
        Check BEFORE making an API call whether it would exceed budget.
        Raises BudgetExceededError if it would.
        """
        projected_cost = self._compute_cost(model, estimated_input_tokens, estimated_output_tokens)
        with self._lock:
            if self.query_spend_usd + projected_cost > self.per_query_cap_usd:
                raise BudgetExceededError(
                    f"Per-query cap ${self.per_query_cap_usd:.4f} would be exceeded. "
                    f"Current spend: ${self.query_spend_usd:.6f}, "
                    f"projected call: ${projected_cost:.6f}"
                )
            if _global_total.value + projected_cost > self.total_cap_usd:
                raise BudgetExceededError(
                    f"Total session cap ${self.total_cap_usd:.4f} would be exceeded. "
                    f"Current total: ${_global_total.value:.6f}, "
                    f"projected call: ${projected_cost:.6f}"
                )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        label: str = "",
    ) -> UsageRecord:
        """
        Record usage AFTER an API call succeeds.
        Updates per-query and global totals. Raises if cap exceeded
        (post-hoc check; use preflight() for pre-check).
        """
        cost = self._compute_cost(model, input_tokens, output_tokens)
        with self._lock:
            self.query_spend_usd += cost
            _global_total.increment(cost)
            self.total_spend_usd = _global_total.value
            rec = UsageRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                label=label,
            )
            self.records.append(rec)

            if self.query_spend_usd > self.per_query_cap_usd:
                raise BudgetExceededError(
                    f"Per-query cap ${self.per_query_cap_usd:.4f} exceeded. "
                    f"Spent ${self.query_spend_usd:.6f} on this query."
                )
            if _global_total.value > self.total_cap_usd:
                raise BudgetExceededError(
                    f"Total session cap ${self.total_cap_usd:.4f} exceeded. "
                    f"Total spent: ${_global_total.value:.6f}"
                )
            return rec

    def summary(self) -> dict:
        with self._lock:
            return {
                "query_spend_usd": round(self.query_spend_usd, 6),
                "total_session_spend_usd": round(_global_total.value, 6),
                "num_api_calls": len(self.records),
                "breakdown_by_label": self._breakdown(),
            }

    def _breakdown(self) -> dict[str, dict]:
        by_label: dict[str, dict] = {}
        for r in self.records:
            key = r.label or "unlabeled"
            if key not in by_label:
                by_label[key] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            by_label[key]["calls"] += 1
            by_label[key]["input_tokens"] += r.input_tokens
            by_label[key]["output_tokens"] += r.output_tokens
            by_label[key]["cost_usd"] += r.cost_usd
        # Round for display
        for v in by_label.values():
            v["cost_usd"] = round(v["cost_usd"], 6)
        return by_label


class _GlobalTotal:
    """Thread-safe global session counter across all BudgetTracker instances."""
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    @property
    def value(self) -> float:
        with self._lock:
            return self._value

    def increment(self, amount: float) -> None:
        with self._lock:
            self._value += amount

    def reset(self) -> None:
        """For tests only."""
        with self._lock:
            self._value = 0.0


# Module-level global. Tracks total session spend across all agent runs.
_global_total = _GlobalTotal()


def get_global_spend() -> float:
    return _global_total.value


def reset_global_spend() -> None:
    """For testing/eval harness to reset between runs."""
    _global_total.reset()