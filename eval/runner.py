"""
Evaluation runner: runs all queries through the agent and grades them.

Produces:
  - Per-query results (plan, trace, answer, judgment)
  - Aggregate metrics (success rate, score distribution, capability breakdown)
  - JSON file in runs/ for reproducibility

Usage:
    python -m eval.runner --prompt-version v2
    python -m eval.runner --prompt-version v1
    python -m eval.runner --prompt-version v2 --queries Q1 Q4 Q7

The resulting JSON is the raw material for the ablation table in the report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.budget import BudgetTracker, reset_global_spend
from agent.config import RUNS_DIR
from agent.orchestrator import AgentResult, Orchestrator, build_default_registry
from eval.judge import Judge, Judgment
from eval.queries import EvalQuery, EVAL_QUERIES, get_all_queries, get_query_by_id

logger = logging.getLogger(__name__)


# ==========================================================
# RESULT STRUCTURES
# ==========================================================

@dataclass
class PerQueryResult:
    query_id: str
    query: str
    capability: str
    agent_success: bool           # did the agent run complete without error?
    judge_passed: bool            # did the answer pass the rubric?
    judge_score: float
    agent_answer: str
    judge_reasoning: str
    wall_time_s: float
    agent_cost_usd: float
    num_api_calls: int
    num_tool_calls: int
    num_tool_failures: int
    tools_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRunSummary:
    prompt_version: str
    timestamp: str
    total_queries: int
    passed: int
    failed: int
    success_rate: float            # passed / total
    mean_score: float
    total_wall_time_s: float
    total_agent_cost_usd: float
    total_judge_cost_usd: float
    per_capability: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[PerQueryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ==========================================================
# RUNNER
# ==========================================================

class EvalRunner:
    def __init__(
        self,
        prompt_version: str = "v2",
        verbose: bool = False,
    ) -> None:
        self.prompt_version = prompt_version
        self.verbose = verbose
        self.registry = build_default_registry()
        self.orchestrator = Orchestrator(
            registry=self.registry,
            prompt_version=prompt_version,
        )
        # Dedicated budget tracker for judge calls so we don't pollute agent budget
        self._judge_budget = BudgetTracker(
            per_query_cap_usd=1.0,   # judge calls are cheap; loose cap
            total_cap_usd=5.0,
        )
        self.judge = Judge(budget=self._judge_budget)

    async def run_all(
        self,
        queries: list[EvalQuery] | None = None,
    ) -> EvalRunSummary:
        queries = queries if queries is not None else get_all_queries()
        results: list[PerQueryResult] = []

        reset_global_spend()
        run_start = time.perf_counter()

        for i, q in enumerate(queries, start=1):
            print(f"\n[{i}/{len(queries)}] {q.id} ({q.capability})")
            print(f"  Query: {q.query[:80]}{'...' if len(q.query) > 80 else ''}")

            per = await self._run_single(q)
            results.append(per)

            status = "PASS" if per.judge_passed else "FAIL"
            print(f"  -> {status} (score={per.judge_score:.2f}) "
                  f"in {per.wall_time_s:.1f}s, ${per.agent_cost_usd:.4f}")
            if not per.judge_passed:
                print(f"     Judge: {per.judge_reasoning[:180]}")

        total_wall = time.perf_counter() - run_start
        summary = self._build_summary(
            results=results,
            total_wall=total_wall,
        )
        self._save_run(summary)
        return summary

    async def _run_single(self, q: EvalQuery) -> PerQueryResult:
        result: AgentResult = await self.orchestrator.run(q.query, reset_global=False)

        # Extract tool usage info from trace
        tools_used: list[str] = []
        tool_failures = 0
        num_tool_calls = 0
        if result.trace is not None:
            num_tool_calls = len(result.trace.executions)
            for ex in result.trace.executions:
                tools_used.append(ex.step.tool)
                if not ex.result.success:
                    tool_failures += 1

        # Grade the answer (even if the agent failed — the judge decides)
        try:
            judgment: Judgment = self.judge.grade(q, result.final_answer or "")
        except Exception as e:
            logger.error(f"Judge crashed on {q.id}: {e}")
            judgment = Judgment(
                query_id=q.id,
                passed=False,
                score=0.0,
                reasoning=f"[JUDGE CRASH] {type(e).__name__}: {e}",
            )

        b = result.budget_summary
        return PerQueryResult(
            query_id=q.id,
            query=q.query,
            capability=q.capability,
            agent_success=result.success,
            judge_passed=judgment.passed,
            judge_score=judgment.score,
            agent_answer=result.final_answer,
            judge_reasoning=judgment.reasoning,
            wall_time_s=result.total_wall_time_s,
            agent_cost_usd=b.get("query_spend_usd", 0.0),
            num_api_calls=b.get("num_api_calls", 0),
            num_tool_calls=num_tool_calls,
            num_tool_failures=tool_failures,
            tools_used=tools_used,
        )

    def _build_summary(
        self,
        results: list[PerQueryResult],
        total_wall: float,
    ) -> EvalRunSummary:
        passed = sum(1 for r in results if r.judge_passed)
        n = len(results)
        success_rate = passed / n if n else 0.0
        mean_score = sum(r.judge_score for r in results) / n if n else 0.0
        total_agent_cost = sum(r.agent_cost_usd for r in results)

        # Per-capability breakdown
        per_cap: dict[str, dict[str, float]] = {}
        for r in results:
            cap = r.capability
            if cap not in per_cap:
                per_cap[cap] = {"n": 0, "passed": 0, "score_sum": 0.0}
            per_cap[cap]["n"] += 1
            per_cap[cap]["passed"] += int(r.judge_passed)
            per_cap[cap]["score_sum"] += r.judge_score
        for cap, stats in per_cap.items():
            stats["success_rate"] = stats["passed"] / stats["n"] if stats["n"] else 0.0
            stats["mean_score"] = stats["score_sum"] / stats["n"] if stats["n"] else 0.0

        return EvalRunSummary(
            prompt_version=self.prompt_version,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            total_queries=n,
            passed=passed,
            failed=n - passed,
            success_rate=success_rate,
            mean_score=mean_score,
            total_wall_time_s=total_wall,
            total_agent_cost_usd=total_agent_cost,
            total_judge_cost_usd=self._judge_budget.query_spend_usd,
            per_capability=per_cap,
            results=results,
        )

    def _save_run(self, summary: EvalRunSummary) -> Path:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        ts = summary.timestamp.replace(":", "-")
        filename = f"eval_{summary.prompt_version}_{ts}.json"
        path = RUNS_DIR / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved run to: {path}")
        return path


# ==========================================================
# SUMMARY PRINTING
# ==========================================================

def print_summary(summary: EvalRunSummary) -> None:
    sep = "=" * 72
    print()
    print(sep)
    print(f"EVAL RUN SUMMARY — prompt_version={summary.prompt_version}")
    print(sep)
    print(f"Passed:         {summary.passed}/{summary.total_queries}")
    print(f"Success rate:   {summary.success_rate * 100:.1f}%")
    print(f"Mean score:     {summary.mean_score:.3f}")
    print(f"Wall time:      {summary.total_wall_time_s:.1f}s total")
    print(f"Agent cost:     ${summary.total_agent_cost_usd:.4f}")
    print(f"Judge cost:     ${summary.total_judge_cost_usd:.4f}")
    print()

    print("Per-query:")
    for r in summary.results:
        mark = "PASS" if r.judge_passed else "FAIL"
        print(f"  {r.query_id}  [{mark}]  score={r.judge_score:.2f}  "
              f"tools={len(r.tools_used)} ({r.num_tool_failures} failed)  "
              f"${r.agent_cost_usd:.4f}")
    print()

    print("Per-capability:")
    for cap, stats in summary.per_capability.items():
        print(f"  {stats['passed']}/{int(stats['n'])}  "
              f"({stats['success_rate'] * 100:.0f}%)  "
              f"mean={stats['mean_score']:.2f}  —  {cap}")
    print(sep)


# ==========================================================
# CLI
# ==========================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval.runner")
    p.add_argument(
        "--prompt-version",
        default="v2",
        choices=["v1", "v2"],
        help="Which prompt version to evaluate (default v2).",
    )
    p.add_argument(
        "--queries",
        nargs="*",
        help="Optional: list of specific query IDs (e.g. Q1 Q4 Q7). Default: all.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    return p.parse_args()


async def _amain() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        for name in ("httpx", "chromadb", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)

    queries = None
    if args.queries:
        queries = [get_query_by_id(qid) for qid in args.queries]

    runner = EvalRunner(prompt_version=args.prompt_version, verbose=args.verbose)
    summary = await runner.run_all(queries=queries)
    print_summary(summary)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()