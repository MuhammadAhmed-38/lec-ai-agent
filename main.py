"""
CLI entry point for the LEC AI agent.

Usage:
    python main.py "your query here"
    python main.py "query" --prompt-version v1
    python main.py "query" --json
    python main.py "query" --verbose

Exit codes:
    0 - agent ran successfully
    1 - agent ran but failed (budget, planner, tool exhaustion, etc.)
    2 - usage / argument error
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from agent.orchestrator import AgentResult, Orchestrator, build_default_registry


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party loggers unless verbose
    if not verbose:
        for name in ("httpx", "chromadb", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.ERROR)


def _print_human_readable(result: AgentResult) -> None:
    sep = "=" * 70
    print(sep)
    print(f"QUERY: {result.query}")
    print(sep)
    print()

    if not result.success:
        print("STATUS: FAILED")
        print(f"ERROR:  {result.error}")
        print()
        return

    # Plan
    if result.plan:
        print(f"PLAN ({len(result.plan.steps)} step(s), "
              f"{len(result.plan.parallel_groups)} group(s)):")
        print(f"  Reasoning: {result.plan.reasoning[:200]}"
              f"{'...' if len(result.plan.reasoning) > 200 else ''}")
        for s in result.plan.steps:
            deps = f" depends_on={s.depends_on}" if s.depends_on else ""
            print(f"  [{s.step_id}] {s.tool}({s.arguments}){deps}")
        print(f"  Parallel groups: {result.plan.parallel_groups}")
        print()

    # Execution trace
    if result.trace:
        print("EXECUTION:")
        for e in result.trace.executions:
            status = "OK" if e.result.success else f"FAIL ({e.result.error[:80]})"
            retry_note = f" [retried {e.retry_count}x]" if e.retry_count else ""
            print(f"  [{e.step.step_id}] {e.step.tool} -> {status}"
                  f" ({e.result.latency_ms:.0f}ms){retry_note}")
        print()

    # Final answer
    print("FINAL ANSWER:")
    print(result.final_answer)
    print()

    # Metrics footer
    b = result.budget_summary
    print(sep)
    print(f"METRICS: {result.total_wall_time_s:.2f}s wall time | "
          f"{b.get('num_api_calls', 0)} LLM calls | "
          f"${b.get('query_spend_usd', 0):.6f} query spend | "
          f"{result.iterations_used} iteration(s)")
    print(sep)


def _print_json(result: AgentResult) -> None:
    print(json.dumps(result.to_dict(), indent=2, default=_json_default, ensure_ascii=False))


def _json_default(obj: Any) -> Any:
    """Fallback for non-JSON-serialisable objects (e.g., dataclasses not converted)."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lec-ai-agent",
        description="Run a query through the LEC AI production agent.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="The user query to run through the agent.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Which prompt version to use (default: v2).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the full result as JSON (useful for eval harnesses).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show INFO-level logs from agent and tools.",
    )
    return parser.parse_args()


async def _amain() -> int:
    args = _parse_args()
    _setup_logging(args.verbose)

    registry = build_default_registry()
    orchestrator = Orchestrator(
        registry=registry,
        prompt_version=args.prompt_version,
    )
    result = await orchestrator.run(args.query, reset_global=True)

    if args.json:
        _print_json(result)
    else:
        _print_human_readable(result)

    return 0 if result.success else 1


def main() -> None:
    try:
        exit_code = asyncio.run(_amain())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()