# Roadmap — Next Week

Five features I would ship given one more week, in priority order. Each
is grounded in an observed limitation from the current eval run or the
development debugging log, not generic wishlist items.

---

## 1. Reflection loop — detect unresolved sub-questions after execution

**Motivation.** Q8 ("CEO of Apple AND iPhone release year") failed in
both v1 and v2 with score 0.5. The planner committed to one KB lookup
(`companies.apple`), saw "iPhone" in the notable-products list, and
never queried `historical_events.iphone_release_year`. The executor
had no mechanism to notice that one of the query's sub-questions was
unanswered. Current `MAX_ITERATIONS` hook exists but is never exercised.

**What I'd build.** After the executor's synthesis step, add a
reflection call (Haiku, `temperature=0`):

1. Take the original query, the plan, and the final answer.
2. Ask: "List any sub-questions in the original query that the final
   answer does not address with concrete data from tool observations.
   Return `[]` if all sub-questions are answered."
3. If non-empty, pass the unresolved sub-questions back to the planner
   as additional context: "Previous plan missed: [...]. Generate a new
   plan that specifically addresses these."
4. Re-execute. Cap at `MAX_ITERATIONS=3` to prevent runaway loops.

**Effort:** ~1 day. Mostly plumbing — reflection call, re-planning
prompt, iteration cap enforcement in `Orchestrator.run()`.

**Expected impact.** Would flip Q8 to pass (fact exists in KB, planner
just needs a second attempt with feedback). Estimated +5-10pp on
overall success rate for multi-hop queries.

---

## 2. Multi-run evaluation with variance reporting

**Motivation.** Q10 showed different outcomes across smoke test and
full run — same query, same prompt, different answer. Single-shot
evaluation gives a point estimate, but LLM agents are stochastic
systems and should be evaluated as distributions.

**What I'd build.** Extend `eval/runner.py` to support `--repeats N`
(default 5). For each query, run N times and report:

- p5 / p50 / p95 scores
- Pass rate (fraction of runs passing)
- Variance in tool selection (did the planner choose different tools
  across runs?)
- Variance in final answer length and content

Aggregate metrics change from "success rate = X%" to "success rate =
X% (p5 Y%, p95 Z%)". The eval summary table grows a variance column.

**Effort:** ~4 hours. Runner change is small; the interesting work is
designing the right variance metrics without blowing the budget
(5× cost).

**Expected impact.** Reveals which queries are reliable (tight
distribution, high floor) vs. which are fragile (wide distribution,
scary p5). Currently we cannot distinguish "90% always" from "90%
sometimes 70% sometimes." Production-readiness needs the distinction.

---

## 3. Formal tool output contract — typed returns, not display strings

**Motivation.** The day-one bug where the knowledge base returned
`'countries.france.population' = 67.97` as a formatted string, and
the calculator tool couldn't consume it. The fix (dual output modes
on the KB tool) is a workaround, not a solution. Any tool returning
human-readable formatted output will recreate this brittleness.

**What I'd build.** Add a `ToolOutput` dataclass that every tool
returns:

```python
@dataclass
class ToolOutput:
    value: Any              # raw, typed — for programmatic consumption
    display: str            # formatted human/LLM-readable version
    type_hint: str          # "scalar" | "list" | "object" | "text"
```

Placeholder substitution uses `.value`. Synthesis and error messages
use `.display`. The executor validates type compatibility: if step 2
depends on step 1's output and step 2's tool schema expects a number,
fail the plan at validation time with a clear error ("Calculator
expects scalar, but step_1 returns object").

**Effort:** ~1 day. Touches every tool (minor), base class (moderate),
executor (moderate). Backward-compatible with a migration flag.

**Expected impact.** Eliminates a whole class of silent pipeline
brittleness. Currently, tool-chain bugs only surface when a downstream
tool fails — and the error message ("Syntax error: invalid syntax")
is unhelpful. Typed contracts fail at plan-validation time with
actionable messages.

---

## 4. Cost/latency routing — v1 for simple queries, v2 for complex

**Motivation.** Ablation showed v2 costs 29% more and runs 24% slower
than v1, for a 10pp success-rate gain concentrated on multi-tool and
grounded-synthesis queries. On Q6 ("What is 15 percent of 2500?")
both versions succeeded identically — v2's overhead was pure waste.

**What I'd build.** A lightweight query-classifier that runs before
the planner (Haiku, `temperature=0`, <200 token prompt):

- Input: the user query
- Output: `{ "complexity": "simple" | "complex", "reasoning": "..." }`
- "Simple" = single tool sufficient, no cross-source synthesis,
  deterministic answer.
- "Complex" = multi-tool, grounding-sensitive, or ambiguous.

Route to v1 prompts for simple, v2 for complex. Log the classification
so we can audit its accuracy over time.

**Effort:** ~half a day.

**Expected impact.** Estimated 15-25% cost reduction on a realistic
query mix (simple queries dominate real usage). Slight classification
error rate acceptable because v1 still handles most queries well —
worst case is under-using v2 on a query that would have benefited,
not outright failure.

---

## 5. OpenTelemetry integration + metric dashboard

**Motivation.** The scaling analysis notes that log-based debugging
stops being viable at ~10+ concurrent users. Current observability
is Python `logging` to stdout and JSON run records in `runs/`. This
is sufficient for single-user interactive use and completely
insufficient for anything production-like.

**What I'd build.**

- Wrap every API call and tool call with OTel spans. Parent span =
  query; children = planning, each tool call, synthesis, judge.
- Export to Tempo/Jaeger for trace view; export metrics (span
  duration p50/p95, tool success rate, cost per query) to Prometheus.
- Grafana dashboard with: query volume, success rate, mean cost,
  p95 latency, per-tool error rate, rate-limit hit frequency.
- Add correlation IDs propagated through the trace (so a user-reported
  failure can be traced end-to-end without grep).

**Effort:** ~1-1.5 days. OTel Python SDK is mature; most of the work
is deciding the span taxonomy and dashboard layout.

**Expected impact.** Non-functional but high-leverage. Prerequisite
for anything resembling production deployment — you cannot operate
what you cannot observe.

---

## What I explicitly would NOT ship in the first week

To be honest about tradeoffs, three tempting features I would defer:

- **Semantic chunking for document_qa.** Would improve retrieval
  quality but the current fixed-size chunking is not the bottleneck
  for Q3 (which fails on synthesis, not retrieval). Defer.
- **Full Docker/gVisor sandboxing for code_executor.** Current
  subprocess isolation is appropriate for the stated threat model.
  Upgrading to kernel-level isolation is a production-deployment
  concern, not a week-one feature.
- **Multi-agent / hierarchical planning.** Glamorous but premature.
  Fix the reflection loop first; measure; only then consider
  decomposing the single agent into multiple specialised agents.

---

## Priority rationale

Ordered by: **(addresses a real observed failure) + (unlocks
downstream work)**.

- **#1 (reflection)** directly fixes Q8 and enables iterative
  planning — prerequisite for harder queries.
- **#2 (multi-run eval)** changes what we can even measure — you
  can't fix what you can't see.
- **#3 (typed outputs)** removes a recurring class of brittleness
  that will keep biting as more tools are added.
- **#4 (routing)** is pure efficiency and can ship independently
  of the above.
- **#5 (observability)** is a long-running investment; would start
  in week one but take multiple weeks to mature.