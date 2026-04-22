# Report Notes

Collected throughout development — paste-ready fragments for the final report.

## Architecture & design decisions

### LLM-as-judge
LLM-as-judge used because ~40% of queries have non-deterministic correct answers
(prices, ranges, natural-language explanations). Judge uses Haiku at temperature 0
with a structured JSON output schema. Judgment includes binary pass/fail, 0-1 score
for partial credit, and reasoning text which feeds into failure analysis.

### Dual-model agent (Sonnet planner, Haiku executor)
Sonnet used for planning because plan quality dominates overall success — a bad plan
cannot be recovered by a good executor. Haiku used for execution synthesis because
the task is structured (formatting observations into a final answer) rather than
open-ended reasoning. This split reduces cost by ~60% versus all-Sonnet while
preserving plan quality.

### Explicit plan-based execution (not native tool use)
Chose explicit plan-based execution over Anthropic's native tool-use loop because:
(a) the assignment requires an explicit planning step, (b) it gives full visibility
into the decision trace for debugging, and (c) it makes prompt ablation surgical —
we ablate the planner prompt without touching the executor, or vice versa.

### Code executor threat model
Code executor uses layered subprocess isolation with explicit threat modeling. This
is not production-grade (would require container/VM isolation), but is appropriate
for the threat model of "LLM generates incorrect code" rather than "attacker
attempts sandbox escape".

### Knowledge base vs web search tool selection
Agent prefers structured KB over web_search for static facts, reducing per-query
latency by ~90% and eliminating API cost for queries where a local lookup suffices.
Tool descriptions explicitly guide this preference to prevent over-use of web search.

### Chunking strategy (document_qa)
Chose fixed-size chunking (500 chars, 50 char overlap) over semantic chunking for
reproducibility and simplicity within the 48-hour timeline. Semantic chunking
(sentence boundaries + similarity-based merging) would likely improve retrieval
quality but adds complexity. This is a conscious tradeoff.

## Eval design

### Query set rationale
Query set was designed for capability coverage, not uniform difficulty. Q10 in
particular is a "trap" query where partial failure is the correct behaviour —
tests that the agent acknowledges unknown facts rather than hallucinating them.

## Placeholders to fill after eval runs

- Baseline (v2) success rate: ___/10 (__%)
- v1 ablation success rate: ___/10 (__%)
- Delta: __ percentage points
- Hypothesis confirmed/refuted: ___
- Worst-performing capability: ___
- Best-performing capability: ___
- Mean cost per query: $___
- Mean latency per query: __s


## Finding from Q10 smoke test (2026-04-22)

Smoke test with Q10 ("population of fictional Atlantis + 5+7") revealed that the
v2 executor's grounding rule is insufficient to prevent hallucination on fictional
entities. Agent answered "12" for the math (correct) but fabricated population
figures ("twenty and forty million") for Atlantis despite KB lookup returning no
match.

Judge correctly assigned partial credit (0.5) — caught the hallucination but
credited the correct math. This demonstrates:
  (a) The judge's nuanced scoring works as designed
  (b) Grounding instructions alone ("do not invent facts") are weaker than
      structural safeguards (e.g., forcing agent to cite specific tool outputs)
  (c) This is a publishable failure mode — the trap query design worked

Future mitigation: add a post-synthesis validator that checks every numerical
claim in the final answer against the tool trace.

## Eval findings — v2 baseline (2026-04-22)

### Overall
- Passed: 9/10 (90%)
- Mean score: 0.95
- Total agent cost: $0.1367
- Total judge cost: $0.0267
- Wall time: 133s (mean ~13s per query)

### Q8 — under-planning / first-lookup bias
Query asked for Apple CEO AND iPhone release year. Both facts exist in KB but in
different paths: companies.apple.ceo and historical_events.iphone_release_year.
Agent only queried the first path, saw iPhone mentioned as a notable product, but
never searched historical_events for the year. Result: partial credit (0.5).

This reveals a real limitation: the planner commits to a single lookup path early
and doesn't iterate when the rubric implies missing data. Mitigation would require
either (a) a reflection step after execution to check if all sub-questions were
answered, or (b) pre-execution KB discovery via the list_topics action.

### Q10 — stochastic behaviour
Smoke test run (first attempt): agent hallucinated Atlantis population → FAIL (0.5)
Full run (second attempt): agent gracefully acknowledged fictional nature, used
web_search as fallback, did NOT hallucinate → PASS (1.0)

Same query, same prompt, different temperature-driven outcomes. Demonstrates that
success rate is a distribution, not a fixed number. Production implication:
reliability testing needs multiple runs per query, not single-shot evaluation.

### Tool usage patterns
- Parallel execution observed in Q4 (3 independent KB lookups) and Q9 (5 calls)
- 2 tool failures recovered via retry mechanism (Q2 and Q5 web_search)
- No plan rejections (all plans passed validation)

## Ablation results — v1 vs v2 (2026-04-22)

### Summary table

| Metric | v2 (structured) | v1 (minimal) | Delta |
|---|---|---|---|
| Success rate | 9/10 (90%) | 8/10 (80%) | +10 pts |
| Mean score | 0.95 | 0.88 | +0.07 |
| Agent cost (total) | $0.137 | $0.106 | +$0.031 |
| Wall time | 133s | 107s | +26s |

### Where v2 helped

**Q3 (document_qa):** v2 PASS (1.0), v1 FAIL (0.30). Judge noted v1's
answer was a "good high-level description" but missed required technical
details (query, key). v2's grounding rule ("any numerical or factual
claim must come from a tool result") forced the synthesis LLM to cite
specific passages from the retrieved chunks rather than produce a
generic summary.

**Q5 (mixed-source synthesis):** v1 calculator failed with
`Name 'india_population' not allowed` — the planner generated a
natural-language variable reference instead of the `{{step_N.output}}`
placeholder. v2's explicit CORRECT/INCORRECT examples in the planner
prompt prevented this. v1 still passed Q5 because the per-step retry
mechanism recovered, but at the cost of extra latency and tool calls.

### Where v2 did NOT help

**Q8 (multi-hop KB lookup):** Both versions FAIL (0.50). Agent retrieved
Apple CEO but never queried `historical_events.iphone_release_year`.
This is a structural planning limitation, not a prompting one. Fix
would require either (a) a reflection step post-execution, or (b)
pre-execution KB discovery. Intentionally noted as out of scope for
this deliverable.

### Cost/latency tradeoff

v2 costs 29% more than v1 ($0.137 vs $0.106) and runs 24% slower.
Justified by the +10 point success rate and +0.07 mean score improvement
— but in latency-sensitive or extreme-scale deployments, this tradeoff
should be re-evaluated per query type. v1 is adequate for simple
single-tool queries; v2's overhead only pays off on multi-tool or
grounded-synthesis queries.

### Tool failures (retry mechanism working)

v1 had 3 recovered tool failures (Q1, Q2, Q5 calculator errors).
v2 had 2 (Q2, Q5 web_search timeouts). In all 5 cases, the per-step
retry mechanism succeeded on second attempt. Zero unrecovered tool
failures across 20 total query runs.

