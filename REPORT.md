# Production Agentic System — Report

**Author:** Muhammad Ahmed
**Assignment:** LEC AI — AI Engineer, Assignment 2
**Timeline:** ~30 working hours across 2 days

## What I built

A multi-tool agent that decomposes a user query into an explicit JSON plan,
executes it (with parallel tool calls where independent), and synthesises a
grounded final answer. Five tools: `calculator` (AST-safe), `web_search`
(Tavily), `knowledge_base_lookup` (local JSON), `document_qa` (ChromaDB +
MiniLM-L6-v2 local embeddings), `code_executor` (subprocess-isolated Python
sandbox). Dual-model split: Claude Sonnet for planning and synthesis, Haiku
for the evaluation judge. Hard budget caps enforced per-query and per-session.
Full details and diagrams in `ARCHITECTURE.md`.

**On the spec's "reflect" step:** Reflection happens implicitly inside the
synthesis call — the Haiku synthesis LLM reasons over all tool
observations (including failures) under strict grounding rules before
producing the final answer. An explicit separate reflection step
between observations and synthesis is scoped as ROADMAP item #1 and
would primarily address multi-hop failures like Q8.

## Results

**Baseline (v2 prompts, 10 queries):** 9/10 passed, mean score 0.95,
mean cost $0.014/query, mean latency ~13s.

**Ablation (v1 minimal prompts, 10 queries):** 8/10 passed, mean score 0.88,
mean cost $0.011/query, latency ~11s.

| Metric | v2 (structured) | v1 (baseline) | Delta |
|---|---|---|---|
| Success rate | 90% | 80% | +10pp |
| Mean score | 0.95 | 0.88 | +0.07 |
| Agent cost (total) | $0.137 | $0.106 | +29% |
| Wall time (total) | 133s | 107s | +24% |

v2's overhead (extra tokens, longer latency) is justified by the success-rate
gain on grounded-synthesis and multi-tool queries. For simple single-tool
queries (Q6), v1 and v2 are identical — the overhead is wasted there.

## What broke (honest failures)

**Build-time failures before reaching eval.** Before the agent was ready for
evaluation, four distinct failure modes surfaced during end-to-end debugging,
all captured in `FAILURE_LOG.md`:
(1) The planner invented argument names (used `key` instead of the tool's
`path` parameter) because it was given tool *descriptions* but not tool
*schemas*. Fixed by injecting the full JSON input_schema into the planner's
user message.
(2) When all tool calls in a query failed, the synthesis LLM generated
fabricated `<function_calls>` XML syntax and invented a plausible-sounding
answer. Fixed with strict grounding rules in the v2 executor prompt.
(3) The planner generated natural-language placeholders like
`<population_from_step_1>` instead of the `{{step_N.output}}` template
syntax. Fixed by adding explicit CORRECT/INCORRECT examples in the v2
planner prompt (left out of v1 intentionally — this is now a real ablation
signal).
(4) Tools returning human-readable formatted strings (e.g., KB returning
`'countries.france.population' = 67.97`) broke placeholder substitution
into calculator tools expecting raw scalars. Fixed with dual output modes
on the KB tool. This is the type of brittleness I expect to recur in any
multi-tool pipeline without a formal output-type contract.

**Q8 — multi-hop KB lookup. Both v1 and v2 failed (score 0.5).** Query asked
for Apple's CEO *and* the iPhone release year. Both facts exist in the KB but
under different top-level paths (`companies.apple.ceo` vs
`historical_events.iphone_release_year`). The planner committed to a single
lookup path, saw "iPhone" in Apple's notable-products list, and never queried
the historical events namespace. This is a **structural planning limitation
that prompt engineering alone cannot fix** — it needs either a reflection
step or pre-execution KB discovery. Noted as a known failure rather than
hidden. Post-fix design sketched in the Roadmap.

**Q3 — document_qa generic summary. v1 failed (score 0.3), v2 passed.** The
v1 executor prompt didn't enforce grounding in tool output, so the synthesis
LLM produced a plausible-sounding generic explanation of self-attention that
missed specific technical terms required by the rubric. v2's explicit
"ground every claim in tool observations" rule forced the model to cite
retrieved passages. This was the largest single-query ablation delta we
observed and is the clearest evidence that the grounding rule matters.

**Q10 — stochastic hallucination (smoke test).** On a preliminary run, the
agent fabricated a population figure ("twenty and forty million") for the
fictional country Atlantis, despite the v2 grounding rule. On the full eval
run, the agent handled the same query correctly (acknowledged fictional
nature, used web_search as fallback). Same prompt, different output. This
means **success rate is a distribution, not a point estimate**, and single-run
evaluation understates reliability risk. Production systems should evaluate
each query N times and report p5/p50/p95.

**Several recoverable tool failures.** v1 had three calculator errors across
Q1, Q2, Q5 — all recovered by the per-step retry mechanism. Notably Q5 in v1
showed a classic planner bug (`CalculatorError: Name 'india_population' not
allowed`) because v1 didn't teach the `{{step_N.output}}` placeholder syntax.
The retry saved it but at the cost of extra latency.

## What I learned

**Plan quality dominates everything.** A bad plan cannot be fixed by a good
executor; a good plan can survive a mediocre executor. Investing in planner
prompt quality and injecting full tool schemas (not just descriptions) was
the highest-ROI change I made. Before schema injection, the planner was
inventing argument names (`key` instead of `path`). After: zero instances.

**Grounding rules are not optional.** Without an explicit rule forbidding the
synthesis LLM from generating content outside tool observations, the model
will confabulate plausible-sounding but ungrounded answers — especially when
upstream tools fail. The cost is a longer system prompt (~300 extra tokens
per synthesis call). The benefit is the difference between a user trusting
the agent and a user being silently misled.

**Tool output formats matter as much as tool logic.** My knowledge-base tool
originally returned formatted strings like `'countries.france.population' =
67.97`. Placeholder substitution piped this whole string into the calculator,
which couldn't parse it. The fix — return the raw scalar for scalar lookups,
formatted text only for objects — was trivial (five lines) but only
discovered by running a real end-to-end query and watching the failure mode.

**LLM-as-judge is usable but non-trivial to deploy.** ~40% of my eval queries
have non-deterministic correct answers (prices, ranges, natural-language
explanations). Exact matching would have made the eval meaningless. The judge
(Haiku, temperature 0, structured JSON schema) worked well — notably giving
partial credit on Q10's hallucinated-but-math-correct smoke-test output. But
judge prompts need the same care as agent prompts; a vague judge prompt
produced inconsistent verdicts in early iterations.

## Ablation takeaway

The structured v2 prompt improves success rate by 10 percentage points over
the minimal v1 prompt. The gain concentrates in two areas: **multi-tool
queries where v2's explicit placeholder syntax prevents malformed arguments**,
and **synthesis queries where v2's grounding rule prevents generic
descriptions that miss required technical content**. For single-tool trivial
queries (Q6: "15 percent of 2500"), v1 and v2 are equivalent, and v2's token
overhead is pure cost. A production deployment should pick the prompt variant
per query class rather than applying v2 uniformly.