# Failure Log — Agent Development

This document captures failure modes observed during agent development
and how they were addressed. It's the raw material for the final report's
"What Broke" section.

## During initial end-to-end integration (~6pm Tuesday, v2 prompts)

### Failure 1: Planner invented argument names
- **Observed:** Planner output `{"action": "lookup", "key": "..."}` for knowledge_base_lookup
- **Root cause:** Planner was given tool descriptions but not input schemas
- **Fix:** Inject full JSON input_schema of each tool into planner's user message, with explicit instruction to use exact parameter names
- **Remaining risk:** Planner could still hallucinate args on novel queries; mitigated by validation step that rejects plans referencing unknown tools/args

### Failure 2: Synthesis LLM hallucinated tool calls on upstream failures
- **Observed:** When all tool calls failed, synthesis output included `<function_calls><invoke name="web_search">...` fabricated syntax, then invented plausible answer
- **Root cause:** Synthesis prompt did not constrain model behaviour when tools failed
- **Fix:** v2 executor prompt explicitly forbids generating tool-call syntax and requires grounding every claim in actual tool observations
- **Signal:** This is a real production risk — user sees confident wrong answer. Must be flagged in eval.

### Failure 3: Natural-language placeholders instead of template syntax
- **Observed:** Planner generated `"expression": "<population_from_step_1> * 2"`
- **Root cause:** v2 prompt described parallel_groups but never specified the placeholder syntax
- **Fix:** Explicit CORRECT/INCORRECT examples in v2 planner prompt
- **Ablation signal:** v1 intentionally left without this guidance

### Failure 4: Display-formatted tool outputs breaking scalar chains
- **Observed:** KB returned `"'countries.france.population_millions' = 67.97"` as a formatted string; placeholder substitution piped this into calculator which couldn't parse it
- **Root cause:** Tool output format was optimised for LLM readability, not for downstream tool consumption
- **Fix:** KB returns raw scalar for scalar lookups, formatted text only for dict/object lookups
- **Remaining risk:** Any tool returning formatted text that another tool tries to parse. Long-term fix: formal "output type" contract per tool.

