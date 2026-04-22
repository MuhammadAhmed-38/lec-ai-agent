# AI-Usage Note

This submission is the output of a tight human-AI engineering loop. I used Claude (Anthropic's assistant, via claude.ai) as an aggressive pair-programmer to compress what would normally be a multi-week build into two days. This note is a specific breakdown of how that collaboration worked — because the email noted that judgment when using AI is what's being evaluated, not the presence of AI.

## The real work: direction, validation, and judgment

What this assignment demonstrates is not "I can write agentic code from scratch alone." It demonstrates **I can architect, direct, and ship a production-grade system using modern tools at a pace that would be impossible solo.** Every line Claude generated passed through my review, testing, and judgment before being accepted. Specifically, the work that was genuinely mine:

### Architectural direction

Every design decision — the dual-model split (Sonnet for planning, Haiku for synthesis), explicit plan-based execution instead of Anthropic's native tool-use, explicit v1/v2 prompt versioning for clean ablation, LLM-as-judge with structured rubrics — was a choice I made after weighing alternatives. When Claude suggested a defensible alternative, I weighed it against the assignment requirements and chose deliberately.

A concrete example: mid-build, Claude suggested formatting `ARCHITECTURE.md` with emojis and color for visual appeal. I declined on the grounds that the document's audience is senior engineers reviewing production design, not a portfolio viewer — Claude's suggestion was cosmetic, not substantive. This kind of judgment call happened dozens of times.

### Every actual bug was caught by me reading output

Four distinct failure modes surfaced during development — none of which
Claude pre-empted:

1. **Planner argument invention** (used `key` instead of `path`) — I 
   caught this by reading the first end-to-end test output and
   noticing the tool error. Only then did we fix it by injecting tool
   schemas into the planner context.
2. **Synthesis LLM hallucinating `<function_calls>` syntax** — When
   all tools failed, the agent fabricated tool-call XML and invented
   an answer. Claude initially treated the test output as "ok"; I
   flagged it as a critical production risk. The grounding rule in
   v2 came from my objection.
3. **Natural-language placeholders** (`<population_from_step_1>`) —
   Identified in a test run by reading plan output. Fix: explicit
   placeholder syntax examples in v2 prompt.
4. **Display-formatted tool output breaking scalar chains** — This
   only surfaced when I ran a real query and traced why the
   calculator kept failing. The fix (dual output modes) was small,
   but the diagnosis was mine.

In every case, the pattern was the same: Claude generated code, I ran it, I read the output carefully, I identified the wrong behaviour, Claude proposed a fix, I tested it. The debugging judgment — what is acceptable output, what is a real bug, what is out-of-scope — was consistently mine.

### Evaluation design

The 10 eval queries, the rubric design, and the judge prompt structure were my domain. Specifically, **Q10 (the fictional Atlantis trap query)** was my call — a deliberate test to see whether the agent would hallucinate a population for a nonexistent country. This single query produced the most valuable finding in the project: that v2's grounding rule reduces but does not eliminate hallucination risk on fictional entities. That insight came from my query design, not from the
evaluation running.

### Scope and shipping decisions

Four times during the build I had to choose between "fix this bug now" vs. "document it as a known limitation and ship." Each of these calls was mine, based on remaining time budget, user impact, and the assignment's explicit invitation to be honest about what didn't work:

- **Q8 multi-hop failure** — documented as structural planning
  limitation rather than hacked around
- **Q10 stochasticity** — flagged as a production risk requiring
  multi-run eval rather than patched
- **Code executor sandboxing** — layered subprocess isolation chosen
  over Docker/gVisor based on threat-model analysis
- **Semantic chunking for document_qa** — deferred to the roadmap
  because retrieval wasn't Q3's failure mode

Shipping a useful system on a hard deadline requires knowing what NOT to fix. That is the judgment this assignment tests.

## What Claude contributed

I'm not going to pretend I typed all 2,000+ lines of code by hand in 2 days. AI tools are here to help, to automate the things for us. What Claude contributed:

- **Code velocity.** Module-by-module scaffolding, boilerplate, error
  handling patterns, JSON repair utilities, async orchestration —
  Claude generated these quickly based on my architectural
  instructions. This is where the 2-month-to-2-day compression
  comes from.
- **Known-good patterns.** The JSON repair + retry on parse failure,
  grounding rules, tool schema injection, and structured Judgment
  dataclass are idioms Claude knows from being trained on a large
  corpus of agentic code. I would have implemented these eventually
  but Claude short-circuited the discovery.
- **Documentation density.** The architecture doc's table structure
  and the report's executive summary format benefit from Claude's
  ability to produce concise technical writing.

## Claude did not contribute:

- Bug discovery (all four major bugs were caught by me reading output)
- Design decisions (every "chosen / rejected" in `ARCHITECTURE.md`
  was my call after hearing alternatives)
- Eval query design (especially the trap queries that produced
  findings)
- Scope decisions (what to ship, what to document, what to defer)
- The engineering judgment to reject AI suggestions that were
  technically plausible but contextually wrong (the emoji example is
  one of many)

## Why this collaboration model is the right answer

Modern engineering is not "human vs AI" or "all AI" — it's a directed-leverage  loop. The engineer supplies architectural taste, domain judgment, debugging instinct, and shipping discipline. The AI supplies typing speed, pattern recall and iteration velocity. A week of work becomes a day; a month's project becomes 48 hours.

This only works if the engineer actually engages critically with every output — if you blindly accept what the AI produces, you ship brittle code with hallucinated tool calls that lie to users. I didn't. Every module was tested. Every eval result was audited. Every design decision was mine to defend.

If the role at LEC AI is for an engineer who can direct this kind of leverage loop and ship production-grade systems on tight timelines, I can evidence it with this project. If the role requires someone who writes every line of code without AI assistance — genuinely, respectfully — that is not the engineering reality I work in, and I suspect it's not the reality LEC AI is hiring for either.