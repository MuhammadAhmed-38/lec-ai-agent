"""
Evaluation query set: 10 multi-step queries with graded success criteria.

Each query is designed to test a specific agent capability. The rubric is
binary (pass/fail) on the key assertion, with optional partial-credit notes
for the LLM judge.

Query mix (capability under test):
  Q1: KB lookup + math            — sequential dependency, scalar chain
  Q2: Web search + math           — fresh data retrieval + compute
  Q3: Document Q&A                — retrieval from ingested PDFs
  Q4: Multi-KB parallel lookup    — parallel execution
  Q5: Web + KB cross-reference    — mixed-source synthesis
  Q6: Pure calculator             — minimal plan (1 step)
  Q7: Code executor               — data analysis requiring programmatic logic
  Q8: Multi-hop web reasoning     — 2+ sequential web searches
  Q9: Ambiguous / underspecified  — tests planner's handling of vagueness
  Q10: Tool-failure resilience    — query that will likely fail some step
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalQuery:
    """One evaluation query with success criteria for the LLM judge."""
    id: str
    query: str
    capability: str                   # Short label for reporting
    expected_answer_contains: list[str]  # Substrings that MUST appear in final answer
    expected_tools: list[str] = field(default_factory=list)  # Tools we expect to see (soft check)
    must_not_contain: list[str] = field(default_factory=list)  # Bad signals (e.g., "I don't know")
    notes: str = ""                   # Human note for the judge / report


# ==========================================================
# THE 10 QUERIES
# ==========================================================

EVAL_QUERIES: list[EvalQuery] = [
    # ----- Q1: KB lookup + math (sequential) -----
    EvalQuery(
        id="Q1",
        query="What is the population of Pakistan in millions, multiplied by 10?",
        capability="kb_lookup + calculator (sequential)",
        expected_answer_contains=["2405", "2,405"],  # 240.5 * 10 = 2405
        expected_tools=["knowledge_base_lookup", "calculator"],
        notes="Pakistan pop is 240.5M in KB. Tests scalar chaining.",
    ),

    # ----- Q2: Web search + math -----
    EvalQuery(
        id="Q2",
        query="What is the current price of Bitcoin in USD, divided by 1000? "
              "Give the answer to 2 decimal places.",
        capability="web_search + calculator",
        expected_answer_contains=[],  # Price varies — judge uses numeric sanity check
        expected_tools=["web_search", "calculator"],
        notes="Answer should be a number around 50-150 as of 2026. "
              "Judge must verify: (a) answer is a number, (b) within 10-500 range.",
    ),

    # ----- Q3: Document Q&A -----
    EvalQuery(
        id="Q3",
        query="According to the indexed documents, what is the self-attention "
              "mechanism in the Transformer architecture?",
        capability="document_qa",
        expected_answer_contains=["attention", "query", "key"],  # Any 2 of these = pass
        expected_tools=["document_qa"],
        must_not_contain=["I don't know", "I cannot find"],
        notes="Attention paper is indexed. Answer should mention attention mechanism "
              "and at least one of query/key/value concepts.",
    ),

    # ----- Q4: Parallel KB lookups -----
    EvalQuery(
        id="Q4",
        query="What are the capitals of France, Japan, and Brazil? "
              "List them together.",
        capability="kb_lookup (parallel, 3 independent)",
        expected_answer_contains=["Paris", "Tokyo", "Brasilia"],
        expected_tools=["knowledge_base_lookup"],
        notes="All 3 capitals must appear. Good planner should parallelise these.",
    ),

    # ----- Q5: Web + KB cross-reference -----
    EvalQuery(
        id="Q5",
        query="Compare the population of India from the knowledge base "
              "with the current world population (from the web). "
              "What percentage of the world population is India?",
        capability="kb + web + calculator (mixed-source synthesis)",
        expected_answer_contains=["%"],  # answer must mention percentage
        expected_tools=["knowledge_base_lookup", "web_search", "calculator"],
        notes="India ~1428M, world ~8000M → ~17-18%. "
              "Judge should accept anything between 15% and 20%.",
    ),

    # ----- Q6: Pure calculator (minimal plan test) -----
    EvalQuery(
        id="Q6",
        query="What is 15 percent of 2500?",
        capability="calculator only (1-step plan)",
        expected_answer_contains=["375"],
        expected_tools=["calculator"],
        notes="Tests: does agent over-plan a trivial query? "
              "Ideal: 1-step plan. Penalty if it uses web_search unnecessarily.",
    ),

    # ----- Q7: Code executor (data analysis) -----
    EvalQuery(
        id="Q7",
        query="Given the list [23, 45, 67, 12, 89, 34, 56, 78, 90, 11], "
              "compute the mean and standard deviation. Show both values.",
        capability="code_executor",
        expected_answer_contains=["50.5"],  # mean = 50.5
        expected_tools=["code_executor"],
        notes="Mean = 50.5, std ≈ 27.7. Mean must appear; std is a nice-to-have.",
    ),

    # ----- Q8: Multi-hop reasoning -----
    EvalQuery(
        id="Q8",
        query="Who is the current CEO of Apple Inc., and in what year did "
              "that company release its first iPhone? Use the knowledge base.",
        capability="multi-hop KB lookup (2 sequential or parallel facts)",
        expected_answer_contains=["Tim Cook", "2007"],
        expected_tools=["knowledge_base_lookup"],
        notes="Both facts are in KB. iPhone release year is 2007.",
    ),

    # ----- Q9: Ambiguous / underspecified -----
    EvalQuery(
        id="Q9",
        query="Tell me about the largest country by area among France, Japan, and Brazil.",
        capability="multi-step: fetch areas, compare, pick max",
        expected_answer_contains=["Brazil"],
        expected_tools=["knowledge_base_lookup"],
        notes="Brazil ~8.5M sq km is largest. Tests agent's ability to compare fetched values.",
    ),

    # ----- Q10: Tool-failure resilience (intentionally tricky) -----
    EvalQuery(
        id="Q10",
        query="What is the population of the fictional country Atlantis, "
              "and what is 5 plus 7?",
        capability="graceful failure (KB lookup will fail) + successful calc",
        expected_answer_contains=["12"],  # 5+7 must succeed
        must_not_contain=["Atlantis has a population of"],  # must not hallucinate
        notes="Atlantis is not in KB. Agent should acknowledge the failure "
              "honestly AND still answer the math part. "
              "Tests grounding rule: does the agent hallucinate a population?",
    ),
]


def get_all_queries() -> list[EvalQuery]:
    """Return all eval queries."""
    return list(EVAL_QUERIES)


def get_query_by_id(query_id: str) -> EvalQuery:
    for q in EVAL_QUERIES:
        if q.id == query_id:
            return q
    raise KeyError(f"No eval query with id {query_id!r}")


def summary() -> str:
    lines = [f"{len(EVAL_QUERIES)} eval queries:"]
    for q in EVAL_QUERIES:
        lines.append(f"  {q.id}: [{q.capability}] {q.query[:70]}...")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())