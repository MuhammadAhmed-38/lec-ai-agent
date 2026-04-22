# LEC AI Agent

A production-grade agentic system that orchestrates 5 tools to answer
multi-step queries reliably. Built for the LEC AI AI Engineer assignment.

## Quick Start

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and TAVILY_API_KEY

# 3. (Optional) ingest PDFs for document_qa tool
python -m tools.document_qa

# 4. Run a query
python main.py "What is the population of France multiplied by 2?"
```

## CLI Usage

```bash
python main.py "your query"                       # human-readable output
python main.py "your query" --json                # JSON output (for eval)
python main.py "your query" --prompt-version v1   # use baseline prompt
python main.py "your query" --verbose             # detailed logs
```

## Tools

The agent has access to 5 tools:

| Tool | Purpose |
|---|---|
| `web_search` | Tavily-backed web search for current information |
| `calculator` | AST-based safe arithmetic evaluator |
| `knowledge_base_lookup` | Structured facts lookup (local JSON KB) |
| `code_executor` | Sandboxed Python execution for data analysis |
| `document_qa` | Vector search over ingested PDFs (ChromaDB + MiniLM-L6-v2) |

## Architecture

Query → **Planner** (Sonnet) produces structured JSON plan →
**Executor** (Haiku) runs tools (sequential or parallel) →
**Synthesis** turns observations into a final answer.

Full details in `ARCHITECTURE.md` (to be added).

## Project Structure
agent/
config.py         Load env vars, model pricing
budget.py         Token + cost tracking, hard caps
prompts.py        v1 (baseline) and v2 (structured) prompts
planner.py        Query → plan JSON (Sonnet)
executor.py       Plan → observations → answer (Haiku)
orchestrator.py   Top-level coordinator
tools/
base.py           Tool ABC + ToolRegistry
calculator.py
web_search.py
knowledge_base.py
code_executor.py
document_qa.py
eval/               (added in evaluation phase)
data/               PDFs + local KB
main.py             CLI entry point

## Running Tests
Pre-flight check:
```bash
python test_setup.py
```

End-to-end agent run:
```bash
python main.py "What is 5 percent of 2000?" --verbose
```