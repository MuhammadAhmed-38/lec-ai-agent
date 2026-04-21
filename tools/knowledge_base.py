"""
Knowledge base tool: lookup structured facts from a local JSON store.

Design rationale:
  - Web search is slow (1-3s) and costs API quota.
  - For static, well-known structured facts (country capitals, company
    info, constants), a local lookup is instant and free.
  - In a real production system, this would wrap an internal company
    KB, CRM, product catalog, or similar structured data source.

The agent should prefer this tool over web_search for:
  - Country/geography facts
  - Basic company info (CEO, HQ, founding year)
  - Scientific constants
  - Well-known historical dates

The tool supports two modes:
  1. list_topics: returns top-level keys so the agent knows what's available
  2. lookup: returns a nested section by path (e.g. 'countries.france')
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.config import DATA_DIR
from tools.base import Tool


_KB_PATH: Path = DATA_DIR / "kb" / "facts.json"


def _load_kb() -> dict[str, Any]:
    if not _KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at {_KB_PATH}")
    with _KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class KnowledgeBaseTool(Tool):
    name = "knowledge_base_lookup"
    description = (
        "Looks up structured facts from a local knowledge base. "
        "Prefer this over web_search for static well-known facts: "
        "country info (capital, population, currency, area), "
        "company info (CEO, headquarters, founded year), "
        "scientific constants, and major historical event dates. "
        "\n\n"
        "Usage:\n"
        "  1. Call with action='list_topics' to discover available categories.\n"
        "  2. Call with action='lookup' and a dot-path like "
        "'countries.france' or 'companies.apple' to fetch facts.\n"
        "  3. Call with action='lookup' and top-level key like "
        "'scientific_constants' to fetch the whole section."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_topics", "lookup"],
                "description": "'list_topics' to see categories, 'lookup' to fetch facts.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Dot-separated path to the fact, e.g. 'countries.france' or "
                    "'companies.apple.ceo'. Required when action='lookup'."
                ),
            },
        },
        "required": ["action"],
    }

    async def _run(self, action: str, path: str = "") -> str:
        kb = _load_kb()

        if action == "list_topics":
            topics = list(kb.keys())
            lines = ["Available top-level topics in knowledge base:"]
            for t in topics:
                sub = kb[t]
                if isinstance(sub, dict):
                    sub_keys = list(sub.keys())
                    preview = ", ".join(sub_keys[:5])
                    if len(sub_keys) > 5:
                        preview += f", ... (+{len(sub_keys) - 5} more)"
                    lines.append(f"  - {t}: [{preview}]")
                else:
                    lines.append(f"  - {t}")
            return "\n".join(lines)

        if action == "lookup":
            if not path:
                return "[ERROR] 'path' is required when action='lookup'"
            return self._lookup(kb, path)

        return f"[ERROR] Unknown action: {action!r}. Use 'list_topics' or 'lookup'."

    @staticmethod
    def _lookup(kb: dict[str, Any], path: str) -> str:
        keys = [k.strip().lower() for k in path.split(".") if k.strip()]
        if not keys:
            return "[ERROR] Empty path"

        current: Any = kb
        traversed: list[str] = []
        for key in keys:
            if not isinstance(current, dict):
                return (
                    f"[ERROR] Cannot descend into '{key}' — "
                    f"parent at '{'.'.join(traversed)}' is not a nested object."
                )
            # Case-insensitive key match
            match = next((k for k in current.keys() if k.lower() == key), None)
            if match is None:
                available = list(current.keys())
                return (
                    f"[ERROR] Key '{key}' not found under '{'.'.join(traversed) or 'root'}'. "
                    f"Available: {available}"
                )
            current = current[match]
            traversed.append(match)

        # Format the result for the LLM.
        # For scalars (int/float/str/bool): return raw value so downstream
        # tools (e.g. calculator) can consume it cleanly.
        # For dicts/lists: return formatted JSON with path context.
        full_path = ".".join(traversed)
        if isinstance(current, dict):
            formatted = json.dumps(current, indent=2, ensure_ascii=False)
            return f"Facts at '{full_path}':\n{formatted}"
        if isinstance(current, list):
            return f"{current}"  # raw list, no prefix
        # Scalar: int, float, str, bool
        return f"{current}"