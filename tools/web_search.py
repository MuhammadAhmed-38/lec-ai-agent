"""
Web search tool backed by Tavily API.

Tavily is chosen because:
  - Built for LLM agents (returns clean, relevance-ranked text)
  - Free tier: 1000 searches/month
  - Simpler than scraping + parsing raw Google/Bing results

Returns a compact formatted summary rather than raw JSON, because:
  - The LLM has to reason over this output; less noise = better decisions
  - Saves tokens (cost)
"""
from __future__ import annotations

import asyncio
from typing import Any

from tavily import TavilyClient

from agent.config import TAVILY_API_KEY
from tools.base import Tool


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Searches the web for current information. Use this for: "
        "recent events, current prices/rates, real-time data, facts "
        "that change over time (weather, stock prices, news), or "
        "any query where up-to-date info matters. "
        "Do NOT use for static knowledge (math, historical facts, "
        "definitions) — use your own knowledge or the knowledge_base tool."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Keep it concise (3-8 words works best).",
            },
            "max_results": {
                "type": "integer",
                "description": "How many results to return. Default 3, max 5.",
                "default": 3,
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, timeout_seconds: int = 20) -> None:
        super().__init__(timeout_seconds=timeout_seconds)
        self._client = TavilyClient(api_key=TAVILY_API_KEY)

    async def _run(self, query: str, max_results: int = 3) -> str:
        # Tavily SDK is synchronous; wrap in thread to avoid blocking the event loop.
        max_results = max(1, min(int(max_results), 5))
        response = await asyncio.to_thread(
            self._client.search,
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        return self._format_results(query, response)

    @staticmethod
    def _format_results(query: str, response: dict[str, Any]) -> str:
        """Produce a compact, LLM-friendly string summary."""
        results = response.get("results", [])
        if not results:
            return f"No web results found for query: {query!r}"

        lines = [f"Web search results for: {query!r}", ""]
        for i, r in enumerate(results, start=1):
            title = r.get("title", "(no title)").strip()
            url = r.get("url", "").strip()
            content = r.get("content", "").strip()
            # Truncate long content — LLMs don't need 2000-char blobs per result
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{i}] {title}")
            lines.append(f"    URL: {url}")
            lines.append(f"    {content}")
            lines.append("")
        return "\n".join(lines).strip()