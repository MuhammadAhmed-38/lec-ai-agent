"""
Base class for all tools.

Every tool in tools/ extends Tool and implements:
  - name: str          (tool identifier used by the agent)
  - description: str   (what the tool does — shown to the LLM)
  - input_schema: dict (JSON schema for parameters, Anthropic format)
  - async _run(**kwargs) -> str  (actual implementation)

The base class handles:
  - timeout enforcement
  - error wrapping (returns structured error instead of raising)
  - logging
  - conversion to Anthropic tool-use format

Design: tools return strings (or JSON-serializable objects) rather than
raising, because the agent needs to *reason* over tool failures — it
shouldn't crash the whole loop.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent.config import TOOL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    Structured result from a tool invocation.

    success=False means the tool ran but failed (timeout, API error,
    invalid input, etc.). The agent can read `error` and decide to
    retry, pick an alternative, or give up.
    """
    tool_name: str
    success: bool
    output: Any = None           # Present when success=True
    error: str = ""              # Present when success=False
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_llm_string(self) -> str:
        """Format for feeding back into the agent's tool-use loop."""
        if self.success:
            return str(self.output)
        return f"[TOOL ERROR in {self.tool_name}] {self.error}"


class Tool(ABC):
    """Abstract base class for all tools."""

    # Subclasses must set these as class attributes.
    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def __init__(self, timeout_seconds: int = TOOL_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must set 'name'")
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} must set 'description'")
        if not self.input_schema:
            raise ValueError(f"{self.__class__.__name__} must set 'input_schema'")

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """
        Actual tool implementation. Subclasses override this.

        Can raise exceptions — the base class's execute() will catch them
        and wrap in a ToolResult with success=False.
        """
        ...

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Public entrypoint. Runs _run() with timeout and error handling.
        Always returns a ToolResult (never raises).
        """
        start = time.perf_counter()
        try:
            logger.info(f"Executing tool '{self.name}' with args keys: {list(kwargs.keys())}")
            output = await asyncio.wait_for(
                self._run(**kwargs),
                timeout=self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(f"Tool '{self.name}' succeeded in {latency_ms:.0f}ms")
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output,
                latency_ms=latency_ms,
            )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start) * 1000
            error = f"Tool timed out after {self.timeout_seconds}s"
            logger.warning(f"Tool '{self.name}' {error}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=error,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            error = f"{type(e).__name__}: {e}"
            logger.error(f"Tool '{self.name}' failed: {error}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=error,
                latency_ms=latency_ms,
            )

    def to_anthropic_schema(self) -> dict:
        """
        Convert to the format Anthropic's tool-use API expects.
        Used when constructing the `tools` parameter for messages.create().
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """
    Holds all registered tools. The orchestrator uses this to:
      - list available tools for the planner
      - look up tools by name during execution
      - produce the `tools=[...]` argument for Anthropic API calls
    """
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: '{name}'. Registered: {list(self._tools.keys())}")
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def anthropic_schemas(self) -> list[dict]:
        """Return list of tool schemas in Anthropic API format."""
        return [t.to_anthropic_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools