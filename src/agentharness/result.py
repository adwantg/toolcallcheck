"""Structured result object returned by AgentRunner.invoke()."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolCall:
    """Record of a single tool invocation during agent execution."""

    name: str
    args: dict[str, Any]
    response: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        d: dict[str, Any] = {"name": self.name, "args": self.args}
        if self.response is not None:
            d["response"] = self.response
        if self.error is not None:
            d["error"] = self.error
        return d


@dataclasses.dataclass(frozen=True)
class TraceEntry:
    """A single entry in the conversation trace."""

    role: str  # "user", "assistant", "tool", "system"
    content: str | None = None
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class AgentResult:
    """Full result of an agent invocation.

    Carries the final text response, all tool calls that occurred,
    model/routing metadata, captured headers, and the full conversation
    trace for debugging.
    """

    response: str = ""
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    model_used: str | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: list[TraceEntry] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    # ---- convenience helpers ------------------------------------------------

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls that occurred."""
        return len(self.tool_calls)

    @property
    def tool_names(self) -> list[str]:
        """Ordered list of tool names that were called."""
        return [tc.name for tc in self.tool_calls]

    def get_tool_call(self, name: str) -> ToolCall | None:
        """Return the first tool call with the given name, or ``None``."""
        for tc in self.tool_calls:
            if tc.name == name:
                return tc
        return None

    def get_all_tool_calls(self, name: str) -> list[ToolCall]:
        """Return all tool calls with the given name."""
        return [tc for tc in self.tool_calls if tc.name == name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "response": self.response,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "model_used": self.model_used,
            "headers": self.headers,
            "metadata": self.metadata,
        }
