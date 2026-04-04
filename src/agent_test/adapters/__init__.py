"""Framework adapter protocol and stubs.

Defines the base protocol for adapting agent-test to different
agent frameworks: OpenAI Agents, LangGraph, PydanticAI, CrewAI.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_test.result import AgentResult


@runtime_checkable
class FrameworkAdapter(Protocol):
    """Protocol that framework adapters must implement.

    An adapter bridges between agent-test's mock infrastructure and
    a specific agent framework's execution model.
    """

    @property
    def framework_name(self) -> str:
        """Return the name of the framework this adapter supports."""
        ...

    async def invoke(
        self,
        message: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent with the given message and return a result."""
        ...


class OpenAIAgentsAdapter:
    """Stub adapter for OpenAI Agents SDK.

    This is a placeholder for the full integration. Override
    ``invoke()`` to wire up the OpenAI Agents SDK.
    """

    @property
    def framework_name(self) -> str:
        return "openai-agents"

    async def invoke(
        self,
        message: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        raise NotImplementedError(
            "OpenAI Agents adapter is a stub. "
            "Implement invoke() to connect to the OpenAI Agents SDK."
        )


class LangGraphAdapter:
    """Stub adapter for LangGraph."""

    @property
    def framework_name(self) -> str:
        return "langgraph"

    async def invoke(
        self,
        message: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        raise NotImplementedError(
            "LangGraph adapter is a stub. Implement invoke() to connect to LangGraph."
        )


class PydanticAIAdapter:
    """Stub adapter for PydanticAI."""

    @property
    def framework_name(self) -> str:
        return "pydantic-ai"

    async def invoke(
        self,
        message: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        raise NotImplementedError(
            "PydanticAI adapter is a stub. Implement invoke() to connect to PydanticAI."
        )


class CrewAIAdapter:
    """Stub adapter for CrewAI."""

    @property
    def framework_name(self) -> str:
        return "crewai"

    async def invoke(
        self,
        message: str,
        *,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        raise NotImplementedError(
            "CrewAI adapter is a stub. Implement invoke() to connect to CrewAI."
        )


__all__ = [
    "CrewAIAdapter",
    "FrameworkAdapter",
    "LangGraphAdapter",
    "OpenAIAgentsAdapter",
    "PydanticAIAdapter",
]
