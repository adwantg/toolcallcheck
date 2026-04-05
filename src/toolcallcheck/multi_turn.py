"""Multi-turn conversation testing.

Allows sending multiple user messages to the same agent session and
asserting behavior after each turn or after the full interaction.
"""

from __future__ import annotations

from typing import Any

from toolcallcheck.result import AgentResult
from toolcallcheck.runner import AgentRunner


class Conversation:
    """Stateful multi-turn conversation session.

    Wraps an :class:`AgentRunner` and preserves conversation history
    across turns so you can test multi-step agent workflows.

    Usage::

        conv = Conversation(runner)
        r1 = await conv.say("Hello")
        assert_no_tool_calls(r1)

        r2 = await conv.say("Create user Jane Doe jane@example.com")
        assert_tool_calls(r2, [...])

        assert conv.turn_count == 2
    """

    def __init__(self, runner: AgentRunner) -> None:
        self._runner = runner
        self._results: list[AgentResult] = []
        self._history: list[dict[str, Any]] = []

    async def say(
        self,
        message: str,
        *,
        access_token: str | None = None,
        site_id: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Send a message and get the agent's response.

        Conversation state is automatically carried forward.
        """
        result = await self._runner.invoke(
            message,
            access_token=access_token,
            site_id=site_id,
            headers=headers,
            metadata=metadata,
            _prior_messages=self._history,
        )

        self._results.append(result)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": result.response})

        return result

    def say_sync(
        self,
        message: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous convenience wrapper for :meth:`say`."""
        result = self._runner.sync_invoke(
            message,
            _prior_messages=self._history,
            **kwargs,
        )
        self._results.append(result)
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": result.response})
        return result

    @property
    def turn_count(self) -> int:
        """Number of conversation turns completed."""
        return len(self._results)

    @property
    def results(self) -> list[AgentResult]:
        """All results from each turn, in order."""
        return list(self._results)

    @property
    def last_result(self) -> AgentResult | None:
        """The result from the most recent turn."""
        return self._results[-1] if self._results else None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Full conversation history as message dicts."""
        return list(self._history)

    @property
    def all_tool_calls(self) -> list[Any]:
        """All tool calls across all turns, in order."""
        calls = []
        for result in self._results:
            calls.extend(result.tool_calls)
        return calls

    def reset(self) -> None:
        """Clear conversation state."""
        self._results.clear()
        self._history.clear()
