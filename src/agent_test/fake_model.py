"""Deterministic fake model adapter.

Removes all nondeterminism by providing scripted or rule-based responses
instead of calling a real LLM endpoint.
"""

from __future__ import annotations

import copy
import re
from typing import Any


class FakeModel:
    """Deterministic model that returns scripted or rule-based responses.

    Supports three modes:

    1. **Scripted sequence** — returns responses in order, one per call.
    2. **Pattern matching** — matches input message against regex rules.
    3. **Default response** — fallback when no rule matches.

    Parameters
    ----------
    responses:
        A list of response dicts to return in sequence.  Each dict should
        have either ``"content"`` (text reply) or ``"tool_calls"``
        (list of ``{"name": ..., "args": ...}`` dicts).
    rules:
        A list of ``(pattern, response_dict)`` tuples.  The first rule
        whose ``pattern`` matches the last user message wins.
    default_response:
        Response dict to return when no scripted response remains and
        no rule matches.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        rules: list[tuple[str, dict[str, Any]]] | None = None,
        default_response: dict[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._rules = list(rules or [])
        self._default = default_response or {"content": ""}
        self._call_index = 0
        self._call_log: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate a deterministic response.

        Parameters
        ----------
        messages:
            Conversation history (list of ``{"role": ..., "content": ...}`` dicts).
        tools:
            Available tool catalog (for context; fake model may or may not use it).
        headers:
            Request headers (for recording / context).

        Returns
        -------
        A dict with either ``"content"`` or ``"tool_calls"`` key.
        """
        self._call_log.append(
            {
                "messages": copy.deepcopy(messages),
                "tools": tools,
                "headers": headers,
            }
        )

        # Mode 1: Scripted sequence
        if self._call_index < len(self._responses):
            response = copy.deepcopy(self._responses[self._call_index])
            self._call_index += 1
            return response

        # Mode 2: Pattern matching — check last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        for pattern, response in self._rules:
            if re.search(pattern, last_user_msg, re.IGNORECASE):
                return copy.deepcopy(response)

        # Mode 3: Default
        return copy.deepcopy(self._default)

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Log of all generate() calls made to this model."""
        return list(self._call_log)

    @property
    def call_count(self) -> int:
        """Number of times generate() was called."""
        return len(self._call_log)

    def reset(self) -> None:
        """Reset the model state (call index and log)."""
        self._call_index = 0
        self._call_log.clear()
