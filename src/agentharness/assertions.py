"""Assertion helpers for agentharness.

Every function raises ``AssertionError`` with a structured, human-readable
diff message on failure so that CI output is immediately actionable.
"""

from __future__ import annotations

import re
from typing import Any

from agentharness.diff import format_tool_call_diff, format_value_diff
from agentharness.result import AgentResult

# ---------------------------------------------------------------------------
# Tool-call assertions
# ---------------------------------------------------------------------------


def assert_tool_calls(
    result: AgentResult,
    expected: list[dict[str, Any]],
    *,
    strict_order: bool = True,
) -> None:
    """Assert that the agent made exactly the expected tool calls.

    Parameters
    ----------
    result:
        The :class:`AgentResult` to inspect.
    expected:
        A list of dicts, each with ``"name"`` and ``"args"`` keys.
    strict_order:
        If ``True`` (default), the order of tool calls must match.
    """
    actual = [{"name": tc.name, "args": tc.args} for tc in result.tool_calls]

    if strict_order:
        if actual != expected:
            diff = format_tool_call_diff(expected, actual)
            raise AssertionError(f"Tool calls do not match (strict order):\n{diff}")
    else:
        # Compare as sets — normalize by sorting
        def _sort_key(d: dict[str, Any]) -> str:
            return d["name"] + str(sorted(d.get("args", {}).items()))

        if sorted(actual, key=_sort_key) != sorted(expected, key=_sort_key):
            diff = format_tool_call_diff(expected, actual)
            raise AssertionError(f"Tool calls do not match (any order):\n{diff}")


def assert_tool_call_count(result: AgentResult, expected_count: int) -> None:
    """Assert the number of tool calls that occurred."""
    actual = result.tool_call_count
    if actual != expected_count:
        raise AssertionError(
            f"Expected {expected_count} tool call(s), got {actual}.\n"
            f"  Actual calls: {result.tool_names}"
        )


def assert_no_tool_calls(result: AgentResult) -> None:
    """Assert that the agent responded without invoking any tool."""
    if result.tool_calls:
        names = result.tool_names
        raise AssertionError(
            f"Expected no tool calls, but {len(names)} call(s) were made:\n  Tools called: {names}"
        )


def assert_tool_call_order(result: AgentResult, expected_names: list[str]) -> None:
    """Assert that tools were called in the specified order.

    Only checks the order of the *named* tools; ignores other calls.
    """
    actual_names = result.tool_names
    if actual_names != expected_names:
        diff = format_value_diff(expected_names, actual_names, label="tool call order")
        raise AssertionError(f"Tool call order mismatch:\n{diff}")


def assert_tool_args_contain(
    result: AgentResult,
    tool_name: str,
    partial_args: dict[str, Any],
    *,
    call_index: int = 0,
) -> None:
    """Assert that a tool call's arguments contain the given subset.

    Useful when the agent adds auto-generated fields (timestamps, IDs)
    that you don't want to assert on.

    Parameters
    ----------
    result:
        The :class:`AgentResult` to inspect.
    tool_name:
        Name of the tool to check.
    partial_args:
        A dict of key-value pairs that must be present in the actual args.
    call_index:
        If the tool was called multiple times, which invocation to check
        (0-indexed, default 0).
    """
    matching = result.get_all_tool_calls(tool_name)
    if not matching:
        raise AssertionError(
            f"Tool '{tool_name}' was never called.\n  Actual calls: {result.tool_names}"
        )

    if call_index >= len(matching):
        raise AssertionError(
            f"Tool '{tool_name}' was called {len(matching)} time(s), "
            f"but call_index={call_index} requested."
        )

    actual_args = matching[call_index].args
    missing: dict[str, Any] = {}
    mismatched: dict[str, tuple[Any, Any]] = {}

    for key, expected_val in partial_args.items():
        if key not in actual_args:
            missing[key] = expected_val
        elif actual_args[key] != expected_val:
            mismatched[key] = (expected_val, actual_args[key])

    if missing or mismatched:
        parts: list[str] = [f"Partial argument mismatch for tool '{tool_name}':"]
        if missing:
            parts.append(f"  Missing keys: {missing}")
        if mismatched:
            for k, (exp, act) in mismatched.items():
                parts.append(f"  Key '{k}': expected {exp!r}, got {act!r}")
        raise AssertionError("\n".join(parts))


# ---------------------------------------------------------------------------
# Response assertions
# ---------------------------------------------------------------------------


def assert_response_contains(result: AgentResult, substring: str) -> None:
    """Assert that the agent's text response contains the given substring."""
    if substring not in result.response:
        raise AssertionError(
            f"Response does not contain expected substring.\n"
            f"  Expected substring: {substring!r}\n"
            f"  Actual response:    {result.response!r}"
        )


def assert_response_matches(result: AgentResult, pattern: str) -> None:
    """Assert that the agent's text response matches the given regex pattern."""
    if not re.search(pattern, result.response):
        raise AssertionError(
            f"Response does not match expected pattern.\n"
            f"  Pattern:         {pattern!r}\n"
            f"  Actual response: {result.response!r}"
        )


def assert_response_equals(result: AgentResult, expected: str) -> None:
    """Assert that the agent's text response exactly equals the expected string."""
    if result.response != expected:
        diff = format_value_diff(expected, result.response, label="response")
        raise AssertionError(f"Response does not match exactly:\n{diff}")


# ---------------------------------------------------------------------------
# Model / routing assertions
# ---------------------------------------------------------------------------


def assert_model_used(result: AgentResult, expected_model: str) -> None:
    """Assert that the agent used the expected model or routing strategy."""
    if result.model_used != expected_model:
        raise AssertionError(f"Expected model '{expected_model}', got '{result.model_used}'.")


# ---------------------------------------------------------------------------
# Header assertions
# ---------------------------------------------------------------------------


def assert_headers(
    result: AgentResult,
    expected_headers: dict[str, str],
    *,
    exact: bool = False,
) -> None:
    """Assert that the captured request headers contain the expected values.

    Parameters
    ----------
    result:
        The :class:`AgentResult` to inspect.
    expected_headers:
        Key-value pairs that must be present.
    exact:
        If ``True``, the headers must match exactly with no extra keys.
    """
    actual = result.headers

    if exact and set(actual.keys()) != set(expected_headers.keys()):
        extra = set(actual.keys()) - set(expected_headers.keys())
        missing_keys = set(expected_headers.keys()) - set(actual.keys())
        parts = ["Headers do not match exactly:"]
        if extra:
            parts.append(f"  Unexpected keys: {extra}")
        if missing_keys:
            parts.append(f"  Missing keys: {missing_keys}")
        raise AssertionError("\n".join(parts))

    missing: dict[str, str] = {}
    mismatched: dict[str, tuple[str, str | None]] = {}

    for key, expected_val in expected_headers.items():
        if key not in actual:
            missing[key] = expected_val
        elif actual[key] != expected_val:
            mismatched[key] = (expected_val, actual[key])

    if missing or mismatched:
        parts = ["Header assertion failed:"]
        if missing:
            parts.append(f"  Missing headers: {missing}")
        if mismatched:
            for k, (exp, act) in mismatched.items():
                parts.append(f"  Header '{k}': expected {exp!r}, got {act!r}")
        raise AssertionError("\n".join(parts))
