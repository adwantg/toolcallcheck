"""Structured diff output for assertion failures.

Provides human-readable, CI-friendly diff formatting for tool calls,
JSON values, and text comparisons.
"""

from __future__ import annotations

import json
from typing import Any


def format_tool_call_diff(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> str:
    """Format a structured diff between expected and actual tool calls."""
    lines: list[str] = []

    max_len = max(len(expected), len(actual))

    for i in range(max_len):
        exp = expected[i] if i < len(expected) else None
        act = actual[i] if i < len(actual) else None

        if exp == act:
            lines.append(f"  [{i}] ✓ {_tool_summary(exp)}")
        elif exp is None:
            lines.append(f"  [{i}] + UNEXPECTED: {_tool_summary(act)}")
        elif act is None:
            lines.append(f"  [{i}] - MISSING:    {_tool_summary(exp)}")
        else:
            lines.append(f"  [{i}] ✗ MISMATCH:")
            lines.append(f"        expected: {_tool_summary(exp)}")
            lines.append(f"        actual:   {_tool_summary(act)}")

            # Show arg-level diff
            if exp and act and exp.get("name") == act.get("name"):
                arg_diff = _format_args_diff(exp.get("args", {}), act.get("args", {}))
                if arg_diff:
                    lines.append(f"        arg diff: {arg_diff}")

    return "\n".join(lines)


def format_value_diff(expected: Any, actual: Any, *, label: str = "value") -> str:
    """Format a generic value comparison diff."""
    lines = [f"  {label}:"]

    if isinstance(expected, str) and isinstance(actual, str):
        lines.append(f"    expected: {expected!r}")
        lines.append(f"    actual:   {actual!r}")

        # Show character-level diff for short strings
        if len(expected) < 200 and len(actual) < 200:
            _add_string_diff(lines, expected, actual)
    elif isinstance(expected, (list, dict)) and isinstance(actual, (list, dict)):
        exp_json = json.dumps(expected, indent=2, default=str)
        act_json = json.dumps(actual, indent=2, default=str)
        lines.append(f"    expected:\n      {exp_json}")
        lines.append(f"    actual:\n      {act_json}")
    else:
        lines.append(f"    expected: {expected!r}")
        lines.append(f"    actual:   {actual!r}")

    return "\n".join(lines)


def _tool_summary(tool: dict[str, Any] | None) -> str:
    """One-line summary of a tool call dict."""
    if tool is None:
        return "<none>"
    name = tool.get("name", "?")
    args = tool.get("args", {})
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{name}({arg_str})"


def _format_args_diff(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    """Show which arguments differ between expected and actual."""
    parts: list[str] = []

    all_keys = sorted(set(expected.keys()) | set(actual.keys()))
    for key in all_keys:
        if key not in expected:
            parts.append(f"+{key}={actual[key]!r}")
        elif key not in actual:
            parts.append(f"-{key}={expected[key]!r}")
        elif expected[key] != actual[key]:
            parts.append(f"~{key}: {expected[key]!r} → {actual[key]!r}")

    return ", ".join(parts) if parts else ""


def _add_string_diff(lines: list[str], expected: str, actual: str) -> None:
    """Add a character-level indicator showing where strings differ."""
    min_len = min(len(expected), len(actual))
    diff_pos = -1
    for i in range(min_len):
        if expected[i] != actual[i]:
            diff_pos = i
            break

    if diff_pos == -1 and len(expected) != len(actual):
        diff_pos = min_len

    if diff_pos >= 0:
        lines.append(f"    first diff at position {diff_pos}")
