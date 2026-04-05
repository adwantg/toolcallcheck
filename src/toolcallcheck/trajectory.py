"""Golden trajectory assertions.

Validate the full sequence of messages and tool calls against expected
trajectories, supporting exact, subset, and ordered-subset matching.
"""

from __future__ import annotations

from typing import Any

from toolcallcheck.result import AgentResult


def assert_trajectory(
    result: AgentResult,
    expected: list[dict[str, Any]],
    *,
    mode: str = "exact",
) -> None:
    """Assert that the agent's execution trajectory matches expectations.

    Each entry in ``expected`` is a dict with:

    - ``{"role": "user", "content": "..."}``
    - ``{"role": "assistant", "content": "..."}``
    - ``{"role": "tool", "name": "...", "args": {...}}``

    Parameters
    ----------
    result:
        The :class:`AgentResult` to inspect.
    expected:
        The expected trajectory as a list of step dicts.
    mode:
        - ``"exact"`` — trajectory must match exactly (default).
        - ``"subset"`` — every expected step must appear, but extras are OK.
        - ``"ordered_subset"`` — expected steps must appear in order, but
          gaps between them are allowed.
    """
    actual = _build_trajectory(result)

    if mode == "exact":
        if actual != expected:
            _raise_trajectory_error(expected, actual, "exact")
    elif mode == "subset":
        for step in expected:
            if step not in actual:
                _raise_trajectory_error(
                    expected,
                    actual,
                    "subset",
                    missing_step=step,
                )
    elif mode == "ordered_subset":
        _check_ordered_subset(expected, actual)
    else:
        raise ValueError(f"Unknown trajectory mode: {mode!r}")


def _build_trajectory(result: AgentResult) -> list[dict[str, Any]]:
    """Convert an AgentResult's trace into a trajectory list."""
    trajectory: list[dict[str, Any]] = []

    for entry in result.trace:
        if entry.role == "user":
            trajectory.append({"role": "user", "content": entry.content or ""})
        elif entry.role == "assistant" and entry.content is not None:
            trajectory.append({"role": "assistant", "content": entry.content})
        elif entry.role == "tool" and entry.tool_call is not None:
            trajectory.append(
                {
                    "role": "tool",
                    "name": entry.tool_call.name,
                    "args": entry.tool_call.args,
                }
            )

    return trajectory


def _check_ordered_subset(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> None:
    """Check that expected steps appear in order within actual."""
    actual_idx = 0
    for exp_step in expected:
        found = False
        while actual_idx < len(actual):
            if actual[actual_idx] == exp_step:
                found = True
                actual_idx += 1
                break
            actual_idx += 1
        if not found:
            _raise_trajectory_error(expected, actual, "ordered_subset", missing_step=exp_step)


def _raise_trajectory_error(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    mode: str,
    missing_step: dict[str, Any] | None = None,
) -> None:
    """Format and raise a trajectory mismatch error."""
    parts = [f"Trajectory mismatch (mode={mode}):"]
    parts.append(f"  Expected ({len(expected)} steps):")
    for i, step in enumerate(expected):
        parts.append(f"    [{i}] {_step_summary(step)}")
    parts.append(f"  Actual ({len(actual)} steps):")
    for i, step in enumerate(actual):
        parts.append(f"    [{i}] {_step_summary(step)}")
    if missing_step:
        parts.append(f"  Missing step: {_step_summary(missing_step)}")
    raise AssertionError("\n".join(parts))


def _step_summary(step: dict[str, Any]) -> str:
    """One-line summary of a trajectory step."""
    role = step.get("role", "?")
    if role == "tool":
        name = step.get("name", "?")
        args = step.get("args", {})
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"tool:{name}({arg_str})"
    content = step.get("content", "")
    if len(content) > 60:
        content = content[:57] + "..."
    return f"{role}: {content!r}"
