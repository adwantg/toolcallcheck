"""Tests for trajectory assertions."""

from __future__ import annotations

import pytest

from agentharness.result import AgentResult, ToolCall, TraceEntry
from agentharness.trajectory import assert_trajectory


def _make_result(trace_entries):
    return AgentResult(trace=trace_entries)


class TestTrajectoryExact:
    def test_exact_match(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="Hello"),
                TraceEntry(role="assistant", content="Hi there"),
            ]
        )
        assert_trajectory(
            result,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

    def test_exact_with_tool(self):
        tc = ToolCall(name="search", args={"q": "test"})
        result = _make_result(
            [
                TraceEntry(role="user", content="Search for test"),
                TraceEntry(role="tool", tool_call=tc),
                TraceEntry(role="assistant", content="Found it"),
            ]
        )
        assert_trajectory(
            result,
            [
                {"role": "user", "content": "Search for test"},
                {"role": "tool", "name": "search", "args": {"q": "test"}},
                {"role": "assistant", "content": "Found it"},
            ],
        )

    def test_exact_mismatch_fails(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="Hello"),
            ]
        )
        with pytest.raises(AssertionError, match="Trajectory mismatch"):
            assert_trajectory(
                result,
                [
                    {"role": "user", "content": "Different"},
                ],
            )

    def test_extra_steps_fail(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="Hello"),
                TraceEntry(role="assistant", content="Hi"),
            ]
        )
        with pytest.raises(AssertionError):
            assert_trajectory(
                result,
                [
                    {"role": "user", "content": "Hello"},
                ],
            )


class TestTrajectorySubset:
    def test_subset_match(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="Hello"),
                TraceEntry(role="assistant", content="Hi"),
                TraceEntry(role="user", content="Bye"),
            ]
        )
        assert_trajectory(
            result,
            [
                {"role": "user", "content": "Hello"},
            ],
            mode="subset",
        )

    def test_subset_missing_step_fails(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="Hello"),
            ]
        )
        with pytest.raises(AssertionError, match="Missing step"):
            assert_trajectory(
                result,
                [
                    {"role": "user", "content": "Not here"},
                ],
                mode="subset",
            )


class TestTrajectoryOrderedSubset:
    def test_ordered_subset_match(self):
        tc = ToolCall(name="t", args={})
        result = _make_result(
            [
                TraceEntry(role="user", content="A"),
                TraceEntry(role="tool", tool_call=tc),
                TraceEntry(role="assistant", content="B"),
                TraceEntry(role="user", content="C"),
            ]
        )
        assert_trajectory(
            result,
            [
                {"role": "user", "content": "A"},
                {"role": "user", "content": "C"},
            ],
            mode="ordered_subset",
        )

    def test_ordered_subset_wrong_order_fails(self):
        result = _make_result(
            [
                TraceEntry(role="user", content="A"),
                TraceEntry(role="user", content="B"),
            ]
        )
        with pytest.raises(AssertionError):
            assert_trajectory(
                result,
                [
                    {"role": "user", "content": "B"},
                    {"role": "user", "content": "A"},
                ],
                mode="ordered_subset",
            )


class TestTrajectoryInvalidMode:
    def test_bad_mode_raises(self):
        result = _make_result([])
        with pytest.raises(ValueError, match="Unknown trajectory mode"):
            assert_trajectory(result, [], mode="invalid")
