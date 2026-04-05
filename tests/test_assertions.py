"""Tests for all assertion helpers."""

from __future__ import annotations

import pytest

from toolcallcheck.assertions import (
    assert_headers,
    assert_model_used,
    assert_no_tool_calls,
    assert_response_contains,
    assert_response_equals,
    assert_response_matches,
    assert_tool_args_contain,
    assert_tool_call_count,
    assert_tool_call_order,
    assert_tool_calls,
)
from toolcallcheck.result import AgentResult, ToolCall

# ---- Fixtures ---------------------------------------------------------------


def _result_with_tools(*tool_tuples):
    """Helper: create result with tool calls from (name, args) tuples."""
    return AgentResult(
        tool_calls=[ToolCall(name=n, args=a) for n, a in tool_tuples],
    )


def _result_with_response(text):
    return AgentResult(response=text)


# ---- assert_tool_calls ------------------------------------------------------


class TestAssertToolCalls:
    def test_exact_match(self):
        result = _result_with_tools(
            ("create_user", {"email": "a@example.com"}),
        )
        assert_tool_calls(result, [{"name": "create_user", "args": {"email": "a@example.com"}}])

    def test_multiple_tools(self):
        result = _result_with_tools(
            ("lookup", {"id": 1}),
            ("update", {"id": 1, "status": "active"}),
        )
        assert_tool_calls(
            result,
            [
                {"name": "lookup", "args": {"id": 1}},
                {"name": "update", "args": {"id": 1, "status": "active"}},
            ],
        )

    def test_wrong_tool_name_fails(self):
        result = _result_with_tools(("actual_tool", {}))
        with pytest.raises(AssertionError, match="Tool calls do not match"):
            assert_tool_calls(result, [{"name": "expected_tool", "args": {}}])

    def test_wrong_args_fails(self):
        result = _result_with_tools(("t", {"a": 1}))
        with pytest.raises(AssertionError):
            assert_tool_calls(result, [{"name": "t", "args": {"a": 2}}])

    def test_extra_tool_call_fails(self):
        result = _result_with_tools(("a", {}), ("b", {}))
        with pytest.raises(AssertionError):
            assert_tool_calls(result, [{"name": "a", "args": {}}])

    def test_missing_tool_call_fails(self):
        result = _result_with_tools(("a", {}))
        with pytest.raises(AssertionError):
            assert_tool_calls(
                result,
                [
                    {"name": "a", "args": {}},
                    {"name": "b", "args": {}},
                ],
            )

    def test_any_order(self):
        result = _result_with_tools(("b", {}), ("a", {}))
        assert_tool_calls(
            result,
            [
                {"name": "a", "args": {}},
                {"name": "b", "args": {}},
            ],
            strict_order=False,
        )

    def test_any_order_mismatch_fails(self):
        result = _result_with_tools(("a", {}))
        with pytest.raises(AssertionError):
            assert_tool_calls(
                result,
                [
                    {"name": "a", "args": {}},
                    {"name": "b", "args": {}},
                ],
                strict_order=False,
            )

    def test_empty_expected_and_actual(self):
        result = AgentResult()
        assert_tool_calls(result, [])


# ---- assert_tool_call_count -------------------------------------------------


class TestAssertToolCallCount:
    def test_correct_count(self):
        result = _result_with_tools(("a", {}), ("b", {}))
        assert_tool_call_count(result, 2)

    def test_zero_count(self):
        result = AgentResult()
        assert_tool_call_count(result, 0)

    def test_wrong_count_fails(self):
        result = _result_with_tools(("a", {}))
        with pytest.raises(AssertionError, match="Expected 2 tool call"):
            assert_tool_call_count(result, 2)


# ---- assert_no_tool_calls ---------------------------------------------------


class TestAssertNoToolCalls:
    def test_no_calls_passes(self):
        result = AgentResult()
        assert_no_tool_calls(result)

    def test_with_calls_fails(self):
        result = _result_with_tools(("surprise", {}))
        with pytest.raises(AssertionError, match="Expected no tool calls"):
            assert_no_tool_calls(result)


# ---- assert_tool_call_order -------------------------------------------------


class TestAssertToolCallOrder:
    def test_correct_order(self):
        result = _result_with_tools(("a", {}), ("b", {}), ("c", {}))
        assert_tool_call_order(result, ["a", "b", "c"])

    def test_wrong_order_fails(self):
        result = _result_with_tools(("b", {}), ("a", {}))
        with pytest.raises(AssertionError, match="Tool call order mismatch"):
            assert_tool_call_order(result, ["a", "b"])


# ---- assert_tool_args_contain -----------------------------------------------


class TestAssertToolArgsContain:
    def test_partial_match(self):
        result = _result_with_tools(
            ("create_user", {"name": "Jane", "role": "admin", "timestamp": "2024-01-01"}),
        )
        assert_tool_args_contain(result, "create_user", {"name": "Jane", "role": "admin"})

    def test_missing_key_fails(self):
        result = _result_with_tools(("t", {"a": 1}))
        with pytest.raises(AssertionError, match="Missing keys"):
            assert_tool_args_contain(result, "t", {"a": 1, "b": 2})

    def test_wrong_value_fails(self):
        result = _result_with_tools(("t", {"a": 1}))
        with pytest.raises(AssertionError, match="expected"):
            assert_tool_args_contain(result, "t", {"a": 999})

    def test_tool_not_called_fails(self):
        result = AgentResult()
        with pytest.raises(AssertionError, match="never called"):
            assert_tool_args_contain(result, "missing", {"a": 1})

    def test_call_index(self):
        result = _result_with_tools(
            ("t", {"i": "first"}),
            ("t", {"i": "second"}),
        )
        assert_tool_args_contain(result, "t", {"i": "second"}, call_index=1)

    def test_bad_call_index_fails(self):
        result = _result_with_tools(("t", {}))
        with pytest.raises(AssertionError, match="call_index=5"):
            assert_tool_args_contain(result, "t", {}, call_index=5)


# ---- assert_response_contains -----------------------------------------------


class TestAssertResponseContains:
    def test_contains(self):
        result = _result_with_response("The user has been created successfully.")
        assert_response_contains(result, "created")
        assert_response_contains(result, "successfully")

    def test_not_contains_fails(self):
        result = _result_with_response("Hello world")
        with pytest.raises(AssertionError, match="does not contain"):
            assert_response_contains(result, "goodbye")


# ---- assert_response_matches ------------------------------------------------


class TestAssertResponseMatches:
    def test_regex_match(self):
        result = _result_with_response("User USR-123 created.")
        assert_response_matches(result, r"USR-\d+")

    def test_regex_no_match_fails(self):
        result = _result_with_response("No ID here")
        with pytest.raises(AssertionError, match="does not match"):
            assert_response_matches(result, r"USR-\d+")


# ---- assert_response_equals -------------------------------------------------


class TestAssertResponseEquals:
    def test_exact_match(self):
        result = _result_with_response("exact text")
        assert_response_equals(result, "exact text")

    def test_exact_mismatch_fails(self):
        result = _result_with_response("actual")
        with pytest.raises(AssertionError, match="does not match exactly"):
            assert_response_equals(result, "expected")


# ---- assert_model_used ------------------------------------------------------


class TestAssertModelUsed:
    def test_correct_model(self):
        result = AgentResult(model_used="Nova-pro")
        assert_model_used(result, "Nova-pro")

    def test_wrong_model_fails(self):
        result = AgentResult(model_used="gpt-4")
        with pytest.raises(AssertionError, match="Expected model"):
            assert_model_used(result, "Nova-pro")


# ---- assert_headers ---------------------------------------------------------


class TestAssertHeaders:
    def test_subset_match(self):
        result = AgentResult(headers={"X-Token": "abc", "X-Other": "val"})
        assert_headers(result, {"X-Token": "abc"})

    def test_missing_header_fails(self):
        result = AgentResult(headers={"X-Token": "abc"})
        with pytest.raises(AssertionError, match="Missing headers"):
            assert_headers(result, {"X-Missing": "val"})

    def test_wrong_value_fails(self):
        result = AgentResult(headers={"X-Token": "abc"})
        with pytest.raises(AssertionError, match="expected"):
            assert_headers(result, {"X-Token": "wrong"})

    def test_exact_mode(self):
        result = AgentResult(headers={"X-Token": "abc"})
        assert_headers(result, {"X-Token": "abc"}, exact=True)

    def test_exact_mode_extra_keys_fail(self):
        result = AgentResult(headers={"X-Token": "abc", "X-Extra": "val"})
        with pytest.raises(AssertionError, match="Unexpected keys"):
            assert_headers(result, {"X-Token": "abc"}, exact=True)
