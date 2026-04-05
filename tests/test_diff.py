"""Tests for structured diff output."""

from __future__ import annotations

from toolcallcheck.diff import format_tool_call_diff, format_value_diff


class TestFormatToolCallDiff:
    def test_matching_calls(self):
        calls = [{"name": "t", "args": {"x": 1}}]
        output = format_tool_call_diff(calls, calls)
        assert "✓" in output

    def test_mismatch(self):
        expected = [{"name": "t", "args": {"x": 1}}]
        actual = [{"name": "t", "args": {"x": 2}}]
        output = format_tool_call_diff(expected, actual)
        assert "MISMATCH" in output

    def test_extra_actual(self):
        expected = [{"name": "a", "args": {}}]
        actual = [{"name": "a", "args": {}}, {"name": "b", "args": {}}]
        output = format_tool_call_diff(expected, actual)
        assert "UNEXPECTED" in output

    def test_missing_actual(self):
        expected = [{"name": "a", "args": {}}, {"name": "b", "args": {}}]
        actual = [{"name": "a", "args": {}}]
        output = format_tool_call_diff(expected, actual)
        assert "MISSING" in output

    def test_different_names(self):
        expected = [{"name": "expected", "args": {}}]
        actual = [{"name": "actual", "args": {}}]
        output = format_tool_call_diff(expected, actual)
        assert "expected" in output
        assert "actual" in output

    def test_arg_level_diff(self):
        expected = [{"name": "t", "args": {"a": 1, "b": 2}}]
        actual = [{"name": "t", "args": {"a": 1, "b": 99}}]
        output = format_tool_call_diff(expected, actual)
        assert "arg diff" in output


class TestFormatValueDiff:
    def test_string_diff(self):
        output = format_value_diff("hello", "world", label="test")
        assert "hello" in output
        assert "world" in output

    def test_list_diff(self):
        output = format_value_diff([1, 2, 3], [1, 2, 4], label="items")
        assert "items" in output

    def test_dict_diff(self):
        output = format_value_diff({"a": 1}, {"a": 2}, label="config")
        assert "config" in output

    def test_position_indicator(self):
        output = format_value_diff("abcd", "abxd", label="val")
        assert "position" in output
