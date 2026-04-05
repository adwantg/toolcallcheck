"""Tests for AgentResult and ToolCall."""

from __future__ import annotations

from toolcallcheck.result import AgentResult, ToolCall, TraceEntry


class TestToolCall:
    def test_create_success(self):
        tc = ToolCall(name="create_user", args={"email": "a@example.com"}, response={"ok": True})
        assert tc.name == "create_user"
        assert tc.args == {"email": "a@example.com"}
        assert tc.response == {"ok": True}
        assert tc.error is None

    def test_create_error(self):
        tc = ToolCall(name="fail", args={}, error="timeout")
        assert tc.error == "timeout"
        assert tc.response is None

    def test_to_dict_success(self):
        tc = ToolCall(name="t", args={"x": 1}, response="ok")
        d = tc.to_dict()
        assert d == {"name": "t", "args": {"x": 1}, "response": "ok"}

    def test_to_dict_error(self):
        tc = ToolCall(name="t", args={}, error="fail")
        d = tc.to_dict()
        assert d == {"name": "t", "args": {}, "error": "fail"}

    def test_to_dict_minimal(self):
        tc = ToolCall(name="t", args={})
        d = tc.to_dict()
        assert d == {"name": "t", "args": {}}

    def test_frozen(self):
        tc = ToolCall(name="t", args={})
        import dataclasses

        with __import__("pytest").raises(dataclasses.FrozenInstanceError):
            tc.name = "other"  # type: ignore[misc]


class TestAgentResult:
    def test_empty_result(self):
        r = AgentResult()
        assert r.response == ""
        assert r.tool_calls == []
        assert r.model_used is None
        assert r.headers == {}
        assert r.trace == []
        assert r.metadata == {}

    def test_tool_call_count(self):
        r = AgentResult(
            tool_calls=[
                ToolCall(name="a", args={}),
                ToolCall(name="b", args={}),
            ]
        )
        assert r.tool_call_count == 2

    def test_tool_names(self):
        r = AgentResult(
            tool_calls=[
                ToolCall(name="x", args={}),
                ToolCall(name="y", args={}),
                ToolCall(name="x", args={}),
            ]
        )
        assert r.tool_names == ["x", "y", "x"]

    def test_get_tool_call(self):
        tc = ToolCall(name="target", args={"k": "v"})
        r = AgentResult(tool_calls=[ToolCall(name="other", args={}), tc])
        assert r.get_tool_call("target") is tc
        assert r.get_tool_call("missing") is None

    def test_get_all_tool_calls(self):
        r = AgentResult(
            tool_calls=[
                ToolCall(name="a", args={"i": 1}),
                ToolCall(name="b", args={}),
                ToolCall(name="a", args={"i": 2}),
            ]
        )
        all_a = r.get_all_tool_calls("a")
        assert len(all_a) == 2
        assert all_a[0].args == {"i": 1}
        assert all_a[1].args == {"i": 2}

    def test_to_dict(self):
        r = AgentResult(
            response="Hello",
            tool_calls=[ToolCall(name="t", args={"x": 1}, response="ok")],
            model_used="gpt-4",
            headers={"X-Key": "val"},
            metadata={"turns": 1},
        )
        d = r.to_dict()
        assert d["response"] == "Hello"
        assert len(d["tool_calls"]) == 1
        assert d["model_used"] == "gpt-4"
        assert d["headers"] == {"X-Key": "val"}
        assert d["metadata"] == {"turns": 1}


class TestTraceEntry:
    def test_user_entry(self):
        e = TraceEntry(role="user", content="hello")
        assert e.role == "user"
        assert e.content == "hello"
        assert e.tool_call is None

    def test_tool_entry(self):
        tc = ToolCall(name="t", args={})
        e = TraceEntry(role="tool", tool_call=tc)
        assert e.tool_call is tc
