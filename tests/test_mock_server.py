"""Tests for MockMCPServer and MockTool."""

from __future__ import annotations

from agentharness.mock_server import MockMCPServer, MockTool


class TestMockTool:
    """Tests for MockTool creation and configuration."""

    def test_create_basic_tool(self):
        tool = MockTool(name="greet", description="Say hello")
        assert tool.name == "greet"
        assert tool.description == "Say hello"
        assert tool.parameters == {}
        assert tool.response is None
        assert tool.error is None

    def test_create_tool_with_params_and_response(self):
        tool = MockTool(
            name="create_user",
            description="Create a user",
            parameters={
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            response={"status": "success", "userId": "USR-123"},
        )
        assert tool.name == "create_user"
        assert "name" in tool.parameters
        assert tool.response == {"status": "success", "userId": "USR-123"}

    def test_create_tool_with_error(self):
        tool = MockTool(
            name="fail_tool",
            error="Service unavailable",
        )
        assert tool.error == "Service unavailable"

    def test_create_tool_with_structured_error(self):
        tool = MockTool(
            name="fail_tool",
            error={"code": 503, "message": "Service unavailable"},
        )
        assert tool.error == {"code": 503, "message": "Service unavailable"}

    def test_tool_error_after(self):
        tool = MockTool(
            name="retry_tool",
            response={"ok": True},
            error="Timeout",
            error_after=2,
        )
        assert tool.error_after == 2


class TestMockMCPServer:
    """Tests for MockMCPServer registration and invocation."""

    def test_add_and_list_tools(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="tool_a", description="Tool A"))
        server.add_tool(MockTool(name="tool_b", description="Tool B"))

        tools = server.list_tools()
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_add_tools_batch(self):
        server = MockMCPServer()
        server.add_tools(
            [
                MockTool(name="a"),
                MockTool(name="b"),
                MockTool(name="c"),
            ]
        )
        assert len(server) == 3

    def test_remove_tool(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="temp"))
        assert "temp" in server
        server.remove_tool("temp")
        assert "temp" not in server

    def test_contains_check(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="exists"))
        assert "exists" in server
        assert "missing" not in server

    def test_canned_response(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="get_weather",
                response={"temp": 72, "unit": "F"},
            )
        )
        result = server.call_tool("get_weather", {"location": "Seattle"})
        assert result == {"result": {"temp": 72, "unit": "F"}}

    def test_conditional_response(self):
        def dynamic_response(args):
            if args.get("city") == "Seattle":
                return {"temp": 55}
            return {"temp": 85}

        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="weather",
                response=dynamic_response,
            )
        )

        r1 = server.call_tool("weather", {"city": "Seattle"})
        assert r1["result"]["temp"] == 55

        r2 = server.call_tool("weather", {"city": "Phoenix"})
        assert r2["result"]["temp"] == 85

    def test_error_injection(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="broken",
                error="Connection failed",
            )
        )
        result = server.call_tool("broken", {})
        assert "error" in result
        assert result["error"]["message"] == "Connection failed"

    def test_structured_error_injection(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="broken",
                error={"code": 500, "message": "Internal error"},
            )
        )
        result = server.call_tool("broken", {})
        assert result["error"] == {"code": 500, "message": "Internal error"}

    def test_error_after_n_calls(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="flaky",
                response={"ok": True},
                error="Timeout",
                error_after=2,
            )
        )

        # First two calls succeed
        assert "result" in server.call_tool("flaky", {})
        assert "result" in server.call_tool("flaky", {})

        # Third call fails
        r3 = server.call_tool("flaky", {})
        assert "error" in r3

    def test_unknown_tool(self):
        server = MockMCPServer()
        result = server.call_tool("nonexistent", {})
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_missing_required_params(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="strict",
                parameters={
                    "required_field": {"type": "string", "required": True},
                },
                response={"ok": True},
            )
        )
        result = server.call_tool("strict", {})
        assert "error" in result
        assert "Missing required" in result["error"]

    def test_optional_params_not_required(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="flexible",
                parameters={
                    "optional_field": {"type": "string", "required": False},
                },
                response={"ok": True},
            )
        )
        result = server.call_tool("flexible", {})
        assert "result" in result

    def test_call_log(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="a", response="ok"))
        server.add_tool(MockTool(name="b", response="ok"))

        server.call_tool("a", {"x": 1})
        server.call_tool("b", {"y": 2})

        log = server.call_log
        assert len(log) == 2
        assert log[0]["name"] == "a"
        assert log[0]["args"] == {"x": 1}
        assert log[1]["name"] == "b"

    def test_reset(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="t", response="ok"))
        server.call_tool("t", {})
        assert len(server.call_log) == 1

        server.reset()
        assert len(server.call_log) == 0

    def test_get_tool(self):
        server = MockMCPServer()
        original = MockTool(name="my_tool", description="desc")
        server.add_tool(original)
        assert server.get_tool("my_tool") is original
        assert server.get_tool("missing") is None

    def test_tool_names_property(self):
        server = MockMCPServer()
        server.add_tools([MockTool(name="x"), MockTool(name="y")])
        assert server.tool_names == ["x", "y"]

    def test_multiple_tools_independent(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="add", response={"sum": 3}))
        server.add_tool(MockTool(name="multiply", response={"product": 6}))

        r1 = server.call_tool("add", {"a": 1, "b": 2})
        r2 = server.call_tool("multiply", {"a": 2, "b": 3})

        assert r1["result"]["sum"] == 3
        assert r2["result"]["product"] == 6

    def test_call_tool_none_args(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="no_args", response="done"))
        result = server.call_tool("no_args")
        assert result["result"] == "done"

    def test_response_is_deep_copied(self):
        """Verify that modifying a returned response doesn't affect future calls."""
        server = MockMCPServer()
        server.add_tool(MockTool(name="t", response={"items": [1, 2, 3]}))

        r1 = server.call_tool("t", {})
        r1["result"]["items"].append(4)  # Mutate the returned response

        r2 = server.call_tool("t", {})
        assert r2["result"]["items"] == [1, 2, 3]  # Original preserved
