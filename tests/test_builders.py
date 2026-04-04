"""Tests for test data builders."""

from __future__ import annotations

import pytest

from agent_test.builders import (
    BuiltScenario,
    ScenarioBuilder,
    ToolResponseBuilder,
    UserMessageBuilder,
)
from agent_test.mock_server import MockTool


class TestUserMessageBuilder:
    def test_basic_message(self):
        msg = UserMessageBuilder("Hello").build()
        assert msg == {"message": "Hello"}

    def test_with_token(self):
        msg = UserMessageBuilder("Hi").with_token("jwt-123").build()
        assert msg["access_token"] == "jwt-123"

    def test_with_site(self):
        msg = UserMessageBuilder("Hi").with_site("site-456").build()
        assert msg["site_id"] == "site-456"

    def test_with_headers(self):
        msg = UserMessageBuilder("Hi").with_header("X-A", "1").with_header("X-B", "2").build()
        assert msg["headers"] == {"X-A": "1", "X-B": "2"}

    def test_with_metadata(self):
        msg = UserMessageBuilder("Hi").with_metadata("test_id", "t1").build()
        assert msg["metadata"] == {"test_id": "t1"}

    def test_full_build(self):
        msg = (
            UserMessageBuilder("Create user Jane")
            .with_token("jwt")
            .with_site("123")
            .with_header("X-Custom", "val")
            .with_metadata("case", "happy")
            .build()
        )
        assert msg["message"] == "Create user Jane"
        assert msg["access_token"] == "jwt"
        assert msg["site_id"] == "123"
        assert msg["headers"]["X-Custom"] == "val"
        assert msg["metadata"]["case"] == "happy"


class TestToolResponseBuilder:
    def test_basic_tool(self):
        tool = ToolResponseBuilder("search").build()
        assert isinstance(tool, MockTool)
        assert tool.name == "search"

    def test_with_description(self):
        tool = ToolResponseBuilder("t").with_description("A tool").build()
        assert tool.description == "A tool"

    def test_with_params(self):
        tool = (
            ToolResponseBuilder("t")
            .with_param("name", "string")
            .with_param("optional", "string", required=False)
            .build()
        )
        assert "name" in tool.parameters
        assert tool.parameters["name"]["required"] is True
        assert tool.parameters["optional"]["required"] is False

    def test_with_response(self):
        tool = ToolResponseBuilder("t").with_response({"ok": True}).build()
        assert tool.response == {"ok": True}

    def test_with_conditional_response(self):
        def fn(args):
            return {"result": args.get("x")}

        tool = ToolResponseBuilder("t").with_conditional_response(fn).build()
        assert callable(tool.response)

    def test_with_error(self):
        tool = ToolResponseBuilder("t").with_error("fail", after=3).build()
        assert tool.error == "fail"
        assert tool.error_after == 3


class TestScenarioBuilder:
    def test_basic_scenario(self):
        s = (
            ScenarioBuilder("test")
            .with_tool(MockTool(name="t", response="ok"))
            .with_model_responses([{"content": "Done"}])
            .with_message("Do it")
            .build()
        )
        assert isinstance(s, BuiltScenario)
        assert s.name == "test"
        assert s.invocation["message"] == "Do it"

    def test_scenario_with_auth(self):
        s = (
            ScenarioBuilder("auth_test")
            .with_message("Hi")
            .with_token("jwt-123")
            .with_site_id("site-456")
            .with_header("X-Custom", "val")
            .with_model_responses([{"content": "ok"}])
            .build()
        )
        assert s.invocation["access_token"] == "jwt-123"
        assert s.invocation["site_id"] == "site-456"
        assert s.invocation["headers"]["X-Custom"] == "val"

    def test_scenario_with_model_name(self):
        s = (
            ScenarioBuilder("model_test")
            .with_model_name("Nova-pro")
            .with_model_responses([{"content": "ok"}])
            .with_message("Hi")
            .build()
        )
        assert s.runner._model_name == "Nova-pro"

    @pytest.mark.asyncio
    async def test_scenario_invoke(self):
        s = (
            ScenarioBuilder("invoke_test")
            .with_tool(MockTool(name="greet", response={"msg": "hello"}))
            .with_model_responses(
                [
                    {"tool_calls": [{"name": "greet", "args": {}}]},
                    {"content": "Greeted!"},
                ]
            )
            .with_message("Say hi")
            .build()
        )
        result = await s.runner.invoke(**s.invocation)
        assert result.response == "Greeted!"
        assert len(result.tool_calls) == 1
