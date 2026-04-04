"""Tests for AgentRunner."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from agentharness.fake_model import FakeModel
from agentharness.mock_server import MockMCPServer, MockTool
from agentharness.runner import AgentRunner


class TestAgentRunnerInit:
    def test_default_init(self):
        runner = AgentRunner()
        assert runner.mcp_server is not None
        assert runner.config == {}

    def test_init_with_mcp_server(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="test_identity", response="ok"))
        runner = AgentRunner(mcp_server=server)
        assert "test_identity" in runner.mcp_server

    def test_init_with_config_dict(self):
        runner = AgentRunner(config={"model_name": "Nova-pro"})
        assert runner.config["model_name"] == "Nova-pro"

    def test_init_with_config_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model_name": "test-model", "timeout": 30}, f)
            f.flush()
            runner = AgentRunner(config=f.name)
            assert runner.config["model_name"] == "test-model"
            assert runner.config["timeout"] == 30
        os.unlink(f.name)

    def test_model_name_from_config(self):
        runner = AgentRunner(config={"model_name": "gpt-4"})
        assert runner.config["model_name"] == "gpt-4"


class TestAgentRunnerInvoke:
    @pytest.mark.asyncio
    async def test_invoke_no_model_returns_empty(self):
        runner = AgentRunner()
        result = await runner.invoke("Hello")
        assert result.response == ""
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_invoke_with_simple_model(self):
        model = FakeModel(responses=[{"content": "Hi there!"}])
        runner = AgentRunner(model=model)
        result = await runner.invoke("Hello")
        assert result.response == "Hi there!"

    @pytest.mark.asyncio
    async def test_invoke_with_tool_call(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="get_weather",
                response={"temp": 72},
            )
        )

        model = FakeModel(
            responses=[
                {"tool_calls": [{"name": "get_weather", "args": {"city": "Seattle"}}]},
                {"content": "It's 72 degrees in Seattle."},
            ]
        )

        runner = AgentRunner(mcp_server=server, model=model, model_name="test-model")
        result = await runner.invoke("What's the weather?")

        assert result.response == "It's 72 degrees in Seattle."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "Seattle"}
        assert result.tool_calls[0].response == {"temp": 72}
        assert result.model_used == "test-model"

    @pytest.mark.asyncio
    async def test_invoke_with_auth_headers(self):
        model = FakeModel(responses=[{"content": "ok"}])
        runner = AgentRunner(
            model=model,
            default_headers={"X-Default": "yes"},
        )

        result = await runner.invoke(
            "Hi",
            access_token="jwt-123",
            site_id="site-456",
            headers={"X-Custom": "val"},
        )

        assert result.headers["X-Access-Token"] == "jwt-123"
        assert result.headers["X-Site-Id"] == "site-456"
        assert result.headers["X-Custom"] == "val"
        assert result.headers["X-Default"] == "yes"

    @pytest.mark.asyncio
    async def test_invoke_metadata(self):
        model = FakeModel(responses=[{"content": "ok"}])
        runner = AgentRunner(model=model)
        result = await runner.invoke("Hi", metadata={"test_id": "t1"})
        assert result.metadata["test_id"] == "t1"
        assert "turns" in result.metadata

    @pytest.mark.asyncio
    async def test_invoke_trace(self):
        model = FakeModel(responses=[{"content": "Response"}])
        runner = AgentRunner(model=model)
        result = await runner.invoke("Hello")

        assert len(result.trace) == 2  # user + assistant
        assert result.trace[0].role == "user"
        assert result.trace[0].content == "Hello"
        assert result.trace[1].role == "assistant"
        assert result.trace[1].content == "Response"

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Agent that keeps calling tools should be stopped by max_turns."""
        server = MockMCPServer()
        server.add_tool(MockTool(name="loop", response="ok"))

        # Model always wants to call tools (infinite loop)
        model = FakeModel(
            default_response={"tool_calls": [{"name": "loop", "args": {}}]},
        )

        runner = AgentRunner(mcp_server=server, model=model, max_turns=3)
        result = await runner.invoke("Start looping")

        # Should stop after max_turns
        assert result.metadata["turns"] == 3

    @pytest.mark.asyncio
    async def test_tool_error_in_trace(self):
        server = MockMCPServer()
        server.add_tool(MockTool(name="broken", error="fail"))

        model = FakeModel(
            responses=[
                {"tool_calls": [{"name": "broken", "args": {}}]},
                {"content": "Tool failed"},
            ]
        )

        runner = AgentRunner(mcp_server=server, model=model)
        result = await runner.invoke("Try")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None


class TestSyncInvoke:
    def test_sync_invoke(self):
        model = FakeModel(responses=[{"content": "Sync reply"}])
        runner = AgentRunner(model=model)
        result = runner.sync_invoke("Hello")
        assert result.response == "Sync reply"
