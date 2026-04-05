"""Tests for multi-turn Conversation."""

from __future__ import annotations

import pytest

from toolcallcheck.assertions import assert_no_tool_calls
from toolcallcheck.fake_model import FakeModel
from toolcallcheck.mock_server import MockMCPServer, MockTool
from toolcallcheck.multi_turn import Conversation
from toolcallcheck.runner import AgentRunner


class TestConversation:
    @pytest.fixture
    def user_runner(self):
        server = MockMCPServer()
        server.add_tool(
            MockTool(
                name="create_user",
                response={"status": "created"},
            )
        )

        model = FakeModel(
            responses=[
                {"content": "I need the email address. What is it?"},
                {"tool_calls": [{"name": "create_user", "args": {"email": "a@example.com"}}]},
                {"content": "User created!"},
            ]
        )

        return AgentRunner(mcp_server=server, model=model)

    @pytest.mark.asyncio
    async def test_multi_turn_flow(self, user_runner):
        conv = Conversation(user_runner)

        r1 = await conv.say("Create a user")
        assert_no_tool_calls(r1)
        assert "email" in r1.response.lower()

        r2 = await conv.say("a@example.com")
        assert r2.response == "User created!"
        assert len(r2.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_turn_count(self, user_runner):
        conv = Conversation(user_runner)
        await conv.say("Turn 1")
        await conv.say("Turn 2")
        assert conv.turn_count == 2

    @pytest.mark.asyncio
    async def test_results_list(self, user_runner):
        conv = Conversation(user_runner)
        r1 = await conv.say("Turn 1")
        r2 = await conv.say("Turn 2")
        assert conv.results == [r1, r2]

    @pytest.mark.asyncio
    async def test_last_result(self, user_runner):
        conv = Conversation(user_runner)
        assert conv.last_result is None
        r1 = await conv.say("Hello")
        assert conv.last_result is r1

    @pytest.mark.asyncio
    async def test_history(self, user_runner):
        conv = Conversation(user_runner)
        await conv.say("Hello")
        h = conv.history
        assert len(h) == 2  # user + assistant
        assert h[0]["role"] == "user"
        assert h[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_history_is_passed_to_later_turns(self):
        model = FakeModel(
            responses=[
                {"content": "First reply"},
                {"content": "Second reply"},
            ]
        )
        runner = AgentRunner(model=model)
        conv = Conversation(runner)

        await conv.say("Hello")
        await conv.say("Follow up")

        assert model.call_log[1]["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "First reply"},
            {"role": "user", "content": "Follow up"},
        ]

    @pytest.mark.asyncio
    async def test_all_tool_calls(self, user_runner):
        conv = Conversation(user_runner)
        await conv.say("Turn 1")  # No tool calls
        await conv.say("Turn 2")  # Has tool call
        all_calls = conv.all_tool_calls
        assert len(all_calls) == 1
        assert all_calls[0].name == "create_user"

    @pytest.mark.asyncio
    async def test_reset(self, user_runner):
        conv = Conversation(user_runner)
        await conv.say("Hello")
        assert conv.turn_count == 1
        conv.reset()
        assert conv.turn_count == 0
        assert conv.history == []

    def test_say_sync(self):
        model = FakeModel(responses=[{"content": "Sync reply"}])
        runner = AgentRunner(model=model)
        conv = Conversation(runner)
        r = conv.say_sync("Hello")
        assert r.response == "Sync reply"
        assert conv.turn_count == 1
        assert conv.last_result is r
        assert conv.history == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Sync reply"},
        ]
