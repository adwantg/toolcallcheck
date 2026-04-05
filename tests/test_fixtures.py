"""Tests for pytest fixtures provided by toolcallcheck."""

from __future__ import annotations

from toolcallcheck.mock_server import MockMCPServer, MockTool
from toolcallcheck.runner import AgentRunner


class TestFixtures:
    def test_mock_mcp_server_fixture(self, mock_mcp_server):
        """Verify the mock_mcp_server fixture provides a working server."""
        assert isinstance(mock_mcp_server, MockMCPServer)
        assert len(mock_mcp_server) == 0

        mock_mcp_server.add_tool(MockTool(name="test", response="ok"))
        assert len(mock_mcp_server) == 1

    def test_agent_runner_fixture(self, agent_runner):
        """Verify the agent_runner fixture provides a working runner."""
        assert isinstance(agent_runner, AgentRunner)
        assert agent_runner.mcp_server is not None

    def test_fixtures_are_isolated(self, mock_mcp_server):
        """Each test should get a fresh server instance."""
        # If fixtures leaked state, this would have tools from other tests
        assert len(mock_mcp_server) == 0

    def test_runner_has_working_server(self, agent_runner):
        """The agent_runner fixture has a working mcp_server."""
        agent_runner.mcp_server.add_tool(MockTool(name="from_runner", response="ok"))
        assert "from_runner" in agent_runner.mcp_server
