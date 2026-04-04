"""Pytest fixtures and plugin registration for agentharness.

This module is automatically discovered by pytest through the
``pytest11`` entry point defined in ``pyproject.toml``.  It provides
ready-to-use fixtures for common testing patterns.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from agentharness.mock_server import MockMCPServer
from agentharness.runner import AgentRunner


def pytest_configure(config: Any) -> None:
    """Register custom markers with pytest."""
    config.addinivalue_line(
        "markers",
        "agent_behavior: tests for standard agent behavior flows",
    )
    config.addinivalue_line(
        "markers",
        "agent_error: tests for agent error handling paths",
    )
    config.addinivalue_line(
        "markers",
        "agent_multi_turn: tests for multi-turn conversation flows",
    )
    config.addinivalue_line(
        "markers",
        "agent_offline: tests that verify offline/network-safe execution",
    )


@pytest.fixture
def mock_mcp_server() -> MockMCPServer:
    """Provide a fresh, isolated :class:`MockMCPServer` for each test."""
    return MockMCPServer()


@pytest.fixture
def agent_runner(mock_mcp_server: MockMCPServer) -> AgentRunner:
    """Provide a fresh :class:`AgentRunner` wired to the ``mock_mcp_server`` fixture."""
    return AgentRunner(mcp_server=mock_mcp_server)


@pytest.fixture
def _agentharness_isolation() -> Generator[None, None, None]:
    """Internal fixture ensuring no shared mutable state between tests.

    This is an autouse fixture when running with ``pytest-xdist`` —
    each test gets its own mock instances.
    """
    yield
