"""Pytest markers for agent-test.

Provides category-based markers for selective test execution in CI.
"""

from __future__ import annotations

import pytest

# Re-export markers for convenience
agent_behavior = pytest.mark.agent_behavior
agent_error = pytest.mark.agent_error
agent_multi_turn = pytest.mark.agent_multi_turn
agent_offline = pytest.mark.agent_offline
