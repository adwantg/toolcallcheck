"""agentharness: Deterministic, pytest-native testing for tool-using AI agents.

Mock MCP tools, assert exact tool calls and trajectories, verify headers
and routing, and reproduce failures locally without depending on cloud
dashboards or live models.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# P0 Core
# ---------------------------------------------------------------------------
from agentharness.assertions import (
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

# ---------------------------------------------------------------------------
# P1 Adoption & Coverage
# ---------------------------------------------------------------------------
from agentharness.builders import ScenarioBuilder, ToolResponseBuilder, UserMessageBuilder
from agentharness.fake_model import FakeModel
from agentharness.mock_server import MockMCPServer, MockTool
from agentharness.multi_turn import Conversation
from agentharness.offline import offline
from agentharness.plugins import register_assertion, run_custom_assertion
from agentharness.recording import Recorder
from agentharness.result import AgentResult, ToolCall
from agentharness.runner import AgentRunner
from agentharness.scenario import scenario
from agentharness.snapshot import assert_snapshot
from agentharness.trajectory import assert_trajectory

__all__ = [
    "AgentResult",
    "AgentRunner",
    "Conversation",
    "FakeModel",
    "MockMCPServer",
    "MockTool",
    "Recorder",
    "ScenarioBuilder",
    "ToolCall",
    "ToolResponseBuilder",
    "UserMessageBuilder",
    "__version__",
    "assert_headers",
    "assert_model_used",
    "assert_no_tool_calls",
    "assert_response_contains",
    "assert_response_equals",
    "assert_response_matches",
    "assert_snapshot",
    "assert_tool_args_contain",
    "assert_tool_call_count",
    "assert_tool_call_order",
    "assert_tool_calls",
    "assert_trajectory",
    "offline",
    "register_assertion",
    "run_custom_assertion",
    "scenario",
]
