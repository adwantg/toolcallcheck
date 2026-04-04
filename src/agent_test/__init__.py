"""agent-test: Deterministic, pytest-native testing for tool-using AI agents.

Mock MCP tools, assert exact tool calls and trajectories, verify headers
and routing, and reproduce failures locally without depending on cloud
dashboards or live models.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# P0 Core
# ---------------------------------------------------------------------------
from agent_test.assertions import (
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
from agent_test.builders import ScenarioBuilder, ToolResponseBuilder, UserMessageBuilder
from agent_test.fake_model import FakeModel
from agent_test.mock_server import MockMCPServer, MockTool
from agent_test.multi_turn import Conversation
from agent_test.offline import offline
from agent_test.plugins import register_assertion, run_custom_assertion
from agent_test.recording import Recorder
from agent_test.result import AgentResult, ToolCall
from agent_test.runner import AgentRunner
from agent_test.scenario import scenario
from agent_test.snapshot import assert_snapshot
from agent_test.trajectory import assert_trajectory

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
