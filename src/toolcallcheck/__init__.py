"""toolcallcheck: Deterministic testing for tool-using AI agents.

Mock MCP tools, assert exact tool calls and trajectories, verify headers
and model metadata, and reproduce failures locally without depending on
cloud dashboards or live models.
"""

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# P0 Core
# ---------------------------------------------------------------------------
from toolcallcheck.assertions import (
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
from toolcallcheck.builders import ScenarioBuilder, ToolResponseBuilder, UserMessageBuilder
from toolcallcheck.fake_model import FakeModel
from toolcallcheck.mock_server import MockMCPServer, MockTool
from toolcallcheck.multi_turn import Conversation
from toolcallcheck.offline import offline
from toolcallcheck.plugins import register_assertion, run_custom_assertion
from toolcallcheck.recording import Recorder
from toolcallcheck.result import AgentResult, ToolCall
from toolcallcheck.runner import AgentRunner
from toolcallcheck.scenario import scenario
from toolcallcheck.snapshot import assert_snapshot
from toolcallcheck.trajectory import assert_trajectory

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
