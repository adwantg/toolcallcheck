# toolcallcheck

> **Deterministic, pytest-native testing for tool-using AI agents.**

[![PyPI](https://img.shields.io/pypi/v/toolcallcheck)](https://pypi.org/project/toolcallcheck/)
[![Python](https://img.shields.io/pypi/pyversions/toolcallcheck)](https://pypi.org/project/toolcallcheck/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/adwantg/toolcallcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/adwantg/toolcallcheck/actions/workflows/ci.yml)

`toolcallcheck` is an open-source Python library for testing tool-using agents. Mock MCP tools, assert exact tool calls and execution trajectories, verify headers and model metadata, and run fully offline in CI without relying on cloud dashboards or live models.

## Why toolcallcheck?

| What you get | How it's different |
|---|---|
| **Exact behavior assertions** | Not score-based judging — assert exact tool names, args, headers |
| **Local/offline execution** | No cloud dashboard dependency — runs in CI and on laptops |
| **MCP & tool contract testing** | Purpose-built for tool-calling agents, not just prompt testing |
| **Python-native ergonomics** | pytest fixtures, markers, and assertion helpers — not config files |
| **Framework-adapter friendly** | Ships a framework adapter protocol plus stubs for OpenAI Agents, LangGraph, PydanticAI, and CrewAI |

## Installation

```bash
# Requires Python 3.10+
pip install toolcallcheck
```

For development:

```bash
pip install toolcallcheck[dev]
```

## Quick Start

```python
import pytest
from toolcallcheck import (
    AgentRunner, MockMCPServer, MockTool, FakeModel,
    assert_tool_calls, assert_response_contains, assert_no_tool_calls,
)

# 1. Define mock tools
server = MockMCPServer()
server.add_tool(MockTool(
    name="create_user",
    description="Create a new user account",
    parameters={
        "name": {"type": "string"},
        "email": {"type": "string"},
        "role": {"type": "string"},
    },
    response={"status": "success", "userId": "USR-123"},
))

# 2. Define deterministic model behavior
model = FakeModel(responses=[
    {"tool_calls": [{"name": "create_user", "args": {
        "name": "Jane Doe", "email": "jane@example.com", "role": "admin",
    }}]},
    {"content": "I've created admin user Jane Doe at jane@example.com."},
])

# 3. Create the test runner
runner = AgentRunner(mcp_server=server, model=model, model_name="Nova-pro")

# 4. Run and assert
@pytest.mark.asyncio
async def test_create_user():
    result = await runner.invoke(
        "Create an admin user for Jane Doe jane@example.com",
        access_token="test-jwt",
        site_id="1234",
    )

    # Assert the correct tool was called with correct arguments
    assert_tool_calls(result, [{
        "name": "create_user",
        "args": {
            "name": "Jane Doe",
            "email": "jane@example.com",
            "role": "admin",
        },
    }])

    # Assert the response is user-friendly
    assert_response_contains(result, "created")
```

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [MockMCPServer & MockTool](#mockmcpserver--mocktool)
- [AgentRunner](#agentrunner)
- [AgentResult](#agentresult)
- [Assertion Helpers](#assertion-helpers)
  - [Tool Call Assertions](#tool-call-assertions)
  - [Response Assertions](#response-assertions)
  - [Header & Auth Assertions](#header--auth-assertions)
  - [Model Selection Assertions](#model-selection-assertions)
- [FakeModel (Deterministic Model Adapter)](#fakemodel-deterministic-model-adapter)
- [Multi-Turn Conversation Testing](#multi-turn-conversation-testing)
- [Snapshot Testing](#snapshot-testing)
- [Golden Trajectory Assertions](#golden-trajectory-assertions)
- [Request/Response Recording](#requestresponse-recording)
- [Test Data Builders](#test-data-builders)
- [Offline Mode](#offline-mode)
- [Parametrized Scenarios](#parametrized-scenarios)
- [Custom Assertion Plugins](#custom-assertion-plugins)
- [Pytest Markers](#pytest-markers)
- [Conditional Tool Responses](#conditional-tool-responses)
- [Partial Argument Matching](#partial-argument-matching)
- [Framework Adapters](#framework-adapters)
- [Structured Diff Output](#structured-diff-output)
- [Dependencies](#dependencies)
- [Architecture](#architecture)
- [Contributing](#contributing)

---

## Core Concepts

`toolcallcheck` provides four building blocks:

1. **`MockMCPServer`** — In-process mock MCP server with registered tools
2. **`FakeModel`** — Deterministic model that returns scripted responses
3. **`AgentRunner`** — Orchestrates the agent loop (model → tool call → mock server → repeat)
4. **`AgentResult`** — Structured result with response, tool calls, headers, trace, and metadata

```
User Message → AgentRunner → FakeModel → Tool Call Intent
                                ↓
                          MockMCPServer → Canned Response
                                ↓
                          FakeModel → Final Text Response
                                ↓
                          AgentResult (for assertions)
```

---

## MockMCPServer & MockTool

The `MockMCPServer` is an in-process mock that requires no TCP networking and no external dependencies.

### Basic Tool Registration

```python
from toolcallcheck import MockMCPServer, MockTool

server = MockMCPServer()

# Register a tool with a canned response
server.add_tool(MockTool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "city": {"type": "string"},
        "unit": {"type": "string", "required": False},
    },
    response={"temp": 72, "unit": "F", "condition": "sunny"},
))

# Register multiple tools at once
server.add_tools([
    MockTool(name="search", response={"results": []}),
    MockTool(name="send_email", response={"sent": True}),
])
```

### Tool Invocation

```python
# Call a tool directly
result = server.call_tool("get_weather", {"city": "Seattle"})
# result == {"result": {"temp": 72, "unit": "F", "condition": "sunny"}}

# Inspect the call log
print(server.call_log)
# [{"name": "get_weather", "args": {"city": "Seattle"}, "result": {...}}]
```

### Error Injection

```python
# Tool that always errors
error_tool = MockTool(
    name="broken_service",
    error="Service temporarily unavailable",
)

# Tool with structured error
structured_error = MockTool(
    name="api_error",
    error={"code": 503, "message": "Service unavailable"},
)

# Tool that fails after N successful calls (for retry testing)
flaky_tool = MockTool(
    name="flaky_api",
    response={"data": "ok"},
    error="Connection timeout",
    error_after=2,  # Succeeds twice, then fails
)
```

### Parameter Validation

```python
# Parameters are validated by default
strict_tool = MockTool(
    name="create_user",
    parameters={
        "name": {"type": "string", "required": True},
        "email": {"type": "string", "required": True},
        "role": {"type": "string", "required": False},
    },
    response={"id": "usr-123"},
)
server.add_tool(strict_tool)

# Missing required parameter → error
result = server.call_tool("create_user", {"name": "Alice"})
# result == {"error": "Missing required parameters: email"}
```

### Server Inspection

```python
server.list_tools()       # Tool catalog (like MCP tools/list)
server.tool_names          # ["get_weather", "search", ...]
server.get_tool("search")  # Get a specific MockTool
len(server)                # Number of registered tools
"search" in server         # Check if tool is registered
server.call_log            # Full invocation history
server.reset()             # Clear call log and counters
```

---

## AgentRunner

The `AgentRunner` orchestrates the full agent loop: model generation → tool dispatch → response collection.

### Basic Usage

```python
from toolcallcheck import AgentRunner, MockMCPServer, MockTool, FakeModel

server = MockMCPServer()
server.add_tool(MockTool(name="lookup", response={"found": True}))

model = FakeModel(responses=[{"content": "I found it!"}])
runner = AgentRunner(mcp_server=server, model=model)

# Async invocation
result = await runner.invoke("Find the record")

# Sync invocation (for simpler tests)
result = runner.sync_invoke("Find the record")
```

### Configuration

```python
# From a dict
runner = AgentRunner(
    mcp_server=server,
    model=model,
    config={"model_name": "Nova-pro", "timeout": 30},
    model_name="Nova-pro",
    default_headers={"X-API-Version": "2024-01"},
    max_turns=5,
)

# From a YAML file
runner = AgentRunner(
    mcp_server=server,
    model=model,
    config="agents/loan-app/config.yaml",
)
```

### Auth & Header Propagation

```python
result = await runner.invoke(
    "Process the loan application",
    access_token="eyJhbGciOiJSUzI1NiIs...",
    site_id="site-1234",
    headers={"X-KVFleet-Strategy": "cheap_cascade"},
    metadata={"test_case": "happy_path"},
)

# Headers are captured in the result for assertion
print(result.headers)
# {"X-Access-Token": "eyJ...", "X-Site-Id": "site-1234", ...}
```

---

## AgentResult

Every `invoke()` returns a structured `AgentResult`:

```python
result = await runner.invoke("Hello")

result.response          # Final agent text response
result.tool_calls        # List of ToolCall objects
result.tool_call_count   # Number of tool calls
result.tool_names        # ["tool_a", "tool_b", ...]
result.model_used        # "Nova-pro" or None
result.headers           # Captured request headers
result.trace             # Full conversation trace (TraceEntry list)
result.metadata          # {"turns": 2, "test_case": "happy_path", ...}

# Access individual tool calls
tc = result.get_tool_call("create_user")
tc.name      # "create_user"
tc.args      # {"name": "Jane Doe", ...}
tc.response  # {"status": "success", ...}
tc.error     # None or error string

# Serialize
result.to_dict()  # Full dict representation
```

---

## Assertion Helpers

### Tool Call Assertions

```python
from toolcallcheck import (
    assert_tool_calls,
    assert_tool_call_count,
    assert_no_tool_calls,
    assert_tool_call_order,
    assert_tool_args_contain,
)

# Assert exact tool calls (name + args)
assert_tool_calls(result, [
    {"name": "create_user", "args": {"name": "Jane", "email": "jane@example.com"}},
])

# Assert tool calls in any order
assert_tool_calls(result, [
    {"name": "tool_b", "args": {}},
    {"name": "tool_a", "args": {}},
], strict_order=False)

# Assert number of tool calls
assert_tool_call_count(result, 2)

# Assert NO tools were called (clarification flow)
assert_no_tool_calls(result)

# Assert tool call ordering
assert_tool_call_order(result, ["lookup", "validate", "submit"])

# Assert partial arguments (ignore auto-generated fields like timestamps)
assert_tool_args_contain(result, "create_record", {
    "firstName": "John",
    "email": "john@test.com",
})

# Check a specific call index when a tool is called multiple times
assert_tool_args_contain(result, "retry_api", {"attempt": 2}, call_index=1)
```

### Response Assertions

```python
from toolcallcheck import (
    assert_response_contains,
    assert_response_matches,
    assert_response_equals,
)

# Substring check
assert_response_contains(result, "invitation sent")

# Regex match
assert_response_matches(result, r"INV-\d{3}")

# Exact match
assert_response_equals(result, "The invitation has been sent successfully.")
```

### Header & Auth Assertions

```python
from toolcallcheck import assert_headers

# Assert specific headers are present with correct values
assert_headers(result, {
    "X-Access-Token": "test-jwt",
    "X-Site-Id": "1234",
})

# Assert exact headers (no extras allowed)
assert_headers(result, {
    "X-Access-Token": "test-jwt",
}, exact=True)
```

### Model Selection Assertions

```python
from toolcallcheck import assert_model_used

# Assert the model or routing strategy used
assert_model_used(result, "Nova-pro")
```

---

## FakeModel (Deterministic Model Adapter)

The `FakeModel` replaces the real LLM with deterministic behavior. It supports three modes:

### Scripted Sequence

```python
from toolcallcheck import FakeModel

# Responses are returned in order, one per model call
model = FakeModel(responses=[
    # First call: model wants to call a tool
    {"tool_calls": [{"name": "search", "args": {"query": "test"}}]},
    # Second call: model returns text (after receiving tool result)
    {"content": "I found 3 results for 'test'."},
])
```

### Pattern Matching

```python
# Match the user's last message against regex rules
model = FakeModel(rules=[
    (r"weather", {"content": "It's sunny and 72°F."}),
    (r"time", {"content": "It's 3:00 PM."}),
    (r"create (\w+)", {"tool_calls": [{"name": "create_user", "args": {"name": "user"}}]}),
])
```

### Default Response

```python
# Fallback when no scripted response remains and no rule matches
model = FakeModel(
    responses=[{"content": "First response"}],
    default_response={"content": "I don't understand, could you rephrase?"},
)
```

### Inspection

```python
model.call_count   # Number of generate() calls
model.call_log     # Full log of all calls with messages, tools, headers
model.reset()      # Reset to initial state
```

---

## Multi-Turn Conversation Testing

Test stateful, multi-step agent workflows:

```python
from toolcallcheck import AgentRunner, FakeModel, Conversation, MockMCPServer, MockTool
from toolcallcheck import assert_no_tool_calls, assert_tool_calls, assert_response_contains

server = MockMCPServer()
server.add_tool(MockTool(name="create_user", response={"status": "created"}))

model = FakeModel(responses=[
    {"content": "Sure! What's their email address?"},
    {"tool_calls": [{"name": "create_user", "args": {"email": "jane@example.com"}}]},
    {"content": "User created!"},
])

runner = AgentRunner(mcp_server=server, model=model)
conv = Conversation(runner)

@pytest.mark.asyncio
async def test_multi_turn_create_user():
    # Turn 1: Agent asks for more info
    r1 = await conv.say("Create a user for Jane")
    assert_no_tool_calls(r1)
    assert_response_contains(r1, "email")

    # Turn 2: User provides email, agent calls tool
    r2 = await conv.say("jane@example.com")
    assert_response_contains(r2, "created")

    # Inspect conversation state
    assert conv.turn_count == 2
    assert len(conv.all_tool_calls) == 1
```

**Properties:**

```python
conv.turn_count      # Number of completed turns
conv.results         # All AgentResult objects in order
conv.last_result     # Most recent result
conv.history         # Full message history as dicts
conv.all_tool_calls  # All tool calls across all turns
conv.reset()         # Clear conversation state
```

---

## Snapshot Testing

Compare agent output against saved golden files:

```python
from toolcallcheck import assert_snapshot

@pytest.mark.asyncio
async def test_response_stability():
    result = await runner.invoke("Get account summary")

    # First run: creates .toolcallcheck_snapshots/account_summary.json
    # Subsequent runs: compares against saved snapshot
    assert_snapshot(result, "account_summary")

    # Options
    assert_snapshot(result, "summary_no_tools", include_tool_calls=False)
    assert_snapshot(result, "full_snapshot", include_model=True)

    # Force update
    assert_snapshot(result, "updated", update=True)
```

Update all snapshots at once:

```bash
TOOLCALLCHECK_UPDATE_SNAPSHOTS=1 pytest tests/
```

---

## Golden Trajectory Assertions

Validate the full sequence of messages and tool calls:

```python
from toolcallcheck import assert_trajectory

@pytest.mark.asyncio
async def test_complete_workflow():
    result = await runner.invoke("Process loan for John Smith")

    # Exact trajectory match
    assert_trajectory(result, [
        {"role": "user", "content": "Process loan for John Smith"},
        {"role": "tool", "name": "lookup_applicant", "args": {"name": "John Smith"}},
        {"role": "tool", "name": "create_loan", "args": {"applicant_id": "A-123"}},
        {"role": "assistant", "content": "Loan LN-456 created for John Smith."},
    ])

    # Subset mode: only required steps must appear
    assert_trajectory(result, [
        {"role": "tool", "name": "create_loan", "args": {"applicant_id": "A-123"}},
    ], mode="subset")

    # Ordered subset: steps must appear in order, gaps allowed
    assert_trajectory(result, [
        {"role": "user", "content": "Process loan for John Smith"},
        {"role": "assistant", "content": "Loan LN-456 created for John Smith."},
    ], mode="ordered_subset")
```

---

## Request/Response Recording

Capture agent executions for debugging:

```python
from toolcallcheck import Recorder

recorder = Recorder()

result = await runner.invoke("Test message")
recorder.record(result, label="test_workflow_flow")

# Inspect in memory
assert recorder.count == 1
entry = recorder.get(0)
print(entry["response"])
print(entry["tool_calls"])
print(entry["trace"])

# Export to files
recorder.export("test_recordings/")        # JSON (default)
recorder.export("test_recordings/", format="yaml")  # YAML

recorder.clear()  # Clear all recordings
```

---

## Test Data Builders

Fluent APIs to reduce test boilerplate:

### UserMessageBuilder

```python
from toolcallcheck import UserMessageBuilder

msg = (
    UserMessageBuilder("Create admin user Jane Doe jane@example.com")
    .with_token("test-jwt-token")
    .with_site("site-1234")
    .with_header("X-KVFleet-Strategy", "cheap_cascade")
    .with_metadata("test_case", "happy_path")
    .build()
)

result = await runner.invoke(**msg)
```

### ToolResponseBuilder

```python
from toolcallcheck import ToolResponseBuilder

tool = (
    ToolResponseBuilder("create_user")
    .with_description("Create a new user account")
    .with_param("name", "string")
    .with_param("role", "string")
    .with_param("email", "string")
    .with_response({"status": "success", "userId": "USR-123"})
    .build()
)
server.add_tool(tool)

# With error injection
error_tool = (
    ToolResponseBuilder("unreliable_api")
    .with_response({"ok": True})
    .with_error("Timeout", after=3)  # Fails after 3 successes
    .build()
)
```

### ScenarioBuilder

```python
from toolcallcheck import ScenarioBuilder, MockTool

scenario = (
    ScenarioBuilder("happy_path_create")
    .with_tool(MockTool(name="create_user", response={"status": "created"}))
    .with_model_responses([
        {"tool_calls": [{"name": "create_user", "args": {"email": "a@example.com"}}]},
        {"content": "User created!"},
    ])
    .with_message("Create user a@example.com")
    .with_token("test-jwt")
    .with_model_name("Nova-pro")
    .build()
)

result = await scenario.runner.invoke(**scenario.invocation)
assert result.response == "User created!"
```

---

## Offline Mode

Block all real network calls during testing:

```python
from toolcallcheck import offline

@pytest.mark.asyncio
async def test_offline_safety():
    with offline():
        # Any real network call here raises NetworkBlockedError
        result = await runner.invoke("Do something")
        # Only mock MCP calls succeed — no accidental live calls

    # Allow specific hosts
    with offline(allow=["localhost", "127.0.0.1"]):
        result = await runner.invoke("Use local service")
```

---

## Parametrized Scenarios

Drive multiple test cases from data:

```python
from toolcallcheck import scenario

CASES = [
    {
        "id": "happy_path",
        "message": "Create user Jane Doe jane@example.com",
        "expect_tool": True,
        "expected_tool_name": "create_user",
    },
    {
        "id": "missing_email",
        "message": "Create user Jane Doe",
        "expect_tool": False,
    },
    {
        "id": "missing_name",
        "message": "Create user",
        "expect_tool": False,
    },
]

@scenario(CASES)
def test_create_user_scenarios(case):
    result = runner.sync_invoke(case["message"])
    if case["expect_tool"]:
        assert result.tool_call_count > 0
    else:
        assert result.tool_call_count == 0
```

---

## Custom Assertion Plugins

Register domain-specific assertions:

```python
from toolcallcheck import register_assertion, run_custom_assertion, AgentResult

# Register a PII detection assertion
def assert_no_pii(result: AgentResult, fields: list[str] | None = None):
    for field in (fields or []):
        if field.lower() in result.response.lower():
            raise AssertionError(f"PII field '{field}' found in response")

register_assertion("no_pii", assert_no_pii)

# Use it in tests
@pytest.mark.asyncio
async def test_no_pii_leakage():
    result = await runner.invoke("Show my account details")
    run_custom_assertion("no_pii", result, fields=["SSN", "password", "credit card"])
```

---

## Pytest Markers

Categorize tests for selective CI execution:

```python
from toolcallcheck.markers import agent_behavior, agent_error, agent_multi_turn, agent_offline

@agent_behavior
async def test_happy_path():
    ...

@agent_error
async def test_tool_failure():
    ...

@agent_multi_turn
async def test_conversation():
    ...

@agent_offline
async def test_network_isolation():
    ...
```

Run specific categories:

```bash
pytest -m agent_behavior       # Only behavior tests
pytest -m agent_error           # Only error tests
pytest -m "not agent_offline"   # Skip offline tests
```

---

## Conditional Tool Responses

Tools can return different responses based on input arguments:

```python
from toolcallcheck import MockTool

def weather_response(args):
    city = args.get("city", "")
    if city == "Seattle":
        return {"temp": 55, "condition": "rainy"}
    elif city == "Phoenix":
        return {"temp": 105, "condition": "sunny"}
    return {"temp": 70, "condition": "unknown"}

tool = MockTool(
    name="get_weather",
    response=weather_response,  # Callable instead of static dict
)
```

---

## Partial Argument Matching

Assert only the arguments you care about:

```python
from toolcallcheck import assert_tool_args_contain

# Agent may add auto-generated fields (timestamps, request IDs)
# Only check the fields you care about
assert_tool_args_contain(result, "create_record", {
    "firstName": "John",
    "email": "john@test.com",
})
# Ignores extra args like {"timestamp": "2024-01-01T...", "requestId": "req-abc"}
```

---

## Framework Adapters

`toolcallcheck` provides a `FrameworkAdapter` protocol plus starter adapter stubs for integrating with specific agent frameworks:

```python
from toolcallcheck.adapters import FrameworkAdapter

# Available adapter stubs (implement .invoke() for your framework):
from toolcallcheck.adapters import (
    OpenAIAgentsAdapter,
    LangGraphAdapter,
    PydanticAIAdapter,
    CrewAIAdapter,
)

# Custom adapter
class MyFrameworkAdapter:
    @property
    def framework_name(self) -> str:
        return "my-framework"

    async def invoke(self, message, *, tools=None, headers=None, metadata=None):
        # Wire up your framework here
        ...
```

---

## Structured Diff Output

All assertion failures produce structured, CI-friendly diff output:

```
Tool calls do not match (strict order):
  [0] ✗ MISMATCH:
        expected: create_user(name='John', email='john@example.com')
        actual:   create_user(name='Jane', email='jane@example.com')
        arg diff: ~name: 'John' → 'Jane', ~email: 'john@example.com' → 'jane@example.com'
  [1] - MISSING:    validateAddress(zip='98101')
```

```
Response does not contain expected substring.
  Expected substring: 'invitation sent'
  Actual response:    'I need more information before proceeding.'
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pyyaml` | Load agent config from YAML files |
| `jsonschema` | JSON Schema validation for tool parameters |
| `deepdiff` | Deep comparison for snapshot and trajectory assertions |

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `pytest` | Test framework |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel test execution |
| `ruff` | Linting and formatting |
| `mypy` | Type checking |
| `hypothesis` | Property-based testing |

---

## Architecture

```
toolcallcheck/
├── src/
│   └── toolcallcheck/
│       ├── __init__.py          # Public API exports
│       ├── mock_server.py       # MockMCPServer, MockTool
│       ├── runner.py            # AgentRunner (orchestration)
│       ├── result.py            # AgentResult, ToolCall, TraceEntry
│       ├── assertions.py        # All assertion helpers
│       ├── diff.py              # Structured diff formatting
│       ├── fixtures.py          # Pytest plugin & fixtures
│       ├── offline.py           # Network safety guardrails
│       ├── fake_model.py        # FakeModel (deterministic adapter)
│       ├── multi_turn.py        # Conversation (multi-turn testing)
│       ├── snapshot.py          # Snapshot testing
│       ├── trajectory.py        # Trajectory assertions
│       ├── recording.py         # Request/response recording
│       ├── builders.py          # Test data builders
│       ├── scenario.py          # Parametrized scenario runner
│       ├── plugins.py           # Custom assertion plugin system
│       ├── markers.py           # Pytest marker re-exports
│       └── adapters/
│           └── __init__.py      # Framework adapter protocol & stubs
├── tests/                       # 176 tests covering all features
├── document/                    # Product docs, roadmap, competitive analysis
├── pyproject.toml               # PEP 621 packaging
├── release.sh                   # Automated release script
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── CITATION.cff
```

---

## Comparison with Alternatives

| Capability | toolcallcheck | Promptfoo | DeepEval | LangSmith | Inspect AI |
|---|:---:|:---:|:---:|:---:|:---:|
| Python-first pytest DX | ✅ | ⚠️ | ✅ | ⚠️ | ⚠️ |
| Deterministic tool-call assertions | ✅ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| In-process MCP mock server | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| Offline local execution | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| Conditional mock responses | ✅ | ❌ | ❌ | ❌ | ❌ |
| Header/auth assertions | ✅ | ❌ | ❌ | ⚠️ | ⚠️ |
| Partial argument matching | ✅ | ❌ | ❌ | ❌ | ❌ |
| Custom assertion plugins | ✅ | ❌ | ❌ | ❌ | ❌ |
| Framework-agnostic | ✅ | ✅ | ✅ | ⚠️ | ✅ |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, quality gates, and PR requirements.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting policy.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite it. See [CITATION.cff](CITATION.cff).
