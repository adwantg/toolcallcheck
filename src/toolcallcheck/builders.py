"""Test data builders and factories.

Fluent APIs for constructing test messages, tool responses, and complete
test scenarios without repetitive boilerplate.
"""

from __future__ import annotations

from typing import Any

from toolcallcheck.fake_model import FakeModel
from toolcallcheck.mock_server import MockMCPServer, MockTool
from toolcallcheck.runner import AgentRunner


class UserMessageBuilder:
    """Fluent builder for constructing agent invocation parameters.

    Usage::

        msg = (
            UserMessageBuilder("Create user Jane")
            .with_token("test-jwt")
            .with_site("1234")
            .with_header("X-Custom", "value")
            .build()
        )
        result = await runner.invoke(**msg)
    """

    def __init__(self, message: str) -> None:
        self._message = message
        self._token: str | None = None
        self._site_id: str | None = None
        self._headers: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}

    def with_token(self, token: str) -> UserMessageBuilder:
        """Set the access token."""
        self._token = token
        return self

    def with_site(self, site_id: str) -> UserMessageBuilder:
        """Set the site ID."""
        self._site_id = site_id
        return self

    def with_header(self, key: str, value: str) -> UserMessageBuilder:
        """Add a request header."""
        self._headers[key] = value
        return self

    def with_metadata(self, key: str, value: Any) -> UserMessageBuilder:
        """Add metadata."""
        self._metadata[key] = value
        return self

    def build(self) -> dict[str, Any]:
        """Build the invocation kwargs dict."""
        result: dict[str, Any] = {"message": self._message}
        if self._token:
            result["access_token"] = self._token
        if self._site_id:
            result["site_id"] = self._site_id
        if self._headers:
            result["headers"] = self._headers
        if self._metadata:
            result["metadata"] = self._metadata
        return result


class ToolResponseBuilder:
    """Fluent builder for constructing mock tool responses.

    Usage::

        tool = (
            ToolResponseBuilder("create_user")
            .with_description("Create a new user")
            .with_param("firstName", "string")
            .with_param("email", "string")
            .with_response({"status": "success"})
            .build()
        )
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._description = ""
        self._params: dict[str, Any] = {}
        self._response: Any = None
        self._error: str | dict[str, Any] | None = None
        self._error_after: int | None = None

    def with_description(self, desc: str) -> ToolResponseBuilder:
        """Set the tool description."""
        self._description = desc
        return self

    def with_param(
        self,
        name: str,
        type_: str = "string",
        *,
        required: bool = True,
    ) -> ToolResponseBuilder:
        """Add a parameter definition."""
        self._params[name] = {"type": type_, "required": required}
        return self

    def with_response(self, response: Any) -> ToolResponseBuilder:
        """Set the canned success response."""
        self._response = response
        return self

    def with_conditional_response(self, fn: Any) -> ToolResponseBuilder:
        """Set a conditional response function ``(args) -> response``."""
        self._response = fn
        return self

    def with_error(
        self,
        error: str | dict[str, Any],
        *,
        after: int | None = None,
    ) -> ToolResponseBuilder:
        """Set error injection."""
        self._error = error
        self._error_after = after
        return self

    def build(self) -> MockTool:
        """Build the MockTool."""
        return MockTool(
            name=self._name,
            description=self._description,
            parameters=self._params,
            response=self._response,
            error=self._error,
            error_after=self._error_after,
        )


class ScenarioBuilder:
    """Fluent builder for constructing complete test scenarios.

    Usage::

        scenario = (
            ScenarioBuilder("user_flow")
            .with_tool(
                ToolResponseBuilder("create_user")
                .with_response({"status": "ok"})
                .build()
            )
            .with_model_responses([
                {"tool_calls": [{"name": "create_user", "args": {"email": "a@example.com"}}]},
                {"content": "Done!"},
            ])
            .with_message("Create user a@example.com")
            .build()
        )
        result = await scenario.runner.invoke(**scenario.invocation)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._tools: list[MockTool] = []
        self._model_responses: list[dict[str, Any]] = []
        self._model_rules: list[tuple[str, dict[str, Any]]] = []
        self._message: str = ""
        self._token: str | None = None
        self._site_id: str | None = None
        self._headers: dict[str, str] = {}
        self._model_name: str | None = None

    def with_tool(self, tool: MockTool) -> ScenarioBuilder:
        """Add a mock tool."""
        self._tools.append(tool)
        return self

    def with_model_responses(self, responses: list[dict[str, Any]]) -> ScenarioBuilder:
        """Set scripted model responses."""
        self._model_responses = responses
        return self

    def with_model_rules(self, rules: list[tuple[str, dict[str, Any]]]) -> ScenarioBuilder:
        """Set pattern-matching model rules."""
        self._model_rules = rules
        return self

    def with_message(self, message: str) -> ScenarioBuilder:
        """Set the user message."""
        self._message = message
        return self

    def with_token(self, token: str) -> ScenarioBuilder:
        """Set the access token."""
        self._token = token
        return self

    def with_site_id(self, site_id: str) -> ScenarioBuilder:
        """Set the site ID."""
        self._site_id = site_id
        return self

    def with_header(self, key: str, value: str) -> ScenarioBuilder:
        """Add a request header."""
        self._headers[key] = value
        return self

    def with_model_name(self, name: str) -> ScenarioBuilder:
        """Set the model name for assertions."""
        self._model_name = name
        return self

    def build(self) -> BuiltScenario:
        """Build the scenario."""
        server = MockMCPServer()
        server.add_tools(self._tools)

        model = FakeModel(
            responses=self._model_responses,
            rules=self._model_rules,
        )

        runner = AgentRunner(
            mcp_server=server,
            model=model,
            model_name=self._model_name,
        )

        invocation: dict[str, Any] = {"message": self._message}
        if self._token:
            invocation["access_token"] = self._token
        if self._site_id:
            invocation["site_id"] = self._site_id
        if self._headers:
            invocation["headers"] = self._headers

        return BuiltScenario(
            name=self._name,
            runner=runner,
            server=server,
            model=model,
            invocation=invocation,
        )


class BuiltScenario:
    """Result of a :class:`ScenarioBuilder.build()` call."""

    def __init__(
        self,
        name: str,
        runner: AgentRunner,
        server: MockMCPServer,
        model: FakeModel,
        invocation: dict[str, Any],
    ) -> None:
        self.name = name
        self.runner = runner
        self.server = server
        self.model = model
        self.invocation = invocation
