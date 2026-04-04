"""In-process mock MCP server and mock tool definitions.

No TCP networking required — everything runs in-process for deterministic,
offline agent testing.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass
class MockTool:
    """Definition of a mock MCP tool.

    Parameters
    ----------
    name:
        Tool name as the agent knows it (e.g. ``"create_user"``).
    description:
        Human-readable description shown to the agent.
    parameters:
        JSON-Schema-style parameter definitions.
    response:
        Canned success response.  Can be a static value **or** a callable
        ``(args: dict) -> Any`` for conditional / dynamic responses.
    error:
        If set, the tool will return this error instead of the response.
        Can be a string message or a dict with ``{"code": ..., "message": ...}``.
    error_after:
        If set to *n*, the tool succeeds for the first *n* calls then
        returns the ``error`` on subsequent calls.  Useful for retry tests.
    validate_params:
        When ``True`` (the default), the mock server will validate incoming
        arguments against ``parameters`` before dispatching.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    response: Any = None
    error: str | dict[str, Any] | None = None
    error_after: int | None = None
    validate_params: bool = True

    # internal counter for error_after logic
    _call_count: int = dataclasses.field(default=0, init=False, repr=False)


class MockMCPServer:
    """In-process mock MCP server.

    Register one or more :class:`MockTool` instances.  When the agent
    invokes a tool, the server dispatches to the matching mock and
    returns its canned (or conditional) response.

    Every invocation is recorded so assertions can inspect the full
    history of tool calls.
    """

    def __init__(self) -> None:
        self._tools: dict[str, MockTool] = {}
        self._call_log: list[dict[str, Any]] = []

    # ---- registration -------------------------------------------------------

    def add_tool(self, tool: MockTool) -> None:
        """Register a mock tool with the server."""
        self._tools[tool.name] = tool

    def add_tools(self, tools: list[MockTool]) -> None:
        """Convenience: register multiple tools at once."""
        for tool in tools:
            self.add_tool(tool)

    def remove_tool(self, name: str) -> None:
        """Remove a previously registered mock tool."""
        self._tools.pop(name, None)

    # ---- tool listing -------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool catalog (analogous to MCP ``tools/list``)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": copy.deepcopy(t.parameters),
            }
            for t in self._tools.values()
        ]

    # ---- invocation ---------------------------------------------------------

    def call_tool(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch a tool call and return the result.

        Parameters
        ----------
        name:
            The tool name to invoke.
        args:
            Arguments passed by the agent.

        Returns
        -------
        dict with either ``{"result": ...}`` or ``{"error": ...}``.
        """
        if args is None:
            args = {}

        if name not in self._tools:
            entry = {"name": name, "args": args, "error": f"Unknown tool: {name}"}
            self._call_log.append(entry)
            return {"error": f"Unknown tool: {name}"}

        tool = self._tools[name]

        # Parameter validation (basic: check required keys exist)
        if tool.validate_params and tool.parameters:
            missing = []
            for param_name, param_def in tool.parameters.items():
                required = True
                if isinstance(param_def, dict):
                    required = param_def.get("required", True)
                if required and param_name not in args:
                    missing.append(param_name)
            if missing:
                err = f"Missing required parameters: {', '.join(missing)}"
                entry = {"name": name, "args": args, "error": err}
                self._call_log.append(entry)
                return {"error": err}

        # Increment call counter
        tool._call_count += 1

        # Error injection
        if tool.error is not None:
            should_error = True
            if tool.error_after is not None and tool._call_count <= tool.error_after:
                should_error = False

            if should_error:
                error_payload = (
                    tool.error if isinstance(tool.error, dict) else {"message": tool.error}
                )
                entry = {"name": name, "args": args, "error": error_payload}
                self._call_log.append(entry)
                return {"error": error_payload}

        # Compute response — support callable (conditional) responses
        if callable(tool.response):
            response_fn: Callable[..., Any] = tool.response
            result = response_fn(args)
        else:
            result = copy.deepcopy(tool.response)

        entry = {"name": name, "args": args, "result": result}
        self._call_log.append(entry)
        return {"result": result}

    # ---- inspection ---------------------------------------------------------

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Full log of all tool invocations (reads only)."""
        return list(self._call_log)

    def get_tool(self, name: str) -> MockTool | None:
        """Look up a registered tool by name."""
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

    def reset(self) -> None:
        """Clear the call log and reset all tool call counters."""
        self._call_log.clear()
        for tool in self._tools.values():
            tool._call_count = 0

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
