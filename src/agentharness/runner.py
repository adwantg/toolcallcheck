"""AgentRunner — the central test harness for agentharness.

Provides a single entry point for initializing an agent under test,
injecting mock MCP servers and fake models, and running deterministic
invocations.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from agentharness.fake_model import FakeModel
from agentharness.mock_server import MockMCPServer
from agentharness.result import AgentResult, ToolCall, TraceEntry


class AgentRunner:
    """Test harness that simulates agent execution against mock MCP tools.

    The runner orchestrates the agent loop: it feeds the user message to the
    model, interprets tool-call intents in the model response, dispatches
    them against the :class:`MockMCPServer`, and collects the structured
    :class:`AgentResult`.

    Parameters
    ----------
    mcp_server:
        A :class:`MockMCPServer` with registered mock tools.
    model:
        A :class:`FakeModel` that provides deterministic model responses.
        If ``None``, a default no-op model is used that simply returns
        an empty response.
    config:
        Optional configuration dict or path to a YAML config file.
    model_name:
        Name of the model/strategy to record in results.
    default_headers:
        Default headers to include on every request.
    offline:
        If ``True``, the runner enforces that no real network calls can
        be made during invocation (default ``True``).
    max_turns:
        Maximum number of model turns before the runner stops to prevent
        infinite loops (default ``10``).
    """

    def __init__(
        self,
        mcp_server: MockMCPServer | None = None,
        model: FakeModel | None = None,
        config: dict[str, Any] | str | None = None,
        model_name: str | None = None,
        default_headers: dict[str, str] | None = None,
        offline: bool = True,
        max_turns: int = 10,
    ) -> None:
        self._mcp_server = mcp_server or MockMCPServer()
        self._model = model
        self._config = self._load_config(config) if config else {}
        self._model_name = model_name or self._config.get("model_name")
        self._default_headers = default_headers or {}
        self._offline = offline
        self._max_turns = max_turns

    # ---- configuration ------------------------------------------------------

    @staticmethod
    def _load_config(config: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(config, dict):
            return config
        import yaml

        with open(config) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return data

    @property
    def mcp_server(self) -> MockMCPServer:
        """The underlying mock MCP server."""
        return self._mcp_server

    @property
    def config(self) -> dict[str, Any]:
        """Loaded configuration."""
        return self._config

    # ---- invocation ---------------------------------------------------------

    async def invoke(
        self,
        message: str,
        *,
        access_token: str | None = None,
        site_id: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run the agent with the given user message.

        This is the primary async test API.  The runner:

        1. Builds the initial context (system prompt + available tools).
        2. Feeds the user message to the model.
        3. If the model returns tool-call intents, dispatches them against the
           mock MCP server and feeds results back.
        4. Repeats until the model produces a final text response or the
           max-turn limit is reached.

        Parameters
        ----------
        message:
            The user message to send to the agent.
        access_token:
            Optional JWT / bearer token for auth propagation testing.
        site_id:
            Optional site identifier.
        headers:
            Per-request headers (merged with default_headers).
        metadata:
            Arbitrary metadata to include in the result.

        Returns
        -------
        AgentResult with response text, tool calls, headers, model info, and trace.
        """
        # Merge headers
        merged_headers = {**self._default_headers}
        if access_token:
            merged_headers["X-Access-Token"] = access_token
        if site_id:
            merged_headers["X-Site-Id"] = site_id
        if headers:
            merged_headers.update(headers)

        # Build tool catalog for model context
        tool_catalog = self._mcp_server.list_tools()

        # Initialize trace and tool call collection
        trace: list[TraceEntry] = []
        all_tool_calls: list[ToolCall] = []

        # Add user message to trace
        trace.append(TraceEntry(role="user", content=message))

        # The conversation history for the model
        conversation: list[dict[str, Any]] = [
            {"role": "user", "content": message},
        ]

        final_response = ""
        turns = 0

        while turns < self._max_turns:
            turns += 1

            if self._model is None:
                # No model — return empty response
                final_response = ""
                trace.append(TraceEntry(role="assistant", content=""))
                break

            # Get model response
            model_response = self._model.generate(
                messages=conversation,
                tools=tool_catalog,
                headers=merged_headers,
            )

            # Check if model wants to call tools
            tool_call_intents = model_response.get("tool_calls", [])

            if not tool_call_intents:
                # Model produced a final text response
                final_response = model_response.get("content", "")
                trace.append(TraceEntry(role="assistant", content=final_response))
                break

            # Process tool calls
            tool_results: list[dict[str, Any]] = []
            for intent in tool_call_intents:
                tool_name = intent["name"]
                tool_args = intent.get("args", {})

                # Dispatch against mock MCP server
                mcp_result = self._mcp_server.call_tool(tool_name, tool_args)

                if "error" in mcp_result:
                    tc = ToolCall(
                        name=tool_name,
                        args=tool_args,
                        error=str(mcp_result["error"]),
                    )
                else:
                    tc = ToolCall(
                        name=tool_name,
                        args=tool_args,
                        response=mcp_result.get("result"),
                    )
                all_tool_calls.append(tc)
                trace.append(TraceEntry(role="tool", tool_call=tc))

                tool_results.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": mcp_result.get("result", mcp_result.get("error")),
                    }
                )

            # Add assistant message with tool calls to conversation
            conversation.append(
                {
                    "role": "assistant",
                    "tool_calls": tool_call_intents,
                }
            )
            # Add tool results to conversation
            conversation.extend(tool_results)

        result_metadata = copy.deepcopy(metadata or {})
        result_metadata["turns"] = turns

        return AgentResult(
            response=final_response,
            tool_calls=all_tool_calls,
            model_used=self._model_name,
            headers=merged_headers,
            trace=trace,
            metadata=result_metadata,
        )

    def sync_invoke(
        self,
        message: str,
        **kwargs: Any,
    ) -> AgentResult:
        """Synchronous convenience wrapper around :meth:`invoke`.

        Useful for simpler test setups that don't need async.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g. pytest-asyncio)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.invoke(message, **kwargs))
                return future.result()
        else:
            return asyncio.run(self.invoke(message, **kwargs))
