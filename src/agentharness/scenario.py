"""Parametrized scenario runner.

Drive multiple test cases from dictionaries or YAML data without
writing separate test functions for each case.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import pytest


def scenario(
    cases: list[dict[str, Any]],
    *,
    id_key: str = "id",
) -> Callable[..., Callable[..., Any]]:
    """Parametrize a test function across multiple scenario cases.

    Each case dict is passed as the ``case`` argument to the test function.
    The ``id_key`` field (default ``"id"``) is used for test ID reporting.

    Usage::

        CASES = [
            {"id": "happy_path", "message": "Create user Jane", "expect_tool": True},
            {"id": "missing_info", "message": "Create user", "expect_tool": False},
        ]

        @scenario(CASES)
        async def test_user_flow(case, agent_runner, mock_mcp_server):
            result = await agent_runner.invoke(case["message"])
            if case["expect_tool"]:
                assert result.tool_call_count > 0
            else:
                assert_no_tool_calls(result)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        ids = [case.get(id_key, f"case_{i}") for i, case in enumerate(cases)]

        @pytest.mark.parametrize("case", cases, ids=ids)
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    return decorator
