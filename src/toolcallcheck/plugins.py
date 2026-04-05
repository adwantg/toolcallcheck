"""Custom assertion plugin system.

Allows users to register domain-specific assertion functions that
integrate with the toolcallcheck assertion workflow.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from toolcallcheck.result import AgentResult

# Global plugin registry
_assertion_registry: dict[str, Callable[..., None]] = {}


def register_assertion(
    name: str,
    fn: Callable[..., None],
) -> None:
    """Register a custom assertion function.

    The function should accept an :class:`AgentResult` as the first
    argument, plus any additional keyword arguments, and raise
    ``AssertionError`` on failure.

    Usage::

        def assert_pii_redacted(result: AgentResult, fields: list[str]) -> None:
            for field in fields:
                if field in result.response:
                    raise AssertionError(f"PII field '{field}' found in response")

        register_assertion("pii_redacted", assert_pii_redacted)

    Parameters
    ----------
    name:
        Unique name for the assertion.
    fn:
        The assertion function.
    """
    _assertion_registry[name] = fn


def run_custom_assertion(
    name: str,
    result: AgentResult,
    **kwargs: Any,
) -> None:
    """Run a registered custom assertion.

    Parameters
    ----------
    name:
        Name of the registered assertion.
    result:
        The :class:`AgentResult` to check.
    **kwargs:
        Additional arguments passed to the assertion function.

    Raises
    ------
    KeyError:
        If no assertion with the given name is registered.
    AssertionError:
        If the custom assertion fails.
    """
    if name not in _assertion_registry:
        raise KeyError(
            f"No custom assertion registered with name '{name}'. "
            f"Available: {list(_assertion_registry.keys())}"
        )

    _assertion_registry[name](result, **kwargs)


def list_assertions() -> list[str]:
    """List all registered custom assertion names."""
    return list(_assertion_registry.keys())


def clear_assertions() -> None:
    """Clear all registered custom assertions (useful in tests)."""
    _assertion_registry.clear()
