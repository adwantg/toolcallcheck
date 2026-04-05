"""Offline mode / network safety guardrails.

Prevents accidental real network calls during agent testing by
monkey-patching ``socket.socket``.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Generator
from typing import Any


class NetworkBlockedError(RuntimeError):
    """Raised when a test tries to make a real network call in offline mode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            "Network call attempted during offline agent test. "
            "Use toolcallcheck.offline() to enforce network isolation, "
            "or add the target to the allow-list."
        )


@contextlib.contextmanager
def offline(
    *,
    allow: list[str] | None = None,
) -> Generator[None, None, None]:
    """Context manager that blocks all real network calls.

    Usage::

        with offline():
            result = await runner.invoke("do something")

    Parameters
    ----------
    allow:
        Optional list of hostnames or IP addresses to allow through
        the block (e.g. ``["localhost", "127.0.0.1"]``).
    """
    allow_set = set(allow or [])
    original_connect = socket.socket.connect

    def _blocked_connect(self: socket.socket, address: Any) -> None:  # type: ignore[override]
        # address is typically (host, port) for TCP
        if isinstance(address, tuple) and len(address) >= 1:
            host = str(address[0])
            if host in allow_set:
                return original_connect(self, address)

        raise NetworkBlockedError()

    socket.socket.connect = _blocked_connect  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[assignment]
