"""Tests for offline mode / network safety."""

from __future__ import annotations

import socket

import pytest

from agent_test.offline import NetworkBlockedError, offline


class TestOfflineMode:
    def test_blocks_connections(self):
        with offline():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(NetworkBlockedError):
                s.connect(("example.com", 80))
            s.close()

    def test_restores_after_context(self):
        original = socket.socket.connect
        with offline():
            pass
        assert socket.socket.connect is original

    def test_allow_list(self):
        with offline(allow=["127.0.0.1", "localhost"]):
            # Should not raise for allowed hosts
            # We don't actually connect, just verify the guard doesn't block
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Trying to connect to a non-listening port will raise ConnectionRefusedError
            # which is different from NetworkBlockedError — that means the allow-list worked
            with pytest.raises((ConnectionRefusedError, OSError)):
                s.connect(("127.0.0.1", 1))
            s.close()

    def test_blocks_non_allowed_host(self):
        with offline(allow=["localhost"]):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(NetworkBlockedError):
                s.connect(("example.com", 80))
            s.close()

    def test_nesting(self):
        with offline(), offline():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(NetworkBlockedError):
                s.connect(("example.com", 80))
            s.close()
