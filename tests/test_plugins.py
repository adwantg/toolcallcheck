"""Tests for custom assertion plugin system."""

from __future__ import annotations

import pytest

from toolcallcheck.plugins import (
    clear_assertions,
    list_assertions,
    register_assertion,
    run_custom_assertion,
)
from toolcallcheck.result import AgentResult


class TestPlugins:
    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Ensure each test starts with a clean registry."""
        clear_assertions()
        yield
        clear_assertions()

    def test_register_and_run(self):
        def check_length(result: AgentResult, min_length: int = 0):
            if len(result.response) < min_length:
                raise AssertionError(f"Response too short: {len(result.response)} < {min_length}")

        register_assertion("min_length", check_length)
        result = AgentResult(response="Hello world")
        run_custom_assertion("min_length", result, min_length=5)

    def test_custom_assertion_fails(self):
        def always_fail(result: AgentResult):
            raise AssertionError("Always fails")

        register_assertion("always_fail", always_fail)
        with pytest.raises(AssertionError, match="Always fails"):
            run_custom_assertion("always_fail", AgentResult())

    def test_unregistered_raises_key_error(self):
        with pytest.raises(KeyError, match="No custom assertion"):
            run_custom_assertion("nonexistent", AgentResult())

    def test_list_assertions(self):
        register_assertion("a", lambda r: None)
        register_assertion("b", lambda r: None)
        names = list_assertions()
        assert "a" in names
        assert "b" in names

    def test_clear(self):
        register_assertion("temp", lambda r: None)
        assert len(list_assertions()) == 1
        clear_assertions()
        assert len(list_assertions()) == 0

    def test_pii_redaction_example(self):
        """Example: domain-specific PII assertion."""

        def assert_no_pii(result: AgentResult, fields: list[str] | None = None):
            if fields is None:
                fields = []
            for field in fields:
                if field.lower() in result.response.lower():
                    raise AssertionError(f"PII field '{field}' found in response")

        register_assertion("no_pii", assert_no_pii)

        safe_result = AgentResult(response="The operation completed successfully.")
        run_custom_assertion("no_pii", safe_result, fields=["SSN", "password"])

        unsafe_result = AgentResult(response="Your SSN is 123-45-6789")
        with pytest.raises(AssertionError, match="PII field"):
            run_custom_assertion("no_pii", unsafe_result, fields=["SSN"])
