"""Tests for FakeModel."""

from __future__ import annotations

from agentharness.fake_model import FakeModel


class TestFakeModelScripted:
    def test_scripted_sequence(self):
        model = FakeModel(
            responses=[
                {"content": "First"},
                {"content": "Second"},
                {"content": "Third"},
            ]
        )
        r1 = model.generate(messages=[{"role": "user", "content": "a"}])
        assert r1["content"] == "First"

        r2 = model.generate(messages=[{"role": "user", "content": "b"}])
        assert r2["content"] == "Second"

        r3 = model.generate(messages=[{"role": "user", "content": "c"}])
        assert r3["content"] == "Third"

    def test_scripted_tool_calls(self):
        model = FakeModel(
            responses=[
                {"tool_calls": [{"name": "search", "args": {"query": "test"}}]},
                {"content": "Done"},
            ]
        )
        r1 = model.generate(messages=[{"role": "user", "content": "search"}])
        assert "tool_calls" in r1
        assert r1["tool_calls"][0]["name"] == "search"

    def test_exhausted_sequence_falls_to_default(self):
        model = FakeModel(
            responses=[{"content": "One"}],
            default_response={"content": "Default"},
        )
        model.generate(messages=[{"role": "user", "content": "a"}])
        r2 = model.generate(messages=[{"role": "user", "content": "b"}])
        assert r2["content"] == "Default"


class TestFakeModelRules:
    def test_pattern_matching(self):
        model = FakeModel(
            rules=[
                (r"weather", {"content": "It's sunny"}),
                (r"time", {"content": "It's noon"}),
            ]
        )
        r1 = model.generate(messages=[{"role": "user", "content": "What's the weather?"}])
        assert r1["content"] == "It's sunny"

        r2 = model.generate(messages=[{"role": "user", "content": "What time is it?"}])
        assert r2["content"] == "It's noon"

    def test_case_insensitive_matching(self):
        model = FakeModel(
            rules=[
                (r"hello", {"content": "Hi"}),
            ]
        )
        r = model.generate(messages=[{"role": "user", "content": "HELLO World"}])
        assert r["content"] == "Hi"

    def test_no_match_returns_default(self):
        model = FakeModel(
            rules=[(r"specific", {"content": "matched"})],
            default_response={"content": "fallback"},
        )
        r = model.generate(messages=[{"role": "user", "content": "unrelated"}])
        assert r["content"] == "fallback"

    def test_first_match_wins(self):
        model = FakeModel(
            rules=[
                (r"test", {"content": "First"}),
                (r"test", {"content": "Second"}),
            ]
        )
        r = model.generate(messages=[{"role": "user", "content": "test"}])
        assert r["content"] == "First"


class TestFakeModelCallLog:
    def test_call_log(self):
        model = FakeModel(default_response={"content": "ok"})
        model.generate(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t"}],
            headers={"X-Key": "val"},
        )
        assert model.call_count == 1
        assert model.call_log[0]["headers"] == {"X-Key": "val"}

    def test_reset(self):
        model = FakeModel(responses=[{"content": "a"}, {"content": "b"}])
        model.generate(messages=[{"role": "user", "content": "x"}])
        assert model.call_count == 1

        model.reset()
        assert model.call_count == 0
        r = model.generate(messages=[{"role": "user", "content": "y"}])
        assert r["content"] == "a"  # Reset to start of sequence


class TestFakeModelDefault:
    def test_default_empty_response(self):
        model = FakeModel()
        r = model.generate(messages=[{"role": "user", "content": "hi"}])
        assert r["content"] == ""
