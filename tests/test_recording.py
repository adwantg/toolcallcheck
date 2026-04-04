"""Tests for recording and export."""

from __future__ import annotations

import json
import tempfile

import pytest

from agentharness.recording import Recorder
from agentharness.result import AgentResult, ToolCall, TraceEntry


class TestRecorder:
    def _sample_result(self, response="Hello"):
        return AgentResult(
            response=response,
            tool_calls=[ToolCall(name="t", args={"x": 1}, response="ok")],
            model_used="gpt-4",
            headers={"X-Key": "val"},
            metadata={"test": True},
            trace=[
                TraceEntry(role="user", content="Hi"),
                TraceEntry(role="assistant", content=response),
            ],
        )

    def test_record_and_count(self):
        rec = Recorder()
        rec.record(self._sample_result(), label="test1")
        assert rec.count == 1

    def test_record_multiple(self):
        rec = Recorder()
        rec.record(self._sample_result("A"), label="a")
        rec.record(self._sample_result("B"), label="b")
        assert rec.count == 2

    def test_get_recording(self):
        rec = Recorder()
        rec.record(self._sample_result(), label="test")
        entry = rec.get(0)
        assert entry["label"] == "test"
        assert entry["response"] == "Hello"
        assert len(entry["tool_calls"]) == 1

    def test_recordings_list(self):
        rec = Recorder()
        rec.record(self._sample_result(), label="a")
        recs = rec.recordings
        assert len(recs) == 1
        assert recs[0]["label"] == "a"

    def test_clear(self):
        rec = Recorder()
        rec.record(self._sample_result())
        rec.clear()
        assert rec.count == 0

    def test_export_json(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder()
            rec.record(self._sample_result(), label="export_test")
            files = rec.export(d, format="json")

            assert len(files) == 1
            assert files[0].name == "export_test.json"
            data = json.loads(files[0].read_text())
            assert data["response"] == "Hello"

    def test_export_default_label(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder()
            rec.record(self._sample_result())  # No label
            files = rec.export(d)
            assert files[0].name == "recording_000.json"

    def test_export_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder()
            rec.record(self._sample_result(), label="yaml_test")
            files = rec.export(d, format="yaml")
            assert files[0].name == "yaml_test.yaml"
            content = files[0].read_text()
            assert "Hello" in content

    def test_export_invalid_format(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder()
            rec.record(self._sample_result())
            with pytest.raises(ValueError, match="Unsupported format"):
                rec.export(d, format="xml")

    def test_trace_in_recording(self):
        rec = Recorder()
        rec.record(self._sample_result(), label="trace_test")
        entry = rec.get(0)
        assert len(entry["trace"]) == 2
        assert entry["trace"][0]["role"] == "user"
