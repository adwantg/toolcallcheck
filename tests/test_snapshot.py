"""Tests for snapshot testing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from agent_test.result import AgentResult, ToolCall
from agent_test.snapshot import assert_snapshot


class TestSnapshot:
    @pytest.fixture
    def snap_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def _result(self):
        return AgentResult(
            response="Hello world",
            tool_calls=[ToolCall(name="greet", args={"name": "Alice"}, response="Hi Alice")],
            model_used="gpt-4",
        )

    def test_create_snapshot_on_first_run(self, snap_dir):
        result = self._result()
        assert_snapshot(result, "test_snap", snapshot_dir=snap_dir)

        snap_path = Path(snap_dir) / "test_snap.json"
        assert snap_path.exists()

        data = json.loads(snap_path.read_text())
        assert data["response"] == "Hello world"
        assert len(data["tool_calls"]) == 1

    def test_match_existing_snapshot(self, snap_dir):
        result = self._result()
        # First run — creates snapshot
        assert_snapshot(result, "test_match", snapshot_dir=snap_dir)
        # Second run — should match
        assert_snapshot(result, "test_match", snapshot_dir=snap_dir)

    def test_mismatch_raises(self, snap_dir):
        result1 = self._result()
        assert_snapshot(result1, "test_drift", snapshot_dir=snap_dir)

        result2 = AgentResult(response="Different response")
        with pytest.raises(AssertionError, match="Snapshot mismatch"):
            assert_snapshot(result2, "test_drift", snapshot_dir=snap_dir)

    def test_update_mode(self, snap_dir):
        result1 = self._result()
        assert_snapshot(result1, "test_update", snapshot_dir=snap_dir)

        result2 = AgentResult(response="Updated response")
        # Should not raise with update=True
        assert_snapshot(result2, "test_update", snapshot_dir=snap_dir, update=True)

        # Verify updated
        data = json.loads(Path(os.path.join(snap_dir, "test_update.json")).read_text())
        assert data["response"] == "Updated response"

    def test_exclude_tool_calls(self, snap_dir):
        result = self._result()
        assert_snapshot(
            result,
            "no_tools",
            snapshot_dir=snap_dir,
            include_tool_calls=False,
        )
        data = json.loads(Path(os.path.join(snap_dir, "no_tools.json")).read_text())
        assert "tool_calls" not in data

    def test_include_model(self, snap_dir):
        result = self._result()
        assert_snapshot(
            result,
            "with_model",
            snapshot_dir=snap_dir,
            include_model=True,
        )
        data = json.loads(Path(os.path.join(snap_dir, "with_model.json")).read_text())
        assert data["model_used"] == "gpt-4"
