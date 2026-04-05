"""Snapshot testing for agent responses.

Compare agent output against saved snapshots for regression detection.
Optionally update snapshots when behavior changes intentionally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from toolcallcheck.result import AgentResult

_SNAPSHOT_DIR = ".toolcallcheck_snapshots"
_UPDATE_ENV_VAR = "TOOLCALLCHECK_UPDATE_SNAPSHOTS"


def assert_snapshot(
    result: AgentResult,
    snapshot_name: str,
    *,
    snapshot_dir: str | Path | None = None,
    update: bool | None = None,
    include_tool_calls: bool = True,
    include_model: bool = False,
) -> None:
    """Compare the result against a saved snapshot.

    On first run (no snapshot file exists), the snapshot is automatically
    created.  On subsequent runs, the result is compared against the saved
    snapshot.

    Parameters
    ----------
    result:
        The :class:`AgentResult` to snapshot.
    snapshot_name:
        A unique name for this snapshot (used as filename).
    snapshot_dir:
        Directory for snapshot files (default: ``.toolcallcheck_snapshots/``).
    update:
        If ``True``, overwrite the existing snapshot.  Can also be set
        via the ``TOOLCALLCHECK_UPDATE_SNAPSHOTS=1`` environment variable.
    include_tool_calls:
        Include tool calls in the snapshot (default ``True``).
    include_model:
        Include model metadata in the snapshot (default ``False``).
    """
    snapshot_dir_path = Path(_SNAPSHOT_DIR) if snapshot_dir is None else Path(snapshot_dir)
    should_update = update if update is not None else os.environ.get(_UPDATE_ENV_VAR) == "1"

    # Build snapshot data
    snap_data: dict[str, Any] = {"response": result.response}
    if include_tool_calls:
        snap_data["tool_calls"] = [tc.to_dict() for tc in result.tool_calls]
    if include_model:
        snap_data["model_used"] = result.model_used

    snap_path = snapshot_dir_path / f"{snapshot_name}.json"

    if not snap_path.exists() or should_update:
        # Create or update snapshot
        snapshot_dir_path.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(snap_data, indent=2, default=str) + "\n")
        return  # First run — snapshot created, test passes

    # Compare against existing snapshot
    existing = json.loads(snap_path.read_text())

    if existing != snap_data:
        # Build a diff message
        parts = [f"Snapshot mismatch for '{snapshot_name}':"]

        if existing.get("response") != snap_data.get("response"):
            parts.append("  Response changed:")
            parts.append(f"    saved:  {existing.get('response')!r}")
            parts.append(f"    actual: {snap_data.get('response')!r}")

        if existing.get("tool_calls") != snap_data.get("tool_calls"):
            parts.append("  Tool calls changed:")
            parts.append(f"    saved:  {json.dumps(existing.get('tool_calls'), default=str)}")
            parts.append(f"    actual: {json.dumps(snap_data.get('tool_calls'), default=str)}")

        parts.append("\n  To update: set TOOLCALLCHECK_UPDATE_SNAPSHOTS=1 or pass update=True")

        raise AssertionError("\n".join(parts))
