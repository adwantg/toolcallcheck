"""Request/response recording for debugging.

Capture the full conversation trace and tool call history, and export
to JSON or YAML files for offline analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_test.result import AgentResult


class Recorder:
    """Records agent invocation results for debugging and replay.

    Usage::

        recorder = Recorder()
        result = await runner.invoke("message")
        recorder.record(result, label="test_workflow_flow")

        # Export all recordings
        recorder.export("test_recordings/")

        # Or inspect in memory
        assert recorder.count == 1
    """

    def __init__(self) -> None:
        self._recordings: list[dict[str, Any]] = []

    def record(self, result: AgentResult, *, label: str | None = None) -> None:
        """Record an agent result.

        Parameters
        ----------
        result:
            The :class:`AgentResult` to record.
        label:
            Optional label for this recording (used as filename on export).
        """
        entry: dict[str, Any] = {
            "label": label,
            "response": result.response,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
            "model_used": result.model_used,
            "headers": result.headers,
            "metadata": result.metadata,
            "trace": [
                {
                    "role": te.role,
                    "content": te.content,
                    "tool_call": te.tool_call.to_dict() if te.tool_call else None,
                }
                for te in result.trace
            ],
        }
        self._recordings.append(entry)

    @property
    def recordings(self) -> list[dict[str, Any]]:
        """All recordings in order."""
        return list(self._recordings)

    @property
    def count(self) -> int:
        """Number of recordings captured."""
        return len(self._recordings)

    def get(self, index: int) -> dict[str, Any]:
        """Get a recording by index."""
        return self._recordings[index]

    def export(
        self,
        output_dir: str | Path,
        *,
        format: str = "json",
    ) -> list[Path]:
        """Export all recordings to files.

        Parameters
        ----------
        output_dir:
            Directory to write recording files.
        format:
            Output format: ``"json"`` (default) or ``"yaml"``.

        Returns
        -------
        List of paths to the written files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for i, rec in enumerate(self._recordings):
            label = rec.get("label") or f"recording_{i:03d}"
            if format == "json":
                file_path = output_path / f"{label}.json"
                file_path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
            elif format == "yaml":
                import yaml

                file_path = output_path / f"{label}.yaml"
                file_path.write_text(yaml.dump(rec, default_flow_style=False))
            else:
                raise ValueError(f"Unsupported format: {format!r}")
            written.append(file_path)

        return written

    def clear(self) -> None:
        """Clear all recordings."""
        self._recordings.clear()
