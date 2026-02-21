"""
ContentCreator - Checkpoint / Stage-Cache Manager

Saves pipeline state after every stage so a crashed or interrupted run
can be resumed from the last successful checkpoint instead of starting
from scratch.

Checkpoint file layout (one per project directory):
    <project_dir>/checkpoint.json

The file stores:
    - original request parameters (prompt, platform, options)
    - which stages have completed
    - paths to every intermediate artifact produced so far
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


class CheckpointManager:
    """Persist and restore pipeline progress."""

    FILENAME = "checkpoint.json"

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.path = str(Path(project_dir) / self.FILENAME)

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    def save(
        self,
        *,
        params: Dict[str, Any],
        completed_stages: List[str],
        artifacts: Dict[str, Any],
        active_stages: List[str],
    ) -> None:
        """Write checkpoint to disk after a stage completes."""
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "params": params,
            "active_stages": active_stages,
            "completed_stages": completed_stages,
            "artifacts": artifacts,
        }
        os.makedirs(self.project_dir, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        # Atomic-ish rename (Windows replaces if dst exists on 3.12+)
        os.replace(tmp, self.path)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    def load(self) -> Optional[Dict[str, Any]]:
        """Load existing checkpoint, or return None."""
        if not os.path.isfile(self.path):
            return None
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != 1:
                return None
            return data
        except (json.JSONDecodeError, OSError):
            return None

    # -----------------------------------------------------------------
    # Validate
    # -----------------------------------------------------------------

    def validate(self, checkpoint: Dict[str, Any]) -> bool:
        """Check that all artifact files referenced still exist."""
        arts = checkpoint.get("artifacts", {})

        def _check(val: Any) -> bool:
            if isinstance(val, str) and (val.endswith((".mp3", ".wav", ".png",
                    ".jpg", ".mp4", ".srt", ".json"))):
                return os.path.isfile(val)
            if isinstance(val, list):
                return all(_check(v) for v in val)
            return True  # non-path values are fine

        return all(_check(v) for v in arts.values())

    # -----------------------------------------------------------------
    # Delete (on successful completion)
    # -----------------------------------------------------------------

    def delete(self) -> None:
        """Remove checkpoint file after a successful full run."""
        try:
            if os.path.isfile(self.path):
                os.remove(self.path)
        except OSError:
            pass

    # -----------------------------------------------------------------
    # Scan output directory for resumable projects
    # -----------------------------------------------------------------

    @staticmethod
    def find_resumable(output_dir: str) -> List[Dict[str, Any]]:
        """Return a list of resumable checkpoints found under *output_dir*.

        Each entry has:
            project_dir, updated_at, params, completed_stages, total_stages
        """
        results: List[Dict[str, Any]] = []
        if not os.path.isdir(output_dir):
            return results

        for entry in os.scandir(output_dir):
            if not entry.is_dir():
                continue
            mgr = CheckpointManager(entry.path)
            cp = mgr.load()
            if cp is None:
                continue
            # Only include if there are remaining stages
            completed = set(cp.get("completed_stages", []))
            active = cp.get("active_stages", [])
            if completed >= set(active):
                continue  # fully done — nothing to resume
            if not mgr.validate(cp):
                continue  # files deleted — can't resume

            results.append({
                "project_dir": entry.path,
                "project_name": entry.name,
                "updated_at": cp.get("updated_at"),
                "params": cp.get("params", {}),
                "completed_stages": list(completed),
                "active_stages": active,
                "remaining_stages": [s for s in active if s not in completed],
            })

        # Most recent first
        results.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return results
