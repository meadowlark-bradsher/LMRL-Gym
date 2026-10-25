"""Structured, append-only logging for Text-Nav x Jev experiment runs.

Writes the output files named in EXPERIMENT_DESIGN.md section 9:

* ``run.json``      — one record of run-level metadata (model id, timestamp,
                      git commit, seeds, step limit, condition set).
* ``episodes.jsonl``— one line per episode: condition, seed, outcome, milestone
                      summary, and aggregate per-episode diagnostics.
* ``steps.jsonl``   — one line per step: the exact state parts sent to the
                      policy, the admissible set, the chosen command, Jev's full
                      option probabilities and confidence, the fact-derived world
                      snapshot, and opportunity flags.
* ``snapshots.jsonl``/``pairwise.jsonl`` — written by the Experiment C prober
                      via :meth:`write`, same mechanism.

Everything is plain JSON lines so the analysis scripts can stream them without
loading a run into memory, and a partial run is still readable. No third-party
dependencies.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Optional, TextIO


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def default_run_dir(root: str, tag: str) -> str:
    """A timestamped run directory under ``root`` (keeps runs from colliding)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(root, "{}_{}".format(stamp, tag))


class Recorder:
    """Owns a run directory and its open JSONL files.

    Use as a context manager so the files are always flushed and closed::

        with Recorder(run_dir, meta) as rec:
            rec.episode({...})
            rec.step({...})
    """

    def __init__(self, run_dir: str, meta: Optional[Dict[str, Any]] = None):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self._files: Dict[str, TextIO] = {}
        self.meta = dict(meta or {})
        self.meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.meta.setdefault("git_commit", _git_commit())
        self._write_run_meta()

    # -- lifecycle -----------------------------------------------------------

    def _write_run_meta(self) -> None:
        with open(os.path.join(self.run_dir, "run.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
            f.write("\n")

    def _file(self, name: str) -> TextIO:
        if name not in self._files:
            self._files[name] = open(os.path.join(self.run_dir, name + ".jsonl"), "a")
        return self._files[name]

    def write(self, stream: str, row: Dict[str, Any]) -> None:
        """Append one JSON row to ``<stream>.jsonl`` and flush it."""
        f = self._file(stream)
        f.write(json.dumps(row, default=str))
        f.write("\n")
        f.flush()

    # -- convenience wrappers for the standard streams -----------------------

    def episode(self, row: Dict[str, Any]) -> None:
        self.write("episodes", row)

    def step(self, row: Dict[str, Any]) -> None:
        self.write("steps", row)

    def snapshot(self, row: Dict[str, Any]) -> None:
        self.write("snapshots", row)

    def pairwise(self, row: Dict[str, Any]) -> None:
        self.write("pairwise", row)

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.flush()
                f.close()
            except Exception:
                pass
        self._files.clear()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
