"""ASPIRE-style multimodal trace: every skill call leaves evidence.

Per run directory:
    trace.jsonl   - one JSON record per skill call (args, duration, result,
                    keyframe paths)
    keyframes/    - JPEG snapshots immediately before/after each call; for
                    motion skills on a rig with a wrist view also one
                    `<step>_<skill>_{before,after}_wrist_<stream>.jpg` per
                    wrist stream (`keyframe_before_wrists` /
                    `keyframe_after_wrists`: `{stream: path}`, null when the
                    rig has none -- the judge then repeats the front image)
    summary.txt   - final human-readable outcome
    run.json      - the run's manifest (`stamp`, written once at build): the
                    ASPIRE note library the run read (`library_sha`), the notes
                    it served, its profile class and purpose. Every row
                    recorded after the stamp carries the same `library_sha`
                    (null when nothing stamped the run: mobile, composed and
                    microduck runtimes, or a manifest that could not be written)

ASPIRE's ablation credits this per-primitive evidence with the single largest
success jump (14% -> 62%); it is also what makes post-hoc debugging of a live
demo possible.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


class TraceLogger:
    #: sha256 of the ASPIRE note library this run read (`stamp`); None until
    #: stamped. A class default, so loggers built with `__new__` read None too.
    library_sha: str | None = None

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._step = 0
        #: optional () -> dict of verified sidecar backends, appended to summaries
        self.backends_fn: Callable[[], dict] | None = None

    def stamp(self, meta: dict[str, Any]) -> None:
        """Write `run.json` = `meta` atomically (a temporary file in the run
        directory, then `os.replace`), then stamp every later row with
        `meta["library_sha"]`. A write that fails raises OSError, leaves any
        earlier run.json as it was and stamps nothing."""
        path = self.run_dir / "run.json"
        tmp = self.run_dir / f".run.json.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        data = (json.dumps(_jsonable(meta), indent=2) + "\n").encode("utf-8")
        try:
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.library_sha = meta.get("library_sha")

    def save_keyframe(self, rgb: np.ndarray | None, tag: str) -> str | None:
        if rgb is None:
            return None
        name = f"keyframes/{self._step:04d}_{tag}.jpg"
        cv2.imwrite(str(self.run_dir / name), rgb)
        return name

    def record(
        self,
        skill: str,
        args: dict[str, Any],
        result: dict[str, Any],
        duration_ms: float,
        keyframe_before: str | None = None,
        keyframe_after: str | None = None,
        tier: str | None = None,
        context: dict[str, Any] | None = None,
        keyframe_before_wrists: dict[str, str] | None = None,
        keyframe_after_wrists: dict[str, str] | None = None,
    ) -> None:
        rec = {
            "step": self._step,
            "t": time.time(),
            "skill": skill,
            "tier": tier,
            "args": _jsonable(args),
            "context": _jsonable(context),
            "duration_ms": round(duration_ms, 1),
            "result": _jsonable(result),
            "keyframe_before": keyframe_before,
            "keyframe_after": keyframe_after,
            # wrist views, {stream name: path}; null (not {}) when there are
            # none so a reader can tell "no wrist camera" from "none saved"
            "keyframe_before_wrists": dict(keyframe_before_wrists) if keyframe_before_wrists else None,
            "keyframe_after_wrists": dict(keyframe_after_wrists) if keyframe_after_wrists else None,
            # the note library this run read (`stamp`); null when unstamped
            "library_sha": self.library_sha,
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._step += 1

    def rows(self) -> list[dict[str, Any]]:
        """The recorded steps, oldest first, read back from `trace.jsonl`.

        The file is the authority (`recall_step` serves what a post-hoc
        reader would see, not an in-memory shadow of it); a line that does
        not parse is skipped rather than failing the whole recall."""
        if not self._trace_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self._trace_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def finish(self, summary: str) -> None:
        # The runtime registers `backends_fn` so every summary carries WHICH
        # sidecars were verifiably in the loop (grasp planner, occupancy
        # map) -- a declared-but-absent backend must not pass for a working
        # one in the artifact people read after the demo.
        if self.backends_fn is not None:
            try:
                b = self.backends_fn()
                summary += "\nbackends: " + ", ".join(f"{k}={v}" for k, v in b.items())
            except Exception:  # noqa: BLE001 -- never let reporting break a run
                pass
        (self.run_dir / "summary.txt").write_text(summary + "\n")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return np.round(obj, 4).tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj
