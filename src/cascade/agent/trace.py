"""ASPIRE-style multimodal trace: every skill call leaves evidence.

Per run directory:
    trace.jsonl   - one JSON record per skill call (args, duration, result,
                    keyframe paths)
    keyframes/    - JPEG snapshots immediately before/after each call
    summary.txt   - final human-readable outcome

ASPIRE's ablation credits this per-primitive evidence with the single largest
success jump (14% -> 62%); it is also what makes post-hoc debugging of a live
demo possible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TraceLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._step = 0

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
    ) -> None:
        rec = {
            "step": self._step,
            "t": time.time(),
            "skill": skill,
            "args": _jsonable(args),
            "duration_ms": round(duration_ms, 1),
            "result": _jsonable(result),
            "keyframe_before": keyframe_before,
            "keyframe_after": keyframe_after,
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._step += 1

    def finish(self, summary: str) -> None:
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
