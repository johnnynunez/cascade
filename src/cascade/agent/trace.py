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
from typing import Any, Callable

import cv2
import numpy as np


class TraceLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        (self.run_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._step = 0
        #: optional () -> dict of verified sidecar backends, appended to summaries
        self.backends_fn: Callable[[], dict] | None = None

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
        }
        with open(self._trace_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._step += 1

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
