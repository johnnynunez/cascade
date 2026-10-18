# SPDX-License-Identifier: Apache-2.0
"""Hidden task checks: did the rollout do the task, measured on the simulator's ground truth.

Harness² scores every execution with a check the improver never sees (answer-free). CASCADE's own
`TaskReport.success` is not that check: it folds in the agent's claim and the decomposed
milestones, and a past-tense milestone ("grasp and lift the cube") can no longer be confirmed at
the end of a task that put the cube down, so a correct execution reads as a failure.

A `TaskCheck` reads the physics truth channel the runtime's verifier already uses
(`SkillRuntime._object_pose`: MuJoCo free bodies, the Isaac bridge's prims) at the start and the
end of the rollout, plus the held flag and the arm's joints. It runs only offline, in the
harnessing loop; nothing of it reaches the agent or the improver.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

_KEYS = {"object", "in_region", "displaced_xy_m", "tol_m", "released", "on_table_tol_m", "arm_home_tol_rad"}


@dataclass(frozen=True)
class TaskCheck:
    """What must hold at the end of the rollout. Every criterion set must pass."""

    object: str | None = None
    #: "delivery_area" (the arm profile's `mj_delivery_area.bounds_xy_m`) or [[x0, y0], [x1, y1]]
    in_region: Any = None
    #: the object's xy displacement from where it started, robot base frame (+y = the robot's left)
    displaced_xy_m: tuple[float, float] | None = None
    tol_m: float = 0.02
    released: bool = False
    #: the object ends at the height it started (on the support, not in the jaws or fallen)
    on_table_tol_m: float | None = 0.02
    arm_home_tol_rad: float | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskCheck | None":
        if not data:
            return None
        unknown = set(data) - _KEYS
        if unknown:
            raise ValueError(f"unknown check keys: {sorted(unknown)}")
        needs_object = data.get("in_region") is not None or data.get("displaced_xy_m") is not None
        if needs_object and not data.get("object"):
            raise ValueError("in_region / displaced_xy_m need an object")
        disp = data.get("displaced_xy_m")
        return cls(object=data.get("object"), in_region=data.get("in_region"),
                   displaced_xy_m=None if disp is None else (float(disp[0]), float(disp[1])),
                   tol_m=float(data.get("tol_m", 0.02)), released=bool(data.get("released", False)),
                   on_table_tol_m=_opt_float(data.get("on_table_tol_m", 0.02)) if data.get("object") else None,
                   arm_home_tol_rad=(None if data.get("arm_home_tol_rad") is None
                                     else float(data["arm_home_tol_rad"])))

    def as_dict(self) -> dict:
        return asdict(self)


def _opt_float(value) -> float | None:
    return None if value is None else float(value)


def _truth(runtime, label: str):
    fn = getattr(runtime, "_object_pose", None)
    if fn is None:
        return None
    try:
        pose = fn(label)
    except Exception:  # noqa: BLE001 -- an unreadable channel is "unknown", never a pass
        return None
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=float).reshape(-1)
    return pose[:3] if pose.size >= 3 and np.isfinite(pose[:3]).all() else None


def _region(check: TaskCheck, runtime) -> tuple[np.ndarray, np.ndarray] | None:
    spec = check.in_region
    if isinstance(spec, str):
        if spec != "delivery_area":
            raise ValueError(f"unknown region {spec!r}")
        area = runtime.cfg.arm.get("mj_delivery_area") or {}
        spec = area.get("bounds_xy_m")
        if spec is None:
            return None
    lo, hi = np.asarray(spec[0], dtype=float), np.asarray(spec[1], dtype=float)
    return np.minimum(lo, hi), np.maximum(lo, hi)


class CheckProbe:
    """Captures the start state when built; `evaluate()` at the end of the rollout."""

    def __init__(self, check: TaskCheck, runtime, arm=None):
        self.check = check
        self.start = None
        if check.object:
            self.start = _truth(runtime, check.object)
            if self.start is None and arm is not None:
                # a lazily bound truth channel binds on the arm's first read
                try:
                    arm.get_state()
                except Exception:  # noqa: BLE001
                    pass
                self.start = _truth(runtime, check.object)

    def evaluate(self, runtime, arm=None) -> tuple[bool | None, dict]:
        c = self.check
        rows: dict[str, dict] = {}
        end = _truth(runtime, c.object) if c.object else None
        if c.object:
            rows["object_pose"] = {"start": None if self.start is None else [round(float(v), 4) for v in self.start],
                                   "end": None if end is None else [round(float(v), 4) for v in end]}
        if c.in_region is not None:
            box = _region(c, runtime)
            if end is None or box is None:
                rows["in_region"] = {"ok": None, "why": "no truth pose" if end is None else "no region bounds"}
            else:
                lo, hi = box
                inside = bool((end[:2] >= lo).all() and (end[:2] <= hi).all())
                rows["in_region"] = {"ok": inside, "bounds": [lo.tolist(), hi.tolist()]}
        if c.displaced_xy_m is not None:
            if end is None or self.start is None:
                rows["displaced"] = {"ok": None, "why": "no truth pose"}
            else:
                moved = end[:2] - self.start[:2]
                err = float(np.linalg.norm(moved - np.asarray(c.displaced_xy_m)))
                rows["displaced"] = {"ok": err <= c.tol_m, "moved_xy_m": [round(float(v), 4) for v in moved],
                                     "err_m": round(err, 4)}
        if c.on_table_tol_m is not None and c.object:
            if end is None or self.start is None:
                rows["on_table"] = {"ok": None, "why": "no truth pose"}
            else:
                dz = float(end[2] - self.start[2])
                rows["on_table"] = {"ok": abs(dz) <= c.on_table_tol_m, "dz_m": round(dz, 4)}
        if c.released:
            held = getattr(runtime, "held_object", None)
            provisional = getattr(runtime, "_held_provisional", None)
            rows["released"] = {"ok": held is None and provisional is None, "held": held}
        if c.arm_home_tol_rad is not None:
            home = runtime.cfg.arm.get("home_q")
            try:
                q = np.asarray((arm or runtime.arm).get_state().q, dtype=float).reshape(-1)
                home = np.asarray(home, dtype=float).reshape(-1)
                n = min(len(q), len(home))
                worst = float(np.max(np.abs(q[:n] - home[:n]))) if n else math.inf
                rows["arm_home"] = {"ok": worst <= c.arm_home_tol_rad, "worst_rad": round(worst, 4)}
            except Exception as exc:  # noqa: BLE001
                rows["arm_home"] = {"ok": None, "why": f"{type(exc).__name__}: {exc}"}
        oks = [r["ok"] for r in rows.values() if "ok" in r]
        passed = None if any(v is None for v in oks) else all(oks)
        return passed, rows
