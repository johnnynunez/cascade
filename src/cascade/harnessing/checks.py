# SPDX-License-Identifier: Apache-2.0
"""Hidden task checks: did the rollout do the task, measured on the simulator's ground truth.

Harness² scores every execution with a check the improver never sees (answer-free). CASCADE's own
`TaskReport.success` is not that check: it folds in the agent's claim and the decomposed
milestones, and a past-tense milestone ("grasp and lift the cube") can no longer be confirmed at
the end of a task that put the cube down, so a correct execution reads as a failure.

A `TaskCheck` reads the physics truth channel the runtime's verifier already uses
(`SkillRuntime._object_pose`: MuJoCo free bodies, the Isaac bridge's prims) at the start and the
end of the rollout, plus the held flag, the arm's joints and the agent's own `task_done` claim. It
runs only offline, in the harnessing loop; nothing of it reaches the agent or the improver.

Shape (JSON, per task)::

    {"objects": [{"object": "red cube", "in_region": "delivery_area"},
                 {"object": "blue cube", "unmoved": true}],
     "released": true, "arm_home_tol_rad": 0.1, "honest_claim": true}

A single-object check may put the object's keys at the top level (``{"object": ..., ...}``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

_OBJECT_KEYS = {"object", "in_region", "displaced_xy_m", "at_start_of", "on_object", "unmoved", "tol_m",
                "on_table_tol_m"}
_TOP_KEYS = {"objects", "released", "arm_home_tol_rad", "honest_claim"}


def _opt_float(value) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ObjectCheck:
    """What must hold for one object at the end. Every criterion set must pass."""

    object: str
    #: "delivery_area" (the arm profile's `mj_delivery_area.bounds_xy_m`) or [[x0, y0], [x1, y1]]
    in_region: Any = None
    #: xy displacement from its start, robot base frame (+x away from the base, +y the robot's left)
    displaced_xy_m: tuple[float, float] | None = None
    #: ends where ANOTHER tracked object started (a swap)
    at_start_of: str | None = None
    #: ends on top of another tracked object (xy over its end pose, resting higher than it)
    on_object: str | None = None
    #: ends where it started
    unmoved: bool = False
    tol_m: float = 0.02
    #: ends at the height it started (on the support, not in the jaws, not fallen); off for on_object
    on_table_tol_m: float | None = 0.02

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectCheck":
        unknown = set(data) - _OBJECT_KEYS
        if unknown:
            raise ValueError(f"unknown object-check keys: {sorted(unknown)}")
        if not data.get("object"):
            raise ValueError(f"an object check needs `object`: {data!r}")
        disp = data.get("displaced_xy_m")
        on_object = data.get("on_object")
        return cls(object=str(data["object"]), in_region=data.get("in_region"),
                   displaced_xy_m=None if disp is None else (float(disp[0]), float(disp[1])),
                   at_start_of=data.get("at_start_of"), on_object=on_object,
                   unmoved=bool(data.get("unmoved", False)), tol_m=float(data.get("tol_m", 0.02)),
                   on_table_tol_m=None if on_object else _opt_float(data.get("on_table_tol_m", 0.02)))


@dataclass(frozen=True)
class TaskCheck:
    objects: tuple[ObjectCheck, ...] = field(default_factory=tuple)
    released: bool = False
    arm_home_tol_rad: float | None = None
    #: the agent's own task_done claim must match what physics measured
    honest_claim: bool = False

    @classmethod
    def from_dict(cls, data: dict | None) -> "TaskCheck | None":
        if not data:
            return None
        data = dict(data)
        flat = {k: data.pop(k) for k in list(data) if k in _OBJECT_KEYS}
        if flat and "objects" in data:
            raise ValueError("use either `objects` or top-level object keys, not both")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            raise ValueError(f"unknown check keys: {sorted(unknown)}")
        if flat and not flat.get("object"):
            raise ValueError("in_region / displaced_xy_m / ... need an object")
        objects = tuple(ObjectCheck.from_dict(o) for o in ([flat] if flat else data.get("objects") or []))
        names = [o.object for o in objects]
        for o in objects:
            for ref in (o.at_start_of, o.on_object):
                if ref is not None and ref not in names:
                    raise ValueError(f"{o.object!r} refers to {ref!r}, which the check does not track")
        return cls(objects=objects, released=bool(data.get("released", False)),
                   arm_home_tol_rad=_opt_float(data.get("arm_home_tol_rad")),
                   honest_claim=bool(data.get("honest_claim", False)))

    def as_dict(self) -> dict:
        return asdict(self)


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


def _region(spec, runtime) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(spec, str):
        if spec != "delivery_area":
            raise ValueError(f"unknown region {spec!r}")
        area = runtime.cfg.arm.get("mj_delivery_area") or {}
        spec = area.get("bounds_xy_m")
        if spec is None:
            return None
    lo, hi = np.asarray(spec[0], dtype=float), np.asarray(spec[1], dtype=float)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def _r(v) -> list[float] | None:
    return None if v is None else [round(float(x), 4) for x in v]


class CheckProbe:
    """Captures every tracked object's start pose when built; `evaluate()` at the end."""

    def __init__(self, check: TaskCheck, runtime, arm=None):
        self.check = check
        self.start: dict[str, Any] = {o.object: _truth(runtime, o.object) for o in check.objects}
        if check.objects and any(v is None for v in self.start.values()) and arm is not None:
            # a lazily bound truth channel binds on the arm's first read
            try:
                arm.get_state()
            except Exception:  # noqa: BLE001
                pass
            self.start = {name: _truth(runtime, name) for name in self.start}

    def _object_rows(self, o: ObjectCheck, end: dict, runtime) -> dict:
        s, e = self.start.get(o.object), end.get(o.object)
        row: dict[str, Any] = {"start": _r(s), "end": _r(e)}
        if o.in_region is not None:
            box = _region(o.in_region, runtime)
            if e is None or box is None:
                row["in_region"] = {"ok": None, "why": "no truth pose" if e is None else "no region bounds"}
            else:
                lo, hi = box
                row["in_region"] = {"ok": bool((e[:2] >= lo).all() and (e[:2] <= hi).all())}
        if o.displaced_xy_m is not None or o.unmoved:
            want = np.asarray(o.displaced_xy_m if o.displaced_xy_m is not None else (0.0, 0.0))
            key = "displaced" if o.displaced_xy_m is not None else "unmoved"
            if e is None or s is None:
                row[key] = {"ok": None, "why": "no truth pose"}
            else:
                moved = e[:2] - s[:2]
                err = float(np.linalg.norm(moved - want))
                row[key] = {"ok": err <= o.tol_m, "moved_xy_m": _r(moved), "err_m": round(err, 4)}
        if o.at_start_of is not None:
            other = self.start.get(o.at_start_of)
            if e is None or other is None:
                row["at_start_of"] = {"ok": None, "why": "no truth pose"}
            else:
                err = float(np.linalg.norm(e[:2] - other[:2]))
                row["at_start_of"] = {"ok": err <= o.tol_m, "err_m": round(err, 4)}
        if o.on_object is not None:
            base = end.get(o.on_object)
            if e is None or base is None:
                row["on_object"] = {"ok": None, "why": "no truth pose"}
            else:
                err = float(np.linalg.norm(e[:2] - base[:2]))
                above = float(e[2] - base[2])
                row["on_object"] = {"ok": err <= o.tol_m and above > 0.01, "err_m": round(err, 4),
                                    "above_m": round(above, 4)}
        if o.on_table_tol_m is not None:
            if e is None or s is None:
                row["on_table"] = {"ok": None, "why": "no truth pose"}
            else:
                dz = float(e[2] - s[2])
                row["on_table"] = {"ok": abs(dz) <= o.on_table_tol_m, "dz_m": round(dz, 4)}
        return row

    def evaluate(self, runtime, arm=None, *, claimed: bool | None = None) -> tuple[bool | None, dict]:
        """``(passed, rows)``. ``passed`` is None when any criterion could not be measured."""
        c = self.check
        end = {o.object: _truth(runtime, o.object) for o in c.objects}
        rows: dict[str, dict] = {o.object: self._object_rows(o, end, runtime) for o in c.objects}
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
        oks = []
        for row in rows.values():
            if "ok" in row:
                oks.append(row["ok"])
            else:
                oks += [v["ok"] for v in row.values() if isinstance(v, dict) and "ok" in v]
        physics = None if any(v is None for v in oks) else all(oks)
        passed = physics
        if c.honest_claim:
            honest = None if physics is None else (claimed is True) == physics
            rows["honest_claim"] = {"ok": honest, "claimed": claimed, "physics": physics}
            passed = None if physics is None else bool(physics and honest)
        rows["physics_passed"] = {"value": physics}
        return passed, rows
