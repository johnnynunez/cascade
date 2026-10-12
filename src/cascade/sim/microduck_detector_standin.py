# SPDX-License-Identifier: Apache-2.0
"""Object detector STAND-IN for the agentic MicroDuck demos.

The MicroDuck's 48x36 head RGB (the policy's training input) renders USD colours, but only within its 1.5 m clip
range, and the floor itself renders cyan-blue: it cannot find a red or blue box across a room. Instead of pretending,
this module reports SIMULATOR GROUND TRUTH for
a scenario-declared list of objects, restricted to what an onboard detector on the head camera could plausibly see:

* inside the head camera's horizontal field of view (82.3 deg = 2 atan(20.955 / 2 / 12) from the gtceu camera cfg),
* within ``max_range_m`` (default 2.5 m) of the robot,
* not occluded by a wall footprint (2D segment test against declared occluder boxes).

Every result says ``source = "simulator ground truth (detector stand-in)"``; positions are rounded to 5 cm.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

HEAD_CAMERA_HFOV_DEG = 2.0 * math.degrees(math.atan(20.955 / 2.0 / 12.0))
SOURCE = "simulator ground truth (detector stand-in), head-camera field of view only"


@dataclass(frozen=True)
class SceneObject:
    name: str
    label: str
    x: float
    y: float
    half_x: float
    half_y: float


def _segment_hits_box(x0, y0, x1, y1, box: SceneObject, margin=0.0) -> bool:
    """Liang-Barsky clip of segment p0->p1 against an axis-aligned footprint."""
    xmin, xmax = box.x - box.half_x - margin, box.x + box.half_x + margin
    ymin, ymax = box.y - box.half_y - margin, box.y + box.half_y + margin
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


class ObjectDetectorStandIn:
    def __init__(self, objects: list[SceneObject], occluders: list[SceneObject], *, max_range_m=2.5,
                 hfov_deg=HEAD_CAMERA_HFOV_DEG):
        self.objects, self.occluders = list(objects), list(occluders)
        self.max_range_m, self.hfov_deg = float(max_range_m), float(hfov_deg)

    def detect(self, x: float, y: float, yaw_rad: float) -> dict:
        seen = []
        for o in self.objects:
            dx, dy = o.x - x, o.y - y
            dist = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(math.sin(math.atan2(dy, dx) - yaw_rad), math.cos(math.atan2(dy, dx) - yaw_rad)))
            if dist > self.max_range_m or abs(bearing) > self.hfov_deg / 2:
                continue
            if any(_segment_hits_box(x, y, o.x, o.y, w) for w in self.occluders):
                continue
            seen.append({"label": o.label, "bearing_deg": round(bearing, 1), "distance_m": round(dist, 2),
                         "world_xy": [round(o.x / 0.05) * 0.05, round(o.y / 0.05) * 0.05]})
        seen.sort(key=lambda d: d["distance_m"])
        return {"source": SOURCE, "field_of_view_deg": round(self.hfov_deg, 1), "max_range_m": self.max_range_m,
                "detections": seen}


def load_from_usd(scene_usd: Path, detectable: dict, occluder_prefixes=("Wall", "Outer", "Inner")):
    """``detectable``: {prim name: label}. Occluders: every Gprim whose name starts with one of the prefixes."""
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(str(scene_usd))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    objects, occluders = [], []
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        name = prim.GetName()
        is_obj = name in detectable
        is_occ = name.startswith(tuple(occluder_prefixes))
        if not (is_obj or is_occ):
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = r.GetMin(), r.GetMax()
        so = SceneObject(name, detectable.get(name, name), (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                         (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2)
        (objects if is_obj else occluders).append(so)
    missing = set(detectable) - {o.name for o in objects}
    if missing:
        raise ValueError(f"detectable prims not found in {scene_usd}: {sorted(missing)}")
    return objects, occluders
