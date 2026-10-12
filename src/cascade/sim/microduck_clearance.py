# SPDX-License-Identifier: Apache-2.0
"""Approximate equipment clearance from RECORDED body poses (no simulation).

For every frame and robot: the minimum distance from any of the 16 rigid-body ORIGINS to each obstacle's ORIENTED
bounding box (course Gprims with a collision API, excluding floor/stage/ramp surfaces; exact for boxes, conservative
for cylinders). Body origins sit inside the links, whose surfaces extend a few centimetres further, so this is an
indicator, not a contact sensor: below ``CONTACT_M`` it is reported as "possible contact".
"""
from __future__ import annotations

import numpy as np

SURFACES = ("Floor", "Ground", "Stage", "Ramp", "Lane")
CONTACT_M = 0.03
METHOD = "recorded body origins vs obstacle oriented bounding boxes (indicator, not a contact sensor)"


def obstacle_boxes(scene_usd):
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(scene_usd))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    boxes = {}
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Gprim) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.GetName().startswith(SURFACES):
            continue
        b = cache.ComputeWorldBound(prim)
        r = b.GetRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        if r.IsEmpty() or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(hi - lo > 1e3):
            continue  # empty / degenerate bound (e.g. a mesh without points)
        boxes[prim.GetName()] = (lo, hi, np.array(b.GetMatrix(), dtype=np.float64).T)  # local range + local->world
    return boxes


def _distance_to_box(pos, box):
    """Exact distance from world points (..., 3) to an (optionally affine-transformed) box (lo, hi[, M])."""
    lo, hi = box[0], box[1]
    if len(box) < 3:
        return np.linalg.norm(np.maximum(np.maximum(lo - pos, 0.0), pos - hi), axis=-1)
    M = box[2]
    Minv = np.linalg.inv(M)
    local = pos @ Minv[:3, :3].T + Minv[:3, 3]
    nearest_local = np.clip(local, lo, hi)
    nearest_world = nearest_local @ M[:3, :3].T + M[:3, 3]
    return np.linalg.norm(pos - nearest_world, axis=-1)


def clearance_report(body_pos: np.ndarray, robot_names: list[str], boxes: dict) -> dict:
    """body_pos: (T, R, B, 3) world positions of body origins."""
    out = {}
    for name, box in boxes.items():
        d = _distance_to_box(body_pos, box)
        per_robot = d.min(axis=(0, 2))
        for ri, r in enumerate(robot_names):
            out.setdefault(r, {})[name] = round(float(per_robot[ri]), 3)
    report = {}
    for r in robot_names:
        per = out.get(r, {})
        if not per:
            report[r] = {"min_body_origin_to_obstacle_box_m": None, "possible_contact": False, "method": METHOD}
            continue
        closest = min(per, key=per.get)
        report[r] = {"min_body_origin_to_obstacle_box_m": per[closest], "closest_obstacle": closest,
                     "possible_contact": per[closest] < CONTACT_M, "per_obstacle_m": per, "method": METHOD}
    return report


def apply_to_verification(verification: dict, scenario_verify: dict, report: dict) -> dict:
    """Robots whose verify spec sets ``no_contact: true`` fail on a possible contact."""
    for r, spec in (scenario_verify or {}).items():
        if spec.get("no_contact") and r in verification and r in report:
            v = verification[r]
            v["clearance_m"] = report[r]["min_body_origin_to_obstacle_box_m"]
            v["possible_contact"] = report[r]["possible_contact"]
            if report[r]["possible_contact"]:
                v["passed"] = False
                v["failed_because"] = f"possible contact with {report[r].get('closest_obstacle')}"
    return verification
