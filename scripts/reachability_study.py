#!/usr/bin/env python3
"""Measured reachability of grasp approach families on the reBot B601-RS (B45).

Seeed feedback (8 Oct 2026): "very limited AREA constraints for skills, the
object has to be really close to the arm". The shipped workspace
(`configs/demo.yaml` `safety.workspace`, x 0.10..0.50, y +-0.30) came from a
TOP-DOWN-only IK probe, while learned grasp candidates (GraspGen-X) already
propose angled approaches. This script measures, offline and on the URDF
only, which TCP grasp poses each approach family can reach when every check
the runtime applies before motion is applied exactly as the runtime applies
it:

  * ``cascade.grasping.selector.select_grasp`` -- pregrasp IK seeded from the
    profile's ``home_q`` (the runtime's grasp seed), grasp IK seeded from the
    pregrasp solution, ``Kinematics.ik`` defaults (limit_margin 0.025 > harness
    joint_margin 0.02, 4 seeded restarts), and both jaw orientations (the
    selector's flip twin);
  * the harness part of ``SkillRuntime.skill_grasp_object``'s candidate vet
    (``_vet``), in its order: ``vet_pose(pregrasp)`` without exemption,
    ``vet_segment(home -> pregrasp)`` over the streamed waypoint profile, and
    the seven descent samples under the grasp exemption cylinder
    (``exempt_radius_m``, floor ``table_z - 0.06``); the carry-lift check when
    the profile enables it.

Live-scene gates (occupancy map, observed-finger gate, cuMotion collision
spheres) are not modelled: they only ever refuse MORE, never admit. Neither
is contact physics: a reachable pose is not a held object (live validation).

Passes (a smaller box can only refuse more, so passes 2 and 3 re-check only
what pass 1 reached):

  1. OPEN: the harness workspace AABB is opened (every other limit kept) to
     map where each family is reachable at all;
  2. DEFAULT: the profile's shipped box, i.e. what the arm can reach today;
  3. ENVELOPE: the derived opt-in box (`derive_envelope`) installed in the
     harness, so the admitted region is verified under the limit that will
     actually gate motion in the opt-in profiles.

Run (CPU only, a few minutes with workers):

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python scripts/reachability_study.py \\
        --out docs/evidence/reach-envelope-20261009/rebot_reachability.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

SCHEMA = "cascade.reachability_study/1"

#: Approach families, in report order. ``tilt_deg`` is the angle between the
#: approach and straight down; ``lean`` names the horizontal direction the
#: approach leans toward, relative to the point's radial direction
#: r = (x, y, 0)/|(x, y)| (from the robot base to the grasp point):
#:   out         leans away from the base, in the vertical plane through r
#:               (rotation about the tangential axis) -- reach outward;
#:   tangential  leans sideways, +-t with t = z x r (rotation ABOUT the radial
#:               axis); either sign counts;
#:   diagonal    leans halfway between the two, (r +- t)/sqrt(2); either sign
#:               counts. This is where the analytic planner leans for a square
#:               footprint yawed 45 degrees to the radial line (its lean is
#:               perpendicular to the jaw axis; see obb_grasp._angled_alternates).
#: ``side`` is a horizontal approach along +r.
FAMILIES: dict[str, dict] = {
    "topdown": {"tilt_deg": 0.0, "lean": "none"},
    "out15": {"tilt_deg": 15.0, "lean": "out"},
    "out30": {"tilt_deg": 30.0, "lean": "out"},
    "out45": {"tilt_deg": 45.0, "lean": "out"},
    "about_radial15": {"tilt_deg": 15.0, "lean": "tangential"},
    "about_radial30": {"tilt_deg": 30.0, "lean": "tangential"},
    "about_radial45": {"tilt_deg": 45.0, "lean": "tangential"},
    "diag30": {"tilt_deg": 30.0, "lean": "diagonal"},
    "diag45": {"tilt_deg": 45.0, "lean": "diagonal"},
    "side": {"tilt_deg": 90.0, "lean": "out"},
}
#: lean kinds evaluated on both sides (either one reaching counts)
_TWO_SIDED = ("tangential", "diagonal")
FAMILY_NAMES = tuple(FAMILIES)

#: Jaw roll about the approach, degrees. 0 = jaw-opening axis horizontal and
#: perpendicular to the lean (for top-down: tangential); that is the pose the
#: analytic planner emits (`obb_grasp.plan_grasps_from_fix`, tilted
#: candidates). The selector also tries each candidate rotated by 180 degrees
#: (its flip twin), so these four cover the full circle in 45 degree steps.
ROLLS_DEG = (0.0, 45.0, 90.0, 135.0)

#: Grasp TCP heights (m above the table) of the objects the shipped configs
#: describe, with the grasp-height rule that produces them (OBB planner:
#: top minus depth_fraction 0.15 of the height; rounded fruit at its widest
#: section).
Z_LEVELS = (0.02, 0.05, 0.07, 0.10)
Z_SOURCES = {
    "0.02": "kitchen lemon (4.3 cm ellipsoid, widest-section grasp) / low fruit",
    "0.05": "kitchen cubes 3.75x3.75x6 cm (0.06 - 0.15*0.06 = 0.051)",
    "0.07": "bare Isaac scene cubes 5x5x8 cm (0.08 - 0.15*0.08 = 0.068); kitchen tomato can 8.25 cm",
    "0.10": "12 cm blocks (demo.yaml pregrasp_offset_m comment: grasped near z~0.10)",
}

#: Grid over the table, base frame (m). Wide enough to contain every point
#: the arm can reach at table height (the outermost reachable sample sits
#: well inside it; see `summary.open`).
GRID_X = (0.00, 0.80, 0.025)
GRID_Y = (-0.80, 0.80, 0.025)

#: The opt-in envelope (`derive_envelope`): the families the opt-in profiles
#: let the analytic planner emit (top-down first, then these tilts), and the
#: rule a grid cell must satisfy to join the workspace box.
ENVELOPE_FAMILIES = ("topdown", "out30", "out45", "side")
ENVELOPE_ROLLS_DEG = (0.0,)


def _axis_values(spec):
    lo, hi, step = spec
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def approach_and_jaw(x: float, y: float, family: str, sign: int = 1, roll_deg: float = 0.0):
    """-> (approach, jaw_opening_axis) unit vectors for a family at (x, y).

    ``sign`` selects the lean side for the tangential families (+t or -t)."""
    spec = FAMILIES[family]
    r = np.array([x, y, 0.0], dtype=float)
    n = float(np.linalg.norm(r))
    r = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else r / n
    t = np.cross([0.0, 0.0, 1.0], r)
    down = np.array([0.0, 0.0, -1.0])
    theta = math.radians(spec["tilt_deg"])
    if spec["lean"] in ("none", "out"):
        lean = r
    elif spec["lean"] == "tangential":
        lean = float(sign) * t
    elif spec["lean"] == "diagonal":
        lean = (r + float(sign) * t) / np.sqrt(2.0)
    else:  # pragma: no cover - table above is closed
        raise ValueError(spec["lean"])
    approach = math.cos(theta) * down + math.sin(theta) * lean
    approach /= np.linalg.norm(approach)
    # Jaw axis at roll 0: horizontal and perpendicular to the lean.
    jaw0 = np.cross([0.0, 0.0, 1.0], lean)
    jaw0 /= np.linalg.norm(jaw0)
    jaw0 -= (jaw0 @ approach) * approach
    jaw0 /= np.linalg.norm(jaw0)
    psi = math.radians(roll_deg)
    jaw = math.cos(psi) * jaw0 + math.sin(psi) * np.cross(approach, jaw0)
    return approach, jaw / np.linalg.norm(jaw)


# ── runtime context (one per worker process) ─────────────────────────────────

_CTX: dict = {}


def _resolved_view(arm: str):
    """(arm profile, that arm's resolved config view) -- the view
    `apps.demo._build_arm` hands to the arm's SafetyHarness."""
    from cascade.config import Cfg, load_demo_config

    cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
    return Cfg(cfg.arms[0]), Cfg(cfg.arms[0]["resolved"])


def build_context(arm: str = "rebot_rs", workspace=None) -> dict:
    """Kinematics + harness exactly as `apps.demo._build_arm` builds them.

    ``workspace`` = (min, max) replaces the profile's box (None keeps it;
    ``"open"`` removes it for the box-free pass)."""
    from cascade.control.kinematics import Kinematics
    from cascade.safety.harness import SafetyHarness, SafetyLimits

    acfg, view = _resolved_view(arm)
    kin = Kinematics(
        model_path=acfg.model,
        ee_frame=acfg.get("ee_frame", "gripper_end"),
        n_controlled=int(acfg.get("n_joints", 6)),
        joint_signs=acfg.get("joint_signs"),
        ik_task_weights=acfg.get("ik_task_weights"),
    )
    limits = SafetyLimits.from_config(view.safety)
    limits.watchdog_s = float("inf")  # no perception in an offline study
    if workspace == "open":
        limits.workspace_min = np.full(3, -np.inf)
        limits.workspace_max = np.full(3, np.inf)
    elif workspace is not None:
        limits.workspace_min = np.asarray(workspace[0], dtype=float)
        limits.workspace_max = np.asarray(workspace[1], dtype=float)
    harness = SafetyHarness(limits, kinematics=kin)
    gcfg = view.grasp
    return {
        "arm": arm,
        "model": str(acfg.model),
        "kin": kin,
        "harness": harness,
        "home_q": np.asarray(acfg.home_q, dtype=float),
        "axis_order": str(acfg.get("tool_axis_order", "down_open")),
        "max_width_m": float((acfg.get("gripper") or {}).get("max_width_m", 0.09)),
        "pregrasp_offset_m": float(gcfg.get("pregrasp_offset_m", 0.12)),
        "exempt_radius_m": float(gcfg.get("exempt_radius_m", 0.07)),
        "move_duration_s": float(gcfg.get("move_duration_s", 2.5)),
        "pre_carry_lift": bool(gcfg.get("pre_carry_lift", False)),
        "carry_height_m": gcfg.get("carry_height_m"),
        "table_z": float(limits.table_z),
        "table_clearance": float(limits.table_clearance),
        "joint_margin": float(limits.joint_margin),
        "max_joint_vel": float(limits.max_joint_vel),
        "keep_out": [[a.tolist(), b.tolist()] for a, b in limits.keep_out],
        "workspace": [np.asarray(limits.workspace_min).tolist(),
                      np.asarray(limits.workspace_max).tolist()],
    }


def descent_samples(q_pre, q_grasp):
    """The joint samples the runtime vets on the descent (`_vet`)."""
    q_pre, q_grasp = np.asarray(q_pre, dtype=float), np.asarray(q_grasp, dtype=float)
    return [q_pre + s * (q_grasp - q_pre) for s in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)]


def runtime_vet(ctx: dict):
    """The harness half of `SkillRuntime.skill_grasp_object._vet`, in its order.

    Kept line-for-line with the runtime (tests/test_reach_envelope.py runs the
    real skill on the mock stack for the same points to pin the equivalence)."""
    from cascade.safety.trajectory import vet_segment

    harness, kin, seed = ctx["harness"], ctx["kin"], ctx["home_q"]
    z_min = ctx["table_z"] - 0.06

    def vet(g, q_pre, q_grasp):
        reason = harness.vet_pose(q_pre)
        if reason:
            return f"pregrasp unsafe: {reason}"
        reason = vet_segment(harness, seed, q_pre, ctx["move_duration_s"])
        if reason:
            return f"approach unsafe: {reason}"
        for q in descent_samples(q_pre, q_grasp):
            reason = harness.vet_pose(q, exempt_xy=g.position[:2],
                                      exempt_radius_m=ctx["exempt_radius_m"], exempt_z_min=z_min)
            if reason:
                return f"descent unsafe: {reason}"
        if ctx["pre_carry_lift"] and ctx["carry_height_m"] is not None:
            pose = kin.fk(q_pre).copy()
            height = float(ctx["carry_height_m"])
            if pose[2, 3] < height:
                pose[2, 3] = height
                lifted = kin.ik(pose, q_pre)
                if not lifted.success or np.max(np.abs(lifted.q - q_pre)) > np.pi:
                    return "grasp cannot reach the configured carry height"
                for fraction in (.15, .3, .45, .6, .75, .9, 1.):
                    reason = harness.vet_pose(q_pre + fraction * (lifted.q - q_pre))
                    if reason:
                        return f"grasp carry lift unsafe: {reason}"
        return None

    return vet


def candidate(ctx: dict, x: float, y: float, z: float, family: str, roll_deg: float, sign: int = 1):
    """The `Grasp` a planner would hand the selector for this family."""
    from cascade.grasping.obb_grasp import tool_rotation
    from cascade.types import Grasp

    approach, jaw = approach_and_jaw(x, y, family, sign, roll_deg)
    return Grasp(position=np.array([x, y, z], dtype=float),
                 rotation=tool_rotation(approach, jaw, ctx["axis_order"]),
                 width_m=0.05, approach=approach, quality=1.0, label=f"{family}/{roll_deg:g}")


def evaluate(ctx: dict, x: float, y: float, z: float, family: str, roll_deg: float):
    """-> (ok, reason_or_None, q_pre, q_grasp) through the real selector."""
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    signs = (1, -1) if FAMILIES[family]["lean"] in _TWO_SIDED else (1,)
    last = None
    for sign in signs:
        g = candidate(ctx, x, y, z, family, roll_deg, sign)
        try:
            _, q_pre, q_grasp = select_grasp(
                [g], ctx["kin"], ctx["home_q"], max_width_m=ctx["max_width_m"],
                pregrasp_offset_m=ctx["pregrasp_offset_m"], validate=runtime_vet(ctx),
                preserve_order=True)
            return True, None, q_pre, q_grasp
        except NoExecutableGrasp as exc:
            last = str(exc)
    return False, last, None, None


def min_link_z(ctx: dict, q_pre, q_grasp) -> float:
    """Lowest joint origin (the harness's link proxies) over the vetted
    descent. Below ``table_z + 0.01`` the pose relies on the grasp exemption
    cylinder (designed for top-down descents); reported, never used to admit."""
    kin = ctx["kin"]
    return float(min(kin.link_positions(q)[1:, 2].min() for q in descent_samples(q_pre, q_grasp)))


def _worker_init(arm, workspace):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CTX.update(build_context(arm, workspace))


def _point_task(args):
    """(x, y, z, families) -> (x, y, z, {family: [roll_mask, min_link_z_mm|None]})."""
    x, y, z, families = args
    out = {}
    for family in families:
        mask, low = 0, None
        for i, roll in enumerate(ROLLS_DEG):
            ok, _, q_pre, q_grasp = evaluate(_CTX, x, y, z, family, roll)
            if ok:
                mask |= 1 << i
                mz = round(1000.0 * min_link_z(_CTX, q_pre, q_grasp))
                low = mz if low is None else max(low, mz)
        out[family] = [mask, low]
    return (x, y, z, out)


def run_grid(arm, workspace, tasks, workers):
    if workers <= 1:
        _worker_init(arm, workspace)
        return [_point_task(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(arm, workspace)) as pool:
        return list(pool.map(_point_task, tasks, chunksize=2))


# ── analysis ─────────────────────────────────────────────────────────────────

def _key(x, y, z):
    return (round(float(x), 4), round(float(y), 4), round(float(z), 4))


def rows_to_table(rows) -> dict:
    """[(x, y, z, {family: [mask, low]})] -> {(x, y, z): {family: mask}}."""
    return {_key(x, y, z): {f: int(v[0]) for f, v in out.items()} for x, y, z, out in rows}


def reachable(table, families=FAMILY_NAMES, rolls=ROLLS_DEG) -> set:
    """Points (x, y, z) some family in ``families`` reaches with a roll in ``rolls``."""
    bits = sum(1 << ROLLS_DEG.index(r) for r in rolls)
    return {p for p, fam in table.items() if any(fam.get(f, 0) & bits for f in families)}


def _bbox(points):
    if not points:
        return None
    a = np.asarray(sorted(points), dtype=float)
    r = np.hypot(a[:, 0], a[:, 1])
    return {"x": [float(a[:, 0].min()), float(a[:, 0].max())],
            "y": [float(a[:, 1].min()), float(a[:, 1].max())],
            "r": [round(float(r.min()), 4), round(float(r.max()), 4)]}


def _in_box_xy(p, box):
    (x0, y0), (x1, y1) = box[0][:2], box[1][:2]
    return x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9


def summarize(table, default_box, z_levels=Z_LEVELS) -> dict:
    """Per z level and family: reachable count, bbox, and what the family
    adds outside the default box and beyond top-down."""
    out = {}
    for z in z_levels:
        at_z = {p: v for p, v in table.items() if abs(p[2] - z) < 1e-9}
        top = reachable(at_z, ("topdown",))
        per = {}
        for f in FAMILY_NAMES:
            pts = reachable(at_z, (f,))
            roll0 = reachable(at_z, (f,), (0.0,))
            outside = {p for p in pts if not _in_box_xy(p, default_box)}
            axis = [p[0] for p in pts if abs(p[1]) < 1e-9]
            per[f] = {
                "reachable": len(pts),
                "max_x_at_y0": max(axis) if axis else None,
                "reachable_roll0": len(roll0),
                "bbox": _bbox(pts),
                "beyond_topdown": len(pts - top),
                "outside_default_box": len(outside),
                "outside_default_box_bbox": _bbox(outside),
            }
        angled = reachable(at_z, tuple(f for f in FAMILY_NAMES if f != "topdown"))
        per["any_angled"] = {"reachable": len(angled), "beyond_topdown": len(angled - top),
                             "bbox": _bbox(angled)}
        out[f"{z:g}"] = per
    return out


def derive_envelope(table, default_box, xs, ys, families=ENVELOPE_FAMILIES,
                    rolls=ENVELOPE_ROLLS_DEG, z_levels=Z_LEVELS) -> dict:
    """The opt-in workspace box: the default box, grown ONLY over grid cells
    where the study measured a reachable approach.

    A cell (x, y) is *admitted* when some family in ``families`` (with a roll
    in ``rolls``: by default the horizontal-jaw pose the analytic planner
    emits) reaches it at EVERY grasp height in ``z_levels`` -- any object the
    configs describe can then be approached there. The box keeps the default
    near-base edge (x_min) and z range: closer to the base the arm folds onto
    itself and neither IK nor the harness models self-collision.

    The result is the largest AABB that contains the default box and whose
    every grid sample outside the default box is admitted (an inscribed box,
    not a bounding box: no unmeasured corner is admitted). Edges sit ON the
    outermost admitted samples (no extrapolation between samples). Ties
    prefer the y-symmetric box, then the larger x extent.
    """
    lo, hi = np.asarray(default_box[0], dtype=float), np.asarray(default_box[1], dtype=float)
    cells = {}
    for (x, y, z), fam in table.items():
        cells.setdefault((x, y), set())
    for z in z_levels:
        at_z = {p: v for p, v in table.items() if abs(p[2] - z) < 1e-9}
        ok = {(p[0], p[1]) for p in reachable(at_z, families, rolls)}
        for c in cells:
            if c not in ok:
                cells[c].add(z)
    admitted = {c for c, missing in cells.items() if not missing}
    xs = [x for x in xs if x >= lo[0] - 1e-9]
    x_hi = [x for x in xs if x >= hi[0] - 1e-9]
    y_hi = [y for y in ys if y >= hi[1] - 1e-9]
    y_lo = [y for y in ys if y <= lo[1] + 1e-9]

    def ok_box(x1, y0, y1):
        for x in xs:
            if x > x1 + 1e-9:
                break
            for y in ys:
                if y < y0 - 1e-9 or y > y1 + 1e-9:
                    continue
                if lo[0] - 1e-9 <= x <= hi[0] + 1e-9 and lo[1] - 1e-9 <= y <= hi[1] + 1e-9:
                    continue  # default box: unchanged, never re-litigated here
                if (x, y) not in admitted:
                    return False
        return True

    best = None
    for x1 in [hi[0], *x_hi]:
        for y1 in [hi[1], *y_hi]:
            for y0 in [lo[1], *y_lo]:
                if not ok_box(x1, y0, y1):
                    continue
                area = (x1 - lo[0]) * (y1 - y0)
                score = (round(area, 6), -round(abs(y1 + y0), 6), x1)
                if best is None or score > best[0]:
                    best = (score, x1, y0, y1)
    assert best is not None, "the default box itself always qualifies"
    _, x1, y0, y1 = best
    ws_min = [float(lo[0]), float(y0), float(lo[2])]
    ws_max = [float(x1), float(y1), float(hi[2])]
    return {
        "workspace_min": [round(v, 4) for v in ws_min],
        "workspace_max": [round(v, 4) for v in ws_max],
        "rule": {
            "families": list(families),
            "rolls_deg": list(rolls),
            "z_levels": list(z_levels),
            "cell": "reachable by some family/roll above at EVERY z level (pass 1, open box)",
            "box": "largest AABB containing the default box whose every grid sample outside "
                   "the default box is admitted; x_min and z range kept from the default box",
        },
        "admitted_cells": len(admitted),
        "admitted_cells_outside_default": sum(1 for c in admitted if not _in_box_xy(c, default_box)),
    }


def compare(table_a, table_b) -> dict:
    """Points reachable in a and not b (by any family), and vice versa."""
    a, b = reachable(table_a), reachable(table_b)
    return {"only_first": len(a - b), "only_second": len(b - a)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return "unknown"


def _rows_json(rows):
    """Compact rows: [x, y, z, mask_per_family..., min_link_z_mm_per_family...];
    points no family reaches are omitted."""
    out = []
    for x, y, z, fam in sorted(rows, key=lambda r: (r[2], r[0], r[1])):
        masks = [int(fam.get(f, [0, None])[0]) for f in FAMILY_NAMES]
        if not any(masks):
            continue
        lows = [fam.get(f, [0, None])[1] for f in FAMILY_NAMES]
        out.append([x, y, z, *masks, *lows])
    return out


def rows_from_json(data_rows):
    n = len(FAMILY_NAMES)
    rows = []
    for r in data_rows:
        x, y, z = r[:3]
        masks, lows = r[3:3 + n], r[3 + n:3 + 2 * n]
        rows.append((x, y, z, {f: [m, lo] for f, m, lo in zip(FAMILY_NAMES, masks, lows)}))
    return rows


def _recheck_tasks(rows):
    """Tasks for a restricted pass: only (point, family) pairs pass 1 reached."""
    tasks = []
    for x, y, z, fam in rows:
        fams = [f for f in FAMILY_NAMES if fam.get(f, [0])[0]]
        if fams:
            tasks.append((x, y, z, fams))
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", default="rebot_rs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--pass1", type=Path, help="reuse pass-1 rows from an earlier output")
    args = ap.parse_args(argv)
    xs, ys = _axis_values(GRID_X), _axis_values(GRID_Y)
    points = [(x, y, z) for z in Z_LEVELS for x in xs for y in ys]
    ctx = build_context(args.arm, None)
    default_box = ctx["workspace"]
    t0 = time.monotonic()
    timings = {}
    if args.pass1:
        rows1 = rows_from_json(json.loads(args.pass1.read_text())["passes"]["open"]["rows"])
        timings["open_s"] = None
    else:
        rows1 = run_grid(args.arm, "open", [(*p, list(FAMILY_NAMES)) for p in points], args.workers)
        timings["open_s"] = round(time.monotonic() - t0, 1)
        print(f"pass 1 (open box): {len(rows1)} points in {timings['open_s']}s", flush=True)
    table1 = rows_to_table(rows1)
    t1 = time.monotonic()
    rows2 = run_grid(args.arm, None, _recheck_tasks(rows1), args.workers)
    timings["default_s"] = round(time.monotonic() - t1, 1)
    print(f"pass 2 (default box): {timings['default_s']}s", flush=True)
    envelope = derive_envelope(table1, default_box, xs, ys)
    box = (envelope["workspace_min"], envelope["workspace_max"])
    t2 = time.monotonic()
    rows3 = run_grid(args.arm, box, _recheck_tasks(rows1), args.workers)
    timings["envelope_s"] = round(time.monotonic() - t2, 1)
    print(f"pass 3 (envelope box {box}): {timings['envelope_s']}s", flush=True)
    table2, table3 = rows_to_table(rows2), rows_to_table(rows3)
    fam_env = ENVELOPE_FAMILIES
    env_pts = reachable(table3, fam_env, ENVELOPE_ROLLS_DEG)
    def_pts = reachable(table2, ("topdown",), (0.0,))
    outside = sorted(p for p in reachable(table3) if not _in_box_xy(p, box))
    envelope["verified"] = {
        "envelope_pass_points_outside_box": len(outside),
        "topdown_roll0_default_box": len(def_pts),
        "envelope_families_roll0_envelope_box": len(env_pts),
        "gained": len(env_pts - def_pts),
        "lost": len(def_pts - env_pts),
        "gained_by_z": {f"{z:g}": len({p for p in env_pts - def_pts if abs(p[2] - z) < 1e-9})
                        for z in Z_LEVELS},
        "default_box_any_family_any_roll": len(reachable(table2)),
        "envelope_box_any_family_any_roll": len(reachable(table3)),
    }
    lows = [v[1] for _, _, _, fam in rows3 for f, v in fam.items()
            if f in fam_env and v[0] & 1 and v[1] is not None]
    envelope["verified"]["min_link_z_mm_envelope_families"] = min(lows) if lows else None
    result = {
        "schema": SCHEMA,
        "provenance": {
            "commit": _commit(),
            "script_sha256": _sha256(Path(__file__)),
            "model": os.path.relpath(ctx["model"], REPO),
            "model_sha256": _sha256(Path(ctx["model"])),
            "arm": args.arm,
            "home_q": ctx["home_q"].tolist(),
            "tool_axis_order": ctx["axis_order"],
            "pregrasp_offset_m": ctx["pregrasp_offset_m"],
            "exempt_radius_m": ctx["exempt_radius_m"],
            "move_duration_s": ctx["move_duration_s"],
            "table_z": ctx["table_z"],
            "table_clearance": ctx["table_clearance"],
            "joint_margin": ctx["joint_margin"],
            "ik_limit_margin": 0.025,
            "max_joint_vel": ctx["max_joint_vel"],
            "default_workspace": default_box,
            "keep_out": ctx["keep_out"],
            "timings_s": timings,
        },
        "grid": {"x": list(GRID_X), "y": list(GRID_Y), "z": list(Z_LEVELS), "z_sources": Z_SOURCES,
                 "rolls_deg": list(ROLLS_DEG), "families": FAMILIES,
                 "row_layout": ["x", "y", "z", *(f"mask:{f}" for f in FAMILY_NAMES),
                                *(f"min_link_z_mm:{f}" for f in FAMILY_NAMES)],
                 "mask_bits": {f"{r:g}": 1 << i for i, r in enumerate(ROLLS_DEG)}},
        "summary": {
            "open": summarize(table1, default_box),
            "default_box": summarize(table2, default_box),
            "envelope_box": summarize(table3, default_box),
        },
        "envelope": envelope,
        "passes": {
            "open": {"workspace": "open", "rows": _rows_json(rows1)},
            "default_box": {"workspace": default_box, "rows": _rows_json(rows2)},
            "envelope_box": {"workspace": [list(box[0]), list(box[1])], "rows": _rows_json(rows3)},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=None, separators=(",", ":")) + "\n")
    print(json.dumps(envelope, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
