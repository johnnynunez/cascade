#!/usr/bin/env python3
"""Measured effect of the B73 gripper clearance vet on the B45 reach grid.

B45 (`scripts/reachability_study.py`, docs/REACH_ENVELOPE.md) measured where
the analytic planner's tilted approaches (30 / 45 / 90 degrees leaning away
from the base, horizontal jaw -- families `out30`, `out45`, `side`, roll 0)
pass IK and the harness vet. B73 adds the clearance vet
(`cascade.grasping.gripper_clearance`): the forearm, wrist, housing and open
fingers' collision hulls along the approach must clear the support plane and
the target's observed box by `width_pad_m / 2`. This script re-runs every
point of the B45 envelope pass (the `*_reach` box) those families reached,
through the REAL selector with the runtime's vet order (harness first, then
the clearance check -- `combined_vet`), with a synthetic upright object box
per grasp height: the objects B45's grasp heights stand for (its `z_sources`),
centred under the TCP and squared to the jaw, standing on the table.

Per point it records whether the candidate (or its flip twin) survives, and
the best orientation's lowest clearance to the plane and to the box with the
part that binds -- so the margin's effect can be read off (`sensitivity`).

Run (CPU only; ~15 s with 32 workers, a few minutes on one):

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python scripts/angled_clearance_study.py \\
        --out docs/evidence/angled-clearance-20261010/rebot_angled_clearance.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import reachability_study as rs  # noqa: E402

SCHEMA = "cascade.angled_clearance_study/1"
ARM = "rebot_rs_reach"
B45_EVIDENCE = REPO / "docs" / "evidence" / "reach-envelope-20261009" / "rebot_reachability.json"
#: the planner's tilts (`grasp.angled_approach_tilts_deg: [30, 45, 90]`), roll 0
FAMILIES = {"out30": 30.0, "out45": 45.0, "side": 90.0}
#: what each B45 grasp height stands for (its `z_sources`), as an upright box:
#: [along the jaw, across it, height] in metres, standing on the table
OBJECTS = {
    "0.02": {"size": [0.043, 0.043, 0.043], "what": "kitchen lemon, 4.3 cm (enclosing box)"},
    "0.05": {"size": [0.0375, 0.0375, 0.06], "what": "kitchen cube 3.75 x 3.75 x 6 cm"},
    "0.07": {"size": [0.05, 0.05, 0.08], "what": "bare-scene cube 5 x 5 x 8 cm"},
    "0.1": {"size": [0.05, 0.05, 0.12], "what": "12 cm block (5 x 5 cm footprint assumed)"},
}
#: margins (mm) for the sensitivity table
SENSITIVITY_MM = (0.0, 5.0, 7.5, 10.0, 15.0, 20.0)
_FINGERS = {"gripper_left": "fingers", "gripper_right": "fingers"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(arm: str = ARM) -> dict:
    """The B45 study context (kinematics + harness exactly as the runtime
    builds them, the profile's own box) plus the clearance vet."""
    from cascade.grasping import gripper_clearance as gc

    ctx = rs.build_context(arm, None)
    acfg, view = rs._resolved_view(arm)
    ctx["vet"] = gc.ApproachClearance(gc.load_geometry(ctx["model"]), ctx["kin"])
    ctx["margin_m"] = gc.margin_m(view.grasp)
    ctx["open_margin_m"] = (acfg.get("gripper") or {}).get("pregrasp_open_margin_m")
    return ctx


def combined_vet(ctx: dict, box, records: list | None = None):
    """The runtime's `_vet` for an angled candidate (`AngledGrasp`, all the
    study evaluates): the harness part (B45's `runtime_vet`), then the
    clearance check. `records` collects every clearance result (one per
    orientation that passed the harness)."""
    from cascade.grasping import gripper_clearance as gc

    harness = rs.runtime_vet(ctx)

    def vet(g, q_pre, q_grasp):
        reason = harness(g, q_pre, q_grasp)
        if reason:
            return reason
        res = ctx["vet"].check(q_pre, q_grasp, support_z=ctx["table_z"], box=box,
                               margin_m=ctx["margin_m"],
                               jaw_gap_m=gc.jaw_gap_m(g.width_m, ctx["max_width_m"], ctx["open_margin_m"]))
        if records is not None:
            records.append(res)
        return None if res.ok else f"gripper clearance: {res.reason}"

    return vet


def object_box(x: float, y: float, z: float, jaw):
    from cascade.grasping.gripper_clearance import Box

    size = np.asarray(OBJECTS[f"{z:g}"]["size"], dtype=float)
    u = np.array([jaw[0], jaw[1], 0.0])
    u /= np.linalg.norm(u)
    return Box(centre=np.array([x, y, size[2] / 2.0]),
               axes=np.column_stack([u, np.cross([0.0, 0.0, 1.0], u), [0.0, 0.0, 1.0]]),
               half=size / 2.0)


def evaluate(ctx: dict, x: float, y: float, z: float, family: str) -> dict:
    """One B45 point through the selector + `combined_vet` -> result row."""
    from cascade.grasping.obb_grasp import AngledGrasp, tool_rotation
    from cascade.grasping.selector import NoExecutableGrasp, select_grasp

    approach, jaw = rs.approach_and_jaw(x, y, family, 1, 0.0)
    g = AngledGrasp(position=np.array([x, y, z], dtype=float),
                    rotation=tool_rotation(approach, jaw, ctx["axis_order"]),
                    width_m=0.05, approach=approach, quality=1.0, label=family)
    records: list = []
    try:
        select_grasp([g], ctx["kin"], ctx["home_q"], max_width_m=ctx["max_width_m"],
                     pregrasp_offset_m=ctx["pregrasp_offset_m"],
                     validate=combined_vet(ctx, object_box(x, y, z, jaw), records), preserve_order=True)
        survived = True
    except NoExecutableGrasp:
        survived = False
    row = {"x": x, "y": y, "z": z, "family": family, "harness_ok": bool(records),
           "survived": survived, "plane_mm": None, "box_mm": None, "limit": None, "part": None}
    if records:
        best = max(records, key=lambda r: min(r.plane_clearance_m, r.box_clearance_m))
        plane = best.plane_clearance_m <= best.box_clearance_m
        row.update(plane_mm=round(best.plane_clearance_m * 1000.0, 2),
                   box_mm=round(best.box_clearance_m * 1000.0, 2),
                   limit="plane" if plane else "box")
        part = best.plane_part if plane else best.box_part
        row["part"] = _FINGERS.get(part, part)
    return row


#: compact JSON row layout (one list per point)
ROW_LAYOUT = ("x", "y", "z", "family", "harness_ok", "survived", "plane_mm", "box_mm", "limit", "part")


def rows_to_json(rows):
    return [[r[k] for k in ROW_LAYOUT] for r in rows]


def rows_from_json(rows):
    return [dict(zip(ROW_LAYOUT, r)) for r in rows]


def summarize(rows, margin_m: float) -> dict:
    """Counts per tilt and grasp height, the binding parts of the refusals,
    and how the survivors would change with the margin."""
    out = {"families": {}, "sensitivity_mm": {}}
    margin_mm = margin_m * 1000.0
    for family in FAMILIES:
        fam = [r for r in rows if r["family"] == family]
        per_z = {}
        for z in sorted({r["z"] for r in fam}):
            at = [r for r in fam if r["z"] == z]
            refused = [r for r in at if not r["survived"]]
            lows = [min(r["plane_mm"], r["box_mm"]) for r in at if r["harness_ok"]]
            per_z[f"{z:g}"] = {
                "reachable": len(at), "survived": len(at) - len(refused),
                "refused_plane": sum(r["limit"] == "plane" for r in refused),
                "refused_box": sum(r["limit"] == "box" for r in refused),
                "refused_parts": dict(sorted(Counter(r["part"] for r in refused).items())),
                "lowest_clearance_mm": round(min(lows), 1) if lows else None,
                "lowest_clearance_part": (min((r for r in at if r["harness_ok"]),
                                              key=lambda r: min(r["plane_mm"], r["box_mm"]))["part"]
                                          if lows else None),
                "refused_r_m": ([round(min(np.hypot(r["x"], r["y"]) for r in refused), 3),
                                 round(max(np.hypot(r["x"], r["y"]) for r in refused), 3)]
                                if refused else None),
            }
        out["families"][family] = {
            "tilt_deg": FAMILIES[family], "by_z": per_z,
            "reachable": len(fam), "survived": sum(r["survived"] for r in fam),
        }
    for m in SENSITIVITY_MM:
        out["sensitivity_mm"][f"{m:g}"] = {
            f: sum(1 for r in rows if r["family"] == f and r["harness_ok"]
                   and min(r["plane_mm"], r["box_mm"]) >= m - 1e-9)
            for f in FAMILIES}
    out["margin_mm"] = round(margin_mm, 3)
    out["reachable"] = len(rows)
    out["survived"] = sum(r["survived"] for r in rows)
    out["non_clearance_refusals"] = sum(1 for r in rows if not r["harness_ok"])
    # The survivors must be exactly the rows clearing the margin (the vet's
    # own decision, read back from the recorded clearances).
    out["decision_consistent"] = all(
        r["survived"] == (r["harness_ok"] and min(r["plane_mm"], r["box_mm"]) >= margin_mm - 0.01)
        for r in rows)
    return out


def envelope_check(rows, b45: dict) -> dict:
    """Does the B45 envelope rule still admit every cell of the opt-in box
    with the vet on? A cell is admitted when some planner family (top-down,
    which this vet never touches, or a SURVIVING tilt) reaches it at every
    grasp height."""
    table = rs.rows_to_table(rs.rows_from_json(b45["passes"]["envelope_box"]["rows"]))
    survived = {(r["x"], r["y"], r["z"], r["family"]) for r in rows if r["survived"]}
    before, after = {}, {}
    for (x, y, z), fam in table.items():
        top = bool(fam.get("topdown", 0) & 1)
        if top or any(fam.get(f, 0) & 1 for f in FAMILIES):
            before.setdefault((x, y), set()).add(z)
        if top or any((x, y, z, f) in survived for f in FAMILIES):
            after.setdefault((x, y), set()).add(z)
    levels = set(rs.Z_LEVELS)
    admitted_before = {c for c, zs in before.items() if zs >= levels}
    admitted_after = {c for c, zs in after.items() if zs >= levels}
    return {"admitted_cells_before": len(admitted_before), "admitted_cells_after": len(admitted_after),
            "cells_losing_admission": sorted([list(c) for c in admitted_before - admitted_after])}


def doc_table(summary: dict) -> list[str]:
    """The REACH_ENVELOPE.md table rows (pinned by the tests)."""
    lines = ["| tilt | z 0.02 | z 0.05 | z 0.07 | z 0.10 | lowest clearance (part) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for family, data in summary["families"].items():
        cells, lows = [], []
        for z, label in (("0.02", "0.02"), ("0.05", "0.05"), ("0.07", "0.07"), ("0.1", "0.10")):
            d = data["by_z"].get(z)
            if d is None:
                cells.append("not reachable")
                continue
            cells.append(f"{d['survived']} / {d['reachable']}")
            lows.append((d["lowest_clearance_mm"], d["lowest_clearance_part"], label))
        low = min(lows)
        lines.append(f"| {data['tilt_deg']:g}° (`{family}`) | " + " | ".join(cells)
                     + f" | {low[0]:+.1f} mm at z {low[2]} ({low[1]}) |")
    return lines


def roadmap_sentence(summary: dict) -> str:
    fam = summary["families"]
    side5 = fam["side"]["by_z"]["0.05"]
    lean = [fam[f]["survived"] == fam[f]["reachable"] for f in ("out30", "out45")]
    return (f"of {summary['reachable']} reachable tilted grasps on the B45 grid "
            f"{summary['survived']} survive: the vet refuses {side5['reachable'] - side5['survived']} of "
            f"{side5['reachable']} side (90°) grasps at 5 cm (wrist motor `link5` "
            f"{side5['lowest_clearance_mm']:+.1f} mm from the table) and "
            + ("no 30°/45° grasp" if all(lean) else "some 30°/45° grasps")
            + " (their lowest point is the fingertips)")


_CTX: dict = {}


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CTX.update(context(ARM))


def _task(args):
    return evaluate(_CTX, *args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args(argv)
    from cascade.grasping import gripper_clearance as gc

    b45 = json.loads(B45_EVIDENCE.read_text())
    tasks = sorted((x, y, z, f) for x, y, z, fam in rs.rows_from_json(b45["passes"]["envelope_box"]["rows"])
                   for f in FAMILIES if fam[f][0] & 1)
    ctx = context(ARM)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        rows = list(pool.map(_task, tasks, chunksize=8))
    elapsed = round(time.monotonic() - t0, 1)
    summary = summarize(rows, ctx["margin_m"])
    result = {
        "schema": SCHEMA,
        "provenance": {
            "commit": rs._commit(), "script_sha256": _sha256(Path(__file__)),
            "arm": ARM, "model": os.path.relpath(ctx["model"], REPO), "model_sha256": _sha256(Path(ctx["model"])),
            "hulls": os.path.relpath(gc.GEOMETRY, REPO), "hulls_sha256": _sha256(gc.GEOMETRY),
            "b45_evidence": os.path.relpath(B45_EVIDENCE, REPO), "b45_evidence_sha256": _sha256(B45_EVIDENCE),
            "margin_m": ctx["margin_m"], "approach_step_m": gc.APPROACH_STEP_M,
            "jaw_gap_m": gc.jaw_gap_m(0.05, ctx["max_width_m"], ctx["open_margin_m"]),
            "table_z": ctx["table_z"], "pregrasp_offset_m": ctx["pregrasp_offset_m"],
            "families": FAMILIES, "objects": OBJECTS, "elapsed_s": elapsed, "workers": args.workers,
        },
        "summary": summary,
        "envelope": envelope_check(rows, b45),
        "row_layout": list(ROW_LAYOUT),
        "rows": rows_to_json(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=None, separators=(",", ":")) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
