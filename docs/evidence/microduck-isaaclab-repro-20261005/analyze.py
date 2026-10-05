#!/usr/bin/env python
"""Aggregate the per-step logs of lab_microduck_walk_test.py into a per-case table (Markdown + JSON).

All metrics are recomputed here from the raw per-step logs (root position, quaternion xyzw, velocities), so they do
not depend on the in-run bookkeeping. Heading = root-link +x axis at the end of the settle phase.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STILL_SPEED = 0.02
STILL_STEPS = 10
DT = 0.02


def yaw_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_xyzw(q):
    x, y, z, w = q
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def planar_speed(r):
    return math.hypot(r["lin_vel_w"][0], r["lin_vel_w"][1])


def still_stats(records):
    """Return (ever_still, first_still_time_s, fraction_of_steps_still) for |v_planar|<0.02 held 0.2 s."""
    n = 0
    first = None
    still_steps = 0
    for k, r in enumerate(records):
        if planar_speed(r) < STILL_SPEED:
            n += 1
            still_steps += 1
        else:
            n = 0
        if n >= STILL_STEPS and first is None:
            first = (k + 1) * DT
    return first is not None, first, still_steps / max(1, len(records))


def analyze_episode(ep: dict) -> dict:
    log = ep["log"]
    phases = [r["phase"] for r in log]
    settle = [r for r in log if r["phase"] == "settle"]
    cmd = [r for r in log if r["phase"] == "command"]
    zero = [r for r in log if r["phase"] == "zero"]
    stand = [r for r in log if r["phase"] == "stand"]
    qkey = "quat_xyzw" if "quat_xyzw" in log[0] else "quat_wxyz"  # old runs stored xyzw under the wrong name
    out = {"kind": ep["kind"], "seed": ep["seed"], "trial": ep.get("trial"), "fell": ep.get("fell"),
           "reset_during_episode": ep.get("reset_during_episode"), "steps": len(log)}
    if not settle:
        out["note"] = "no settle data"
        return out
    s_end = settle[-1]
    p0 = np.array(s_end["pos"][:2])
    yaw0 = yaw_xyzw(s_end[qkey])
    h = np.array([math.cos(yaw0), math.sin(yaw0)])
    lat = np.array([-math.sin(yaw0), math.cos(yaw0)])
    out["settle_tilt_deg"] = math.degrees(tilt_xyzw(s_end[qkey]))
    out["settle_height_m"] = s_end["pos"][2]
    out["settle_planar_speed"] = planar_speed(s_end)
    if stand:
        p1 = np.array(stand[-1]["pos"][:2])
        ever, first, frac = still_stats(stand)
        out.update(
            {
                "stand_duration_s": len(stand) * DT,
                "stand_drift_m": float(np.linalg.norm(p1 - p0)),
                "stand_drift_along_m": float((p1 - p0) @ h),
                "stand_drift_lateral_m": float((p1 - p0) @ lat),
                "stand_yaw_change_deg": math.degrees(wrap(yaw_xyzw(stand[-1][qkey]) - yaw0)),
                "stand_max_planar_speed": max(planar_speed(r) for r in stand),
                "stand_max_tilt_deg": math.degrees(max(tilt_xyzw(r[qkey]) for r in stand)),
                "stand_still_fraction": frac,
                "stand_ever_still": ever,
            }
        )
        return out
    if not cmd:
        return out
    c_end = cmd[-1]
    pc = np.array(c_end["pos"][:2])
    out.update(
        {
            "command": c_end["cmd"],
            "command_steps": len(cmd),
            "command_duration_s": len(cmd) * DT,
            "command_reached": ep.get("command_reached"),
            "during_along_mm": 1000 * float((pc - p0) @ h),
            "during_lateral_mm": 1000 * float((pc - p0) @ lat),
            "during_yaw_change_deg": math.degrees(wrap(yaw_xyzw(c_end[qkey]) - yaw0)),
            "during_max_planar_speed": max(planar_speed(r) for r in cmd),
            "during_mean_vb_x": float(np.mean([r["lin_vel_b"][0] for r in cmd])),
            "during_mean_vb_y": float(np.mean([r["lin_vel_b"][1] for r in cmd])),
            "during_max_tilt_deg": math.degrees(max(tilt_xyzw(r[qkey]) for r in cmd)),
        }
    )
    if not zero:
        out["note"] = ep.get("note", "no zero phase")
        return out
    z_end = zero[-1]
    pz = np.array(z_end["pos"][:2])
    ever, first, frac = still_stats(zero)
    # path length while the command is zero
    prev = c_end
    path = 0.0
    for r in zero:
        path += math.hypot(r["pos"][0] - prev["pos"][0], r["pos"][1] - prev["pos"][1])
        prev = r
    last1s = zero[-50:]
    out.update(
        {
            "zero_duration_s": len(zero) * DT,
            "post_zero_along_mm": 1000 * float((pz - pc) @ h),
            "post_zero_lateral_mm": 1000 * float((pz - pc) @ lat),
            "post_zero_total_mm": 1000 * float(np.linalg.norm(pz - pc)),
            "post_zero_path_mm": 1000 * path,
            "post_zero_yaw_change_deg": math.degrees(wrap(yaw_xyzw(z_end[qkey]) - yaw_xyzw(c_end[qkey]))),
            "post_zero_max_planar_speed": max(planar_speed(r) for r in zero),
            "post_zero_ever_still": ever,
            "post_zero_first_still_s": first,
            "post_zero_still_fraction": frac,
            "post_zero_last1s_disp_mm": 1000 * float(np.linalg.norm(pz - np.array(last1s[0]["pos"][:2]))),
            "post_zero_last1s_mean_speed": float(np.mean([planar_speed(r) for r in last1s])),
            "post_zero_max_tilt_deg": math.degrees(max(tilt_xyzw(r[qkey]) for r in zero)),
            "total_along_mm": 1000 * float((pz - p0) @ h),
            "total_lateral_mm": 1000 * float((pz - p0) @ lat),
            "total_yaw_change_deg": math.degrees(wrap(yaw_xyzw(z_end[qkey]) - yaw0)),
        }
    )
    return out


def main():
    files = sorted(f for f in glob.glob(os.path.join(RES_DIR, "*.json")) if "probe" not in f and "OLD_" not in f)
    table = []
    runs = {}
    for f in files:
        d = json.load(open(f))
        tag = os.path.basename(f)[:-5]
        cfg = d["cfg"]
        runs[tag] = {
            "task": cfg["task"], "variant": cfg["variant"], "seed": cfg["seed"], "policy": d["policy"]["stem"],
            "terrain": cfg["terrain_type"], "sub_terrains": cfg.get("terrain_sub_terrains"),
            "solver_iterations": cfg["solver"]["iterations"], "bam": cfg["bam_actuator_cfg"],
            "bam_readback": d.get("bam_readback_after_first_reset"), "obs_patch_max_abs_diff": d.get("obs_patch_max_abs_diff"),
            "episodes": [],
        }
        for ep in d["episodes"]:
            a = analyze_episode(ep)
            a.update({"run": tag, "policy": d["policy"]["stem"], "task": args_short(cfg["task"]), "variant": cfg["variant"],
                      "run_seed": cfg["seed"]})
            table.append(a)
            runs[tag]["episodes"].append(a)
    json.dump({"runs": runs, "table": table}, open(os.path.join(RES_DIR, "..", "analysis.json"), "w"), indent=1)
    print(markdown(table))


def args_short(task):
    return "flat-task(plane)" if "Flat" in task else "rough-task(plane)"


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:.{nd}f}"


def markdown(table):
    lines = []
    lines.append("### Walk-then-zero episodes (heading = root +x at the end of the 1 s settle)")
    lines.append("| policy | env | variant | seed | case | cmd steps | reached | during along [mm] | during lat [mm] | during yaw [deg] | post-zero along [mm] | post-zero lat [mm] | post-zero path [mm] | post-zero yaw [deg] | still (0.2 s) at [s] | last-1 s disp [mm] | max tilt [deg] | fell |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in table:
        if a["kind"] == "stand" or "during_along_mm" not in a:
            continue
        still = fmt(a.get("post_zero_first_still_s"), 2) if a.get("post_zero_ever_still") else "never"
        lines.append(
            f"| {a['policy']} | {a['task']} | {a['variant']} | {a['run_seed']} | {a['kind']} | {a['command_steps']} | {fmt(a.get('command_reached'))} | "
            f"{fmt(a['during_along_mm'])} | {fmt(a['during_lateral_mm'])} | {fmt(a['during_yaw_change_deg'])} | "
            f"{fmt(a.get('post_zero_along_mm'))} | {fmt(a.get('post_zero_lateral_mm'))} | {fmt(a.get('post_zero_path_mm'))} | {fmt(a.get('post_zero_yaw_change_deg'))} | "
            f"{still} | {fmt(a.get('post_zero_last1s_disp_mm'), 2)} | {fmt(max(a.get('during_max_tilt_deg', 0), a.get('post_zero_max_tilt_deg', 0)))} | {fmt(a['fell'])} |"
        )
    lines.append("")
    lines.append("### 10 s zero-command standing")
    lines.append("| policy | env | variant | seed | drift [mm] | along [mm] | lateral [mm] | yaw change [deg] | max planar speed [m/s] | still fraction | max tilt [deg] | fell |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in table:
        if a["kind"] != "stand" or "stand_drift_m" not in a:
            continue
        lines.append(
            f"| {a['policy']} | {a['task']} | {a['variant']} | {a['run_seed']} | {fmt(1000*a['stand_drift_m'], 2)} | {fmt(1000*a['stand_drift_along_m'], 2)} | "
            f"{fmt(1000*a['stand_drift_lateral_m'], 2)} | {fmt(a['stand_yaw_change_deg'], 2)} | {fmt(a['stand_max_planar_speed'], 3)} | {fmt(a['stand_still_fraction'], 3)} | {fmt(a['stand_max_tilt_deg'])} | {fmt(a['fell'])} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
