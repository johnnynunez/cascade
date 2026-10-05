#!/usr/bin/env python
"""Build report.json (aggregates + pointers to the raw per-step logs) from analysis.json."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(HERE, "analysis.json")))
table = a["table"]


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": st.mean(vals), "min": min(vals), "max": max(vals)}


summary = {}
for policy in ("velocity_flat", "velocity_rough"):
    for direction in ("fwd", "rev"):
        rows = [r for r in table if r["policy"] == policy and r["kind"] in (f"{direction}_cl", f"{direction}_ol")]
        rows_1s = [r for r in table if r["policy"] == policy and r["kind"] == f"{direction}_1s"]
        summary[f"{policy}/{direction}/pm30mm"] = {
            "episodes": len(rows),
            "during_along_mm": agg(rows, "during_along_mm"),
            "during_lateral_mm": agg(rows, "during_lateral_mm"),
            "post_zero_along_mm": agg(rows, "post_zero_along_mm"),
            "post_zero_lateral_mm": agg(rows, "post_zero_lateral_mm"),
            "post_zero_path_mm": agg(rows, "post_zero_path_mm"),
            "post_zero_yaw_change_deg": agg(rows, "post_zero_yaw_change_deg"),
            "post_zero_first_still_s": agg(rows, "post_zero_first_still_s"),
            "post_zero_last1s_disp_mm": agg(rows, "post_zero_last1s_disp_mm"),
            "max_tilt_deg": agg(rows, "post_zero_max_tilt_deg"),
            "all_ever_still": all(r.get("post_zero_ever_still") for r in rows),
            "any_fell": any(r.get("fell") for r in rows),
        }
        if rows_1s:
            summary[f"{policy}/{direction}/1s_steady_gait"] = {
                "episodes": len(rows_1s),
                "during_along_mm": agg(rows_1s, "during_along_mm"),
                "post_zero_along_mm": agg(rows_1s, "post_zero_along_mm"),
                "post_zero_first_still_s": agg(rows_1s, "post_zero_first_still_s"),
                "any_fell": any(r.get("fell") for r in rows_1s),
            }
    stand = [r for r in table if r["policy"] == policy and r["kind"] == "stand"]
    summary[f"{policy}/stand_10s"] = {
        "episodes": len(stand),
        "drift_mm": agg([{"d": 1000 * r["stand_drift_m"]} for r in stand], "d"),
        "yaw_change_deg": agg(stand, "stand_yaw_change_deg"),
        "max_planar_speed": agg(stand, "stand_max_planar_speed"),
        "any_fell": any(r.get("fell") for r in stand),
    }

report = {
    "title": "MicroDuck +-30 mm walk-then-zero-twist reproduction directly in Isaac Lab (Newton MJWarp + BAM)",
    "date": "2026-10-05",
    "fork": "AntoineRichard/IsaacLab antoiner/feat/microduck-rough-velocity @ eafc80dfac8c45dc4392e84432d837b47ab54cc4 (contains upstream develop 4aa39c103)",
    "runtime": {
        "mode": "kit-less Isaac Lab (uv environment of the fork), Newton 1.6.1rc1 (newton-physics/newton@release-1.6 dff2963) with MJWarp (mujoco-warp 3.12.0, mujoco 3.12.0), warp 1.17.0, torch 2.12.0+cu130, onnxruntime 1.30.0 CPU",
        "gpu": "CUDA_VISIBLE_DEVICES=GPU-4c811d02-a79b-2837-425e-5f62ac3c578a (physical GPU 1)",
        "isaac_sim_used": "none (Isaac Lab's Newton backend runs without Kit; neither Isaac Sim build was modified)",
    },
    "policies": {
        "velocity_flat": {"onnx_sha256": "6c3d02661819fdf9abe5129d75f7fb47bf3b9c73605bcca74ca410ee9d6c7779", "iteration": 49999},
        "velocity_rough": {"onnx_sha256": "efdc851c50e52a550581fb93cb7d0987af10ba65e06e0447c90e4a620475cb9f", "iteration": 49999},
    },
    "protocol": {
        "policy_rate_hz": 50, "physics_dt_s": 0.005, "decimation": 4,
        "settle_s": 1.0, "command_speed_m_s": 0.3, "closed_loop_target_m": 0.025, "open_loop_steps": 15,
        "steady_gait_steps": 50, "zero_hold_s": 3.0, "standing_test_s": 10.0,
        "still_criterion": "|v_planar(root link)| < 0.02 m/s for 0.2 s (10 consecutive policy steps)",
        "heading": "root link +x axis at the end of the settle phase; displacement decomposed along/lateral to it",
    },
    "variants": {
        "lab_play": "task cfg + play_mode() exactly as `isaaclab play` applies it (noise off, pushes off, startup/reset randomization ON: encoder bias +-0.015 rad, IMU misalignment <=6 deg, trunk mass 0.95-1.05, CoM +-3 mm, BAM friction scale 0.9-1.1, armature 0.9-1.1, foot friction 0.7-1.3, reset yaw uniform; BAM deployment vin 6.5-8.2 V sampled, sag gain 0-0.2 V/Nm, delay 3-6 physics steps, max current 1.75 A, IMU obs lag 0-1 step, joint-vel obs lag 1 step)",
        "norand": "all randomization ranges zeroed (identity IMU, no encoder bias, nominal mass/CoM/friction/armature, reset pose = identity), IMU obs lag fixed 0, joint-vel lag 1 step; BAM deployment still sampled from Lab's ranges at startup (vin 7.178 V, sag 0.194 V/Nm for seed 0), delay 3-6, max current 1.75 A",
        "cascade_nominal": "norand + BAM vin 7.4 V fixed, sag gain 0.1 V/Nm, delay 0, max current disabled (0) -> max_effort 0.9634 N.m, i.e. CASCADE's `official_infer_nominal_no_delay` profile",
    },
    "cascade_reference_numbers": {
        "source": "cascade origin/main docs/evidence/microduck-isaaclab-usd-20261005/locomotion/summary.json + the task brief",
        "velocity_flat_forward": {"during_along_mm": 25.6, "during_lateral_mm": -5.8, "after_along_mm": 25.8, "after_lateral_mm": -8.8, "after_yaw_rad": 0.131},
        "velocity_flat_reverse": {"during_along_mm": -27.6, "during_lateral_mm": -10.1, "after_along_mm": -48.5, "after_lateral_mm": -15.6, "after_yaw_rad": -0.228},
        "velocity_rough_forward_(task_brief)": {"during_along_mm": 25.9, "after_along_mm": 26.9, "after_yaw_rad": 0.006},
        "velocity_rough_reverse_(task_brief)": {"during_along_mm": -26.7, "after_along_mm": -55.4},
    },
    "summary": summary,
    "per_episode_table": table,
    "runs": a["runs"],
    "raw_per_step_logs": sorted(os.path.relpath(f, HERE) for f in glob.glob(os.path.join(HERE, "results", "*.json")) if "probe" not in f),
    "raw_log_record_fields": ["i", "phase", "t", "cmd", "obs(61)", "action(14)", "terminated", "truncated", "pos", "com_pos", "quat_xyzw", "yaw", "tilt", "lin_vel_w", "ang_vel_w", "lin_vel_b", "ang_vel_b", "joint_pos(14, policy order)", "joint_vel(14, policy order)"],
    "note_on_old_file": "results/OLD_wrongquat_flat_velocity_flat_lab_play_seed0.json is the first run; its stored key 'quat_wxyz' actually holds the xyzw quaternion (Isaac Lab 3.0 convention) and its in-run yaw/tilt fields are wrong; analyze.py recomputes everything from the raw quaternion, and the run was repeated as flat_velocity_flat_lab_play_seed0.json.",
}
json.dump(report, open(os.path.join(HERE, "report.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
