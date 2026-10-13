# SPDX-License-Identifier: Apache-2.0
"""Render an HONEST (onboard-only) single-robot MicroDuck episode, walker or skater, to a 1080p review video.

Panels (English, burned in):
* "illustrative camera view": OVRTX follow camera of the RECORDED Newton physics (never an input to anything);
* "ROBOT'S OWN HEAD CAMERA": the exact 640x360 frames the robot's odometry and detector used (and the agent saw),
  with the boxes of the latest detect_objects call;
* "ROBOT'S OWN ToF 8x8": its own ToF frame;
* "OWN POSE ESTIMATE vs SIMULATOR TRUTH": the robot's estimate (start frame) and, for evaluation only, the truth;
* "<robot>'s OWN MAP": its ToF occupancy map built on its own estimate;
* the agent panel: task, active controller, the latest LLM decision and receipts.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import yaml

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade.sim.microduck_ovrtx_replay import ReplayPoses, build_render_scene, look_at  # noqa: E402
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxRenderer, SceneSnapshot  # noqa: E402
from render_episode import (BG, DIM, PANEL, SDK_051, TXT, YEL, Events, current_map, font, map_image,  # noqa: E402
                            tof_panel, wrap, yaw_of)

HOME = Path.home()
WALK_USD = HOME / "Projects/microducks/microduck_gtceu/assets/MicroDuck.usd"
SKATE_USD = HOME / "Projects/microducks/train/skillsft/assets/microduck_rollers.usd"
MODE_LABEL = {"stand": "standing (learned walker, zero command)", "walk": "learned walker velocity_flat_existing",
              "onboard_hero": "onboard perception actor (ToF + RGB)", "getup": "learned get-up policy",
              "coast": "coasting (learned skating policy)", "skate": "skating (learned skating policy)",
              "brake": "braking (learned skating policy)"}


def read_video(path, w=640, h=360):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def traj_panel(size, est, truth, k_est, color=(80, 190, 255)):
    """Top view of the robot's own estimate (blue) and the simulator truth (grey, evaluation only)."""
    w, h = size
    img = Image.new("RGB", size, (14, 15, 19))
    d = ImageDraw.Draw(img)
    pts = np.concatenate([est[:, 1:3], truth])
    lo, hi = pts.min(0) - 0.3, pts.max(0) + 0.3
    span = max(hi[0] - lo[0], (hi[1] - lo[1]) * w / h, 1.0)
    sc = (w - 10) / span

    def px(x, y):
        return 5 + (x - lo[0]) * sc, h - 5 - (y - lo[1]) * sc

    if len(truth) > 1:
        d.line([px(*p) for p in truth[: k_est + 1]], fill=(150, 150, 150), width=2)
    if k_est > 0:
        d.line([px(x, y) for x, y in est[: k_est + 1, 1:3]], fill=color, width=2)
    if k_est >= 0:
        u, v = px(*est[k_est, 1:3])
        d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=color)
        u2, v2 = px(*truth[k_est])
        d.ellipse([u2 - 4, v2 - 4, u2 + 4, v2 + 4], outline=(220, 220, 220), width=2)
        err = float(np.hypot(*(est[k_est, 1:3] - truth[k_est])))
        d.text((6, 4), f"error now {err:.2f} m", fill=TXT, font=font(12))
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--hold_s", type=float, default=0.8)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--llm_label", default="GPT-6-Astra via Codex CLI (Cascade codex_astra profile)")
    a = p.parse_args()
    run = a.run
    with np.load(run / "recording.npz") as z:
        rec = {k: z[k] for k in z.files}
    meta = json.loads((run / "recording.json").read_text())
    scen = yaml.safe_load((run / "scenario.yaml").read_text())
    episode = json.loads((run / "episode.json").read_text())
    events = Events(run / "events.jsonl")
    resolved = json.loads((run / "scenario_resolved.json").read_text())
    skate = meta.get("robot_kind") == "skates"
    robot_usd = SKATE_USD if skate else WALK_USD
    robots = [str(r) for r in rec["robot_names"]]
    r0 = robots[0]
    modes = [str(m) for m in rec["modes"]]
    times = rec["time_s"]
    body_pose = rec["body_pose"]
    T = len(times)
    with np.load(run / "onboard.npz") as z:
        onb = {k: z[k] for k in z.files}
    est = onb[f"est_{r0}"]  # (N, 6): t, x, y, yaw, tracking, features
    tofs = onb.get(f"tof_{r0}")
    cam = read_video(run / f"headcam_{r0}.mp4")
    n_cam = min(len(cam), len(est))
    # truth in the robot's start frame (evaluation only)
    k0 = int(np.searchsorted(times, est[0, 0]))
    s0 = body_pose[k0, 0, 0]
    yaw0 = yaw_of(s0[3:7])
    c, s_ = math.cos(yaw0), math.sin(yaw0)
    kk = np.clip(np.searchsorted(times, est[:n_cam, 0]), 0, T - 1)
    dx, dy = body_pose[kk, 0, 0, 0] - s0[0], body_pose[kk, 0, 0, 1] - s0[1]
    truth = np.stack([c * dx + s_ * dy, -s_ * dx + c * dy], 1)
    maps = {}
    with np.load(run / f"maps_{r0}.npz") as z:
        m = {k: z[k] for k in z.files}
    maps[r0] = {"t": m["t"], "lo": m["logodds"], "drop": m["drop"], "bounds": m["bounds"], "res": float(m["res"]),
                "visited": json.loads(str(m["visited"])), "landmarks": json.loads(str(m["landmarks"]))}
    rdir = run / "render"
    rdir.mkdir(exist_ok=True)
    layout = build_render_scene(rdir / "scene.usdc", course_usd=Path(resolved["scene_path"]), microduck_usd=robot_usd,
                                robots=robots, floor_size=24.0)
    replay = ReplayPoses(layout, [str(b) for b in rec["body_names"]], robots)
    W, H = 1920, 1080
    f = (720 / 2) / math.tan(math.radians(50) / 2)
    cams = [CameraSpec("main", 1280, 720, f, f, np.eye(4), near_m=0.02, far_m=60.0)]
    fol = scen.get("render", {}).get("follow", {"distance": 1.0, "height": 0.6})
    step = max(1, int(round(1.0 / (a.fps * float(times[1] - times[0])))))
    frames_idx = list(range(0, T, step))[: a.max_frames or None]
    decisions = [e for e in events.rows if e["kind"] == "llm_reply"]
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
                           str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt",
                           "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    yaw_s = None
    seq = dec_ptr = 0
    t0w = time.time()
    with OvrtxRenderer(rdir / "scene.usdc", cams, dynamic_paths=replay.paths, device=0, render_dt=1.0 / a.fps,
                       sdk_versions=SDK_051) as renderer:
        for n, k in enumerate(frames_idx):
            t = float(times[k])
            root = body_pose[k, 0, 0]
            yaw = yaw_of(root[3:7])
            yaw_s = yaw if yaw_s is None else yaw_s + 0.08 * ((yaw - yaw_s + math.pi) % (2 * math.pi) - math.pi)
            eye = root[:3] + np.array([-fol["distance"] * math.cos(yaw_s - 0.5), -fol["distance"] * math.sin(yaw_s - 0.5),
                                       fol["height"]])
            poses = {"main": look_at(eye, root[:3] + np.array([0, 0, 0.02]))}
            seq += 1
            fr = renderer.render(SceneSnapshot("cascade.microduck.honest_replay", "main", seq, t + 1e-3 * seq,
                                               time.monotonic(), replay.transforms(body_pose[k]), poses))
            main_v = Image.fromarray(np.ascontiguousarray(fr["main"].rgb[..., ::-1]))
            ke = int(np.searchsorted(est[:n_cam, 0], t + 1e-6, side="right")) - 1
            new = []
            while dec_ptr < len(decisions) and decisions[dec_ptr]["sim_time_s"] <= t + 1e-6:
                new.append(decisions[dec_ptr])
                dec_ptr += 1
            for dec, nhold in ([(dd, int(a.hold_s * a.fps)) for dd in new] or [(None, 1)]):
                img = compose(W, H, main_v, rec, k, t, r0, modes, maps, events, scen, a.llm_label, dec, cam, ke, est,
                              truth, tofs, skate)
                buf = np.asarray(img, np.uint8).tobytes()
                for _ in range(nhold):
                    ff.stdin.write(buf)
            if n % 100 == 0:
                print(f"frame {n}/{len(frames_idx)} t={t:.2f} wall={time.time() - t0w:.0f}s", flush=True)
        card = end_card(W, H, scen, episode, r0, a.llm_label, skate)
        buf = np.asarray(card, np.uint8).tobytes()
        for _ in range(5 * a.fps):
            ff.stdin.write(buf)
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    rep = {"frames_rendered": seq, "fps": a.fps, "wall_s": round(time.time() - t0w, 1), "sdk": SDK_051,
           "robot_usd_sha256": hashlib.sha256(robot_usd.read_bytes()).hexdigest(), "pose_repair": False}
    (run / "render_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("RENDER DONE", json.dumps(rep), flush=True)


def last_detection(events, t, r):
    rows = [e for e in events.upto(t, r, ("tool_result",)) if e["text"] == "detect_objects"]
    if not rows or t - rows[-1]["sim_time_s"] > 3.0:
        return None
    return (rows[-1].get("result") or {}).get("detections") or []


def compose(W, H, main_v, rec, k, t, r, modes, maps, events, scen, llm_label, thinking, cam, ke, est, truth, tofs, skate):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 58], fill=(12, 13, 17))
    d.text((16, 6), scen.get("title", scen["name"]) + "  -  ONBOARD-ONLY RUN", fill=TXT, font=font(25, bold=True))
    d.text((16, 38), f"Cascade agent ({llm_label}). The agent and every controller used only this robot's own camera, "
                     f"ToF, IMU and joint encoders. Isaac Lab 3.0 + Newton physics.", fill=DIM, font=font(14))
    d.text((W - 270, 14), f"sim time {t:6.2f} s", fill=YEL, font=font(24, bold=True, mono=True))
    img.paste(main_v, (0, 60))
    d.rectangle([0, 60 + 720 - 26, 820, 60 + 720], fill=(0, 0, 0))
    d.text((8, 60 + 720 - 23), "illustrative camera view - OVRTX render of the recorded physics (not a robot input)",
           fill=TXT, font=font(15))
    # own head camera
    x0 = 1280
    if ke >= 0:
        hc = Image.fromarray(cam[min(ke, len(cam) - 1)])
        hd = ImageDraw.Draw(hc)
        for det in last_detection(events, t, r) or []:
            b = det.get("bbox_px")
            if b:
                hd.rectangle(b, outline=(255, 220, 0), width=2)
                hd.text((b[0] + 3, max(0, b[1] - 14)), f"{det['label'][:24]} {det.get('distance_m')} m",
                        fill=(255, 220, 0), font=font(12, bold=True))
        img.paste(hc, (x0, 60))
    d.rectangle([x0, 60 + 360 - 22, x0 + 640, 60 + 360], fill=(0, 0, 0))
    d.text((x0 + 6, 60 + 360 - 20), "ROBOT'S OWN HEAD CAMERA 640x360 15 Hz (agent, odometry and detector input)",
           fill=YEL, font=font(13, bold=True))
    # ToF + estimate panel
    y = 430
    d.text((x0 + 6, y), "ROBOT'S OWN ToF 8x8", fill=YEL, font=font(13, bold=True))
    if tofs is not None and ke >= 0 and ke < len(tofs):
        img.paste(tof_panel(tofs[ke][:64], tofs[ke][64:], 200), (x0 + 6, y + 20))
    elif not skate:
        img.paste(tof_panel(rec["tof_mm"][k, 0], rec["tof_status"][k, 0], 200), (x0 + 6, y + 20))
    d.text((x0 + 220, y), "OWN POSE ESTIMATE (blue) vs SIMULATOR TRUTH", fill=YEL, font=font(13, bold=True))
    d.text((x0 + 220, y + 16), "(grey, evaluation only - never shown to the robot)", fill=DIM, font=font(11))
    img.paste(traj_panel((410, 186), est, truth, ke), (x0 + 222, y + 34))
    if ke >= 0:
        loc = "tracking" if est[ke, 4] > 0.5 else "VO LOST"
        d.text((x0 + 222, y + 224), f"camera odometry: {loc}, {int(est[ke, 5])} features", fill=DIM, font=font(11))
    y2 = y + 244
    d.text((x0 + 6, y2), f"{r}'s OWN MAP (its ToF frames placed with its own pose estimate)", fill=YEL,
           font=font(13, bold=True))
    snap, lms, visited = current_map(maps, r, t)
    if snap is not None and ke >= 0:
        mi = map_image(snap, maps[r]["bounds"], maps[r]["res"], (est[ke, 1], est[ke, 2], est[ke, 3]),
                       (630, 1080 - y2 - 26), lms, visited, (255, 170, 40))
        img.paste(mi, (x0 + 4, y2 + 20))
    agent_panel(img, rec, k, t, r, modes, events, thinking, skate)
    return img


def summarize_result(e):
    rr = e.get("result") or {}
    if "legs" in rr:
        return f"{e['text']}: {len(rr['legs'])} legs, last {str(rr['legs'][-1].get('stop_reason'))[:60]}"
    if "stop_reason" in rr:
        return f"{e['text']}: {str(rr['stop_reason'])[:80]}, moved {rr.get('traveled_m')} m (own estimate)"
    if e["text"] == "detect_objects":
        return f"detect_objects: {len(rr.get('detections') or [])} detections"
    return f"{e['text']}: " + ("ok" if rr.get("ok") else str(rr.get("error", ""))[:90])


def agent_panel(img, rec, k, t, r, modes, events, thinking, skate):
    d = ImageDraw.Draw(img)
    x0, y0, x1 = 0, 784, 1280
    d.rectangle([x0, y0, x1, 1080], fill=PANEL)
    mode = modes[int(rec["mode"][k, 0])]
    c = rec["command"][k, 0]
    extra = (f"   throttle {c[0]:+.2f}, heading error {math.degrees(c[2]):+.0f} deg" if skate else
             (f"   command vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f}" if mode == "walk" else ""))
    d.rectangle([x0 + 10, y0 + 8, x1 - 10, y0 + 36], fill=(50, 80, 50) if mode not in ("stand", "coast") else (60, 60, 70))
    d.text((x0 + 18, y0 + 12), "ACTIVE CONTROLLER: " + MODE_LABEL.get(mode, mode) + extra, fill=TXT, font=font(16, bold=True))
    task = next((e["text"] for e in events.rows if e["robot"] == r and e["kind"] == "task"), "")
    tl = wrap("TASK: " + task, 150)[:2]
    for i, ln in enumerate(tl):
        d.text((x0 + 12, y0 + 42 + i * 15), ln, fill=DIM, font=font(13))
    if thinking is not None:
        call = (thinking.get("tool_calls") or [{}])[0]
        d.text((x0 + 12, y0 + 76), f"AGENT DECIDES (LLM {thinking.get('latency_s', 0):.1f} s, simulation paused): "
                                   f"{call.get('name')}({json.dumps(call.get('args'))[:80]})", fill=YEL, font=font(17, bold=True))
        lines = wrap(thinking.get("text", ""), 150)[:5]
    else:
        last = events.upto(t, r, ("llm_reply",))
        lines = wrap(last[-1]["text"], 150)[:5] if last else []
        lt = [e for e in events.upto(t, r, ("tool_call", "tool_result"))]
        if lt:
            e = lt[-1]
            d.text((x0 + 12, y0 + 76), ("EXECUTING: " if e["kind"] == "tool_call" else "LAST: ")
                   + f"{e['text']}({json.dumps(e.get('args', {}))[:80]})", fill=TXT, font=font(17, bold=True))
    for i, ln in enumerate(lines):
        d.text((x0 + 12, y0 + 102 + i * 18), ln, fill=TXT, font=font(14))
    yy = y0 + 102 + 5 * 18 + 10
    for e in events.upto(t, r, ("tool_result",))[-3:]:
        d.text((x0 + 12, yy), f"t={e['sim_time_s']:.1f}s {summarize_result(e)}"[:165], fill=DIM, font=font(13, mono=True))
        yy += 17


def end_card(W, H, scen, episode, r, llm_label, skate):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 60), scen.get("title", scen["name"]) + " - onboard-only run - outcome", fill=TXT, font=font(32, bold=True))
    v = episode.get("verification", {}).get(r, {})
    rep = episode.get("reports", {}).get(r, {})
    ok = v.get("passed")
    y = 130
    d.text((80, y), f"simulator check (offline): {'PASSED' if ok else 'FAILED'}"
                    + (f" - {v.get('failed_because')}" if v.get("failed_because") else ""),
           fill=(120, 230, 120) if ok else (240, 120, 110), font=font(26, bold=True))
    y += 44
    lines = []
    if v.get("type") == "reach":
        lines.append(f"final distance to the goal centre {v.get('distance_m')} m (limit {v.get('radius_m')} m)")
    if v.get("type") == "slalom":
        lines.append("cones (closest approach, need <= 0.45 m): " + ", ".join(
            f"{c['cone']} {c['closest_m']} m {'ok' if c['passed'] else 'MISSED'}" for c in v.get("cones", [])))
        lines.append(f"stop mark: {v.get('stop_distance_m')} m from centre (need <= {v.get('stop_radius_m')}), "
                     f"{'stopped' if v.get('stopped') else 'not stopped'}")
    if v.get("fell") or v.get("fallen"):
        lines.append("the robot FELL" + (" (no get-up on skates: run over)" if skate else ""))
    cl = (episode.get("clearance") or {}).get(r)
    if cl and cl.get("min_body_origin_to_obstacle_box_m") is not None:
        lines.append(f"equipment clearance (approx.): {cl['min_body_origin_to_obstacle_box_m']:.3f} m to "
                     f"{cl.get('closest_obstacle')}; possible contact: {'YES' if cl.get('possible_contact') else 'no'}")
    le = (episode.get("localization_eval") or {}).get(r)
    if le:
        lines.append(f"own localization vs truth (offline): final error {le['position_error_final_m']} m, max "
                     f"{le['position_error_max_m']} m, RMS {le['position_error_rms_m']} m over {le['true_path_m']} m of path")
    claim = rep.get("success_claimed")
    lines += wrap(f"agent's own claim: {'success' if claim else ('failure' if claim is False else rep.get('error', 'none'))}"
                  f" - {rep.get('summary', '')}", 165)[:3]
    for ln in lines:
        d.text((100, y), ln, fill=TXT, font=font(18))
        y += 30
    llm = episode.get("llm", {})
    d.text((80, y + 14), f"LLM: {llm_label}; {sum(x['calls'] for x in llm.values())} calls, "
                         f"{sum(x['wall_s'] for x in llm.values()):.0f} s of thinking (simulation paused meanwhile). "
                         f"Simulated time {episode.get('sim_time_s', 0):.1f} s.", fill=TXT, font=font(17))
    notes = ["ONBOARD-ONLY: the agent and every controller used only the robot's own sensors: OVRTX head camera 640x360 "
             "15 Hz, VL53L8CX ToF model, IMU model, joint encoders.",
             "Localization = the robot's own estimate: gyro heading + " + ("" if skate else "leg odometry (encoders) + ")
             + "cuVSLAM visual odometry; it drifts. Simulator truth only in this offline check.",
             "Detector: YOLOE (open vocabulary) on the head image. Learned policies move the robot; scripted parts "
             "(waypoint/steering loop, guards) run on the estimate.",
             ("Skate-wheel MicroDuck (canonical robot plus passive wheels). No get-up policy on skates." if skate else
              "Falls are handled by the learned get-up policy."),
             "Courses carry set dressing (tape marks, posters; no colliders) so the floor has texture like a real one."]
    for i, n in enumerate(notes):
        d.text((80, y + 64 + i * 28), n, fill=DIM, font=font(16))
    return img


if __name__ == "__main__":
    main()
