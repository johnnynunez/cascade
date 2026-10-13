# SPDX-License-Identifier: Apache-2.0
"""Render an agentic SKATING MicroDuck episode to a 1080p review video: OVRTX views of the recorded physics + overlays.

Panels and their labels (all English, burned in):
* "illustrative camera view": OVRTX renders (follow / overview cameras) of the RECORDED Newton poses on the skate-wheel
  MicroDuck (microduck_rollers.usd, sha256 pinned); never an input to anything;
* "head-height view": an OVRTX camera at each robot's head (illustrative; the skating policy is blind and the agent
  does not see it);
* "POLICY OBSERVATION (exact actor inputs)": the recorded 61-D actor input of that control step: the twist command
  [throttle, 0, heading error] plus proprioception (angular velocity, gravity, joint positions/velocities, last action);
* "<robot>'s OWN memory": the agent's landmarks (from its detector stand-in / notes) and its trail;
* the agent panel: task, active skate mode, the latest LLM decision and the receipts.
Simulation time is paused while an agent thinks; the video shows each decision as a short labelled hold.
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

from cascade.sim.microduck_ovrtx_replay import (  # noqa: E402
    MICRODUCK_ROLLERS_USD_SHA256, ReplayPoses, build_render_scene, look_at)
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxRenderer, SceneSnapshot  # noqa: E402
from cascade.skills.microduck_skate_runtime import Oval  # noqa: E402
from render_episode import (BG, DIM, PANEL, ROBOT_COLORS, SDK_051, TXT, YEL, Events, font,  # noqa: E402
                            map_image, wrap, yaw_of)

MODE_LABEL = {"coast": "coasting (learned skating policy, throttle 0)",
              "skate": "skating (learned skating policy, throttle > 0)",
              "brake": "braking (learned skating policy, throttle < 0)"}
OBS_GROUPS = [("ang vel", 0, 3), ("gravity", 3, 6), ("joint pos", 6, 20), ("joint vel", 20, 34),
              ("last action", 34, 48), ("COMMAND", 48, 51)]


def cam(name, w, h, vfov_deg):
    f = (h / 2) / math.tan(math.radians(vfov_deg) / 2)
    return CameraSpec(name, w, h, f, f, np.eye(4), near_m=0.02, far_m=60.0)


def head_pose(body_pose_r, head_idx, yaw):
    p = body_pose_r[head_idx, :3]
    eye = p + np.array([0.10 * math.cos(yaw), 0.10 * math.sin(yaw), 0.03])  # just in front of the head shell
    tgt = eye + np.array([math.cos(yaw), math.sin(yaw), -0.18])
    return look_at(eye, tgt)


def obs_panel(obs, size):
    """61-D actor input as labelled heat rows + the command values."""
    w, h = size
    img = Image.new("RGB", (w, h), (12, 13, 16))
    d = ImageDraw.Draw(img)
    row = (h - 30) // len(OBS_GROUPS)
    for gi, (lab, a, b) in enumerate(OBS_GROUPS):
        y = gi * row
        d.text((4, y + 2), lab, fill=YEL if lab == "COMMAND" else DIM, font=font(11, bold=lab == "COMMAND"))
        n = b - a
        cw = (w - 92) / n
        for j in range(n):
            v = float(obs[a + j])
            m = max(-1.0, min(1.0, v / (1.0 if lab != "joint vel" else 5.0)))
            col = (int(128 + 127 * m), int(128 - 60 * abs(m)), int(128 - 127 * m))
            d.rectangle([88 + j * cw, y + 2, 88 + (j + 1) * cw - 1, y + row - 3], fill=col)
    c = obs[48:51]
    d.text((4, h - 26), f"command: throttle {c[0]:+.2f}  (unused) {c[1]:+.2f}  heading error {math.degrees(c[2]):+.0f} deg",
           fill=TXT, font=font(12, mono=True))
    return img


def summarize_result(e):
    rr = e.get("result") or {}
    name = e["text"]
    if "legs" in rr:
        return (f"{name}: {len(rr['legs'])} legs, last {rr['legs'][-1]['stop_reason']}, closest "
                f"{[lg['closest_approach_m'] for lg in rr['legs']]}")
    if "stop_reason" in rr:
        s = f"{name}: {rr['stop_reason'][:70]}, moved {rr.get('traveled_m')} m"
        if "closest_approach_m" in rr:
            s += f", closest {rr['closest_approach_m']} m"
        sp = ((rr.get("after") or {}).get("pose") or {}).get("speed_m_s")
        return s + (f", now {sp} m/s" if sp is not None else "")
    return f"{name}: " + ("ok" if rr.get("ok") else str(rr.get("error", ""))[:90])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--robot_usd", type=Path,
                   default=Path.home() / "Projects/microducks/train/skillsft/assets/microduck_rollers.usd")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hold_s", type=float, default=0.8)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--overview_eye", type=float, nargs=3, default=None, help="override the scenario's overview camera")
    p.add_argument("--overview_target", type=float, nargs=3, default=None)
    p.add_argument("--llm_label", default="GPT-6-Astra via Codex CLI (Cascade codex_astra profile)")
    a = p.parse_args()
    run = a.run
    with np.load(run / "recording.npz") as z:
        rec = {k: z[k] for k in z.files}
    meta = json.loads((run / "recording.json").read_text())
    scen = yaml.safe_load((run / "scenario.yaml").read_text())
    episode = json.loads((run / "episode.json").read_text())
    events = Events(run / "events.jsonl")
    if hashlib.sha256(a.robot_usd.read_bytes()).hexdigest() != MICRODUCK_ROLLERS_USD_SHA256:
        raise SystemExit("robot USD is not the pinned skate-wheel variant")
    robots = [str(r) for r in rec["robot_names"]]
    modes = [str(m) for m in rec["modes"]]
    body_names = [str(b) for b in rec["body_names"]]
    head_idx = body_names.index("yaw_roll_motion")
    times = rec["time_s"]
    body_pose = rec["body_pose"]
    T = len(times)
    multi = len(robots) > 1
    resolved = json.loads((run / "scenario_resolved.json").read_text())
    course_path = Path(resolved["scene_path"])
    oval = Oval(scen["oval"]["half"], scen["oval"]["lanes"]) if scen.get("oval") else None
    rdir = run / "render"
    rdir.mkdir(exist_ok=True)
    layout = build_render_scene(rdir / "scene.usdc", course_usd=course_path, microduck_usd=a.robot_usd, robots=robots,
                                floor_size=16.0)
    replay = ReplayPoses(layout, body_names, robots)
    maps = {}
    for r in robots:
        with np.load(run / f"maps_{r}.npz") as z:
            m = {k: z[k] for k in z.files}
        maps[r] = {"t": m["t"], "lo": m["logodds"], "drop": m["drop"], "bounds": m["bounds"], "res": float(m["res"]),
                   "visited": json.loads(str(m["visited"])), "landmarks": json.loads(str(m["landmarks"]))}
    W, H = 1920, 1080
    cams = [cam("main", 1280, 720, 50)]
    if not multi:
        cams.append(cam("aux", 640, 360, 55))
        cams.append(cam("pov_0", 320, 240, 70))
    else:
        for ri in range(len(robots)):
            cams.append(cam(f"pov_{ri}", 240, 180, 70))
    ov = dict(scen.get("render", {}).get("overview", {"eye": [-1, -2, 2], "target": [1, 0, 0]}))
    if a.overview_eye:
        ov["eye"] = a.overview_eye
    if a.overview_target:
        ov["target"] = a.overview_target
    fol = scen.get("render", {}).get("follow", {"distance": 1.0, "height": 0.6})
    T_over = look_at(ov["eye"], ov["target"])
    step = max(1, int(round(1.0 / (a.fps * float(times[1] - times[0])))))
    frames_idx = list(range(0, T, step))
    if a.max_frames:
        frames_idx = frames_idx[: a.max_frames]
    decisions = [e for e in events.rows if e["kind"] == "llm_reply"]
    # lap progress from the recorded trajectory (same track coordinate as the runtime)
    progress = None
    if oval is not None:
        progress = np.zeros((T, len(robots)))
        for ri in range(len(robots)):
            last, acc = None, 0.0
            for k in range(T):
                f = oval.fraction(float(body_pose[k, ri, 0, 0]), float(body_pose[k, ri, 0, 1]))
                if last is not None:
                    acc += (f - last + 2.0) % 4.0 - 2.0
                last = f
                progress[k, ri] = acc / 4.0
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    yaw_s = [None] * len(robots)
    t0 = time.time()
    seq = rendered = dec_ptr = 0
    with OvrtxRenderer(rdir / "scene.usdc", cams, dynamic_paths=replay.paths, device=a.device,
                       render_dt=1.0 / a.fps, sdk_versions=SDK_051) as renderer:
        for n, k in enumerate(frames_idx):
            t = float(times[k])
            for ri in range(len(robots)):
                yaw = yaw_of(body_pose[k, ri, 0, 3:7])
                ys = yaw_s[ri]
                yaw_s[ri] = yaw if ys is None else ys + 0.08 * ((yaw - ys + math.pi) % (2 * math.pi) - math.pi)
            if multi:
                cam_poses = {"main": T_over}
            else:
                root = body_pose[k, 0, 0]
                eye = root[:3] + np.array([-fol["distance"] * math.cos(yaw_s[0] - 0.5),
                                           -fol["distance"] * math.sin(yaw_s[0] - 0.5), fol["height"]])
                cam_poses = {"main": look_at(eye, root[:3] + np.array([0, 0, 0.02])), "aux": T_over}
            for ri in range(len(robots)):
                cam_poses[f"pov_{ri}"] = head_pose(body_pose[k, ri], head_idx, yaw_s[ri])
            seq += 1
            snap = SceneSnapshot("cascade.microduck_agentic.skate_replay", meta.get("scenario", "episode"), seq,
                                 t + 1e-3 * seq, time.monotonic(), replay.transforms(body_pose[k]), cam_poses)
            frames = renderer.render(snap)
            rendered += 1
            views = {name: Image.fromarray(np.ascontiguousarray(f.rgb[..., ::-1])) for name, f in frames.items()}
            new = []
            while dec_ptr < len(decisions) and decisions[dec_ptr]["sim_time_s"] <= t + 1e-6:
                new.append(decisions[dec_ptr])
                dec_ptr += 1
            for dec, nhold in ([(d, int(a.hold_s * a.fps)) for d in new] or [(None, 1)]):
                img = compose(W, H, views, rec, k, t, robots, modes, maps, events, scen, multi, a.llm_label, dec,
                              cams, cam_poses, progress)
                buf = np.asarray(img, dtype=np.uint8).tobytes()
                for _ in range(nhold):
                    ffmpeg.stdin.write(buf)
            if n % 100 == 0:
                print(f"frame {n}/{len(frames_idx)} t={t:.2f} wall={time.time() - t0:.0f}s", flush=True)
        card = end_card(W, H, scen, episode, robots, a.llm_label)
        buf = np.asarray(card, dtype=np.uint8).tobytes()
        for _ in range(4 * a.fps):
            ffmpeg.stdin.write(buf)
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise SystemExit("ffmpeg failed")
    report = {"frames_rendered": rendered, "fps": a.fps, "sample_step": step, "wall_s": round(time.time() - t0, 1),
              "renderer": "cascade.sim.ovrtx_renderer.OvrtxRenderer (RealTimePathTracing)", "sdk": SDK_051,
              "robot_usd_sha256": layout["microduck_usd_sha256"], "robot_variant": layout.get("robot_variant"),
              "decision_holds": len(decisions), "physics_steps": 0, "pose_repair": False}
    (run / "render_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("RENDER DONE", json.dumps(report), flush=True)


def project(c, T_cam, pt):
    q = np.linalg.inv(T_cam) @ np.array([pt[0], pt[1], pt[2], 1.0])
    if q[2] <= 0.05:
        return None
    return c.fx * q[0] / q[2] + (c.width - 1) / 2, c.fy * q[1] / q[2] + (c.height - 1) / 2


def current_map(maps, r, t):
    m = maps[r]
    j = int(np.searchsorted(m["t"], t, side="right")) - 1
    if j < 0:
        return None, [], []
    return (m["lo"][j], m["drop"][j]), m["landmarks"][j], m["visited"][j]


def speed_at(rec, k, ri):
    k0 = max(0, k - 5)
    p = rec["body_pose"]
    dt = float(rec["time_s"][k] - rec["time_s"][k0]) or 1.0
    return float(np.linalg.norm(p[k, ri, 0, :2] - p[k0, ri, 0, :2]) / dt)


def compose(W, H, views, rec, k, t, robots, modes, maps, events, scen, multi, llm_label, thinking, cams, cam_poses,
            progress):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 58], fill=(12, 13, 17))
    d.text((16, 8), scen.get("title", scen["name"]), fill=TXT, font=font(26, bold=True))
    d.text((16, 38), f"Cascade agent ({llm_label}) chooses tools; unchanged learned SKATING policies move the legs "
                     f"(skate-wheel MicroDuck). Isaac Lab 3.0 + Newton physics, OVRTX render.", fill=DIM, font=font(14))
    d.text((W - 270, 14), f"sim time {t:6.2f} s", fill=YEL, font=font(24, bold=True, mono=True))
    img.paste(views["main"], (0, 60))
    if multi:
        c = cams[0]
        for ri, r in enumerate(robots):
            uv = project(c, cam_poses["main"], rec["body_pose"][k, ri, 0, :3] + np.array([0, 0, 0.30]))
            if uv:
                u, v = uv
                d.rectangle([u - 34, 60 + v - 22, u + 34, 60 + v - 2], fill=ROBOT_COLORS[ri % 4])
                d.text((u - 30, 60 + v - 21), r, fill=(0, 0, 0), font=font(15, bold=True))
    tag = ("illustrative camera view - OVRTX render of the recorded physics ("
           + ("overview" if multi else "follow camera") + ")")
    d.rectangle([0, 60 + 720 - 26, 760, 60 + 720], fill=(0, 0, 0))
    d.text((8, 60 + 720 - 23), tag, fill=TXT, font=font(15))
    if multi:
        right_multi(img, views, rec, k, t, robots, modes, maps, progress)
        agent_panel_multi(img, k, t, robots, events, thinking)
    else:
        img.paste(views["aux"], (1280, 60))
        d.rectangle([1280, 60 + 360 - 22, 1280 + 330, 60 + 360], fill=(0, 0, 0))
        d.text((1286, 60 + 360 - 20), "illustrative camera view - overview", fill=TXT, font=font(14))
        right_single(img, views, rec, k, t, robots[0], maps)
        agent_panel_single(img, rec, k, t, robots[0], modes, events, thinking)
    return img


def right_single(img, views, rec, k, t, r, maps):
    d = ImageDraw.Draw(img)
    x, y = 1284, 426
    img.paste(views["pov_0"], (x, y + 20))
    d.text((x, y), "head-height view (OVRTX, illustrative)", fill=TXT, font=font(13, bold=True))
    d.text((x, y + 262), "the skating policy is blind; the agent", fill=DIM, font=font(11))
    d.text((x, y + 276), "does not see this image", fill=DIM, font=font(11))
    d.text((x + 330, y), "POLICY OBSERVATION", fill=YEL, font=font(14, bold=True))
    d.text((x + 330, y + 17), "(exact actor inputs, 61-D)", fill=YEL, font=font(12))
    img.paste(obs_panel(rec["actor_obs"][k, 0].astype(np.float32), (300, 230)), (x + 330, y + 36))
    y2 = y + 296
    d.text((x, y2), f"{r}'s OWN memory: landmarks + trail (pose from simulator)", fill=YEL, font=font(14, bold=True))
    snap, lms, visited = current_map(maps, r, t)
    if snap is not None:
        root = rec["body_pose"][k, 0, 0]
        mi = map_image(snap, maps[r]["bounds"], maps[r]["res"], (root[0], root[1], yaw_of(root[3:7])),
                       (630, 1080 - y2 - 26), lms, visited, ROBOT_COLORS[0])
        img.paste(mi, (x, y2 + 22))


def right_multi(img, views, rec, k, t, robots, modes, maps, progress):
    d = ImageDraw.Draw(img)
    row_h = 1020 // len(robots)
    for ri, r in enumerate(robots):
        y = 60 + ri * row_h
        col = ROBOT_COLORS[ri % 4]
        d.rectangle([1280, y, 1920, y + row_h - 4], fill=PANEL)
        mode = modes[int(rec["mode"][k, ri])]
        status = f"{MODE_LABEL[mode].split(' (')[0]}, {speed_at(rec, k, ri):.2f} m/s"
        if progress is not None:
            status += f", lap {max(progress[k, ri], 0.0):.2f}"
        d.text((1288, y + 4), r, fill=col, font=font(18, bold=True))
        d.text((1376, y + 8), status, fill=TXT, font=font(13))
        img.paste(views[f"pov_{ri}"], (1288, y + 30))
        d.text((1288, y + 212), "head-height view (illustrative)", fill=DIM, font=font(11))
        img.paste(obs_panel(rec["actor_obs"][k, ri].astype(np.float32), (360, 170)), (1536, y + 30))
        d.text((1536, y + 204), "POLICY OBSERVATION (exact actor inputs)", fill=YEL, font=font(11, bold=True))
        snap, lms, visited = current_map(maps, r, t)
        if snap is not None and row_h > 300:
            root = rec["body_pose"][k, ri, 0]
            mi = map_image(snap, maps[r]["bounds"], maps[r]["res"], (root[0], root[1], yaw_of(root[3:7])),
                           (600, row_h - 250), lms, visited, col)
            img.paste(mi, (1288, y + 228))
            d.text((1288 + mi.width + 8, y + 232), f"{r}'s OWN", fill=YEL, font=font(11, bold=True))
            d.text((1288 + mi.width + 8, y + 246), "memory", fill=YEL, font=font(11, bold=True))


def last_tool(events, t, r):
    calls = events.upto(t, r, ("tool_call", "tool_result"))
    return calls[-1] if calls else None


def agent_panel_single(img, rec, k, t, r, modes, events, thinking):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 0, 784, 1280, 1080
    d.rectangle([x0, y0, x1, y1], fill=PANEL)
    mode = modes[int(rec["mode"][k, 0])]
    c = rec["command"][k, 0]
    badge = MODE_LABEL[mode] + f"   throttle {c[0]:+.2f}, heading error {math.degrees(c[2]):+.0f} deg, " \
                               f"speed {speed_at(rec, k, 0):.2f} m/s"
    d.rectangle([x0 + 10, y0 + 8, x1 - 10, y0 + 36], fill=(50, 80, 50) if mode != "coast" else (60, 60, 70))
    d.text((x0 + 18, y0 + 12), "SKATE CONTROL: " + badge, fill=TXT, font=font(16, bold=True))
    task = next((e["text"] for e in events.rows if e["robot"] == r and e["kind"] == "task"), "")
    tl = wrap("TASK: " + task, 150)
    if len(tl) > 2:
        tl = tl[:2]
        tl[1] = tl[1][:146].rstrip() + " ..."
    for i, ln in enumerate(tl):
        d.text((x0 + 12, y0 + 42 + i * 15), ln, fill=DIM, font=font(13))
    if thinking is not None:
        call = (thinking.get("tool_calls") or [{}])[0]
        d.text((x0 + 12, y0 + 76), f"AGENT DECIDES (LLM {thinking.get('latency_s', 0):.1f} s, simulation paused): "
                                   f"{call.get('name')}({json.dumps(call.get('args'))[:80]})", fill=YEL,
               font=font(17, bold=True))
        lines = wrap(thinking.get("text", ""), 150)[:5]
    else:
        last = events.upto(t, r, ("llm_reply",))
        lines = wrap(last[-1]["text"], 150)[:5] if last else []
        lt = last_tool(events, t, r)
        if lt is not None:
            lab = f"{lt['text']}({json.dumps(lt.get('args', {}))[:80]})"
            d.text((x0 + 12, y0 + 76), ("EXECUTING: " if lt["kind"] == "tool_call" else "LAST: ") + lab, fill=TXT,
                   font=font(17, bold=True))
    for i, ln in enumerate(lines):
        d.text((x0 + 12, y0 + 102 + i * 18), ln, fill=TXT, font=font(14))
    yy = y0 + 102 + 5 * 18 + 10
    for e in events.upto(t, r, ("tool_result",))[-3:]:
        d.text((x0 + 12, yy), f"t={e['sim_time_s']:.1f}s {summarize_result(e)}"[:165], fill=DIM, font=font(13, mono=True))
        yy += 17


def agent_panel_multi(img, k, t, robots, events, thinking):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 0, 784, 1280, 1080
    d.rectangle([x0, y0, x1, y1], fill=PANEL)
    d.text((x0 + 12, y0 + 6), "AGENTS (one Cascade agent per robot, each with its own memory; shared physics world)",
           fill=YEL, font=font(15, bold=True))
    yy = y0 + 30
    for ri, r in enumerate(robots):
        col = ROBOT_COLORS[ri % 4]
        last = events.upto(t, r, ("llm_reply",))
        lt = last_tool(events, t, r)
        hot = thinking is not None and thinking["robot"] == r
        head = f"{r}: "
        if hot:
            tc = (thinking.get("tool_calls") or [{}])[0]
            head += f"DECIDES -> {tc.get('name')}({json.dumps(tc.get('args'))[:70]})  [LLM {thinking.get('latency_s', 0):.1f} s]"
        elif lt is not None:
            head += ("executing " if lt["kind"] == "tool_call" else "done ") + f"{lt['text']}({json.dumps(lt.get('args', {}))[:60]})"
        d.text((x0 + 12, yy), head[:150], fill=col, font=font(15, bold=True))
        txt = thinking["text"] if hot else (last[-1]["text"] if last else "")
        for i, ln in enumerate(wrap(txt, 165)[:2]):
            d.text((x0 + 24, yy + 20 + i * 16), ln, fill=TXT if hot else DIM, font=font(13))
        res = events.upto(t, r, ("tool_result",))
        if res:
            d.text((x0 + 24, yy + 52), summarize_result(res[-1])[:160], fill=DIM, font=font(12, mono=True))
        yy += 20 + 2 * 16 + 34
    board = [e for e in events.upto(t, None, ("tool_call",)) if e["text"] == "share_with_fleet"][-1:]
    for e in board:
        d.text((x0 + 12, yy), f"fleet board <- {e['robot']}: {str(e.get('args', {}).get('message', ''))[:140]}",
               fill=(200, 200, 120), font=font(12))


def end_card(W, H, scen, episode, robots, llm_label):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 60), scen.get("title", scen["name"]) + " - outcome", fill=TXT, font=font(32, bold=True))
    y = 120
    ver = episode.get("verification", {})
    for r in robots:
        v = ver.get(r, {})
        rep = episode.get("reports", {}).get(r, {})
        ok = v.get("passed")
        d.text((80, y), f"{r}: simulator check {'PASSED' if ok else ('FAILED' if ok is False else 'n/a')} "
                        f"({v.get('type', '')}{', FELL at ' + str(v.get('fell_at_sim_time_s')) + ' s' if v.get('fell') else ''})",
               fill=(120, 230, 120) if ok else (240, 120, 110), font=font(23, bold=True))
        y += 32
        det = []
        if v.get("type") == "slalom":
            det.append("cones (closest approach, need <= 0.45 m): " + ", ".join(
                f"{c['cone']} {c['closest_m']} m {'ok' if c['passed'] else 'MISSED'}" for c in v.get("cones", []))
                + f"; in order: {'yes' if v.get('in_order') else 'NO'}")
            det.append(f"stop mark: {v.get('stop_distance_m')} m from centre (need <= {v.get('stop_radius_m')}), final "
                       f"speed {v.get('final_speed_m_s')} m/s ({'stopped' if v.get('stopped') else 'NOT stopped'})")
        if v.get("type") == "race_lap":
            det.append(f"laps {v.get('laps_done')}, finished: {'yes, race time ' + str(v.get('race_time_s')) + ' s' if v.get('finished') else 'no'}"
                       f"; time off the track (r < 0.45 or > 1.35 m): {100 * (v.get('off_track_fraction') or 0):.0f} %")
        cl = (episode.get("clearance") or {}).get(r)
        if cl and cl.get("min_body_origin_to_obstacle_box_m") is not None:
            det.append(f"equipment clearance (approx., body origins vs obstacle boxes): {cl['min_body_origin_to_obstacle_box_m']:.3f} m "
                       f"to {cl.get('closest_obstacle')}; possible contact: {'YES' if cl.get('possible_contact') else 'no'}")
        claim = rep.get("success_claimed")
        cl_lines = wrap(f"agent's own claim: {'success' if claim else ('failure' if claim is False else rep.get('error', 'none'))}"
                        f" - {rep.get('summary', '')}", 170)[: (3 if len(robots) == 1 else 2)]
        for ln in det:
            d.text((100, y), ln[:180], fill=TXT, font=font(16))
            y += 22
        for ln in cl_lines:
            d.text((100, y), ln, fill=DIM, font=font(15))
            y += 20
        y += 12
    sk = ver.get("_skaters")
    if sk:
        d.text((80, y), "closest skater-skater distance (trunk centres): " + ", ".join(
            f"{k} {v} m" for k, v in sk["min_trunk_distance_m"].items())
            + ("  -> possible contact" if sk["possible_skater_contact"] else ""), fill=TXT, font=font(16))
        y += 26
    llm = episode.get("llm", {})
    d.text((80, y + 6), f"LLM: {llm_label}; calls {sum(v['calls'] for v in llm.values())}, "
                        f"{sum(v['wall_s'] for v in llm.values()):.0f} s of thinking (simulation paused meanwhile). "
                        f"Simulated time {episode.get('sim_time_s', 0):.1f} s.", fill=TXT, font=font(17))
    pol = episode.get("skate_policy", {})
    notes = [f"Learned: every joint target, from the skating policy {pol.get('name')} (ONNX sha256 {str(pol.get('sha256'))[:12]}...),"
             " trained in Isaac Lab + Newton; its only command is [throttle, 0, heading error].",
             "Agent: tool choice, targets, throttle, lanes (LLM). Scripted: the steering loop / race pilot that turns a "
             "target into the heading error,",
             "brake-to-halt, the obstacle brake reflex, scenario and simulator check.",
             "Robot: skate-wheel MicroDuck (microduck_rollers.usd), the canonical robot plus passive wheels under the feet.",
             "Agent senses are stand-ins: simulator pose/speed; objects in the head camera's field of view from simulator "
             "ground truth; range ahead",
             "from scene geometry. The skating policy itself is blind. One shared Newton world: skaters can collide. "
             "No get-up on skates: a fall ends the run."]
    for i, n in enumerate(notes):
        d.text((80, y + 44 + i * 26), n, fill=DIM, font=font(16))
    return img


if __name__ == "__main__":
    main()
