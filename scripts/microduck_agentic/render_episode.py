# SPDX-License-Identifier: Apache-2.0
"""Render an agentic MicroDuck episode to a 1080p review video: OVRTX views of the recorded physics + overlays.

Panels and their labels (all English, burned in):
* "illustrative camera view": OVRTX RTX renders (follow / overview cameras) of the RECORDED Newton physics poses on
  the canonical MicroDuck.usd; never an input to anything;
* "POLICY OBSERVATION": the exact held ToF 8x8 frame and the exact 48x36 head RGB tensor of each robot (inputs of the
  onboard perception actor; the agent reads the same ToF frame and sees the same RGB, upscaled);
* "<robot>'s own map": the robot's own ToF occupancy map at that time (pose from the simulator);
* the agent panel: task, active controller (which learned policy is driving), the latest LLM decision and receipts.
Simulation time is paused while an agent thinks; the video shows each decision as a short labelled hold.

Run in the Isaac Lab venv (OVRTX 0.5.1 + pxr + PIL):
    python scripts/microduck_agentic/render_episode.py --run runs/backstage_001 --output runs/backstage_001/video.mp4
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

for extra in filter(None, os.environ.get("CASCADE_APPEND_SYSPATH", "").split(os.pathsep)):
    sys.path.append(extra)

from cascade.sim.microduck_ovrtx_replay import ReplayPoses, build_render_scene, look_at  # noqa: E402
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxRenderer, SceneSnapshot  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG = (18, 20, 25)
PANEL = (30, 33, 40)
TXT = (230, 232, 236)
DIM = (150, 155, 165)
YEL = (255, 205, 60)
ROBOT_COLORS = [(255, 170, 40), (80, 190, 255), (150, 230, 110), (240, 110, 200)]
MODE_LABEL = {
    "stand": "standing balance (learned walker, zero command)",
    "walk": "learned walker velocity_flat_existing.onnx",
    "onboard_hero": "onboard perception actor onboard_hero.pt (ToF+RGB -> joints)",
    "getup": "learned get-up policy getup_kitfloor_ft_model_1100",
    "depth_nav": "learned depth navigation v4 -> walker",
}
SDK_051 = (("ovrtx", "0.5.1.385782"), ("ovstage", "0.2.1.385922"))


def font(size, bold=False, mono=False):
    return ImageFont.truetype(MONO if mono else (FONT_B if bold else FONT), size)


def wrap(text, width):
    out = []
    for para in str(text).splitlines():
        out += textwrap.wrap(para, width) or [""]
    return out


def tof_panel(mm, status, size=250):
    img = Image.new("RGB", (size, size), (10, 10, 12))
    d = ImageDraw.Draw(img)
    cell = size // 8
    f = font(max(9, cell // 3), mono=True)
    for z in range(64):
        r, c = divmod(z, 8)
        st, v = int(status[z]), int(mm[z])
        x0, y0 = c * cell, r * cell
        if st in (5, 9) and v >= 100:
            m = min(v / 1000.0, 2.0) / 2.0
            col = (int(255 * (1 - m)), int(80 + 120 * m), int(40 + 200 * m))
            label = f"{v / 1000:.2f}"
        elif st == 255:
            col, label = (20, 30, 70), "--"
        else:
            col, label = (0, 0, 0), "x"
        d.rectangle([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2], fill=col)
        d.text((x0 + 3, y0 + cell // 3), label, fill=(255, 255, 255), font=f)
    return img


def map_image(snap, bounds, res, robot_xy_yaw, size, landmarks, visited, color):
    lo, drop = snap
    h, w = lo.shape
    img = np.full((h, w, 3), 72, np.uint8)
    img[lo < -0.6] = (226, 229, 233)
    img[lo > 0.6] = (25, 25, 30)
    img[drop] = (220, 60, 50)
    im = Image.fromarray(np.flipud(img)).resize((size[0], int(size[0] * h / w)), Image.NEAREST)
    if im.height > size[1]:
        im = Image.fromarray(np.flipud(img)).resize((int(size[1] * w / h), size[1]), Image.NEAREST)
    d = ImageDraw.Draw(im)
    x0, y0 = bounds[0], bounds[1]
    sx, sy = im.width / (w * res), im.height / (h * res)

    def px(x, y):
        return (x - x0) * sx, im.height - (y - y0) * sy

    if visited and len(visited) > 1:
        d.line([px(v[0], v[1]) for v in visited], fill=(40, 110, 230), width=2)
    for lm in landmarks:
        u, v = px(lm["x"], lm["y"])
        d.ellipse([u - 5, v - 5, u + 5, v + 5], outline=(255, 210, 0), width=2)
        d.text((u + 6, v - 8), lm["label"][:14], fill=(255, 210, 0), font=font(11))
    x, y, yaw = robot_xy_yaw
    u, v = px(x, y)
    tip = (u + 12 * math.cos(yaw), v - 12 * math.sin(yaw))
    d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=color)
    d.line([(u, v), tip], fill=color, width=3)
    return im


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Events:
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)] if Path(path).exists() else []

    def upto(self, t, robot=None, kinds=None):
        return [e for e in self.rows if e["sim_time_s"] <= t + 1e-6 and (robot is None or e["robot"] == robot)
                and (kinds is None or e["kind"] in kinds)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--microduck_usd", type=Path,
                   default=Path.home() / "Projects/microducks/microduck_gtceu/assets/MicroDuck.usd")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hold_s", type=float, default=0.8)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--render_course", type=Path, default=None,
                   help="render this course USD instead of the physics course (must match --render_course_sha256)")
    p.add_argument("--render_course_sha256", default=None)
    p.add_argument("--llm_label", default="GPT-6-Astra via Codex CLI (Cascade codex_astra profile)")
    a = p.parse_args()
    run = a.run
    with np.load(run / "recording.npz") as z:  # materialize once: NpzFile indexing re-decompresses every access
        rec = {k: z[k] for k in z.files}
    meta = json.loads((run / "recording.json").read_text())
    scen = yaml.safe_load((run / "scenario.yaml").read_text())
    episode = json.loads((run / "episode.json").read_text())
    events = Events(run / "events.jsonl")
    robots = [str(r) for r in rec["robot_names"]]
    modes = [str(m) for m in rec["modes"]]
    times = rec["time_s"]
    body_pose = rec["body_pose"]  # (T, R, 16, 7)
    T = len(times)
    multi = len(robots) > 1
    course = scen.get("scene")
    repo = Path(__file__).resolve().parents[2]
    course_path = None
    resolved = run / "scenario_resolved.json"
    if resolved.exists() and json.loads(resolved.read_text()).get("scene_path"):
        course_path = Path(json.loads(resolved.read_text())["scene_path"])
    elif course:
        course_path = Path(course) if Path(course).is_absolute() else repo / course
    render_course, render_sha = a.render_course, a.render_course_sha256
    if render_course is None and scen.get("render_scene_env"):
        import os

        render_course, render_sha = os.environ.get(scen["render_scene_env"]), scen.get("render_scene_sha256")
        if not render_course:
            raise SystemExit(f"set {scen['render_scene_env']} (sha256 {render_sha})")
    if render_course is not None:
        import hashlib

        render_course = Path(render_course).resolve(strict=True)
        digest = hashlib.sha256(render_course.read_bytes()).hexdigest()
        if digest != render_sha:
            raise SystemExit(f"{render_course} sha256 {digest} != expected {render_sha}")
        print(f"render course: {render_course} (physics course was {course_path})", flush=True)
        scen["_render_course_note"] = (f"Rendered course: {render_course.name} (visual source asset); physics ran on "
                                       f"{Path(course_path).name} derived from it (same frame).")
        course_path = render_course
    rdir = run / "render"
    rdir.mkdir(exist_ok=True)
    layout = build_render_scene(rdir / "scene.usdc", course_usd=course_path, microduck_usd=a.microduck_usd,
                                robots=robots)
    replay = ReplayPoses(layout, [str(b) for b in rec["body_names"]], robots)
    maps = {}
    for r in robots:
        with np.load(run / f"maps_{r}.npz") as z:
            m = {k: z[k] for k in z.files}
        maps[r] = {"t": m["t"], "lo": m["logodds"], "drop": m["drop"], "bounds": m["bounds"], "res": float(m["res"]),
                   "visited": json.loads(str(m["visited"])), "landmarks": json.loads(str(m["landmarks"]))}

    W, H = 1920, 1080
    main_w, main_h = 1280, 720
    vfov = math.radians(50)
    cams = [CameraSpec("main", main_w, main_h, (main_h / 2) / math.tan(vfov / 2), (main_h / 2) / math.tan(vfov / 2),
                       np.eye(4), near_m=0.02, far_m=60.0)]
    if not multi:
        cams.append(CameraSpec("aux", 640, 360, 180 / math.tan(math.radians(55) / 2), 180 / math.tan(math.radians(55) / 2),
                               np.eye(4), near_m=0.02, far_m=60.0))
    ov = scen.get("render", {}).get("overview", {"eye": [-1, -2, 2], "target": [1, 0, 0]})
    fol = scen.get("render", {}).get("follow", {"distance": 0.9, "height": 0.55})
    T_over = look_at(ov["eye"], ov["target"])
    step = max(1, int(round(1.0 / (a.fps * float(times[1] - times[0])))))
    frames_idx = list(range(0, T, step))
    if a.max_frames:
        frames_idx = frames_idx[: a.max_frames]
    decisions = [e for e in events.rows if e["kind"] == "llm_reply"]
    ffmpeg = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                               "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)], stdin=subprocess.PIPE)
    yaw_s = None
    t0 = time.time()
    seq = 0
    rendered = 0
    dec_ptr = 0
    with OvrtxRenderer(rdir / "scene.usdc", cams, dynamic_paths=replay.paths, device=a.device,
                       render_dt=1.0 / a.fps, sdk_versions=SDK_051) as renderer:
        for n, k in enumerate(frames_idx):
            t = float(times[k])
            root = body_pose[k, 0, 0]
            yaw = yaw_of(root[3:7])
            yaw_s = yaw if yaw_s is None else yaw_s + 0.08 * ((yaw - yaw_s + math.pi) % (2 * math.pi) - math.pi)
            if multi:
                cam_poses = {"main": T_over}
            else:
                eye = root[:3] + np.array([-fol["distance"] * math.cos(yaw_s - 0.5), -fol["distance"] * math.sin(yaw_s - 0.5),
                                           fol["height"]])
                eye[2] = max(eye[2], root[2] + 0.25)
                cam_poses = {"main": look_at(eye, root[:3] + np.array([0, 0, 0.02])), "aux": T_over}
            seq += 1
            snap = SceneSnapshot("cascade.microduck_agentic.replay", meta.get("scenario", "episode"), seq, t + 1e-3 * seq,
                                 time.monotonic(), replay.transforms(body_pose[k]), cam_poses)
            frames = renderer.render(snap)
            rendered += 1
            views = {name: Image.fromarray(np.ascontiguousarray(f.rgb[..., ::-1])) for name, f in frames.items()}
            new_decisions = []
            while dec_ptr < len(decisions) and decisions[dec_ptr]["sim_time_s"] <= t + 1e-6:
                new_decisions.append(decisions[dec_ptr])
                dec_ptr += 1
            holds = [(d, int(a.hold_s * a.fps)) for d in new_decisions] or [(None, 1)]
            for dec, nhold in holds:
                img = compose(W, H, views, rec, k, t, robots, modes, maps, events, scen, episode, multi, a.llm_label,
                              thinking=dec, cams=cams, cam_poses=cam_poses)
                buf = np.asarray(img, dtype=np.uint8).tobytes()
                for _ in range(nhold):
                    ffmpeg.stdin.write(buf)
            if n % 100 == 0:
                print(f"frame {n}/{len(frames_idx)} t={t:.2f} wall={time.time() - t0:.0f}s", flush=True)
        # end card: 3 s
        card = end_card(W, H, scen, episode, robots, a.llm_label)
        buf = np.asarray(card, dtype=np.uint8).tobytes()
        for _ in range(3 * a.fps):
            ffmpeg.stdin.write(buf)
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise SystemExit("ffmpeg failed")
    report = {"frames_rendered": rendered, "fps": a.fps, "sample_step": step, "wall_s": round(time.time() - t0, 1),
              "renderer": "cascade.sim.ovrtx_renderer.OvrtxRenderer (RealTimePathTracing)", "sdk": SDK_051,
              "microduck_usd_sha256": layout["microduck_usd_sha256"], "decision_holds": len(decisions),
              "physics_steps": 0, "pose_repair": False}
    (run / "render_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("RENDER DONE", json.dumps(report), flush=True)


def project(cam, T_cam, p):
    Tinv = np.linalg.inv(T_cam)
    q = Tinv @ np.array([p[0], p[1], p[2], 1.0])
    if q[2] <= 0.05:
        return None
    return cam.fx * q[0] / q[2] + (cam.width - 1) / 2, cam.fy * q[1] / q[2] + (cam.height - 1) / 2


def compose(W, H, views, rec, k, t, robots, modes, maps, events, scen, episode, multi, llm_label, thinking, cams,
            cam_poses):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 58], fill=(12, 13, 17))
    d.text((16, 8), scen.get("title", scen["name"]), fill=TXT, font=font(26, bold=True))
    d.text((16, 38), f"Cascade agent ({llm_label}) chooses tools; unchanged learned MicroDuck policies move the robot. "
                     f"Isaac Lab 3.0 + Newton physics, OVRTX render.", fill=DIM, font=font(14))
    d.text((W - 270, 14), f"sim time {t:6.2f} s", fill=YEL, font=font(24, bold=True, mono=True))
    main = views["main"]
    img.paste(main, (0, 60))
    md = ImageDraw.Draw(img)
    if multi:
        cam = cams[0]
        for ri, r in enumerate(robots):
            root = rec["body_pose"][k, ri, 0, :3]
            uv = project(cam, cam_poses["main"], root + np.array([0, 0, 0.32]))
            if uv:
                u, v = uv
                col = ROBOT_COLORS[ri % 4]
                md.rectangle([u - 34, 60 + v - 22, u + 34, 60 + v - 2], fill=col)
                md.text((u - 30, 60 + v - 21), r, fill=(0, 0, 0), font=font(15, bold=True))
    tag = "illustrative camera view - OVRTX render of the recorded physics (" + ("overview" if multi else "follow camera") + ")"
    md.rectangle([0, 60 + 720 - 26, 760, 60 + 720], fill=(0, 0, 0))
    md.text((8, 60 + 720 - 23), tag, fill=TXT, font=font(15))
    if multi:
        right_multi(img, rec, k, t, robots, modes, maps)
        agent_panel_multi(img, rec, k, t, robots, modes, events, thinking)
    else:
        img.paste(views["aux"], (1280, 60))
        d.rectangle([1280, 60 + 360 - 22, 1280 + 330, 60 + 360], fill=(0, 0, 0))
        d.text((1286, 60 + 360 - 20), "illustrative camera view - overview", fill=TXT, font=font(14))
        right_single(img, rec, k, t, robots[0], maps)
        agent_panel_single(img, rec, k, t, robots[0], modes, events, thinking, scen)
    return img


def obs_block(img, x, y, rec, k, ri, tof_size=200, rgb_scale=5, label=True):
    d = ImageDraw.Draw(img)
    if label:
        d.text((x, y), "POLICY OBSERVATION (exact actor inputs)", fill=YEL, font=font(15, bold=True))
        y += 22
    tof = tof_panel(rec["tof_mm"][k, ri], rec["tof_status"][k, ri], tof_size)
    img.paste(tof, (x, y))
    rgb = Image.fromarray(rec["rgb"][k, ri]).resize((48 * rgb_scale, 36 * rgb_scale), Image.NEAREST)
    img.paste(rgb, (x + tof_size + 12, y))
    d.text((x, y + tof_size + 3), "ToF 8x8, metres (VL53L8CX model, 15 Hz)", fill=DIM, font=font(12))
    d.text((x + tof_size + 12, y + 36 * rgb_scale + 3), "head RGB 48x36 (agent sees it too)", fill=DIM, font=font(12))
    return y + tof_size + 20


def current_map(maps, r, t):
    m = maps[r]
    j = int(np.searchsorted(m["t"], t, side="right")) - 1
    if j < 0:
        return None, [], []
    return (m["lo"][j], m["drop"][j]), m["landmarks"][j], m["visited"][j]


def right_single(img, rec, k, t, r, maps):
    d = ImageDraw.Draw(img)
    x = 1290
    y = obs_block(img, x, 430, rec, k, 0, tof_size=200, rgb_scale=5)
    d.text((x, y + 2), f"{r}'s OWN map (its ToF frames; pose from simulator)", fill=YEL, font=font(15, bold=True))
    snap, lms, visited = current_map(maps, r, t)
    if snap is not None:
        root = rec["body_pose"][k, 0, 0]
        mi = map_image(snap, maps[r]["bounds"], maps[r]["res"], (root[0], root[1], yaw_of(root[3:7])),
                       (620, 1080 - y - 30), lms, visited, ROBOT_COLORS[0])
        img.paste(mi, (x, y + 24))


def last_tool(events, t, r):
    calls = events.upto(t, r, ("tool_call", "tool_result"))
    return calls[-1] if calls else None


def agent_panel_single(img, rec, k, t, r, modes, events, thinking, scen):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 0, 784, 1280, 1080
    d.rectangle([x0, y0, x1, y1], fill=PANEL)
    mode = modes[int(rec["mode"][k, 0])]
    cmd = rec["command"][k, 0]
    badge = MODE_LABEL[mode] + (f"  cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f}" if mode in ("walk", "depth_nav") else "")
    d.rectangle([x0 + 10, y0 + 8, x1 - 10, y0 + 36], fill=(50, 80, 50) if mode != "stand" else (60, 60, 70))
    d.text((x0 + 18, y0 + 12), "ACTIVE CONTROLLER: " + badge, fill=TXT, font=font(16, bold=True))
    task = next((e["text"] for e in events.rows if e["robot"] == r and e["kind"] == "task"), "")
    task_lines = wrap("TASK: " + task, 150)
    if len(task_lines) > 2:
        task_lines = task_lines[:2]
        task_lines[1] = task_lines[1][:146].rstrip() + " ..."
    for i, ln in enumerate(task_lines):
        d.text((x0 + 12, y0 + 42 + i * 15), ln, fill=DIM, font=font(13))
    if thinking is not None:
        tc = thinking.get("tool_calls") or [{}]
        call = tc[0]
        d.text((x0 + 12, y0 + 76), f"AGENT DECIDES (LLM {thinking.get('latency_s', 0):.1f} s, simulation paused): "
                                   f"{call.get('name')}({json.dumps(call.get('args'))[:80]})", fill=YEL,
               font=font(17, bold=True))
        lines = wrap(thinking.get("text", ""), 150)[:6]
    else:
        last = [e for e in events.upto(t, r, ("llm_reply",))]
        lines = wrap(last[-1]["text"], 150)[:6] if last else []
        lt = last_tool(events, t, r)
        if lt is not None:
            lab = f"{lt['text']}({json.dumps(lt.get('args', {}))[:70]})" if lt["kind"] == "tool_call" else f"{lt['text']} done"
            d.text((x0 + 12, y0 + 76), "EXECUTING: " + lab if lt["kind"] == "tool_call" else "LAST: " + lab,
                   fill=TXT, font=font(17, bold=True))
    for i, ln in enumerate(lines):
        d.text((x0 + 12, y0 + 102 + i * 18), ln, fill=TXT, font=font(14))
    res = [e for e in events.upto(t, r, ("tool_result",))][-3:]
    yy = y0 + 102 + 6 * 18 + 8
    for e in res:
        rr = e.get("result") or {}
        s = f"t={e['sim_time_s']:.1f}s {e['text']}: " + (
            f"{rr.get('stop_reason')} moved {rr.get('traveled_m')} m, turned {rr.get('yaw_change_deg')} deg"
            if "stop_reason" in rr else ("ok" if rr.get("ok") else str(rr.get("error", ""))[:90]))
        d.text((x0 + 12, yy), s[:160], fill=DIM, font=font(13, mono=True))
        yy += 17


def right_multi(img, rec, k, t, robots, modes, maps):
    d = ImageDraw.Draw(img)
    row_h = 1020 // len(robots)
    for ri, r in enumerate(robots):
        y = 60 + ri * row_h
        col = ROBOT_COLORS[ri % 4]
        d.rectangle([1280, y, 1920, y + row_h - 4], fill=PANEL)
        mode = modes[int(rec["mode"][k, ri])]
        d.text((1288, y + 4), f"{r}", fill=col, font=font(18, bold=True))
        d.text((1370, y + 7), MODE_LABEL[mode][:58], fill=TXT, font=font(13))
        tof = tof_panel(rec["tof_mm"][k, ri], rec["tof_status"][k, ri], 136)
        img.paste(tof, (1288, y + 30))
        rgb = Image.fromarray(rec["rgb"][k, ri]).resize((48 * 3, 36 * 3), Image.NEAREST)
        img.paste(rgb, (1288 + 142, y + 30))
        d.text((1288, y + 30 + 138), "POLICY OBSERVATION: ToF 8x8 | RGB 48x36", fill=YEL, font=font(11, bold=True))
        snap, lms, visited = current_map(maps, r, t)
        if snap is not None:
            root = rec["body_pose"][k, ri, 0]
            mi = map_image(snap, maps[r]["bounds"], maps[r]["res"], (root[0], root[1], yaw_of(root[3:7])),
                           (330, row_h - 50), lms, visited, col)
            img.paste(mi, (1582, y + 30))
        d.text((1582, y + row_h - 22), f"{r}'s own map", fill=YEL, font=font(11, bold=True))


def agent_panel_multi(img, rec, k, t, robots, modes, events, thinking):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = 0, 784, 1280, 1080
    d.rectangle([x0, y0, x1, y1], fill=PANEL)
    d.text((x0 + 12, y0 + 6), "AGENTS (one Cascade agent per robot, each with its own memory and map)", fill=YEL,
           font=font(15, bold=True))
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
        txt = (thinking["text"] if hot else (last[-1]["text"] if last else ""))
        for i, ln in enumerate(wrap(txt, 165)[:3]):
            d.text((x0 + 24, yy + 20 + i * 16), ln, fill=TXT if hot else DIM, font=font(13))
        yy += 20 + 3 * 16 + 8
    board = [e for e in events.upto(t, None, ("tool_call",)) if e["text"] == "share_with_fleet"][-2:]
    for e in board:
        d.text((x0 + 12, yy), f"fleet board <- {e['robot']}: {str(e.get('args', {}).get('message', ''))[:140]}",
               fill=(200, 200, 120), font=font(12))
        yy += 15


def end_card(W, H, scen, episode, robots, llm_label):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((80, 70), scen.get("title", scen["name"]) + " - outcome", fill=TXT, font=font(34, bold=True))
    y = 140
    for r in robots:
        v = episode.get("verification", {}).get(r, {})
        rep = episode.get("reports", {}).get(r, {})
        ok = v.get("passed")
        d.text((80, y), f"{r}: simulator check {'PASSED' if ok else ('FAILED' if ok is False else 'n/a')}"
                        f" ({v.get('type', '')}{', distance ' + str(v.get('distance_m')) + ' m' if 'distance_m' in v else ''}"
                        f"{', fallen' if v.get('fallen') else ''})", fill=(120, 230, 120) if ok else (240, 120, 110),
               font=font(24, bold=True))
        y += 36
        cl = (episode.get("clearance") or {}).get(r)
        if cl and cl.get("min_body_origin_to_obstacle_box_m") is not None:
            d.text((100, y), f"equipment clearance (approx., body origins vs obstacle boxes): "
                             f"{cl['min_body_origin_to_obstacle_box_m']:.3f} m to {cl.get('closest_obstacle')}; possible "
                             f"contact: {'YES' if cl.get('possible_contact') else 'no'}"
                             f"{' (checked by the verifier)' if 'possible_contact' in v else ' (not part of this check)'}",
                   fill=TXT, font=font(16))
            y += 24
        claim = rep.get("success_claimed")
        per_robot = 4 if len(robots) == 1 else 2
        claim_lines = wrap(f"agent's own claim: {'success' if claim else ('failure' if claim is False else rep.get('error', 'none'))}"
                           f" - {rep.get('summary', '')}", 175)
        if len(claim_lines) > per_robot:
            claim_lines = claim_lines[:per_robot]
            claim_lines[-1] = claim_lines[-1][:170].rstrip() + " ..."
        for ln in claim_lines:
            d.text((100, y), ln, fill=DIM, font=font(16))
            y += 22
        y += 18
    llm = episode.get("llm", {})
    d.text((80, y + 10), f"LLM: {llm_label}; calls {sum(v['calls'] for v in llm.values())}, "
                         f"{sum(v['wall_s'] for v in llm.values()):.0f} s of thinking (simulation paused meanwhile). "
                         f"Simulated time {episode.get('sim_time_s', 0):.1f} s.", fill=TXT, font=font(18))
    notes = ["Learned: every joint target (walker, onboard ToF+RGB actor, get-up, depth navigation).",
             "Agent: tool choice and parameters (LLM). Scripted: ToF stop guard, turn-angle loop, waypoint heading/speed",
             "loop (it only picks the walker's velocity command), A* route proposals on the robot's own map, scenario and",
             "simulator check.",
             "Maps: each robot's own ToF occupancy grid with the simulator pose (no SLAM in this demo).",
             "Robots are separate Newton worlds: no robot-robot contact, robots do not see each other."]
    if scen.get("_render_course_note"):
        notes.append(scen["_render_course_note"])
    if scen.get("detectable"):
        notes.append("Object detections: simulator ground truth within the head camera's field of view "
                     "(detector stand-in; the 48x36 head RGB only draws the first 1.5 m).")
    for i, n in enumerate(notes):
        d.text((80, y + 60 + i * 28), n, fill=DIM, font=font(17))
    return img


if __name__ == "__main__":
    main()
