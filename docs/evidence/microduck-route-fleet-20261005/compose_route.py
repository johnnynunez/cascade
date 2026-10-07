"""Compose a route-run video from the shared owner's overview frames + a measured track map.

Left: RTX overview JPEGs (sim-time aligned, 1x). Right: top-down plot of every robot's
measured world position (owner physics rows), lanes, start/finish lines, per-robot verdict.
Labels the clip honestly: the map is a plot of recorded state, not a render.

Usage: compose_route.py RUN_DIR OUT.mp4 [--fps 10] [--route 5.0] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def load_rows(kit):
    rows = []
    with open(kit / 'physics.jsonl') as f:
        for line in f:
            d = json.loads(line)
            rows.append((d['step'], d['sim_time_s'], {r: (v['position'], v['orientation_wxyz'], v.get('fallen'))
                                                      for r, v in d['robots'].items()}))
    return rows


def yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--route', type=float, default=5.0)
    ap.add_argument('--title', default='CASCADE · MicroDuck fleet · Isaac Sim 6.2 / Newton · one shared world')
    ap.add_argument('--speed', type=float, default=1.0, help='playback speed relative to sim time')
    args = ap.parse_args()
    run = Path(args.run_dir)
    kit = run / 'owner' / 'kit'
    exp = json.load(open(run / 'experiment.json'))
    summary = {s['duck']: s for s in exp.get('summary', [])}
    frames = [json.loads(l) for l in open(kit / 'frames.jsonl')]
    rows = load_rows(kit)
    robots = sorted(rows[0][2])
    starts = {r: np.array(rows[0][2][r][0][:2]) for r in robots}
    # per-robot verdict timing: the walk step's sim window from the driver report
    windows = {}
    for r in robots:
        w = run / r / 'walk.json'
        if w.exists():
            d = json.load(open(w))
            if d.get('steps'):
                st = d['steps'][0]
                windows[r] = (st.get('before_sim_t'), st.get('after_sim_t'))
    # track-map geometry
    ys = np.array([starts[r][1] for r in robots])
    y_min, y_max = ys.min() - 0.6, ys.max() + 0.6
    x_min, x_max = -0.6, args.route + 1.0
    W, H = 1920, 1080
    map_w = 760
    img_w = W - map_w
    scale = min((map_w - 80) / (x_max - x_min), (H - 200) / (y_max - y_min))
    def to_px(p):
        return (img_w + 40 + (p[0] - x_min) * scale, 120 + (y_max - p[1]) * scale)
    out_dir = run / 'composed'
    out_dir.mkdir(exist_ok=True)
    f_small, f_mid, f_big = font(22), font(26), font(34, True)
    row_i = 0
    for k, fr in enumerate(frames):
        t = fr['sim_time_s']
        while row_i + 1 < len(rows) and rows[row_i + 1][1] <= t:
            row_i += 1
        state = rows[row_i][2]
        canvas = Image.new('RGB', (W, H), (18, 18, 22))
        rtx = Image.open(kit / fr['file']).convert('RGB')
        rtx = rtx.resize((img_w, int(rtx.height * img_w / rtx.width)))
        canvas.paste(rtx, (0, (H - rtx.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, W, 70], fill=(10, 10, 14))
        d.text((20, 18), args.title, font=f_big, fill=(240, 240, 240))
        d.text((20, H - 40), f'sim t = {t:6.2f} s   RTX overview (left) · measured track map (right, plot of recorded state)',
               font=f_small, fill=(200, 200, 200))
        # map
        d.rectangle([img_w, 70, W, H], fill=(28, 30, 36))
        for x in np.arange(0, args.route + 1e-6, 1.0):
            p0, p1 = to_px((x, y_min)), to_px((x, y_max))
            col = (120, 200, 120) if x in (0.0, args.route) else (70, 74, 84)
            d.line([p0, p1], fill=col, width=3 if x in (0.0, args.route) else 1)
            d.text((p0[0] - 10, p1[1] - 26), f'{x:.0f} m', font=f_small, fill=(170, 170, 180))
        for r in robots:
            y0 = starts[r][1]
            for yb in (y0 - 0.5, y0 + 0.5):
                d.line([to_px((x_min, yb)), to_px((x_max, yb))], fill=(50, 54, 62), width=1)
        for r in robots:
            pos, q, fallen = state[r]
            px = to_px(pos[:2])
            trail = [to_px(rows[j][2][r][0][:2]) for j in range(max(0, row_i - 400), row_i + 1, 4)]
            if len(trail) > 1:
                d.line(trail, fill=(90, 140, 220), width=2)
            s = summary.get(r, {})
            color = (90, 200, 110) if s.get('confirmed') else (220, 170, 60)
            if fallen:
                color = (230, 70, 70)
            d.ellipse([px[0]-7, px[1]-7, px[0]+7, px[1]+7], fill=color)
            hx, hy = math.cos(yaw(q)), math.sin(yaw(q))
            d.line([px, (px[0] + 18*hx, px[1] - 18*hy)], fill=color, width=3)
            # label: progress now; verdict once its window ended
            dx = pos[0] - starts[r][0]
            label = f'{r}  {dx:+.2f} m'
            win = windows.get(r)
            if win and win[1] is not None and t >= win[1]:
                label += '  ' + ('CONFIRMED' if s.get('confirmed') else (s.get('post_status') or 'n/a').upper())
            d.text((img_w + 48, 120 + (y_max - starts[r][1]) * scale - 32), label, font=f_small, fill=(225, 225, 230))
        canvas.save(out_dir / f'f_{k:05d}.jpg', quality=88)
    fps = max(1, int(round(args.fps * args.speed)))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(out_dir / 'f_%05d.jpg'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', '-movflags', '+faststart', args.out], check=True)
    print('wrote', args.out, 'frames', len(frames))


if __name__ == '__main__':
    main()
