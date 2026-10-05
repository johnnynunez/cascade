"""Compose the keynote showcase video from one shared-owner run: RTX overview frames at 1x sim time,
a lower-third with the exact scope, a small top-down formation inset (measured robot positions from the
owner's physics rows + the presenter's analytic pose), and a final summary card.

Usage: compose_showcase.py RUN_DIR OUT.mp4 [--fps 10]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--inset', type=int, default=420)
    args = ap.parse_args()
    run = Path(args.run_dir)
    kit = run / 'owner' / 'kit'
    receipt = json.load(open(kit / 'receipt.json'))
    choreo = receipt['backend']['showcase']['choreography']
    slots = choreo['formation']['slots']
    frames = [json.loads(l) for l in open(kit / 'frames.jsonl')]
    phys = [json.loads(l) for l in open(kit / 'physics.jsonl')]
    robots = sorted(phys[0]['robots'])
    # robot positions at or before each frame's sim time
    def robot_rows(t):
        row = phys[0]
        for r in phys:
            if r['sim_time_s'] <= t + 1e-9:
                row = r
            else:
                break
        return row
    W, H = Image.open(kit / frames[0]['file']).size
    f_small, f_mid, f_big = font(max(18, W // 96)), font(max(22, W // 80), True), font(max(30, W // 60), True)
    out_dir = run / 'composed'
    out_dir.mkdir(exist_ok=True)
    # inset geometry: world window around the presenter path
    xs = [w[0] for w in choreo['presenter']['waypoints']]
    ys = [w[1] for w in choreo['presenter']['waypoints']]
    x0, x1 = min(xs) - 3.0, max(xs) + 1.0
    y0, y1 = min(ys) - 2.0, max(ys) + 1.5
    inset_w = args.inset
    inset_h = int(inset_w * (y1 - y0) / (x1 - x0))
    def to_inset(x, y):
        return (int((x - x0) / (x1 - x0) * inset_w), int((y1 - y) / (y1 - y0) * inset_h))
    fell = {r: False for r in robots}
    faulted = {r: False for r in robots}
    n = 0
    for k, fr in enumerate(frames):
        t = fr['sim_time_s']
        img = Image.open(kit / fr['file']).convert('RGB')
        d = ImageDraw.Draw(img)
        # lower third
        d.rectangle([0, H - 92, W, H], fill=(8, 8, 12))
        d.text((24, H - 84), 'CASCADE · 12 MicroDucks follow a presenter · Isaac Sim 6.2 / Newton · one world, 1× sim time',
               font=f_mid, fill=(240, 240, 240))
        d.text((24, H - 46), 'presenter = kinematic proxy (pose from the simulation, no perception) · robot twists scripted '
                             'in-process · policy rough_walk_e + BAM · not physical admission', font=f_small, fill=(185, 185, 195))
        d.text((W - 190, H - 84), f't = {t:5.1f} s', font=f_mid, fill=(240, 240, 240))
        # inset
        ix, iy = W - inset_w - 24, 24
        d.rectangle([ix - 6, iy - 6, ix + inset_w + 6, iy + inset_h + 6], fill=(10, 10, 14), outline=(60, 60, 70))
        row = robot_rows(t)
        pres = fr.get('presenter')
        if pres:
            px, py, ph = pres['x'], pres['y'], pres['heading_rad']
            c, s = math.cos(ph), math.sin(ph)
            for sx, sy in slots:
                wx, wy = px + c * sx - s * sy, py + s * sx + c * sy
                u, v = to_inset(wx, wy)
                d.ellipse([ix + u - 3, iy + v - 3, ix + u + 3, iy + v + 3], outline=(90, 90, 110))
            u, v = to_inset(px, py)
            d.ellipse([ix + u - 7, iy + v - 7, ix + u + 7, iy + v + 7], fill=(230, 230, 240))
            d.line([ix + u, iy + v, ix + u + int(14 * c), iy + v - int(14 * s)], fill=(230, 230, 240), width=2)
        for r in robots:
            st = row['robots'][r]
            x, y = st['position'][0], st['position'][1]
            fell[r] = fell[r] or bool(st.get('fallen'))
            faulted[r] = faulted[r] or (st['controller'].get('controller') == 'fault')
            u, v = to_inset(x, y)
            color = (230, 70, 70) if fell[r] else (235, 170, 60) if faulted[r] else (118, 185, 0)
            d.ellipse([ix + u - 4, iy + v - 4, ix + u + 4, iy + v + 4], fill=color)
        d.text((ix, iy + inset_h + 10), 'top view · measured positions · slots ○ · presenter ●',
               font=f_small, fill=(185, 185, 195))
        img.save(out_dir / f'f_{k:05d}.jpg', quality=88)
        n = k + 1
    # summary card
    card = Image.new('RGB', (W, H), (10, 10, 14))
    d = ImageDraw.Draw(card)
    d.text((40, 40), 'Summary', font=f_big, fill=(240, 240, 240))
    last = phys[-1]
    first = phys[0]
    y = 120
    ok = 0
    for r in robots:
        a, b = first['robots'][r]['position'], last['robots'][r]['position']
        net = math.hypot(b[0] - a[0], b[1] - a[1])
        pres = frames[-1].get('presenter') or {}
        px, py, ph = pres.get('x', 0.), pres.get('y', 0.), pres.get('heading_rad', 0.)
        i = robots.index(r)
        sx, sy = slots[i]
        c, s = math.cos(ph), math.sin(ph)
        tx, ty = px + c * sx - s * sy, py + s * sx + c * sy
        slot_err = math.hypot(b[0] - tx, b[1] - ty)
        state = 'FELL' if fell[r] else 'FAULT' if faulted[r] else 'walking/standing'
        ok += not (fell[r] or faulted[r])
        d.text((40, y), f'{r}: net {net:4.2f} m · final slot error {slot_err:4.2f} m · {state}', font=f_small,
               fill=(230, 70, 70) if fell[r] else (235, 170, 60) if faulted[r] else (118, 185, 0))
        y += 28
    d.text((40, y + 16), f'{ok}/{len(robots)} robots neither fell nor faulted over {last["sim_time_s"]:.0f} s of simulation; '
                         f'presenter path {choreo["presenter"]["length_m"]:.1f} m at {choreo["presenter"]["speed_m_s"]:.2f} m/s. '
                         'No verifier verdict: scripted showcase, not an agent task.', font=f_small, fill=(200, 200, 210))
    for j in range(args.fps * 4):
        card.save(out_dir / f'f_{n + j:05d}.jpg', quality=88)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', str(out_dir / 'f_%05d.jpg'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', args.out], check=True)
    print('wrote', args.out, 'frames', n + args.fps * 4, 'ok', ok, 'of', len(robots), 'fell', sum(fell.values()),
          'faulted', sum(faulted.values()))


if __name__ == '__main__':
    main()
