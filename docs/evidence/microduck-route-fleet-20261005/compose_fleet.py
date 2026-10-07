"""Compose the fleet route video: 4x3 mosaic of each world's RTX overview, aligned on each robot's
own sim clock at its walk start, 1x sim time, with measured progress labels and a final summary card.

Usage: compose_fleet.py RUN_DIR OUT.mp4 [--fps 10]
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


def load_robot(run, robot_id):
    d = run / robot_id
    kit = d / 'owner' / 'kit'
    walk = json.load(open(d / 'walk.json'))
    frames = [json.loads(l) for l in open(kit / 'frames.jsonl')]
    phys = []
    for l in open(kit / 'physics.jsonl'):
        r = json.loads(l)
        (rid, v), = r['robots'].items()
        phys.append((r['sim_time_s'], v['position'], v.get('fallen')))
    step = walk['steps'][0] if walk.get('steps') else None
    t_start = step['before_sim_t'] if step and step.get('before_sim_t') is not None else frames[0]['sim_time_s']
    t_end = step['after_sim_t'] if step and step.get('after_sim_t') is not None else None
    x0 = phys[0][1][0] if phys else 0.
    return dict(id=robot_id, kit=kit, walk=walk, step=step, frames=frames, phys=phys, t_start=t_start, t_end=t_end, x0=x0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('out')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--tail-s', type=float, default=4.0)
    args = ap.parse_args()
    run = Path(args.run_dir)
    exp = json.load(open(run / 'experiment.json'))
    robots = [load_robot(run, r) for r in exp['robot_ids'] if (run / r / 'walk.json').exists()]
    dt = 1.0 / args.fps
    # relative time: 0 = each robot's walk start; clip length = longest walk + tail
    spans = [((r['t_end'] or r['frames'][-1]['sim_time_s']) - r['t_start']) for r in robots]
    length = max(spans) + args.tail_s
    n = int(length / dt) + 1
    W, H = 1920, 1080
    cols, rows = 4, 3
    tile_w, tile_h = W // cols, (H - 90) // rows
    f_small, f_mid, f_big = font(19), font(22, True), font(30, True)
    out_dir = run / 'composed'
    out_dir.mkdir(exist_ok=True)
    for k in range(n):
        t_rel = k * dt
        canvas = Image.new('RGB', (W, H), (14, 14, 18))
        d = ImageDraw.Draw(canvas)
        for idx, r in enumerate(robots):
            t = r['t_start'] + t_rel
            # frame at or before t
            fr = None
            for f in r['frames']:
                if f['sim_time_s'] <= t + 1e-9:
                    fr = f
                else:
                    break
            if fr is None:
                fr = r['frames'][0]
            tile = Image.open(r['kit'] / fr['file']).convert('RGB').resize((tile_w, tile_h))
            x, y = (idx % cols) * tile_w, 90 + (idx // cols) * tile_h
            canvas.paste(tile, (x, y))
            # measured progress from the owner's physics rows (<= t)
            pos = None
            for ts, p, fallen in r['phys']:
                if ts <= t + 1e-9:
                    pos = (p, fallen)
                else:
                    break
            prog = 0. if pos is None else pos[0][0] - r['x0']
            fell = bool(pos and pos[1])
            st = r['step'] or {}
            done = r['t_end'] is not None and t >= r['t_end']
            verdict = ''
            color = (235, 235, 235)
            if done:
                if st.get('confirmed'):
                    verdict, color = 'CONFIRMED', (90, 210, 110)
                else:
                    verdict, color = (st.get('post_status') or 'FAILED').upper(), (230, 170, 60)
            if fell:
                verdict, color = 'FELL', (230, 70, 70)
            label = f"{r['id']}  {r['walk']['distance_m']:.1f} m route   x {prog:+.2f} m"
            d.rectangle([x, y, x + tile_w, y + 30], fill=(0, 0, 0))
            d.text((x + 8, y + 5), label, font=f_small, fill=(235, 235, 235))
            if verdict:
                d.text((x + tile_w - 8 - d.textlength(verdict, font=f_mid), y + 3), verdict, font=f_mid, fill=color)
            d.rectangle([x, y, x + tile_w - 1, y + tile_h - 1], outline=(40, 40, 48))
        d.rectangle([0, 0, W, 90], fill=(8, 8, 12))
        d.text((20, 14), 'CASCADE · 12 MicroDucks · independent 3.5–5.5 m routes · Isaac Sim 6.2 / Newton · one world per robot',
               font=f_big, fill=(240, 240, 240))
        d.text((20, 58), f'walk time t = {t_rel:5.1f} s (each tile on its own world clock, 1x)   RTX overview per world · '
                         f'verdicts from the unchanged independent verifier · not physical admission', font=f_small, fill=(190, 190, 200))
        canvas.save(out_dir / f'f_{k:05d}.jpg', quality=86)
    # summary card (3 s)
    card = Image.new('RGB', (W, H), (10, 10, 14))
    d = ImageDraw.Draw(card)
    d.text((40, 40), 'Summary', font=f_big, fill=(240, 240, 240))
    y = 110
    conf = 0
    for r in robots:
        st = r['step'] or {}
        m = (st.get('post_metrics') or {}).get('metrics') or {}
        ok = st.get('confirmed')
        conf += bool(ok)
        bd = m.get('body_displacement_m') or [float('nan'), float('nan')]
        net = None
        if st.get('before_pos') and st.get('after_pos'):
            net = math.hypot(st['after_pos'][0] - st['before_pos'][0], st['after_pos'][1] - st['before_pos'][1])
        line = (f"{r['id']}  route {r['walk']['distance_m']:.1f} m  ->  verifier body {bd[0]:+.2f} m (lateral {bd[1]:+.2f})"
                f"   net world {net if net is None else round(net, 2)} m   {('CONFIRMED' if ok else (st.get('post_status') or 'n/a').upper())}"
                f"   {st.get('error') or ''}")
        d.text((40, y), line, font=f_small, fill=(90, 210, 110) if ok else (230, 170, 60))
        y += 30
    d.text((40, y + 20), f'{conf}/{len(robots)} routes confirmed by the independent verifier (unchanged limits). '
                          'One Isaac Sim world per robot; twelve robots in ONE world remain blocked by the owner\'s RPC latency (see report).',
           font=f_small, fill=(200, 200, 210))
    for j in range(args.fps * 4):
        card.save(out_dir / f'f_{n + j:05d}.jpg', quality=86)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', str(out_dir / 'f_%05d.jpg'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24', '-movflags', '+faststart', args.out], check=True)
    print('wrote', args.out, 'frames', n + args.fps * 4, 'confirmed', conf, 'of', len(robots))


if __name__ == '__main__':
    main()
