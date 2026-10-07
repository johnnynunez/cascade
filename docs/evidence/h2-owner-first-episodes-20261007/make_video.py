"""Encode an H2 episode video from the owner's overview frames with an honest overlay.

Overlay per frame: sim time, physics step, the admitted command active at that sim time
(from the client episode report) and the independent verifier's verdict for that command.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

run = Path(sys.argv[1])
out_mp4 = Path(sys.argv[2])
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
kit = run / 'kit'
frames = [json.loads(l) for l in (kit / 'frames.jsonl').read_text().splitlines()]
episode = json.loads((run / 'episode.json').read_text())
steps = episode['steps']
font = ImageFont.load_default(size=26)
small = ImageFont.load_default(size=20)
work = run / 'video-frames'
work.mkdir(exist_ok=True)


def active(sim_t):
    for s in steps:
        t0, t1 = s.get('before_sim_t'), s.get('after_sim_t')
        if t0 is not None and t1 is not None and t0 <= sim_t <= t1:
            return s
    return None


for i, f in enumerate(frames):
    img = Image.open(kit / f['file']).convert('RGB')
    draw = ImageDraw.Draw(img)
    t = f['sim_time_s']
    lines = [f"Unitree H2 · NVIDIA Velocity-H2-History-v0 · PhysX (Isaac Sim 6.2 build) · CASCADE owner",
             f"sim {t:7.2f} s   step {f['step']:6d}   policy 50 Hz / physics 200 Hz"]
    s = active(t)
    if s is not None:
        a = s['args']
        lines.append(f"walk_velocity vx={a['vx']:+.1f} vy={a['vy']:+.1f} wz={a['wz']:+.1f} for {a['duration_s']:.0f} s  "
                     f"-> verifier: {s['post_status'].upper()}")
    else:
        lines.append('no admitted command: standing (zero twist, balance policy running)')
    lines.append('candidate profile, no physical admission · x markers 1 m (orange), y (blue)')
    y = 12
    for k, line in enumerate(lines):
        fnt = font if k < 3 else small
        box = draw.textbbox((16, y), line, font=fnt)
        draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(0, 0, 0))
        draw.text((16, y), line, font=fnt, fill=(255, 255, 255) if 'REFUTED' not in line else (255, 170, 90))
        y = box[3] + 10
    img.save(work / f'{i:06d}.jpg', quality=90)

subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', str(work / '%06d.jpg'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '26', '-preset', 'medium', '-movflags', '+faststart',
                str(out_mp4)], check=True)
print(out_mp4, out_mp4.stat().st_size, 'bytes', len(frames), 'frames')
