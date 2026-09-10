#!/usr/bin/env python3
"""Record the cascade agentic loop to an MP4.

Streams the isaac cam0 + side + wrist cameras while a background agent task
runs,
writing annotated frames to /tmp/wrc_frames, then muxes them to
/tmp/wrc_graspgenx_demo.mp4 with ffmpeg.
Run from the models/ dir (mobileclip CWD dependency) with PYTHONPATH=../src.
"""
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")
from cascade.sim.bridge_client import BridgeClient  # noqa: E402

FPS = 6
DUR = 100  # seconds -- full pick cycle: approach, grasp, transport, drop
OUT_DIR = "/tmp/wrc_frames"
subprocess.run(["rm", "-rf", OUT_DIR], check=False)
subprocess.run(["mkdir", "-p", OUT_DIR], check=False)

c = BridgeClient(port=8611)
c.connect()

# Grab a caption to overlay describing the pipeline state
caption = {"text": "cascade | Isaac Sim PhysX | reBot RS + Qwen3.6-27B + GraspGenX 6-DOF"}


def _label(img, lines):
    img = img.copy()
    y = 28
    for ln in lines:
        cv2.putText(img, ln, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, ln, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 30
    return img


n = FPS * DUR
for i in range(n):
    t0 = time.time()
    bgr0, _, _ = c.frame("cam0")
    try:
        bgrs, _, _ = c.frame("side")
    except Exception:
        bgrs = bgr0
    try:
        bgrw, _, _ = c.frame("wrist")
    except Exception:
        bgrw = bgr0
    h = 480
    def _rs(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
    left = _rs(bgr0)
    mid = _rs(bgrs)
    right = _rs(bgrw)
    combo = np.hstack([left, mid, right])
    combo = _label(combo, [caption["text"],
                           f"frame {i+1}/{n}  cameras: cam0 | side | wrist"])
    cv2.imwrite(f"{OUT_DIR}/f{i:04d}.jpg", combo)
    dt = 1.0 / FPS - (time.time() - t0)
    if dt > 0:
        time.sleep(dt)

c.close()

# mux (pad to even dimensions -- libx264/yuv420p needs width & height even)
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(FPS), "-i", f"{OUT_DIR}/f%04d.jpg",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
    "/tmp/wrc_graspgenx_demo.mp4",
], check=True)
print("WROTE /tmp/wrc_graspgenx_demo.mp4")
