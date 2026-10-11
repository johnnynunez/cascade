#!/usr/bin/env python3
"""Record what nvblox sees next to what the cameras see (read-only).

    scripts/record_live_map.py --seconds 60 --out runs/videos/live_map.mp4
    scripts/record_live_map.py --cameras d455f_scene,d455f_wrist --fps 10

Runs beside a cascade MCP server (`scripts/openshell_rig.py serve-mcp`,
`scripts/nemoclaw_mcp.py serve`, `scripts/launch.sh`) and WATCHES it:

- cameras: the live-view server's annotated MJPEG streams
  (http://127.0.0.1:8090/stream/<camera>), never the devices;
- the map: the occupancy bridge (127.0.0.1:5557) through
  cascade.apps.live_map_video.ReadOnlyBridge, which can only `probe` and
  `query` (it raises on `clear`/`integrate_*`);
- status: GET /state (objects, what is held, the task).

The map shows what the SERVER integrated. With CASCADE_OCCUPANCY=0 the
bridge holds nothing new, and the video says so (voxel count and map age).

Output: H.264 MP4 (yuv420p), a 2 x 2 grid of 640 x 360 tiles: the cameras
on top; nvblox's occupied voxels (a slow orbit, the safety workspace in
green) and a top-down height map with the agent's objects below. Stop with
Ctrl-C/SIGTERM; the file is finalized either way.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.apps import live_map_video as lmv

STALE_S = 1.5
TILE = (640, 360)


class Latest:
    """The newest value a background reader produced, with its arrival time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.value, self.t, self.note = None, 0.0, "connecting"

    def set(self, value, note=""):
        with self._lock:
            self.value, self.t, self.note = value, time.monotonic(), note

    def fail(self, note):
        with self._lock:
            self.note = note

    def get(self):
        with self._lock:
            return self.value, time.monotonic() - self.t, self.note


def read_camera(url: str, out: Latest, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                for jpeg in lmv.iter_mjpeg(resp):
                    if stop.is_set():
                        return
                    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        out.set(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception as exc:  # noqa: BLE001 - a dropped stream is shown as missing, then retried
            out.fail(f"stream error: {type(exc).__name__}")
        stop.wait(1.0)


def read_map(bridge: lmv.ReadOnlyBridge, lo, hi, period: float, out: Latest, stop: threading.Event) -> None:
    while not stop.is_set():
        t0 = time.monotonic()
        try:
            pts = bridge.occupied_points(lo, hi)
            out.set(pts, note=f"{1000 * (time.monotonic() - t0):.0f} ms")
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 - shown on the frame, retried
            out.fail(f"bridge: {type(exc).__name__}")
        stop.wait(max(0.0, period - (time.monotonic() - t0)))


def read_state(url: str, out: Latest, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                out.set(json.loads(resp.read()))
        except Exception as exc:  # noqa: BLE001
            out.fail(f"state: {type(exc).__name__}")
        stop.wait(1.0)


def visible_objects(state) -> list:
    objs = (state or {}).get("objects") or []
    return [o for o in objs if isinstance(o, dict) and o.get("state") == "visible"][:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", default="http://127.0.0.1:8090", help="cascade live-view server")
    ap.add_argument("--bridge", default="127.0.0.1:5557", help="occupancy bridge host:port (read-only use)")
    ap.add_argument("--cameras", default="d455f_scene,d455f_wrist")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--map-period", type=float, default=0.25, help="seconds between map queries")
    ap.add_argument("--region-min", type=float, nargs=3, default=[-0.15, -0.5, -0.06])
    ap.add_argument("--region-max", type=float, nargs=3, default=[0.75, 0.5, 0.6])
    ap.add_argument("--ws-min", type=float, nargs=3, default=[0.1, -0.3, -0.01], help="safety workspace (drawn)")
    ap.add_argument("--ws-max", type=float, nargs=3, default=[0.5, 0.3, 0.55])
    ap.add_argument("--out", default=None, help="MP4 path (default runs/videos/live_map_<time>.mp4)")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    args = ap.parse_args(argv)

    from cascade.perception.occupancy import OccupancyClient

    host, port = args.bridge.rsplit(":", 1)
    bridge = lmv.ReadOnlyBridge(OccupancyClient(host, int(port), timeout_ms=2000))
    out_path = Path(args.out or REPO / "runs" / "videos" / f"live_map_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cams = [c for c in args.cameras.split(",") if c][:2]
    cam_latest = {c: Latest() for c in cams}
    map_latest, state_latest = Latest(), Latest()
    threads = [threading.Thread(target=read_camera, args=(f"{args.live}/stream/{c}", cam_latest[c], stop),
                                daemon=True) for c in cams]
    threads.append(threading.Thread(target=read_map, args=(bridge, args.region_min, args.region_max,
                                                           args.map_period, map_latest, stop), daemon=True))
    threads.append(threading.Thread(target=read_state, args=(f"{args.live}/state", state_latest, stop), daemon=True))
    for t in threads:
        t.start()

    W, H = 2 * TILE[0], 2 * TILE[1]
    ff = subprocess.Popen([args.ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out_path)],
                          stdin=subprocess.PIPE)
    t_start = time.monotonic()
    period = 1.0 / args.fps
    n = 0
    try:
        while not stop.is_set() and time.monotonic() - t_start < args.seconds:
            t_frame = t_start + n * period
            now = time.monotonic()
            if t_frame > now:
                time.sleep(t_frame - now)
            t = time.monotonic() - t_start
            panels, titles = [], []
            for c in cams:
                img, age, note = cam_latest[c].get()
                if img is None or age > STALE_S:
                    panels.append(lmv.camera_panel(None, TILE, c, note if img is None else f"STALE {age:.1f} s"))
                else:
                    panels.append(lmv.camera_panel(img, TILE, c))
                titles.append(f"camera: {c}")
            while len(panels) < 2:
                panels.append(lmv.camera_panel(None, TILE, "(no camera)"))
                titles.append("")
            pts, map_age, map_note = map_latest.get()
            pts = np.zeros((0, 3), np.float32) if pts is None else pts
            view = lmv.orbit_view(t, size=TILE)
            m3d = lmv.render_points(pts, view)
            lmv.draw_box(m3d, view, args.ws_min, args.ws_max)
            map_ok = map_age <= 2.0 and map_latest.value is not None
            label = (f"{len(pts)} occupied voxels · query {map_note}" if map_ok
                     else f"map: {map_note if map_latest.value is None else f'STALE {map_age:.1f} s'}")
            cv2.putText(m3d, label, (10, TILE[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230) if map_ok else (255, 90, 90), 1, cv2.LINE_AA)
            state, _, _ = state_latest.get()
            td = lmv.topdown(pts, args.ws_min, args.ws_max, visible_objects(state), size=TILE)
            held = (state or {}).get("holding")
            cv2.putText(td, f"t {t:5.1f} s   holding: {held or '-'}", (10, TILE[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
            frame = lmv.compose(panels + [m3d, td], titles + ["nvblox: occupied voxels (3-D)",
                                                              "nvblox: top-down height + objects"], tile=TILE)
            ff.stdin.write(frame.tobytes())
            n += 1
    finally:
        stop.set()
        ff.stdin.close()
        ff.wait(timeout=60)
    print(f"wrote {n} frames ({n / args.fps:.1f} s) -> {out_path}")
    return 0 if ff.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
