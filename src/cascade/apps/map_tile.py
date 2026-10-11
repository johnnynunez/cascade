"""The nvblox map as one more live-view tile: the virtual camera `nvblox_map`.

The demo screen is OpenClaw in Chromium with the cascade camera extension
beside it. On the real rig the extension discovers its tiles from the live
view's `/state` (apps/stream_server.py) and streams `/stream/<name>` as MJPEG.
This module adds one more "camera" there that shows what the runtime's own
occupancy map holds -- the cloud the safety gate's clearance check reads --
so the audience sees the robot's 3-D model of the table next to the cameras.

    +-------------------------------------+-----------------+
    | slowly orbiting 3-D occupied voxels | top-down height |
    | + safety workspace (green)          | map + objects   |
    |                                     +-----------------+
    | "N occupied voxels · map 0.4 s old" | backend, voxels |
    |                                     | age, arm, clock |
    +-------------------------------------+-----------------+

It never slows or changes the map:

- it reads `OccupancyMap.view_snapshot()`: no lock (the perception thread
  holds the map's locks through a bridge round trip, the harness takes them
  at 50 Hz), no bridge traffic (that cache IS what the gate uses), and a
  read-only view of the cloud;
- it renders only when a viewer asks for a frame (`jpeg()`, called by the
  MJPEG handler of a connected client), at most `fps` times a second however
  many viewers share it; `/state` (`stats()`) never renders. No viewer, no
  cost;
- the renderer (apps/live_map_video.py, shared with
  scripts/record_live_map.py) is numpy-vectorized, so the frame is spent in
  C code that releases the GIL, not in a Python loop the motion stream
  would wait behind.

And it never shows a frozen picture. Every state that is not a live map has
its own placeholder, each carrying a clock so a live placeholder cannot be
mistaken for a frozen frame:

    off          nvblox off (CASCADE_OCCUPANCY=0)
    bridge_down  nvblox bridge not answering
    standby      map starts when the arm powers on (the arm must be masked out)
    waiting      waiting for the first masked depth frame
    stale        the last map, with its age in red

The tile is marked `{"virtual": true, "kind": "map"}` in `/state`; agent-facing
camera lists (MCP world_state, camera_snapshot, analyze, capabilities) come
from the camera rig and never contain it.
"""

from __future__ import annotations

import os
import textwrap
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import live_map_video as lmv

#: Stream name of the virtual camera (also its /stream/<name> route).
MAP_CAMERA = "nvblox_map"
#: Human label (the extension's LABELS map carries the same text).
MAP_LABEL = "3D map (nvblox)"

#: Shipped defaults: a modest rate (the map refreshes at the perception rate,
#: ~3 Hz) and a 16:9 frame that fills an extension tile.
DEFAULT_FPS = 3.0
DEFAULT_STALE_S = 2.0
DEFAULT_QUALITY = 75
DEFAULT_SIZE = (960, 540)
#: Safety workspace drawn when neither the config nor the map names one.
DEFAULT_WORKSPACE = ((0.1, -0.3, -0.01), (0.5, 0.3, 0.55))

_ON = ("1", "on", "true", "yes")
_OFF = ("0", "off", "false", "no")

TITLE = (118, 185, 0)       # RGB, the deck's green
AMBER = (240, 180, 60)
DIM = (150, 156, 160)
RULE = (52, 58, 64)


@dataclass(frozen=True)
class MapTileSettings:
    """Resolved `stream.nvblox_map` (see `map_tile_settings`)."""

    enabled: bool
    fps: float = DEFAULT_FPS
    stale_s: float = DEFAULT_STALE_S
    quality: int = DEFAULT_QUALITY
    size: tuple[int, int] = DEFAULT_SIZE
    #: Why the runtime has no occupancy map, for the "nvblox off (...)" frame.
    off_reason: str = "occupancy off"


def _switch(value, where: str) -> bool | None:
    """auto -> None, on -> True, off -> False; anything else fails loud."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in _ON:
        return True
    if text in _OFF:
        return False
    raise ValueError(f"{where}: expected auto, true or false, got {value!r}")


def _plain(value):
    data = getattr(value, "_data", value)   # cascade.config wraps nested mappings
    return dict(data) if isinstance(data, Mapping) else data


def map_tile_settings(stream_cfg, *, occupancy, occupancy_cfg=None, env=None) -> MapTileSettings:
    """Decide whether the live view announces `nvblox_map`, and how it renders.

    `stream.nvblox_map` is `auto` (default), `true`, `false`, or a mapping
    `{enabled, fps, stale_s, quality, width, height}`. `CASCADE_NVBLOX_MAP`
    (`0`/`1`/`auto`) overrides `enabled` without editing YAML.

    `auto` announces the tile exactly when this runtime HAS an occupancy map
    (`occupancy` is not None). Justification: with occupancy off there is
    nothing to show, and a tile that only ever says so would sit on every
    mock and simulator dashboard (and change the dashboard the golden tests
    pin). `true` forces it anyway; the frame then says why the map is off
    (`off_reason`), e.g. "nvblox off (CASCADE_OCCUPANCY=0)".
    """
    env = os.environ if env is None else env
    raw = _plain(stream_cfg.get("nvblox_map", "auto")) if stream_cfg is not None else "auto"
    block = raw if isinstance(raw, dict) else {"enabled": raw}
    forced = _switch(block.get("enabled", "auto"), "stream.nvblox_map")
    env_raw = str(env.get("CASCADE_NVBLOX_MAP", "") or "").strip()
    if env_raw:
        forced = _switch(env_raw, "CASCADE_NVBLOX_MAP")
    enabled = (occupancy is not None) if forced is None else forced

    occ_env = str(env.get("CASCADE_OCCUPANCY", "") or "").strip()
    occ_cfg = _plain(occupancy_cfg)
    if occ_env.lower() in _OFF:
        off_reason = f"CASCADE_OCCUPANCY={occ_env}"
    elif not isinstance(occ_cfg, dict):
        off_reason = "occupancy not configured"
    elif not occ_cfg.get("enabled", True):
        off_reason = "occupancy.enabled: false"
    else:
        off_reason = "no occupancy client: pyzmq / msgpack-numpy missing"

    def num(key, default, lo, hi, cast=float):
        return cast(min(max(float(block.get(key, default)), lo), hi))

    return MapTileSettings(
        enabled=bool(enabled),
        fps=num("fps", DEFAULT_FPS, 0.5, 10.0),
        stale_s=num("stale_s", DEFAULT_STALE_S, 0.1, 3600.0),
        quality=num("quality", DEFAULT_QUALITY, 30, 95, int),
        size=(num("width", DEFAULT_SIZE[0], 320, 1920, int), num("height", DEFAULT_SIZE[1], 180, 1080, int)),
        off_reason=off_reason,
    )


@dataclass(frozen=True)
class MapView:
    """What the tile shows right now (classification only, nothing drawn)."""

    state: str                      # off | bridge_down | standby | waiting | live | stale
    headline: str = ""
    detail: tuple[str, ...] = ()
    points: np.ndarray | None = field(default=None, repr=False)
    age_s: float | None = None
    voxels: int = 0
    caption: str = ""
    ok: bool = True
    status: dict | None = None
    arm_up: bool | None = None
    last_error: str | None = None
    region: tuple | None = None


def classify(snap: dict | None, *, occupancy_on: bool, arm_up: bool | None, stale_s: float,
             off_reason: str) -> MapView:
    """Map snapshot -> what to show. Order matters:

    1. no occupancy map in this runtime            -> off
    2. a cloud younger than `stale_s`              -> live (even in standby:
       a simulated arm is masked without powering a LazyArm)
    3. arm in standby                              -> standby (nothing is
       integrated until the arm can be masked out, by design)
    4. no probe answer / a refresh that timed out  -> bridge_down
    5. an older cloud                              -> stale (age in red)
    6. otherwise                                   -> waiting
    """
    if not occupancy_on:
        return MapView("off", f"nvblox off ({off_reason})",
                       ("this runtime runs without an occupancy map",))
    snap = snap or {}
    pts, age = snap.get("points"), snap.get("age_s")
    status, err = snap.get("status"), snap.get("last_error")
    n = 0 if pts is None else len(pts)
    region = ((snap["region_min"], snap["region_max"])
              if snap.get("region_min") and snap.get("region_max") else None)
    if pts is not None and age is not None and age <= stale_s:
        return MapView("live", points=pts, age_s=age, voxels=n, status=status, arm_up=arm_up,
                       caption=f"{n:,} occupied voxels · map {age:.1f} s old", last_error=err,
                       region=region)
    if arm_up is False:
        return MapView("standby", "map starts when the arm powers on", ("(the arm must be masked out)",),
                       age_s=age, voxels=n, status=status, arm_up=arm_up)
    if status is None or "timed out" in (err or ""):
        why = err if (err and "timed out" in err) else (snap.get("probe_error") or err or "no probe answer")
        return MapView("bridge_down", "nvblox bridge not answering", (why,), age_s=age, voxels=n,
                       arm_up=arm_up, last_error=err)
    if pts is not None and age is not None:
        return MapView("stale", points=pts, age_s=age, voxels=n, status=status, arm_up=arm_up, ok=False,
                       caption=f"{n:,} occupied voxels · map {age:.1f} s old", last_error=err,
                       region=region)
    why = snap.get("body_error") or err or "no query result yet"
    return MapView("waiting", "waiting for the first masked depth frame", (why,), status=status,
                   arm_up=arm_up, last_error=err)


def _text(img, text, org, scale, color, thickness=1):
    cv2.putText(img, text, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def _wrap(text: str, width_px: int, scale: float) -> list[str]:
    char_px = max(1.0, 19.0 * scale)        # FONT_HERSHEY_SIMPLEX averages ~19 px per char at scale 1
    return textwrap.wrap(str(text), max(8, int(width_px / char_px))) or [""]


class MapTile:
    """One virtual camera over the runtime's occupancy map (see module doc)."""

    def __init__(self, occupancy, settings: MapTileSettings, *, arm_up_fn=None, objects_fn=None,
                 workspace=None, clock=time.monotonic, wall=time.time):
        self.name = MAP_CAMERA
        self.label = MAP_LABEL
        self.settings = settings
        self._occ = occupancy
        self._arm_up_fn = arm_up_fn
        self._objects_fn = objects_fn
        self._workspace = workspace
        self._clock, self._wall = clock, wall
        self._t0 = clock()
        # One render at a time, shared by every viewer. /state (stats) never
        # takes it, so a slow render cannot delay a status poll.
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._t_frame = 0.0
        self._recent: deque[float] = deque(maxlen=8)
        #: frames rendered so far: advances only while someone watches
        self.frame_id = 0
        #: seconds the last render + JPEG encode took (diagnostics)
        self.last_render_s = 0.0
        #: why the last frame is an error frame (None after a good render)
        self.error: str | None = None

    @property
    def fps(self) -> float:
        return self.settings.fps

    # ── classification (cheap: no drawing, no bridge, no lock) ───────────

    def _arm_up(self) -> bool | None:
        if self._arm_up_fn is None:
            return None
        try:
            return bool(self._arm_up_fn())
        except Exception:  # noqa: BLE001 -- unknown is not standby
            return None

    def view(self) -> MapView:
        snap = None
        if self._occ is not None:
            try:
                snap = self._occ.view_snapshot()
            except Exception as e:  # noqa: BLE001 -- a viewer must never break the live view
                snap = {"status": None, "probe_error": f"map snapshot failed: {e}"}
        return classify(snap, occupancy_on=self._occ is not None, arm_up=self._arm_up(),
                        stale_s=self.settings.stale_s, off_reason=self.settings.off_reason)

    def stats(self) -> dict:
        """The /state entry. Never renders and never waits for a render."""
        v = self.view()
        now = self._clock()
        recent = list(self._recent)
        fps = 0.0
        if len(recent) >= 2 and now - recent[-1] <= 2.0 / self.fps:
            fps = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
        return {
            "virtual": True, "kind": "map", "label": MAP_LABEL,
            "fps": round(fps, 1), "frame_id": self.frame_id, "depth": False, "error": self.error,
            "map_state": v.state, "voxels": v.voxels,
            "map_age_s": None if v.age_s is None else round(float(v.age_s), 1),
        }

    # ── frames ───────────────────────────────────────────────────────────

    def jpeg(self) -> bytes:
        """The current frame as JPEG, rendered at most `fps` times a second;
        every viewer within one period gets the same bytes."""
        with self._lock:
            now = self._clock()
            if self._jpeg is not None and now - self._t_frame < 0.9 / self.fps:
                return self._jpeg
            t0 = time.perf_counter()
            try:
                rgb = self._render(self.view())
                self.error = None
            except Exception as e:  # noqa: BLE001 -- an error frame, never a dead stream
                self.error = f"map view failed: {type(e).__name__}: {e}"[:200]
                rgb = self._error_frame(self.error)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(self.settings.quality)])
            if not ok:  # pragma: no cover - cv2 encodes any uint8 image
                raise RuntimeError("JPEG encoding failed")
            self.last_render_s = time.perf_counter() - t0
            self._jpeg, self._t_frame = buf.tobytes(), now
            self._recent.append(now)
            self.frame_id += 1
            return self._jpeg

    def render_rgb(self) -> np.ndarray:
        """One frame (RGB uint8, H x W x 3) of the current state."""
        return self._render(self.view())

    def _render(self, v: MapView) -> np.ndarray:
        w, h = self.settings.size
        img = np.empty((h, w, 3), np.uint8)
        img[:] = lmv.BACKGROUND
        k = h / 540.0
        if v.points is None:
            self._placeholder(img, v, k)
        else:
            self._map(img, v, k)
        return img

    def _error_frame(self, message: str) -> np.ndarray:
        w, h = self.settings.size
        img = np.empty((h, w, 3), np.uint8)
        img[:] = lmv.BACKGROUND
        self._placeholder(img, MapView("error", "map view failed", (message,)), h / 540.0)
        return img

    def _workspace_box(self, v: MapView):
        """The safety workspace from the config, else the map's own region."""
        return self._workspace or v.region or DEFAULT_WORKSPACE

    def _objects(self):
        if self._objects_fn is None:
            return []
        try:
            return lmv.visible_objects(self._objects_fn())
        except Exception:  # noqa: BLE001 -- beliefs are decoration here
            return []

    def _clock_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self._wall()))

    def _map(self, img, v: MapView, k: float) -> None:
        h, w = img.shape[:2]
        side = round(w * 0.35)
        td_h = min(side, h)
        t = self._clock() - self._t0
        m3d, td = lmv.map_panels(v.points, t_s=t, workspace=self._workspace_box(v),
                                 objects=self._objects(), size=(w - side, h), caption=v.caption,
                                 caption_ok=v.ok, topdown_size=(side, td_h))
        img[:, :w - side] = m3d
        img[:td_h, w - side:] = td
        img[:, w - side] = RULE
        if td_h < h:
            img[td_h, w - side:] = RULE
        _text(img, MAP_LABEL, (12, 26 * k + 4), 0.7 * k, TITLE, max(1, round(1.4 * k)))
        # status box under the top-down view: backend, voxels, age, arm, clock
        st = v.status or {}
        backend = "bridge: ?"
        if st:
            backend = f"{st.get('backend', '?')} on {st.get('device', '?')}"
            if st.get("voxel"):
                backend += f" · {float(st['voxel']) * 100:.1f} cm"
        age = f"map age {float(v.age_s or 0.0):.1f} s" + ("" if v.ok else "  STALE")
        lines = [(backend, lmv.TEXT, 0.5), (f"{v.voxels:,} occupied voxels", lmv.TEXT, 0.5),
                 (age, lmv.TEXT if v.ok else lmv.ALERT, 0.5),
                 ({True: "arm: up", False: "arm: standby"}.get(v.arm_up, "arm: ?"), lmv.TEXT, 0.5)]
        if not v.ok and v.last_error:
            lines += [(row, lmv.ALERT, 0.42) for row in _wrap(v.last_error, side - 20, 0.42 * k)[:2]]
        x, y = w - side + 12, td_h + 24 * k
        for text, color, scale in lines:
            if y > h - 30 * k:
                break
            _text(img, text, (x, y), scale * k, color)
            y += 22 * k
        _text(img, self._clock_text(), (x, h - 10 * k), 0.45 * k, DIM)

    def _placeholder(self, img, v: MapView, k: float) -> None:
        h, w = img.shape[:2]
        _text(img, MAP_LABEL, (16, 30 * k + 4), 0.75 * k, TITLE, max(1, round(1.4 * k)))
        color = {"bridge_down": lmv.ALERT, "error": lmv.ALERT, "off": DIM}.get(v.state, AMBER)
        scale = 0.95 * k
        rows = _wrap(v.headline, w - 60, scale)
        y = h * 0.45 - (len(rows) - 1) * 20 * k
        for row in rows:
            (tw, _), _ = cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, round(2 * k)))
            _text(img, row, ((w - tw) / 2, y), scale, color, max(1, round(2 * k)))
            y += 42 * k
        for line in v.detail:
            for row in _wrap(line, w - 80, 0.55 * k)[:3]:
                (tw, _), _ = cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, 1)
                _text(img, row, ((w - tw) / 2, y), 0.55 * k, lmv.TEXT)
                y += 28 * k
        _text(img, f"live · {self._clock_text()}", (16, h - 14 * k), 0.5 * k, DIM)
