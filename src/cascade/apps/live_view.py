"""FrameHub: continuous camera streaming + always-on visualization window.

Wraps any CameraBase. A background thread grabs frames at camera rate and
(optionally) renders them in an OpenCV window with detection overlays and a
status line, while the skill runtime keeps its normal `get_frame()` calls --
served fresh from the same stream, so the camera is never double-opened.

The runtime feeds annotations back via `set_overlay(...)` (duck-typed: it
only calls it when the camera object has the method), so the window shows
what the agent currently sees and does. All GUI calls stay inside the hub
thread; if the display is unavailable the hub silently degrades to a
headless frame pump.
"""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

from ..types import Frame


def depth_colormap(depth_m: np.ndarray, max_m: float = 2.0) -> np.ndarray:
    d = np.clip(depth_m, 0, max_m) / max_m
    img = cv2.applyColorMap((255 - d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_m <= 0] = 0
    return img


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 26
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (60, 255, 60), 1, cv2.LINE_AA)
        y += 26


def draw_detections(img: np.ndarray, dets) -> None:
    for d in dets:
        x0, y0, x1, y1 = d.bbox.astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(img, f"{d.label} {d.conf:.2f}", (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
        if d.mask is not None and d.mask.shape == img.shape[:2]:
            overlay = img.copy()
            overlay[d.mask] = (0, 200, 255)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)


def _stack_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    """Compose per-camera tiles into one panel.

    Tiles are stacked VERTICALLY (one row per camera) because an RGB+depth
    tile is already ~2x as wide as the sensor: hstacking three of those
    yields a ~7700 px panel that no screen can show without shrinking each
    camera into illegibility. Rows are padded to the widest tile so a mixed
    rig (RGB+depth next to RGB-only) still stacks.
    """
    if len(tiles) == 1:
        return tiles[0]
    width = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] < width:
            pad = np.zeros((t.shape[0], width - t.shape[1], 3), dtype=t.dtype)
            t = np.hstack([t, pad])
        padded.append(t)
    return np.vstack(padded)


def _opencv_has_gui() -> bool:
    """True when this cv2 build can open windows (not the -headless wheel)."""
    try:
        info = cv2.getBuildInformation()
    except Exception:  # noqa: BLE001
        return False
    for line in info.splitlines():
        t = line.strip()
        if t.startswith("GUI:"):
            val = t.split(":", 1)[1].strip().upper()
            return bool(val) and val not in ("NONE", "NO")
    return False


def _cv2_window_unavailable_reason() -> str | None:
    """Why cv2.imshow cannot run from THIS thread, or None when it can.

    - `opencv-python-headless` (the base dependency) has no highgui at all;
    - on macOS, Cocoa windows may only be created on the MAIN thread, and
      the RigViewer runs on a worker (under the MCP server the main thread
      is the JSON-RPC loop) -> highgui throws an opaque C++ exception."""
    if not _opencv_has_gui():
        return ("this venv has opencv-python-headless (no highgui); install "
                "`opencv-python` in its place for a native camera window")
    import platform

    # The viewer ALWAYS paints from its own worker thread (see RigViewer._loop),
    # whichever thread calls start(), so on macOS this is unconditional.
    if platform.system() == "Darwin":
        return ("macOS Cocoa windows must be created on the main thread and the "
                "camera viewer paints from a worker (the demo CLI and the MCP server "
                "both keep the main thread for control)")
    return None


class RigViewer:
    """cv2 window over a CameraRig: all streams side by side, annotated.

    Each tile is RGB (with detections) beside its depth colormap, matching
    what FrameHub shows for a single camera -- a depth-capable sensor whose
    depth is invisible in the live window is the one failure mode nobody
    notices until the grasp is already wrong. RGB-only cameras render as a
    single pane, so a mixed rig (D455F + a plain UVC) still tiles cleanly.

    Render-only -- frame pumping lives in each CameraStream. Degrades to a
    silent no-op when no display is available, exactly like FrameHub."""

    def __init__(self, rig, title: str = "cascade :: live", scale: float = 0.7,
                 rate_hz: float = 20.0, tile_h: int = 480,
                 show_depth: bool = True, depth_max_m: float = 2.0):
        self._rig = rig
        self._title = title
        self._scale = scale
        self._period = 1.0 / rate_hz
        self._tile_h = tile_h
        self._show_depth = show_depth
        self._depth_max_m = float(depth_max_m)
        self._stop = False
        self._thread: threading.Thread | None = None
        self._gui_ok = True

    def start(self) -> None:
        import os

        if not os.environ.get("DISPLAY") or os.environ.get("CASCADE_VIEW") == "0":
            self._gui_ok = False
            return
        reason = _cv2_window_unavailable_reason()
        if reason:
            # Two structural cases, both of which used to surface as
            # "[rig-viewer] viewer disabled: Unknown C++ exception from OpenCV
            # code" in every run log -- a crash-shaped line for a known
            # limitation. Name it once, up front.
            self._gui_ok = False
            print(f"[rig-viewer] camera window skipped: {reason}. The MuJoCo physics "
                  "window and the browser dashboard (live_view_url) are unaffected.",
                  file=sys.stderr)
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rig-viewer")
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _render_tile(self, stream) -> np.ndarray | None:
        frame = stream.latest()
        if frame is None:
            return None
        dets, status = stream.overlay()
        img = frame.rgb.copy()
        draw_detections(img, dets)
        h, w = img.shape[:2]
        depth_txt = f"depth: {frame.depth_source}"
        if frame.has_depth:
            valid = frame.depth_m > 0
            frac = float(valid.mean())
            med = float(np.median(frame.depth_m[valid])) if frac > 0 else 0.0
            depth_txt = (f"depth: {frame.depth_source}  {frac * 100:4.1f}% valid"
                         f"  med {med:.2f} m")
        draw_hud(img, [
            f"{stream.name}  {w}x{h}  {stream.fps:4.1f} fps",
            depth_txt,
            f"agent: {status}",
        ])
        if self._show_depth and frame.has_depth:
            dimg = depth_colormap(frame.depth_m, max_m=self._depth_max_m)
            draw_hud(dimg, [f"{stream.name} depth  0-{self._depth_max_m:.1f} m"])
            img = np.hstack([img, dimg])
            h, w = img.shape[:2]
        if h != self._tile_h:
            img = cv2.resize(img, (int(w * self._tile_h / h), self._tile_h))
        return img

    def _loop(self) -> None:
        window_up = False
        while not self._stop:
            t0 = time.monotonic()
            tiles = [t for s in self._rig if (t := self._render_tile(s)) is not None]
            if tiles and self._gui_ok:
                try:
                    panel = _stack_tiles(tiles)
                    if self._scale != 1.0:
                        panel = cv2.resize(panel, None, fx=self._scale, fy=self._scale)
                    if not window_up:
                        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
                        window_up = True
                    cv2.imshow(self._title, panel)
                    cv2.waitKey(1)
                except Exception as e:
                    self._gui_ok = False
                    print(f"[rig-viewer] viewer disabled: {e}", file=sys.stderr)
                    return
            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)
        if window_up:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:
                pass


class FrameHub:
    """Camera wrapper: background pump + optional live window.

    Superseded by perception.stream.CameraStream + RigViewer for the rig
    path; kept for single-camera embedding and back-compat."""

    def __init__(self, camera, title: str = "cascade", show: bool = True,
                 scale: float = 0.7, rate_hz: float = 30.0):
        self._camera = camera
        self._title = title
        self._show = show
        self._scale = scale
        self._period = 1.0 / rate_hz
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._latest_seq = 0
        self._overlay_dets: list = []
        self._overlay_status = "starting..."
        self._stop = False
        self._thread: threading.Thread | None = None
        self._gui_ok = show

    # ── CameraBase-compatible surface ────────────────────────────────────

    @property
    def has_depth(self) -> bool:
        return self._camera.has_depth

    def open(self) -> None:
        self._camera.open()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="framehub")
        self._thread.start()

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._camera.close()

    def warm_up(self, n: int = 10) -> None:
        deadline = time.monotonic() + 10.0
        with self._cond:
            while self._latest_seq < n and time.monotonic() < deadline:
                self._cond.wait(timeout=0.2)

    def get_frame(self, timeout_s: float = 3.0) -> Frame:
        """Return a frame captured AFTER this call (fresh observation)."""
        with self._cond:
            want = self._latest_seq + 1
            deadline = time.monotonic() + timeout_s
            while self._latest_seq < want:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self._latest is not None:
                        return self._latest  # degrade: latest known frame
                    raise RuntimeError("frame hub produced no frames")
                self._cond.wait(timeout=remaining)
            return self._latest

    # ── annotations from the runtime ─────────────────────────────────────

    def set_overlay(self, detections=None, status: str | None = None) -> None:
        with self._cond:
            if detections is not None:
                self._overlay_dets = list(detections)
            if status is not None:
                self._overlay_status = status

    # ── pump/render loop (owns ALL GUI calls) ────────────────────────────

    def _loop(self) -> None:
        fps, t_prev = 0.0, time.monotonic()
        window_up = False
        while not self._stop:
            t0 = time.monotonic()
            try:
                frame = self._camera.get_frame()
            except Exception as e:
                print(f"[framehub] frame grab failed: {e}", file=sys.stderr)
                time.sleep(0.2)
                continue
            with self._cond:
                self._latest = frame
                self._latest_seq += 1
                dets = list(self._overlay_dets)
                status = self._overlay_status
                self._cond.notify_all()

            if self._gui_ok:
                try:
                    now = time.monotonic()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                    t_prev = now
                    rgb = frame.rgb.copy()
                    draw_detections(rgb, dets)
                    h, w = rgb.shape[:2]
                    draw_hud(rgb, [
                        f"{w}x{h}  {fps:4.1f} fps  depth: {frame.depth_source}",
                        f"agent: {status}",
                    ])
                    panel = (
                        np.hstack([rgb, depth_colormap(frame.depth_m)])
                        if frame.has_depth else rgb
                    )
                    if self._scale != 1.0:
                        panel = cv2.resize(panel, None, fx=self._scale, fy=self._scale)
                    if not window_up:
                        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
                        window_up = True
                    cv2.imshow(self._title, panel)
                    cv2.waitKey(1)
                except Exception as e:
                    self._gui_ok = False  # headless / display gone: keep pumping
                    print(f"[framehub] viewer disabled: {e}", file=sys.stderr)

            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                time.sleep(self._period - elapsed)
        if window_up:
            try:
                cv2.destroyWindow(self._title)
                cv2.waitKey(1)
            except Exception:
                pass
