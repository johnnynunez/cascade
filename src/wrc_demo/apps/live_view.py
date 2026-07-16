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


class FrameHub:
    """Camera wrapper: background pump + optional live window."""

    def __init__(self, camera, title: str = "wrc-demo", show: bool = True,
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
