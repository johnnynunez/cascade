"""WorldWatcher: the always-on perception loop behind the livestream.

A background thread round-robins over every rig camera that can lift pixels
to 3D (depth + extrinsics), runs the detector on the newest frame, names each
detection's color, and fuses everything into the shared BeliefStore. The
result: when a command like "pick and place pink object" arrives, the world
model is already warm -- resolution against beliefs is instantaneous instead
of waiting on an observe->detect round trip.

It also:
- pushes detection overlays to each stream (the MJPEG viewers show them);
- heartbeats the safety harness on EVERY processed frame (fresh perception
  = motion allowed) -- an empty table must not starve the watchdog, and the
  loop keeps ticking during arm motion precisely because multi-move skills
  outlast `watchdog_s` (adversarial review finding, 2026-07-18);
- while "paused" (arm in motion) it keeps detecting and heartbeating but
  SKIPS belief fusion, so the held object is not re-registered at a bogus
  mid-air position (ignore_label additionally covers it by detector label).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .colors import detection_color
from .freshness import capture_marker, frames_after_reset, newer_capture
from .grounding import Extrinsics, mask_to_points_cam, oriented_bbox
from .thread_join import cancel_stop_before_exit, join_thread, stop_before_exit
from .workspace import WorkspaceFilter
from ..types import Frame, transform_points


class LockedDetector:
    """Serialize detector access across threads (YOLO predict is not
    re-entrant); shared by the watcher and the skill runtime."""

    def __init__(self, detector):
        self._detector = detector
        self._lock = threading.Lock()

    def detect(self, frame, classes=None):
        with self._lock:
            return self._detector.detect(frame, classes=classes)

    def set_classes(self, classes) -> None:
        with self._lock:
            self._detector.set_classes(classes)


@dataclass
class WatchedCamera:
    stream: object  # CameraStream
    depth: object  # DepthProvider
    extrinsics: Extrinsics
    last_seq: int = -1
    fuse: bool = True  # False: overlay/heartbeat only (no 3D fusion), e.g.
    # an audience camera with placeholder extrinsics
    last_error: str | None = None
    fusion_floor: Frame | None = None
    reset_pending: bool = False
    last_capture: dict | None = None
    map_depth: bool | None = None  # None follows fuse; explicitly independent of semantic beliefs

    @property
    def maps_depth(self):
        return self.fuse if self.map_depth is None else self.map_depth


class WorldWatcher:
    """Always-on perception: keeps the BeliefStore warm.

    `classes=None` (the default) scans open-world, which is what a public
    booth needs: a visitor's object must land in the world model without
    anyone having named it in advance. Passing an explicit list makes this a
    CLOSED set -- anything not on it is invisible to the whole system.
    """

    def __init__(
        self,
        cameras: list[WatchedCamera],
        detector,
        beliefs,
        classes: list[str] | None = None,
        rate_hz: float = 3.0,
        harness=None,
        workspace: "WorkspaceFilter | None" = None,
        occupancy=None,
    ):
        self._cams = cameras
        self._detector = detector
        self._beliefs = beliefs
        self._classes = list(classes) if classes else None
        self._workspace = workspace or WorkspaceFilter()
        self._period = 1.0 / max(rate_hz, 0.1)
        self._harness = harness
        # OccupancyMap | None -- refreshed here (perception rate_hz), never
        # from the 50 Hz motion stream; see perception/occupancy.py.
        self._occupancy = occupancy
        self._stop = False
        self._pause_count = 0
        self._pause_lock = threading.Lock()
        self._fusion_epoch = 0
        self._thread: threading.Thread | None = None
        self._exit_hook = None
        self._wake = threading.Event()
        self._ignore: set[str] = set()
        self.ticks = 0
        self.last_dets: dict[str, list] = {}
        self.last_update_t: float | None = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop = False
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="world-watcher")
        self._thread.start()
        # Safety net for callers that never reach stop() (an exception between
        # build_runtime and shutdown_runtime, a script that just returns):
        # stop() still runs at exit, while the interpreter is intact.
        self._exit_hook = stop_before_exit(self)

    def stop(self, timeout_s: float | None = None) -> None:
        """Stop the loop and WAIT until the tick in flight has finished.

        A tick runs native code (YOLOE/torch/CUDA, cv2, the occupancy
        client). This used to be `join(timeout=5)`, which returned while the
        first inference of a cold process (> 5 s while CUDA JIT-compiles its
        kernels) was still running. The launcher's runtime check then reached
        interpreter exit, and the watcher thread aborted the process with
        "terminate called without an active exception" when it came back
        from torch (3 of 20 kitchen launches; 3 of 3 with a cold kernel
        cache; see perception/thread_join.py). Every call in a tick is
        bounded by its own I/O timeout or by compute, so waiting is the safe
        stop; join_thread logs where the thread is every 10 s meanwhile.
        `timeout_s` exists for the atexit safety net only; if it expires the
        thread handle is kept, so a later stop() can still join it.
        """
        self._stop = True
        self._wake.set()
        thread = self._thread
        if thread is not None and join_thread(thread, what="watcher", timeout_s=timeout_s):
            self._thread = None
        if self._thread is None:
            cancel_stop_before_exit(self._exit_hook)
            self._exit_hook = None

    @contextlib.contextmanager
    def paused(self):
        """Hold BELIEF FUSION while the arm moves (re-entrant). Detection,
        overlays and the watchdog heartbeat keep running: motion skills
        depend on the heartbeat staying alive longer than watchdog_s."""
        with self._pause_lock:
            self._pause_count += 1
            self._fusion_epoch += 1
        try:
            yield
        finally:
            with self._pause_lock:
                self._pause_count -= 1
                self._fusion_epoch += 1

    @property
    def is_paused(self) -> bool:
        return self._pause_count > 0

    def ignore_label(self, label: str | None) -> None:
        """Skip fusing this label (e.g. the object currently in the jaws)."""
        self._ignore = {label} if label else set()

    def reset_camera_frames(self, primary, *, timeout_s=5):
        """Fence every fusing camera while reset owns the existing pause.

        A camera that times out remains fenced. If it later recovers, its
        first delivery establishes a floor and only a newer capture can fuse.
        """
        watched = [cam for cam in self._cams if cam.fuse or cam.maps_depth or cam.stream is primary]
        cameras = [primary] + [cam.stream for cam in watched if cam.stream is not primary]
        with self._pause_lock:
            if not self._pause_count:
                raise RuntimeError("Reset camera barrier requires paused belief fusion")
            for cam in watched:
                cam.fusion_floor = None
                cam.reset_pending = True

        def floor_ready(stream, frame):
            with self._pause_lock:
                for cam in watched:
                    if cam.stream is stream:
                        cam.fusion_floor = frame

        resetting_map = getattr(self._occupancy, "begin_scene_reset", lambda: False)()
        observed = frames_after_reset(cameras, timeout_s=timeout_s, on_floor=floor_ready)
        if resetting_map:
            self._occupancy.finish_scene_reset([floor for _, floor, _ in observed])
            for stream, _, fresh in observed:
                cam = next((c for c in watched if c.stream is stream and c.maps_depth), None)
                if cam is not None:
                    fresh = cam.depth.ensure_depth(fresh)
                    T = fresh.T_base_cam if fresh.T_base_cam is not None else cam.extrinsics.cam_to_base()
                    self._occupancy.refresh(fresh, T)
        with self._pause_lock:
            for cam in watched:
                cam.reset_pending = False
        return observed

    def _fusion_allowed(self, cam, frame, epoch):
        """Called under _pause_lock, including immediately before each commit."""
        if self._pause_count or epoch != self._fusion_epoch:
            return False
        if cam.reset_pending and cam.fusion_floor is None:
            cam.fusion_floor = frame
            return False
        if cam.fusion_floor is not None and not newer_capture(frame, cam.fusion_floor):
            return False
        cam.reset_pending = False
        return True

    # ── the loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop:
            t0 = time.monotonic()
            for cam in self._cams:
                if self._stop:
                    break
                try:
                    self._tick(cam)
                    cam.last_error = None
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    if msg != cam.last_error:  # log state changes, not 3 Hz spam
                        print(f"[watcher:{cam.stream.name}] {msg}", file=sys.stderr)
                    cam.last_error = msg
            elapsed = time.monotonic() - t0
            if elapsed < self._period:
                # Interruptible: stop() wakes this at once, so stopping an
                # idle watcher costs no inter-tick sleep.
                self._wake.wait(self._period - elapsed)

    def _tick(self, cam: WatchedCamera) -> None:
        with self._pause_lock:
            epoch = self._fusion_epoch
        frame = cam.stream.latest()
        if frame is None:
            return
        # Local frame IDs count deliveries. An Isaac bridge can deliver the
        # same rendered capture at the client's much faster polling rate.
        if frame.frame_id == cam.last_seq:
            return
        cam.last_seq = frame.frame_id
        if (getattr(frame, "capture", None) or {}).get("backend") == "isaac":
            marker = capture_marker(frame)
            if marker == cam.last_capture:
                return
            cam.last_capture = marker
        else:
            cam.last_capture = None
        frame = cam.depth.ensure_depth(frame)
        T = None
        if frame.has_depth and (cam.fuse or cam.maps_depth):
            # Mask geometry before waiting for semantic inference/its lock.
            # Otherwise the robot can move during detection and a current
            # joint pose masks an OLD image, permanently fusing self ghosts.
            T = (frame.T_base_cam if frame.T_base_cam is not None
                 else cam.extrinsics.cam_to_base())
            if self._occupancy is not None and cam.maps_depth:
                # Keep geometry fresh during motion; only beliefs are paused.
                self._occupancy.refresh(frame, T)
        dets = self._detector.detect(frame, classes=self._classes)
        cam.stream.set_overlay(detections=dets)
        self.last_dets[cam.stream.name] = dets
        self.ticks += 1
        # Perception is alive: a processed frame is a heartbeat even with an
        # empty table or the arm mid-motion (motion skills outlast the
        # watchdog window and rely on this).
        if self._harness is not None:
            self._harness.heartbeat()
        if T is None or not cam.fuse:
            return
        with self._pause_lock:
            if not self._fusion_allowed(cam, frame, epoch):
                return
        fused = 0
        for d in dets:
            if d.label in self._ignore:
                continue
            mask = d.mask
            if mask is None:
                h, w = frame.rgb.shape[:2]
                from .cuda_math import enabled, bbox_mask
                if enabled():
                    mask = bbox_mask((h, w), d.bbox)
                else:
                    mask = np.zeros((h, w), dtype=bool)
                    x0, y0, x1, y1 = d.bbox.astype(int)
                    mask[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < 10:
                continue
            pts_base = transform_points(T, pts_cam)
            center, extents, _ = oriented_bbox(pts_base)
            why = self._workspace.reject(
                center, extents,
                mask_frac=float(mask.sum()) / float(mask.numel() if hasattr(mask, "numel") else mask.size),
            )
            if why is not None:
                continue  # scenery, the robot itself, or out of reach
            color = detection_color(frame.rgb, d)
            with self._pause_lock:
                if not self._fusion_allowed(cam, frame, epoch):
                    return
                self._beliefs.update(
                    d.label, center, d.conf, extent=extents,
                    top_z=float(pts_base[:, 2].max()), t=frame.t,
                    color=color,
                    points=pts_base if d.mask is not None else None,
                )
            fused += 1
        if fused:
            self.last_update_t = time.monotonic()

    def stats(self) -> dict:
        return {
            "ticks": self.ticks,
            "paused": self.is_paused,
            "cameras": [c.stream.name for c in self._cams],
            "last_update_s_ago": (
                round(time.monotonic() - self.last_update_t, 1)
                if self.last_update_t is not None else None
            ),
        }
