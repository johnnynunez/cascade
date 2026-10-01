"""TCP client for the Isaac Sim bridge (newline-delimited JSON).

Protocol (one JSON object per line, request -> response):
    {"op": "ping"}                          -> {"ok": true}
    {"op": "frame", "camera": "<name>"}     -> {"ok": true, "K": [[..]x3],
        "rgb_jpeg_b64": ..., "depth_z_b64": ...(zlib float32 HxW), "width",
        "height", "t"}
    {"op": "state"}                         -> {"ok": true, "q": [...],
        "dq": [...], "gripper_pos": x}
    {"op": "set_joints", "q": [...]}        -> {"ok": true}
    {"op": "gripper", "pos": x, "effort": e}-> {"ok": true}
    {"op": "stop"}                          -> {"ok": true}

The same protocol is exercised by tests with an in-process fake server, so
the client is fully covered without a running sim.
"""

from __future__ import annotations

import base64
import copy
import json
import socket
import threading
import time
import zlib

import numpy as np


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8611, timeout_s: float = 10.0):
        self._addr = (host, int(port))
        self._timeout = timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()  # one in-flight request at a time

    def connect(self) -> None:
        with self._lock:
            if self._sock is not None:
                return
            try:
                sock = socket.create_connection(self._addr, timeout=self._timeout)
            except OSError as e:
                raise BridgeError(
                    f"cannot reach the Isaac Sim bridge at {self._addr[0]}:{self._addr[1]} "
                    f"({e}); start it inside Isaac Sim: scripts/isaac_bridge.py"
                ) from e
            sock.settimeout(self._timeout)
            self._sock = sock

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def request(self, payload: dict, timeout_s: float | None = None) -> dict:
        """Send one request within one wall-time budget, including lock wait.

        Some ops legitimately take much longer than a normal round trip:
        `reset_props` settles and verifies props on the simulation's main
        thread and can need tens of seconds. Timing out client-side mid-reset
        leaves the caller reading a half-reset scene, which is how a sweep
        ends up measuring the previous episode's end state.
        """
        budget = self._timeout if timeout_s is None else timeout_s
        if type(budget) not in (int, float) or not np.isfinite(budget) or budget <= 0:
            raise BridgeError("request wall-time budget must be finite and positive")
        deadline = time.monotonic() + budget
        if not self._lock.acquire(timeout=budget):
            raise BridgeError("bridge request deadline expired waiting for transport lock")
        try:
            if self._sock is None:
                raise BridgeError("bridge not connected")
            sock = self._sock
            previous_timeout = sock.gettimeout()

            def remaining():
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("bridge request wall-time deadline expired")
                sock.settimeout(left)

            try:
                wire = json.dumps(payload).encode() + b"\n"
                remaining()
                sock.sendall(wire)
                chunks = bytearray()
                while b"\n" not in chunks:
                    remaining()  # slow-drip responses cannot renew the deadline
                    part = sock.recv(65536)
                    if not part:
                        raise OSError("bridge closed the connection")
                    chunks.extend(part)
                    if len(chunks) > 128 * 1024 * 1024:
                        raise ValueError("bridge response exceeds 128 MiB")
                remaining()
                line, extra = bytes(chunks).split(b"\n", 1)
                if extra:
                    raise ValueError("unsolicited data after bridge response")
                resp = json.loads(line)
                if not isinstance(resp, dict):
                    raise ValueError("bridge response must be an object")
                remaining()
            except (OSError, ValueError) as e:
                # A late response after timeout must never become the next
                # request's response. Reconnect explicitly after the caller
                # has handled the failed/possibly-delivered operation.
                self._close_locked()
                raise BridgeError(f"bridge I/O failed: {e}") from e
            finally:
                if self._sock is sock:
                    sock.settimeout(previous_timeout)
        finally:
            self._lock.release()
        if not resp.get("ok", False):
            raise BridgeError(str(resp.get("error", "bridge error")))
        return resp

    # ── typed helpers ────────────────────────────────────────────────────

    def ping(self) -> bool:
        return bool(self.request({"op": "ping"}).get("ok"))

    def frame(self, camera: str = "cam0") -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """-> (bgr uint8 HxWx3, depth float32 HxW meters or None, K 3x3)."""
        frame = self.observation(camera)
        # Legacy tuple API; new frame consumers must use observation() so
        # per-frame data cannot race through a mutable `last_*` side channel.
        self.last_T_base_cam = frame.T_base_cam
        return frame.rgb, frame.depth_m, frame.K

    def observation(self, camera: str = "cam0"):
        """One atomic Frame, including producer state and local source binding.

        Missing/malformed proprioception is preserved for the safety consumer
        to reject; never fill it using a later `state` request. Frame.t is
        client receipt time, capture.t uses the explicitly named remote clock.
        """
        import cv2
        from ..types import Frame

        r = self.request({"op": "frame", "camera": camera})
        jpg = np.frombuffer(base64.b64decode(r["rgb_jpeg_b64"]), dtype=np.uint8)
        bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if bgr is None:
            raise BridgeError("bridge sent an undecodable rgb frame")
        depth = None
        if r.get("depth_z_b64"):
            raw = zlib.decompress(base64.b64decode(r["depth_z_b64"]))
            depth = np.frombuffer(raw, dtype=np.float32).reshape(r["height"], r["width"]).copy()
        K = np.asarray(r["K"], dtype=np.float64).reshape(3, 3)
        T_base_cam = (
            np.asarray(r["T_base_cam"], dtype=np.float64).reshape(4, 4)
            if r.get("T_base_cam") is not None else None
        )
        robot_mask = None
        if r.get("robot_pixel_mask") is not None:
            try:
                m = r["robot_pixel_mask"]
                state = r.get("proprioception") or {}
                if (m.get("version") != 1 or m.get("encoding") != "zlib-u8-base64"
                        or not m.get("robot_id") or m["robot_id"] != state.get("robot_id")
                        or m.get("t") != r.get("t") or m.get("t") != state.get("t")
                        or m.get("shape") != list(bgr.shape[:2])):
                    raise ValueError("render self-mask identity/shape/clock mismatch")
                mask_bytes = zlib.decompress(base64.b64decode(m["data"], validate=True))
                mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(bgr.shape[:2])
                if np.any(mask > 1):
                    raise ValueError("render self-mask is not binary")
                robot_mask = mask.astype(bool)
            except Exception as exc:
                raise BridgeError(f"invalid render self-mask: {exc}") from exc
        return Frame(
            rgb=bgr, depth_m=depth, K=K,
            depth_source="sensor" if depth is not None else "none",
            T_base_cam=T_base_cam,
            robot_mask=robot_mask,
            capture={"backend": "isaac", "source": self._addr, "camera": camera,
                     "t": r.get("t"), "proprioception": copy.deepcopy(r.get("proprioception")),
                     "render_reference": copy.deepcopy(r.get("render_reference")),
                     "contact_paths": copy.deepcopy((r.get("robot_pixel_mask") or {}).get("contact_paths", []))},
        )

    def state(self, *, timeout_s: float | None = None) -> dict:
        return self.request({"op": "state"}, timeout_s=timeout_s)

    def set_joints(self, q: np.ndarray, *, timeout_s: float | None = None) -> None:
        self.request({"op": "set_joints", "q": [float(x) for x in np.asarray(q).ravel()]},
                     timeout_s=timeout_s)

    def gripper(self, pos: float, effort: float = 1.0) -> None:
        self.request({"op": "gripper", "pos": float(pos), "effort": float(effort)})

    def stop(self) -> None:
        self.request({"op": "stop"})
