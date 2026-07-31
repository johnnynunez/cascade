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
import json
import socket
import threading
import zlib

import numpy as np


class BridgeError(RuntimeError):
    pass


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8611, timeout_s: float = 10.0):
        self._addr = (host, int(port))
        self._timeout = timeout_s
        self._sock: socket.socket | None = None
        self._file = None
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
            self._file = sock.makefile("rwb")

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def request(self, payload: dict, timeout_s: float | None = None) -> dict:
        """Send one request. `timeout_s` overrides the socket timeout for
        this call only.

        Some ops legitimately take much longer than a normal round trip:
        `reset_props` under Newton does a timeline Stop -> Play (the only way
        a resting body's pose actually sticks on that engine) and needs tens
        of seconds. Timing out client-side while the bridge is mid-reset
        leaves the caller reading a half-reset scene, which is how a sweep
        ends up measuring the previous episode's end state.
        """
        with self._lock:
            if self._file is None:
                raise BridgeError("bridge not connected")
            _prev = None
            if timeout_s is not None and self._sock is not None:
                _prev = self._sock.gettimeout()
                self._sock.settimeout(timeout_s)
            try:
                self._file.write(json.dumps(payload).encode() + b"\n")
                self._file.flush()
                line = self._file.readline()
            except OSError as e:
                raise BridgeError(f"bridge I/O failed: {e}") from e
            finally:
                if _prev is not None and self._sock is not None:
                    self._sock.settimeout(_prev)
        if not line:
            raise BridgeError("bridge closed the connection")
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise BridgeError(str(resp.get("error", "bridge error")))
        return resp

    # ── typed helpers ────────────────────────────────────────────────────

    def ping(self) -> bool:
        return bool(self.request({"op": "ping"}).get("ok"))

    def frame(self, camera: str = "cam0") -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """-> (bgr uint8 HxWx3, depth float32 HxW meters or None, K 3x3)."""
        import cv2

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
        # Eye-in-hand cameras serve per-frame extrinsics (the camera moves
        # with the arm); stashed rather than returned to keep the 3-tuple
        # signature every existing caller expects.
        self.last_T_base_cam = (
            np.asarray(r["T_base_cam"], dtype=np.float64).reshape(4, 4)
            if r.get("T_base_cam") is not None else None
        )
        return bgr, depth, K

    def state(self) -> dict:
        return self.request({"op": "state"})

    def set_joints(self, q: np.ndarray) -> None:
        self.request({"op": "set_joints", "q": [float(x) for x in np.asarray(q).ravel()]})

    def gripper(self, pos: float, effort: float = 1.0) -> None:
        self.request({"op": "gripper", "pos": float(pos), "effort": float(effort)})

    def stop(self) -> None:
        self.request({"op": "stop"})
