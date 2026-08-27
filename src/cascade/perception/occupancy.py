"""nvblox-backed occupancy map: is a candidate arm position too close to
something in the scene that isn't tracked as an object belief?

Mirrors `grasping/graspgenx_backend.py`'s wiring on purpose: the heavy 3D
reconstruction (nvblox) lives in its own process/venv, reachable over a
self-contained ZMQ/msgpack wire client (no nvblox/ROS import in this venv).
Any server error degrades to "no occupancy data" rather than blocking motion
(booth rule) -- SafetyHarness treats a missing/stale map exactly like the
geometry checks it already runs when constructed without one.

There is no NVIDIA-shipped ZMQ/msgpack front end for nvblox: nvblox ships as
a library (Isaac ROS node, or nvblox_torch bindings), not this wire protocol.
`scripts/serve_nvblox_bridge.py` (launched by `scripts/serve_nvblox.sh`) is a
real, working bridge that speaks it today -- an Open3D tensor voxel
accumulator that auto-picks CUDA:0 when the process can see an NVIDIA GPU
and CPU:0 otherwise (same code path either device), with a zero-dependency
numpy fallback when Open3D isn't installed. Device-agnostic by construction,
not by placeholder: this client (and everything below) is identical whether
the bridge ends up choosing CPU or GPU, and swapping in real nvblox later
means replacing that one bridge process's backend class, not this file. Wire
protocol:

    action "integrate": {points: (N,3) float32, base frame} -> {} (ack)
    action "query": {region_min: (3,), region_max: (3,)}
        -> {points: (M,3) float32} occupied voxel centers in that AABB

`OccupancyMap.refresh()` is called off the motion hot path (from
`WorldWatcher._tick`, at the perception loop's rate_hz, same as belief
fusion) -- a network round trip inside the 50 Hz `SafetyHarness.approve()`
stream would itself be a latency/availability hazard. The harness instead
reads the last-fetched occupied-point cloud with plain numpy distance
checks; past `max_age_s` the cache is treated as absent (skip the check),
the same fallback shape as the existing perception watchdog.
"""

from __future__ import annotations

import time

import numpy as np


class OccupancyError(RuntimeError):
    pass


class OccupancyClient:
    """Minimal REQ/REP msgpack wire client (mirrors GraspGenXClient)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5557, timeout_ms: int = 500):
        try:
            import msgpack  # noqa: F401
            import msgpack_numpy
            import zmq  # noqa: F401
        except ImportError as e:
            raise OccupancyError(
                f"occupancy backend needs pyzmq+msgpack-numpy in this venv ({e})"
            ) from e
        msgpack_numpy.patch()
        self._host, self._port, self._timeout = host, int(port), int(timeout_ms)
        self._sock = None

    def _connect(self):
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self._timeout)
        sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._sock = sock

    def request(self, payload: dict) -> dict:
        import msgpack
        import zmq

        if self._sock is None:
            self._connect()
        try:
            self._sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = self._sock.recv()
        except zmq.error.Again as e:
            self.close()  # REQ socket is wedged after a timeout
            raise OccupancyError(
                f"nvblox bridge at {self._host}:{self._port} timed out "
                f"({self._timeout} ms); is scripts/serve_nvblox.sh running?"
            ) from e
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise OccupancyError(f"nvblox bridge error: {resp['error']}")
        return resp

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class OccupancyMap:
    """Cached obstacle-point snapshot, refreshed off the motion hot path.

    `clearance()` is a pure-numpy nearest-neighbour distance to the cached
    cloud -- cheap enough to call from `SafetyHarness.approve()` per
    waypoint. It returns None (meaning "skip the check") whenever there is
    no fresh cache, so a dead/slow bridge degrades exactly like running
    with no occupancy map at all rather than freezing the arm.
    """

    def __init__(
        self,
        client: OccupancyClient | None = None,
        region_min: np.ndarray | None = None,
        region_max: np.ndarray | None = None,
        max_age_s: float = 10.0,
        stride: int = 8,
    ):
        self._client = client
        self._region_min = None if region_min is None else np.asarray(region_min, dtype=float)
        self._region_max = None if region_max is None else np.asarray(region_max, dtype=float)
        self.max_age_s = float(max_age_s)
        self.stride = max(int(stride), 1)
        self._occupied: np.ndarray | None = None  # (M, 3), last successful query
        self._last_refresh: float | None = None
        self.last_error: str | None = None

    @classmethod
    def from_config(cls, cfg, workspace_min=None, workspace_max=None) -> "OccupancyMap | None":
        """Build from an `occupancy:` config block. Returns None (not
        wired into the harness) unless explicitly enabled -- there is no
        nvblox bridge running by default, and an unconfigured map must not
        silently gate motion."""
        if cfg is None or not bool(cfg.get("enabled", False)):
            return None
        client = OccupancyClient(
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 5557)),
            timeout_ms=int(cfg.get("timeout_ms", 500)),
        )
        region_min = cfg.get("region_min", None)
        region_max = cfg.get("region_max", None)
        return cls(
            client=client,
            region_min=region_min if region_min is not None else workspace_min,
            region_max=region_max if region_max is not None else workspace_max,
            max_age_s=float(cfg.get("max_age_s", 10.0)),
            stride=int(cfg.get("stride", 8)),
        )

    def refresh(self, frame, T_base_cam: np.ndarray) -> None:
        """Push this frame's depth (subsampled) to the bridge, then pull
        back the occupied-point snapshot for the configured region.

        Best-effort: any failure records `last_error` and leaves the
        existing cache in place (it simply ages toward `max_age_s`).
        """
        if self._client is None:
            return
        try:
            pts_base = self._depth_to_base_points(frame, T_base_cam)
            if pts_base.shape[0]:
                self._client.request({"action": "integrate", "points": pts_base})
            region_min = self._region_min
            region_max = self._region_max
            if region_min is None or region_max is None:
                return  # nothing to query without a region of interest
            resp = self._client.request({
                "action": "query",
                "region_min": np.asarray(region_min, dtype=np.float32),
                "region_max": np.asarray(region_max, dtype=np.float32),
            })
            self._occupied = np.asarray(resp["points"], dtype=np.float32).reshape(-1, 3)
            self._last_refresh = time.monotonic()
            self.last_error = None
        except OccupancyError as e:
            self.last_error = str(e)

    def _depth_to_base_points(self, frame, T_base_cam: np.ndarray) -> np.ndarray:
        if not frame.has_depth:
            return np.empty((0, 3), dtype=np.float32)
        depth = frame.depth_m[:: self.stride, :: self.stride]
        ys, xs = np.nonzero(depth > 0)
        if ys.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        zs = depth[ys, xs]
        ys_full = ys * self.stride
        xs_full = xs * self.stride
        K = frame.K
        pts_cam = np.stack(
            [(xs_full - K[0, 2]) / K[0, 0] * zs, (ys_full - K[1, 2]) / K[1, 1] * zs, zs],
            axis=-1,
        )
        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
        return (pts_h @ T_base_cam.T)[:, :3].astype(np.float32)

    def is_stale(self) -> bool:
        return self._last_refresh is None or (time.monotonic() - self._last_refresh) > self.max_age_s

    def clearance(self, points: np.ndarray) -> np.ndarray | None:
        """Min distance from each query point to the cached occupied cloud.

        Returns None if there is no usable cache (never refreshed, stale,
        or the cloud is empty) -- callers must treat that as "no data",
        not "clear".
        """
        if self.is_stale() or self._occupied is None or self._occupied.shape[0] == 0:
            return None
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        d = np.linalg.norm(pts[:, None, :] - self._occupied[None, :, :], axis=-1)
        return d.min(axis=1)
