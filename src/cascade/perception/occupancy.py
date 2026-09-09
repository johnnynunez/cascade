"""Occupancy / clearance map: is a candidate arm position too close to
something in the scene that isn't tracked as an object belief?

The heavy 3-D reconstruction lives in its own process, reachable over a
self-contained ZMQ/msgpack wire client (no nvblox / warp / torch import in
this venv): `scripts/serve_occupancy_bridge.py`, launched by
`scripts/serve_occupancy.sh` and by `scripts/launch.sh`. Its backends
(`perception/occupancy_backends.py`):

    nvblox   real nvblox (nvblox_torch) TSDF + ESDF -- P0 on NVIDIA GPUs
    warp     dense projective TSDF with free-space carving + exact EDT in
             Warp kernels; CPU on macOS/Linux/aarch64, CUDA on Jetson/x86
    voxel    numpy occupancy hash (no carving, no distance field)

The bridge answers `query` with a DISTANCE GRID over the requested region:
distance to the nearest known obstacle at every voxel centre (0 inside an
obstacle, inf where nothing is known). `clearance(points)` is then a
trilinear read of that cached grid -- identical whichever backend built it,
and O(points) instead of O(points x obstacle cloud). `points` (occupied voxel
centres) still ride along for consumers that want a cloud.

`refresh()` ships the DEPTH FRAME + K + T_base_cam (`integrate_depth`) so
the backend can ray-cast: that is what lets a removed object disappear from
the map (carving). It runs off the motion hot path (WorldWatcher._tick, at
the perception rate); SafetyHarness.approve() at 50 Hz only ever reads the
cache. Past `max_age_s` the cache is treated as absent (skip the check),
the same fallback shape as the perception watchdog.

WHAT IS AND IS NOT DEGRADED. A bridge that is down degrades to "no
occupancy check" -- the arm keeps every geometric gate it always had -- but
it must never degrade SILENTLY: `probe()` at startup records which backend
answered (or that none did) in `status`, the demo prints it and writes it
to the run summary, and the launcher refuses `--require-occupancy` runs
without a live bridge. A default that nobody runs looks exactly like one
that works; the status line is what makes the difference visible.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


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

    def request(self, payload: dict, timeout_ms: int | None = None) -> dict:
        import msgpack
        import zmq

        if self._sock is None:
            self._connect()
        sock = self._sock
        if timeout_ms is not None:
            sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
            sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        try:
            sock.send(msgpack.packb(payload, use_bin_type=True))
            raw = sock.recv()
        except zmq.error.Again as e:
            self.close()  # REQ socket is wedged after a timeout
            raise OccupancyError(
                f"occupancy bridge at {self._host}:{self._port} timed out "
                f"({timeout_ms if timeout_ms is not None else self._timeout} ms); "
                f"is scripts/serve_occupancy.sh running?"
            ) from e
        finally:
            if timeout_ms is not None and self._sock is not None:
                self._sock.setsockopt(zmq.RCVTIMEO, self._timeout)
                self._sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise OccupancyError(f"occupancy bridge error: {resp['error']}")
        return resp

    def probe(self, timeout_ms: int = 300) -> dict:
        """Ask the bridge what it is. Raises OccupancyError when nothing
        answers -- a SHORT timeout on purpose: this runs at startup, once."""
        resp = self.request({"action": "probe"}, timeout_ms=timeout_ms)
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise OccupancyError(f"occupancy bridge probe returned {resp!r}")
        return resp

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class OccupancyMap:
    """Cached distance grid, refreshed off the motion hot path.

    `clearance()` trilinearly reads the cached grid -- cheap enough for
    `SafetyHarness.approve()` per waypoint. It returns None (meaning "skip
    the check") whenever there is no fresh cache, so a dead/slow bridge
    degrades exactly like running with no occupancy map at all rather than
    freezing the arm.
    """

    def __init__(
        self,
        client: OccupancyClient | None = None,
        region_min: np.ndarray | None = None,
        region_max: np.ndarray | None = None,
        max_age_s: float = 10.0,
        stride: int = 8,
        depth_stride: int = 2,
    ):
        self._client = client
        self._region_min = None if region_min is None else np.asarray(region_min, dtype=float)
        self._region_max = None if region_max is None else np.asarray(region_max, dtype=float)
        self.max_age_s = float(max_age_s)
        self.stride = max(int(stride), 1)            # legacy cloud subsampling
        self.depth_stride = max(int(depth_stride), 1)  # depth frame subsampling on the wire
        self._occupied: np.ndarray | None = None      # (M, 3), last query's occupied centres
        self._grid: np.ndarray | None = None          # (nx, ny, nz) distance [m]
        self._grid_origin: np.ndarray | None = None
        self._grid_voxel: float = 0.0
        self._last_refresh: float | None = None
        self.last_error: str | None = None
        #: what answered the startup probe: {"backend": "warp", "device": ...}
        #: or None if nothing did. Read by the demo banner / run summary.
        self.status: dict | None = None
        self.probe_error: str | None = None
        self._depth_supported: bool | None = None
        #: robot-body masking (perception/robot_mask.py): callables returning
        #: (L,3) base-frame link points for each arm (or None when that arm
        #: is not up yet), and the per-arm radius. The camera sees the arm;
        #: unmasked, the robot's own links read as obstacles at 0 m from its
        #: collision proxies and the clearance gate refuses every motion.
        self._body_fns: list = []
        self._body_radii: list[float] = []
        self.last_masked_px: int = 0

    def add_robot_body(self, link_points_fn, radius_m: float = 0.06) -> None:
        """Register an arm to mask out of the depth before integration.
        `link_points_fn() -> (L,3) | None` (None = arm not available now)."""
        self._body_fns.append(link_points_fn)
        self._body_radii.append(float(radius_m))

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg, workspace_min=None, workspace_max=None) -> "OccupancyMap | None":
        """Build from an `occupancy:` config block. Returns None (not wired
        into the harness) when disabled -- and, booth rule, when the client
        cannot even be built because the wire deps (pyzmq + msgpack-numpy,
        the `grasping` extra) are missing from this venv.

        CASCADE_OCCUPANCY overrides the config (same contract as
        CASCADE_BELIEFS): "0/false/no/off" disables without editing YAML --
        the test suite pins it off so mock runs never wait out ZMQ timeouts
        against a bridge that isn't there -- and any other value forces it
        on.

        The map PROBES the bridge once here (300 ms). No answer = the map is
        still built (the bridge may come up later; refresh() keeps trying)
        but `status` is None and `probe_error` says why, so callers can show
        "occupancy: none" instead of implying a clearance gate exists."""
        import os

        env = os.environ.get("CASCADE_OCCUPANCY", "").strip().lower()
        if env:
            if env in ("0", "false", "no", "off"):
                return None
            enabled = True
        else:
            enabled = cfg is not None and bool(cfg.get("enabled", True))
        if cfg is None or not enabled:
            return None
        try:
            client = OccupancyClient(
                host=str(cfg.get("host", "127.0.0.1")),
                port=int(cfg.get("port", 5557)),
                timeout_ms=int(cfg.get("timeout_ms", 500)),
            )
        except OccupancyError as e:
            logger.warning(
                "occupancy map disabled: %s (install the `grasping` extra "
                "to enable the clearance gate)", e,
            )
            return None
        region_min = cfg.get("region_min", None)
        region_max = cfg.get("region_max", None)
        m = cls(
            client=client,
            region_min=region_min if region_min is not None else workspace_min,
            region_max=region_max if region_max is not None else workspace_max,
            max_age_s=float(cfg.get("max_age_s", 10.0)),
            stride=int(cfg.get("stride", 8)),
            depth_stride=int(cfg.get("depth_stride", 2)),
        )
        m.probe(timeout_ms=int(cfg.get("probe_timeout_ms", 300)))
        return m

    def probe(self, timeout_ms: int = 300) -> dict | None:
        """One startup round trip that names the backend (or records that
        nothing answered). Never raises."""
        if self._client is None:
            return None
        try:
            self.status = self._client.probe(timeout_ms=timeout_ms)
            self.probe_error = None
            self._depth_supported = True
            logger.info("occupancy bridge: %s", self.status.get("describe", self.status.get("backend")))
        except OccupancyError as e:
            self.status = None
            self.probe_error = str(e)
            logger.warning("occupancy bridge NOT reachable -- clearance gate is OFF until it is: %s", e)
        return self.status

    def describe(self) -> str:
        """One line for banners/summaries: 'warp on cpu (1.0 cm)' or 'none (...)'."""
        if self.status:
            return (f"{self.status.get('backend')} on {self.status.get('device', '?')} "
                    f"({float(self.status.get('voxel', 0))*100:.1f} cm voxels"
                    f"{', ESDF' if self.status.get('esdf') else ', occupancy-only'})")
        return f"none ({self.probe_error or 'not probed'})"

    # ── refresh ────────────────────────────────────────────────────────

    def refresh(self, frame, T_base_cam: np.ndarray) -> None:
        """Push this frame's depth (+K, pose) to the bridge, then pull back
        the distance grid for the configured region.

        Best-effort: any failure records `last_error` and leaves the
        existing cache in place (it simply ages toward `max_age_s`).
        """
        if self._client is None:
            return
        try:
            if frame.has_depth:
                if self._depth_supported is not False:
                    try:
                        self._integrate_depth(frame, T_base_cam)
                    except OccupancyError as e:
                        if "unknown action" in str(e):
                            # an old bridge: fall back to the cloud protocol
                            self._depth_supported = False
                            self._integrate_points(frame, T_base_cam)
                        else:
                            raise
                else:
                    self._integrate_points(frame, T_base_cam)
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
            if "grid" in resp:
                self._grid = np.asarray(resp["grid"], dtype=np.float32)
                self._grid_origin = np.asarray(resp["origin"], dtype=np.float32)
                self._grid_voxel = float(resp["voxel"])
            else:
                self._grid = None
            self._last_refresh = time.monotonic()
            self.last_error = None
            if self.status is None:
                # the bridge came up after startup: name it now
                self.probe()
        except OccupancyError as e:
            self.last_error = str(e)

    def _integrate_depth(self, frame, T_base_cam: np.ndarray) -> None:
        s = self.depth_stride
        depth = np.ascontiguousarray(frame.depth_m[::s, ::s], dtype=np.float32)
        K = np.asarray(frame.K, dtype=np.float64).copy()
        K[:2, :] /= s
        depth = self._mask_robot(depth, K, T_base_cam)
        self._client.request({
            "action": "integrate_depth", "depth": depth,
            "K": K.astype(np.float32), "T_base_cam": np.asarray(T_base_cam, dtype=np.float32),
        })

    def _mask_robot(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
        """Zero the depth pixels on the robot's own body (no measurement:
        neither free nor occupied for the ray-casting backends)."""
        if not self._body_fns:
            return depth
        from .robot_mask import robot_mask

        total = np.zeros(depth.shape, dtype=bool)
        for fn, r in zip(self._body_fns, self._body_radii):
            try:
                lp = fn()
            except Exception:  # noqa: BLE001 -- an arm in standby is not an error
                lp = None
            if lp is None:
                continue
            total |= robot_mask(depth, K, T_base_cam, np.asarray(lp), radius_m=r)
        self.last_masked_px = int(total.sum())
        if self.last_masked_px:
            depth = depth.copy()
            depth[total] = 0.0
        return depth

    def _integrate_points(self, frame, T_base_cam: np.ndarray) -> None:
        pts_base = self._depth_to_base_points(frame, T_base_cam)
        if pts_base.shape[0] and self._body_fns:
            # legacy cloud path: drop points within the body radius of any link
            from .robot_mask import _segment_distances

            keep = np.ones(len(pts_base), dtype=bool)
            for fn, r in zip(self._body_fns, self._body_radii):
                try:
                    lp = fn()
                except Exception:  # noqa: BLE001
                    lp = None
                if lp is None:
                    continue
                lp = np.asarray(lp, dtype=float)
                for i in range(max(len(lp) - 1, 0)):
                    keep &= _segment_distances(pts_base.astype(float), lp[i], lp[i + 1]) > r
            pts_base = pts_base[keep]
        if pts_base.shape[0]:
            self._client.request({"action": "integrate", "points": pts_base})

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

    # ── reads (hot path) ────────────────────────────────────────────────

    def is_stale(self) -> bool:
        return self._last_refresh is None or (time.monotonic() - self._last_refresh) > self.max_age_s

    def clearance(self, points: np.ndarray) -> np.ndarray | None:
        """Min distance from each query point to the nearest known obstacle.

        Returns None if there is no usable cache (never refreshed, stale, or
        nothing known) -- callers must treat that as "no data", not "clear".
        With a distance grid: trilinear interpolation inside the mapped
        region; a point OUTSIDE the region reads +inf ("nothing known
        there"), never the clamped border value -- clamping would let a
        point far outside the map inherit an obstacle distance it has no
        relation to, or hide a real one. Unknown (inf) voxels read as far.
        Without a grid (old bridge): brute-force distance to the occupied cloud.
        """
        if self.is_stale():
            return None
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if self._grid is not None and self._grid.size and np.isfinite(self._grid).any():
            return self._sample_grid(pts)
        if self._occupied is None or self._occupied.shape[0] == 0:
            return None
        d = np.linalg.norm(pts[:, None, :] - self._occupied[None, :, :], axis=-1)
        return d.min(axis=1)

    def _sample_grid(self, pts: np.ndarray) -> np.ndarray:
        g = self._grid
        shape = np.asarray(g.shape)
        f = (pts - self._grid_origin) / self._grid_voxel
        # half a voxel of slack: the grid's cells extend +-voxel/2 around centres
        inside = np.all((f >= -0.5) & (f <= shape - 0.5), axis=1)
        f = np.clip(f, 0, shape - 1 - 1e-6)
        i0 = np.floor(f).astype(int)
        i1 = np.minimum(i0 + 1, shape - 1)
        t = f - i0
        out = np.zeros(len(pts))
        for dx in (0, 1):
            wx = t[:, 0] if dx else 1 - t[:, 0]
            ix = i1[:, 0] if dx else i0[:, 0]
            for dy in (0, 1):
                wy = t[:, 1] if dy else 1 - t[:, 1]
                iy = i1[:, 1] if dy else i0[:, 1]
                for dz in (0, 1):
                    wz = t[:, 2] if dz else 1 - t[:, 2]
                    iz = i1[:, 2] if dz else i0[:, 2]
                    v = g[ix, iy, iz]
                    # inf (unknown) corners must not poison a finite neighbour:
                    # treat them as "far" for the interpolation
                    v = np.where(np.isfinite(v), v, 1e3)
                    out += wx * wy * wz * v
        out[~inside] = np.inf
        return out
