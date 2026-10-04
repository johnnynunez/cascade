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
the map (carving). Negotiated nvblox `integrate_masked_depth` keeps measured
robot/payload rays with an inactive surface mask: front free space survives,
while the excluded surface and space behind it receive no observations.
Other backends receive zero depth for excluded pixels. It runs off the motion hot path (WorldWatcher._tick, at
the perception rate); SafetyHarness.approve() at 50 Hz only ever reads the
cache. By default, past `max_age_s` the cache is treated as absent (skip
the check). With `required: true`, missing/stale/failed observations block
motion, and unknown distance-grid support is not free space.

An unavailable registered ROBOT BODY POSE is different from an absent
bridge: no frame may be integrated without its mask, and clearance raises
SafetyViolation until a masked depth refresh succeeds. It must not age into
"no data, skip"; the arm is known to be in the image but cannot be removed.

An optional bridge that is down degrades to "no occupancy check", with its
startup status visible in the demo and run summary. The opt-in
`occupancy.required` policy instead refuses startup without a live bridge
and refuses motion without a fresh observed grid. A successful probe alone
does not provide observed geometry or admit a robot trajectory.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class OccupancyError(RuntimeError):
    pass


class _MaskedDepthError(OccupancyError):
    """A negotiated native-mask operation must never downgrade to an old wire."""


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
        self._wire_lock = threading.RLock()

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
        # Startup probe and the perception watcher can run concurrently.
        # A REQ socket must complete send/recv before another request starts.
        with self._wire_lock:
            return self._request(payload, timeout_ms)

    def _request(self, payload: dict, timeout_ms: int | None = None) -> dict:
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
        with self._wire_lock:
            if self._sock is not None:
                self._sock.close()
                self._sock = None


class OccupancyMap:
    """Cached distance grid, refreshed off the motion hot path.

    `clearance()` trilinearly reads the cached grid -- cheap enough for
    `SafetyHarness.approve()` per waypoint. By default it returns None (meaning "skip
    the check") whenever there is no fresh cache, so a dead/slow bridge
    degrades exactly like running with no occupancy map at all rather than
    freezing the arm. A missing registered robot pose instead fails closed;
    its frame cannot safely be integrated and is not an empty scene.
    `required=True` also rejects unavailable maps and reports unobserved
    grid support as unknown for the harness to reject outside contact exemptions.
    """

    def __init__(
        self,
        client: OccupancyClient | None = None,
        region_min: np.ndarray | None = None,
        region_max: np.ndarray | None = None,
        max_age_s: float = 10.0,
        stride: int = 8,
        depth_stride: int = 2,
        required: bool = False,
    ):
        if type(required) is not bool:
            raise OccupancyError("occupancy.required must be a boolean")
        self.required = required
        self._client = client
        self._region_min = None if region_min is None else np.asarray(region_min, dtype=float)
        self._region_max = None if region_max is None else np.asarray(region_max, dtype=float)
        self.max_age_s = float(max_age_s)
        if required and (not np.isfinite(self.max_age_s) or self.max_age_s <= 0):
            raise OccupancyError("required occupancy max_age_s must be finite and positive")
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
        self._frame_body_fns: list = []
        self.last_masked_px: int = 0
        self._body_error: str | None = None
        self.allowed_contact_paths: set[str] = set()
        self._refresh_lock = threading.RLock()
        self._payload_pose_fn = None
        self._payload_samples = {}
        self._contact_paths = None
        self._contact_floor = -np.inf
        self._latest_contact_stamp = -np.inf
        self._depth_history = {}
        self._prop_history_floor = {}
        self.last_replayed_frames = 0
        self._reset_pending = False
        self._reset_floors = {}
        self.last_payload_query = None
        self.scene_reset_generation = 0
        # Captures admitted to the CURRENT successful depth + ESDF query.
        # Camera delivery and an attempted RPC are not map evidence.
        self._integrated_captures = {}
        self._payload_identity = None
        self._payload_epoch = None
        self._payload_epoch_error = None
        self._transition_pending = None
        self._capture_refresh_binding = None
        self.last_capture_refresh = None
        self._scene_reset_invalidated = False

    @staticmethod
    def _capture_key(marker):
        return (tuple(marker["source"]), marker["camera"], marker["robot_id"], marker["clock"])

    @property
    def scene_reset_pending(self) -> bool:
        """A reset still needs its capture barrier or valid captured geometry."""
        with self._refresh_lock:
            return (self._reset_pending or self._scene_reset_invalidated or self._capture_refresh_binding is not None
                    or bool(self._reset_floors and self._body_error))

    @staticmethod
    def _capture_epoch(frame):
        state = (getattr(frame, "capture", None) or {}).get("proprioception") or {}
        epoch = state.get("producer_epoch")
        if not isinstance(epoch, str) or not epoch:
            raise OccupancyError("capture refresh requires the producer physics epoch")
        return epoch

    def begin_capture_refresh(self, producer_clock):
        """Fence fresh geometry without discarding measured, unchanged-scene anchors.

        This contract is only for a confirmed attachment before any prop reset.
        A real scene reset must use begin_scene_reset, which discards this state.
        """
        if not self.tracks_payload:
            return False
        with self._refresh_lock:
            if self._scene_reset_invalidated:
                raise OccupancyError("scene reset cannot be downgraded to retained-anchor refresh")
            self._reset_pending = True
            self._body_error = "waiting for fresh retained-payload geometry"
            self._grid = self._occupied = None
            self._last_refresh = None
            self.last_capture_refresh = None
            binding = self._capture_refresh_binding
            if binding is not None and binding.get("failure"):
                raise OccupancyError(binding["failure"])
            if binding is None:
                if not self._contact_paths or self._payload_identity is None or not self._depth_history:
                    raise OccupancyError("capture refresh requires an observed attached payload and anchors")
                keys, epochs = set(), set()
                for camera, history in self._depth_history.items():
                    if not history:
                        raise OccupancyError("capture refresh is missing a measured camera anchor")
                    for old in history:
                        marker = old.get("marker") or {}
                        identity = (tuple(marker.get("source", ())), marker.get("robot_id"), marker.get("clock"))
                        epoch = old.get("producer_epoch")
                        if (identity != self._payload_identity or marker.get("camera") != camera
                                or not isinstance(epoch, str) or not epoch
                                or set(old["props"]) != self.allowed_contact_paths):
                            raise OccupancyError("retained anchor source, robot, clock or prop identity is invalid")
                        keys.add(self._capture_key(marker))
                        epochs.add(epoch)
                if len(epochs) != 1:
                    raise OccupancyError("retained anchors span different producer physics epochs")
                binding = {"keys": keys, "epoch": epochs.pop(), "paths": self._contact_paths,
                           "identity": self._payload_identity, "floor": None, "failure": None,
                           "anchors": {name: dict(rows[0]["marker"]) for name, rows in self._depth_history.items()}}
                self._capture_refresh_binding = binding
            if (not isinstance(producer_clock, dict) or producer_clock.get("epoch") != binding["epoch"]
                    or tuple(producer_clock.get("source", ())) != binding["identity"][0]
                    or producer_clock.get("robot_id") != binding["identity"][1]):
                binding["failure"] = "retained anchors differ from the current validated producer clock"
                raise OccupancyError(binding["failure"])
            self._integrated_captures.clear()
            self._reset_floors.clear()
            # The next validated frame replays measured background before it
            # can publish a new query. A failed replay remains fenced.
            self._transition_pending = (binding["paths"], self._latest_contact_stamp)
            return True

    def fence_capture_refresh(self, reason):
        """Keep a failed/deadline-expired refresh unusable until explicit retry."""
        with self._refresh_lock:
            self._reset_pending = True
            self._body_error = "retained-payload capture refresh failed: " + str(reason)
            self._grid = self._occupied = None
            self._last_refresh = None
            self._integrated_captures.clear()

    def begin_scene_reset(self):
        """Invalidate attached geometry/history before draining pre-reset captures."""
        if self._payload_pose_fn is None:
            return False
        with self._refresh_lock:
            self.scene_reset_generation += 1
            self._scene_reset_invalidated = True
            self._reset_pending = True
            self._body_error = "waiting for post-reset captured geometry"
            self._grid = self._occupied = None
            self._last_refresh = None
            self._depth_history.clear()
            self._prop_history_floor.clear()
            self._payload_samples.clear()
            self._integrated_captures.clear()
            self._payload_identity = None
            self._payload_epoch = None
            self._payload_epoch_error = None
            self._transition_pending = None
            self._capture_refresh_binding = None
            self.last_capture_refresh = None
            self._contact_paths = None
            self._contact_floor = self._latest_contact_stamp = -np.inf
            self._reset_floors.clear()
            self._client.request({"action": "clear"})
            return True

    def finish_scene_reset(self, floor_frames):
        from .freshness import capture_marker

        with self._refresh_lock:
            floors = {}
            for frame in floor_frames:
                marker = capture_marker(frame)
                floors[self._capture_key(marker)] = marker["t"]
            if not floors:
                raise OccupancyError("reset requires captured camera floors")
            binding = self._capture_refresh_binding
            if binding is not None:
                try:
                    if len(floor_frames) != len(floors) or set(floors) != binding["keys"]:
                        raise OccupancyError("capture refresh camera set or identity changed")
                    for frame in floor_frames:
                        if (self._capture_epoch(frame) != binding["epoch"]
                                or tuple(sorted(frame.capture.get("contact_paths", []))) != binding["paths"]):
                            raise OccupancyError("capture refresh producer epoch or contact changed")
                    binding["floor"] = max(floors.values())
                    floors = {key: binding["floor"] for key in floors}
                except (OccupancyError, TypeError, ValueError) as exc:
                    binding["failure"] = str(exc)
                    raise
            self._reset_floors = floors
            self._reset_pending = False

    def track_payload(self, frame_tcp_pose_fn) -> None:
        """Use captured depth of attached props as robot geometry, not world obstacles."""
        if self._payload_pose_fn is not None:
            raise ValueError("payload tracking currently requires a single arm")
        self._payload_pose_fn = frame_tcp_pose_fn
        self._body_error = "payload segmentation not yet validated"

    @property
    def tracks_payload(self) -> bool:
        return self._payload_pose_fn is not None

    def wait_payload_ready(self, floors, *, deadline, guard, expected_paths=None):
        """Wait for every post-close source to enter one coherent attached map.

        Floors are collected AFTER close, before calling this method. All
        cameras must progress beyond the latest floor on their shared producer
        clock. Waiting never holds the refresh lock or changes an RPC timeout.
        """
        return self._wait_capture_ready(floors, deadline=deadline, guard=guard,
                                        expected_paths=expected_paths)

    def wait_released_ready(self, floors, *, deadline, guard, producer_epoch, prop_floors):
        """Require successful fresh commits of the empty-hand state from every source.

        This does not clear/rebuild the map or discard anchors. The ordinary
        attachment transition must have committed the released prop's history
        floor before any withdrawal may use its original contact cylinder.
        """
        return self._wait_capture_ready(floors, deadline=deadline, guard=guard,
            expected_paths=(), released=(producer_epoch, dict(prop_floors)))

    def _wait_capture_ready(self, floors, *, deadline, guard, expected_paths=None, released=None):
        from .freshness import capture_marker

        markers = [capture_marker(frame) for frame in floors]
        if not markers or any(m.get("backend") != "isaac" for m in markers):
            raise OccupancyError("payload barrier requires Isaac capture floors")
        identities = {(tuple(m["source"]), m["robot_id"], m["clock"]) for m in markers}
        keys = {self._capture_key(m) for m in markers}
        if len(identities) != 1 or len(keys) != len(markers):
            raise OccupancyError("payload barrier camera/robot/clock identity mismatch")
        floor = max(m["t"] for m in markers)
        expected = None if expected_paths is None else tuple(sorted(expected_paths))
        reason = "waiting for all post-close map captures"
        while True:
            guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OccupancyError(f"post-close payload geometry deadline: {reason}")
            if not self._refresh_lock.acquire(timeout=min(.05, remaining)):
                continue
            try:
                entries = [self._integrated_captures.get(key) for key in keys]
                ready = (all(e is not None and e["marker"]["t"] > floor for e in entries)
                         and (self._contact_paths == () if released else bool(self._contact_paths))
                         and all(e["paths"] == self._contact_paths for e in entries if e)
                         and (released is None or (
                             self._payload_epoch == released[0]
                             and all(e["marker"].get("producer_epoch") == released[0] for e in entries if e)
                             and all(self._prop_history_floor.get(path, -np.inf) >= stamp
                                     for path, stamp in released[1].items())))
                         and (expected is None or self._contact_paths == expected))
                if ready and not self._body_error and not self.last_error and not self.is_stale():
                    if (self._grid is not None and self._grid.size
                            and np.isfinite(self._grid).any()
                            and (released is not None or sum(len(v) for v in self._payload_samples.values()) > 0)):
                        guard()
                        if time.monotonic() >= deadline:
                            raise OccupancyError("post-close payload geometry deadline expired")
                        return {"floors": markers, "shared_floor": floor,
                                "integrated": [dict(e["marker"]) for e in entries],
                                "contact_paths": list(self._contact_paths),
                                "surfaces_by_camera": {k: len(v) for k, v in self._payload_samples.items()}}
                reason = self._body_error or self.last_error or "missing coherent fresh captures or observed surface"
            finally:
                self._refresh_lock.release()
            time.sleep(min(.01, max(0., deadline - time.monotonic())))

    def payload_points(self, T_base_tcp: np.ndarray) -> np.ndarray:
        """Transform measured attached surfaces with a candidate TCP pose."""
        with self._refresh_lock:
            if not self._payload_samples:
                return np.empty((0, 3))
            points = np.concatenate(list(self._payload_samples.values()))
            return points @ T_base_tcp[:3, :3].T + T_base_tcp[:3, 3]

    def payload_clearance(self, points: np.ndarray) -> np.ndarray | None:
        """A carried surface needs observed ESDF support; unknown is not free."""
        with self._refresh_lock:
            distance = self._clearance(points)
            if distance is None or self._grid is None:
                return None
            known = self._observed_grid_support(points)
            self.last_payload_query = {
                "samples": len(points), "unobserved_samples": int((~known).sum()),
                "minimum_observed_clearance_m": float(distance[known].min()) if known.any() else None,
                "surfaces_by_camera": {name: len(p) for name, p in self._payload_samples.items()},
            }
            result = np.where(known, distance, np.nan)
            self._check_required_age()
            return result

    def _observed_grid_support(self, points: np.ndarray) -> np.ndarray:
        """Require every contributing interpolation corner to be observed.

        Called under the cache lock by both carried-surface queries and the
        opt-in required-map policy. Outside the grid is always unknown.
        """
        fractional = (np.atleast_2d(points) - self._grid_origin) / self._grid_voxel
        shape = np.asarray(self._grid.shape)
        inside = ((fractional >= 0) & (fractional <= shape - 1)).all(axis=1)
        low = np.floor(np.clip(fractional, 0, shape - 1)).astype(int)
        high = np.minimum(low + 1, shape - 1)
        blend = np.clip(fractional, 0, shape - 1) - low
        known = inside.copy()
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    index = np.column_stack((
                        (low if dx == 0 else high)[:, 0],
                        (low if dy == 0 else high)[:, 1],
                        (low if dz == 0 else high)[:, 2]))
                    weight = ((blend[:, 0] if dx else 1-blend[:, 0])
                              * (blend[:, 1] if dy else 1-blend[:, 1])
                              * (blend[:, 2] if dz else 1-blend[:, 2]))
                    known &= np.isfinite(self._grid[tuple(index.T)]) | (weight <= 1e-9)
        return known

    def add_robot_body(self, link_points_fn, radius_m: float = 0.06, *, frame_link_points_fn=None) -> None:
        """Register an arm to mask out of the depth before integration.
        `link_points_fn() -> (L,3) | None` (None defers integration and blocks
        clearance until a masked depth refresh succeeds). If supplied,
        `frame_link_points_fn(frame)` is REQUIRED for integration: no fallback
        to current proprioception when a capture snapshot is absent/invalid.
        Backends without this contract retain the no-argument callback."""
        self._body_fns.append(link_points_fn)
        self._body_radii.append(float(radius_m))
        self._frame_body_fns.append(frame_link_points_fn)
        self._body_error = "robot body pose not yet validated by a masked depth refresh"

    def _robot_bodies(self, frame=None) -> list:
        """Read ALL registered bodies before sending any depth to the map.

        Unknown pose is not an empty scene: integrating it would persist the
        robot as an obstacle. Latch the fault until a masked refresh succeeds,
        so an ageing cache cannot silently turn it into a clearance bypass.
        """
        bodies = []
        for i, (fn, radius) in enumerate(zip(self._body_fns, self._body_radii)):
            try:
                frame_fn = self._frame_body_fns[i]
                points = np.asarray(frame_fn(frame) if frame_fn is not None else fn(), dtype=float)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                    raise ValueError("missing or invalid link points")
            except Exception as e:  # noqa: BLE001 -- includes standby / bridge failures
                self.last_masked_px = 0
                self._body_error = f"robot body pose {i} unavailable: {e}"
                raise OccupancyError(self._body_error) from e
            bodies.append((points, radius))
        return bodies

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
        "occupancy: none" instead of implying a clearance gate exists.
        With `required: true`, disabling the map, missing client dependencies
        or a failed probe raises instead of constructing an optional runtime."""
        import os

        required = False if cfg is None else cfg.get("required", False)
        if type(required) is not bool:
            raise OccupancyError("occupancy.required must be a boolean")
        if required and cfg.get("enabled", True) is not True:
            raise OccupancyError("required occupancy cannot be disabled in configuration")
        env = os.environ.get("CASCADE_OCCUPANCY", "").strip().lower()
        if env:
            if env in ("0", "false", "no", "off"):
                if required:
                    raise OccupancyError("required occupancy conflicts with CASCADE_OCCUPANCY=off")
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
            if required:
                raise OccupancyError(f"required occupancy client unavailable: {e}") from e
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
            required=required,
        )
        m.allowed_contact_paths = set(cfg.get("allowed_contact_paths", []))
        m.probe(timeout_ms=int(cfg.get("probe_timeout_ms", 300)))
        if required and m.status is None:
            client.close()
            raise OccupancyError(f"required occupancy bridge unavailable: {m.probe_error}")
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
            logger.warning("occupancy bridge NOT reachable (%s): %s",
                           "required map blocks motion" if self.required else "optional gate unavailable", e)
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
        with self._refresh_lock:
            self._refresh(frame, T_base_cam)

    def _refresh(self, frame, T_base_cam: np.ndarray) -> None:
        if self._client is None:
            return
        if self._reset_pending:
            return
        if self._reset_floors:
            from .freshness import capture_marker

            marker = capture_marker(frame)
            key = self._capture_key(marker)
            if key not in self._reset_floors:
                self._body_error = "camera identity changed after reset"
                if self._capture_refresh_binding is not None:
                    self._capture_refresh_binding["failure"] = self._body_error
                raise OccupancyError(self._body_error)
            if marker["t"] <= self._reset_floors[key]:
                return
        if self._payload_pose_fn is not None:
            capture = getattr(frame, "capture", None) or {}
            stamp = capture.get("t")
            if (self._transition_pending is not None and isinstance(stamp, (int, float))
                    and stamp < self._transition_pending[1]):
                return  # an interrupted clear cannot be undone by delayed old-state evidence
            if isinstance(stamp, (int, float)) and stamp < self._contact_floor:
                return  # an older camera must not undo an attachment transition
            paths = tuple(sorted(capture.get("contact_paths", [])))
            if (isinstance(stamp, (int, float)) and stamp < self._latest_contact_stamp
                    and paths != self._contact_paths):
                return  # newer evidence of the SAME state also supersedes an old transition
        try:
            prepared = None
            if self.required:
                capture_t = float(frame.t)
                age = time.monotonic() - capture_t
                if (not frame.has_depth or not np.isfinite(age)
                        or age < 0 or age > self.max_age_s
                        or not np.any(np.isfinite(frame.depth_m) & (frame.depth_m > 0))):
                    raise OccupancyError("required occupancy needs a fresh nonempty depth frame")
            if frame.has_depth:
                if self._depth_supported is not False:
                    try:
                        prepared = self._integrate_depth(frame, T_base_cam)
                    except OccupancyError as e:
                        if self._payload_pose_fn is not None or isinstance(e, _MaskedDepthError):
                            raise  # payload transitions require the depth + clear contract
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
            occupied = np.asarray(resp["points"], dtype=np.float32).reshape(-1, 3)
            if not np.isfinite(occupied).all():
                raise ValueError("nonfinite occupied points")
            if "grid" in resp:
                grid = np.asarray(resp["grid"], dtype=np.float32)
                origin = np.asarray(resp["origin"], dtype=np.float32)
                voxel = float(resp["voxel"])
                if (grid.ndim != 3 or min(grid.shape) < 1 or np.isnan(grid).any()
                        or origin.shape != (3,) or not np.isfinite(origin).all()
                        or not np.isfinite(voxel) or voxel <= 0):
                    raise ValueError("invalid ESDF query geometry")
            else:
                grid, origin, voxel = None, None, 0.
            self._occupied, self._grid = occupied, grid
            self._grid_origin, self._grid_voxel = origin, voxel
            self._last_refresh = capture_t if self.required else time.monotonic()
            self.last_error = None
            if frame.has_depth:
                self._body_error = None
            if prepared is not None:
                paths, stamp, camera, points, floors, marker = prepared
                if paths != self._contact_paths:
                    self._payload_samples.clear()
                    self._contact_floor = stamp
                self._contact_paths = paths
                self._transition_pending = None
                self._payload_identity = (tuple(marker["source"]), marker["robot_id"], marker["clock"])
                self._payload_epoch = marker.get("producer_epoch")
                self._latest_contact_stamp = max(self._latest_contact_stamp, stamp)
                self._prop_history_floor = floors
                if paths and len(points):
                    self._payload_samples[camera] = points
                self._remember_depth(frame, T_base_cam)
                self._integrated_captures[self._capture_key(marker)] = {"marker": marker, "paths": paths}
                if paths and not any(len(p) for p in self._payload_samples.values()):
                    self._body_error = "attached object has no measured depth surface"
                    self.last_error = self._body_error
                binding = self._capture_refresh_binding
                if binding is not None:
                    entries = [self._integrated_captures.get(key) for key in binding["keys"]]
                    complete = (binding["floor"] is not None and not binding["failure"]
                        and all(entry is not None and entry["marker"]["t"] > binding["floor"]
                                and entry["paths"] == binding["paths"] for entry in entries)
                        and self._grid is not None and np.isfinite(self._grid).any()
                        and any(len(points) for points in self._payload_samples.values()))
                    if complete and not self._body_error:
                        self.last_capture_refresh = {
                            "producer_epoch": binding["epoch"], "shared_floor": binding["floor"],
                            "contact_paths": list(binding["paths"]),
                            "anchors_before": binding["anchors"],
                            "anchors_after": {name: dict(rows[0]["marker"])
                                              for name, rows in self._depth_history.items()},
                            "integrated": [dict(entry["marker"]) for entry in entries],
                            "replayed_frames": self.last_replayed_frames,
                            "prop_history_floor": dict(self._prop_history_floor),
                        }
                        self._capture_refresh_binding = None
                    else:
                        self._body_error = binding["failure"] or "waiting for all retained-payload camera commits"
                elif self._scene_reset_invalidated:
                    complete = bool(self._reset_floors) and all(
                        key in self._integrated_captures
                        and self._integrated_captures[key]["marker"]["t"] > floor
                        and self._integrated_captures[key]["paths"] == self._contact_paths
                        for key, floor in self._reset_floors.items())
                    if complete and not self._body_error:
                        self._scene_reset_invalidated = False
                    else:
                        self._body_error = "waiting for every post-reset camera integration"
            if self.status is None:
                # the bridge came up after startup: name it now
                self.probe()
        except (OccupancyError, KeyError, TypeError, ValueError) as e:
            self.last_error = str(e)
            if self.tracks_payload:
                self._body_error = f"payload tracking unavailable: {e}"
                camera = (getattr(frame, "capture", None) or {}).get("camera")
                self._integrated_captures = {k: v for k, v in self._integrated_captures.items()
                                             if k[1] != camera}

    def _integrate_depth(self, frame, T_base_cam: np.ndarray) -> None:
        s = self.depth_stride
        depth = np.ascontiguousarray(frame.depth_m[::s, ::s], dtype=np.float32)
        K = np.asarray(frame.K, dtype=np.float64).copy()
        K[:2, :] /= s
        excluded = self._robot_depth_mask(depth, K, T_base_cam, frame=frame)
        prepared = self._prepare_payload(frame, T_base_cam) if self.tracks_payload else None
        self._send_depth(depth, K, T_base_cam, excluded)
        return prepared

    def _send_depth(self, depth, K, T_base_cam, excluded):
        # Raw self/payload depth may only go to an explicitly negotiated
        # native masking operation. An older bridge rejects this action;
        # it cannot silently ignore an extra mask on integrate_depth.
        if self.status is None:
            self.probe()
        native_mask = (self.status is not None
                       and self.status.get("backend") == "nvblox"
                       and self.status.get("masked_depth") is True)
        packet = {
            "action": "integrate_masked_depth" if native_mask else "integrate_depth",
            "depth": np.ascontiguousarray(depth if native_mask else np.where(excluded, 0., depth),
                                          dtype=np.float32),
            "K": np.asarray(K, dtype=np.float32),
            "T_base_cam": np.asarray(T_base_cam, dtype=np.float32),
        }
        if native_mask:
            # nvblox's inactive rays integrate observed free space only
            # before the positive TSDF truncation band. They add neither
            # the excluded surface nor invented geometry behind it.
            packet["active_mask"] = np.ascontiguousarray(~excluded, dtype=np.uint8)
        try:
            self._client.request(packet)
        except OccupancyError as exc:
            if native_mask:
                self._body_error = f"native depth masking unavailable: {exc}"
                raise _MaskedDepthError(self._body_error) from exc
            raise

    def _remember_depth(self, frame, T_base_cam):
        capture = frame.capture
        camera, stamp = capture.get("camera", "primary"), float(capture["t"])
        history = self._depth_history.setdefault(camera, [])
        if history and history[-1]["t"] == stamp:
            return
        from .freshness import capture_marker
        history.append({"t": stamp, "marker": capture_marker(frame),
                        "producer_epoch": capture.get("proprioception", {}).get("producer_epoch"),
                        "depth": frame.depth_m.copy(), "K": frame.K.copy(),
                        "T": np.asarray(T_base_cam).copy(),
                        "robot": frame.robot_mask & ~frame.payload_mask,
                        "props": {path: mask.copy() for path, mask in frame.prop_masks.items()}})
        # Keep an anchor from before the gripper occludes the scene plus the
        # newest view. Exact prop masks prevent old object poses resurfacing.
        if len(history) > 2:
            del history[1:-1]

    def _replay_background(self, paths, stamp):
        # Reconstruct from actual prior views, removing only exact pixels of
        # this payload. A world-map clear must not erase every other camera's
        # recent observation while the gripper occludes the current view.
        # Old poses of released props remain filtered from historical frames.
        floors = self._prop_history_floor.copy()
        for path in set(paths) | set(self._contact_paths or ()):
            floors[path] = stamp
        self._grid = self._occupied = None
        self._last_refresh = None
        self._integrated_captures.clear()
        if self._capture_refresh_binding is None or paths != self._contact_paths:
            self._payload_samples.clear()
        self._client.request({"action": "clear"})
        self.last_replayed_frames = 0
        history = sorted((f for frames in self._depth_history.values() for f in frames),
                         key=lambda f: f["t"])
        for old in history:
            mask = old["robot"].copy()
            for path, pixels in old["props"].items():
                if path in paths or old["t"] < floors.get(path, -np.inf):
                    mask |= pixels
            s = self.depth_stride
            depth = old["depth"][::s, ::s]
            K = old["K"].copy()
            K[:2, :] /= s
            self._send_depth(depth, K, old["T"], mask[::s, ::s])
            self.last_replayed_frames += 1
        return floors

    def _prepare_payload(self, frame, T_base_cam) -> None:
        try:
            from .freshness import capture_marker
            marker = capture_marker(frame)
            if marker.get("backend") != "isaac":
                raise ValueError("payload requires an Isaac producer capture")
            identity = (tuple(marker["source"]), marker["robot_id"], marker["clock"])
            epoch = (getattr(frame, "capture", None) or {}).get("proprioception", {}).get("producer_epoch")
            if self._payload_epoch_error:
                raise ValueError(self._payload_epoch_error)
            if ((epoch is not None and (not isinstance(epoch, str) or not epoch))
                    or (self._payload_epoch is not None and epoch != self._payload_epoch)):
                self._payload_epoch_error = "payload producer epoch changed; a real scene reset is required"
                raise ValueError(self._payload_epoch_error)
            if epoch is not None:
                marker["producer_epoch"] = epoch
            if self._payload_identity is not None and identity != self._payload_identity:
                if self._capture_refresh_binding is not None:
                    self._capture_refresh_binding["failure"] = "payload capture source/robot/clock identity changed"
                raise ValueError("payload capture source/robot/clock identity changed")
            mask = getattr(frame, "payload_mask", None)
            capture = frame.capture or {}
            stamp = float(capture["t"])
            paths = tuple(sorted(capture.get("contact_paths", [])))
            binding = self._capture_refresh_binding
            if binding is not None:
                try:
                    if (binding["failure"] or self._capture_key(marker) not in binding["keys"]
                            or identity != binding["identity"] or paths != binding["paths"]
                            or self._capture_epoch(frame) != binding["epoch"]):
                        raise OccupancyError(binding["failure"] or "retained-payload capture identity or contact changed")
                except (OccupancyError, ValueError, TypeError) as exc:
                    binding["failure"] = str(exc)
                    raise
            props = getattr(frame, "prop_masks", None)
            if (props is None or set(props) != self.allowed_contact_paths
                    or any(m.dtype != np.bool_ or m.shape != frame.depth_m.shape for m in props.values())):
                raise ValueError("missing or invalid captured per-prop masks")
            if (mask is None or mask.dtype != np.bool_ or mask.shape != frame.depth_m.shape
                    or not np.isfinite(stamp) or (mask.any() and not paths)
                    or np.any(mask & ~frame.robot_mask)):
                raise ValueError("missing or invalid captured payload mask")
            expected = np.zeros(mask.shape, dtype=bool)
            for path in paths:
                expected |= props[path]
            if not np.array_equal(expected, mask):
                raise ValueError("payload mask does not match captured prop identities")
            T_tcp = np.asarray(self._payload_pose_fn(frame), dtype=float)
            if T_tcp.shape != (4, 4) or not np.isfinite(T_tcp).all():
                raise ValueError("invalid captured TCP pose")
            valid = mask & np.isfinite(frame.depth_m) & (frame.depth_m > 0)
            y, x = np.nonzero(valid)
            points = np.empty((0, 3))
            if len(x):
                z = frame.depth_m[y, x]
                K = frame.K
                camera = np.column_stack(((x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z))
                base = camera @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
                local = (base - T_tcp[:3, 3]) @ T_tcp[:3, :3]
                # Keep spatial coverage with one measured point per 3 mm cell.
                _, index = np.unique(np.floor(local/.003).astype(np.int64), axis=0, return_index=True)
                points = local[index]
            if paths != self._contact_paths or self._transition_pending is not None:
                self._transition_pending = (paths, stamp)
                floors = self._replay_background(paths, stamp)
            else:
                floors = self._prop_history_floor.copy()
            return paths, stamp, capture["camera"], points, floors, marker
        except Exception as exc:
            self._body_error = f"payload tracking unavailable: {exc}"
            raise OccupancyError(self._body_error) from exc

    def _render_robot_mask(self, frame):
        mask = getattr(frame, "robot_mask", None)
        if mask is None:
            return None
        contacts = (getattr(frame, "capture", None) or {}).get("contact_paths", [])
        if (not isinstance(contacts, list) or any(not isinstance(p, str) for p in contacts)
                or not set(contacts).issubset(self.allowed_contact_paths)):
            self._body_error = "render mask includes unapproved contact paths"
            raise OccupancyError(self._body_error)
        mask = np.asarray(mask)
        if (mask.dtype != np.bool_ or frame.depth_m is None
                or mask.shape != frame.depth_m.shape):
            self._body_error = "invalid render self-mask shape/type"
            raise OccupancyError(self._body_error)
        return mask

    def _robot_depth_mask(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, *, frame=None) -> np.ndarray:
        """Identify excluded measured surfaces before choosing backend semantics."""
        bodies = self._robot_bodies(frame)
        from .robot_mask import robot_mask

        total = np.zeros(depth.shape, dtype=bool)
        pixels = self._render_robot_mask(frame)
        if pixels is not None:
            total = pixels[::self.depth_stride, ::self.depth_stride]
        else:
            for lp, r in bodies:
                total |= robot_mask(depth, K, T_base_cam, lp, radius_m=r)
        self.last_masked_px = int(total.sum())
        return total

    def _integrate_points(self, frame, T_base_cam: np.ndarray) -> None:
        bodies = self._robot_bodies(frame)
        pixels = self._render_robot_mask(frame)
        if pixels is not None:
            import copy

            frame = copy.copy(frame)
            frame.depth_m = frame.depth_m.copy()
            frame.depth_m[pixels] = 0
            self.last_masked_px = int(pixels.sum())
        pts_base = self._depth_to_base_points(frame, T_base_cam)
        if pts_base.shape[0] and bodies and pixels is None:
            # legacy cloud path: drop points within the body radius of any link
            from .robot_mask import _segment_distances

            keep = np.ones(len(pts_base), dtype=bool)
            for lp, r in bodies:
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
        Required maps raise SafetyViolation if missing, stale or failed, and
        return NaN for points without observed grid support. A pending/failed robot-body mask raises SafetyViolation even when
        the cache has aged out; this known fault must never become a bypass.
        """
        with self._refresh_lock:
            distance = self._clearance(points)
            if self.required:
                distance = np.where(self._observed_grid_support(points), distance, np.nan)
                self._check_required_age()
            return distance

    def _check_required_age(self) -> None:
        """Interpolation must finish inside the same observation lifetime."""
        if self.required and self.is_stale():
            from ..types import SafetyViolation

            raise SafetyViolation("required occupancy unavailable: distance grid is stale")

    def _clearance(self, points: np.ndarray) -> np.ndarray | None:
        if self._body_error is not None:
            from ..types import SafetyViolation

            raise SafetyViolation(f"occupancy unsafe: {self._body_error}")
        stale = self.is_stale()
        usable_grid = (not stale and self._grid is not None and self._grid.size
                       and np.isfinite(self._grid).any())
        if self.required:
            from ..types import SafetyViolation

            reason = None if self.last_error is None else f"refresh failed: {self.last_error}"
            if stale:
                reason = reason or "distance grid is missing or stale"
            if not usable_grid:
                reason = reason or "fresh observed distance grid required"
            if reason:
                raise SafetyViolation(f"required occupancy unavailable: {reason}")
        if stale:
            return None
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if usable_grid:
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
        # i1 is already capped at the last centre. An epsilon here would
        # mix an unobserved neighbour into an exactly observed border cell.
        f = np.clip(f, 0, shape - 1)
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
