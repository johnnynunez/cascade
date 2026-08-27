"""GraspGen-X grasp backend: learned 6-DoF grasps behind the same Grasp API.

Talks to a GraspGen-X ZMQ server (NVlabs/GraspGenX client-server mode) with a
self-contained msgpack wire client -- the heavy model stack lives in its own
venv/process (~/Projects/demo/.graspgenx), cascade only needs pyzmq +
msgpack-numpy. Launch the server with scripts/serve_graspgenx.sh.

Flow: the segmented object points from an ObjectFix (already in the BASE
frame) go up; ranked (K,4,4) grasp poses come back in the SAME frame
("grasps are returned in the input point cloud's frame"). Poses convert to
cascade's Grasp convention:

    GraspGen-X gripper frame: +Z = approach axis, +X = jaw closing axis,
        origin at the GRIPPER BASE.
    cascade Grasp: rotation columns [x=approach, y=jaw-opening, z=x*y],
        position = the point BETWEEN the jaws (the arm's gripper_end frame).

so R_wrc = [R[:,2], R[:,0], R[:,1]] (even permutation) and the position
shifts along the approach axis by `tip_offset_m` (gripper-base -> jaw-center
distance for the gripper the server was asked to emulate; calibrate in sim).

The OBB planner stays as the always-available fallback: any server error
degrades to the analytic path instead of failing the grasp (booth rule).
GraspGen-X's own default planner (GraspMoE) is itself diffusion + OBB
heuristics rescored by a learned discriminator, so this wiring mirrors the
paper's intended usage.
"""

from __future__ import annotations

import time

import numpy as np

from ..types import Grasp, ObjectFix


class GraspGenXError(RuntimeError):
    pass


class GraspGenXClient:
    """Minimal REQ/REP msgpack wire client (mirrors graspgenx.serving)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5556, timeout_ms: int = 8000):
        try:
            import msgpack  # noqa: F401
            import msgpack_numpy
            import zmq  # noqa: F401
        except ImportError as e:
            raise GraspGenXError(
                f"graspgenx backend needs pyzmq+msgpack-numpy in this venv ({e})"
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
            raise GraspGenXError(
                f"graspgenx server at {self._host}:{self._port} timed out "
                f"({self._timeout} ms); is scripts/serve_graspgenx.sh running?"
            ) from e
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise GraspGenXError(f"graspgenx server error: {resp['error']}")
        return resp

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class GraspGenXPlanner:
    def __init__(self, cfg):
        g = cfg.get("graspgenx", None)
        get = (lambda k, d: g.get(k, d)) if g is not None else (lambda k, d: d)
        self._client = GraspGenXClient(
            host=str(get("host", "127.0.0.1")),
            port=int(get("port", 5556)),
            timeout_ms=int(get("timeout_ms", 8000)),
        )
        self.gripper = str(get("gripper", "franka_panda"))
        self.tip_offset_m = float(get("tip_offset_m", 0.10))
        self.num_grasps = int(get("num_grasps", 100))
        self.topk = int(get("topk", 32))
        self.min_score = float(get("min_score", 0.0))
        self.last_latency_s: float | None = None
        # Cross-embodiment mode: a `sweep` block describes OUR gripper by
        # its swept volume (12 numbers) -- no name lookup, no borrowed
        # Franka. Convention: origin at the JAW CENTER (tip_offset then 0),
        # +Z = approach, +X = closing direction.
        sweep = get("sweep", None)
        self.sweep_params = None
        if sweep is not None:
            self.sweep_params = {
                "extents_open": [float(v) for v in sweep.get("extents_open")],
                "offset_open": [float(v) for v in sweep.get("offset_open", [0, 0, 0])],
                "extents_mid": [float(v) for v in sweep.get("extents_mid")],
                "offset_mid": [float(v) for v in sweep.get("offset_mid", [0, 0, 0])],
                "gripper_type": int(sweep.get("gripper_type", 0)),
                "fingertip_depth": float(sweep.get("fingertip_depth", 0.0)),
            }
            self.tip_offset_m = float(get("tip_offset_m", 0.0))

    def plan(self, fix: ObjectFix, max_width_m: float = 0.09) -> list[Grasp]:
        """Segmented base-frame object points -> ranked wrc Grasps."""
        pts = np.asarray(fix.points, dtype=np.float32)
        if pts.shape[0] < 50:
            raise GraspGenXError(f"only {pts.shape[0]} object points (<50)")
        t0 = time.monotonic()
        if self.sweep_params is not None:
            payload = {
                "action": "infer_object",
                "point_cloud": pts,
                "sweep_volume_params": self.sweep_params,
                "num_grasps": self.num_grasps,
                "planner": "graspmoe",
            }
        else:
            payload = {
                "action": "infer",
                "point_cloud": pts,
                "gripper_name": self.gripper,
                "num_grasps": self.num_grasps,
                "grasp_threshold": float(self.min_score),
                "topk_num_grasps": self.topk,
            }
        # Diffusion sampling is stochastic: a borderline cloud (e.g. a slab
        # near the jaw limit) can land EVERY sample under the server's
        # internal score threshold on one run and return 200+ on the next.
        # One retry is cheap; a repeat empty means the cloud itself is the
        # problem, so dump it for offline replay.
        poses = scores = None
        for _attempt in range(2):
            resp = self._client.request(payload)
            poses = np.asarray(resp["grasps"], dtype=np.float32).reshape(-1, 4, 4)
            scores = np.asarray(resp["confidences"], dtype=np.float32).reshape(-1)
            if poses.shape[0]:
                break
        self.last_latency_s = round(time.monotonic() - t0, 3)
        if poses.shape[0] == 0:
            dump = f"/tmp/wrc_ggx_empty_{int(time.time())}.npz"
            try:
                np.savez_compressed(dump, points=pts, label=str(fix.label))
            except OSError:
                dump = "unsaved"
            raise GraspGenXError(
                f"graspgenx returned no grasps twice (cloud dumped: {dump})"
            )

        # Jaw-opening width from the object's span along each grasp's jaw
        # axis (the wire protocol does not carry per-grasp widths; the
        # runtime's air-grasp verification and select_grasp need one).
        grasps: list[Grasp] = []
        order = np.argsort(-scores)
        for i in order:
            T = poses[i].astype(float)
            R, p = T[:3, :3], T[:3, 3]
            approach = R[:, 2] / np.linalg.norm(R[:, 2])
            # Tabletop sanity: the model sees a floating cloud (no table), so
            # it happily proposes approaches from below. Those are physically
            # unreachable here -- drop them before wasting IK attempts.
            if approach[2] > 0.2:
                continue
            open_axis = R[:, 0] / np.linalg.norm(R[:, 0])
            R_wrc = np.column_stack([approach, open_axis, np.cross(approach, open_axis)])
            tcp = p + approach * self.tip_offset_m
            span = (fix.points - fix.position) @ open_axis
            width = float(
                min(np.quantile(span, 0.95) - np.quantile(span, 0.05) + 0.015,
                    max_width_m)
            )
            grasps.append(
                Grasp(
                    position=tcp,
                    rotation=R_wrc,
                    width_m=width,
                    approach=approach,
                    quality=float(scores[i]),
                    label=fix.label,
                )
            )
        return grasps
