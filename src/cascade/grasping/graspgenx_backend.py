"""GraspGen-X grasp backend: learned 6-DoF grasps behind the same Grasp API.

Talks to a GraspGen-X ZMQ server (NVlabs/GraspGenX client-server mode) with a
self-contained msgpack wire client -- the heavy model stack lives in its own
venv/process (.graspgenx inside this checkout), cascade only needs pyzmq +
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

The Spark profile requires a real CUDA server and diffusion-only candidates.
Other profiles may explicitly allow an analytic fallback. GraspMoE remains
available as an upstream planner choice, with its own diffusion/OBB provenance.

"""

from __future__ import annotations

import os
import time

import numpy as np

from ..types import Grasp, ObjectFix
from . import evidence


class GraspGenXError(RuntimeError):
    pass


class NoEligibleGrasps(GraspGenXError):
    """A valid nonempty learned batch was removed by the unchanged filters."""


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

    def request(self, payload: dict, timeout_ms: int | None = None, *,
                deadline: float | None = None, check=None) -> dict:
        import msgpack
        import zmq

        if deadline is not None:
            return self._request_bounded(payload, timeout_ms, deadline, check)
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
            raise GraspGenXError(
                f"graspgenx server at {self._host}:{self._port} timed out "
                f"({timeout_ms if timeout_ms is not None else self._timeout} ms); "
                f"is scripts/serve_graspgenx.sh running?"
            ) from e
        finally:
            if timeout_ms is not None and self._sock is not None:
                self._sock.setsockopt(zmq.RCVTIMEO, self._timeout)
                self._sock.setsockopt(zmq.SNDTIMEO, self._timeout)
        resp = msgpack.unpackb(raw, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise GraspGenXError(f"graspgenx server error: {resp['error']}")
        return resp

    def _request_bounded(self, payload, timeout_ms, deadline, check):
        """Send/receive share one deadline; cancelled responses are discarded."""
        import math
        import msgpack
        import zmq

        if not isinstance(deadline, (int, float)) or isinstance(deadline, bool) or not math.isfinite(deadline):
            raise GraspGenXError("invalid grasp planning deadline")
        end = min(deadline, time.monotonic() + (self._timeout if timeout_ms is None else timeout_ms) / 1000.)

        def remaining_ms():
            if check is not None:
                check()  # pure cancellation/deadline callback, never a robot RPC
            left = end - time.monotonic()
            if left <= 0:
                raise GraspGenXError("graspgenx planning request timed out")
            return max(1, min(50, math.ceil(left * 1000)))

        try:
            remaining_ms()
            encoded = msgpack.packb(payload, use_bin_type=True)
            remaining_ms()
            if self._sock is None:
                self._connect()
            sock = self._sock
            for event, operation in (
                    (zmq.POLLOUT, lambda: sock.send(encoded, flags=zmq.NOBLOCK)),
                    (zmq.POLLIN, lambda: sock.recv(flags=zmq.NOBLOCK))):
                while True:
                    ready = sock.poll(remaining_ms(), event)
                    remaining_ms()
                    if not ready:
                        continue
                    try:
                        raw = operation()
                        remaining_ms()
                        break
                    except zmq.error.Again:
                        continue
            response = msgpack.unpackb(raw, raw=False)
            remaining_ms()
            if isinstance(response, dict) and "error" in response:
                raise GraspGenXError(f"graspgenx server error: {response['error']}")
            return response
        except BaseException:
            # A late response cannot be matched to the next planning batch.
            self.close()
            raise

    def probe(self, timeout_ms: int = 300, *, deadline=None, check=None) -> dict:
        """One SHORT round trip at startup: is a GraspGen-X server (or the
        protocol stub) answering? Raises GraspGenXError otherwise. Without
        this every grasp paid the full inference timeout (8 s) before
        falling back to the analytic planner -- and the fallback was a
        buried memory note, so a demo without the server looked like one
        with it, just slow.

        Wire: the real server (graspgenx/serving/zmq_server.py) answers
        `{"action": "health"}` with `{"status": "ok"}`; the repo's protocol
        stub answers the same (and marks itself `"stub": true`)."""
        kwargs = {} if deadline is None else {"deadline": deadline, "check": check}
        resp = self.request({"action": "health"}, timeout_ms=timeout_ms, **kwargs)
        ok = isinstance(resp, dict) and (resp.get("status") == "ok" or resp.get("ok") is True)
        if not ok:
            raise GraspGenXError(f"graspgenx health returned {resp!r}")
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
            host=os.environ.get("CASCADE_GRASPGENX_HOST", str(get("host", "127.0.0.1"))),
            port=int(os.environ.get("CASCADE_GRASPGENX_PORT", get("port", 5556))),
            timeout_ms=int(get("timeout_ms", 8000)),
        )
        self.gripper = str(get("gripper", "franka_panda"))
        self.tip_offset_m = float(get("tip_offset_m", 0.10))
        self.num_grasps = int(get("num_grasps", 100))
        self.topk = int(get("topk", 32))
        self.min_score = float(get("min_score", 0.0))
        self.approach_z_max = float(get("approach_z_max", 0.2))
        self.required = bool(get("required", False))
        self.planner = str(get("planner", "graspmoe"))
        if self.planner not in {"diffusion", "graspmoe"}:
            raise ValueError(f"unknown GraspGen-X planner: {self.planner}")
        self.probe_timeout_ms = int(get("probe_timeout_ms", 300))
        self.last_branch_counts: dict[str, int] = {}
        self.last_latency_s: float | None = None
        #: what the startup probe saw: {"ok": True, "stub": bool, ...} or None
        self.status: dict | None = None
        # Cross-embodiment mode: a `sweep` block describes OUR gripper by
        # its swept volume (12 numbers) -- no name lookup, no borrowed
        # Franka. Config volumes use the JAW CENTER; the wire uses the model's
        # GRIPPER BASE. Translate conditioning by the same tip offset used
        # to translate returned poses back. Applying only the return offset
        # describes a different gripper to the generator/discriminator.
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
            for key in ("offset_open", "offset_mid"):
                self.sweep_params[key][2] += self.tip_offset_m
            self.sweep_params["fingertip_depth"] += self.tip_offset_m

    def probe(self, timeout_ms: int | None = None, *, deadline=None, check=None) -> dict:
        kwargs = {} if deadline is None else {"deadline": deadline, "check": check}
        self.status = self._client.probe(timeout_ms=self.probe_timeout_ms if timeout_ms is None else timeout_ms, **kwargs)
        if self.required and self.status.get("stub"):
            self.status = None
            raise GraspGenXError("real GraspGen-X required; analytic protocol stub rejected")
        return self.status

    def describe(self) -> str:
        if self.status is None:
            return "graspgenx (unprobed)"
        return "graspgenx-stub (analytic protocol double)" if self.status.get("stub") else "graspgenx (learned 6-DoF)"

    @staticmethod
    def _validate_bounded_response(response):
        """Malformed batches are terminal, even when every score is low."""
        try:
            if not isinstance(response, dict):
                raise ValueError("expected a mapping")
            poses = np.asarray(response["grasps"])
            scores = np.asarray(response["confidences"])
            if (poses.ndim != 3 or poses.shape[1:] != (4, 4)
                    or scores.shape != (len(poses),)
                    or poses.dtype.kind not in "fiu" or scores.dtype.kind not in "fiu"
                    or not np.isfinite(poses).all() or not np.isfinite(scores).all()):
                raise ValueError("expected finite numeric (N,4,4) poses and (N,) scores")
            R = poses[:, :3, :3].astype(float)
            # Float32 model output; the same rigid-transform tolerance as
            # observed RGBD calibration, never a collision-margin adjustment.
            if (not np.all(poses[:, 3, :] == [0, 0, 0, 1])
                    or not np.all(np.linalg.det(R) > 0)
                    or not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=3e-6, rtol=0)):
                raise ValueError("non-rigid grasp transform")
        except (KeyError, ValueError, TypeError) as exc:
            raise GraspGenXError(f"malformed GraspGen-X batch: {exc}") from exc

    def plan(self, fix: ObjectFix, max_width_m: float = 0.09, *, deadline=None, check=None) -> list[Grasp]:
        """Segmented base-frame object points -> ranked wrc Grasps."""
        if check is not None:
            check()
        source_points = fix.points
        tensor_points = hasattr(source_points, "detach")
        # ZMQ/msgpack is a host-memory transport between independent Python
        # processes. Perception stays on CUDA; explicitly copy only the
        # segmented request cloud at this serialization boundary.
        pts = np.asarray(source_points.detach().cpu().numpy() if tensor_points
                         else source_points, dtype=np.float32)
        evidence.array("ggx_request_points_base_m", pts)
        evidence.event("ggx_configuration", gripper=self.gripper, tip_offset_m=self.tip_offset_m,
                       num_grasps=self.num_grasps, topk=self.topk, min_score=self.min_score,
                       approach_z_max=self.approach_z_max, planner=self.planner,
                       sweep_params=self.sweep_params, status=self.status)
        if pts.shape[0] < 50:
            raise GraspGenXError(f"only {pts.shape[0]} object points (<50)")
        t0 = time.monotonic()
        if self.sweep_params is not None:
            payload = {
                "action": "infer_object",
                "point_cloud": pts,
                "sweep_volume_params": self.sweep_params,
                "num_grasps": self.num_grasps,
                "planner": self.planner,
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
        for _attempt in range(2 if deadline is None else 1):
            if check is not None:
                check()
            kwargs = {} if deadline is None else {"deadline": deadline, "check": check}
            resp = self._client.request(payload, **kwargs)
            if check is not None:
                check()
            evidence.ggx_response(resp, _attempt)
            if deadline is not None:
                self._validate_bounded_response(resp)
            poses = np.asarray(resp["grasps"], dtype=np.float32).reshape(-1, 4, 4)
            scores = np.asarray(resp["confidences"], dtype=np.float32).reshape(-1)
            if len(scores) != len(poses) or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores)):
                raise GraspGenXError("invalid/non-finite GraspGen-X poses or scores")
            tags = resp.get("branch_tags", [])
            if self.required and self.sweep_params is not None:
                if len(tags) != len(poses) or (self.planner == "diffusion" and any(t != "diff" for t in tags)):
                    raise GraspGenXError("missing or incorrect learned grasp provenance")
            self.last_branch_counts = {tag: tags.count(tag) for tag in set(tags)}
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
                (f"graspgenx returned no grasps twice (cloud dumped: {dump})" if deadline is None
                 else f"graspgenx returned no grasps in bounded batch (cloud dumped: {dump})")
            )

        # Jaw-opening width from the object's span along each grasp's jaw
        # axis (the wire protocol does not carry per-grasp widths; the
        # runtime's air-grasp verification and select_grasp need one).
        grasps: list[Grasp] = []
        order = np.argsort(-scores)
        for i in order:
            if check is not None:
                check()
            if scores[i] < self.min_score:
                continue
            T = poses[i].astype(float)
            R, p = T[:3, :3], T[:3, 3]
            approach = R[:, 2] / np.linalg.norm(R[:, 2])
            # Tabletop sanity: the model sees a floating cloud (no table), so
            # it happily proposes approaches from below. Those are physically
            # unreachable here -- drop them before wasting IK attempts.
            if approach[2] > self.approach_z_max:
                continue
            open_axis = R[:, 0] / np.linalg.norm(R[:, 0])
            R_wrc = np.column_stack([approach, open_axis, np.cross(approach, open_axis)])
            tcp = p + approach * self.tip_offset_m
            if tensor_points:
                import torch

                device, dtype = source_points.device, source_points.dtype
                center = torch.as_tensor(fix.position, device=device, dtype=dtype)
                axis = torch.as_tensor(open_axis, device=device, dtype=dtype)
                span = (source_points - center) @ axis
                bounds = torch.quantile(span, torch.tensor([.05, .95], device=device, dtype=dtype))
                width = min(float((bounds[1] - bounds[0]).item()) + .015, max_width_m)
                if device.type == "cuda":
                    from ..perception.cuda_math import record
                    record("graspgenx_width", source_points, span)
            else:
                span = (pts - fix.position) @ open_axis
                width = float(min(np.quantile(span, .95) - np.quantile(span, .05) + .015,
                                  max_width_m))
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
            if len(grasps) >= self.topk:
                break
        if deadline is not None:
            vertical = poses[:, 2, 2] / np.linalg.norm(poses[:, :3, 2], axis=1)
            evidence.event("ggx_filtered_counts", raw=len(scores),
                score_pass=int(np.count_nonzero(scores >= self.min_score)),
                vertical_pass=int(np.count_nonzero(vertical <= self.approach_z_max)),
                both_pass=int(np.count_nonzero((scores >= self.min_score) & (vertical <= self.approach_z_max))),
                retained=len(grasps), min_score=self.min_score, approach_z_max=self.approach_z_max)
        if not grasps:
            error = GraspGenXError if deadline is None else NoEligibleGrasps
            raise error("no GraspGen-X candidates satisfy the tabletop/score constraints")
        if check is not None:
            check()
        return grasps
