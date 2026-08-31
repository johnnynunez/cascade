#!/usr/bin/env python
"""A GraspGen-X ZMQ server that runs anywhere -- no CUDA, no checkpoints.

    python scripts/serve_graspgenx_stub.py [--port 5556] [--gripper so101]

WHY THIS EXISTS. `configs/demo.yaml` defaults `grasp.backend: graspgenx`, but
the real server (`scripts/serve_graspgenx.sh`) needs an NVIDIA GPU, a separate
venv with its own torch, and downloaded checkpoints. On any machine without
those -- a laptop, a CPU booth box, CI -- the client raises and the runtime
silently falls back to the OBB planner. That fallback is correct, but it means
the ENTIRE GraspGen-X code path (wire protocol, pose conversion, tip offset,
approach filtering, width estimation, re-ranking) is never exercised outside
the DGX, and a mistake in it would only ever surface at a demo.

This speaks the same wire protocol and returns geometrically sensible grasps
computed analytically from the point cloud, so the path can be run and tested
on any machine. It is a TEST DOUBLE, not a model:

  - it does NOT learn anything and its `confidences` are a geometric heuristic;
  - it will never find the clever grasp a diffusion model finds.

Use it to validate wiring and to develop the client; use the real server for
grasp QUALITY. The response format is identical, so swapping is a port change.

Protocol (mirrors graspgenx.serving; see grasping/graspgenx_backend.py):

  request   {"action": "infer"|"infer_object"|"ping", "point_cloud": (N,3),
             "gripper_name"|"sweep_volume_params": ..., "num_grasps": int, ...}
  response  {"grasps": (K,4,4) float32, "confidences": (K,) float32}
            or {"error": "..."} which the client raises as GraspGenXError.

Grasp frame convention returned here is GraspGen-X's OWN (not cascade's):
+Z = approach, +X = jaw closing axis, origin at the GRIPPER BASE, i.e. one
`tip_offset_m` behind the jaw centre. Getting this wrong is the single most
likely bug in the client, which is exactly why the stub emits the real
convention rather than a convenient one.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _principal_axes(pts: np.ndarray):
    """(centre, axes as columns, extents) of the cloud's oriented box."""
    centre = pts.mean(axis=0)
    centred = pts - centre
    cov = np.cov(centred.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    proj = centred @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    centre = centre + axes @ ((mins + maxs) / 2)
    return centre, axes, maxs - mins


def plan_grasps(points: np.ndarray, num_grasps: int = 32,
                tip_offset_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Analytic top-down grasps around the cloud's vertical axis.

    Returns (K,4,4) poses in the GRIPPER-BASE frame and (K,) confidences.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    centre, axes, extents = _principal_axes(pts)
    top_z = float(pts[:, 2].max())

    # Grasp a little below the top face: closing at the very top slips off.
    grasp_z = top_z - 0.3 * max(float(extents[2]), 0.01)
    approach = np.array([0.0, 0.0, -1.0])          # straight down

    poses, confs = [], []
    # Sweep the jaw yaw; prefer closing across the object's NARROW horizontal
    # axis, which is what a real planner converges to.
    horiz = []
    for i in range(3):
        a = axes[:, i]
        if abs(a[2]) < 0.7:                        # roughly horizontal axis
            horiz.append((float(extents[i]), a / np.linalg.norm(a)))
    # Sort on the extent ONLY. Sorting the tuples lets Python fall through to
    # comparing the numpy axis vectors whenever two extents tie -- which is
    # exactly what a symmetric object (a cube: 0.0339 == 0.0339) produces --
    # and `array < array` raises "truth value of an array is ambiguous".
    # A hand-built asymmetric cloud never hits it; the real perception cloud
    # of a cube hits it every time.
    horiz.sort(key=lambda t: t[0])
    best_axis = horiz[0][1] if horiz else np.array([1.0, 0.0, 0.0])
    best_yaw = float(np.arctan2(best_axis[1], best_axis[0]))

    n = max(1, int(num_grasps))
    for k in range(n):
        # Spread candidates around the best yaw, best first.
        offset = (k // 2 + 1) * (np.pi / max(n, 8)) * (1 if k % 2 else -1)
        yaw = best_yaw + (0.0 if k == 0 else offset)
        close = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        third = np.cross(approach, close)
        # GraspGen-X frame: +X = closing, +Y = third, +Z = approach.
        R = np.column_stack([close, third, approach])
        # Origin at the GRIPPER BASE: back off along -approach from the jaw
        # centre by tip_offset_m (the client adds it straight back).
        jaw_centre = np.array([centre[0], centre[1], grasp_z])
        base = jaw_centre - approach * tip_offset_m
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = base
        poses.append(T)
        # Confidence: 1.0 on the narrow axis, decaying with yaw deviation.
        confs.append(float(np.clip(1.0 - abs(offset) / np.pi, 0.05, 1.0)))

    return (np.asarray(poses, dtype=np.float32),
            np.asarray(confs, dtype=np.float32))


def serve(port: int, gripper: str, verbose: bool = True) -> None:
    try:
        import msgpack
        import msgpack_numpy
        import zmq
    except ImportError as e:
        print(f"needs the grasping extra: uv pip install -e '.[grasping]' ({e})",
              file=sys.stderr)
        raise SystemExit(2) from e
    msgpack_numpy.patch()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{port}")
    if verbose:
        print(f"[graspgenx-stub] listening on tcp://127.0.0.1:{port} "
              f"(gripper={gripper}) -- ANALYTIC, not the learned model")

    while True:
        try:
            req = msgpack.unpackb(sock.recv(), raw=False)
        except KeyboardInterrupt:
            break
        action = req.get("action", "infer")
        try:
            if action == "ping":
                sock.send(msgpack.packb({"ok": True, "stub": True},
                                        use_bin_type=True))
                continue
            pts = np.asarray(req["point_cloud"], dtype=np.float64).reshape(-1, 3)
            if pts.shape[0] < 3:
                raise ValueError(f"point cloud has {pts.shape[0]} points")
            # BOTH actions return a GRIPPER-BASE pose. The real server keeps
            # its base-frame convention even when given a centred sweep box --
            # that is precisely why `tip_offset_m: 0.098` in configs/demo.yaml
            # is annotated "even with centered sweep boxes", and why the client
            # applies the offset unconditionally. An earlier version of this
            # stub returned a jaw-centre pose for `infer_object`, which made
            # the client's (correct) offset push every grasp 9.8 cm BELOW the
            # table. Match the real server, not what seems tidier.
            poses, confs = plan_grasps(
                pts, num_grasps=int(req.get("num_grasps", 32)),
                tip_offset_m=0.098,
            )
            if verbose:
                print(f"[graspgenx-stub] {action}: {pts.shape[0]} pts -> "
                      f"{poses.shape[0]} grasps")
            sock.send(msgpack.packb(
                {"grasps": poses, "confidences": confs}, use_bin_type=True))
        except Exception as e:  # noqa: BLE001 - report like the real server
            sock.send(msgpack.packb({"error": f"{type(e).__name__}: {e}"},
                                    use_bin_type=True))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GraspGen-X protocol stub (analytic, runs without CUDA)"
    )
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--gripper", default="so101")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    serve(a.port, a.gripper, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
