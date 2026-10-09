#!/usr/bin/env python3
"""Run the upstream learned server with CUDA identity in its health reply.

The first inference of a fresh process is slow: measured on this x86 rig
(RTX PRO 6000 Blackwell, torch 2.7.0+cu128, 9 Oct 2026) the first
`infer_object` of a 5 cm cube took 15.53 s and every later one 0.09 s. The
CASCADE client waits 8 s (`grasp.graspgenx.timeout_ms`), so the first pick of
every session fell back to the analytic OBB planner (B36 signature 4). The
server therefore runs one synthetic inference BEFORE it binds its port
(`--no-warmup` restores the old start-up): a client that can connect gets a
warm model, and a launcher that waits for the port waits for the warm-up too.

The warm-up is advisory. A warm-up that raises or returns no grasps is logged
loudly and the server binds cold, exactly as before this change (requests
that fail return their error and the client falls back per request); the
`health` reply says `warmed_up: false` with the `warmup_error`. A host
without CUDA or without a loadable model still refuses before any warm-up,
as it always did; `check_graspgenx.py` stays the launcher's fail-closed gate.
"""
import argparse
import platform
import sys
import time

#: Swept volume of the reBot RS jaw with the default `tip_offset_m` folded in,
#: as `graspgenx_backend` sends it (configs/demo.yaml `grasp.graspgenx.sweep`).
#: The warm-up only needs SOME valid volume: the model is gripper-independent.
WARMUP_SWEEP = {
    "extents_open": [0.090, 0.020, 0.045],
    "offset_open": [0.0, 0.0, -0.0225 + 0.098],
    "extents_mid": [0.045, 0.020, 0.045],
    "offset_mid": [0.0, 0.0, -0.0225 + 0.098],
    "gripper_type": 0,
    "fingertip_depth": 0.098,
}


def warmup_cloud(side: float = 0.05, per_face: int = 700, seed: int = 0):
    """Surface points of a cube resting on a table (bottom face hidden)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            p = rng.uniform(-side / 2, side / 2, size=(per_face, 3))
            p[:, axis] = sign * side / 2
            faces.append(p)
    cloud = np.concatenate(faces) + np.array([0.0, 0.0, side / 2])
    return cloud[cloud[:, 2] > 1e-3].astype(np.float32)


def warmup_request() -> dict:
    return {"action": "infer_object", "point_cloud": warmup_cloud(),
            "sweep_volume_params": dict(WARMUP_SWEEP), "num_grasps": 100}


def warm(server) -> float:
    """One synthetic inference through the server's own dispatch; returns
    its wall time. Raises if the model fails or returns no grasps (the
    caller logs it and binds cold)."""
    t0 = time.monotonic()
    reply = server._dispatch(warmup_request())
    if not isinstance(reply, dict) or "grasps" not in reply:
        raise RuntimeError(f"no grasps in the reply: {str(reply)[:300]}")
    return time.monotonic() - t0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--default-gripper")
    parser.add_argument("--no-warmup", action="store_true",
                        help="bind before the first inference (the old start-up)")
    args = parser.parse_args(argv)
    import torch
    from graspgenx.serving.zmq_server import GraspGenXZMQServer

    if not torch.cuda.is_available():
        raise RuntimeError("GraspGen-X requires CUDA; CPU fallback is disabled")

    class CudaServer(GraspGenXZMQServer):
        def _dispatch(self, request):
            reply = super()._dispatch(request)
            if request.get("action") == "health":
                device = next(self._shared_model.parameters()).device
                if device.type != "cuda":
                    raise RuntimeError(f"GraspGen-X model is on {device}, expected CUDA")
                reply.update(learned=True, device=str(device),
                             gpu=torch.cuda.get_device_name(device),
                             capability=list(torch.cuda.get_device_capability(device)),
                             torch=torch.__version__, cuda=torch.version.cuda,
                             architecture=platform.machine(),
                             warmed_up=self._warmup_s is not None,
                             warmup_s=self._warmup_s,
                             warmup_error=self._warmup_error)
            return reply

    server = CudaServer(config_path=args.config, assets_dir=args.assets_dir,
                        host=args.host, port=args.port,
                        default_gripper=args.default_gripper)
    server._warmup_s = None
    server._warmup_error = None
    if args.no_warmup:
        print(f"[graspgenx] warm-up skipped (--no-warmup): binding {args.host}:{args.port} "
              "cold", flush=True)
    else:
        try:
            server._warmup_s = round(warm(server), 3)
        except Exception as exc:  # noqa: BLE001 -- advisory: fail open to the old start-up
            server._warmup_error = f"{type(exc).__name__}: {exc}"[:300]
            print(f"[graspgenx] WARNING: warm-up inference failed ({server._warmup_error}); "
                  f"binding {args.host}:{args.port} COLD as before: the first request may "
                  "outlast the client timeout and fall back to the analytic planner",
                  file=sys.stderr, flush=True)
        else:
            print(f"[graspgenx] warm-up inference {server._warmup_s:.2f} s before binding "
                  f"{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
