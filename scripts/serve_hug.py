#!/usr/bin/env python3
"""HUG (Human Universal Grasping) behind CASCADE's REQ/REP grasp protocol.

    # the learned model, inside HUG's own environment (Python 3.10, CUDA):
    python scripts/serve_hug.py --checkpoint <hug clone>/checkpoints/hug_full.safetensors
    # an analytic protocol double: no torch, no weights, no MANO, any host
    python scripts/serve_hug.py --stub

HUG (Wu et al., arXiv:2606.17054; github.com/KevinyWu/hug @ 8d1c52d, MIT;
weights HF kevinywu/hug @ 1415c9e, MIT) ships inference and a click-to-grasp
app, not a server. The real engine below runs HUG's DOCUMENTED path for one
RGB-D frame and one query pixel -- `hug.prepare_inputs.prepare_pkl`
(centre-crop + 224 px resize + K), `GraspDataset.get_inference_data`, the
body of `hug.app.handle_click` (query depth from the 224 px depth, point
cloud cropped around the query at the model's radius), `model.sample`
and `mano_params_to_grasp_dict` -- and answers with camera-frame hands.
CASCADE's extensions, labelled as such:

  - N samples of ONE query per request (one batched `model.sample` call; the
    paper's limitations section names "generating many candidates and
    selecting the best" as the natural extension, the app samples one);
  - `crop: query` (opt-in) pre-crops the square that contains the query
    pixel before HUG's own centre crop, for objects outside the centre;
  - refusals instead of silent clamps: a query outside the crop, a query
    without depth, CUDA absent without an explicit `--device cpu`.

Wire (msgpack + msgpack-numpy, same shape as GraspGen-X's server):

  request   {"action": "health"}
            {"action": "infer", "rgb": (H,W,3) uint8 RGB, "depth_mm": (H,W)
             uint16 millimetres (0 = invalid), "K": (3,3) at the RGB
             resolution, "query_px": [u, v], "num_samples": N,
             "crop": "center" | "query"}
  response  {"landmarks_3d": (N,21,3) f32 camera frame (OpenCV), metres,
             manotorch SNAP order; "T_camera_wrist": (N,4,4) f32;
             "mano_params": (N,99) f32 (real engine); "query": what the
             server received and how it mapped the query; "stub": bool;
             "latency_s": float}
            or {"error": "..."}, which the client raises as HugError.

HUG has no confidence score; the reply carries none. MANO (needed by HUG's
hand layer) is licensed per user at https://mano.is.tue.mpg.de/ and is NOT
redistributable: the operator installs it into the HUG clone; CASCADE never
downloads, vendors or commits it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TARGET_SIZE = 224
DEFAULT_PORT = 5558
N_LANDMARKS = 21
MAX_SAMPLES = 256
HUG_COMMIT = "8d1c52d4c24bfae5a369e32e3f134f5601a02630"
#: HF kevinywu/hug @ 1415c9e, hug_full.safetensors (241,359,400 B), LFS sha256
PINNED_CHECKPOINT_SHA256 = "515b5c3bc7987739aec019e754c15df5fbf3eff9daefb93924da098ae4bd1eae"
MANO_URL = "https://mano.is.tue.mpg.de/"
CROP_MODES = ("center", "query")


class RequestError(ValueError):
    """The request cannot be answered as asked; reported, never clamped."""


# ── preprocessing geometry (pure numpy; mirrors hug.prepare_inputs) ──────────


def crop_geometry(height: int, width: int, mode: str = "center", query=None):
    """-> (x_off, y_off, size) of the square HUG will see.

    `center` is HUG's own `_center_crop_square`: the shorter side, centred.
    `query` (CASCADE's extension) shifts the same-size square to contain the
    query pixel, as centred on it as the image allows."""
    size = min(int(height), int(width))
    if mode == "center":
        return (int(width) - size) // 2, (int(height) - size) // 2, size
    if mode == "query":
        u, v = (float(c) for c in query)
        x_off = int(np.clip(int(round(u)) - size // 2, 0, int(width) - size))
        y_off = int(np.clip(int(round(v)) - size // 2, 0, int(height) - size))
        return x_off, y_off, size
    raise RequestError(f"unknown crop mode {mode!r} ({' | '.join(CROP_MODES)})")


def adjust_K(K, x_off: float, y_off: float, scale: float) -> np.ndarray:
    """HUG's `_adjust_K`: shift the principal point by the crop, then scale."""
    K_new = np.asarray(K, dtype=np.float64).copy()
    K_new[0, 2] -= x_off
    K_new[1, 2] -= y_off
    K_new[:2, :] *= scale
    return K_new


def query_to_224(u: float, v: float, x_off: int, y_off: int, size: int):
    """Full-resolution query pixel -> HUG's 224 px grid, with the SAME
    arithmetic as K (so both describe one camera ray). A query outside the
    square is invisible to the model: refuse rather than clamp."""
    du, dv = float(u) - x_off, float(v) - y_off
    if not (0.0 <= du < size and 0.0 <= dv < size):
        raise RequestError(
            f"query pixel ({u:.0f}, {v:.0f}) lies outside HUG's {size}x{size} crop at "
            f"x={x_off}, y={y_off}: the object is not visible to the model "
            "(crop: query keeps it in view)")
    scale = TARGET_SIZE / float(size)
    return du * scale, dv * scale


def validate_infer_request(req: dict) -> dict:
    """Type/shape checks plus the query mapping, shared by both engines."""
    try:
        rgb = np.asarray(req["rgb"])
        depth = np.asarray(req["depth_mm"])
        K = np.asarray(req["K"], dtype=np.float64)
        u, v = (float(c) for c in req["query_px"])
        n = int(req.get("num_samples", 1))
        mode = str(req.get("crop", "center"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RequestError(f"malformed infer request: {exc}") from exc
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise RequestError(f"rgb must be (H,W,3) uint8, got {rgb.shape} {rgb.dtype}")
    if depth.shape != rgb.shape[:2] or depth.dtype != np.uint16:
        raise RequestError(f"depth_mm must be (H,W) uint16 matching rgb, got "
                           f"{depth.shape} {depth.dtype}")
    if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise RequestError("K must be a finite (3,3) intrinsic matrix")
    h, w = depth.shape
    if not (np.isfinite(u) and np.isfinite(v) and 0 <= u < w and 0 <= v < h):
        raise RequestError(f"query pixel ({u}, {v}) is outside the {w}x{h} image")
    if not 1 <= n <= MAX_SAMPLES:
        raise RequestError(f"num_samples must be in 1..{MAX_SAMPLES}")
    x_off, y_off, size = crop_geometry(h, w, mode, (u, v))
    u224, v224 = query_to_224(u, v, x_off, y_off, size)
    depth_mm = int(depth[int(v), int(u)])
    if depth_mm <= 0:
        raise RequestError(f"no valid depth at the query pixel ({u:.0f}, {v:.0f})")
    return {
        "rgb": rgb, "depth_mm": depth, "K": K, "query_px": (u, v), "num_samples": n,
        "crop": mode, "crop_box": (x_off, y_off, size), "px_224": (u224, v224),
        "K_224": adjust_K(K, x_off, y_off, TARGET_SIZE / float(size)),
        "echo": {"px": [u, v], "px_224": [u224, v224], "crop": [x_off, y_off, size],
                 "rgb": [int(c) for c in rgb[int(v), int(u)]], "depth_mm": depth_mm},
    }


# ── policy: device, MANO licence, pinned weights ─────────────────────────────


def resolve_device(requested: str, cuda_available: bool) -> str:
    """CUDA unless the operator EXPLICITLY asks for the CPU. HUG's own app
    silently picks the CPU when CUDA is absent; this server refuses that."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_available:
            raise RuntimeError(
                "HUG inference requires CUDA and no CUDA device is visible; CPU "
                "fallback is disabled (pass --device cpu to run on the CPU explicitly)")
        return requested
    raise ValueError(f"--device must be cuda, cuda:N or cpu (got {requested!r})")


def check_mano(mano_models_folder) -> Path:
    """MANO is required by HUG's hand layer and is NOT redistributable."""
    path = Path(mano_models_folder) / "models" / "MANO_RIGHT.pkl"
    if not path.is_file():
        raise RuntimeError(
            f"MANO right-hand model missing at {path}. MANO is licensed per user "
            f"(register at {MANO_URL}, non-commercial research licence) and is not "
            "redistributable: download it yourself and copy the contents of "
            "mano_v*_*/ into the HUG clone's assets/mano_models/. CASCADE never "
            "downloads, vendors or commits it.")
    return path


def verify_checkpoint(path, expected_sha256: str) -> str:
    """sha256 of the weights; refuses anything but the pinned release unless
    the operator passes --checkpoint-sha256 '' (reported as unpinned)."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"HUG checkpoint not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(f"HUG checkpoint {path} sha256 {actual} != pinned {expected_sha256}")
    return actual


# ── engines ───────────────────────────────────────────────────────────────────


class StubEngine:
    """Analytic protocol double. NOT the model: every hand is a synthetic
    pinch 1.5 cm behind the observed query point along the camera ray, palm
    toward the camera, jaw yaw spread over the samples. It exists so the
    whole client path (wire, BGR->RGB, mm depth, frames, pinch mapping,
    ranking, fail-visible rules) runs on any machine."""

    stub = True
    APERTURE_M = 0.04
    DEPTH_M = 0.015

    def health(self) -> dict:
        return {"status": "ok", "ok": True, "stub": True, "learned": False, "model": "hug-stub"}

    def infer(self, req: dict) -> dict:
        u, v = req["query_px"]
        K = req["K"]
        d = req["echo"]["depth_mm"] / 1000.0
        q = np.array([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])
        r = q / np.linalg.norm(q)                       # camera ray: palm -> pinch
        x0 = np.array([1.0, 0.0, 0.0]) - r[0] * r
        x0 /= np.linalg.norm(x0)
        c = q + self.DEPTH_M * r
        n = req["num_samples"]
        hands, wrists = [], []
        for k in range(n):
            th = np.pi * k / n
            x = np.cos(th) * x0 + np.sin(th) * np.cross(r, x0)
            y = np.cross(r, x)
            L = np.zeros((N_LANDMARKS, 3))
            L[0] = c - 0.10 * r
            thumb, index = c - self.APERTURE_M / 2 * x, c + self.APERTURE_M / 2 * x
            for i, f in zip((1, 2, 3), (0.25, 0.5, 0.75)):
                L[i] = L[0] + f * (thumb - L[0])
            L[4] = thumb
            for mcp, off in ((5, 0.03), (9, 0.01), (13, -0.01), (17, -0.03)):
                L[mcp] = c - 0.07 * r + off * y
                tip = index if mcp == 5 else c + 0.01 * x - 0.02 * r + off * y
                for j in (1, 2, 3):
                    L[mcp + j] = L[mcp] + j / 3 * (tip - L[mcp])
            hands.append(L)
            T = np.eye(4)
            T[:3, :3] = np.column_stack([x, y, r])
            T[:3, 3] = L[0]
            wrists.append(T)
        return {"landmarks_3d": np.asarray(hands, np.float32),
                "T_camera_wrist": np.asarray(wrists, np.float32)}


class HugEngine:
    """The learned model, through HUG's own preprocessing and sampling code.
    Needs HUG's environment (`pip install -e <clone>`), the weights and the
    operator's MANO model."""

    stub = False

    def __init__(self, checkpoint, device: str = "cuda", sampling_steps: int = 1,
                 expected_sha256: str = PINNED_CHECKPOINT_SHA256, use_ema: bool = True):
        import torch
        from hug.utils import data_keys

        self._torch = torch
        self.root = Path(data_keys.PROJECT_ROOT)
        check_mano(data_keys.MANO_MODELS_FOLDER)
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = verify_checkpoint(self.checkpoint, expected_sha256)
        self.pinned = bool(expected_sha256)
        from hug.dataloader.grasp_dataset import GraspDataset
        from hug.inference import load_model, resolve_checkpoint_path
        from hug.models.mano import mano_params_to_grasp_dict
        from hug.prepare_inputs import prepare_pkl
        from hug.utils.pcl_utils import depth_to_pcl_tensors, pixel_to_xyz

        self._prepare_pkl, self._to_grasp = prepare_pkl, mano_params_to_grasp_dict
        self._depth_to_pcl, self._pixel_to_xyz = depth_to_pcl_tensors, pixel_to_xyz
        self.device = device
        self.sampling_steps = int(sampling_steps)
        self.model = load_model(resolve_checkpoint_path(self.checkpoint), use_ema, device)
        actual = next(self.model.parameters()).device
        if device != "cpu" and actual.type != "cuda":
            raise RuntimeError(f"HUG model is on {actual}, expected CUDA")
        self.use_rgb = getattr(self.model, "use_rgb", True)
        self.use_depth = getattr(self.model, "use_depth", False)
        self.pcl_use_rgb = getattr(self.model, "pcl_use_rgb", False)
        # One private folder for the per-request pkl HUG's dataset reads.
        self._workdir = Path(tempfile.mkdtemp(prefix="cascade-hug-"))
        self._dataset = GraspDataset(str(self._workdir), split="val",
                                     use_rgb=self.use_rgb, use_depth=self.use_depth)
        try:
            self.hug_commit = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            self.hug_commit = None

    def close(self):
        shutil.rmtree(self._workdir, ignore_errors=True)

    def health(self) -> dict:
        torch = self._torch
        actual = next(self.model.parameters()).device
        reply = {"status": "ok", "ok": True, "stub": False, "learned": True, "model": "hug",
                 "device": str(actual), "torch": torch.__version__,
                 "checkpoint": str(self.checkpoint),
                 "checkpoint_sha256": self.checkpoint_sha256, "pinned": self.pinned,
                 "hug_root": str(self.root), "hug_commit": self.hug_commit,
                 "hug_commit_expected": HUG_COMMIT, "sampling_steps": self.sampling_steps,
                 "architecture": platform.machine()}
        if actual.type == "cuda":
            reply.update(gpu=torch.cuda.get_device_name(actual), cuda=torch.version.cuda)
        return reply

    def infer(self, req: dict) -> dict:
        torch = self._torch
        x_off, y_off, size = req["crop_box"]
        rgb, depth, K = req["rgb"], req["depth_mm"], req["K"]
        if req["crop"] == "query":
            # CASCADE's extension: hand HUG the square that contains the query;
            # its own centre crop of a square is then the identity.
            rgb = rgb[y_off:y_off + size, x_off:x_off + size]
            depth = depth[y_off:y_off + size, x_off:x_off + size]
            K = adjust_K(K, x_off, y_off, 1.0)
        self._prepare_pkl(np.ascontiguousarray(rgb), np.ascontiguousarray(depth),
                          np.asarray(K, dtype=np.float64), "query", self._workdir)
        sample = self._dataset.get_inference_data("query")
        K_224 = np.asarray(sample["camera_K"], dtype=np.float64)
        if not np.allclose(K_224, req["K_224"], rtol=0.0, atol=1e-6):
            raise RuntimeError("CASCADE's query mapping disagrees with HUG's preprocessing "
                               f"(K_224 {K_224.tolist()} vs {req['K_224'].tolist()})")
        u224, v224 = req["px_224"]
        # hug.app.handle_click: the query depth comes from HUG's 224 px depth.
        depth_image = sample["depth_image"]
        dh, dw = depth_image.shape[:2]
        du = int(np.clip(u224 * dw / float(TARGET_SIZE), 0, dw - 1))
        dv = int(np.clip(v224 * dh / float(TARGET_SIZE), 0, dh - 1))
        depth_m = float(depth_image[dv, du]) / 1000.0
        if not depth_m > 0.0:
            raise RequestError("no valid depth at the query pixel after HUG's 224 px resize")
        dev = self.device
        point_uv = torch.tensor([[u224, v224, depth_m]], dtype=torch.float32, device=dev)
        camera_K = torch.from_numpy(K_224).float().unsqueeze(0).to(dev)
        rgb_t = sample["rgb"].unsqueeze(0).to(dev) if self.use_rgb else None
        pcl_xyz = pcl_rgb = None
        if self.use_depth:
            radius = getattr(self.model, "pcl_crop_radius", None)
            if radius is not None:
                centre = self._pixel_to_xyz(u224, v224, depth_m, K_224)
                xyz, colours = self._depth_to_pcl(
                    depth_image.astype(np.float32) / 1000.0, sample["rgb_original"], K_224,
                    center=centre, crop_radius=radius)
                pcl_xyz = xyz.unsqueeze(0).to(dev)
                pcl_rgb = colours.unsqueeze(0).to(dev) if self.pcl_use_rgb else None
            else:
                pcl_xyz = sample["pcl_xyz"].unsqueeze(0).to(dev)
                pcl_rgb = sample["pcl_rgb"].unsqueeze(0).to(dev) if self.pcl_use_rgb else None
        n = req["num_samples"]

        def batch(t):   # one query, N independent noise draws (our extension)
            return None if t is None else t.repeat(n, *([1] * (t.dim() - 1)))

        with torch.no_grad():
            preds = self.model.sample(batch(point_uv), batch(camera_K), steps=self.sampling_steps,
                                      rgb=batch(rgb_t), pcl_xyz=batch(pcl_xyz),
                                      pcl_rgb=batch(pcl_rgb))
            betas = self.model.fixed_betas.squeeze(0)
            hands = [self._to_grasp(preds[i], betas, self.model.mano, K_224, self.model.mesh_faces)
                     for i in range(n)]
        return {"landmarks_3d": np.stack([np.asarray(h["landmarks_3d"], np.float32) for h in hands]),
                "T_camera_wrist": np.stack([np.asarray(h["T_camera_wrist"], np.float32)
                                            for h in hands]),
                "mano_params": preds.float().cpu().numpy().astype(np.float32),
                "depth_m_224": depth_m}


# ── protocol ──────────────────────────────────────────────────────────────────


def handle(engine, request) -> dict:
    """One request -> one reply; every failure is an explicit error reply."""
    try:
        if not isinstance(request, dict):
            raise RequestError("request must be a mapping")
        action = request.get("action", "infer")
        if action in ("health", "ping"):
            return engine.health()
        if action != "infer":
            raise RequestError(f"unknown action {action!r}")
        req = validate_infer_request(request)
        t0 = time.monotonic()
        reply = dict(engine.infer(req))
        reply.update(query=req["echo"], stub=bool(engine.stub),
                     latency_s=round(time.monotonic() - t0, 3))
        return reply
    except Exception as exc:  # noqa: BLE001 -- reported to the client, like GraspGen-X
        return {"error": f"{type(exc).__name__}: {exc}"}


def serve(engine, host: str = "127.0.0.1", port: int = DEFAULT_PORT, verbose: bool = True):
    try:
        import msgpack
        import msgpack_numpy
        import zmq
    except ImportError as e:
        print(f"serve_hug needs pyzmq + msgpack-numpy ({e})", file=sys.stderr)
        raise SystemExit(2) from e
    msgpack_numpy.patch()
    sock = zmq.Context.instance().socket(zmq.REP)
    if int(port) == 0:
        port = sock.bind_to_random_port(f"tcp://{host}")
    else:
        sock.bind(f"tcp://{host}:{int(port)}")
    kind = "ANALYTIC STUB, not the learned model" if engine.stub else "learned HUG"
    print(f"[hug-serve] listening on tcp://{host}:{port} ({kind})", flush=True)
    while True:
        try:
            raw = sock.recv()
        except KeyboardInterrupt:
            break
        try:
            request = msgpack.unpackb(raw, raw=False)
        except Exception as exc:  # noqa: BLE001
            request = None
            reply = {"error": f"undecodable request: {exc}"}
        else:
            reply = handle(engine, request)
        if verbose and isinstance(request, dict) and request.get("action") == "infer":
            print(f"[hug-serve] infer: {reply.get('error') or len(reply['landmarks_3d'])} "
                  f"({reply.get('latency_s', '-')} s)", file=sys.stderr, flush=True)
        sock.send(msgpack.packb(reply, use_bin_type=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="HUG grasp server for CASCADE (REQ/REP msgpack)")
    ap.add_argument("--stub", action="store_true",
                    help="analytic protocol double (no torch, weights or MANO)")
    ap.add_argument("--checkpoint", help="hug_full.safetensors (HF kevinywu/hug)")
    ap.add_argument("--checkpoint-sha256", default=PINNED_CHECKPOINT_SHA256,
                    help="expected sha256 of the weights; '' disables the pin")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (cpu only explicitly)")
    ap.add_argument("--sampling-steps", type=int, default=1,
                    help="Euler steps (HUG's app/inference default since 8d1c52d: 1)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CASCADE_HUG_PORT", DEFAULT_PORT)),
                    help="0 binds a free port and prints it")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.stub:
        engine = StubEngine()
    else:
        if not args.checkpoint:
            ap.error("--checkpoint is required (or --stub)")
        try:
            import torch
        except ImportError as e:
            print(f"serve_hug: torch is missing; run inside HUG's environment ({e})",
                  file=sys.stderr)
            raise SystemExit(2) from e
        try:
            device = resolve_device(args.device, bool(torch.cuda.is_available()))
        except (RuntimeError, ValueError) as e:
            print(f"serve_hug: {e}", file=sys.stderr)
            raise SystemExit(2) from e
        try:
            engine = HugEngine(args.checkpoint, device=device,
                               sampling_steps=args.sampling_steps,
                               expected_sha256=args.checkpoint_sha256)
        except (RuntimeError, ImportError) as e:
            print(f"serve_hug: {e}", file=sys.stderr)
            raise SystemExit(2) from e
    serve(engine, args.host, args.port, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
