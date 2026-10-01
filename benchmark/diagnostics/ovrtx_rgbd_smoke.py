#!/usr/bin/env python3
"""Real RTX smoke; no robot transport, simulator process, or actuation.

Run in its own process under an external timeout (initial shader compilation
can block in native code). All input geometry is authored here, in metres.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxError, OvrtxRenderer, SceneSnapshot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_depth(camera, cube_world, size=.5, pixel_offset=0.):
    """Analytic ray/box intersection, preserving ray Z parameter (not radial)."""
    v, u = np.mgrid[:camera.height, :camera.width]
    rays = np.stack(((u + pixel_offset - camera.K[0, 2]) / camera.fx,
                     (v + pixel_offset - camera.K[1, 2]) / camera.fy, np.ones_like(u)), -1)
    to_cube = np.linalg.inv(cube_world) @ camera.T_base_cam
    direction = rays @ to_cube[:3, :3].T
    origin = to_cube[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (-size/2 - origin) / direction
        b = (size/2 - origin) / direction
    near = np.minimum(a, b).max(-1)
    far = np.maximum(a, b).min(-1)
    return np.where((far >= near) & (near > 0), near, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0, help="CUDA-visible ordinal")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"version": 1, "pass": False, "scope": "isolated renderer RGBD, no physics integration",
               "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
               "pid": os.getpid(), "device": args.device,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "started_monotonic": time.monotonic(), "frames": [], "errors": [], "cleanup_errors": [],
               "packages": {p: importlib.metadata.version(p) for p in ("ovrtx", "ovstage", "numpy")},
               "source": {p: sha(ROOT / p) for p in (
                   "src/cascade/sim/ovrtx_renderer.py", "src/cascade/perception/ovrtx_camera.py",
                   "benchmark/diagnostics/ovrtx_rgbd_smoke.py")}}
    def save():
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    save()
    renderer = None
    try:
        receipt["gpu_inventory"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], text=True)
        scene = args.output / "scene.usda"
        scene.write_text('''#usda 1.0
(
 metersPerUnit = 1
 upAxis = "Z"
)
def Xform "World" {
 def Xform "Parent" {
  matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(.2,.1,0,1))
  uniform token[] xformOpOrder = ["xformOp:transform"]
  def Cube "Cube" {
   double size = .5
   color3f[] primvars:displayColor = [(.8,.1,.05)]
   matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(-.2,-.1,0,1))
   uniform token[] xformOpOrder = ["xformOp:transform"]
  }
 }
 def DomeLight "Light" {
  float inputs:intensity = 700
 }
}
''')
        pose = np.diag([1., -1., -1., 1.])
        pose[2, 3] = 3.
        try:
            CameraSpec("unsupported", 160, 120, 180., 135., pose)
        except OvrtxError:
            receipt["unequal_focal_lengths_rejected"] = True
        else:
            raise ValueError("Unvalidated unequal focal lengths were accepted")
        cameras = [CameraSpec("square", 160, 120, 180., 180., pose),
                   CameraSpec("wide", 160, 120, 135., 135., pose)]
        renderer = OvrtxRenderer(scene, cameras, dynamic_paths=["/World/Parent/Cube"], device=args.device)
        parent = np.eye(4)
        parent[:3, 3] = [.2, .1, 0]
        start = time.monotonic()
        for index, local_position in enumerate(([-.2, -.1, 0], [.1, -.02, .4]), 1):
            local = np.eye(4)
            local[:3, 3] = local_position
            moving_camera = pose.copy()
            moving_camera[:2, 3] = [0.08 * (index - 1), -.05 * (index - 1)]
            if index == 2:
                angle = .18
                local[:3, :3] = [[np.cos(angle), 0, np.sin(angle)],
                                     [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
                angle = .12
                ry = np.array([[np.cos(angle), 0, np.sin(angle)],
                               [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
                moving_camera[:3, :3] = ry @ pose[:3, :3]
            # A single caller snapshot supplies BOTH scene and camera transforms.
            camera_poses = {c.name: moving_camera.copy() for c in cameras}
            camera_poses["wide"][0, 3] += .15
            snapshot = SceneSnapshot("analytic-scene", "run-1", index, index / 30,
                time.monotonic(), {"/World/Parent/Cube": local},
                camera_poses, {"cube_size_m": .5})
            frames = renderer.render(snapshot)
            repeat = renderer.render(snapshot)
            for c in cameras:
                frame = frames[c.name]
                reference = expected_depth(replace(c, T_base_cam=camera_poses[c.name]), parent @ local)
                half_pixel_reference = expected_depth(replace(c, T_base_cam=camera_poses[c.name]),
                                                      parent @ local, pixel_offset=.5)
                hit, observed = reference > 0, frame.depth_m > 0
                # Boundary pixels can be antialiased; evaluate interior metric
                # depth and permit ONE raster edge pixel, never a metric offset.
                interior = hit.copy()
                for axis in (0, 1):
                    interior &= np.roll(hit, 1, axis) & np.roll(hit, -1, axis)
                pixels = np.argwhere(observed)
                predicted = np.argwhere(hit)
                if not pixels.size or not predicted.size or not interior.any():
                    raise ValueError("No finite rendered/reference depth")
                bbox_error = max(abs(pixels.min(0) - predicted.min(0)).max(),
                                 abs(pixels.max(0) - predicted.max(0)).max())
                depth_error = float(np.max(abs(frame.depth_m[interior] - reference[interior])))
                alternate_error = float(np.max(abs(frame.depth_m[interior] - half_pixel_reference[interior])))
                duplicate = (frame.capture == repeat[c.name].capture and frame.t == repeat[c.name].t
                             and frame.frame_id == repeat[c.name].frame_id
                             and all(np.array_equal(getattr(frame, k), getattr(repeat[c.name], k))
                                     for k in ("rgb", "depth_m", "K", "T_base_cam")))
                stem = f"{index:02d}-{c.name}"
                np.savez_compressed(args.output / f"{stem}.npz", rgb=frame.rgb, depth_m=frame.depth_m,
                                    K=frame.K, T_base_cam=frame.T_base_cam, expected_depth=reference)
                Image.fromarray(frame.rgb[..., ::-1]).save(args.output / f"{stem}.png")
                record = {"camera": c.name, "snapshot": index, "capture": frame.capture,
                          "bbox_error_px": int(bbox_error), "max_interior_depth_error_m": depth_error,
                          "diagnostic_half_pixel_depth_error_m": alternate_error,
                          "primary_projection": "integer pixel indices and Frame.K, same as Cascade backprojection",
                          "valid_depth_pixels": int(observed.sum()), "duplicate_whole_packet": duplicate,
                          "pass": bool(bbox_error <= 1 and depth_error <= 1e-4 and duplicate)}
                receipt["frames"].append(record)
                save()
        receipt["render_elapsed_s"] = time.monotonic() - start
        receipt["pass"] = len(receipt["frames"]) == 4 and all(f["pass"] for f in receipt["frames"])
    except BaseException as e:
        receipt["errors"].append(f"{type(e).__name__}: {e}")
        receipt["pass"] = False
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as e:
                receipt["cleanup_errors"].append(str(e))
                receipt["pass"] = False
        receipt["finished_monotonic"] = time.monotonic()
        receipt["source_unchanged"] = all(sha(ROOT / p) == h for p, h in receipt["source"].items())
        receipt["pass"] = receipt["pass"] and receipt["source_unchanged"]
        receipt["artifacts"] = {p.name: sha(p) for p in args.output.iterdir() if p.is_file() and p.name != "receipt.json"}
        save()
    print(json.dumps({"pass": receipt["pass"], "receipt": str(args.output / "receipt.json"),
                      "errors": receipt["errors"], "cleanup_errors": receipt["cleanup_errors"]}))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
