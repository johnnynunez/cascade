#!/usr/bin/env python3
"""Finite, read-only CUDA perception probe; never constructs an arm or moves it.

The live result uses real neural detections of RTX RGB-D frames. CPU reference
geometry is evaluated only as a numeric parity test, outside demo execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


@contextmanager
def cpu_reference():
    previous = os.environ.get("CASCADE_REQUIRE_CUDA")
    os.environ["CASCADE_REQUIRE_CUDA"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CASCADE_REQUIRE_CUDA", None)
        else:
            os.environ["CASCADE_REQUIRE_CUDA"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Evidence must stay inside the authorized project")
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"pass": False, "kind": "read-only live CUDA perception and numerical parity",
               "started_unix": time.time(), "pid": os.getpid(), "motion_commands": 0,
               "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (ROOT/"src/cascade/perception").glob("*.py")}}
    lock = open("/tmp/cascade-kitchen-gpu.lock", "a+")
    deadline = time.monotonic() + 30
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU experiment lock unavailable for 30 seconds")
            time.sleep(.1)
    receipt["gpu_lock_held"] = True
    try:
        os.environ.update(CASCADE_REQUIRE_CUDA="1", CASCADE_DEVICE="cuda:0",
                          CASCADE_GPU_PERCEPTION_EVIDENCE_DIR=str(out))
        import cv2
        import numpy as np
        import torch
        from cascade.sim.bridge_client import BridgeClient
        from cascade.config import load_profile
        from cascade.perception.detector import OpenVocabDetector
        from cascade.perception import colors, grounding, pixel_target, visual_diff, probe
        from cascade.perception import cuda_math as gpu
        from cascade.grasping import obb_grasp
        from cascade.skills.runtime import SkillRuntime
        from types import SimpleNamespace

        client = BridgeClient(host="127.0.0.1", port=8611, timeout_s=8)
        try:
            client.connect()
            receipt["ping"] = client.ping()
            frame = client.observation("cam0")
            later = client.observation("side")
        finally:
            client.close()
        assert frame.has_depth
        if frame.T_base_cam is None:
            cfg = load_profile("cameras", "isaac")
            frame.T_base_cam = grounding.Extrinsics.from_config(cfg.extrinsics).cam_to_base()
            receipt["static_camera_extrinsics_sha256"] = hashlib.sha256((ROOT/"configs/cameras/isaac.yaml").read_bytes()).hexdigest()
        np.savez_compressed(out/"live-rgbd.npz", rgb=frame.rgb, depth=frame.depth_m,
                            K=frame.K, transform=frame.T_base_cam)
        cv2.imwrite(str(out/"live-worktop.jpg"), frame.rgb)
        cv2.imwrite(str(out/"live-side.jpg"), later.rgb)
        receipt["capture"] = {"shape": list(frame.rgb.shape), "t": frame.t,
                              "depth_source": frame.depth_source,
                              "depth_sha256": hashlib.sha256(frame.depth_m.tobytes()).hexdigest()}
        os.chdir(ROOT/"models")  # existing text encoder resolves its installed relative asset here
        detector = OpenVocabDetector(str(ROOT/"models/yoloe-11s-seg.pt"), classes=["pink cube"], device="cuda:0")
        started = time.monotonic()
        detections = detector.detect(frame, classes=["pink cube"])
        torch.cuda.synchronize()
        receipt["target_detector_seconds"] = time.monotonic() - started
        candidates = [d for d in detections if d.mask is not None and int(d.mask.sum()) >= 24]
        assert candidates, "No actual learned object mask in live RTX frame"
        det = max(candidates, key=lambda d: d.conf)
        receipt["detections"] = [{"label": d.label, "conf": d.conf, "bbox": d.bbox.tolist(),
            "mask_device": str(d.mask.device) if d.mask is not None else None} for d in detections]
        assert det.mask.device.type == "cuda"
        mask_cpu = det.mask.detach().cpu().numpy()

        # Disable random subsampling here: compare exactly the same measured
        # pixels/point cloud, then exercise production sampling in localization.
        cloud = grounding.mask_to_points_cam(frame, det.mask, max_points=1_000_000)
        transformed = gpu.transform(frame.T_base_cam, cloud)
        center, extents, axes = grounding.oriented_bbox(transformed)
        adjusted = grounding._recentre_by_size(center, transformed, extents, frame.T_base_cam[:3, 3])
        color = colors.mask_color(frame.rgb, det.mask, max_px=1_000_000)
        fix = pixel_target.fix_from_mask(frame, frame.T_base_cam, det.mask)
        with cpu_reference():
            reference_cloud = grounding.mask_to_points_cam(frame, mask_cpu, max_points=1_000_000)
            reference_transformed = reference_cloud @ frame.T_base_cam[:3, :3].T + frame.T_base_cam[:3, 3]
            ref_center, ref_extents, ref_axes = grounding.oriented_bbox(reference_transformed)
            ref_adjusted = grounding._recentre_by_size(ref_center, reference_transformed, ref_extents, frame.T_base_cam[:3, 3])
            ref_color = colors.mask_color(frame.rgb, mask_cpu, max_px=1_000_000)
            ref_fix = pixel_target.fix_from_mask(frame, frame.T_base_cam, mask_cpu)
        parity = {
            "backprojection_max_abs_m": float(np.max(np.abs(cloud.cpu().numpy()-reference_cloud))),
            "center_max_abs_m": float(np.max(np.abs(center-ref_center))),
            "extent_max_abs_m": float(np.max(np.abs(extents-ref_extents))),
            "recenter_max_abs_m": float(np.max(np.abs(adjusted-ref_adjusted))),
            "pixel_center_max_abs_m": float(np.max(np.abs(fix.position-ref_fix.position))),
            "pixel_extent_max_abs_m": float(np.max(np.abs(fix.extent-ref_fix.extent))),
            "color": color, "cpu_reference_color": ref_color,
        }
        assert all(value < 1e-6 for name, value in parity.items() if name.endswith("_m")), parity
        assert color == ref_color, parity

        pixels = np.random.default_rng(20260912).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        hsv = gpu.hsv8(pixels).cpu().numpy()
        parity["hsv_max_uint8_error"] = int(np.max(np.abs(hsv-cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).astype(int))))
        assert parity["hsv_max_uint8_error"] == 0, parity
        before, after = frame.rgb, np.roll(frame.rgb, 1, axis=1)
        changed = visual_diff._changed_fraction(before, after)
        depth_probe = probe._median_depth(frame.depth_m, frame.rgb.shape[1]//2, frame.rgb.shape[0]//2)
        with cpu_reference():
            ref_changed = visual_diff._changed_fraction(before, after)
            ref_depth_probe = probe._median_depth(frame.depth_m, frame.rgb.shape[1]//2, frame.rgb.shape[0]//2)
        parity["visual_change_abs_error"] = abs(changed-ref_changed)
        parity["probe_depth_abs_error_m"] = abs(depth_probe-ref_depth_probe)
        assert parity["visual_change_abs_error"] < 1e-12 and parity["probe_depth_abs_error_m"] < 1e-6

        localized = grounding.localize_object(frame, "pink cube", SimpleNamespace(detect=lambda *args, **kwargs: detections),
            grounding.Extrinsics(mode="eye_to_hand", T=frame.T_base_cam))
        remembered = gpu.remember_cloud(localized.points)
        belief = SimpleNamespace(position=localized.position, points=remembered, label="pink cube", conf=det.conf)
        recalled = SkillRuntime._fix_from_belief("pink cube", belief)
        assert recalled.points.device.type == "cuda"
        receipt["localization"] = {"position_m": localized.position.tolist(), "extent_m": localized.extent.tolist(),
            "point_device": str(localized.points.device), "remembered_device": str(remembered.device),
            "recalled_device": str(recalled.points.device), "physical_truth_claimed": False}
        # Planning computes candidate geometry only; it never creates a robot.
        grasps = obb_grasp.plan_grasps_from_fix(localized, table_z=0.)
        receipt["non_actuating_grasp_candidates"] = len(grasps)
        open_world = detector.detect(frame, classes=None)
        receipt["open_world_detections"] = len(open_world)
        receipt["parity"] = parity
        receipt["nvidia_telemetry"] = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        receipt["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        receipt["pass"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_unix"] = time.time()
        (out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(json.dumps({"pass": receipt["pass"], "out": str(out), "parity": receipt["parity"]}))


if __name__ == "__main__":
    main()
