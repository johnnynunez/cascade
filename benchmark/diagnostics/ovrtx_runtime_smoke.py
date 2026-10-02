#!/usr/bin/env python3
"""Native owned-process OVRTX snapshot/mask/binding smoke, no robot actuation."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from cascade.sim.ovrtx_process import OvrtxProcess
from cascade.sim.ovrtx_renderer import CameraSpec, SceneSnapshot, OvrtxError
from cascade.sim.render_binding import valid_render_binding
from isaac_ovrtx import bridge_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Interpreter with pinned OVRTX SDK")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.output / "scene.usda"
    scene.write_text('''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" {
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(1,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
    def Cube "Robot" {
        double size = .3
        color3f[] primvars:displayColor = [(0.1,0.4,0.9)]
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(-.3,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
def Xform "World_Props" {
    def Cube "target" {
        double size = .3
        color3f[] primvars:displayColor = [(0.9,0.1,0.05)]
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(.3,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
def DomeLight "Dome" {
    float inputs:intensity = 1000
}
def DistantLight "Key" {
    float inputs:intensity = 1500
}
''')
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 2.
    camera = CameraSpec("cam0", 320, 240, 280., 280., pose)
    paths = ["/World/Robot", "/World_Props/target"]
    receipt = {"version": 1, "pass": False, "scope": "native snapshot renderer and bridge packet; synthetic physical state",
               "parent_pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "device": args.device, "frames": [], "started_monotonic": time.monotonic(),
               "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
               "source": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "src/cascade/sim/ovrtx_renderer.py",
                                    ROOT / "src/cascade/sim/ovrtx_masks.py",
                                    ROOT / "src/cascade/sim/ovrtx_process.py",
                                    ROOT / "src/cascade/sim/ovrtx_worker.py",
                                    ROOT / "src/cascade/sim/render_binding.py",
                                    ROOT / "scripts/isaac_ovrtx.py", Path(__file__)]}}
    owner = None
    try:
        owner = OvrtxProcess(args.python, dict(scene=str(scene), cameras=[camera], dynamic_paths=paths,
                            semantic_paths={p: p for p in paths}, world_paths=paths, device=args.device))
        receipt["worker_pid"] = owner.process.pid
        captures = []
        for step in (1, 2):
            t = time.monotonic()
            state = {"version": 1, "backend": "isaac", "robot_id": paths[0],
                     "producer_epoch": "native-smoke", "joint_convention": "asset", "q": [step * .1] * 6,
                     "t": t, "time_source": "physics_loop_monotonic"}
            contact = dict(error=None, tracking=True, paths=[], scene_prop_paths=[paths[1]])
            transforms = {p: np.eye(4) for p in paths}
            transforms[paths[0]][0, 3] = -.3
            transforms[paths[1]][:3, 3] = [.3 + (step-1)*.15, 0, (step-1)*.25]
            snapshot = SceneSnapshot("native-smoke", "native-smoke", step, step / 60, t,
                transforms, {"cam0": pose}, dict(proprioception=state, contact_state=contact,
                                                snapshot_finished_monotonic=time.monotonic()))
            frame = owner.render(snapshot)["cam0"]
            packet = bridge_packet(frame, snapshot, robot_id=paths[0], base_z=0.)
            capture = {"t": packet["t"], "proprioception": packet["proprioception"],
                       "render_reference": packet["render_reference"]}
            assert valid_render_binding(capture)
            assert capture["proprioception"]["q"] == state["q"]
            mask = frame.prop_masks[paths[1]]
            robot = frame.prop_masks[paths[0]]
            Image.fromarray(frame.rgb[..., ::-1]).save(args.output / f"frame-{step}.png")
            Image.fromarray(mask.astype(np.uint8)*255).save(args.output / f"mask-{step}.png")
            np.savez(args.output / f"frame-{step}.npz", depth=frame.depth_m, mask=mask, robot=robot)
            receipt["last_frame_summary"] = {"mask": int(mask.sum()), "robot": int(robot.sum()),
                                             "semantic_labels": frame.capture.get("semantic_labels"),
                                             "valid_depth": int((frame.depth_m > 0).sum()),
                                             "mean_rgb": float(frame.rgb.mean())}
            assert mask.sum() > 500 and robot.sum() > 500
            assert not np.any(mask & robot)
            assert np.median(np.where(robot)[1]) < camera.K[0, 2], "parent composed twice over reset body"
            measured = float(np.median(frame.depth_m[mask]))
            expected = 2. - transforms[paths[1]][2, 3] - .15
            assert abs(measured - expected) < .01
            assert frame.rgb.mean() > 1 and np.isfinite(frame.depth_m).all()
            repeated = owner.render(snapshot)["cam0"]
            assert repeated.t == frame.t and repeated.capture == frame.capture
            np.testing.assert_array_equal(repeated.rgb, frame.rgb)
            Image.fromarray(frame.rgb[..., ::-1]).save(args.output / f"frame-{step}.png")
            Image.fromarray(mask.astype(np.uint8)*255).save(args.output / f"mask-{step}.png")
            row = dict(step=step, snapshot_sha256=frame.capture["scene_state_sha256"],
                       target_pixels=int(mask.sum()), robot_pixels=int(robot.sum()),
                       median_depth_m=measured, expected_depth_m=expected,
                       render_reference=packet["render_reference"])
            receipt["frames"].append(row)
            captures.append(capture)
        stale = replace(snapshot, sequence=1, sim_time=1/60)
        try:
            owner.render(stale)
        except OvrtxError:
            receipt["stale_rejected"] = True
        else:
            raise AssertionError("Regressed physical snapshot was accepted")
        receipt["pass"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if owner is not None:
            owner.close()
            receipt["worker_exit_code"] = owner.process.returncode
        receipt["finished_monotonic"] = time.monotonic()
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
