"""Replay of two real rebot_grasp runs on Seeed's B601 rig through the
camera-frame planner.

Ported from WRC `tests/test_camera_grasp_replay.py`. The captured runs (Orbbec
Gemini 2 overhead camera, reBot RS arm, a yellow banana; both ended in
"[Grasp] Holding object" on 2026-08-07):

  run A: centre (596, 402) px, depth 0.642 m, position_cam (-0.045, 0.040,
         0.642), jaw 38.9 mm, length 160.9 mm, angle 4.14 deg
  run B: centre (612, 395) px, depth 0.648 m, position_cam (-0.031, 0.034,
         0.648), jaw 40.3 mm, length 160.4 mm, angle 0.00 deg

K is the overhead Gemini 2's colour intrinsics (WRC
data/calibration/orbbec_overhead.npz). T_cam2base is the rig's hand-eye solve
shipped as the reference extrinsic of configs/cameras/orbbec_overhead.yaml.

NOT ported: WRC's base-frame position/pregrasp/approach replays and its IK
check. WRC itself marks them xfail -- the recorded base-frame poses predate a
30 mm finger change and a re-calibration (2026-09-02), so they no longer match
any T on disk. What still binds to the recording is asserted here: the
camera-frame back-projection and the jaw width (both independent of T), and
the line-of-sight geometry on the real tilted mount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from cascade.config import load_profile
from cascade.grasping.camera_grasp import plan_grasp_from_mask

K_GEMINI2 = np.array([[691.01849365, 0.0, 644.87585449],
                      [0.0, 691.27801514, 358.46099854],
                      [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Run:
    name: str
    center_px: tuple[int, int]
    depth_m: float
    position_cam: tuple[float, float, float]
    length_m: float
    width_m: float
    angle_deg: float


RUNS = [
    Run("run_A", (596, 402), 0.642, (-0.045, 0.040, 0.642), 0.1609, 0.0389, 4.14),
    Run("run_B", (612, 395), 0.648, (-0.031, 0.034, 0.648), 0.1604, 0.0403, 0.00),
]


@pytest.fixture(scope="module")
def T_rig():
    return np.asarray(load_profile("cameras", "orbbec_overhead").extrinsics.T, float)


def _scene(run: Run):
    """The captured object as a yawed rectangle mask with uniform depth."""
    H, W = 720, 1280
    cx, cy = run.center_px
    yaw = math.radians(run.angle_deg)
    fx = K_GEMINI2[0, 0]
    half_l, half_w = run.length_m * fx / run.depth_m / 2, run.width_m * fx / run.depth_m / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = xx - cx, yy - cy
    rx = math.cos(yaw) * dx + math.sin(yaw) * dy
    ry = -math.sin(yaw) * dx + math.cos(yaw) * dy
    mask = (np.abs(rx) <= half_l) & (np.abs(ry) <= half_w)
    depth = np.zeros((H, W), np.float32)
    depth[mask] = run.depth_m
    return mask, depth


@pytest.mark.parametrize("run", RUNS, ids=[r.name for r in RUNS])
def test_jaw_width_matches_the_recorded_jaw_plus_the_pad(run, T_rig):
    mask, depth = _scene(run)
    (g,) = plan_grasp_from_mask(mask, depth, K_GEMINI2, T_rig, insertion_depth_m=0.0)
    assert g.width_m == pytest.approx(run.width_m + 0.015, abs=2e-3)


@pytest.mark.parametrize("run", RUNS, ids=[r.name for r in RUNS])
def test_camera_frame_point_matches_rebot_grasps_recorded_position(run, T_rig):
    """Undo the lift: the planned surface point, in the camera frame, is the
    position_cam rebot_grasp logged for that run (to its 1 mm print precision)."""
    mask, depth = _scene(run)
    (g,) = plan_grasp_from_mask(mask, depth, K_GEMINI2, T_rig, insertion_depth_m=0.0)
    p_cam = np.linalg.inv(T_rig) @ np.r_[g.position, 1.0]
    np.testing.assert_allclose(p_cam[:3], run.position_cam, atol=1.5e-3)


@pytest.mark.parametrize("run", RUNS, ids=[r.name for r in RUNS])
def test_on_the_tilted_rig_mount_the_dive_follows_the_camera_ray(run, T_rig):
    """The rig camera is tilted ~32 deg: the approach is the camera ray to the
    object (~29 deg off vertical here), not a snapped top-down vector."""
    mask, depth = _scene(run)
    (g,) = plan_grasp_from_mask(mask, depth, K_GEMINI2, T_rig, insertion_depth_m=0.015)
    surface = g.position - g.approach * 0.015
    ray = surface - T_rig[:3, 3]
    np.testing.assert_allclose(g.approach, ray / np.linalg.norm(ray), atol=1e-6)
    tilt = math.degrees(math.acos(-g.approach[2]))
    assert 25.0 < tilt < 33.0, tilt
    np.testing.assert_allclose(g.position - surface, 0.015 * g.approach, atol=1e-9)
