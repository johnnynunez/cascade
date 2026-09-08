"""The always-on robot matrix (2026-09-03 requirement): AgileX PiPER,
SO-101, reBot dev arm, Unitree H1 and H1-2 must ALL flow through the one
loader/factory/safety stack in every suite run -- no robot is "the" robot.

Layers, cheapest first:
  1. profile completeness through the REAL loader (extends resolved),
  2. kinematics: the vendored URDF loads, FK at home_q lands where the
     profile says it does, chain order pins the n_controlled slice contract,
  3. E2E on the mock transport: a motion skill drives each robot through
     build_runtime -> SafetyHarness -> mock backend; the humanoids (no hand)
     must REFUSE a grasp honestly instead of miming one.

"Unitree H2" note: no public H2 model exists (unitree_ros / Menagerie,
checked 2026-09-03); the second-generation H1 ships as H1-2 and that is
what the h1_2 profiles drive. See assets/urdf/h1_2/PROVENANCE.md.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from conftest import has_pinocchio

from cascade.apps.demo import build_runtime, shutdown_runtime
from cascade.config import load_demo_config

needs_pin = pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")

#: profile -> (n_joints, ee_frame, gripper max_width_m)
MATRIX = {
    "mock":       (6, "gripper_end", 0.09),   # reBot dev arm (RS model)
    "so101_mock": (5, "gripper_frame_link", 0.055),
    "piper_mock": (6, "tcp_link", 0.065),     # AgileX PiPER
    "h1_mock":    (4, "tcp_link", 0.0),       # Unitree H1 right arm (no hand)
    "h1_2_mock":  (7, "tcp_link", 0.0),       # Unitree H1-2 right arm (flange)
}


@pytest.mark.parametrize("name", sorted(MATRIX))
def test_profile_completeness_through_the_real_loader(name):
    """Every matrix robot declares its own DOF, keyframes and gripper --
    via the loader (extends resolved), never raw YAML."""
    n, ee, width = MATRIX[name]
    cfg = load_demo_config(camera="mock", arm=name, llm="mock")
    a = cfg.arm
    assert a.type == "mock"
    assert int(a.n_joints) == n
    assert a.get("ee_frame") == ee
    assert len(a.get("home_q")) == n
    assert len(a.get("handover_q")) == n
    assert len(a.get("joint_signs")) == n
    g = a.get("gripper")
    assert g is not None and float(g.get("max_width_m")) == pytest.approx(width)
    # every robot must bring its own workspace: the reBot's box only for the
    # reBot (regression guard on forgotten `overrides:`)
    ws_max = [float(v) for v in cfg.safety.workspace.max]
    if name == "mock":
        assert ws_max == [0.50, 0.30, 0.55]
    else:
        assert ws_max != [0.50, 0.30, 0.55], f"{name} inherited the reBot workspace"


@needs_pin
@pytest.mark.parametrize("name", sorted(MATRIX))
def test_kinematics_load_and_home_pose(name):
    """The vendored URDF builds, the arm chain occupies the FIRST n q
    indices (the kinematics slice contract), and FK(home_q) is a sane
    in-workspace TCP -- catches a swapped chain or a mis-sized keyframe."""
    from cascade.control.kinematics import Kinematics

    n, ee, _ = MATRIX[name]
    cfg = load_demo_config(camera="mock", arm=name, llm="mock")
    a = cfg.arm
    kin = Kinematics(a.model, ee_frame=ee, n_controlled=n,
                     joint_signs=a.get("joint_signs"),
                     ik_task_weights=a.get("ik_task_weights"))
    lo, hi = kin.joint_limits
    home = np.asarray(a.get("home_q"), dtype=float)
    assert home.shape == (n,)
    assert np.all(home >= lo - 1e-9) and np.all(home <= hi + 1e-9), (
        f"{name} home_q outside its own URDF limits")
    T = kin.fk(home)
    p = T[:3, 3]
    ws_min = np.asarray([float(v) for v in cfg.safety.workspace.min])
    ws_max = np.asarray([float(v) for v in cfg.safety.workspace.max])
    assert np.all(p >= ws_min - 1e-6) and np.all(p <= ws_max + 1e-6), (
        f"{name} FK(home_q)={np.round(p, 3).tolist()} outside its own "
        f"workspace {ws_min.tolist()}..{ws_max.tolist()}")


def _wait_for_belief(runtime, label="red object", timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.beliefs.find(label) is not None:
            return True
        time.sleep(0.05)
    return False


@needs_pin
@pytest.mark.parametrize("name", sorted(MATRIX))
def test_e2e_motion_on_every_robot(name, tmp_path):
    """build_runtime + one real motion per robot on the mock transport.

    wave exercises profile keyframes -> kinematics -> harness -> backend on
    every chain length without depending on scene geometry; the gripper
    check then splits by embodiment: arms with jaws must pick, the handless
    humanoid arms must refuse with a reasoned error (never mime a grasp).
    """
    camera = "mock_small" if name in ("so101_mock", "piper_mock") else "mock"
    cfg = load_demo_config(camera=camera, arm=name, llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    # E2E convention: the kinematic mock's jaws jam at 0.5 = an object is in
    # the gripper. Without it every close reads as an air grasp.
    arm.object_stop_frac = 0.5
    try:
        res = runtime.execute("wave", {})
        assert res.get("ok"), f"{name}: wave failed: {res}"

        _, _, width = MATRIX[name]
        if width == 0.0:
            assert _wait_for_belief(runtime), f"{name}: mock object never fused"
            res = runtime.execute("grasp_object", {"label": "red object"})
            assert res.get("ok") is False, (
                f"{name} has no hand but claimed a grasp: {res}")
        else:
            assert _wait_for_belief(runtime), f"{name}: mock object never fused"
            res = runtime.execute("pick_and_place", {"object": "red object"})
            assert res.get("ok"), f"{name}: pick_and_place failed: {res}"
    finally:
        shutdown_runtime(runtime, arm)
