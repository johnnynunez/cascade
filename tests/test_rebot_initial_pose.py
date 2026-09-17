"""The RS ready pose faces forward and survives an Isaac editor restart.

FK and the real safety gates run offline; Kit boundaries are stubbed only
for the restart contract, which does not certify simulated dynamics.
"""

import ast
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import REPO, load_isaac_bridge_definitions, needs_pin

from cascade.config import load_demo_config, load_profile
from cascade.safety.harness import SafetyHarness, SafetyLimits


@needs_pin
@pytest.mark.parametrize("profile", ["mock", "isaac", "rebot_rs", "rebot_rs_mb"])
def test_rebot_home_faces_forward_inside_its_safety_envelope(profile):
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm=profile)
    kin = Kinematics(cfg.arm.model, cfg.arm.ee_frame,
                     joint_signs=cfg.arm.joint_signs)
    home = np.asarray(cfg.arm.home_q, dtype=float)
    pose = kin.fk(home)

    # In the Seeed ready pose the fingers extend forward and open sideways.
    # Checking both axes catches a sideways wrist and a rolled gripper.
    np.testing.assert_allclose(pose[:3, 0], [1.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(pose[:3, 1], [0.0, 1.0, 0.0], atol=1e-4)

    # vet_pose enforces margins, TCP workspace and table/link clearance
    # without approve()'s recovery exceptions for an already-invalid pose.
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    assert harness.vet_pose(home) is None

    nearby = home + np.array([0.02, -0.03, 0.04, 0.02, -0.01, 0.02])
    solved = kin.ik(pose, q_init=nearby, retries=0)
    assert solved.success, f"home IK failed: {solved.error}"
    np.testing.assert_allclose(kin.fk(solved.q), pose, atol=2e-4)
    assert harness.vet_pose(solved.q) is None


def _bridge_startup_state(no_targets):
    """Execute only pose declarations, without importing or starting Kit."""
    path = REPO / "scripts" / "isaac_bridge.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(target, ast.Name) and target.id in {"HOME_Q", "_targets"}
               for target in targets):
            nodes.append(node)
    env = {"_NO_TARGETS": no_targets}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def test_bridge_home_matches_isaac_profile_in_asset_joint_convention():
    cfg = load_profile("arms", "isaac")
    expected_asset_q = np.asarray(cfg.home_q) * np.asarray(cfg.joint_signs)
    env = _bridge_startup_state(no_targets=False)
    np.testing.assert_allclose(env["HOME_Q"], expected_asset_q, atol=1e-12)
    np.testing.assert_allclose(env["_targets"]["q"], expected_asset_q, atol=1e-12)


@pytest.mark.parametrize("no_targets", [False, True], ids=["ready", "asset-inspection"])
def test_editor_restart_preserves_initial_pose_mode(monkeypatch, no_targets):
    env = _bridge_startup_state(no_targets)
    startup_q = env["_targets"]["q"]
    startup_grip = env["_targets"]["grip_frac"]
    if no_targets:
        assert startup_q is None
        assert startup_grip is None
    else:
        assert startup_grip == 1.0

    prims = ModuleType("isaacsim.core.experimental.prims")
    prims.RigidPrim = Mock()
    monkeypatch.setitem(sys.modules, prims.__name__, prims)
    articulation = object()
    env.update({
        "args": SimpleNamespace(prim="/test/robot"),
        "Articulation": Mock(return_value=articulation),
        "_init_wrist_cam": Mock(),
        "_state_lock": threading.Lock(),
        "_PROP_SPAWNS": {},
    })
    load_isaac_bridge_definitions({"_resume_scene"}, env)

    # A previous command must not override the configured startup mode.
    env["_targets"].update(q=[0.1] * 6, grip_frac=0.25, stopped=True)
    env["_resume_scene"]()

    assert env["art"] is articulation
    env["Articulation"].assert_called_once_with("/test/robot")
    env["_init_wrist_cam"].assert_called_once_with()
    assert env["_targets"]["stopped"] is False
    assert env["_targets"]["grip_frac"] == startup_grip
    if no_targets:
        assert env["_targets"]["q"] is None
    else:
        np.testing.assert_allclose(env["_targets"]["q"], startup_q, atol=1e-12)
