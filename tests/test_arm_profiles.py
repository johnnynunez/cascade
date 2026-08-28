"""Arm profiles describe the robot; nothing else may assume a specific one.

This repo drives a reBot DevArm B601, but the code is meant to be
robot-agnostic: `control/arm_base.py` defines the interface, `configs/arms/`
describes each robot, and the skill layer talks to `SafeArm` without knowing
which machine is on the other end.

The failure mode is not "someone imports the reBot class in a skill" (nobody
did). It is subtler: a harness loads a profile that describes a DIFFERENT
robot and patches the mismatched fields at runtime, so every field it forgets
stays silently wrong. That cost two full LIBERO benchmark runs
(`topdown_z_max` clamping place targets 65 cm below the table, and an RS-sized
jaw width applied to a Panda gripper).

These tests pin the invariants that keep profiles authoritative.
"""

from __future__ import annotations


from pathlib import Path

import pytest

_ARMS = Path(__file__).resolve().parents[1] / "configs" / "arms"
_SRC = Path(__file__).resolve().parents[1] / "src" / "cascade"


def _profiles():
    return sorted(_ARMS.glob("*.yaml"))


def _load(p: Path) -> dict:
    """Profile as the framework sees it, with any `extends:` parent resolved.

    Reading the raw YAML would fail every transport variant of a robot
    (so101_mock, so101_mujoco), which deliberately inherit the kinematics and
    the gripper measurements of the SAME arm from so101.yaml. What these tests
    forbid is inheriting them from a DIFFERENT robot, and `extends:` cannot
    express that: the resolved dict is the honest thing to assert on.
    """
    from cascade.config import _load_profile_raw

    return _load_profile_raw("arms", p.stem, _ARMS.parent)


def test_every_arm_profile_declares_its_own_gripper_width():
    """max_width_m gates grasp candidates and the air-grasp check.

    Inheriting it from another robot's profile admits grasps the fingers
    cannot close on. The Panda opens 0.08 m, the RS arm 0.09 m.
    """
    missing = []
    for p in _profiles():
        cfg = _load(p)
        if "max_width_m" not in (cfg.get("gripper") or {}):
            missing.append(p.name)
    assert not missing, (
        f"arm profiles without gripper.max_width_m: {missing}. "
        "The value would fall back to a default sized for a different robot."
    )


def test_every_arm_profile_declares_its_dof():
    missing = [p.name for p in _profiles() if "n_joints" not in _load(p)]
    assert not missing, f"arm profiles without n_joints: {missing}"


def test_home_pose_length_matches_declared_dof():
    """A 6-element home_q on a 7-DoF arm broadcasts wrong through every
    move_joints and harness approval downstream."""
    bad = []
    for p in _profiles():
        cfg = _load(p)
        home, n = cfg.get("home_q"), cfg.get("n_joints")
        if home is not None and n is not None and len(home) != int(n):
            bad.append(f"{p.name}: home_q has {len(home)}, n_joints={n}")
    assert not bad, bad


def test_joint_signs_length_matches_declared_dof():
    bad = []
    for p in _profiles():
        cfg = _load(p)
        signs, n = cfg.get("joint_signs"), cfg.get("n_joints")
        if signs is not None and n is not None and len(signs) != int(n):
            bad.append(f"{p.name}: joint_signs has {len(signs)}, n_joints={n}")
    assert not bad, bad


def test_libero_panda_profile_matches_robosuite_hardware():
    """Pins the values this profile exists to get right.

    0.08 m is measured from robosuite's panda_gripper.xml (each finger travels
    0..0.04 m). 7 DoF is the Franka. topdown_z_max is absolute here because
    LIBERO's table is at z = 0.80, not 0.
    """
    cfg = _load(_ARMS / "libero_panda.yaml")
    assert cfg["n_joints"] == 7
    assert cfg["gripper"]["max_width_m"] == pytest.approx(0.08)
    assert cfg["topdown_z_max"] > cfg["table_z"], (
        "a top-down ceiling below the table makes every place unreachable"
    )


def _motorbridge_profiles():
    return [p for p in _profiles() if _load(p).get("type") == "rebot_rs_mb"]


def test_motorbridge_profiles_declare_their_mit_gains():
    """The motorbridge path drops reBotArm_control_py and with it the SDK's
    rebotarm_rs.yaml, so there is no vendor default to inherit per-joint gains
    from. `rebot_rs.yaml` can leave `mit_kp: null` precisely because the SDK
    fills it in; here that would send kp=None to a 48 V motor, so the backend
    refuses to construct without them.
    """
    bad = []
    for p in _motorbridge_profiles():
        cfg = _load(p)
        n = int(cfg["n_joints"])
        for key in ("mit_kp", "mit_kd"):
            gains = cfg.get(key)
            if gains is None:
                bad.append(f"{p.name}: {key} missing (SDK defaults do not exist here)")
            elif len(gains) != n:
                bad.append(f"{p.name}: {key} has {len(gains)}, n_joints={n}")
    assert not bad, bad


def test_wire_signs_length_matches_declared_dof():
    """wire_signs maps mechPos to LOCAL joint angles. A short list would
    broadcast against the joint vector and silently mirror the wrong joints."""
    bad = []
    for p in _profiles():
        cfg = _load(p)
        signs, n = cfg.get("wire_signs"), cfg.get("n_joints")
        if signs is not None and n is not None and len(signs) != int(n):
            bad.append(f"{p.name}: wire_signs has {len(signs)}, n_joints={n}")
    assert not bad, bad


def test_rs_motorbridge_base_yaw_sign_stays_as_measured():
    """Pins the one wire sign that was confirmed against the physical arm.

    Jogged on the rig 2026-08-27: a commanded +0.100 rad moved mechPos +0.0947
    rad, and a +0.5 rad jog rotated the base CLOCKWISE viewed from above. FK on
    this profile puts +q1 toward -y (y is left in the base frame), so clockwise
    is what the model predicts and +1 is correct.

    This is pinned separately from the length check because the length check
    would happily pass on an inverted sign, and an inverted base yaw sends every
    reach to the mirror image of its target -- a failure that looks like bad
    calibration rather than a sign error.

    Joints 2..6 are deliberately NOT pinned: their signs are inferred from
    matching scale and zero, not observed, so a test asserting them would
    manufacture confidence that nobody earned.
    """
    for p in _motorbridge_profiles():
        signs = _load(p).get("wire_signs")
        assert signs is not None, f"{p.name}: wire_signs missing"
        assert signs[0] == 1, (
            f"{p.name}: base yaw wire sign was measured as +1 on the rig "
            f"(clockwise from above for +q1, matching FK); got {signs[0]}. "
            "If the arm was rebuilt or a motor re-zeroed, re-jog it and update "
            "this pin together with the profile comment."
        )


def test_rs_motorbridge_gripper_opens_in_the_positive_direction():
    """Pins a MEASUREMENT, not a preference.

    Hand-sweeping the RS jaws end to end on 2026-08-27 gave travel 0.0 ->
    +6.39 rad with closed at 0. The DM-derived `rebot_rs.yaml` says open is
    -6.8, and that profile's own comment flags it as unverified for RS.
    Copying the DM polarity into the motorbridge profile does not merely open
    the wrong amount: `open_gripper()` then drives the jaws into their CLOSED
    hard stop and holds there under effort.

    If this test fails because someone re-measured, update the number here and
    in the profile together -- do not delete the pin.
    """
    for p in _motorbridge_profiles():
        g = _load(p).get("gripper") or {}
        assert g["open_pos"] > g["closed_pos"], (
            f"{p.name}: RS jaws open toward POSITIVE mechPos "
            f"(measured 0 -> +6.39 rad); got open={g['open_pos']} "
            f"closed={g['closed_pos']}, which is the DM build's polarity"
        )
        assert 0 < g["open_pos"] <= 6.39, (
            f"{p.name}: open_pos {g['open_pos']} is outside the measured "
            "travel; it must keep margin off the +6.39 rad hard stop"
        )


def test_no_robot_specific_imports_outside_the_control_layer():
    """Skills, perception, agent and safety must not depend on a robot backend.

    Robot-specific behaviour belongs behind ArmBase, not inline in a skill.

    Checks CODE, not prose: docstrings are allowed (and expected) to describe
    the hardware, e.g. `grasping/force.py` explains that the gripper is a
    RobStride motor in MIT mode. That is documentation of a physical fact, not
    a dependency. Parsing the AST is what separates the two; a line-by-line
    grep flagged both docstrings and would have trained everyone to ignore it.
    """
    import ast

    offenders = []
    names = {"rebot_rs_arm", "RebotRSArm", "robstride", "RobStride"}
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC)
        if rel.parts[0] in {"control", "sim", "apps"}:
            continue          # factories and backends may name backends
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if any(n in a.name for n in names):
                        offenders.append(f"{rel}:{node.lineno}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(n in node.module for n in names):
                    offenders.append(f"{rel}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Name) and node.id in names:
                offenders.append(f"{rel}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in names:
                offenders.append(f"{rel}:{node.lineno}: .{node.attr}")
    assert not offenders, (
        "robot-specific code outside the control layer:\n  "
        + "\n  ".join(offenders)
    )
