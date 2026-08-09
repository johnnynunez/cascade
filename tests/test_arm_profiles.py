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
import yaml

_ARMS = Path(__file__).resolve().parents[1] / "configs" / "arms"
_SRC = Path(__file__).resolve().parents[1] / "src" / "wrc_demo"


def _profiles():
    return sorted(_ARMS.glob("*.yaml"))


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text()) or {}


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
