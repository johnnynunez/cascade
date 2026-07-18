"""The USD asset is what Newton/Isaac Sim simulates; the URDF package is what
host FK/IK loads. These tests parse the USD physics layer directly (no USD
runtime needed) and pin the two assets together, so a re-export of either one
that changes joints, limits or masses fails CI instead of silently splitting
sim and host kinematics."""

import math
from xml.etree import ElementTree as ET

import numpy as np

from conftest import JOINT_SIGNS, URDF, USD, needs_pin

from wrc_demo.control.usd_model import apply_joint_signs, urdf_xml_from_usd

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _movable(root):
    return [j for j in root.iter("joint") if j.get("type") != "fixed"]


def test_usd_converter_extracts_full_joint_tree():
    root = ET.fromstring(urdf_xml_from_usd(USD))
    joints = _movable(root)
    assert [j.get("name") for j in joints] == ARM_JOINTS + ["joint_left", "joint_right"]
    assert [j.get("type") for j in joints] == ["revolute"] * 6 + ["prismatic"] * 2
    # gripper_end rides on link6 through a fixed joint (it is the IK frame)
    fixed = {j.get("name") for j in root.iter("joint") if j.get("type") == "fixed"}
    assert "j_gripper_end" in fixed
    names = {link.get("name") for link in root.iter("link")}
    assert {"base_link", "link6", "gripper_end", "gripper_left", "gripper_right"} <= names


def test_usd_converter_limits_and_masses():
    root = ET.fromstring(urdf_xml_from_usd(USD))
    limits = {j.get("name"): j.find("limit") for j in _movable(root)}
    # USD authors limits in degrees; URDF wants rad (values from physics.usda)
    assert math.isclose(float(limits["joint1"].get("lower")), math.radians(-160.42818), rel_tol=1e-6)
    assert math.isclose(float(limits["joint2"].get("lower")), -math.pi, rel_tol=1e-3)
    assert float(limits["joint2"].get("upper")) == 0.0
    # prismatic limits stay in meters
    assert math.isclose(float(limits["joint_left"].get("upper")), 0.05, rel_tol=1e-9)
    masses = {
        link.get("name"): float(link.find("inertial/mass").get("value"))
        for link in root.iter("link") if link.find("inertial/mass") is not None
    }
    assert math.isclose(masses["base_link"], 1.1774, rel_tol=1e-6)
    assert math.isclose(masses["link2"], 1.552, rel_tol=1e-6)  # masses PR#3 value


def test_apply_joint_signs_mirrors_axis_and_limits():
    xml = urdf_xml_from_usd(USD)
    flipped = ET.fromstring(apply_joint_signs(xml, JOINT_SIGNS))
    j2 = next(j for j in flipped.iter("joint") if j.get("name") == "joint2")
    lo, hi = float(j2.find("limit").get("lower")), float(j2.find("limit").get("upper"))
    assert lo == 0.0 and math.isclose(hi, math.pi, rel_tol=1e-3)
    # fingers keep their sign (only 6 arm entries in JOINT_SIGNS)
    jl = next(j for j in flipped.iter("joint") if j.get("name") == "joint_left")
    assert float(jl.find("limit").get("lower")) == 0.0


@needs_pin
def test_usd_matches_urdf_kinematics():
    """Sim asset (USD) and host asset (URDF) must be the same robot."""
    from wrc_demo.control.kinematics import Kinematics

    ku = Kinematics(str(URDF), "gripper_end")
    ks = Kinematics(str(USD), "gripper_end")
    assert ku.nq == ks.nq == 8
    assert np.allclose(ku.joint_limits[0], ks.joint_limits[0], atol=1e-5)
    assert np.allclose(ku.joint_limits[1], ks.joint_limits[1], atol=1e-5)
    rng = np.random.default_rng(7)
    for _ in range(50):
        q = rng.uniform(ku.joint_limits[0], ku.joint_limits[1])
        assert np.allclose(ku.fk(q), ks.fk(q), atol=1e-5)
        assert np.allclose(ku.link_positions(q), ks.link_positions(q), atol=1e-5)
    mu = [float(i.mass) for i in ku.model.inertias]
    ms = [float(i.mass) for i in ks.model.inertias]
    assert np.allclose(mu, ms, atol=1e-6)


@needs_pin
def test_usd_local_convention_round_trip():
    """joint_signs is an exact mirror: FK_local(q) == FK_asset(-q), and the
    repo's home_q is inside the local-convention limits."""
    from wrc_demo.control.kinematics import Kinematics

    raw = Kinematics(str(USD), "gripper_end")
    loc = Kinematics(str(USD), "gripper_end", joint_signs=JOINT_SIGNS)
    rng = np.random.default_rng(11)
    for _ in range(50):
        q = rng.uniform(loc.joint_limits[0], loc.joint_limits[1])
        assert np.allclose(loc.fk(q), raw.fk(-q), atol=1e-8)
    home = np.array([0.0, 1.2, 1.2, 0.0, 0.75, 0.0])
    lo, hi = loc.joint_limits
    assert np.all(home >= lo) and np.all(home <= hi)
