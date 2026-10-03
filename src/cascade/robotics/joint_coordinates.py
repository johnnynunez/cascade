"""Explicit generalized-coordinate conventions; no kinematics or control.

Frames name the parent/child *joint* frames supplied by the physical producer.
This declaration contains no joint origins or transforms into world/link frames.
"""
from collections.abc import Mapping

from .contracts import freeze_json, plain_json


MULTI_DOF_JOINTS = frozenset({"planar", "spherical", "floating"})
QUATERNION_NORM_SQUARED_TOLERANCE = 1e-6  # accepts float32 roundoff; never renormalizes


def coordinate_convention(kind):
    """Return a detached convention, never infer coordinates from a measurement."""
    if not isinstance(kind, str):
        raise ValueError("generalized joint type must be a string")
    if kind in {"revolute", "continuous"}:
        q, v, effort = ["angle"], ["angular_speed"], ["torque"]
        qu, vu, eu, rotation = ["rad"], ["rad/s"], ["N*m"], "right_handed_about_declared_axis"
    elif kind == "prismatic":
        q, v, effort = ["displacement"], ["linear_speed"], ["force"]
        qu, vu, eu, rotation = ["m"], ["m/s"], ["N"], "none"
    elif kind == "planar":
        q, v, effort = ["x", "y", "angle_z"], ["vx", "vy", "wz"], ["fx", "fy", "tz"]
        qu, vu, eu, rotation = ["m", "m", "rad"], ["m/s", "m/s", "rad/s"], ["N", "N", "N*m"], "right_handed_about_parent_z"
    elif kind == "spherical":
        q, v, effort = ["qw", "qx", "qy", "qz"], ["wx", "wy", "wz"], ["tx", "ty", "tz"]
        qu, vu, eu, rotation = ["1"] * 4, ["rad/s"] * 3, ["N*m"] * 3, "hamilton_wxyz_child_to_parent"
    elif kind == "floating":
        q = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        v, effort = ["vx", "vy", "vz", "wx", "wy", "wz"], ["fx", "fy", "fz", "tx", "ty", "tz"]
        qu, vu, eu, rotation = ["m"] * 3 + ["1"] * 4, ["m/s"] * 3 + ["rad/s"] * 3, ["N"] * 3 + ["N*m"] * 3, "hamilton_wxyz_child_to_parent"
    else:
        raise ValueError("unsupported generalized joint type")
    return {"nq": len(q), "nv": len(v), "q_order": q, "v_order": v, "effort_order": effort,
            "q_units": qu, "v_units": vu, "effort_units": eu,
            "configuration_frame": "parent_joint", "motion_frame": "parent_joint",
            "motion_relation": "child_relative_to_parent",
            "motion_reference": "child_joint_origin", "rotation": rotation}


def validate_coordinates(kind, value):
    """Require the complete convention; never silently normalize or convert it."""
    expected = coordinate_convention(kind)
    if (not isinstance(value, Mapping) or type(value.get("nq")) is not int
            or type(value.get("nv")) is not int or plain_json(value) != expected):
        raise ValueError("generalized joint coordinate convention mismatch")
    return freeze_json(value)
