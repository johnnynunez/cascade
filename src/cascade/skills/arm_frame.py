"""The per-call arm frame (spec 2026-10-17 base_pose runtime frame, §3.1-3.2).

Two frames meet in the skill runtime:

- the TABLE frame W: camera extrinsics, the occupancy map, beliefs, the MuJoCo
  world and every coordinate an agent sends or reads;
- the running arm's BASE frame B: FK and IK, the harness workspace, `table_z`,
  per-arm config positions and the held offset.

`T_world_base` (the arm's `base_pose` as a 4x4, W <- B) maps one to the other:
p_W = T p_B. `None` means "no transform": no base_pose, or the exact identity.
Every helper here then returns its input OBJECT unchanged, so an arm without a
base_pose (the reBot, the kitchen arm, mock, so101_mock) runs byte for byte as
before. `_rigid_apply` is the only place that multiplies by a pose.

`execute(arm=X)` binds an `ArmFrame` for that one call: X's `T_world_base` and
its `ArmBinding` (Kinematics, resolved profile, gripper constants). The
binding is built once per arm by `build_runtime`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..types import SkillError

#: Sections a non-primary arm reads from its own resolved view (D3).
ARM_SCOPED_SECTIONS: tuple[str, ...] = ("arm", "grasp", "safety", "relative_motion")
#: Profile `overrides:` keys that describe the whole rig and stay top-level
#: (`agent`: the ASPIRE sim retry ledger is built once per runtime).
RIG_SCOPED_OVERRIDES: frozenset[str] = frozenset({
    "workspace_filter", "memory", "perception_loop", "detector", "object_descriptions", "occupancy", "agent"})
#: `grasp` keys naming services the rig builds ONCE from the primary's view
#: (GraspGen-X, HUG, the VLA executor; apps/demo.py): every arm's view keeps
#: the top-level values, so no arm names a service the rig never built (R13).
RIG_GRASP_SERVICE_KEYS: frozenset[str] = frozenset({"backend", "executor", "graspgenx", "hug", "vla"})

#: |R^T R - I| tolerance of a rigid base_pose, and of "level" (#349's).
_RIGID_ATOL = 1e-9


class ArmFrameError(SkillError):
    """The running arm's frame cannot be resolved: execute() refuses the call."""


@dataclass(frozen=True)
class ArmBinding:
    """One arm's Kinematics and resolved profile, built once per arm.

    `grip` is (open_pos, closed_pos, max_width_m); None = use the runtime's own
    (the primary's)."""

    kin: Any
    cfg: Any
    grip: tuple[float, float, float] | None


@dataclass(frozen=True, eq=False)
class ArmFrame:
    """The frame bound for one execute() call."""

    name: str | None
    T_world_base: np.ndarray | None
    binding: ArmBinding


# ── reading the pose ──────────────────────────────────────────────────────


def world_from_base(arm) -> np.ndarray | None:
    """The running arm's `T_world_base`: its harness's `base_pose`, or None.

    Reads instance dicts only (the #352 `_mounted_off_table_origin` pattern):
    a standby LazyArm is never materialized and a Mock never "has" a pose.
    No harness, no base_pose and the exact identity are None. Anything that is
    not a finite rigid 4x4 (last row [0, 0, 0, 1], R^T R = I within 1e-9,
    det R > 0) raises ArmFrameError: fail closed.
    """
    harness = (getattr(arm, "__dict__", None) or {}).get("harness")
    pose = (getattr(harness, "__dict__", None) or {}).get("base_pose")
    if pose is None:
        return None
    try:
        T = np.array(pose, dtype=float)
    except (TypeError, ValueError):
        raise ArmFrameError("its base_pose is not a rigid transform (not a numeric matrix)") from None
    if T.shape != (4, 4):
        raise ArmFrameError(f"its base_pose is not a rigid transform (shape {T.shape}, not 4x4)")
    if not np.isfinite(T).all():
        raise ArmFrameError("its base_pose is not a rigid transform (non-finite values)")
    R = T[:3, :3]
    if (not np.array_equal(T[3], [0.0, 0.0, 0.0, 1.0])
            or not np.allclose(R.T @ R, np.eye(3), rtol=0.0, atol=_RIGID_ATOL)
            or np.linalg.det(R) <= 0.0):
        raise ArmFrameError("its base_pose is not a rigid transform (rotation not orthonormal)")
    if np.array_equal(T, np.eye(4)):
        return None
    return T


def is_level(T) -> bool:
    """The base's z axis is the table's: the z row of R is [0, 0, 1]."""
    return T is None or bool(np.allclose(np.asarray(T, dtype=float)[2, :3], [0.0, 0.0, 1.0],
                                         rtol=0.0, atol=_RIGID_ATOL))


def tilt_deg(T) -> tuple[float, float]:
    """(roll, pitch) of the base in degrees (URDF rpy, `types.pose_to_transform`)."""
    if T is None:
        return 0.0, 0.0
    R = np.asarray(T, dtype=float)[:3, :3]
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    return float(np.degrees(roll)), float(np.degrees(pitch))


# ── the only multiply ────────────────────────────────────────────────────


def _rigid_apply(T, x) -> np.ndarray:
    """T applied to x: a 4x4 pose composes (T @ x); (3,) or (N, 3) points map
    as T p. The ONLY place a base_pose multiplies anything; callers with
    T None never reach it."""
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape == (4, 4):
        return T @ x
    return x @ T[:3, :3].T + T[:3, 3]


def _inverse(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _rotation(T) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = np.asarray(T, dtype=float)[:3, :3]
    return out


# ── converters (T None: the input object, unchanged) ─────────────────────


def to_arm(T, p):
    """A table-frame point (3,) or points (N, 3) in the arm's base frame."""
    return p if T is None else _rigid_apply(_inverse(T), p)


def to_world(T, p):
    """An arm-frame point (3,) or points (N, 3) in the table frame."""
    return p if T is None else _rigid_apply(T, p)


def dir_to_arm(T, v):
    """A table-frame direction in the arm frame (rotation only)."""
    return v if T is None else _rigid_apply(_rotation(_inverse(T)), v)


def dir_to_world(T, v):
    """An arm-frame direction in the table frame (rotation only)."""
    return v if T is None else _rigid_apply(_rotation(T), v)


def cam_to_arm(T, T_table_cam):
    """A camera's table-frame pose (cam -> W) as cam -> B: T^-1 T_table_cam."""
    return T_table_cam if T is None else _rigid_apply(_inverse(T), T_table_cam)


def frame_to_arm(T, frame, extrinsics):
    """The frame with `T_base_cam` in the arm frame: T^-1 (the frame's own
    per-capture pose, else the camera's static `extrinsics.cam_to_base()`)."""
    if T is None:
        return frame
    T_table_cam = frame.T_base_cam if frame.T_base_cam is not None else extrinsics.cam_to_base()
    return dataclasses.replace(frame, T_base_cam=cam_to_arm(T, T_table_cam))


def _fix_through(M, fix):
    return dataclasses.replace(
        fix,
        position=_rigid_apply(M, fix.position),
        points=_rigid_apply(M, np.asarray(fix.points, dtype=float).reshape(-1, 3)),
        axes=_rigid_apply(_rotation(M), np.asarray(fix.axes, dtype=float).T).T,
    )


def fix_to_arm(T, fix):
    """An ObjectFix in the arm frame: position, points and axes move; extent,
    detection, label and t are kept."""
    return fix if T is None else _fix_through(_inverse(T), fix)


def fix_to_world(T, fix):
    """An ObjectFix in the table frame (see `fix_to_arm`)."""
    return fix if T is None else _fix_through(T, fix)


def table_z_in_world(T, table_z):
    """The table-frame height of the arm's base-frame `table_z` plane (level
    mounts; the harness lifts it the same way)."""
    return table_z if T is None else float(_rigid_apply(T, [0.0, 0.0, float(table_z)])[2])


# ── bindings ─────────────────────────────────────────────────────────────


def arm_view_cfg(top_cfg, acfg):
    """The Cfg a non-primary arm runs with: the top-level data, with the
    `ARM_SCOPED_SECTIONS` from the arm's resolved view, and the rig's grasp
    services (`RIG_GRASP_SERVICE_KEYS`) kept as the top level has them.

    The top-level `arms` list stays, so #357's running-arm lookup agrees (R7).
    A profile without a resolved view (a hand-built config) contributes its
    own `arm` section only."""
    from ..config import Cfg

    top = dict(top_cfg._data)
    resolved = acfg.get("resolved")
    if resolved is not None:
        source = resolved._data if hasattr(resolved, "_data") else dict(resolved)
        sections = {key: source[key] for key in ARM_SCOPED_SECTIONS if key in source}
    else:
        own = acfg._data if hasattr(acfg, "_data") else dict(acfg)
        sections = {"arm": {k: v for k, v in own.items() if k != "resolved"}}
    if "grasp" in sections:
        top_grasp = top.get("grasp") or {}
        grasp = {k: v for k, v in sections["grasp"].items() if k not in RIG_GRASP_SERVICE_KEYS}
        grasp.update({k: top_grasp[k] for k in RIG_GRASP_SERVICE_KEYS if k in top_grasp})
        sections["grasp"] = grasp
    return Cfg({**top, **sections})


def arm_binding(top_cfg, acfg, kin) -> ArmBinding:
    """A non-primary arm's binding: its pre-pass Kinematics (the one its
    harness uses), its view Cfg (built once, so per-arm identities such as
    `held_observation.binding()` stay stable) and its gripper constants."""
    view = arm_view_cfg(top_cfg, acfg)
    grip = view.arm.get("gripper") or {}
    return ArmBinding(kin, view, (float(grip.get("open_pos", 0.0)), float(grip.get("closed_pos", 1.0)),
                                  float(grip.get("max_width_m", 0.09))))
