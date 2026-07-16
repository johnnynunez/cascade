"""Analytic top-down grasp planning from an ObjectFix (improved baseline).

The baseline estimated grasps in the camera frame from 2D OBBs (approach
along the camera ray). Here the object's 3D OBB already lives in the base
frame, so the plan is done where it is physically meaningful:

- approach: straight down (-z base) for tabletop scenes;
- jaw opening axis: the OBB minor horizontal axis (grip across the object's
  short side), yaw = atan2 of that axis; a 90-degree-rotated alternate is
  also emitted for the recovery path ("try the other yaw" after a failure);
- grasp height: object top minus a fraction of its height, clamped above the
  table so the jaws wrap the object instead of pinching its rim.

Width feasibility against the gripper's max opening is annotated, not
silently dropped: the agent should know an object is too wide to pinch (and
maybe push it instead).
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, ObjectFix


def _yaw_rotation(yaw: float, tool_down: np.ndarray | None = None) -> np.ndarray:
    """TCP rotation for a top-down grasp with jaw-opening yaw.

    Convention: tool forward (+x of gripper_end on the RS arm) points down
    (-z base); the jaw opening axis lies in the table plane.
    """
    down = np.array([0.0, 0.0, -1.0]) if tool_down is None else tool_down
    open_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    open_axis -= open_axis @ down * down
    open_axis /= np.linalg.norm(open_axis)
    third = np.cross(down, open_axis)
    # Columns: x=approach(down), y=open axis, z=completes right-handed frame.
    return np.column_stack([down, open_axis, third])


def plan_grasps_from_fix(
    fix: ObjectFix,
    table_z: float,
    max_width_m: float = 0.09,
    depth_fraction: float = 0.5,
    min_grasp_z_above_table: float = 0.005,
    width_pad_m: float = 0.015,
) -> list[Grasp]:
    """-> ranked candidate grasps (primary yaw first, then alternates)."""
    # Horizontal footprint: project OBB axes into the table plane.
    axes, extents = fix.axes, fix.extent
    horiz: list[tuple[float, np.ndarray]] = []
    for i in range(3):
        a = axes[:, i].copy()
        a[2] = 0.0
        n = np.linalg.norm(a)
        if n < 0.3:  # mostly vertical axis: not a jaw direction
            continue
        # Jaws only span the HORIZONTAL projection of this extent.
        horiz.append((float(extents[i]) * n, a / n))
    if not horiz:  # degenerate (looking at a sphere/top of a cylinder)
        horiz = [(float(extents[1]), np.array([1.0, 0.0, 0.0]))]
    horiz.sort(key=lambda t: t[0])  # narrowest horizontal extent first

    obj_top_z = float(fix.points[:, 2].max())
    points_min_z = float(fix.points[:, 2].min())
    # Top-down cameras only see the object's top surface, so points_min sits
    # near the top and would collapse the height estimate. Objects rest on
    # the table: unless the cloud clearly starts well above it (stacked),
    # take the table as the bottom.
    if points_min_z - table_z < 0.5 * max(obj_top_z - table_z, 0.01):
        obj_bottom_z = table_z
    else:
        obj_bottom_z = points_min_z
    height = max(obj_top_z - obj_bottom_z, 0.01)
    grasp_z = max(
        obj_top_z - depth_fraction * height,
        table_z + min_grasp_z_above_table,
    )

    grasps: list[Grasp] = []
    for rank, (width, axis) in enumerate(horiz[:2]):
        yaw = float(np.arctan2(axis[1], axis[0]))
        required = width + width_pad_m
        feasible = required <= max_width_m
        pos = fix.position.copy()
        pos[2] = grasp_z
        grasps.append(
            Grasp(
                position=pos,
                rotation=_yaw_rotation(yaw),
                width_m=required,
                approach=np.array([0.0, 0.0, -1.0]),
                quality=(1.0 if feasible else 0.2) * (1.0 - 0.1 * rank) * fix.detection.conf,
                label=fix.label,
            )
        )
    return grasps
