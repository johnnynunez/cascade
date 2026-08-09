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


def _rim_grasp_width(points: np.ndarray, obj_top_z: float,
                     band_m: float = 0.006) -> tuple[float, float] | None:
    """Wall thickness and grasp height for an open container, or None.

    MEASURED CASE this exists for: LIBERO's akita bowl is 112.4 mm across
    against an 80 mm jaw opening, so closing on its footprint is impossible
    and the planner correctly refused every grasp. A bowl is not picked
    across its diameter though: it is pinched by the wall, one finger inside
    and one outside. Measured from the mesh, that wall is 5.2 mm thick:

        z band (frac of height)   r_min   r_max   wall
        0.98                      51.2    56.2     5.0 mm
        0.85                      47.4    54.7     7.3 mm
        0.50                      40.0    46.8     6.8 mm
        0.25                      25.2    38.6    13.4 mm

    A solid object has material all the way to the axis, so its `r_min` in
    the top band is near zero; a container leaves an annulus. That gap is the
    signal, and it is visible in a top-down point cloud, which is why this
    works from perception rather than needing the mesh.

    Returns `(wall_thickness_m, grasp_z)` where grasp_z sits just below the
    rim so the fingers straddle it.
    """
    if points is None or len(points) < 60:
        return None
    band = points[np.abs(points[:, 2] - obj_top_z) < band_m]
    # MEASURED: the belief store caps clouds at 384 points spread over the
    # whole surface, which left 53 in the rim band of a real bowl. An earlier
    # cut of this check demanded 100 and therefore never fired on the object
    # it was written for. The ratio test below is what discriminates hollow
    # from solid; this count only needs to be enough for percentiles to mean
    # something.
    if len(band) < 24:
        return None
    centre = band[:, :2].mean(axis=0)
    r = np.linalg.norm(band[:, :2] - centre, axis=1)
    r_out = float(np.percentile(r, 99))
    r_in = float(np.percentile(r, 1))
    if r_out < 1e-6:
        return None
    # A hollow rim: material sits in an annulus, so the inner radius is a
    # large fraction of the outer one. Solids fail this and fall through to
    # the normal footprint grasp.
    if r_in / r_out < 0.5:
        return None
    wall = r_out - r_in
    if wall <= 1e-4:
        return None
    return wall, float(obj_top_z - 0.5 * band_m)


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

    # Open containers cannot be grasped across their footprint: pinch the rim
    # instead. Only attempted when the footprint genuinely does not fit, so
    # solid objects keep today's behaviour exactly.
    if grasps and all(g.width_m > max_width_m for g in grasps):
        rim = _rim_grasp_width(fix.points, obj_top_z)
        if rim is not None:
            wall, rim_z = rim
            rim_required = wall + width_pad_m
            if rim_required <= max_width_m:
                # Offset the grasp point onto the rim itself: the centre is
                # empty air for a container.
                band = fix.points[np.abs(fix.points[:, 2] - obj_top_z) < 0.006]
                centre_xy = (band[:, :2].mean(axis=0) if len(band)
                             else fix.position[:2])
                r_out = (float(np.percentile(
                    np.linalg.norm(band[:, :2] - centre_xy, axis=1), 99))
                    if len(band) else 0.0)
                for rank, yaw in enumerate((0.0, np.pi / 2)):
                    d = np.array([np.cos(yaw), np.sin(yaw)])
                    pos = np.array([centre_xy[0] + d[0] * (r_out - 0.5 * wall),
                                    centre_xy[1] + d[1] * (r_out - 0.5 * wall),
                                    rim_z])
                    grasps.insert(rank, Grasp(
                        position=pos,
                        # Jaws close ACROSS the wall, i.e. radially.
                        rotation=_yaw_rotation(yaw),
                        width_m=rim_required,
                        approach=np.array([0.0, 0.0, -1.0]),
                        quality=0.9 * (1.0 - 0.1 * rank) * fix.detection.conf,
                        label=fix.label,
                    ))
    return grasps
