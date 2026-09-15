"""Analytic top-down grasp planning from an ObjectFix (improved baseline).

The baseline estimated grasps in the camera frame from 2D OBBs (approach
along the camera ray). Here the object's 3D OBB already lives in the base
frame, so the plan is done where it is physically meaningful:

- approach: straight down (-z base) for tabletop scenes;
- jaw opening axis: the OBB minor horizontal axis (grip across the object's
  short side), yaw = atan2 of that axis; a 90-degree-rotated alternate is
  also emitted for the recovery path ("try the other yaw" after a failure),
  and each yaw is emitted twice, 180 degrees apart, since the jaw axis is a
  line and the flip is free reach on a roll-limited wrist;
- grasp height: object top minus a fraction of its height, clamped above the
  table so the jaws wrap the object instead of pinching its rim.

Width feasibility against the gripper's max opening is annotated, not
silently dropped: the agent should know an object is too wide to pinch (and
maybe push it instead).
"""

from __future__ import annotations

import numpy as np

from ..types import Grasp, ObjectFix


def _yaw_rotation(yaw: float, tool_down: np.ndarray | None = None,
                  axis_order: str = "down_open") -> np.ndarray:
    """TCP rotation for a top-down grasp with jaw-opening yaw.

    `axis_order` names the tool-frame convention, because it is NOT universal:

      "down_open"  columns [approach, open, third]  (reBot / RS arm: tool
                   forward is +x of gripper_end and points down)
      "open_down"  columns [open, third, approach]  (Franka Panda in LIBERO,
                   and the SO-101 -- see below)
      "third_open_down"  columns [third, open, approach]  (Franka FR3 with the
                   Franka Hand as described by franka_description: the two
                   prismatic fingers slide along hand +y, `fr3_hand_tcp` is a
                   pure +z translation of `fr3_hand`, so the OPENING axis is
                   col1 and the approach is col2. This is NOT the LIBERO
                   Panda's frame: robosuite's `gripper0_grip_site` is rotated
                   so its col0 is the opening axis. Same robot family, two
                   descriptions, two conventions -- measure, do not assume.)

    MEASURED for the SO-101 from the Menagerie collision geometry, by placing
    the two fingertip sphere sets in the URDF TCP frame (`gripper_frame_link`)
    across the gripper joint's whole range:

      - the fingers extend along col2 (the fixed jaw's root is at col2 = -76 mm
        and its tip at +3 mm), so col2 is the APPROACH;
      - the tip-to-tip separation is along col0, within 3.5 degrees of it over
        the entire useful aperture (16-55 mm), so col0 is the OPENING axis;
      - the col1 component of the separation is 0 at every angle, because col1
        is the jaw HINGE axis (a single-hinge "beak", not parallel fingers).

    That is exactly the Panda's order, so the SO-101 reuses "open_down" rather
    than needing a convention of its own.

    MEASURED on LIBERO's Panda at rest, the site frame reads:

        col 0 [0, 1, 0]           <- jaw opening axis (confirmed against the
                                     vector between the two finger pads)
        col 2 [-0.06, 0, -1.0]    <- approach, pointing down

    Assuming the reBot order there asked for a frame 92.6 degrees away from
    anything the arm holds naturally. Position-only IK hid that by silently
    discarding the requested rotation; once orientation was actually solved
    for, the same request made the solver diverge to 1466 mm.

    The convention belongs to the arm, so callers pass it from the arm profile
    rather than this function guessing per robot.
    """
    down = np.array([0.0, 0.0, -1.0]) if tool_down is None else tool_down
    open_axis = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    open_axis -= open_axis @ down * down
    open_axis /= np.linalg.norm(open_axis)
    third = np.cross(down, open_axis)
    if axis_order == "open_down":
        # Right-handed with approach last: [open, third x open ... ] worked out
        # so that col0 = opening, col2 = approach, matching the measurement.
        return np.column_stack([open_axis, np.cross(down, open_axis), down])
    if axis_order == "third_open_down":
        # col1 = opening, col2 = approach; col0 = open x down keeps it
        # right-handed (det = +1), which IK needs for a valid rotation.
        return np.column_stack([np.cross(open_axis, down), open_axis, down])
    if axis_order != "down_open":
        raise ValueError(
            f"unknown tool_axis_order {axis_order!r} "
            "(down_open | open_down | third_open_down)"
        )
    return np.column_stack([down, open_axis, third])


def _rim_grasp_width(points: np.ndarray, obj_top_z: float,
                     band_m: float = 0.006, up: np.ndarray | None = None):
    """Rim geometry for an open container, or None.

    Returns `(wall_thickness_m, grasp_z, r_out, r_in, centre_xy)`, all measured
    AT THE GRASP DEPTH rather than at the lip.

    `up` is the container's own axis. It defaults to world vertical, which is
    right for an upright object and wrong for a tilted one: the band would then
    cut the wall diagonally and smear the annulus.

    MEASURED across libero_spatial, annulus ratio by frame:

        task 5, tilted 15.9 deg   world frame: n/a      object frame: 0.840
        the other nine, upright   world frame: 0.840    object frame: 0.840

    So the object frame reproduces the upright answer exactly and additionally
    recovers the tilted case, which the world frame could not see at all.

    MEASURED CASE this exists for: LIBERO's akita bowl is 112.4 mm across
    against an 80 mm jaw opening, so closing on its footprint is impossible
    and the planner correctly refused every grasp. A bowl is not picked
    across its diameter though: it is pinched by the wall, one finger inside
    and one outside.

    A solid object has material all the way to the axis, so its inner radius
    in the top band is near zero; a container leaves an annulus. That gap is
    the signal, and it is visible in a point cloud, which is why this works
    from perception rather than needing the mesh.

    The returned geometry is measured AT THE GRASP DEPTH, not at the lip,
    because containers taper (see below), and grasp_z sits below the lip so
    the pads straddle the wall instead of resting on its top edge.
    """
    import os
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from ..perception.cuda_math import rim_grasp_width
        return rim_grasp_width(points, obj_top_z, band_m=band_m, up=up)
    if points is None or len(points) < 60:
        return None
    axis = (np.array([0.0, 0.0, 1.0]) if up is None
            else np.asarray(up, float) / (np.linalg.norm(up) or 1.0))
    origin = points.mean(axis=0)
    h = (points - origin) @ axis
    band = points[np.abs(h - float(h.max())) < band_m]
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
    # Drop the pads below the lip, then re-measure the wall AT THAT DEPTH.
    #
    # MEASURED on the akita bowl: the wall tapers inward with depth, so a
    # radius taken at the lip is wrong where the fingers actually close.
    #
    #     depth below rim    r_in   r_out   wall centre
    #                0 mm    48.5    59.9    54.2 mm
    #               12 mm    42.0    55.7    48.9 mm
    #
    # Aiming at the lip's 54.2 mm put the TCP 3.8 mm outboard of the wall at
    # the grasp depth, and the jaws closed past it in mid-air while the skill
    # reported a verified 9 mm grasp. Containers taper; a single-height rim
    # model does not survive that.
    grasp_z = float(obj_top_z - 0.012)
    deep = points[np.abs(points[:, 2] - grasp_z) < 0.0025]
    if len(deep) >= 24:
        rd = np.linalg.norm(deep[:, :2] - centre, axis=1)
        d_out = float(np.percentile(rd, 99))
        d_in = float(np.percentile(rd, 1))
        if d_out > d_in > 0.0:
            r_out, r_in = d_out, d_in
            wall = r_out - r_in
    return wall, grasp_z, r_out, r_in, centre


def plan_grasps_from_fix(
    fix: ObjectFix,
    table_z: float,
    max_width_m: float = 0.09,
    depth_fraction: float = 0.5,
    min_grasp_z_above_table: float = 0.005,
    width_pad_m: float = 0.015,
    axis_order: str = "down_open",
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
    # A top-only cloud cannot locate the bottom. Keep the tabletop prior
    # unless side geometry resolves the vertical extent.
    if (obj_top_z - points_min_z <= 0.01
            or points_min_z - table_z < 0.5 * max(obj_top_z - table_z, 0.01)):
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
        # Both yaw and yaw+pi describe the SAME physical grasp: the jaw axis is
        # a line, so flipping it swaps which jaw is on which side and nothing
        # else. Emitting the flip costs nothing and buys reach on arms whose
        # wrist roll cannot cover a full turn -- MEASURED on the SO-101 (roll
        # range 320 deg), the flip lifts the top-down solve rate over its
        # workspace from 0.672 to 0.801. Ranked just behind its own primary so
        # a physically better yaw is never displaced by a flipped worse one.
        for sub, y in enumerate((yaw, yaw + np.pi)):
            grasps.append(
                Grasp(
                    position=pos,
                    rotation=_yaw_rotation(y, axis_order=axis_order),
                    width_m=required,
                    approach=np.array([0.0, 0.0, -1.0]),
                    quality=(1.0 if feasible else 0.2)
                    * (1.0 - 0.1 * rank - 0.01 * sub)
                    * fix.detection.conf,
                    label=fix.label,
                )
            )

    # Open containers cannot be grasped across their footprint: pinch the rim
    # instead. Only attempted when the footprint genuinely does not fit, so
    # solid objects keep today's behaviour exactly.
    if grasps and all(g.width_m > max_width_m for g in grasps):
        # The container's own axis: whichever OBB axis is closest to vertical.
        # Using world up here loses tilted objects entirely (measured: a bowl
        # at 15.9 deg read as solid, ratio 0.108 against 0.840 upright).
        up = None
        axes = getattr(fix, "axes", None)
        if axes is not None:
            A = np.asarray(axes, float)
            if A.shape == (3, 3):
                up = A[int(np.argmax(np.abs(A @ np.array([0.0, 0.0, 1.0]))))]
                if up @ np.array([0.0, 0.0, 1.0]) < 0:
                    up = -up
        rim = _rim_grasp_width(fix.points, obj_top_z, up=up)
        if rim is not None:
            wall, rim_z, r_out, r_in, centre_xy = rim
            rim_required = wall + width_pad_m
            if rim_required <= max_width_m:
                # Put the TCP on the MIDDLE of the wall, using the radii
                # measured at the grasp depth (the wall tapers; see
                # `_rim_grasp_width`). The container's centre is empty air and
                # `fix.position` is the whole object's centroid, so neither is
                # a valid reference here.
                r_mid = 0.5 * (r_out + r_in)
                for rank, yaw in enumerate((0.0, np.pi / 2)):
                    d = np.array([np.cos(yaw), np.sin(yaw)])
                    pos = np.array([centre_xy[0] + d[0] * r_mid,
                                    centre_xy[1] + d[1] * r_mid,
                                    rim_z])
                    grasps.insert(rank, Grasp(
                        position=pos,
                        # Jaws close ACROSS the wall, i.e. radially.
                        rotation=_yaw_rotation(yaw, axis_order=axis_order),
                        # NO approach padding here. `width_pad_m` exists so
                        # the jaws clear a solid object on the way down, and
                        # for a footprint grasp it is harmless: the closing
                        # stage still drives the fingers onto the surface.
                        #
                        # MEASURED: padding a 9.2 mm rim to 24.2 mm left the
                        # jaws 15 mm wider than the material, so they closed
                        # on air. The skill reported ok=True with
                        # grip_verified=True, the object never moved
                        # (lift 0.0 cm, moved 0.0 cm on 8/10 tasks), and the
                        # air-grasp check missed it because 24 mm of gap looks
                        # exactly like holding a 24 mm object.
                        #
                        # The wall thickness IS the closing width for a rim.
                        width_m=wall,
                        approach=np.array([0.0, 0.0, -1.0]),
                        quality=0.9 * (1.0 - 0.1 * rank) * fix.detection.conf,
                        label=fix.label,
                    ))
    return grasps
