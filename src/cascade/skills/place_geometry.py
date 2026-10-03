"""Pure proposed placement IK; no commands, scene writes or release authority."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grasping.obb_grasp import _yaw_rotation
from ..types import SkillError, make_transform


@dataclass(frozen=True)
class PlaceGeometry:
    lift: object
    pre: object
    low: object
    retreat: object
    retreat_target: object
    rotation: np.ndarray
    hover: np.ndarray
    target: np.ndarray | None = None


def plan(runtime, q_now, target, *, x, y, release_z, z_cap,
         attachment_translation_tool=None):
    # Error classes remain those of the ordinary skill: refusal has exactly
    # the same terminal carry/retreat semantics for point and area requests.
    from .runtime import _PostPlaceRetreatPlanError, _PreCarryLiftError

    gcfg = runtime.cfg.grasp
    attachment = None
    if attachment_translation_tool is not None:
        attachment = np.asarray(attachment_translation_tool, dtype=float)
        if attachment.shape != (3,) or not np.isfinite(attachment).all():
            raise SkillError("placement attachment translation must be a finite tool-frame vector")
    table_z = float(runtime.cfg.safety.get("table_z", 0.0))
    tcp_now = runtime.kin.fk(q_now)
    hover = target + np.array([0.0, 0.0, float(gcfg.get("pregrasp_offset_m", 0.12))])
    # Finish the horizontal carry before lowering the held object. A
    # low release target must not also lower a can through the worktop's
    # other props while it is still crossing to that target.
    hover[2] = max(hover[2], float(tcp_now[2, 3]))
    hover[2] = min(hover[2], z_cap)  # same wrist ceiling as the release
    carry_height = gcfg.get("carry_height_m")
    if carry_height is not None:
        carry_height = float(carry_height)
        if not np.isfinite(carry_height) or not table_z < carry_height <= z_cap:
            raise _PreCarryLiftError("carry height must be above the table and within the wrist ceiling")
        # Release height controls the final descent, independently of
        # the clearance needed while crossing other objects.
        hover[2] = max(hover[2], carry_height)
    lift = None
    carry_start = q_now
    if (bool(gcfg.get("pre_carry_lift", False))
            and float(hover[2]) > float(tcp_now[2, 3]) + .001):
        # Reaching the clearance height only at the far end of the
        # horizontal chord can catch a tall payload on a low platform.
        # First reach the SAME already-planned height at the current XY
        # and orientation. Plan and vet this phase before any motion.
        lift_pose = tcp_now.copy()
        lift_pose[2, 3] = float(hover[2])
        lift = runtime.kin.ik(lift_pose, q_now)
        if (not lift.success or np.max(np.abs(lift.q - q_now)) > np.pi):
            raise _PreCarryLiftError("pre-carry lift is unreachable; keeping the grasp")
        for fraction in (.15, .3, .45, .6, .75, .9, 1.):
            reason = runtime.arm.harness.vet_pose(q_now + fraction * (lift.q - q_now))
            if reason:
                raise _PreCarryLiftError(f"pre-carry lift is unsafe: {reason}")
        carry_start = lift.q
    # Plan empty-gripper clearance before release so the trip home cannot
    # tip the placed object. Keep release yaw and XY through retraction.
    retreat_target = None
    retreat_offset = gcfg.get("post_place_retreat_offset_m")
    if retreat_offset is not None:
        retreat_offset = float(retreat_offset)
        if not np.isfinite(retreat_offset) or retreat_offset <= 0:
            raise _PostPlaceRetreatPlanError("post-place retreat offset must be finite and positive")
        retreat_target = target.copy()
        retreat_target[2] = max(float(hover[2]), release_z + retreat_offset)
    # Preserve held yaw to avoid loading an off-center grasp. Nearby yaws
    # avoid wrist unwinding when the original yaw approaches joint limits.
    radial = float(np.arctan2(y, x))
    yaws = [radial, 0.0, np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2]
    approach_col = 0 if runtime._tool_axis_order == "down_open" else 2
    opening_col = 0 if runtime._tool_axis_order == "open_down" else 1
    if float(-tcp_now[2, approach_col]) > 0.95:
        held_yaw = float(np.arctan2(tcp_now[1, opening_col], tcp_now[0, opening_col]))
        near_yaws = [held_yaw]
        for delta in (np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
            near_yaws.extend((held_yaw - delta, held_yaw + delta))
        yaws = near_yaws + yaws
    pre = low = retreat = None
    for yaw in yaws:
        R = _yaw_rotation(yaw, axis_order=runtime._tool_axis_order)
        # The observed attachment is geometric aiming evidence only. Rotate
        # its translation into the destination frame; z keeps its existing
        # release-height meaning. The legacy world-offset path is unchanged.
        candidate_target = target.copy()
        candidate_hover = hover.copy()
        candidate_retreat = None if retreat_target is None else retreat_target.copy()
        if attachment is not None:
            candidate_target[:2] = np.array([x, y]) - (R @ attachment)[:2]
            candidate_hover[:2] = candidate_target[:2]
            if candidate_retreat is not None:
                candidate_retreat[:2] = candidate_target[:2]
        cand_pre = runtime.kin.ik(make_transform(R, candidate_hover), carry_start)
        if (not cand_pre.success
                or np.max(np.abs(cand_pre.q - carry_start)) > np.pi):
            continue
        cand_low = runtime.kin.ik(make_transform(R, candidate_target), cand_pre.q)
        if (cand_low.success
                and np.max(np.abs(cand_low.q - cand_pre.q)) <= np.pi):
            cand_retreat = None
            if retreat_target is not None:
                # Reuse the existing pose exactly when the hover already
                # provides this clearance (e.g. the kitchen pink cube).
                cand_retreat = (cand_pre if np.array_equal(candidate_retreat, candidate_hover)
                                else runtime.kin.ik(make_transform(R, candidate_retreat), cand_low.q))
                if (not cand_retreat.success
                        or np.max(np.abs(cand_retreat.q - cand_low.q)) > np.pi):
                    continue
            pre, low, retreat = cand_pre, cand_low, cand_retreat
            target, hover, retreat_target = candidate_target, candidate_hover, candidate_retreat
            break
    if pre is None or low is None:
        retreat_detail = (f", post-place retreat {retreat_target.round(3).tolist()}"
                          if retreat_target is not None else "")
        error_type = _PostPlaceRetreatPlanError if retreat_target is not None else SkillError
        raise error_type(
            f"place pose unreachable at {target.round(3).tolist()} "
            f"(hover {hover.round(3).tolist()}{retreat_detail}, all yaws tried)"
        )

    return PlaceGeometry(lift, pre, low, retreat, retreat_target, R, hover, target)
