"""Hardware facts shared by the planner and MCP discovery; no device imports."""

import numpy as np


def arm_profiles(cfg):
    return [a.as_dict() if hasattr(a, "as_dict") else a
            for a in (cfg.get("arms") or [cfg.arm])]


def _moves_the_base(pose) -> bool:
    """A declared `base_pose` that moves the arm off the table origin. The
    exact identity counts as none (base_pose spec §4); a pose that does not
    parse counts as one (the build refuses it before any arm is made)."""
    if pose is None:
        return False
    from .types import pose_to_transform

    try:
        T = pose_to_transform(pose)
    except (TypeError, ValueError):
        return True
    return not bool((T == np.eye(4)).all())


def declares_base_pose(cfg) -> bool:
    """Any arm of this rig declares a base_pose (base_pose spec D7): its agent
    text then says that coordinates are in the shared TABLE frame. A rig
    without one keeps every string byte-identical."""
    if cfg is None:
        return False
    try:
        arms = arm_profiles(cfg)
    except (AttributeError, KeyError, TypeError):    # no arm section (a viewer's config double)
        return False
    return any(_moves_the_base(arm.get("base_pose")) for arm in arms)


def describe_robot(cfg):
    lines = []
    table_frame = declares_base_pose(cfg)
    for raw in arm_profiles(cfg):
        arm = raw.as_dict() if hasattr(raw, "as_dict") else raw
        grip = arm.get("gripper", {})
        name = arm.get("name", "primary")
        if table_frame:
            pose = arm.get("base_pose")
            where = (f"stands at base_pose {list(pose)} (x, y, z m; roll, pitch, yaw rad) in it"
                     if _moves_the_base(pose) else "stands at its origin")
            targets = ("Cartesian tool targets and reported positions use the shared table frame, in metres: "
                       "the same numbers for every arm on this rig. This arm's base "
                       f"({arm.get('base_frame', 'profile base')}) {where}.")
        else:
            targets = (f"Cartesian tool targets use the robot base frame ({arm.get('base_frame', 'profile base')}), "
                       "in metres.")
        lines.append(
            f"Arm {name}: {arm.get('display_name', name)}, {arm['n_joints']}-DOF; "
            f"transport {arm['type']}. Joint numbers are 1..{arm['n_joints']}. "
            "Joint vectors use radians; rotate_joint takes relative degrees. "
            f"TCP frame: {arm.get('ee_frame', 'profile TCP')}. " + targets
        )
        if grip:
            lines.append(
                f"Gripper: {grip.get('name', 'profile gripper')}; "
                f"maximum aperture {grip.get('max_width_m', 'unspecified')} m. "
                f"Driver position: open={grip.get('open_pos')}, closed={grip.get('closed_pos')}. "
                + str(grip.get("position_description", "Use the high-level open/close/grasp tools."))
            )
        if arm.get("tool_axis_order") == "third_open_down":
            lines.append("In the TCP frame, finger approach is +Z and the jaws open along ±Y. "
                         "Base-frame directions are +X forward, +Y left, +Z up.")
        if arm.get("return_home_after_motion"):
            lines.append(
                "Home is the operator-taught overview pose for the wrist camera. "
                "After each top-level motion command, the runtime returns to this home, "
                "including an ordinary failed grasp, while preserving the gripper state. "
                "Move targets are visited before returning home. STOP, hardware faults "
                "and stale feedback prevent automatic return. Read the return_home result; "
                "do not assume a failed return succeeded or repeatedly retry it. "
                "Reobserve from home before selecting the next target."
            )
        if arm["type"] == "ros2":
            lines.append(
                f"ROS2 arm state: {arm.get('ros_state_topic')}; "
                f"commands: {arm.get('ros_command_topic')}; "
                f"gripper action: {arm.get('ros_gripper_action', 'not configured')}."
            )
            if arm.get("ros_trajectory_action"):
                lines.append(
                    f"Arm motions use {arm['ros_trajectory_action']} (FollowJointTrajectory), "
                    "with a complete smooth trajectory and execution feedback. Native robot "
                    "faults block motion; get_observation reports motion_control readiness "
                    "and hardware errors. A force reflex requires operator recovery before "
                    "another motion; do not retry a pick while it remains active."
                )
            if arm.get("ros_gripper_move_action"):
                lines.append(f"Opening uses {arm['ros_gripper_move_action']} "
                             "with full aperture in metres; the driver waits for action completion.")
            if arm.get("ros_gripper_grasp_action"):
                lines.append(f"Force-controlled closing uses {arm['ros_gripper_grasp_action']} "
                             "with full aperture in metres; the driver waits for action completion.")
            if arm.get("robot_model") == "fr3":
                lines.append(
                    "Control is through franka_bringup and ros2_control, with the Franka Gripper "
                    "action server. MoveIt is not used. FCI must be active in Franka Desk. "
                    "An application reset_stop checks readiness, clears CASCADE's stop latch and "
                    "returns to saved home while preserving the gripper; it cannot "
                    "enable FCI or clear a Franka hardware fault."
                )
        if arm.get("unsupported_skills"):
            lines.append("Unsupported skills on this arm: " + ", ".join(arm["unsupported_skills"]) + ".")
    cameras = cfg.get("cameras") or [cfg.camera]
    for raw in cameras:
        camera = raw.as_dict() if hasattr(raw, "as_dict") else raw
        mode = camera.get("extrinsics", {}).get("mode", "unspecified")
        lines.append(f"Camera {camera.get('name', camera.get('type'))}: {mode}.")
        if mode == "eye_in_hand":
            lines.append(
                "The camera moves with the hand. Reobserve after arm motion; do not reuse "
                "old image pixels as current targets or assume home clears the view. "
                "Image right/down are optical axes, not robot base directions. "
                "Use locate_pixel/preview_grasp and the configured hand-eye transform."
            )
    lines.append("A failed settle or stale feedback is a failed motion, not success. "
                 "Do not blindly repeat relative motions or relax safety tolerances.")
    return "\n".join(lines)
