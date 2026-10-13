"""Hardware facts shared by the planner and MCP discovery; no device imports."""


def arm_profiles(cfg):
    return [a.as_dict() if hasattr(a, "as_dict") else a
            for a in (cfg.get("arms") or [cfg.arm])]


def describe_robot(cfg):
    lines = []
    for raw in arm_profiles(cfg):
        arm = raw.as_dict() if hasattr(raw, "as_dict") else raw
        grip = arm.get("gripper", {})
        name = arm.get("name", "primary")
        lines.append(
            f"Arm {name}: {arm.get('display_name', name)}, {arm['n_joints']}-DOF; "
            f"transport {arm['type']}. Joint numbers are 1..{arm['n_joints']}. "
            "Joint vectors use radians; rotate_joint takes relative degrees. "
            f"TCP frame: {arm.get('ee_frame', 'profile TCP')}. "
            f"Cartesian tool targets use the robot base frame ({arm.get('base_frame', 'profile base')}), in metres."
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
