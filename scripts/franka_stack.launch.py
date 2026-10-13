"""FR3 and Franka Gripper through stock franka_bringup, without MoveIt."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    bringup = Path(get_package_share_directory("franka_bringup"))
    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.11"),
        DeclareLaunchArgument("use_fake_hardware", default_value="false"),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(bringup / "launch/franka.launch.py")),
            launch_arguments={
                "robot_type": "fr3", "namespace": "", "arm_prefix": "",
                "robot_ip": LaunchConfiguration("robot_ip"),
                "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
                "load_gripper": LaunchConfiguration("load_gripper"), "joint_state_rate": "100",
                "controllers_yaml": str(root / "configs/ros2/fr3_controllers.yaml"),
            }.items(),
        ),
        Node(package="controller_manager", executable="spawner",
             arguments=["fr3_arm_controller", "--controller-manager", "/controller_manager",
                        "--controller-manager-timeout", "60"], output="screen"),
    ])
