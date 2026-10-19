#!/usr/bin/env bash
# Run a command in the ROS 2 environment of a native-controller arm (the
# Franka FR3, configs/arms/fr3.yaml): source the ROS distribution and the
# franka_ros2 workspace, default ROS_DOMAIN_ID to 31, then exec the command.
#
#   scripts/ros_env.sh python -m cascade.apps.mcp_server --http 127.0.0.1:18741 ...
#
# scripts/openshell_rig.py mcp-start --ros-env runs the robot MCP server's
# user unit through it: a systemd unit inherits none of the launcher's shell.
# Same defaults as scripts/bringup_franka.sh, which starts the bringup itself.
set -eo pipefail
source "${CASCADE_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
source "${CASCADE_FRANKA_SETUP:-$HOME/franka_ros2_ws/install/setup.bash}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
exec "$@"
