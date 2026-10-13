#!/usr/bin/env bash
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CASCADE_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
source "${CASCADE_FRANKA_SETUP:-$HOME/franka_ros2_ws/install/setup.bash}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
exec ros2 launch "$REPO/scripts/franka_stack.launch.py" "$@"
