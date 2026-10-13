#!/usr/bin/env bash
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CASCADE_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
source "${CASCADE_FRANKA_SETUP:-$HOME/franka_ros2_ws/install/setup.bash}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
unset CASCADE_ARMS CASCADE_ROBOT CASCADE_BASE
export CASCADE_ARM=fr3
export CASCADE_CAMERA="${CASCADE_CAMERA_PROFILE:-d435i}"
export CASCADE_CAMERAS="$CASCADE_CAMERA"
export CASCADE_VIEW=0
export CASCADE_RUN_DIR="$REPO/runs/franka_mcp"
export CASCADE_GRASP_MEMORY_PATH="$REPO/runs/franka_mcp/grasp_memory.json"
export CASCADE_ENVELOPE_PATH="$REPO/runs/franka_mcp/envelope.json"
export CASCADE_BELIEFS_PATH="$REPO/runs/franka_mcp/beliefs.json"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO/models"
exec "${PY:-$REPO/.venv-hardware/bin/python}" -m cascade.apps.mcp_server "$@"
