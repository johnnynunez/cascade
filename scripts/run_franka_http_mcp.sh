#!/usr/bin/env bash
# One shared native FR3 runtime for every OpenClaw session.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CASCADE_FRANKA_MOCK:-0}" == 1 ]]; then
    # Some hardware venvs obtain Pinocchio from the ROS installation. An
    # explicitly supplied setup path is also useful for an offline smoke;
    # sourcing it imports no ROS node and opens no hardware connection.
    if [[ -n "${CASCADE_ROS_SETUP:-}" ]]; then source "$CASCADE_ROS_SETUP"; fi
    export CASCADE_ARM=fr3_mock CASCADE_CAMERAS=mock CASCADE_OCCUPANCY=0
    unset CASCADE_GROUNDING_LLM_PROFILE
else
    source "${CASCADE_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
    source "${CASCADE_FRANKA_SETUP:-$HOME/franka_ros2_ws/install/setup.bash}"
    export CASCADE_ARM=fr3 CASCADE_CAMERAS=d435i
    # MCP uses a mock chat model because OpenClaw is the brain. Preserve the
    # local Cosmos image-grounding fallback used by native localization.
    export CASCADE_GROUNDING_LLM_PROFILE=local_cosmos
fi
# A previous multi-arm/mobile shell or setup hook cannot override this rig.
unset CASCADE_ARMS CASCADE_ROBOT CASCADE_BASE
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-31}"
export CASCADE_VIEW=0 CASCADE_STREAM=eager CASCADE_START_STOPPED=1 CASCADE_LLM=mock
# Holding is inferred from this runtime's observations, never injected by launch.
unset CASCADE_START_HOLDING
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
cd "$REPO/models"
exec "${PY:-$REPO/.venv-hardware/bin/python}" -m cascade.apps.mcp_server "$@"
