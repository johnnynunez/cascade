#!/usr/bin/env bash
# GraspGen-X ZMQ grasp server for cascade (set grasp.backend: graspgenx).
#
#   ./scripts/serve_graspgenx.sh [gripper] [port]
#
# Private inference environment, installed by scripts/install_graspgenx.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${CASCADE_GRASPGENX_SOURCE:-$REPO/.graspgenx-src}"
PY="${CASCADE_GRASPGENX_PYTHON:-$REPO/.graspgenx/bin/python}"
[[ -x "$PY" && -d "$SOURCE/graspgenx" ]] || {
    printf 'Missing GraspGen-X installation; run scripts/install_graspgenx.sh\n' >&2
    exit 1
}
export GRASPGENX_CHECKPOINT_DIR="${GRASPGENX_CHECKPOINT_DIR:-$SOURCE/ext/graspgenx_checkpoints}"
export GRASPGENX_GRIPPER_CFG_DIR="${GRASPGENX_GRIPPER_CFG_DIR:-$SOURCE/ext/gripper_descriptions}"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
# Cascade describes the active gripper with per-request sweep volumes.
ARGS=()
[[ -z "${1:-}" ]] || ARGS+=(--default-gripper "$1")
exec "$PY" "$REPO/scripts/graspgenx_server.py" \
    --config "$GRASPGENX_CHECKPOINT_DIR/release" \
    --assets-dir "$GRASPGENX_GRIPPER_CFG_DIR" \
    --port "${2:-${CASCADE_GRASPGENX_PORT:-5556}}" "${ARGS[@]}"
