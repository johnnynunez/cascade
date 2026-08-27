#!/usr/bin/env bash
# GraspGen-X ZMQ grasp server for cascade (set grasp.backend: graspgenx).
#
#   ./scripts/serve_graspgenx.sh [gripper] [port]
#
# Model + deps live in ~/Projects/demo/.graspgenx (own venv: its torch is
# pinned independently of .demo). Checkpoints auto-downloaded to
# GraspGenX/ext on first import.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export GRASPGENX_CHECKPOINT_DIR="$ROOT/GraspGenX/ext/graspgenx_checkpoints"
exec "$ROOT/.graspgenx/bin/python" "$ROOT/GraspGenX/client-server/graspgenx_server.py" \
    --config "$GRASPGENX_CHECKPOINT_DIR/release/gen/config.yaml" \
    --assets_dir "$ROOT/GraspGenX/ext/gripper_descriptions" \
    --default_gripper "${1:-franka_panda}" \
    --port "${2:-5556}"
