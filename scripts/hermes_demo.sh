#!/usr/bin/env bash
# One-shot Hermes demo: register the wrc-demo MCP server, verify the
# connection, and drop into a Hermes chat where you can talk to the arm.
#
#   ./scripts/hermes_demo.sh                    # L515 camera, mock arm
#   ./scripts/hermes_demo.sh --arm rebot_rs     # real arm (onsite only!)
#
# Try saying:
#   "take a camera snapshot and tell me what you see"
#   "what objects are on the table?"
#   "grasp the red cube and place it on the plate"
set -euo pipefail

CAMERA="l515"
ARM="mock"
PY="/home/spark/Projects/demo/.demo/bin/python"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DETECTOR="${WRC_DETECTOR_MODEL:-/home/spark/Projects/demo/reBot-DevArm-Grasp/models/yolo11n.pt}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --detector) DETECTOR="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 1 ;;
    esac
done

# If the ultralytics CLIP fork is installed, prefer the open-vocabulary
# YOLOE model (free-text vocabulary); otherwise stay on closed-set yolo11n.
if "$PY" -c "import clip" 2>/dev/null; then
    DETECTOR="${WRC_DETECTOR_MODEL:-/home/spark/Projects/demo/reBot-DevArm-Grasp/models/yoloe-26s-seg.pt}"
    echo "[+] CLIP available: using open-vocabulary detector $DETECTOR"
else
    echo "[i] CLIP fork not installed: using closed-set COCO detector."
    echo "    For YOUR OWN vocabulary (any object by name), run once:"
    echo "    uv pip install --python $PY 'git+https://github.com/ultralytics/CLIP.git'"
fi

echo "[+] registering MCP server (camera=$CAMERA arm=$ARM)"
hermes mcp remove wrc-demo >/dev/null 2>&1 || true
hermes mcp add wrc-demo \
    --command "$PY" \
    --env "PYTHONPATH=$REPO/src" "WRC_CAMERA=$CAMERA" "WRC_ARM=$ARM" \
          "WRC_DETECTOR_MODEL=$DETECTOR" \
    --args -m wrc_demo.apps.mcp_server

echo "[+] testing the connection"
hermes mcp test wrc-demo

echo "[+] opening chat (ctrl+d to exit)"
exec hermes chat -q "Take a camera snapshot and list the objects you can see on the table." || exec hermes chat
