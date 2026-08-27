#!/usr/bin/env bash
# One-shot Hermes demo: register the cascade MCP server, verify the
# connection, and drop into a Hermes chat where you can talk to the arm.
#
#   ./scripts/hermes_demo.sh                              # default model
#   ./scripts/hermes_demo.sh --model anthropic/claude-sonnet-4-5
#   ./scripts/hermes_demo.sh --arm rebot_rs               # real arm (onsite!)
#
# The BRAIN is whatever vision-capable model Hermes is pointed at:
#   - Claude:      needs ANTHROPIC_API_KEY known to Hermes (`hermes model`),
#                  then --model anthropic/<claude model>
#   - GPT (Codex): already logged in on this rig (gpt-5.6-sol) - just works
#   - local VLM:   serve Qwen3.6 (scripts/serve_qwen_llamacpp.sh), define a
#                  custom provider in ~/.hermes/config.yaml `providers:`
#                  pointing at http://127.0.0.1:8080/v1, then --provider it
#
# Try saying:
#   "take a camera snapshot and tell me what you see"
#   "what objects are on the table?"
#   "grasp the red cube and place it on the plate"
set -euo pipefail

CAMERA="d455f"
ARM="mock"
MODEL=""
PROVIDER=""
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shared uv venv sits next to the checkout (…/Projects/demo/.demo) on every
# rig; override with PY=... for a non-standard layout
PY="${PY:-$(cd "$REPO/.." && pwd)/.demo/bin/python}"
# closed-set fallback; ultralytics auto-downloads it into models/ on first
# (online) use if absent. DETECTOR_SET tracks an explicit user choice so the
# CLIP branch below never overrides --detector / CASCADE_DETECTOR_MODEL.
DETECTOR="${CASCADE_DETECTOR_MODEL:-$REPO/models/yolo11n.pt}"
DETECTOR_SET="${CASCADE_DETECTOR_MODEL:+1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --detector) DETECTOR="$2"; DETECTOR_SET=1; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --provider) PROVIDER="$2"; shift 2 ;;
        *) echo "unknown flag $1"; exit 1 ;;
    esac
done

# If the ultralytics CLIP fork is installed, prefer the open-vocabulary
# YOLOE model (free-text vocabulary); otherwise stay on closed-set yolo11n.
if "$PY" -c "import clip" 2>/dev/null; then
    [[ -z "${DETECTOR_SET:-}" ]] && DETECTOR="$REPO/models/yoloe-11s-seg.pt"
    echo "[+] CLIP available: using open-vocabulary detector $DETECTOR"
else
    echo "[i] CLIP fork not installed: using closed-set COCO detector."
    echo "    For YOUR OWN vocabulary (any object by name), run once:"
    echo "    uv pip install --python $PY 'git+https://github.com/ultralytics/CLIP.git'"
fi

echo "[+] registering MCP server (camera=$CAMERA arm=$ARM)"
hermes mcp remove cascade >/dev/null 2>&1 || true
hermes mcp add cascade \
    --command "$PY" \
    --env "PYTHONPATH=$REPO/src" "CASCADE_CAMERAS=$CAMERA" "CASCADE_ARM=$ARM" \
          "CASCADE_DETECTOR_MODEL=$DETECTOR" "DISPLAY=${DISPLAY:-:1}" \
          "YOLO_OFFLINE=True" "ULTRALYTICS_OFFLINE=True" \
    --args -m cascade.apps.mcp_server

echo "[+] testing the connection"
hermes mcp test cascade

CHAT_ARGS=()
[[ -n "$MODEL" ]] && CHAT_ARGS+=(-m "$MODEL")
[[ -n "$PROVIDER" ]] && CHAT_ARGS+=(--provider "$PROVIDER")

cat <<'EOF'
[+] opening interactive chat (ctrl+d to exit). Ask things like:
      take a camera snapshot and tell me what you see
      what objects are on the table?
      localize the bottle
      grasp the red cube
EOF
exec hermes chat "${CHAT_ARGS[@]}"
