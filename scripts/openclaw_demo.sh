#!/usr/bin/env bash
# One-shot OpenClaw demo: register the wrc-demo MCP server, point OpenClaw's
# agent at a LOCAL brain, bring up the gateway, and print the web-chat URL.
# The user then talks to the arm from OpenClaw's web chat (not this repo's
# MJPEG dashboard — that stays the camera/narration big screen).
#
#   ./scripts/openclaw_demo.sh                          # sim rig, cosmos brain
#   ./scripts/openclaw_demo.sh --arm rebot_rs --cameras l515   # real arm (onsite!)
#   ./scripts/openclaw_demo.sh --brain qwen             # Qwen3.6 as primary
#   ./scripts/openclaw_demo.sh --brain skip             # keep current model
#
# Brains are the local servers from this repo (start them first):
#   cosmos  scripts/serve_cosmos_vllm.sh   -> http://127.0.0.1:8082/v1  (cosmos3-edge)
#   qwen    scripts/serve_qwen_llamacpp.sh -> http://127.0.0.1:8080/v1
#           NOTE: OpenClaw's agent system prompt overflows llama.cpp's default
#           16k/slot — serve Qwen with CTX=65536 for OpenClaw.
#
# Verified on OpenClaw 2026.7.1-2 (2026-07-21). Platform gotchas encoded here:
#   - OpenClaw BLOCKS the PYTHONPATH env for stdio servers ("startup safety"),
#     so wrc_demo must be editable-installed in the venv (done below).
#   - `--cwd <repo>` matters: YOLOE's text encoder resolves relative to CWD.
#   - A half-onboarded config (missing gateway.mode) blocks gateway start.
#   - Custom-provider contextWindow defaults to 128000; it must match what
#     the local server actually allocated or turns die with context overflow.
set -euo pipefail

CAMERAS="isaac,isaac_side"
ARM="isaac"
BRAIN="cosmos"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$(cd "$REPO/.." && pwd)/.demo/bin/python}"
DETECTOR="${WRC_DETECTOR_MODEL:-$REPO/models/yoloe-11s-seg.pt}"
CLASSES="${WRC_DETECT_CLASSES:-cube,banana,bottle,cup,bowl,box,plate,toy}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cameras) CAMERAS="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --brain) BRAIN="$2"; shift 2 ;;   # cosmos | qwen | skip
        *) echo "unknown flag $1"; exit 1 ;;
    esac
done

# 0. OpenClaw ignores PYTHONPATH -> the package itself must be importable.
if ! "$PY" -c "import wrc_demo" 2>/dev/null; then
    echo "[+] installing wrc_demo editable into $PY (OpenClaw blocks PYTHONPATH)"
    uv pip install --python "$PY" --no-deps -e "$REPO"
fi

# 1. Register the MCP server (probes the connection before saving).
echo "[+] registering MCP server (cameras=$CAMERAS arm=$ARM)"
openclaw mcp add wrc-demo \
    --command "$PY" \
    --arg -m --arg wrc_demo.apps.mcp_server \
    --cwd "$REPO" \
    --connect-timeout 120 \
    --env "WRC_CAMERAS=$CAMERAS" --env "WRC_ARM=$ARM" \
    --env "WRC_DETECTOR_MODEL=$DETECTOR" --env "WRC_DETECT_CLASSES=$CLASSES" \
    --env "YOLO_OFFLINE=True" --env "ULTRALYTICS_OFFLINE=True" \
    --env "DISPLAY=${DISPLAY:-:1}"

# 2. Point the agent at a local brain (custom OpenAI-compatible provider).
case "$BRAIN" in
    cosmos) BASE_URL="http://127.0.0.1:8082/v1"; MODEL_ID="cosmos3-edge"; CTX=32768 ;;
    qwen)   BASE_URL="http://127.0.0.1:8080/v1"
            MODEL_ID="$(curl -sf -m 5 http://127.0.0.1:8080/v1/models \
                        | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["models"][0]["model"])')"
            CTX=65536 ;;
    skip)   BASE_URL="" ;;
    *) echo "unknown --brain $BRAIN (cosmos|qwen|skip)"; exit 1 ;;
esac
if [[ -n "$BASE_URL" ]]; then
    if ! curl -sf -m 5 "$BASE_URL/models" >/dev/null; then
        echo "[!] no server on $BASE_URL — start scripts/serve_${BRAIN}_*.sh first"; exit 1
    fi
    echo "[+] onboarding custom provider $BASE_URL ($MODEL_ID)"
    openclaw onboard --non-interactive --accept-risk --mode local \
        --auth-choice custom-api-key \
        --custom-base-url "$BASE_URL" \
        --custom-model-id "$MODEL_ID" \
        --custom-compatibility openai \
        --custom-image-input
    # onboard writes contextWindow=128000 regardless of the server; fix it to
    # what the local server actually allocated or long turns overflow.
    CTX="$CTX" MODEL_ID="$MODEL_ID" "$PY" - <<'PY'
import json, os, pathlib
p = pathlib.Path.home() / ".openclaw/openclaw.json"
cfg = json.loads(p.read_text())
mid, ctx = os.environ["MODEL_ID"], int(os.environ["CTX"])
for prov in cfg.get("models", {}).get("providers", {}).values():
    for m in prov.get("models", []):
        if m.get("id") == mid:
            m["contextWindow"] = ctx
p.write_text(json.dumps(cfg, indent=2))
print(f"[+] contextWindow={ctx} for {mid}")
PY
fi

# 3. Gateway as a persistent service + web chat.
openclaw config set gateway.mode local >/dev/null 2>&1 || true
openclaw gateway install 2>/dev/null || true
openclaw gateway restart 2>/dev/null || openclaw gateway start
openclaw config validate

cat <<EOF
[+] done. Open the web chat:
      http://127.0.0.1:18789/
    auth token (local): jq -r .gateway.auth.token ~/.openclaw/openclaw.json
    (or just run: openclaw dashboard)
    Try: "pick up the pink cube", "describe the scene", "wave".
    Test a turn headless: openclaw agent --agent main -m "describe the scene"
EOF
