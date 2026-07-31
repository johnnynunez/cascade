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
#   - `--cwd <repo>/models` matters, and it must be models/ NOT the repo root:
#     YOLOE resolves its `mobileclip_blt.ts` text encoder RELATIVE TO CWD. From
#     the repo root ultralytics cannot see the 600 MB file in models/, tries to
#     download it (even with YOLO_OFFLINE=True), writes a truncated 9 MB file
#     into the repo root, and every get_observation then dies with
#     "PytorchStreamReader failed reading zip archive". The agent surfaces that
#     as "a runtime error loading a checkpoint" and quietly stops perceiving.
#     If you hit it: rm the stray mobileclip_blt.ts from the repo root.
#   - A half-onboarded config (missing gateway.mode) blocks gateway start.
#   - The gateway must be RUNNING before `openclaw onboard --non-interactive`,
#     and must be RESTARTED after `openclaw mcp add` or it serves a stale tool
#     list (the agent then answers from prose with no tools at all).
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
# cwd is models/ because YOLOE's text encoder resolves relative to it (see the
# header note); DETECTOR is passed absolute so it works from anywhere.
echo "[+] registering MCP server (cameras=$CAMERAS arm=$ARM)"
openclaw mcp add wrc-demo \
    --command "$PY" \
    --arg -m --arg wrc_demo.apps.mcp_server \
    --cwd "$REPO/models" \
    --connect-timeout 120 \
    --env "WRC_CAMERAS=$CAMERAS" --env "WRC_ARM=$ARM" \
    --env "WRC_DETECTOR_MODEL=$DETECTOR" --env "WRC_DETECT_CLASSES=$CLASSES" \
    --env "YOLO_OFFLINE=True" --env "ULTRALYTICS_OFFLINE=True" \
    --env "DISPLAY=${DISPLAY:-:1}"

# 2. Gateway FIRST: `openclaw onboard --non-interactive` health-checks it and
#    aborts if nothing is listening, and a gateway started before `mcp add`
#    serves a stale tool list. Start it, then onboard, then restart so the new
#    MCP server is actually picked up.
openclaw config set gateway.mode local >/dev/null 2>&1 || true
openclaw gateway install 2>/dev/null || true
if ! ss -ltn 2>/dev/null | grep -q ':18789'; then
    echo "[+] starting gateway"
    openclaw gateway start 2>/dev/null || nohup openclaw gateway run >/tmp/openclaw-gateway.log 2>&1 &
    for _ in $(seq 1 30); do
        ss -ltn 2>/dev/null | grep -q ':18789' && break
        sleep 1
    done
fi
ss -ltn 2>/dev/null | grep -q ':18789' \
    || { echo "[!] gateway never came up on 18789 — see /tmp/openclaw-gateway.log"; exit 1; }

# 3. Point the agent at a local brain (custom OpenAI-compatible provider).
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
    # what the local server actually allocated or long turns overflow. Ask the
    # server rather than trusting the constant above: llama.cpp reports its
    # real n_ctx on /props, and a stale guess overflows mid-turn.
    REAL_CTX="$(curl -sf -m 5 "${BASE_URL%/v1}/props" 2>/dev/null \
        | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("default_generation_settings",{}).get("n_ctx") or "")' 2>/dev/null || true)"
    [[ -n "$REAL_CTX" ]] && CTX="$REAL_CTX"
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

# 4. Restart so the gateway picks up the MCP server registered in step 1.
echo "[+] restarting gateway to load the wrc-demo tools"
openclaw gateway restart >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    ss -ltn 2>/dev/null | grep -q ':18789' && break
    sleep 1
done
openclaw config validate

# 5. Prove the tools are actually reachable before telling the user it works.
echo "[+] probing MCP tools (Isaac + YOLOE load takes ~40 s on first call)"
openclaw mcp probe wrc-demo 2>&1 | tail -3

cat <<EOF
[+] done. Open the web chat:
      http://127.0.0.1:18789/
    auth token (local): jq -r .gateway.auth.token ~/.openclaw/openclaw.json
    (or just run: openclaw dashboard)
    Try: "pick up the pink cube", "describe the scene", "wave".
    Test a turn headless: openclaw agent --agent main -m "describe the scene"
EOF
