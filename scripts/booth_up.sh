#!/usr/bin/env bash
# Booth pre-flight: catch every silent-degradation mode BEFORE the first
# attendee group, and print exactly what to launch. This script changes no
# state.
#
#   ./scripts/booth_up.sh                 # full pre-flight, mock arm assumed
#   ARM=rebot_rs ./scripts/booth_up.sh    # also checks can0
#
# Every check explains its failure mode: the worst booth bugs here are the
# SILENT ones (OBB fallback, vanished detections, watcher stalls).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$(cd "$REPO/.." && pwd)/.demo/bin/python}"
ARM="${ARM:-rebot_rs}"
PORT="${CASCADE_STREAM_PORT:-8090}"
GGX_PORT="${CASCADE_GRASPGENX_PORT:-5556}"   # the port the robot server dials
FAIL=0

ok()   { echo "[+] $*"; }
warn() { echo "[!] $*"; FAIL=1; }

echo "=== cascade booth pre-flight ($(date '+%H:%M')) ==="

# 1. interpreter + package
if [[ -x "$PY" ]] && "$PY" -c "import cascade" 2>/dev/null; then
    ok "venv python: $PY"
else
    warn "no working interpreter at $PY (set PY=... to your .demo venv)"
fi

# 2. offline mode: without these, ultralytics (SAM) looks its assets up on
#    GitHub and can stall the 3 Hz watcher for seconds per tick.
if grep -q "YOLO_OFFLINE" "$REPO/.mcp.json" 2>/dev/null; then
    ok ".mcp.json exports YOLO_OFFLINE"
else
    warn ".mcp.json lacks YOLO_OFFLINE -- regenerate:" \
         "$PY scripts/setup_agents.py --host claude --camera l515 --arm $ARM" \
         "--detect-classes '<prop nouns>' --hide-tools reset_stop --write"
fi

# 2b. booth tuning overlay: bounded worst cases for 15-min sessions
#     (configs/booth.yaml via CASCADE_BOOTH; see BOOTH_RUNBOOK.md §1)
if [[ ! -f "$REPO/configs/booth.yaml" ]]; then
    warn "configs/booth.yaml MISSING (broken checkout) -- the MCP server" \
         "will refuse to start with CASCADE_BOOTH set"
fi
BOOTH_VAL="$(grep -oE '"CASCADE_BOOTH": *"[^"]*"' "$REPO/.mcp.json" 2>/dev/null \
             | sed -E 's/.*"CASCADE_BOOTH": *"([^"]*)".*/\1/' | tr -d ' ' \
             | tr '[:upper:]' '[:lower:]')"
case "$BOOTH_VAL" in
    ""|0|false|no|off)
        warn "booth tuning NOT active -- dev timing budgets (120 s picks)." \
             "Regenerate .mcp.json with: --env CASCADE_BOOTH=1" ;;
    *)  ok "booth tuning active (CASCADE_BOOTH=$BOOTH_VAL in .mcp.json)" ;;
esac

# 3. .mcp.json must pin the rig you mean to drive (it has pinned the Isaac
#    sim stack before -- confusing, though harmless).
if grep -q '"CASCADE_ARM": "'"$ARM"'"' "$REPO/.mcp.json" 2>/dev/null; then
    ok ".mcp.json arm profile: $ARM"
else
    warn ".mcp.json CASCADE_ARM is not '$ARM' -- attendees would drive the wrong arm"
fi

# 4. GraspGen-X server: grasping SILENTLY falls back to the analytic OBB
#    planner when :5556 is down (booth rule) -- quality drops and each pick
#    gains the client timeout in dead air. Loud is the whole point:
if (exec 3<>"/dev/tcp/127.0.0.1/$GGX_PORT") 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    ok "GraspGen-X server on :$GGX_PORT"
else
    warn "GraspGen-X DOWN on :$GGX_PORT -- learned grasps unavailable, silent OBB" \
         "fallback active. Start it: scripts/serve_graspgenx.sh"
fi

# 5. local LLM fallback (venue internet dies -> point the MCP host here)
if curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    ok "local Qwen serving on :8080"
else
    warn "local Qwen NOT serving on :8080 -- no offline brain if venue" \
         "internet drops. Start it: scripts/serve_qwen_llamacpp.sh"
fi

# 5b. SAM weights (configs/demo.yaml segmenter.model = <repo>/models/sam2.1_l.pt):
#     without them the brain's boxes carry no masks (observed-finger gate,
#     colour identity and learned grasp clouds degrade), and offline
#     ultralytics never downloads them.
SAM="$REPO/models/sam2.1_l.pt"
if [[ -s "$SAM" ]]; then
    ok "SAM weights: $SAM"
else
    warn "no SAM weights: detections without masks; run the installer" \
         "(scripts/install.sh --profile spark places $SAM)"
fi

# 5c. perception brain: object names come from the Qwen brain at
#     configs/demo.yaml's endpoint (same variable, same default). The key
#     FILE goes the way the detector sends it -- on stdin, never in argv.
PERCEPTION_URL="${CASCADE_PERCEPTION_URL:-http://127.0.0.1:8080/v1}"
PERCEPTION_KEY_FILE="${CASCADE_PERCEPTION_KEY_FILE:-$HOME/.cascade/nemoclaw/llama.key}"
PERCEPTION_KEY_FILE="${PERCEPTION_KEY_FILE/#\~/$HOME}"
PERCEPTION_MODELS="${PERCEPTION_URL%/}/models"
perception_answers() {
    if [[ -r "$PERCEPTION_KEY_FILE" ]]; then
        printf 'Authorization: Bearer %s\n' "$(<"$PERCEPTION_KEY_FILE")" \
            | curl -sf -m 2 -H @- -o /dev/null "$PERCEPTION_MODELS" 2>/dev/null
    else
        curl -sf -m 2 -o /dev/null "$PERCEPTION_MODELS" 2>/dev/null
    fi
}
if perception_answers; then
    ok "perception brain answers at $PERCEPTION_URL"
else
    warn "no object names: start a Qwen brain or set CASCADE_PERCEPTION_URL" \
         "($PERCEPTION_MODELS did not answer within 2 s)"
fi

# 6. CAN bus (real arm only)
if [[ "$ARM" == "rebot_rs" ]]; then
    if ip link show can0 2>/dev/null | grep -q "state UP"; then
        ok "can0 up"
    else
        warn "can0 not up: sudo ip link set can0 up type can bitrate 1000000" \
             "(and kill motorbridge-gateway/Studio: host-id 0xFD conflict)"
    fi
fi

# 7. learned state: the day-long learning arc is part of the demo -- keep
#    it, but snapshot the morning baseline so a bad rehearsal can be undone
#    (scripts/booth_reset.sh --restore-brain).
GM="$HOME/.cascade/grasp_memory.json"
if [[ -f "$GM" && ! -f "$GM.morning" ]]; then
    cp "$GM" "$GM.morning"
    ok "grasp memory baseline snapshotted -> $GM.morning"
elif [[ -f "$GM.morning" ]]; then
    ok "grasp memory baseline exists ($GM.morning)"
else
    ok "no grasp memory yet (fresh brain: today's first groups are the control group)"
fi

# 8. dashboard URL for the big screen / host phone. lan_ip() inside the
#    server probes 8.8.8.8 and can return 127.0.0.1 on an air-gapped LAN --
#    trust this instead:
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "=== dashboard (big screen + staff phone STOP button) ==="
echo "    http://${IP:-<booth-ip>}:$PORT/"
echo
if [[ "$FAIL" -eq 0 ]]; then
    echo "=== ALL CHECKS GREEN. Launch the attendee session: ==="
else
    echo "=== FIX THE [!] LINES ABOVE, then launch: ==="
fi
echo "    cd $REPO && claude       # or your MCP host of choice"
echo "    (the cascade MCP server starts automatically from .mcp.json;"
echo "     perception pre-warms, motors stay off until the first motion)"
exit "$FAIL"
