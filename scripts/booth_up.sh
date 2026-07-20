#!/usr/bin/env bash
# Booth pre-flight: catch every silent-degradation mode BEFORE the first
# attendee group, and print exactly what to launch. This script changes no
# state except ensuring the YOLOE text encoder is reachable from the repo
# root (where MCP hosts launch the server).
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
PORT="${WRC_STREAM_PORT:-8090}"
FAIL=0

ok()   { echo "[+] $*"; }
warn() { echo "[!] $*"; FAIL=1; }

echo "=== wrc_demo booth pre-flight ($(date '+%H:%M')) ==="

# 1. interpreter + package
if [[ -x "$PY" ]] && "$PY" -c "import wrc_demo" 2>/dev/null; then
    ok "venv python: $PY"
else
    warn "no working interpreter at $PY (set PY=... to your .demo venv)"
fi

# 2. YOLOE text encoder must be reachable from the LAUNCH CWD. MCP hosts
#    start the server from the repo root; ultralytics resolves the encoder
#    relative to CWD and every detection silently vanishes if it's missing.
if [[ -e "$REPO/mobileclip_blt.ts" ]]; then
    ok "text encoder reachable from repo root"
elif [[ -e "$REPO/models/mobileclip_blt.ts" ]]; then
    if ln -s "models/mobileclip_blt.ts" "$REPO/mobileclip_blt.ts" 2>/dev/null; then
        ok "text encoder symlinked into repo root (launch dir for MCP hosts)"
    else
        warn "could not symlink the text encoder into $REPO (permissions?)"
    fi
else
    warn "models/mobileclip_blt.ts MISSING: open-vocab detections will all" \
         "silently vanish (it is gitignored -- copy it from another rig)"
fi

# 3. offline mode: without these, ultralytics phones GitHub on class
#    re-embeds and stalls the 3 Hz watcher for seconds per tick.
if grep -q "YOLO_OFFLINE" "$REPO/.mcp.json" 2>/dev/null; then
    ok ".mcp.json exports YOLO_OFFLINE"
else
    warn ".mcp.json lacks YOLO_OFFLINE -- regenerate:" \
         "$PY scripts/setup_agents.py --host claude --camera l515 --arm $ARM" \
         "--detect-classes '<prop nouns>' --hide-tools reset_stop --write"
fi

# 3b. booth tuning overlay: bounded worst cases for 15-min sessions
#     (configs/booth.yaml via WRC_BOOTH; see BOOTH_RUNBOOK.md §1)
if [[ ! -f "$REPO/configs/booth.yaml" ]]; then
    warn "configs/booth.yaml MISSING (broken checkout) -- the MCP server" \
         "will refuse to start with WRC_BOOTH set"
fi
BOOTH_VAL="$(grep -oE '"WRC_BOOTH": *"[^"]*"' "$REPO/.mcp.json" 2>/dev/null \
             | sed -E 's/.*"WRC_BOOTH": *"([^"]*)".*/\1/' | tr -d ' ' \
             | tr '[:upper:]' '[:lower:]')"
case "$BOOTH_VAL" in
    ""|0|false|no|off)
        warn "booth tuning NOT active -- dev timing budgets (120 s picks)." \
             "Regenerate .mcp.json with: --env WRC_BOOTH=1" ;;
    *)  ok "booth tuning active (WRC_BOOTH=$BOOTH_VAL in .mcp.json)" ;;
esac

# 4. .mcp.json must pin the rig you mean to drive (it has pinned the Isaac
#    sim stack before -- confusing, though harmless).
if grep -q '"WRC_ARM": "'"$ARM"'"' "$REPO/.mcp.json" 2>/dev/null; then
    ok ".mcp.json arm profile: $ARM"
else
    warn ".mcp.json WRC_ARM is not '$ARM' -- attendees would drive the wrong arm"
fi

# 5. GraspGen-X server: grasping SILENTLY falls back to the analytic OBB
#    planner when :5556 is down (booth rule) -- quality drops and each pick
#    gains the client timeout in dead air. Loud is the whole point:
if (exec 3<>/dev/tcp/127.0.0.1/5556) 2>/dev/null; then
    exec 3>&- 2>/dev/null || true
    ok "GraspGen-X server on :5556"
else
    warn "GraspGen-X DOWN on :5556 -- learned grasps unavailable, silent OBB" \
         "fallback active. Start it: scripts/serve_graspgenx.sh"
fi

# 6. local LLM fallback (venue internet dies -> point the MCP host here)
if curl -sf -m 2 "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    ok "local Qwen serving on :8080"
else
    warn "local Qwen NOT serving on :8080 -- no offline brain if venue" \
         "internet drops. Start it: scripts/serve_qwen_llamacpp.sh"
fi

# 7. CAN bus (real arm only)
if [[ "$ARM" == "rebot_rs" ]]; then
    if ip link show can0 2>/dev/null | grep -q "state UP"; then
        ok "can0 up"
    else
        warn "can0 not up: sudo ip link set can0 up type can bitrate 1000000" \
             "(and kill motorbridge-gateway/Studio: host-id 0xFD conflict)"
    fi
fi

# 8. learned state: the day-long learning arc is part of the demo -- keep
#    it, but snapshot the morning baseline so a bad rehearsal can be undone
#    (scripts/booth_reset.sh --restore-brain).
GM="$HOME/.wrc_demo/grasp_memory.json"
if [[ -f "$GM" && ! -f "$GM.morning" ]]; then
    cp "$GM" "$GM.morning"
    ok "grasp memory baseline snapshotted -> $GM.morning"
elif [[ -f "$GM.morning" ]]; then
    ok "grasp memory baseline exists ($GM.morning)"
else
    ok "no grasp memory yet (fresh brain: today's first groups are the control group)"
fi

# 9. dashboard URL for the big screen / host phone. lan_ip() inside the
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
echo "    (the wrc-demo MCP server starts automatically from .mcp.json;"
echo "     perception pre-warms, motors stay off until the first motion)"
exit "$FAIL"
