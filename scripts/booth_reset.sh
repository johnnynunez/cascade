#!/usr/bin/env bash
# Between-groups reset (target: under 60 seconds, run while Q&A wraps up).
#
#   ./scripts/booth_reset.sh                  # normal reset: KEEP learned state
#   ./scripts/booth_reset.sh --wipe-brain     # deliberate fresh-brain demo day
#   ./scripts/booth_reset.sh --restore-brain  # undo a contaminated session
#                                             # (restores the booth_up.sh baseline)
#
# MEMORY POLICY (deliberate, not an accident): ~/.cascade/grasp_memory.json
# and runs/experience.json persist across groups because the robot getting
# measurably better over the day IS the long-running-agent story. Wipe only
# for a cold-start narrative; restore if picks degrade after a bad streak.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$(cd "$REPO/.." && pwd)/.demo/bin/python}"
PORT="${CASCADE_STREAM_PORT:-8090}"
GM="$HOME/.cascade/grasp_memory.json"

case "${1:-}" in
    --wipe-brain)
        rm -f "$GM" "$REPO/runs/experience.json"
        echo "[+] learned state wiped (grasp memory + tier-2 habits): cold start"
        echo "[!] RESTART the MCP session now: a running server holds the"
        echo "    memory in RAM and re-saves it on the next grasp, silently"
        echo "    undoing this wipe"
        ;;
    --restore-brain)
        if [[ -f "$GM.morning" ]]; then
            cp "$GM.morning" "$GM"
            echo "[+] grasp memory restored to this morning's baseline"
            echo "    (takes effect on the NEXT server start: the running"
            echo "     server keeps its in-memory copy)"
        else
            echo "[!] no baseline at $GM.morning (booth_up.sh creates it)"; exit 1
        fi
        ;;
    "") ;;
    *)  echo "usage: booth_reset.sh [--wipe-brain|--restore-brain]"; exit 2 ;;
esac

# Refresh the "robot diary" for the Q&A beat: export_markdown is not called
# automatically anywhere in the pipeline, so regenerate it here from the
# session's persisted grasp outcomes.
if [[ -f "$GM" ]]; then
    "$PY" - <<'EOF' || echo "[!] diary refresh failed (non-fatal)"
from pathlib import Path
from cascade.memory.grasp_memory import GraspOutcomeMemory

d = Path.home() / ".cascade"
GraspOutcomeMemory(d / "grasp_memory.json").export_markdown(d / "GRASP_MEMORY.md")
print(f"[+] robot diary refreshed: {d / 'GRASP_MEMORY.md'}")
EOF
fi

echo
echo "=== reset checklist (hands, ~60 s, parallel with Q&A) ==="
echo "  1. arm parked?           last command must have been move_home /"
echo "                           'go home' -- NEVER disconnect a loaded arm"
echo "  2. props back on tape:   every prop on its taped zone, bin emptied"
echo "  3. cup + spare props     back at the host position"
echo "  4. /clear in the MCP host chat (fresh context, server keeps running)"
echo
echo "=== perception pre-flight for the NEXT group ==="
STATE="$(curl -sf -m 3 "http://127.0.0.1:$PORT/state" 2>/dev/null)" || {
    echo "[!] dashboard /state unreachable on :$PORT -- is the MCP session up?"
    exit 1
}
echo "$STATE" | python3 -c '
import json, sys
state = json.load(sys.stdin)
objs = state.get("objects", [])
if not objs:
    print("[!] world model EMPTY: check props on table, then the silent killers:")
    print("    encoder CWD + YOLO_OFFLINE (scripts/booth_up.sh checks both)")
    sys.exit(1)
stale = [o for o in objs if o.get("state") == "remembered"]
print(f"[+] {len(objs)} objects in the world model:")
for o in objs:
    mark = "LIVE" if o.get("state") != "remembered" else "remembered (ghost? nudge it)"
    print(f"      {o.get('label','?'):<16} {mark}")
if stale:
    print("[!] remembered-state rows above are invisible to the camera right now:")
    print("    off-table, occluded, or ghosts -- fix before seating the group")
    sys.exit(1)
'
echo "[+] table matches the world model: seat the next group"
