#!/usr/bin/env bash
# One-click bring-up for the cascade demo: the simulator (when there is one),
# the OpenClaw 2.0 chat with cascade's robot tools registered, and the proof
# that the tools answer -- in ONE command.
#
#   ./scripts/launch.sh                     # auto: Isaac Sim if installed, else MuJoCo
#   ./scripts/launch.sh --sim isaac         # Isaac Sim (bridge + editor window) + OpenClaw
#   ./scripts/launch.sh --sim mujoco        # MuJoCo (CPU physics, rendered camera) + OpenClaw
#   ./scripts/launch.sh --sim none --arm rebot_rs --cameras l515   # REAL ARM: OpenClaw only
#   ./scripts/launch.sh --down              # stop what this script started
#   ./scripts/launch.sh --dry-run           # print the plan, touch nothing
#
# What "one click" means here, in dependency order (ASPIRE's launch_servers.py
# and RPent's `--dashboard` are the reference shape: start services, WAIT on
# a readiness signal for each, refuse to declare victory without a probe):
#   1. deps        venv python that imports cascade (+ mujoco for --sim mujoco)
#   2. assets      robot meshes for the sim arm (scripts/fetch_robot_assets.py)
#   3. simulator   isaac: scripts/isaac_bridge.py under Isaac's python.sh, wait
#                  for the TCP bridge on :8611 (default 240 s -- Kit is slow)
#                  mujoco: nothing to start; the MCP server owns the world and
#                  opens the passive viewer itself (CASCADE_VIEW=1 + DISPLAY)
#                  none: nothing
#   4. openclaw    install/upgrade guard (>= 2.0), gateway up, cascade MCP
#                  server registered IDEMPOTENTLY (`mcp set`, not `mcp add`),
#                  brain resolved (--brain auto keeps whatever auth OpenClaw
#                  already has; a local server is only used when it answers)
#   5. proof       `openclaw mcp probe cascade` must list the robot tools
#   6. chat        `openclaw dashboard` opens the browser (--no-open to skip)
#
# Platform notes (all verified on macOS 26 / OpenClaw 2026.9.3, 2026-09-09):
#   - no `ss` on macOS: port checks use a python socket probe
#   - `openclaw mcp add` FAILS when the entry exists; `mcp set <name> <json>`
#     is the idempotent form and is what this script uses
#   - `openclaw update` runs `doctor` afterwards and doctor FAILS while the
#     gateway service is running -> the guard stops the gateway first
#   - OpenClaw blocks PYTHONPATH for stdio servers, so cascade must be
#     editable-installed into the venv it launches (checked, fixed if not)
#   - the gateway serves a STALE tool list until restarted after `mcp set`
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM="auto"            # auto | isaac | mujoco | none
ARM=""                # default depends on SIM
CAMERAS=""            # default depends on SIM
BRAIN="auto"          # auto | keep | cosmos | cosmos-sglang | qwen
OPEN_CHAT=1
DRY=0
DOWN=0
ISAAC_GUI=1
ISAAC_WAIT_S="${ISAAC_WAIT_S:-240}"
BRIDGE_PORT="${CASCADE_BRIDGE_PORT:-8611}"
GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
STATE_DIR="${CASCADE_LAUNCH_STATE:-$REPO/runs/.launch}"
MCP_NAME="${CASCADE_MCP_NAME:-cascade}"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim) SIM="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --cameras) CAMERAS="$2"; shift 2 ;;
        --brain) BRAIN="$2"; shift 2 ;;
        --no-open) OPEN_CHAT=0; shift ;;
        --headless) ISAAC_GUI=0; shift ;;
        --dry-run) DRY=1; shift ;;
        --down) DOWN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown flag $1"; usage; exit 2 ;;
    esac
done

log()  { printf '[launch] %s\n' "$*"; }
warn() { printf '[launch] WARNING: %s\n' "$*" >&2; }
die()  { printf '[launch] ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [[ $DRY == 1 ]]; then printf '        $ %s\n' "$*"; else "$@"; fi; }

# ── python: the repo venv first, then whatever `python3` is ─────────────────
pick_python() {
    local cand
    for cand in "${PY:-}" "$REPO/.venv/bin/python" "$(cd "$REPO/.." 2>/dev/null && pwd)/.demo/bin/python" "$(command -v python3 || true)"; do
        [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return; }
    done
    die "no python found; create the venv first: uv venv $REPO/.venv && uv pip install -e '$REPO[sim]'"
}
PY="$(pick_python)"

port_open() {  # port_open <port> [host]  -- macOS has no `ss`
    "$PY" - "$1" "${2:-127.0.0.1}" <<'PYEOF'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
try:
    s.connect((sys.argv[2], int(sys.argv[1]))); sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PYEOF
}
wait_port() {  # wait_port <port> <seconds> <what>
    local port=$1 secs=$2 what=$3 i
    for ((i = 0; i < secs; i++)); do
        port_open "$port" && return 0
        (( i % 15 == 14 )) && log "still waiting for $what on :$port (${i}s)"
        sleep 1
    done
    return 1
}

# ── --down: stop what a previous run started, nothing else ──────────────────
if [[ $DOWN == 1 ]]; then
    if [[ -f "$STATE_DIR/isaac_bridge.pid" ]]; then
        pid=$(cat "$STATE_DIR/isaac_bridge.pid")
        if kill -0 "$pid" 2>/dev/null; then log "stopping Isaac bridge (pid $pid)"; run kill "$pid"; fi
        run rm -f "$STATE_DIR/isaac_bridge.pid"
    fi
    if [[ -f "$STATE_DIR/gateway.started" ]]; then
        log "stopping OpenClaw gateway (this script started it)"
        run openclaw gateway stop || true
        run rm -f "$STATE_DIR/gateway.started"
    else
        log "gateway was already running before launch.sh; leaving it up"
    fi
    log "down."
    exit 0
fi

# ── resolve --sim auto ───────────────────────────────────────────────────────
find_isaac_python() {
    local c
    for c in "${ISAACSIM_PATH:-}" "$HOME/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release" \
             "$HOME/isaacsim" "$HOME/.local/share/ov/pkg"/isaac-sim-* /isaac-sim; do
        [[ -n "$c" && -x "$c/python.sh" ]] && { echo "$c/python.sh"; return 0; }
    done
    # pip-installed Isaac Sim into the active python?
    "$PY" -c "import isaacsim" >/dev/null 2>&1 && { echo "$PY"; return 0; }
    return 1
}
ISAAC_PY=""
if [[ "$SIM" == "auto" ]]; then
    if ISAAC_PY="$(find_isaac_python)"; then SIM="isaac"
    elif "$PY" -c "import mujoco" >/dev/null 2>&1; then SIM="mujoco"
    else SIM="none"; warn "neither Isaac Sim nor mujoco importable from $PY -> --sim none (real hardware)"
    fi
    log "--sim auto -> $SIM"
elif [[ "$SIM" == "isaac" ]]; then
    ISAAC_PY="$(find_isaac_python)" || die "Isaac Sim not found (set ISAACSIM_PATH to the folder holding python.sh)"
fi

case "$SIM" in
    isaac)  ARM="${ARM:-isaac}";        CAMERAS="${CAMERAS:-isaac,isaac_side}" ;;
    mujoco) ARM="${ARM:-so101_mujoco}"; CAMERAS="${CAMERAS:-mujoco_scene}" ;;
    none)   [[ -n "$ARM" && -n "$CAMERAS" ]] || die "--sim none drives REAL hardware: pass --arm <profile> --cameras <profile[,profile]> explicitly (no guessing which robot is plugged in)" ;;
    *) die "unknown --sim $SIM (auto|isaac|mujoco|none)" ;;
esac
log "plan: sim=$SIM arm=$ARM cameras=$CAMERAS brain=$BRAIN python=$PY"
[[ $DRY == 1 ]] && log "(dry run: commands are printed, nothing is executed)"
mkdir -p "$STATE_DIR" 2>/dev/null || true

# ── 1. deps ─────────────────────────────────────────────────────────────────
if ! "$PY" -c "import cascade" >/dev/null 2>&1; then
    log "cascade not importable from $PY -> editable install (OpenClaw blocks PYTHONPATH)"
    if command -v uv >/dev/null; then run uv pip install --python "$PY" --no-deps -e "$REPO"
    else run "$PY" -m pip install --no-deps -e "$REPO"; fi
fi
if [[ "$SIM" == "mujoco" ]] && ! "$PY" -c "import mujoco" >/dev/null 2>&1; then
    log "mujoco missing -> installing the [sim] extra"
    if command -v uv >/dev/null; then run uv pip install --python "$PY" -e "$REPO[sim]"
    else run "$PY" -m pip install -e "$REPO[sim]"; fi
fi

# ── 2. assets ───────────────────────────────────────────────────────────────
if [[ "$SIM" == "mujoco" ]]; then
    ROBOT="${ARM%%_*}"   # so101_mujoco -> so101, piper_mujoco -> piper
    MJCF="$("$PY" - "$ARM" <<'PYEOF' 2>/dev/null || true
import sys
from cascade.config import load_demo_config
print(load_demo_config(arm=sys.argv[1]).arm.get("mjcf") or "")
PYEOF
)"
    if [[ -n "$MJCF" && ! -f "$MJCF" ]]; then
        log "robot MJCF missing ($MJCF) -> fetching assets for $ROBOT"
        run "$PY" "$REPO/scripts/fetch_robot_assets.py" "$ROBOT"
    fi
fi

# ── 3. simulator ────────────────────────────────────────────────────────────
if [[ "$SIM" == "isaac" ]]; then
    if port_open "$BRIDGE_PORT"; then
        log "Isaac bridge already answering on :$BRIDGE_PORT -> reusing it"
    else
        gui_flag=(); [[ $ISAAC_GUI == 1 ]] && gui_flag=(--gui)
        log "starting Isaac Sim bridge (${gui_flag[*]:-headless}) -> $STATE_DIR/isaac_bridge.log"
        if [[ $DRY == 1 ]]; then
            printf '        $ %s %s --port %s %s &\n' "$ISAAC_PY" "$REPO/scripts/isaac_bridge.py" "$BRIDGE_PORT" "${gui_flag[*]:-}"
        else
            nohup "$ISAAC_PY" "$REPO/scripts/isaac_bridge.py" --port "$BRIDGE_PORT" "${gui_flag[@]}" \
                >"$STATE_DIR/isaac_bridge.log" 2>&1 &
            echo $! >"$STATE_DIR/isaac_bridge.pid"
            wait_port "$BRIDGE_PORT" "$ISAAC_WAIT_S" "Isaac bridge" \
                || die "Isaac bridge never listened on :$BRIDGE_PORT after ${ISAAC_WAIT_S}s -- see $STATE_DIR/isaac_bridge.log"
            log "Isaac bridge up on :$BRIDGE_PORT"
        fi
    fi
fi

# ── 4. openclaw ─────────────────────────────────────────────────────────────
if ! command -v openclaw >/dev/null; then
    log "OpenClaw not installed -> npm install -g openclaw@latest"
    run npm install -g openclaw@latest
fi
OC_VER="$(openclaw --version 2>/dev/null | grep -oE '20[0-9]{2}\.[0-9]+\.[0-9]+' | head -1 || true)"
if [[ -n "$OC_VER" ]]; then
    # CalVer: 2.0 == 2026.8.x and later. Refuse to run the 2.0-only flags on 2026.7.
    if "$PY" -c "import sys; y,m,_=map(int,'$OC_VER'.split('.')); sys.exit(0 if (y,m)>=(2026,8) else 1)"; then
        log "OpenClaw $OC_VER (>= 2.0)"
    else
        log "OpenClaw $OC_VER is pre-2.0 -> upgrading (gateway stopped first: doctor fails while it runs)"
        run openclaw gateway stop || true
        run openclaw update --yes --no-restart
        run openclaw doctor --fix || true
    fi
fi

# gateway up (needed before onboarding and before the probe)
run openclaw config set gateway.mode local >/dev/null 2>&1 || true
if port_open "$GATEWAY_PORT"; then
    log "gateway already listening on :$GATEWAY_PORT"
else
    log "starting gateway"
    run openclaw gateway install >/dev/null 2>&1 || true
    run openclaw gateway start >/dev/null 2>&1 || true
    if [[ $DRY == 0 ]]; then
        wait_port "$GATEWAY_PORT" 30 "OpenClaw gateway" || die "gateway never came up on :$GATEWAY_PORT (openclaw gateway status)"
        touch "$STATE_DIR/gateway.started"
    fi
fi

# register the MCP server -- idempotent (`mcp set` replaces; `mcp add` errors on an existing name)
DETECTOR="${CASCADE_DETECTOR_MODEL:-$REPO/models/yoloe-11s-seg.pt}"
CLASSES="${CASCADE_DETECT_CLASSES:-cube,banana,bottle,cup,bowl,box,plate,toy}"
# The MCP server is what opens the MuJoCo viewer, and on macOS
# `mujoco.viewer.launch_passive` REFUSES to run under plain python ("requires
# mjpython") -- the arm then logs a warning and runs headless, i.e. no sim
# window for the audience. mjpython ships in the venv next to python.
MCP_PY="$PY"
if [[ "$SIM" == "mujoco" && "$(uname -s)" == "Darwin" && -x "$(dirname "$PY")/mjpython" ]]; then
    MCP_PY="$(dirname "$PY")/mjpython"
    log "macOS + mujoco: MCP server runs under mjpython so the viewer can open"
fi
MCP_JSON="$("$PY" - "$MCP_PY" "$REPO" "$CAMERAS" "$ARM" "$DETECTOR" "$CLASSES" "$SIM" <<'PYEOF'
import json, os, sys
py, repo, cams, arm, det, classes, sim = sys.argv[1:8]
env = {
    "CASCADE_CAMERAS": cams, "CASCADE_ARM": arm,
    "CASCADE_DETECTOR_MODEL": det, "CASCADE_DETECT_CLASSES": classes,
    "YOLO_OFFLINE": "True", "ULTRALYTICS_OFFLINE": "True",
}
# Sim runs open the physics viewer from the MCP server (CASCADE_VIEW=1 needs
# DISPLAY set; macOS has no DISPLAY, so give it one -- mujoco.viewer ignores
# the value, it only gates the "is there a screen" check).
if sim in ("mujoco", "isaac"):
    env["CASCADE_VIEW"] = "1"
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
print(json.dumps({
    "command": py, "args": ["-m", "cascade.apps.mcp_server"],
    "cwd": os.path.join(repo, "models"),   # YOLOE resolves its text encoder relative to cwd
    "env": env, "connectionTimeoutMs": 120000,
}))
PYEOF
)"
log "registering MCP server '$MCP_NAME' (cameras=$CAMERAS arm=$ARM)"
mkdir -p "$REPO/models"
run openclaw mcp set "$MCP_NAME" "$MCP_JSON"

# brain
brain_local() {  # brain_local <base_url> <model_id> <ctx>
    local base=$1 mid=$2 ctx=$3
    curl -sf -m 5 "$base/models" >/dev/null || die "no model server on $base -- start scripts/serve_*.sh first, or use --brain auto"
    log "onboarding local provider $base ($mid)"
    run openclaw onboard --non-interactive --accept-risk --mode local \
        --auth-choice custom-api-key --custom-base-url "$base" \
        --custom-model-id "$mid" --custom-compatibility openai --custom-image-input --skip-health
    CTX="$ctx" MODEL_ID="$mid" run "$PY" - <<'PYEOF'
import json, os, pathlib
p = pathlib.Path.home() / ".openclaw/openclaw.json"
cfg = json.loads(p.read_text()); mid, ctx = os.environ["MODEL_ID"], int(os.environ["CTX"])
for prov in cfg.get("models", {}).get("providers", {}).values():
    for m in prov.get("models", []):
        if m.get("id") == mid: m["contextWindow"] = ctx
p.write_text(json.dumps(cfg, indent=2)); print(f"[launch] contextWindow={ctx} for {mid}")
PYEOF
}
case "$BRAIN" in
    keep) log "brain: keeping OpenClaw's current model" ;;
    auto)
        # Prefer a local model server that is actually answering; otherwise keep
        # whatever auth OpenClaw already has, but VERIFY it can answer a turn.
        if curl -sf -m 3 http://127.0.0.1:8082/v1/models >/dev/null 2>&1; then brain_local http://127.0.0.1:8082/v1 cosmos3-edge 32768
        elif curl -sf -m 3 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
            MID="$(curl -sf -m 5 http://127.0.0.1:8080/v1/models | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or d.get("models"))[0].get("id") or (d.get("models"))[0]["model"])')"
            brain_local http://127.0.0.1:8080/v1 "$MID" 65536
        else
            DEFAULT_MODEL="$(openclaw models status --json 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("defaultModel") or "")' 2>/dev/null || true)"
            if [[ -z "$DEFAULT_MODEL" ]]; then
                die "no local model server and OpenClaw has no default model: run 'openclaw onboard' once (pick a provider you have auth for), then re-run"
            fi
            log "brain: no local server; keeping OpenClaw's $DEFAULT_MODEL"
        fi ;;
    cosmos)        brain_local http://127.0.0.1:8082/v1 cosmos3-edge 32768 ;;
    cosmos-sglang) brain_local http://127.0.0.1:8083/v1 cosmos3-edge 32768 ;;
    qwen)          MID="$(curl -sf -m 5 http://127.0.0.1:8080/v1/models | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print((d.get("data") or d.get("models"))[0].get("id") or d["models"][0]["model"])')"
                   brain_local http://127.0.0.1:8080/v1 "$MID" 65536 ;;
    *) die "unknown --brain $BRAIN (auto|keep|cosmos|cosmos-sglang|qwen)" ;;
esac

# the gateway serves a stale tool list until restarted after `mcp set`
log "restarting gateway so it loads the '$MCP_NAME' tools"
run openclaw gateway restart >/dev/null 2>&1 || true
[[ $DRY == 0 ]] && { wait_port "$GATEWAY_PORT" 30 "OpenClaw gateway" || die "gateway did not come back after restart"; }
run openclaw config validate >/dev/null

# ── 5. proof: the robot tools must actually be listed ───────────────────────
if [[ $DRY == 0 ]]; then
    log "probing MCP tools (first call loads perception; up to ~60 s)"
    # 2.0's plain `mcp probe` prints only a COUNT; `--json` carries the names,
    # namespaced as <server>__<tool>.
    PROBE_JSON="$(openclaw mcp probe "$MCP_NAME" --json 2>/dev/null || true)"
    read -r NTOOLS HAVE_ROBOT <<<"$(printf '%s' "$PROBE_JSON" | "$PY" -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print("0 0"); sys.exit()
tools = [t.split("__", 1)[-1] for t in d.get("tools", [])]
need = {"pick_and_place", "get_observation", "analyze_scene"}
print(len(tools), int(need <= set(tools)))
' 2>/dev/null || echo "0 0")"
    if [[ "$HAVE_ROBOT" == "1" ]]; then
        log "tools reachable ($NTOOLS listed, incl. pick_and_place / get_observation / analyze_scene)"
    else
        openclaw mcp doctor "$MCP_NAME" --probe 2>&1 | tail -8 >&2
        die "the MCP probe did not list the robot tools ($NTOOLS found) -- see above"
    fi
    # a real turn through the brain, so 'it works' means the LLM answered, not just the wiring
    log "one headless turn through the brain (Reply OK)..."
    ANSWER="$(openclaw agent exec 'Reply with exactly the word OK and nothing else.' --json 2>/dev/null \
        | "$PY" -c 'import json,sys
d=json.load(sys.stdin)
def walk(o):
    if isinstance(o,str): yield o
    elif isinstance(o,dict): [ (yield from walk(v)) for v in o.values() ]
    elif isinstance(o,list): [ (yield from walk(v)) for v in o ]
print(" ".join(s for s in walk(d) if "OK" in s)[:80])' 2>/dev/null || true)"
    if [[ "$ANSWER" == *OK* ]]; then log "brain answered."
    else warn "the brain did not answer a trivial turn -- check 'openclaw models status' (auth) before blaming the robot tools"; fi
fi

# ── 6. chat ─────────────────────────────────────────────────────────────────
cat <<EOF
[launch] READY   sim=$SIM  arm=$ARM  cameras=$CAMERAS
         chat:      http://127.0.0.1:$GATEWAY_PORT/   (openclaw dashboard)
         headless:  openclaw agent exec "describe the scene"
         try:       "what do you see?"  "pick and place the red object"  "did it actually move?"
$( [[ "$SIM" == "mujoco" ]] && echo '         viewer:    the MuJoCo window opens on the FIRST motion command (arm is lazy until then)' )
$( [[ "$SIM" == "isaac"  ]] && echo "         isaac:     bridge :$BRIDGE_PORT, log $STATE_DIR/isaac_bridge.log" )
         stop:      ./scripts/launch.sh --down
EOF
if [[ $OPEN_CHAT == 1 && $DRY == 0 ]]; then
    openclaw dashboard >/dev/null 2>&1 || warn "could not open the browser; use the URL above"
fi
