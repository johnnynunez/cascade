#!/usr/bin/env bash
# One-click bring-up for the cascade demo: the simulator (when there is one),
# the OpenClaw 2.0 chat with cascade's robot tools registered, and the proof
# that the tools answer -- in ONE command.
#
#   ./scripts/launch.sh                     # auto: Isaac Sim if installed, else MuJoCo
#   ./run.sh                               # ONE CLICK on a fresh clone: --setup --sim auto
#   ./scripts/launch.sh --sim isaac         # Isaac Sim (bridge + editor window) + OpenClaw
#   ./scripts/launch.sh --setup --sim isaac # same, but first create the venv, install the
#                                          # extras this mode needs, fetch assets, install
#                                          # OpenClaw -- idempotent, safe to re-run
#   ./scripts/launch.sh --check --sim isaac # preflight only: report what is missing, exit
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
#   3b. sidecars   occupancy bridge on :5557 (scripts/serve_occupancy.sh:
#                  nvblox on NVIDIA GPUs, warp TSDF+EDT anywhere; --occupancy
#                  none to skip) and, in SIM modes only, the GraspGen-X
#                  protocol stub on :5556 (the real server is a CUDA sidecar
#                  you start yourself; --graspgenx none|stub|external). Each
#                  is waited on and PROBED; the demo banner names what answered.
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
SETUP=0               # --setup: create venv + install extras + fetch assets + install OpenClaw
ROBOT_TURN=1          # --no-robot-turn: skip the real pick_and_place proof (sim modes only)
CHECK=0               # --check: preflight report only
OCCUPANCY="auto"      # auto | nvblox | warp | voxel | none   (bridge backend, or skip)
GRASPGENX="auto"      # auto | stub | external | none  (auto = stub in sim, external otherwise)
OCC_PORT="${CASCADE_OCCUPANCY_PORT:-5557}"
GGX_PORT="${CASCADE_GRASPGENX_PORT:-5556}"
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
        --setup) SETUP=1; shift ;;
        --no-robot-turn) ROBOT_TURN=0; shift ;;
        --check) CHECK=1; shift ;;
        --occupancy) OCCUPANCY="$2"; shift 2 ;;
        --graspgenx) GRASPGENX="$2"; shift 2 ;;
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
    return 1
}
ensure_uv() {
    command -v uv >/dev/null 2>&1 && return 0
    [[ -x "$HOME/.local/bin/uv" ]] && { export PATH="$HOME/.local/bin:$PATH"; return 0; }
    log "installing uv (https://astral.sh/uv)"
    run bash -c 'curl -fsSL https://astral.sh/uv/install.sh | sh' || die "could not install uv"
    export PATH="$HOME/.local/bin:$PATH"
}
if [[ $SETUP == 1 && ! -x "$REPO/.venv/bin/python" && -z "${PY:-}" ]]; then
    ensure_uv
    log "creating $REPO/.venv (python 3.12)"
    run uv venv --python 3.12 "$REPO/.venv" || die "uv venv failed"
fi
PY="$(pick_python)" || die "no python found -- run with --setup (creates $REPO/.venv), or: uv venv $REPO/.venv"

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
    for svc in occupancy_bridge graspgenx_stub; do
        if [[ -f "$STATE_DIR/$svc.pid" ]]; then
            pid=$(cat "$STATE_DIR/$svc.pid")
            if kill -0 "$pid" 2>/dev/null; then log "stopping $svc (pid $pid)"; run kill "$pid"; fi
            run rm -f "$STATE_DIR/$svc.pid"
        fi
    done
    # The gateway keeps one MCP server process per chat SESSION and never
    # reaps them while it runs (measured: two `cascade.apps.mcp_server`
    # processes from finished sessions, each rendering its camera at
    # `fps`, 60 % CPU apiece, hours after the last turn). They are ours:
    # stop them, and leave the gateway itself to whoever started it.
    ORPHANS="$("$PY" - "$REPO" <<'PYEOF' 2>/dev/null || true
import subprocess, sys
repo = sys.argv[1]
out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
for line in out.splitlines():
    pid, _, cmd = line.strip().partition(" ")
    if "cascade.apps.mcp_server" in cmd and repo in cmd:
        print(pid)
PYEOF
)"
    for pid in $ORPHANS; do
        log "stopping MCP server (pid $pid) left behind by a finished chat session"
        run kill "$pid" 2>/dev/null || true
    done
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
    ISAAC_PY="$(find_isaac_python)" || {
        [[ $CHECK == 1 || $DRY == 1 ]] || die "Isaac Sim not found (set ISAACSIM_PATH to the folder holding python.sh); ./run.sh check isaac lists everything missing"
        ISAAC_PY=""
    }
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
pip_install() {  # pip_install <spec...>  -- uv when present (fast, no pip needed in the venv)
    if command -v uv >/dev/null 2>&1 || [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        run uv pip install --python "$PY" "$@"
    else
        run "$PY" -m pip install "$@"
    fi
}
# Extras each mode needs. `kinematics` = pinocchio FK/IK (every motion skill;
# measured: a fresh clone's first pick answered "pinocchio module is missing"
# because only the wire extras were installed); `grasping` = the ZMQ wire for
# the GraspGen-X / occupancy sidecars; `occupancy` = Warp+scipy backend; `llm`
# = OpenAI client (brain + Robo-Dopamine judge); `perception` = YOLOE
# (ultralytics; torch from the host) -- the Isaac cameras use it, the MuJoCo
# scene uses the colour-threshold mock detector and does not. Isaac needs no
# `sim` extra: physics lives in Isaac's own python, cascade talks TCP.
case "$SIM" in
    isaac)  EXTRAS="kinematics,grasping,occupancy,llm,perception" ;;
    mujoco) EXTRAS="sim,kinematics,grasping,occupancy,llm" ;;
    *)      EXTRAS="kinematics,grasping,occupancy,llm,perception" ;;
esac
MISSING_MODS="$("$PY" - "$SIM" <<'PYEOF' 2>/dev/null || echo "cascade"
import importlib.util, sys
sim = sys.argv[1]
need = {"cascade": "cascade", "pinocchio": "kinematics", "zmq": "grasping", "msgpack_numpy": "grasping",
        "openai": "llm", "cv2": "core", "yaml": "core", "warp": "occupancy", "scipy": "occupancy"}
if sim == "mujoco":
    need["mujoco"] = "sim"
else:
    need["ultralytics"] = "perception"
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))
PYEOF
)"
if [[ -n "$MISSING_MODS" ]]; then
    if [[ $SETUP == 1 || $CHECK == 0 ]]; then
        log "python deps missing ($MISSING_MODS) -> installing cascade[$EXTRAS] into $PY"
        pip_install -e "$REPO[$EXTRAS]" || die "dependency install failed"
    else
        warn "python deps missing: $MISSING_MODS (run with --setup)"
    fi
fi
# `--check` is a REPORT: it must never mutate the venv. (Measured: a
# `--check --sim isaac` on a MuJoCo-only venv installed 121 MB of torch
# because these two steps had no CHECK gate, unlike the extras step above.)
if [[ "$SIM" == "mujoco" ]] && ! "$PY" -c "import mujoco" >/dev/null 2>&1; then
    if [[ $CHECK == 1 && $SETUP == 0 ]]; then
        warn "mujoco missing (run with --setup)"
    else
        log "mujoco missing -> installing the [sim] extra"
        pip_install -e "$REPO[sim]"
    fi
fi
if [[ "$EXTRAS" == *perception* ]] && ! "$PY" -c "import torch" >/dev/null 2>&1; then
    # torch is deliberately NOT pinned in pyproject (per-platform builds); the
    # PyPI default is right for macOS (MPS) and CPU Linux, and CUDA x86 gets
    # the CUDA wheel from the default index too. Jetson needs NVIDIA's wheel:
    # install it first and this step is skipped.
    if [[ $CHECK == 1 && $SETUP == 0 ]]; then
        warn "torch missing (YOLOE detector needs it; run with --setup)"
    else
        log "torch missing (YOLOE detector needs it) -> installing the default build for this platform"
        pip_install torch torchvision || die "torch install failed -- install the wheel for this platform, then re-run"
    fi
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
        if [[ $CHECK == 1 && $SETUP == 0 ]]; then
            warn "robot MJCF missing ($MJCF); run with --setup to fetch assets for $ROBOT"
        else
            log "robot MJCF missing ($MJCF) -> fetching assets for $ROBOT"
            run "$PY" "$REPO/scripts/fetch_robot_assets.py" "$ROBOT"
        fi
    fi
fi

if [[ "$SIM" == "isaac" ]]; then
    USD="${CASCADE_USD:-$REPO/assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda}"
    [[ -f "$USD" || $CHECK == 1 ]] || die "Isaac scene USD missing: $USD (it is tracked in git -- is this a partial checkout? set CASCADE_USD to your asset)"
fi
DETECTOR_PT="${CASCADE_DETECTOR_MODEL:-$REPO/models/yoloe-11s-seg.pt}"
[[ -f "$DETECTOR_PT" || $CHECK == 1 ]] || die "detector weights missing: $DETECTOR_PT (tracked in git; set CASCADE_DETECTOR_MODEL to override)"

# ── preflight report (--check) ─────────────────────────────────────────────
if [[ $CHECK == 1 ]]; then
    ok()  { printf '  [ok]      %s\n' "$*"; }
    bad() { printf '  [MISSING] %s\n' "$*"; PREFLIGHT_FAIL=1; }
    PREFLIGHT_FAIL=0
    echo "[launch] preflight for sim=$SIM (python=$PY)"
    "$PY" -c "import cascade" >/dev/null 2>&1 && ok "cascade importable" || bad "cascade not importable (--setup)"
    [[ -z "$MISSING_MODS" ]] && ok "python extras [$EXTRAS]" || bad "python modules: $MISSING_MODS (--setup)"
    command -v openclaw >/dev/null 2>&1 && ok "openclaw $(openclaw --version 2>/dev/null | grep -oE '20[0-9]{2}\.[0-9]+\.[0-9]+' | head -1)" || bad "openclaw CLI (--setup installs it)"
    if [[ "$SIM" == "isaac" ]]; then
        [[ -n "$ISAAC_PY" ]] && ok "Isaac Sim python: $ISAAC_PY" || bad "Isaac Sim (set ISAACSIM_PATH to the folder holding python.sh)"
        [[ -f "$USD" ]] && ok "scene USD: $USD" || bad "scene USD $USD"
        command -v nvidia-smi >/dev/null 2>&1 && ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" || bad "nvidia-smi (Isaac Sim needs an NVIDIA GPU)"
        [[ -n "${DISPLAY:-}" || "$(uname -s)" == "Darwin" || $ISAAC_GUI == 0 ]] && ok "display for the editor window (or --headless)" || bad "no DISPLAY: pass --headless or run from the desktop session"
    fi
    [[ -f "$DETECTOR_PT" ]] && ok "detector weights" || bad "detector weights $DETECTOR_PT"
    curl -sf -m 5 -o /dev/null https://openclaw.ai 2>/dev/null && ok "internet (openclaw.ai reachable)" || warn "no internet: fine if OpenClaw + models are already installed"
    if [[ $PREFLIGHT_FAIL == 1 ]]; then echo "[launch] preflight FAILED -- fix the [MISSING] lines (most: ./scripts/launch.sh --setup --sim $SIM)"; exit 3; fi
    echo "[launch] preflight OK -> ./scripts/launch.sh --sim $SIM"
    exit 0
fi

# ── 3. simulator ────────────────────────────────────────────────────────────
if [[ "$SIM" == "isaac" ]]; then
    if port_open "$BRIDGE_PORT"; then
        log "Isaac bridge already answering on :$BRIDGE_PORT -> reusing it"
    else
        gui_flag=""; [[ $ISAAC_GUI == 1 ]] && gui_flag="--gui"   # string, not array: bash 3.2 + set -u
        [[ -n "$ISAAC_PY" ]] || ISAAC_PY='${ISAACSIM_PATH}/python.sh'   # dry run on a box without Isaac
        log "starting Isaac Sim bridge (${gui_flag:-headless}) -> $STATE_DIR/isaac_bridge.log"
        if [[ $DRY == 1 ]]; then
            printf '        $ %s %s --port %s --usd %s %s &\n' "$ISAAC_PY" "$REPO/scripts/isaac_bridge.py" "$BRIDGE_PORT" "$USD" "${gui_flag:-}"
        else
            # shellcheck disable=SC2086  # $gui_flag is empty or exactly --gui
            nohup "$ISAAC_PY" "$REPO/scripts/isaac_bridge.py" --port "$BRIDGE_PORT" --usd "$USD" $gui_flag \
                >"$STATE_DIR/isaac_bridge.log" 2>&1 &
            echo $! >"$STATE_DIR/isaac_bridge.pid"
            wait_port "$BRIDGE_PORT" "$ISAAC_WAIT_S" "Isaac bridge" \
                || die "Isaac bridge never listened on :$BRIDGE_PORT after ${ISAAC_WAIT_S}s -- see $STATE_DIR/isaac_bridge.log"
            # a listening port is not a working bridge: ping through the real client
            "$PY" - "$BRIDGE_PORT" <<'PYEOF' || die "Isaac bridge on :$BRIDGE_PORT did not answer ping -- see $STATE_DIR/isaac_bridge.log"
import sys
from cascade.sim.bridge_client import BridgeClient
c = BridgeClient(port=int(sys.argv[1]), timeout_s=20.0)
pong = c.request({"op": "ping"})
assert pong.get("ok"), f"ping answered {pong!r}"
st = c.request({"op": "state"})
print(f"[launch] Isaac bridge answers: engine={pong.get('engine')} dofs={len(pong.get('dofs') or [])} state_ok={bool(st.get('ok', True))}")
PYEOF
            log "Isaac bridge up on :$BRIDGE_PORT"
        fi
    fi
fi

# ── 3b. sidecars: occupancy bridge + (sim) GraspGen-X stub ─────────────────
start_sidecar() {  # start_sidecar <name> <port> <wait_s> <cmd...>
    local name="$1" port="$2" wait_s="$3"; shift 3
    if port_open "$port"; then
        log "$name already answering on :$port -> reusing it"
        return 0
    fi
    log "starting $name -> $STATE_DIR/$name.log"
    if [[ $DRY == 1 ]]; then
        printf '        $ %s &\n' "$*"
        return 0
    fi
    nohup "$@" >"$STATE_DIR/$name.log" 2>&1 &
    echo $! >"$STATE_DIR/$name.pid"
    wait_port "$port" "$wait_s" "$name" || die "$name never listened on :$port after ${wait_s}s -- see $STATE_DIR/$name.log"
    log "$name up on :$port"
}

if [[ "$OCCUPANCY" != "none" ]]; then
    # first Warp kernel compile can take ~30 s cold; nvblox loads CUDA kernels
    start_sidecar occupancy_bridge "$OCC_PORT" 120 \
        "$PY" "$REPO/scripts/serve_occupancy_bridge.py" --port "$OCC_PORT" --backend "$OCCUPANCY"
    if [[ $DRY == 0 ]]; then
        OCC_DESC="$("$PY" - "$OCC_PORT" <<'PYEOF'
import sys
from cascade.perception.occupancy import OccupancyClient
try:
    st = OccupancyClient(port=int(sys.argv[1])).probe(timeout_ms=2000)
    print(st.get("describe") or st.get("backend"))
except Exception as e:
    print(f"NOT ANSWERING ({e})")
PYEOF
)"
        log "occupancy: $OCC_DESC"
    fi
else
    log "occupancy bridge skipped (--occupancy none): the clearance gate will be OFF"
fi

case "$GRASPGENX" in
    auto) [[ "$SIM" == "none" ]] && GRASPGENX="external" || GRASPGENX="stub" ;;
esac
case "$GRASPGENX" in
    stub)
        start_sidecar graspgenx_stub "$GGX_PORT" 30 \
            "$PY" "$REPO/scripts/serve_graspgenx_stub.py" --port "$GGX_PORT" --quiet
        log "grasp planner: GraspGen-X PROTOCOL STUB on :$GGX_PORT (analytic grasps; the learned model needs a CUDA sidecar)" ;;
    external)
        if port_open "$GGX_PORT"; then log "grasp planner: GraspGen-X server answering on :$GGX_PORT"
        else warn "no GraspGen-X server on :$GGX_PORT -- grasps will use the analytic OBB planner (the demo banner will say so). Start scripts/serve_graspgenx.sh on a CUDA box, or pass --graspgenx stub"; fi ;;
    none) log "GraspGen-X skipped (--graspgenx none): analytic OBB planner" ;;
    *) die "--graspgenx must be auto|stub|external|none" ;;
esac

# ── 4. openclaw ─────────────────────────────────────────────────────────────
if ! command -v openclaw >/dev/null; then
    log "OpenClaw not installed -> npm install -g openclaw@latest"
    run npm install -g openclaw@latest
fi
if ! command -v openclaw >/dev/null 2>&1; then
    if [[ $SETUP == 1 ]]; then
        log "installing OpenClaw CLI (https://openclaw.ai/install.sh, unattended)"
        run bash -c 'curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard' || die "OpenClaw install failed"
        export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:$PATH"
        command -v openclaw >/dev/null 2>&1 || die "openclaw not on PATH after install -- open a new shell and re-run"
    else
        die "openclaw CLI not found -- run with --setup, or: curl -fsSL https://openclaw.ai/install.sh | bash"
    fi
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
if sim == "mujoco":
    # the physics window itself (the arm profile defaults to view: false)
    env["CASCADE_MJ_VIEW"] = "1"
# requestTimeoutMs: the OpenClaw per-CALL budget (default 60 s). pick_and_place
# PERSISTS for up to grasp.persist_seconds (120 s) by design; at 60 s the
# host sends notifications/cancelled, the server treats a cancel mid-motion
# as the operator walking away and LATCHES THE E-STOP, and every later
# motion fails "e-stop latched" until reset_stop. Measured on a real chat
# turn: the pick completed at 60.0 s, confirmed by physics, and was
# reported as cancelled. Budget = persistence + place + home, with margin.
print(json.dumps({
    "command": py, "args": ["-m", "cascade.apps.mcp_server"],
    "cwd": os.path.join(repo, "models"),   # YOLOE resolves its text encoder relative to cwd
    "env": env, "connectionTimeoutMs": 120000, "requestTimeoutMs": 300000,
}))
PYEOF
)"
# Prune OpenClaw MCP entries whose command (or `-m` module) no longer
# exists (renamed venvs, deleted checkouts, renamed packages). Each dead entry costs EVERY turn a failed spawn
# ("[bundle-mcp] failed to start server ... Connection closed") and a
# catalog retry; on this machine a stale `wrc-demo` pointing at a removed
# package did exactly that on every visitor turn.
if [[ $DRY == 0 ]]; then
    # NB: `cmd | python - <<'EOF'` loses the pipe -- the heredoc IS stdin
    # (the script), so sys.stdin.read() saw the program, matched nothing
    # and pruned nothing in the first live run. Hand the listing over in
    # a file instead.
    openclaw mcp show > "$STATE_DIR/mcp_show.txt" 2>/dev/null || true
    DEAD="$(MCP_SHOW="$STATE_DIR/mcp_show.txt" "$PY" - <<'PYEOF' 2>/dev/null || true
import json, os, re, shutil, sys
raw = open(os.environ["MCP_SHOW"], errors="replace").read()
m = re.search(r"\{.*\}", raw, re.S)
if not m:
    sys.exit()
try:
    servers = json.loads(m.group(0))
except Exception:
    sys.exit()
servers = servers.get("servers", servers) if isinstance(servers, dict) else {}
for name, ent in servers.items():
    if not isinstance(ent, dict):
        continue
    cmd = ent.get("command")
    if not cmd:
        continue  # remote (url) servers: nothing to check
    if os.path.isabs(cmd):
        ok = os.path.exists(cmd)
    else:
        ok = shutil.which(cmd) is not None
    args = ent.get("args") or []
    # `python -m pkg.module`: the interpreter may well exist while the
    # package it points at was renamed/removed (this machine: a `wrc-demo`
    # entry whose venv still had python but no `wrc_demo` package). Ask
    # THAT interpreter whether the module resolves -- a static path check
    # cannot see it, and this is exactly what fails on every turn.
    if ok and len(args) >= 2 and args[0] == "-m" and os.path.basename(cmd).startswith("python"):
        import subprocess

        try:
            probe = subprocess.run(
                [cmd, "-c", f"import importlib.util, sys; sys.exit(0 if importlib.util.find_spec({args[1].split('.')[0]!r}) else 3)"],
                capture_output=True, timeout=20, cwd=ent.get("cwd") if os.path.isdir(ent.get("cwd") or "") else None,
            )
            ok = probe.returncode == 0
        except Exception:
            ok = True  # cannot tell; leave it alone
    if not ok:
        print(name)
PYEOF
)"
    for dead in $DEAD; do
        [[ "$dead" == "$MCP_NAME" ]] && continue
        warn "MCP server '$dead' points at a command/module that no longer exists -- removing it (it failed on every turn)"
        run openclaw mcp unset "$dead" >/dev/null 2>&1 || true
    done
fi
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
                # Fresh machine, no local model server: the ONE interactive
                # step. OpenClaw's own onboarding picks a provider and logs
                # in (OAuth or API key); we cannot and should not guess
                # credentials. Only under --setup and only on a terminal.
                if [[ $SETUP == 1 && -t 0 && $DRY == 0 ]]; then
                    log "OpenClaw has no model yet -> its onboarding wizard (pick a provider you have access to; the robot tools are already registered)"
                    openclaw onboard --mode local || die "onboarding did not complete"
                    DEFAULT_MODEL="$(openclaw models status --json 2>/dev/null | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("defaultModel") or "")' 2>/dev/null || true)"
                    [[ -n "$DEFAULT_MODEL" ]] || die "still no default model after onboarding -- 'openclaw models status'"
                else
                    die "no local model server and OpenClaw has no default model: run 'openclaw onboard' once (pick a provider you have auth for), or start a local brain (scripts/serve_cosmos_vllm.sh), then re-run"
                fi
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
    # Warm the interpreter ONCE. In a freshly created venv the first import of
    # cascade.skills.runtime compiles bytecode for the whole stack (torch,
    # ultralytics, cv2): measured 34 s cold vs 0.1 s warm for tools/list, and
    # OpenClaw's probe allows 1.5 s -- so a fresh clone failed its own proof
    # step with "MCP tool listing timed out" although nothing was wrong.
    log "warming the MCP server's imports (first run in a fresh venv compiles bytecode; up to ~60 s)"
    (cd "$REPO/models" && "$MCP_PY" -c "import cascade.apps.mcp_server, cascade.skills.runtime" >/dev/null 2>&1) \
        || warn "import warm-up failed -- the probe below will show whether the server starts"
    # The tool probe only LISTS tools; the robot stack is built on the first
    # call. Build it once here, with the exact env the MCP server gets, so
    # "pinocchio missing" / "asset not found" / "camera failed" surface in
    # the launcher and not in the visitor's first chat message.
    log "building the robot runtime once (cameras=$CAMERAS arm=$ARM) -- proves the stack, not just the tool list"
    if [[ "$SIM" == "isaac" && -z "$ISAAC_PY" ]]; then
        warn "no Isaac python: skipping the runtime build check"
    else
        RUNTIME_CHECK="$(cd "$REPO/models" && CASCADE_CAMERAS="$CAMERAS" CASCADE_ARM="$ARM" CASCADE_DETECTOR_MODEL="$DETECTOR" \
            CASCADE_DETECT_CLASSES="$CLASSES" YOLO_OFFLINE=True ULTRALYTICS_OFFLINE=True CASCADE_STREAM=0 CASCADE_VIEW=0 CASCADE_BELIEFS=0 \
            "$PY" - <<'PYEOF' 2>&1 | grep -v "ARB_clip\|linesearch\|^Warp\|Module .* load\|^$" | tail -5
import os, sys, tempfile
from cascade.config import load_demo_config
from cascade.apps.demo import build_runtime
cams = os.environ["CASCADE_CAMERAS"].split(",")
cfg = load_demo_config(cameras=cams, arm=os.environ["CASCADE_ARM"], llm="mock")
rt, arm = build_runtime(cfg, tempfile.mkdtemp(prefix="cascade-check-"), view=False, lazy_arm=True, serve=False)
print("[launch] runtime builds:", rt.backends())
try:
    rt.camera.close()
except Exception:
    pass
PYEOF
)"
        printf '%s\n' "$RUNTIME_CHECK" | sed 's/^/        /'
        [[ "$RUNTIME_CHECK" == *"runtime builds:"* ]] || die "the robot runtime does not build with cameras=$CAMERAS arm=$ARM -- see the error above (missing extra? asset? camera?)"
    fi
    log "probing MCP tools"
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
    # Right after `gateway restart` the first agent run can fail while the
    # gateway re-warms provider auth; retry a couple of times before calling
    # it a failure, and read the envelope's `final`/`payloads` explicitly
    # rather than grepping the whole JSON for the letters OK.
    BRAIN_OK=0
    for attempt in 1 2 3; do
        ANSWER="$(openclaw agent exec 'Reply with exactly the word OK and nothing else.' --json --timeout 120 2>/dev/null \
            | "$PY" -c 'import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    print(""); sys.exit()
txt = d.get("final") or " ".join((p.get("text") or "") for p in (d.get("payloads") or []))
print((txt or "").strip()[:80])' 2>/dev/null || true)"
        if [[ "$ANSWER" == *OK* ]]; then BRAIN_OK=1; break; fi
        (( attempt < 3 )) && { log "brain attempt $attempt got '${ANSWER:-<nothing>}' -- retrying in 5 s"; sleep 5; }
    done
    if [[ $BRAIN_OK == 1 ]]; then log "brain answered."
    else warn "the brain did not answer a trivial turn after 3 attempts -- check 'openclaw models status' (auth) and the network before blaming the robot tools"; fi

    # One REAL robot turn through the chat host in sim modes (--no-robot-turn
    # skips it; never on real hardware). This is the proof a visitor cares
    # about: the brain picked the robot tool and physics confirmed the
    # effect. `toolSummary.failures` counts tool calls that returned an
    # error -- each one is a line in <run_dir>/server.log.
    if [[ $ROBOT_TURN == 1 && "$SIM" != "none" && $BRAIN_OK == 1 ]]; then
        log "one real robot turn: 'pick and place the red object' (up to ~2 min: the sim arm moves)"
        TURN_T0="$(date +%s)"
        TURN_JSON="$(openclaw agent exec 'pick and place the red object' --json --timeout 240 2>/dev/null || true)"
        # (the JSON travels in an env var: a here-doc IS python's stdin, so a
        # pipe into `python - <<EOF` is silently shadowed)
        TURN_SUMMARY="$(TURN_JSON="$TURN_JSON" TURN_T0="$TURN_T0" "$PY" - "$REPO" <<'PYEOF' 2>/dev/null || echo "no JSON envelope from agent exec"
import glob, json, os, sys
try:
    d = json.loads(os.environ.get("TURN_JSON") or "")
except Exception:
    print("no JSON envelope from agent exec"); sys.exit()
ts = d.get("toolSummary") or {}
final = (d.get("final") or "").strip().replace("\n", " ")[:120]
t0 = float(os.environ.get("TURN_T0") or 0)
# Only traces written DURING this turn count. The first live run of this
# block read the newest trace on disk -- from a finished visitor session --
# and printed "CONFIRMED by physics" over a turn whose server had crashed
# (failures=2, no trace of its own).
runs = [t for t in glob.glob(os.path.join(sys.argv[1], "runs", "mcp_*", "trace.jsonl")) if os.path.getmtime(t) >= t0 - 1]
runs.sort(key=os.path.getmtime)
verdict = "no trace written by this turn"
if runs:
    rows = [json.loads(l) for l in open(runs[-1]) if l.strip()]
    picks = [r for r in rows if r.get("skill") == "pick_and_place"]
    if picks:
        pc = (picks[-1].get("result") or {}).get("postcondition") or {}
        verdict = f"pick_and_place ok={picks[-1]['result'].get('ok')} postcondition={pc.get('status')} channel={pc.get('channel')}"
    else:
        verdict = "no pick_and_place call in the trace (the brain answered without moving the robot)"
    verdict += f" | log: {os.path.dirname(runs[-1])}/server.log"
else:
    logs = [l for l in glob.glob(os.path.join(sys.argv[1], "runs", "mcp_*", "server.log")) if os.path.getmtime(l) >= t0 - 1]
    if logs:
        verdict += f" | newest server log: {sorted(logs, key=os.path.getmtime)[-1]}"
failures = ts.get("failures") or 0
ok = failures == 0 and "postcondition=confirmed" in verdict
print(f"{'OK' if ok else 'NOT-OK'} tools={ts.get('calls')} failures={failures} | {verdict} | brain: {final}")
PYEOF
)"
        log "robot turn: ${TURN_SUMMARY#* }"
        case "$TURN_SUMMARY" in
            OK*) log "robot turn CONFIRMED by physics." ;;
            *"failures=0"*) warn "the robot turn did not end in a physics-confirmed pick -- read the log path above before demoing" ;;
            *) warn "the robot turn had FAILED tool calls (the tool server may have crashed mid-call: see the gateway log, ~/.openclaw/logs or /tmp/openclaw) -- do not demo until a rerun is clean" ;;
        esac
        # The proof moved a prop into the drop zone. Put the scene back so the
        # first visitor starts from the spawn layout, not from the aftermath.
        log "resetting the scene after the proof turn (props back on spawn, memory cleared)"
        RESET_JSON="$(openclaw agent exec 'Call the reset_scene tool once and reply with exactly its props_reset list.' --json --timeout 120 2>/dev/null || true)"
        RESET_OK="$(RESET_JSON="$RESET_JSON" "$PY" - <<'PYEOF' 2>/dev/null || echo ""
import json, os
try:
    d = json.loads(os.environ.get("RESET_JSON") or "")
    ts = d.get("toolSummary") or {}
    print("ok" if ts.get("calls") and not ts.get("failures") else "")
except Exception:
    print("")
PYEOF
)"
        if [[ "$RESET_OK" == "ok" ]]; then log "scene reset."
        else warn "scene reset did not go through -- say 'reset the scene' in the chat before the first visitor"; fi
    fi
fi

# ── 6. chat ─────────────────────────────────────────────────────────────────
# Tell the truth about the window: the proof turn's server log says whether
# the MuJoCo viewer opened, was skipped (display asleep: it retries on the
# next motion after the screen wakes), or is unavailable on this box.
VIEWER_NOTE="the MuJoCo window opens on the FIRST motion command (arm is lazy until then)"
if [[ "$SIM" == "mujoco" ]]; then
    NEWEST_LOG="$(ls -t "$REPO"/runs/mcp_*/server.log 2>/dev/null | head -1 || true)"
    if [[ -n "$NEWEST_LOG" ]]; then
        if grep -q "viewer window open" "$NEWEST_LOG" 2>/dev/null; then
            VIEWER_NOTE="the MuJoCo window is open (it follows every motion)"
        elif grep -q "viewer skipped: no ACTIVE display" "$NEWEST_LOG" 2>/dev/null; then
            VIEWER_NOTE="display was asleep/locked during setup -- the MuJoCo window opens on the first motion AFTER the screen is awake"
        elif grep -q "viewer unavailable" "$NEWEST_LOG" 2>/dev/null; then
            VIEWER_NOTE="no MuJoCo window on this machine ($(grep -o 'viewer unavailable ([^)]*)' "$NEWEST_LOG" | head -1)); use the dashboard live view"
        fi
    fi
fi
cat <<EOF
[launch] READY   sim=$SIM  arm=$ARM  cameras=$CAMERAS
         chat:      http://127.0.0.1:$GATEWAY_PORT/   (openclaw dashboard)
         headless:  openclaw agent exec "describe the scene"
         try:       "what do you see?"  "pick and place the red object"  "did it actually move?"
$( [[ "$CAMERAS" == *scene_two* ]] && echo '         memory:    "put both cubes in the drop zone, one at a time; call task_memory before each action; then tell me how many you moved and how you know"' )
         reset:     "reset the scene"  (between visitors: props back on spawn, memory cleared)
$( [[ "$SIM" == "mujoco" ]] && echo "         viewer:    $VIEWER_NOTE" )
$( [[ "$SIM" == "isaac"  ]] && echo "         isaac:     bridge :$BRIDGE_PORT, log $STATE_DIR/isaac_bridge.log" )
$( [[ "$OCCUPANCY" != "none" ]] && echo "         occupancy: :$OCC_PORT ${OCC_DESC:-(dry run)}" )
$( [[ "$GRASPGENX" != "none" ]] && echo "         graspgenx: :$GGX_PORT ($GRASPGENX)" )
         stop:      ./scripts/launch.sh --down
EOF
if [[ $OPEN_CHAT == 1 && $DRY == 0 ]]; then
    openclaw dashboard >/dev/null 2>&1 || warn "could not open the browser; use the URL above"
fi
