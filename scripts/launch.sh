#!/usr/bin/env bash
# One-click bring-up for the cascade demo: the simulator (when there is one),
# the OpenClaw 2.0 chat with cascade's robot tools registered, and the proof
# that the tools answer -- in ONE command.
#
#   ./scripts/launch.sh                     # auto: Isaac Sim if installed, else MuJoCo
#   ./run.sh                               # ONE CLICK on a fresh clone: --setup --sim auto
#   ./scripts/launch.sh --sim isaac         # Isaac Sim (bridge + editor window) + OpenClaw
#   ./scripts/launch.sh --sim isaac --engine physx --scene-config demo/scene/kitchen_config.json
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
#                  for the TCP bridge on :8611 (cold Newton setup can exceed 10 minutes)
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
#   5. proof       `openclaw mcp probe cascade` must list the robot tools; the
#                  brain answers a turn; in sim modes ONE real pick is checked
#                  against physics, the scene is reset, and the outcome JUDGE
#                  (eval.judge) scores the pick's keyframes -- fn>0 in the
#                  banner means the pictures missed physics-confirmed progress
#                  (--no-robot-turn / --no-judge skip these)
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
# Prefer the pinned, user-local CLI installed with this checkout.
[[ ! -x "$REPO/.openclaw-cli/bin/openclaw" ]] || export PATH="$REPO/.openclaw-cli/bin:$PATH"
# A managed Spark install uses its own chat profile on subsequent run/down.
if [[ -x "$REPO/.isaacsim/bin/python" && -z "${CASCADE_OPENCLAW_PROFILE+x}" ]]; then
    export CASCADE_OPENCLAW_PROFILE=cascade-demo
fi
SIM="auto"            # auto | isaac | mujoco | none
ARM=""                # default depends on SIM
CAMERAS=""            # default depends on SIM
BRAIN="auto"          # auto | keep | cosmos | cosmos-sglang | qwen
OPEN_CHAT=1
SETUP=0               # --setup: create venv + install extras + fetch assets + install OpenClaw
ROBOT_TURN=1          # --no-robot-turn: skip the real pick_and_place proof (sim modes only)
NO_JUDGE=0            # --no-judge: skip scoring the proof turn with eval.judge
CHECK=0               # --check: preflight report only
OCCUPANCY="auto"      # auto | nvblox | warp | voxel | none   (bridge backend, or skip)
GRASPGENX="auto"      # auto | stub | external | none  (auto = stub in sim, external otherwise)
OCC_PORT="${CASCADE_OCCUPANCY_PORT:-5557}"
GGX_PORT="${CASCADE_GRASPGENX_PORT:-5556}"
DRY=0
DOWN=0
ISAAC_GUI=1
ISAAC_ENGINE=""       # explicit --engine newton|physx; otherwise bridge default
SCENE_CONFIG=""       # explicit --scene-config JSON; Isaac only
SCENE_CONFIG_SHA256=""
ISAAC_WAIT_S="${ISAAC_WAIT_S:-1200}"
BRIDGE_PORT="${CASCADE_BRIDGE_PORT:-8611}"
GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
STATE_ROOT="${CASCADE_LAUNCH_STATE:-$REPO/runs/.launch}"
MCP_NAME="${CASCADE_MCP_NAME:-cascade}"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sim|--arm|--cameras|--brain|--occupancy|--graspgenx|--engine|--scene-config)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { printf '[launch] ERROR: missing value for %s\n' "$1" >&2; exit 2; } ;;
    esac
    case "$1" in
        --sim) SIM="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --cameras) CAMERAS="$2"; shift 2 ;;
        --brain) BRAIN="$2"; shift 2 ;;
        --engine) ISAAC_ENGINE="$2"; shift 2 ;;
        --scene-config) SCENE_CONFIG="$2"; shift 2 ;;
        --no-open) OPEN_CHAT=0; shift ;;
        --setup) SETUP=1; shift ;;
        --no-robot-turn) ROBOT_TURN=0; shift ;;
        --no-judge)      NO_JUDGE=1; shift ;;
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

# A report flag must never turn into an installer because --setup was added.
if [[ $CHECK == 1 && $SETUP == 1 ]]; then
    printf '[launch] ERROR: --check and --setup are mutually exclusive\n' >&2
    exit 2
fi
case "$BRAIN" in auto|keep|cosmos|cosmos-sglang|qwen) ;; *) printf 'unknown --brain %s\n' "$BRAIN" >&2; exit 2 ;; esac
case "$SIM" in auto|isaac|mujoco|none) ;; *) printf 'unknown --sim %s\n' "$SIM" >&2; exit 2 ;; esac
case "$OCCUPANCY" in auto|nvblox|warp|voxel|none) ;; *) printf 'unknown --occupancy %s\n' "$OCCUPANCY" >&2; exit 2 ;; esac
case "$GRASPGENX" in auto|stub|external|none) ;; *) printf 'unknown --graspgenx %s\n' "$GRASPGENX" >&2; exit 2 ;; esac
case "$ISAAC_ENGINE" in ""|newton|physx) ;; *) printf 'unknown --engine %s (newton|physx)\n' "$ISAAC_ENGINE" >&2; exit 2 ;; esac
if [[ -n "$ISAAC_ENGINE$SCENE_CONFIG" && "$SIM" != isaac && "$SIM" != auto ]]; then
    printf '[launch] ERROR: --engine and --scene-config apply only to Isaac Sim (--sim isaac)\n' >&2
    exit 2
fi
[[ "$ISAAC_WAIT_S" =~ ^[1-9][0-9]*$ ]] || { printf 'ISAAC_WAIT_S must be a positive integer\n' >&2; exit 2; }
if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark && $DOWN == 0 ]]; then
    [[ "$SIM" != auto ]] || SIM=isaac
    [[ "$BRAIN" != auto ]] || BRAIN=cosmos
    [[ "$SIM" == isaac && "$BRAIN" == cosmos ]] || { printf '[launch] ERROR: Spark delivery requires Isaac + Cosmos, not a substituted demo\n' >&2; exit 2; }
    [[ "$ISAAC_ENGINE" != physx ]] || { printf '[launch] ERROR: Spark delivery requires the Newton engine\n' >&2; exit 2; }
fi

log()  { printf '[launch] %s\n' "$*"; }
warn() { printf '[launch] WARNING: %s\n' "$*" >&2; }
die()  { printf '[launch] ERROR: %s\n' "$*" >&2; exit 1; }
run()  { if [[ $DRY == 1 ]]; then printf '        $ %s\n' "$*"; else "$@"; fi; }
oc() {
    if [[ -n "${CASCADE_OPENCLAW_PROFILE:-}" ]]; then
        command openclaw --profile "$CASCADE_OPENCLAW_PROFILE" "$@"
    else
        command openclaw "$@"
    fi
}
if [[ -n "${CASCADE_OPENCLAW_PROFILE:-}" ]]; then
    GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18790}"
fi

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
ownerctl() {
    "$PY" "$REPO/src/cascade/apps/process_owner.py" --repo "$REPO" \
        --state-root "$STATE_ROOT" --profile "${CASCADE_OPENCLAW_PROFILE:-}" "$@"
}
STATE_DIR="$(ownerctl state-dir)" || die "invalid launch state/profile"
export CASCADE_MCP_NAME="$MCP_NAME"

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
wait_port() {  # wait_port <port> <seconds> <what> [owned_pid] [log]
    local port=$1 secs=$2 what=$3 pid="${4:-}" logfile="${5:-}" i
    for ((i = 0; i < secs; i++)); do
        port_open "$port" && return 0
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            warn "$what process $pid exited during startup; see $logfile"
            return 1
        fi
        if (( i % 15 == 14 )); then
            log "still waiting for $what on :$port (${i}/${secs}s); log=$logfile"
            if [[ -n "$logfile" && -f "$logfile" ]]; then
                "$PY" - "$logfile" <<'PYEOF'
import pathlib, sys
with pathlib.Path(sys.argv[1]).open('rb') as stream:
    stream.seek(0, 2)
    stream.seek(max(0, stream.tell() - 4096))
    print('\n'.join(stream.read().decode(errors='replace').splitlines()[-3:]), flush=True)
PYEOF
            fi
        fi
        sleep 1
    done
    return 1
}

# ── --down: stop what a previous run started, nothing else ──────────────────
if [[ $DOWN == 1 ]]; then
    if [[ $DRY == 1 || $CHECK == 1 ]]; then ownerctl down --dry-run
    else ownerctl down; fi
    log "down."
    exit 0
fi

# ── resolve --sim auto ───────────────────────────────────────────────────────
find_isaac_python() {
    local c
    if [[ -n "${ISAACSIM_PYTHON_EXE:-}" ]]; then
        [[ -x "$ISAACSIM_PYTHON_EXE" ]] || return 1
        echo "$ISAACSIM_PYTHON_EXE"; return 0
    fi
    if [[ -n "${ISAACSIM_PATH:-}" ]]; then
        [[ -x "$ISAACSIM_PATH/python.sh" ]] || return 1
        echo "$ISAACSIM_PATH/python.sh"; return 0
    fi
    if [[ -x "$REPO/.isaacsim/bin/python" ]]; then
        echo "$REPO/.isaacsim/bin/python"; return 0
    fi
    for c in "$HOME/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release" \
             "$HOME/isaacsim" "$HOME/.local/share/ov/pkg"/isaac-sim-* /isaac-sim; do
        [[ -n "$c" && -x "$c/python.sh" ]] && { echo "$c/python.sh"; return 0; }
    done
    # pip-installed Isaac Sim into the active python?
    "$PY" -c "import importlib.metadata; importlib.metadata.version('isaacsim')" >/dev/null 2>&1 && { echo "$PY"; return 0; }
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

if [[ -n "$ISAAC_PY" ]]; then
    # find_isaac_python runs in command substitution; exporting inside it
    # would be lost. Propagate the chosen source root here, to BOTH check
    # and bridge. A wheel must not inherit somebody else's source apps.
    if [[ "$(basename "$ISAAC_PY")" == python.sh ]]; then
        export ISAACSIM_PATH="$(cd "$(dirname "$ISAAC_PY")" && pwd)"
    else
        unset ISAACSIM_PATH
    fi
fi

if [[ -n "$ISAAC_ENGINE$SCENE_CONFIG" && "$SIM" != isaac ]]; then
    printf '[launch] ERROR: --engine and --scene-config require Isaac; --sim auto selected %s\n' "$SIM" >&2
    exit 2
fi
if [[ "$SIM" == isaac && -n "${CASCADE_ISAAC_DT+x}" ]]; then
    # Reject invalid explicit timing before launch state/dependencies/services.
    "$PY" - "$CASCADE_ISAAC_DT" <<'PYEOF' || exit 2
import math, sys
try:
    value = float(sys.argv[1])
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("must be finite and in (0, 1] seconds")
except (TypeError, ValueError) as exc:
    print(f"[launch] ERROR: invalid CASCADE_ISAAC_DT: {exc}", file=sys.stderr)
    sys.exit(2)
PYEOF
fi

if [[ -n "$SCENE_CONFIG" ]]; then
    # Canonical identity is checked again against the running bridge. A changed
    # file at the same path must not let a stale scene masquerade as this one.
    SCENE_INFO="$("$PY" - "$SCENE_CONFIG" <<'PYEOF'
import hashlib, json, pathlib, sys
try:
    path = pathlib.Path(sys.argv[1]).expanduser().resolve(strict=True)
    if '\n' in str(path) or '\r' in str(path):
        raise ValueError('scene configuration path cannot contain line breaks')
    data = path.read_bytes()
    if not isinstance(json.loads(data), dict):
        raise ValueError('scene configuration must be a JSON object')
    print(path)
    print(hashlib.sha256(data).hexdigest())
except (OSError, ValueError) as exc:
    print(f'[launch] ERROR: invalid --scene-config: {exc}', file=sys.stderr)
    sys.exit(2)
PYEOF
)" || exit 2
    SCENE_CONFIG="${SCENE_INFO%$'\n'*}"
    SCENE_CONFIG_SHA256="${SCENE_INFO##*$'\n'}"
fi

case "$SIM" in
    isaac)  ARM="${ARM:-isaac}";        CAMERAS="${CAMERAS:-isaac,isaac_side}" ;;
    mujoco) ARM="${ARM:-so101_mujoco}"; CAMERAS="${CAMERAS:-mujoco_scene}" ;;
    none)   [[ -n "$ARM" && -n "$CAMERAS" ]] || die "--sim none drives REAL hardware: pass --arm <profile> --cameras <profile[,profile]> explicitly (no guessing which robot is plugged in)" ;;
    *) die "unknown --sim $SIM (auto|isaac|mujoco|none)" ;;
esac
log "plan: sim=$SIM arm=$ARM cameras=$CAMERAS brain=$BRAIN python=$PY"
[[ "$SIM" != isaac ]] || log "Isaac startup budget=${ISAAC_WAIT_S}s (cold collision preprocessing/shaders); ISAAC_WAIT_S overrides it"
[[ "$SIM" != isaac ]] || log "Isaac selection: engine=${ISAAC_ENGINE:-newton (bridge default)} scene-config=${SCENE_CONFIG:-default} sha256=${SCENE_CONFIG_SHA256:-none}"
[[ $DRY == 1 ]] && log "(dry run: commands are printed, nothing is executed)"
if [[ $DRY == 1 ]]; then
    log "would install missing extras/assets, start $SIM and sidecars, register OpenClaw, verify brain + motion + reset, then open chat"
    log "Isaac target: 6.1.0 (package 6.1.0.0); profile=${CASCADE_OPENCLAW_PROFILE:-default}; no commands executed"
    exit 0
fi
if [[ $CHECK == 0 ]]; then
    LAUNCH_OWNER="$(ownerctl init)" || die "launch owner initialization failed"
    export CASCADE_LAUNCH_OWNER="$LAUNCH_OWNER"
    # Dependency/boot failures invalidate READY too, before reaching proof.
    "$PY" - "$STATE_DIR" "$SIM" <<'PYEOF'
import json, os, pathlib, sys, time, uuid
state, sim = pathlib.Path(sys.argv[1]), sys.argv[2]
attempt = 'launch-attempt-' + uuid.uuid4().hex
evidence = state / attempt
evidence.mkdir(mode=0o700)
previous = state / 'proof.json'
if previous.is_file():
    (evidence / 'previous-proof.json').write_bytes(previous.read_bytes())
report = {'verified': False, 'sim': sim, 'started_at': time.time(), 'attempt': attempt,
          'profile': os.environ.get('CASCADE_OPENCLAW_PROFILE', ''), 'note': 'launch in progress; no current proof'}
temporary = evidence / 'initial-proof.json'
temporary.write_text(json.dumps(report) + '\n')
temporary.replace(previous)
PYEOF
fi

# ── 1. deps ─────────────────────────────────────────────────────────────────
pip_install() {  # pip_install <spec...>  -- uv when present (fast, no pip needed in the venv)
    [[ "${CASCADE_INSTALL_PROFILE:-}" != spark ]] || die "Spark dependencies are missing; repair via scripts/install.sh --profile spark --accept-eula (no unpinned installs at launch)"
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
isaac_metadata_check() {
    # python.sh honors PYTHONEXE and loader variables even for metadata-only
    # checks. Do not let an agent/app environment select or contaminate Kit.
    env -u PYTHONEXE -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV -u CONDA_PREFIX \
        -u LD_LIBRARY_PATH -u LD_PRELOAD \
        PYTHONDONTWRITEBYTECODE=1 "$ISAAC_PY" "$REPO/scripts/isaac_runtime.py" --check
}
if [[ $CHECK == 1 ]]; then
    ok()  { printf '  [ok]      %s\n' "$*"; }
    bad() { printf '  [MISSING] %s\n' "$*"; PREFLIGHT_FAIL=1; }
    PREFLIGHT_FAIL=0
    echo "[launch] preflight for sim=$SIM (python=$PY)"
    "$PY" -c "import cascade" >/dev/null 2>&1 && ok "cascade importable" || bad "cascade not importable (--setup)"
    [[ -z "$MISSING_MODS" ]] && ok "python extras [$EXTRAS]" || bad "python modules: $MISSING_MODS (--setup)"
    command -v openclaw >/dev/null 2>&1 && ok "openclaw $(oc --version 2>/dev/null | grep -oE '20[0-9]{2}\.[0-9]+\.[0-9]+' | head -1)" || bad "openclaw CLI (--setup installs it)"
    if [[ "$SIM" == "isaac" ]]; then
        if [[ -n "$ISAAC_PY" ]]; then
            if ISAAC_INFO="$(isaac_metadata_check 2>&1)"; then ok "Isaac Sim: $ISAAC_INFO"
            else bad "Isaac Sim: $ISAAC_INFO"; fi
        else
            bad "Isaac Sim 6.1.0 (run scripts/install.sh --profile spark --accept-eula)"
        fi
        [[ -f "$USD" ]] && ok "scene USD: $USD" || bad "scene USD $USD"
        command -v nvidia-smi >/dev/null 2>&1 && ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" || bad "nvidia-smi (Isaac Sim needs an NVIDIA GPU)"
        [[ -n "${DISPLAY:-}" || "$(uname -s)" == "Darwin" || $ISAAC_GUI == 0 ]] && ok "display for the editor window (or --headless)" || bad "no DISPLAY: pass --headless or run from the desktop session"
    fi
    [[ -f "$DETECTOR_PT" ]] && ok "detector weights" || bad "detector weights $DETECTOR_PT"
    if BRAIN_CHECK="$("$PY" "$REPO/scripts/demo_proof.py" --check-brain "$BRAIN" 2>&1)"; then
        ok "brain: $BRAIN_CHECK"
    else
        bad "brain: $BRAIN_CHECK"
    fi
    curl -sf -m 5 -o /dev/null https://openclaw.ai 2>/dev/null && ok "internet (openclaw.ai reachable)" || warn "no internet: fine if OpenClaw + models are already installed"
    if [[ $PREFLIGHT_FAIL == 1 ]]; then echo "[launch] preflight FAILED -- fix the [MISSING] lines (most: ./scripts/launch.sh --setup --sim $SIM)"; exit 3; fi
    echo "[launch] preflight OK -> ./scripts/launch.sh --sim $SIM"
    exit 0
fi

# ── 3. simulator ────────────────────────────────────────────────────────────
if [[ "$SIM" == "isaac" ]]; then
    isaac_metadata_check || die "Isaac Sim 6.1.0.0 installation is incomplete"
    if [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 && -f /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
        export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    if port_open "$BRIDGE_PORT"; then
        log "Isaac bridge already answering on :$BRIDGE_PORT -> reusing it"
    else
        # This array is always nonempty, including on macOS's bash 3.2.
        isaac_args=(--port "$BRIDGE_PORT" --usd "$USD")
        [[ $ISAAC_GUI != 1 ]] || isaac_args+=(--gui)
        [[ -z "$ISAAC_ENGINE" ]] || isaac_args+=(--engine "$ISAAC_ENGINE")
        [[ -z "$SCENE_CONFIG" ]] || isaac_args+=(--scene-config "$SCENE_CONFIG")
        [[ -n "$ISAAC_PY" ]] || ISAAC_PY='${ISAACSIM_PATH}/python.sh'   # dry run on a box without Isaac
        log "starting Isaac Sim bridge (engine=${ISAAC_ENGINE:-newton}, gui=$ISAAC_GUI) -> $STATE_DIR/isaac_bridge.log"
        if [[ $DRY == 1 ]]; then
            printf '        $ %q ' "$ISAAC_PY"
            printf '%q ' "$REPO/scripts/isaac_bridge.py" "${isaac_args[@]}"
            printf '&\n'
        else
            nohup "$PY" "$REPO/scripts/isaac_launch.py" --python "$ISAAC_PY" -- \
                "$REPO/scripts/isaac_bridge.py" "${isaac_args[@]}" \
                >"$STATE_DIR/isaac_bridge.log" 2>&1 &
            bridge_pid=$!
            ownerctl record --pid "$bridge_pid" --role isaac_bridge >/dev/null
            wait_port "$BRIDGE_PORT" "$ISAAC_WAIT_S" "Isaac bridge" "$bridge_pid" "$STATE_DIR/isaac_bridge.log" \
                || die "Isaac bridge never listened on :$BRIDGE_PORT after ${ISAAC_WAIT_S}s -- see $STATE_DIR/isaac_bridge.log"
            ownerctl record --pid "$bridge_pid" --role isaac_bridge >/dev/null
            log "Isaac bridge up on :$BRIDGE_PORT"
        fi
    fi
    # Probe borrowed bridges too. An open port is neither health nor proof.
    "$PY" - "$BRIDGE_PORT" "$ISAAC_ENGINE" "$SCENE_CONFIG" "$SCENE_CONFIG_SHA256" <<'PYEOF' || die "Isaac bridge on :$BRIDGE_PORT failed health/scene identity -- see $STATE_DIR/isaac_bridge.log"
import math, os, sys
from cascade.sim.bridge_client import BridgeClient
expected_dt = None
if 'CASCADE_ISAAC_DT' in os.environ:
    try:
        expected_dt = float(os.environ['CASCADE_ISAAC_DT'])
    except (TypeError, ValueError) as exc:
        raise AssertionError("invalid CASCADE_ISAAC_DT: expected finite timestep in (0, 1] seconds") from exc
    assert math.isfinite(expected_dt) and 0 < expected_dt <= 1, "invalid CASCADE_ISAAC_DT: expected finite timestep in (0, 1] seconds"
c = BridgeClient(port=int(sys.argv[1]), timeout_s=20.0)
c.connect()
try:
    pong = c.request({"op": "ping"})
    assert pong.get("ok"), f"ping answered {pong!r}"
    expected_engine, expected_scene, expected_sha = (sys.argv[2:] + ['', '', ''])[:3]
    if expected_engine:
        assert pong.get('engine') == expected_engine, f"requested Isaac engine {expected_engine!r}, received {pong.get('engine')!r}"
    if expected_scene:
        assert pong.get('scene_config') == expected_scene, f"requested scene {expected_scene!r}, received {pong.get('scene_config')!r}"
        assert pong.get('scene_config_sha256') == expected_sha, 'running Isaac scene config differs from requested bytes; restart that scene explicitly'
    if expected_dt is not None:
        actual_dt = pong.get('physics_dt_s')
        assert type(actual_dt) in (int, float) and math.isfinite(actual_dt) and 0 < actual_dt <= 1, "running Isaac bridge has no valid actual physics timestep; restart it explicitly"
        assert math.isclose(actual_dt, expected_dt, rel_tol=1e-6, abs_tol=1e-12), (
            f"requested Isaac timestep {expected_dt:.12g}s, received {actual_dt:.12g}s; restart that bridge explicitly")
    st = c.request({"op": "state"})
    assert st.get('ok') is True and isinstance(st.get('q'), list) and st['q'], f"invalid joint state: {st!r}"
    assert all(isinstance(q, (int, float)) and math.isfinite(q) for q in st['q']), "non-finite joint state"
    if os.environ.get('CASCADE_INSTALL_PROFILE') == 'spark':
        assert pong.get('engine') == 'newton', f"Spark requires Newton, received {pong.get('engine')!r}"
    print(f"[launch] Isaac bridge answers: engine={pong.get('engine')} dofs={len(pong.get('dofs') or [])} state_ok={bool(st.get('ok', True))} physics_dt_s={pong.get('physics_dt_s')}")
finally:
    c.close()
PYEOF
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
    local service_pid=$!
    ownerctl record --pid "$service_pid" --role "$name" >/dev/null
    wait_port "$port" "$wait_s" "$name" || die "$name never listened on :$port after ${wait_s}s -- see $STATE_DIR/$name.log"
    ownerctl record --pid "$service_pid" --role "$name" >/dev/null
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
if ! command -v openclaw >/dev/null 2>&1; then
    [[ $SETUP == 1 ]] || die "OpenClaw CLI missing; run the installer or launch with --setup"
    log "installing pinned OpenClaw 2026.9.3 in $REPO/.openclaw-cli"
    OC_INSTALLER="$STATE_DIR/openclaw-install-cli.sh"
    curl --fail --show-error --location --retry 3 --connect-timeout 15 \
        https://openclaw.ai/install-cli.sh -o "$OC_INSTALLER" || die "OpenClaw installer download failed"
    bash "$OC_INSTALLER" --prefix "$REPO/.openclaw-cli" --version 2026.9.3 \
        || die "OpenClaw installation failed"
    export PATH="$REPO/.openclaw-cli/bin:$PATH"
fi
OC_VER="$(oc --version 2>/dev/null | grep -oE '20[0-9]{2}\.[0-9]+\.[0-9]+' | head -1 || true)"
[[ -n "$OC_VER" ]] || die "cannot determine OpenClaw version"
"$PY" -c "import sys; y,m,_=map(int,'$OC_VER'.split('.')); sys.exit(0 if (y,m)>=(2026,8) else 1)" \
    || die "OpenClaw $OC_VER is unsupported; use the pinned installer (no automatic global migrations)"
log "OpenClaw $OC_VER (profile=${CASCADE_OPENCLAW_PROFILE:-default})"

# Use exactly the same resolver as --check, including endpoint overrides.
# An answering endpoint with the wrong model is a hard error, not permission
# to switch providers. Resolve before touching the gateway configuration.
BRAIN_INFO="$("$PY" "$REPO/scripts/demo_proof.py" --check-brain "$BRAIN")" \
    || die "brain resolution failed; configure the selected model/endpoint explicitly"
log "brain resolution: $BRAIN_INFO"

# gateway up (needed before onboarding and before the probe)
GATEWAY_OWNED=0
if ownerctl owns-gateway; then GATEWAY_OWNED=1; fi
run oc config set gateway.mode local >/dev/null
run oc config set gateway.port "$GATEWAY_PORT" --strict-json >/dev/null
if port_open "$GATEWAY_PORT"; then
    "$PY" "$REPO/scripts/demo_proof.py" --wait-gateway 45 >/dev/null \
        || die "port :$GATEWAY_PORT does not answer for this OpenClaw profile; not touching that process"
    log "gateway already healthy on :$GATEWAY_PORT"
else
    log "starting gateway"
    run oc gateway install >/dev/null 2>&1 || true
    run oc gateway start >/dev/null 2>&1 || true
    if [[ $DRY == 0 ]]; then
        wait_port "$GATEWAY_PORT" 30 "OpenClaw gateway" || die "gateway never came up on :$GATEWAY_PORT (openclaw gateway status)"
        ownerctl record-gateway >/dev/null || die "cannot identify the gateway we started"
        GATEWAY_OWNED=1
    fi
fi

# register the MCP server -- idempotent (`mcp set` replaces; `mcp add` errors on an existing name)
DETECTOR="${CASCADE_DETECTOR_MODEL:-$REPO/models/yoloe-11s-seg.pt}"
CLASSES="${CASCADE_DETECT_CLASSES:-}"
if [[ -n "${CASCADE_OPENCLAW_PROFILE:-}" ]]; then
    export CASCADE_GRASP_MEMORY_PATH="${CASCADE_GRASP_MEMORY_PATH:-$STATE_DIR/memory/grasp_memory.json}"
    export CASCADE_ENVELOPE_PATH="${CASCADE_ENVELOPE_PATH:-$STATE_DIR/memory/envelope.json}"
    export CASCADE_BELIEFS_PATH="${CASCADE_BELIEFS_PATH:-$STATE_DIR/memory/beliefs.json}"
fi
# The MCP server is what opens the MuJoCo viewer, and on macOS
# `mujoco.viewer.launch_passive` REFUSES to run under plain python ("requires
# mjpython") -- the arm then logs a warning and runs headless, i.e. no sim
# window for the audience. mjpython ships in the venv next to python.
MCP_PY="$PY"
if [[ "$SIM" == "mujoco" && "$(uname -s)" == "Darwin" && -x "$(dirname "$PY")/mjpython" ]]; then
    MCP_PY="$(dirname "$PY")/mjpython"
    log "macOS + mujoco: MCP server runs under mjpython so the viewer can open"
fi
MCP_JSON="$("$PY" - "$MCP_PY" "$REPO" "$CAMERAS" "$ARM" "$DETECTOR" "$CLASSES" "$SIM" "$STATE_DIR" "$LAUNCH_OWNER" "$ISAAC_GUI" "$OCCUPANCY" <<'PYEOF'
import json, os, sys
py, repo, cams, arm, det, classes, sim = sys.argv[1:8]
state_dir, owner = sys.argv[8:10]
env = {
    "CASCADE_CAMERAS": cams, "CASCADE_ARM": arm,
    "CASCADE_DETECTOR_MODEL": det,
    "YOLO_OFFLINE": "True", "ULTRALYTICS_OFFLINE": "True",
    "CASCADE_OPENCLAW_PROFILE": os.environ.get("CASCADE_OPENCLAW_PROFILE", ""),
}
if classes:
    env["CASCADE_DETECT_CLASSES"] = classes  # explicit operator vocabulary only
for key in ("CASCADE_GRASP_MEMORY_PATH", "CASCADE_ENVELOPE_PATH", "CASCADE_BELIEFS_PATH", "CASCADE_BELIEFS"):
    if key in os.environ:
        env[key] = os.environ[key]
if sys.argv[11] == "none":
    # --occupancy none is an explicit runtime opt-out, not just permission
    # to omit its server. Otherwise lazy-arm preflight still asks the dead
    # map for robot body poses and refuses every motion before connect().
    env["CASCADE_OCCUPANCY"] = "0"
# Sim runs open the physics viewer from the MCP server (CASCADE_VIEW=1 needs
# DISPLAY set; macOS has no DISPLAY, so give it one -- mujoco.viewer ignores
# the value, it only gates the "is there a screen" check).
if sim in ("mujoco", "isaac"):
    env["CASCADE_VIEW"] = os.environ.get("CASCADE_VIEW", "1") if sys.argv[10] == "1" else "0"
    if "DISPLAY" in os.environ:
        env["DISPLAY"] = os.environ["DISPLAY"]
    elif sys.platform == "darwin":
        env["DISPLAY"] = ":0"  # gates the macOS viewer, not an X11 connection
if sim == "mujoco":
    # the physics window itself (the arm profile defaults to view: false)
    env["CASCADE_MJ_VIEW"] = os.environ.get("CASCADE_MJ_VIEW", "1") if sys.argv[10] == "1" else "0"
# requestTimeoutMs: the OpenClaw per-CALL budget (default 60 s). pick_and_place
# PERSISTS for up to grasp.persist_seconds (120 s) by design; at 60 s the
# host sends notifications/cancelled, the server treats a cancel mid-motion
# as the operator walking away and LATCHES THE E-STOP, and every later
# motion fails "e-stop latched" until reset_stop. Measured on a real chat
# turn: the pick completed at 60.0 s, confirmed by physics, and was
# reported as cancelled. Budget = persistence + place + home, with margin.
print(json.dumps({
    "command": py, "args": ["-m", "cascade.apps.mcp_server", "--launch-owner", owner, "--launch-state-dir", state_dir],
    "cwd": os.path.join(repo, "models"),   # YOLOE resolves its text encoder relative to cwd
    "env": env, "connectionTimeoutMs": 120000, "requestTimeoutMs": 300000,
}))
PYEOF
)"
# Replace only this demo entry. Unrelated MCP servers belong to their owner;
# a dedicated Spark profile starts empty and does not need global pruning.
log "registering MCP server '$MCP_NAME' (cameras=$CAMERAS arm=$ARM)"
mkdir -p "$REPO/models"
run oc mcp set "$MCP_NAME" "$MCP_JSON"

# brain
brain_local() {  # brain_local <base_url> <model_id> <ctx>
    local base=$1 mid=$2 ctx=$3 active_cfg providers
    "$PY" "$REPO/scripts/demo_proof.py" --probe-native "$base" --model "$mid" \
        || die "requested model lacks working native tool calls; refusing to substitute another brain"
    log "onboarding local provider $base ($mid)"
    run oc onboard --non-interactive --accept-risk --mode local \
        --auth-choice custom-api-key --custom-base-url "$base" \
        --custom-model-id "$mid" --custom-compatibility openai --custom-image-input --skip-health \
        --skip-ui --suppress-gateway-token-output --skip-channels --skip-hooks \
        --skip-daemon --skip-bootstrap --skip-skills
    active_cfg="$(oc config file)" || die "cannot resolve active OpenClaw config"
    providers="$("$PY" - "$active_cfg" "$mid" "$ctx" <<'PYEOF'
import json, pathlib, sys
cfg = json.loads(pathlib.Path(sys.argv[1]).expanduser().read_text())
providers = cfg.get("models", {}).get("providers", {})
found = False
for provider in providers.values():
    for model in provider.get("models", []):
        if model.get("id") == sys.argv[2]:
            model["contextWindow"] = int(sys.argv[3])
            model["maxTokens"] = min(int(model.get("maxTokens", 4096)), 4096)
            found = True
if not found:
    raise SystemExit("onboarding did not register the requested model")
print(json.dumps(providers))
PYEOF
)" || die "local provider config is incomplete"
    run oc config set models.providers "$providers" --strict-json >/dev/null
    log "contextWindow=$ctx for $mid (selected profile only)"
}
RESOLVED_BRAIN="$(printf '%s' "$BRAIN_INFO" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["brain"])')"
if [[ "$RESOLVED_BRAIN" == keep ]]; then
    log "brain: keeping the resolved OpenClaw model"
else
    BASE="$(printf '%s' "$BRAIN_INFO" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["base_url"])')"
    MID="$(printf '%s' "$BRAIN_INFO" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model"])')"
    CTX="$(printf '%s' "$BRAIN_INFO" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["context_window"])')"
    brain_local "$BASE" "$MID" "$CTX"
fi

# Dedicated Spark profile: the brain sees this robot, not a general-purpose
# coding agent's tools or personal skills (also keeps Cosmos context bounded).
if [[ -n "${CASCADE_OPENCLAW_PROFILE:-}" ]]; then
    run oc config set agents.defaults.skills '[]' --strict-json >/dev/null
    TOOL_ALLOW="$(MCP_PREFIX="$MCP_NAME" "$PY" -c 'import json,os; print(json.dumps([os.environ["MCP_PREFIX"]+"__*"]))')"
    run oc config set tools.allow "$TOOL_ALLOW" --strict-json >/dev/null
    ACTIVE_CONFIG="$(oc config file)" || die "cannot resolve selected OpenClaw profile"
    PROFILE_WORKSPACE="$("$PY" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().parent / "workspace")' "$ACTIVE_CONFIG")"
    # Workspace attestations outlive the checkout/runs directory. Preserve the
    # profile-home workspace and bind both default and main-agent overrides.
    mkdir -p "$PROFILE_WORKSPACE"
    run oc config set agents.defaults.workspace "$PROFILE_WORKSPACE" >/dev/null
    run oc config set agents.entries.main.workspace "$PROFILE_WORKSPACE" >/dev/null
fi

# the gateway serves a stale tool list until restarted after `mcp set`
log "restarting gateway so it loads the '$MCP_NAME' tools"
run oc gateway restart >/dev/null 2>&1 || true
[[ $DRY == 0 ]] && { "$PY" "$REPO/scripts/demo_proof.py" --wait-gateway 60 >/dev/null || die "gateway did not become healthy after restart"; }
if [[ $GATEWAY_OWNED == 1 ]]; then
    ownerctl record-gateway >/dev/null || die "cannot identify our restarted gateway"
fi
run oc config validate >/dev/null

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
            "$PY" - <<'PYEOF' 2>&1 | tee "$STATE_DIR/runtime-check.log" | grep -v "ARB_clip\|linesearch\|^Warp\|Module .* load\|^$" | tail -5
import os, sys, tempfile
from cascade.config import load_demo_config
from cascade.apps.demo import build_runtime, shutdown_runtime
cams = os.environ["CASCADE_CAMERAS"].split(",")
cfg = load_demo_config(cameras=cams, arm=os.environ["CASCADE_ARM"], llm="mock")
rt, arm = build_runtime(cfg, tempfile.mkdtemp(prefix="cascade-check-"), view=False, lazy_arm=True, serve=False)
try:
    print("[launch] runtime builds:", rt.backends())
finally:
    shutdown_runtime(rt, arm)
PYEOF
)" || {
            printf '%s\n' "$RUNTIME_CHECK" >&2
            die "the robot runtime check failed; full diagnostic: $STATE_DIR/runtime-check.log"
        }
        printf '%s\n' "$RUNTIME_CHECK" | sed 's/^/        /'
        [[ "$RUNTIME_CHECK" == *"runtime builds:"* ]] || die "the robot runtime does not build with cameras=$CAMERAS arm=$ARM -- see the error above (missing extra? asset? camera?)"
    fi
    log "probing MCP tools"
    # 2.0's plain `mcp probe` prints only a COUNT; `--json` carries the names,
    # namespaced as <server>__<tool>.
    PROBE_JSON="$(oc mcp probe "$MCP_NAME" --json 2>/dev/null || true)"
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
        oc mcp doctor "$MCP_NAME" --probe 2>&1 | tail -8 >&2
        die "the MCP probe did not list the robot tools ($NTOOLS found) -- see above"
    fi
    # One gateway session owns the proof world. Never use isolated `agent
    # exec` calls for pick/reset: those may each spawn a different simulator.
    log "proving brain, manipulation and reset in one persistent chat session"
    proof_flags=""
    [[ $ROBOT_TURN == 0 ]] && proof_flags="--no-robot-turn"
    # shellcheck disable=SC2086
    PROOF_JSON="$("$PY" "$REPO/scripts/demo_proof.py" --run --repo "$REPO" \
        --state-dir "$STATE_DIR" --sim "$SIM" $proof_flags)" \
        || die "demo verification failed; NOT READY. Evidence: $STATE_DIR/cascade-proof-*/"
    printf '%s\n' "$PROOF_JSON"
    PROOF_VERIFIED="$(printf '%s' "$PROOF_JSON" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["verified"]))')"
    PROOF_TRACE="$(printf '%s' "$PROOF_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("trace", ""))')"
    BRAIN_DESC="$(printf '%s' "$PROOF_JSON" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model"])')"

    # Preserve the optional outcome metric, without making a judge outage
    # fatal or calling macOS-only caffeinate on Linux. Physics is the gate.
    JUDGE_NOTE=""
    if [[ $NO_JUDGE == 0 && $PROOF_VERIFIED == 1 ]]; then
        TURN_LOG="$(dirname "$PROOF_TRACE")"
        log "judging proof keyframes (optional metric, not the readiness gate)"
        JUDGE_OUT="$("$PY" "$REPO/scripts/judge_run.py" "$TURN_LOG" --strict 2>&1 || true)"
        JUDGE_NOTE="$(printf '%s' "$JUDGE_OUT" | "$PY" -c 'import sys; lines=[x.split("appended to summary.txt: ",1)[1] for x in sys.stdin.read().splitlines() if "appended to summary.txt: " in x]; print(lines[-1] if lines else "unavailable; physical verification remains valid")')"
        log "judge: $JUDGE_NOTE"
    fi
fi

# ── 6. chat ─────────────────────────────────────────────────────────────────
# Tell the truth about the window: the proof turn's server log says whether
# the MuJoCo viewer opened, was skipped (display asleep: it retries on the
# next motion after the screen wakes), or is unavailable on this box.
VIEWER_NOTE="the MuJoCo window opens on the FIRST motion command (arm is lazy until then)"
if [[ "$SIM" == "mujoco" ]]; then
    NEWEST_LOG=""
    if [[ -n "${PROOF_TRACE:-}" ]]; then NEWEST_LOG="$(dirname "$PROOF_TRACE")/server.log"; fi
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
LAUNCH_STATUS="STARTED (UNVERIFIED: robot proof skipped)"
[[ "${PROOF_VERIFIED:-0}" == 1 ]] && LAUNCH_STATUS="READY"
OC_HINT="openclaw"
[[ -n "${CASCADE_OPENCLAW_PROFILE:-}" ]] && OC_HINT="openclaw --profile $CASCADE_OPENCLAW_PROFILE"
cat <<EOF
[launch] $LAUNCH_STATUS   sim=$SIM  arm=$ARM  cameras=$CAMERAS
         brain:     ${BRAIN_DESC:-not verified}
         chat:      http://127.0.0.1:$GATEWAY_PORT/   ($OC_HINT dashboard)
         evidence:  $STATE_DIR/proof.json
         headless:  $OC_HINT agent --session-id cascade-demo -m "describe the scene"
         try:       "what do you see?"  "pick and place the red object"  "did it actually move?"
$( [[ "$CAMERAS" == *scene_two* ]] && echo '         memory:    "put both cubes in the drop zone, one at a time; call task_memory before each action; then tell me how many you moved and how you know"' )
         reset:     "reset the scene"  (between visitors: props back on spawn, memory cleared)
$( [[ "$SIM" == "mujoco" ]] && echo "         viewer:    $VIEWER_NOTE" )
$( [[ -n "${JUDGE_NOTE:-}" ]] && echo "         judge:     $JUDGE_NOTE" )
$( [[ "$SIM" == "isaac"  ]] && echo "         isaac:     bridge :$BRIDGE_PORT, log $STATE_DIR/isaac_bridge.log" )
$( [[ "$OCCUPANCY" != "none" ]] && echo "         occupancy: :$OCC_PORT ${OCC_DESC:-(dry run)}" )
$( [[ "$GRASPGENX" != "none" ]] && echo "         graspgenx: :$GGX_PORT ($GRASPGENX)" )
         stop:      ./scripts/launch.sh --down
EOF
if [[ $OPEN_CHAT == 1 && $DRY == 0 ]]; then
    oc dashboard >/dev/null 2>&1 || warn "could not open the browser; use the URL above"
fi
