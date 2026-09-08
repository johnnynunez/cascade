#!/usr/bin/env bash
# CASCADE one-click installer -- the `curl | bash` entry point:
#
#   curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/install.sh | bash
#
# Flags survive the pipe via `bash -s --`:
#   curl -fsSL .../install.sh | bash -s -- --dir ~/robots/cascade --profile spark
#   curl -fsSL .../install.sh | bash -s -- --profile laptop --no-verify
#
# What it does, in order (each step idempotent -- re-running upgrades):
#   1. deps      git + uv (installs uv from astral.sh if missing)
#   2. clone     github.com/johnnynunez/cascade into --dir (or pull)
#   3. venv      uv venv + editable install with profile-matched extras
#   4. assets    scripts/fetch_robot_assets.py so101 piper h1 (pinned commits)
#   5. verify    the offline mock episode -- proof the install actually works
#   6. next      prints the bring-up lines for sim / real / OpenClaw / ROS2
#
# Profiles (--profile):
#   laptop   (default) dev + kinematics + sim + grasping     no GPU, no robot
#   spark    laptop + arm-feetech + arm + perception + llm   the DGX rig
#   ci       dev + kinematics only                            minimal
#
# GPU stacks (Cosmos3-Edge vLLM brain, Isaac, GraspGen-X server) are NOT
# installed here: they are per-machine and live behind scripts/bootstrap.sh,
# which this installer chains to when --with-openclaw is passed on a GPU box.
set -euo pipefail

REPO_URL="https://github.com/johnnynunez/cascade.git"
DIR="${CASCADE_HOME:-$HOME/cascade}"
PROFILE="laptop"
VERIFY=1
WITH_OPENCLAW=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) DIR="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --no-verify) VERIFY=0; shift ;;
        --with-openclaw) WITH_OPENCLAW=1; shift ;;
        *) echo "unknown flag $1 (--dir --profile laptop|spark|ci --no-verify --with-openclaw)" >&2; exit 1 ;;
    esac
done

case "$PROFILE" in
    laptop) EXTRAS="dev,kinematics,sim,grasping" ;;
    spark)  EXTRAS="dev,kinematics,sim,grasping,arm-feetech,arm,perception,llm" ;;
    ci)     EXTRAS="dev,kinematics" ;;
    *) echo "unknown --profile $PROFILE (laptop|spark|ci)" >&2; exit 1 ;;
esac

log() { printf '\033[1;32m[cascade-install]\033[0m %s\n' "$*" >&2; }

log "1/6 base tooling"
command -v git >/dev/null 2>&1 || { echo "git is required; install it first" >&2; exit 1; }
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv (astral.sh)"
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

log "2/6 source -> $DIR"
if [[ -d "$DIR/.git" ]]; then
    git -C "$DIR" pull --ff-only || log "pull failed (dirty tree?) -- continuing with the existing checkout"
else
    git clone --depth 1 "$REPO_URL" "$DIR"
fi
cd "$DIR"

log "3/6 venv + editable install [$EXTRAS]"
[[ -d .venv ]] || uv venv
uv pip install -e ".[$EXTRAS]"
PY="$DIR/.venv/bin/python"

log "4/6 robot assets (MJCF + meshes, pinned commits)"
"$PY" scripts/fetch_robot_assets.py so101 piper h1 || {
    log "asset fetch failed (offline?) -- kinematics still work from vendored URDFs; rerun later"
}

if [[ "$VERIFY" == "1" ]]; then
    log "5/6 verify: offline mock episode (no hardware, no GPU, no network)"
    "$PY" -m cascade.apps.demo --task "pick and place pink object" --no-view --no-serve --llm mock
    log "verify OK"
else
    log "5/6 verify skipped (--no-verify)"
fi

log "6/6 done."
cat >&2 <<EOF

  cascade is installed at: $DIR
  activate:                source $DIR/.venv/bin/activate

  try it (no hardware):
    python -m cascade.apps.demo --interactive --no-view --no-serve
    python -m cascade.apps.demo --arm so101_mujoco --task "pick up the cube"
    python -m cascade.apps.demo --arm piper_mock --camera mock_small \\
        --task "pick and place the red object"

  robots via ROS2 (any ros2_control robot; needs a sourced ROS2 env):
    python -m cascade.apps.demo --arm so101_ros2      # or piper / h1 / h1_2
  dual arm:
    python -m cascade.apps.demo --arms so101_left,so101_right

  OpenClaw 2.0 as the chat/orchestrator front end (GPU box):
    $DIR/scripts/bootstrap.sh            # brain + nvblox + OpenClaw wiring
  tests:
    $PY -m pytest tests/ -q
EOF

if [[ "$WITH_OPENCLAW" == "1" ]]; then
    log "chaining scripts/bootstrap.sh (--with-openclaw)"
    PY="$PY" "$DIR/scripts/bootstrap.sh"
fi
