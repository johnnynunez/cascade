#!/usr/bin/env bash
# CASCADE front door: safe under `curl .../install.sh | bash -s -- ...`.
# --profile spark is deliberately not a laptop/GPU-autodetection fallback.
set -euo pipefail

PROFILE=spark
DIR="${CASCADE_HOME:-$HOME/cascade}"
if [[ -z "${CASCADE_HOME:-}" && -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    [[ ! -f "$LOCAL_REPO/pyproject.toml" ]] || DIR="$LOCAL_REPO"
fi
REF=main
REF_SET=0
BRAIN=""
DRY=0
CHECK=0
ACCEPT=0
PREPARE=0
NO_OPEN=0

log() { printf '[cascade-install] %s\n' "$*"; }
die() { printf '[cascade-install] ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
    printf '%s\n' 'Usage: install.sh [--profile spark|laptop|ci] [--dir PATH] [--ref REF]' \
        '  --accept-eula   consent to NVIDIA Isaac Sim / Omniverse EULA' \
        '  --dry-run       show plan without downloads or filesystem writes' \
        '  --check         read-only preflight; nonzero when prerequisites missing' \
        '  --prepare-only  install/cache + Spark shortcuts; no services or proof' \
        '  --no-open       do not open the browser' \
        '  --brain qwen|keep  default: qwen on Spark, keep on laptop'
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile|--dir|--ref|--brain)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "missing value for $1"
            case "$1" in
                --profile) PROFILE="$2" ;;
                --dir) DIR="$2" ;;
                --ref) REF="$2"; REF_SET=1 ;;
                --brain) BRAIN="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --check) CHECK=1; shift ;;
        --accept-eula) ACCEPT=1; shift ;;
        --prepare-only) PREPARE=1; shift ;;
        --no-open) NO_OPEN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown flag $1 (see --help)" ;;
    esac
done
case "$PROFILE" in spark) BRAIN="${BRAIN:-qwen}" ;; laptop|ci) BRAIN="${BRAIN:-keep}" ;; *) die "unknown --profile $PROFILE" ;; esac
case "$BRAIN" in qwen|keep) ;; *) die "unknown --brain $BRAIN" ;; esac
[[ "$PROFILE" != spark || "$BRAIN" == qwen ]] || die 'Spark delivery requires --brain qwen; use --profile laptop explicitly for other brains'
[[ "$PROFILE" == spark || "$BRAIN" == keep ]] || die "--brain qwen requires --profile spark"
[[ "$REF" =~ ^[A-Za-z0-9_][A-Za-z0-9_./-]*$ && "$REF" != *..* && "$REF" != */ && "$REF" != *. && "$REF" != *.lock && "$REF" != */.* && "$REF" != *//* ]] || die "invalid --ref $REF"
case "$DIR" in /*) ;; *) DIR="$PWD/$DIR" ;; esac
log "profile=$PROFILE brain=$BRAIN source=$REF dir=$DIR"
log "CASCADE: Python 3.12 in $DIR/.venv; source changes are preserved"
if [[ "$PROFILE" == spark ]]; then
    log "Isaac Sim 6.1.0: isaacsim[all,extscache]==6.1.0.0 in $DIR/.isaacsim (Python 3.12); explicit ISAACSIM_PATH is honored"
    log "YOLOE: CUDA aarch64 cu130 torch + torchvision, promptable/prompt-free weights + mobileclip_blt.ts"
    log "Qwen Q4 + vision projector: pinned downloads in $DIR/models/qwen3.8-27b; verified CUDA llama.cpp in $DIR/.llama.cpp, loopback :8080"
    log 'Kitchen: download and verify the 166-file kitchen-v1 release automatically.'
fi
[[ "$PROFILE" == ci ]] || log "OpenClaw 2026.9.3: rootless private install, profile cascade-demo on Spark"
log 'No driver/OS changes. GPU rehearsal remains required; installation is not physical proof.'
if [[ "$DRY" == 1 ]]; then exit 0; fi
HOST_BAD=0
if [[ "$PROFILE" == spark ]]; then
  (
    [[ "$ACCEPT" == 1 || "$CHECK" == 1 ]] || die 'Spark installation requires --accept-eula (https://docs.omniverse.nvidia.com/eula)'
    [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || die 'Spark requires Linux aarch64; use --profile laptop or ci explicitly on other hosts'
    LIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
    [[ "$LIBC" =~ ^glibc\ ([0-9]+)\.([0-9]+)$ ]] || die "cannot determine glibc version: $LIBC"
    (( BASH_REMATCH[1] > 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] >= 35) )) || die 'Isaac requires glibc >=2.35'
    GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)" || die 'NVIDIA driver is not operational; ask the administrator (no drivers are changed)'
    [[ -n "$GPU" ]] || die 'NVIDIA GPU not found'
  ) || { [[ "$CHECK" == 1 ]] || exit 2; HOST_BAD=1; }
fi
if [[ "$CHECK" == 1 ]]; then
    MISSING="$HOST_BAD"
    missing() { log "MISSING: $*"; MISSING=1; }
    [[ -f "$DIR/pyproject.toml" && -f "$DIR/scripts/install_support.py" ]] || missing "source checkout at $DIR"
    [[ -x "$DIR/.venv/bin/python" ]] || missing "$DIR/.venv/bin/python"
    if [[ "$PROFILE" == spark ]]; then
        if [[ -f "$DIR/scripts/install_isaac.sh" ]]; then
            bash "$DIR/scripts/install_isaac.sh" --dir "$DIR" --check || MISSING=1
        else
            missing "$DIR/.isaacsim/bin/python or a valid ISAACSIM_PATH (Isaac 6.1.0.0)"
        fi
        [[ -f "$DIR/scripts/serve_qwen_llamacpp.sh" ]] || missing 'local model / llama-server setup helper'
    fi
    [[ "$PROFILE" == ci || -x "$DIR/.openclaw-cli/bin/openclaw" ]] || missing 'OpenClaw 2026.9.3 private CLI'
    if [[ -f "$DIR/scripts/install_support.py" && -x "$DIR/.venv/bin/python" ]]; then
        "$DIR/.venv/bin/python" -B "$DIR/scripts/install_support.py" check --repo "$DIR" --profile "$PROFILE" --brain "$BRAIN" || MISSING=1
    fi
    [[ "$MISSING" == 0 ]] || exit 3
    log 'Installation preflight passed; runtime/GPU/physical proof is still separate.'
    exit 0
fi
command -v git >/dev/null 2>&1 || die 'git is required; install it with administrator approval, then retry'
command -v curl >/dev/null 2>&1 || die 'curl is required; install it with administrator approval, then retry'
git check-ref-format --allow-onelevel "$REF" >/dev/null 2>&1 || die "invalid --ref $REF"

retry() {
    local attempt=1 status=0
    while true; do
        if "$@"; then return 0; else status=$?; fi
        [[ "$attempt" -lt 3 ]] || return "$status"
        log "retry $attempt/3: $*" >&2
        sleep "$attempt"
        attempt=$((attempt + 1))
    done
}
# Publish a complete clone only. A failed download leaves no half-checkout.
STAGING=""
SOURCE_REF="$REF"
cleanup() { [[ -z "$STAGING" ]] || rm -rf -- "$STAGING"; }
trap cleanup EXIT
if [[ ! -e "$DIR" ]]; then
    mkdir -p -- "$(dirname "$DIR")"
    STAGING="$(mktemp -d "$(dirname "$DIR")/.cascade-clone.XXXXXX")"
    retry git clone --no-checkout --filter=blob:none https://github.com/johnnynunez/cascade.git "$STAGING/source"
    retry git -C "$STAGING/source" fetch --depth 1 origin "$REF"
    git -C "$STAGING/source" checkout --detach FETCH_HEAD
    [[ ! -e "$DIR" ]] || die "destination appeared while cloning: $DIR"
    mv -- "$STAGING/source" "$DIR"
else
    [[ -e "$DIR/.git" && -f "$DIR/pyproject.toml" ]] || die "source directory exists but is not a checkout: $DIR"
    # No implicit pull: a rerun keeps the exact source, including dirty files.
    log "reusing checkout without updating sources: $DIR"
    [[ "$REF_SET" == 1 ]] || SOURCE_REF='existing checkout'
    if [[ "$REF_SET" == 1 ]]; then
        CURRENT="$(git -C "$DIR" rev-parse HEAD)"
        REQUESTED="$(git -C "$DIR" rev-parse --verify "$REF^{commit}" 2>/dev/null || true)"
        if [[ "$CURRENT" != "$REQUESTED" ]]; then
            DIRTY="$(GIT_OPTIONAL_LOCKS=0 git -C "$DIR" status --porcelain -- . ':!.venv' ':!.isaacsim' ':!.cosmos' ':!.openclaw-cli' ':!runs' ':!models/Cosmos3-Edge-hf')"
            [[ -z "$DIRTY" ]] || die "dirty checkout: cannot change to --ref $REF; commit/stash yourself or use another --dir"
            retry git -C "$DIR" fetch --depth 1 origin "$REF"
            git -C "$DIR" checkout --detach FETCH_HEAD
        fi
    fi
fi
[[ -f "$DIR/pyproject.toml" ]] || die "not a CASCADE source checkout: $DIR"
cd "$DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    UV_INSTALLER="$(curl -fLsS --retry 3 --retry-delay 1 https://astral.sh/uv/install.sh)"
    UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh -c "$UV_INSTALLER"
fi
PY="$DIR/.venv/bin/python"
record_install() {
    local consent=""
    [[ "$ACCEPT" != 1 ]] || consent="--accept-eula"
    # $consent is empty or the single literal flag, never user input.
    "$PY" "$DIR/scripts/install_support.py" record --repo "$DIR" --profile "$PROFILE" --brain "$BRAIN" --ref "$SOURCE_REF" $consent
    if [[ "$PROFILE" == spark ]]; then
        "$PY" "$DIR/scripts/desktop.py" register --repo "$DIR"
        log "Desktop entries installed: Install CASCADE (Spark), CASCADE (Spark)"
    fi
}
if [[ ! -x "$PY" ]]; then
    [[ ! -e "$DIR/.venv" ]] || die 'incomplete .venv: move it aside explicitly; installer will not reset it'
    retry uv venv --python 3.12 "$DIR/.venv"
fi
[[ "$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == 3.12 ]] || die 'existing .venv must use Python 3.12; it was not reset'
case "$PROFILE" in
    ci) EXTRAS=dev,kinematics ;;
    laptop) EXTRAS=dev,kinematics,sim,grasping,occupancy,llm ;;
    spark) EXTRAS=kinematics,grasping,occupancy,llm,perception ;;
esac
if [[ "$PROFILE" == spark ]]; then
    # Resolve source payloads before the large runtime/package downloads.
    "$PY" "$DIR/scripts/install_support.py" robot-assets --repo "$DIR"
    retry uv pip install --python "$PY" 'torch==2.14.0+cu130' 'torchvision==0.29.0+cu130' \
        --index-url https://download.pytorch.org/whl/cu130
fi
retry uv pip install --python "$PY" -e "$DIR[$EXTRAS]"
if [[ "$PROFILE" == ci ]]; then
    record_install
    log "CI dependencies installed (no services). Python: $PY"
    exit 0
fi
if [[ "$PROFILE" == spark ]]; then
    # MobileCLIPTS uses this tokenizer; no training/mobileclip stack is needed.
    retry uv pip install --python "$PY" 'git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7'
    "$PY" -B -c 'import torch, torchvision
assert torch.__version__ == "2.14.0+cu130", torch.__version__
assert torchvision.__version__ == "0.29.0+cu130", torchvision.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.is_available(), "CUDA unavailable: repair the NVIDIA driver with administrator approval; no CPU fallback"
print("[cascade-install] Detector CUDA wheel imports checked; no model inference or physics proof performed.")'
    bash "$DIR/scripts/install_isaac.sh" --dir "$DIR" --accept-eula
    "$PY" "$DIR/scripts/install_support.py" assets --repo "$DIR"
else
    retry "$PY" "$DIR/scripts/fetch_robot_assets.py" so101
fi

# Never upgrade the user's unrelated OpenClaw installation/configuration.
OC_PREFIX="$DIR/.openclaw-cli"
export PATH="$OC_PREFIX/bin:$PATH"
OC_VERSION="$("$OC_PREFIX/bin/openclaw" --version 2>/dev/null || true)"
if [[ ! "$OC_VERSION" =~ (^|[[:space:]])2026\.9\.3($|[[:space:]]) ]]; then
    CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/cascade/installers"
    mkdir -p "$CACHE"
    if [[ ! -s "$CACHE/openclaw-cli.sh" ]]; then
        curl -fLsS --retry 3 --retry-delay 1 https://openclaw.ai/install-cli.sh -o "$CACHE/openclaw-cli.sh.part"
        mv "$CACHE/openclaw-cli.sh.part" "$CACHE/openclaw-cli.sh"
    fi
    # The rootless installer normally refreshes a loaded gateway even with
    # --no-onboard. Suppress that hook: preparation must not start services.
    OPENCLAW_INSTALL_CLI_SH_NO_RUN=1 bash -c '
        source "$1"
        refresh_gateway_service_if_loaded() { :; }
        shift
        main "$@"
    ' -- "$CACHE/openclaw-cli.sh" --prefix "$OC_PREFIX" --version 2026.9.3 --install-method npm --no-onboard
fi
OC_VERSION="$("$OC_PREFIX/bin/openclaw" --version)"
[[ "$OC_VERSION" =~ (^|[[:space:]])2026\.9\.3($|[[:space:]]) ]] || die "OpenClaw version mismatch: $OC_VERSION (expected 2026.9.3)"
if [[ "$BRAIN" == qwen ]]; then
    retry uv pip install --python "$PY" 'cmake==4.1.0' 'ninja==1.13.0'
    PY="$PY" MODEL_ROOT="$DIR" CASCADE_INSTALL_PROFILE="$PROFILE" "$DIR/scripts/serve_qwen_llamacpp.sh" --setup-only
fi
if [[ "$PREPARE" == 1 ]]; then
    record_install
    log 'PREPARED: dependencies/assets cached. No services started, no runtime or physical proof performed.'
    if [[ "$PROFILE" == spark ]]; then
        printf '[cascade-install] Next: python3 %q launch --repo %q\n' "$DIR/scripts/desktop.py" "$DIR"
    fi
    exit 0
fi
record_install
# Proof and READY belong exclusively to launch.sh, not to the installer.
set -- launch --repo "$DIR" --profile "$PROFILE" --brain "$BRAIN"
[[ "$NO_OPEN" != 1 ]] || set -- "$@" --no-open
# Replace the front-door shell so TERM/HUP reaches the supervising helper.
cleanup
STAGING=""
trap - EXIT
exec "$PY" "$DIR/scripts/install_support.py" "$@"
