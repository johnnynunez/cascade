#!/usr/bin/env bash
# Reuse an existing Isaac release or install pinned wheels in an isolated venv.
set -euo pipefail
DIR="${CASCADE_HOME:-$PWD}"
ACCEPT=0
DRY=0
CHECK=0
die() { printf '[install-isaac] ERROR: %s\n' "$*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir) [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'missing --dir value'; DIR="$2"; shift 2 ;;
        --accept-eula) ACCEPT=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --check) CHECK=1; shift ;;
        -h|--help) printf '%s\n' 'install_isaac.sh --dir CHECKOUT --accept-eula [--dry-run|--check]'; exit 0 ;;
        *) die "unknown flag $1" ;;
    esac
done
case "$DIR" in /*) ;; *) DIR="$PWD/$DIR" ;; esac
PY="$DIR/.isaacsim/bin/python"
REUSE=0
# Explicit paths are authoritative, including a broken path. Never repair a
# source/standalone Kit environment with pip or silently select another one.
if [[ -n "${ISAACSIM_PYTHON_EXE:-}" ]]; then
    PY="$ISAACSIM_PYTHON_EXE"
    [[ "$PY" == "$DIR/.isaacsim/bin/python" ]] || REUSE=1
elif [[ -n "${ISAACSIM_PATH:-}" ]]; then
    PY="$ISAACSIM_PATH/python.sh"; REUSE=1
elif [[ ! -x "$PY" ]]; then
    for release in "$HOME/Projects/isaac/IsaacSim/_build/linux-$(uname -m)/release" \
                   "$HOME/isaacsim" "$HOME/.local/share/ov/pkg"/isaac-sim-* /isaac-sim; do
        if [[ -x "$release/python.sh" ]]; then PY="$release/python.sh"; REUSE=1; break; fi
    done
fi
if [[ "$REUSE" == 1 ]]; then
    printf '[install-isaac] Reuse existing Isaac Sim 6.1.0 without package writes: %s\n' "$PY"
    [[ "$DRY" != 1 ]] || exit 0
    [[ "$CHECK" == 1 || "$ACCEPT" == 1 ]] || die 'requires --accept-eula; environment variables alone are not consent'
    [[ -x "$PY" ]] || die "selected Isaac Python is missing: $PY"
    if [[ "$(basename "$PY")" == python.sh ]]; then
        export ISAACSIM_PATH="$(cd "$(dirname "$PY")" && pwd)"
    else
        unset ISAACSIM_PATH
    fi
    env -u PYTHONEXE -u PYTHONHOME -u PYTHONPATH -u VIRTUAL_ENV -u CONDA_PREFIX \
        -u LD_LIBRARY_PATH -u LD_PRELOAD \
        PYTHONDONTWRITEBYTECODE=1 "$PY" "$DIR/scripts/isaac_runtime.py" --check \
        || die 'selected Isaac release is incomplete or incompatible; it was not modified'
    printf '[install-isaac] Existing release metadata checked; no pip install, Kit startup or GPU proof.\n'
    exit 0
fi
printf '[install-isaac] Isaac Sim 6.1.0, isaacsim[all,extscache]==6.1.0.0, Python 3.12 -> %s\n' "$PY"
[[ "$DRY" != 1 ]] || exit 0
verify() {
    [[ -x "$PY" ]] || return 1
    # Metadata only: importing isaacsim may prompt for the EULA or initialize Kit.
    "$PY" -B -c 'import sys, importlib.metadata as m, ctypes
try:
    ctypes.CDLL("libgomp.so.1")
except OSError as exc:
    raise SystemExit(f"libgomp.so.1 is required on Spark; ask the administrator to install libgomp1 (no OS packages changed): {exc}")
assert sys.version_info[:2] == (3, 12), "Isaac requires CPython 3.12"
d = m.distribution("isaacsim")
assert d.version == "6.1.0.0", d.version
assert any(str(p).endswith("isaacsim/apps/isaacsim.exp.full.newton.kit") and d.locate_file(p).is_file() for p in d.files or []), "Newton experience missing from installed wheel"
for name in ("isaacsim-app", "isaacsim-core", "isaacsim-extscache-kit", "isaacsim-extscache-kit-sdk", "isaacsim-extscache-physics"):
    assert m.version(name) == "6.1.0.0", name'
}
if [[ "$CHECK" == 1 ]]; then
    verify || { printf '[install-isaac] MISSING: exact Isaac wheel / Newton experience in %s\n' "$PY" >&2; exit 3; }
    exit 0
fi
[[ "$ACCEPT" == 1 ]] || die 'requires --accept-eula; environment variables alone are not consent'
[[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || die 'Spark Isaac install requires Linux aarch64'
LIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
[[ "$LIBC" =~ ^glibc\ ([0-9]+)\.([0-9]+)$ ]] || die 'cannot determine glibc'
(( BASH_REMATCH[1] > 2 || (BASH_REMATCH[1] == 2 && BASH_REMATCH[2] >= 35) )) || die 'requires glibc >=2.35'
GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)" || die 'NVIDIA driver is not operational'
[[ -n "$GPU" ]] || die 'NVIDIA GPU not found'
command -v uv >/dev/null 2>&1 || die 'uv is missing; use scripts/install.sh'
export OMNI_KIT_ACCEPT_EULA=YES
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}" UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
retry() {
    local attempt=1 status=0
    while true; do
        if "$@"; then return 0; else status=$?; fi
        [[ "$attempt" -lt 3 ]] || return "$status"
        printf '[install-isaac] retry %s/3\n' "$attempt" >&2
        sleep "$attempt"; attempt=$((attempt + 1))
    done
}
if [[ ! -x "$PY" ]]; then
    [[ ! -e "$DIR/.isaacsim" ]] || die 'incomplete .isaacsim; move it aside explicitly (never reset automatically)'
    retry uv venv --python 3.12 "$DIR/.isaacsim"
fi
[[ "$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == 3.12 ]] || die 'existing .isaacsim is not Python 3.12; refusing to replace it'
# The latest Isaac6.1 core pins torch2.11; select its explicit CUDA13 wheel.
# This environment is intentionally independent of app and Cosmos torch.
retry uv pip install --python "$PY" 'torch==2.11.0+cu130' --index-url https://download.pytorch.org/whl/cu130
# Both indexes are required: e.g. mujoco-usd-converter==0.5.0 is on PyPI,
# while NVIDIA has a different version. uv's first-index strategy cannot
# resolve that exact dependency; the Isaac family remains pinned to 6.1.0.0.
retry uv pip install --python "$PY" 'isaacsim[all,extscache]==6.1.0.0' \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
verify
printf '[install-isaac] Installed metadata and Newton experience checked; no Kit/GPU execution performed.\n'
