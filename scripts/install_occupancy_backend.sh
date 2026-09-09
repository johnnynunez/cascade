#!/usr/bin/env bash
# Installs the deps for scripts/serve_occupancy_bridge.py (the cascade
# occupancy bridge -- see src/cascade/perception/occupancy.py) into the
# target interpreter. Idempotent: only installs packages.
#
#   ./scripts/install_occupancy_backend.sh [--python /path/to/python] [--nvblox]
#
# Always:   pyzmq msgpack msgpack-numpy   (wire)   +  warp-lang scipy
#           -> the `warp` backend: dense TSDF with carving + exact EDT, CPU
#              on macOS/Linux/aarch64, CUDA on Jetson/x86 (PyPI wheels).
# --nvblox: additionally installs nvblox_torch from the nvidia-isaac/nvblox
#           GitHub release (linux x86_64 + CUDA 12/13 wheels only; needs
#           torch). Jetson: build from source per the nvblox docs. The
#           bridge's `auto` backend picks nvblox when it imports AND a CUDA
#           device is present, warp otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=""
WITH_NVBLOX=0
NVBLOX_VER="${NVBLOX_VER:-0.0.10}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PY="$2"; shift 2 ;;
        --nvblox) WITH_NVBLOX=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3 || true)"; fi
fi
[[ -n "$PY" && -x "$PY" ]] || { echo "error: no usable python; pass --python /path/to/python" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv (https://astral.sh/uv)..." >&2
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "installing occupancy bridge deps into: $PY" >&2
uv pip install --python "$PY" pyzmq msgpack msgpack-numpy "warp-lang>=1.16" scipy

if [[ "$WITH_NVBLOX" == "1" ]]; then
    os="$(uname -s)"; arch="$(uname -m)"
    if [[ "$os" != "Linux" || "$arch" != "x86_64" ]]; then
        echo "error: nvblox_torch pip wheels exist only for linux x86_64 (this is $os/$arch)." >&2
        echo "       Jetson: build nvblox_torch from source (https://nvidia-isaac.github.io/nvblox/)." >&2
        exit 2
    fi
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "error: nvblox needs a CUDA-enabled torch in $PY (install torch for your CUDA first)." >&2
        exit 2
    fi
    cu="$("$PY" -c "import torch; print('cu13' if torch.version.cuda.startswith('13') else 'cu12')")"
    ub="$(. /etc/os-release && echo "${VERSION_ID%%.*}")"
    tag="${cu}ubuntu${ub}"
    url="https://github.com/nvidia-isaac/nvblox/releases/download/v${NVBLOX_VER}/nvblox_torch-${NVBLOX_VER}+${tag}-py3-none-linux_x86_64.whl"
    echo "installing nvblox_torch ${NVBLOX_VER} (${tag}) from ${url}" >&2
    uv pip install --python "$PY" "$url"
    "$PY" -c "import nvblox_torch; print('nvblox_torch OK')"
fi

echo "" >&2
echo "done. Launch the bridge with:  PY=$PY ./scripts/serve_occupancy.sh   (auto-picks nvblox > warp > voxel)" >&2
