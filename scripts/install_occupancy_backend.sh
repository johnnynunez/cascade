#!/usr/bin/env bash
# Installs the Python deps for scripts/serve_nvblox_bridge.py (the wrc_demo
# occupancy bridge -- see src/wrc_demo/perception/occupancy.py) into the
# target interpreter. Safe to re-run (idempotent: only installs packages,
# never touches repo files, config, or running processes).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/johnnynunez/wrc_demo/main/scripts/install_occupancy_backend.sh | bash
#
# Or locally:
#   ./scripts/install_occupancy_backend.sh [--python /path/to/python]
#
# What it installs:
#   pyzmq, msgpack, msgpack-numpy   -- wire client/bridge, always needed
#   open3d                          -- the default bridge backend: same code
#                                      path on CPU and NVIDIA GPU (CUDA:0
#                                      auto-detected at runtime by
#                                      serve_nvblox_bridge.py, no separate
#                                      install for the GPU case)
#
# What it does NOT do: install nvblox itself. Real nvblox (NVIDIA's
# TSDF/ESDF library) ships as a from-source CUDA build with no pip wheel,
# and running it needs a CUDA GPU (there is no CPU-mode nvblox to fall back
# to). The Open3D backend installed here already gets the GPU-when-available
# behavior for the occupancy/clearance-checking use case this project needs
# (see occupancy.py's docstring); if you specifically need real nvblox's
# TSDF/ESDF mesh reconstruction, build github.com/NVlabs/nvblox_torch by
# hand on a CUDA machine -- that is a separate, much heavier undertaking
# this script deliberately does not attempt silently.
set -euo pipefail

PY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PY="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$PY" ]]; then
    PY="${PY_OVERRIDE:-/home/johnny/Projects/demo/.demo/bin/python}"
    [[ -x "$PY" ]] || PY="$(command -v python3 || true)"
fi
if [[ -z "$PY" || ! -x "$(command -v "$PY" 2>/dev/null || echo "$PY")" ]]; then
    echo "error: no usable Python interpreter found. Pass one explicitly:" >&2
    echo "  ./scripts/install_occupancy_backend.sh --python /path/to/python" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv (https://astral.sh/uv)..." >&2
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "installing occupancy bridge deps into: $PY" >&2
uv pip install --python "$PY" pyzmq msgpack msgpack-numpy open3d

echo "" >&2
echo "done. Launch the bridge with:" >&2
echo "  PY=$PY ./scripts/serve_nvblox.sh" >&2
echo "It auto-picks CUDA:0 if this machine has a visible NVIDIA GPU, CPU:0 otherwise -- no flag needed." >&2
