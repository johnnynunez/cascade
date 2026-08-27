#!/usr/bin/env bash
# nvblox occupancy bridge for cascade (set occupancy.enabled: true).
#
#   ./scripts/serve_nvblox.sh [port] [voxel_size_m] [backend]
#
# Launches serve_nvblox_bridge.py -- see that file's docstring for what its
# backends are (Open3D tensor voxel accumulation, auto-picking CUDA:0 when
# this process can see an NVIDIA GPU and CPU:0 otherwise, or a zero-dep
# numpy fallback) and NOT (neither is real nvblox: no TSDF/ESDF, no
# free-space carving). Needs pyzmq + msgpack-numpy always, plus open3d for
# the default/GPU-capable backend (`uv pip install open3d`); falls back to
# the numpy backend automatically if open3d isn't installed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/johnny/Projects/demo/.demo/bin/python}"
exec "$PY" "$ROOT/scripts/serve_nvblox_bridge.py" \
    --port "${1:-5557}" \
    --voxel-size "${2:-0.02}" \
    --backend "${3:-auto}"
