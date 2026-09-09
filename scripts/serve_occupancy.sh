#!/usr/bin/env bash
# cascade occupancy bridge (set occupancy.enabled: true in configs/demo.yaml).
#
#   ./scripts/serve_occupancy.sh [port] [voxel_size_m] [backend]
#
# Launches serve_occupancy_bridge.py from the repo venv (or $PY). Backends:
# nvblox (P0 on NVIDIA GPUs, real TSDF+ESDF), warp (hardware-agnostic
# default: CPU on macOS/Linux/aarch64, CUDA where present), voxel (numpy
# fallback). `auto` picks the best one that can start on THIS machine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PY:-}" ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3)"; fi
fi
exec "$PY" "$ROOT/scripts/serve_occupancy_bridge.py" \
    --port "${1:-5557}" \
    --voxel-size "${2:-0.01}" \
    --backend "${3:-auto}"
