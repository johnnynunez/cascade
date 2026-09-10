#!/usr/bin/env bash
# Newton screw demonstration: isolated packages, never CASCADE's .venv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' 'uv is required to create the isolated Newton demo environment.' >&2
    exit 2
fi
exec uv run --no-project --isolated --python 3.12 \
    --with 'newton[sim]==1.5.1' \
    --with 'warp-lang==1.17.0' \
    --with 'mujoco==3.11.0' \
    --with 'mujoco-warp==3.11.0' \
    --with 'numpy==2.5.3' \
    --with 'scipy==1.18.1' \
    --with 'trimesh==4.12.2' \
    --with 'viser==1.1.0' \
    python "$REPO/scripts/demo_newton_screw.py" "$@"
