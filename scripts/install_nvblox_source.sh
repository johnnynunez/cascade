#!/usr/bin/env bash
# Private CUDA 13.2.2 + Torch 2.14.1 build on Linux x86_64 or DGX Spark aarch64.
# Requires git, a C++ compiler, tar/xz, uv, and a CUDA-capable NVIDIA driver.
# Does not install drivers, use sudo, or modify Cascade/Isaac/GraspGen-X venvs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${CASCADE_NVBLOX_PREFIX:-$REPO/.nvblox}"
JOBS="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) printf 'Usage: %s [--prefix DIR] [--jobs N]\n' "$0" >&2; exit 2 ;;
    esac
done
[[ "$(uname -s)" == Linux ]] || { echo 'Linux is required' >&2; exit 2; }
case "$(uname -m)" in x86_64|aarch64) ;; *) echo 'Unsupported architecture' >&2; exit 2 ;; esac
command -v uv >/dev/null || { echo 'uv is required' >&2; exit 2; }
mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd)"
VENV="$PREFIX/venv"
SOURCE="$PREFIX/source"
export CUDA_HOME="$PREFIX/cuda-13.2.2" CUDA_VERSION=13.2.2
export CUDA_PATH="$CUDA_HOME"
[[ -x "$VENV/bin/python" ]] || uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
"$PY" "$REPO/scripts/fetch_cuda_redist.py" "$CUDA_HOME"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu132 'torch==2.14.1'
uv pip install --python "$PY" 'cmake==3.31.10' ninja setuptools wheel numpy scipy \
    transforms3d imageio nvtx pyzmq msgpack msgpack-numpy open3d pyyaml pydantic opencv-python
REV=c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f
if [[ ! -d "$SOURCE/.git" ]]; then
    git clone --branch v0.0.10 --depth 1 https://github.com/nvidia-isaac/nvblox.git "$SOURCE"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$REV" ]] || {
    echo "Unexpected nvblox source revision in $SOURCE; keeping it unchanged" >&2; exit 2;
}
PATCH="$REPO/scripts/nvblox-pytorch214.patch"
if ! git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
    git -C "$SOURCE" apply --check "$PATCH"
    git -C "$SOURCE" apply "$PATCH"
fi
ARCH="$("$PY" -c 'import torch; assert torch.cuda.is_available(); a,b=torch.cuda.get_device_capability(); print(a*10+b)')"
TORCH_PREFIX="$("$PY" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')"
# Upstream's package metadata caps torch at 2.9.1. Install only its Python
# package: the tested wrapper compatibility patch above supplies C++20.
uv pip install --python "$PY" --no-deps --no-build-isolation -e "$SOURCE/nvblox_torch"
cmake --fresh -S "$SOURCE" -B "$SOURCE/build" -G Ninja \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_HOME" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DBUILD_RENDERER=OFF \
    -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF
cmake --build "$SOURCE/build" --target py_nvblox -j "$JOBS"
"$PY" "$REPO/scripts/check_nvblox.py" --output "$PREFIX/check.json"
uv pip freeze --python "$PY" > "$PREFIX/requirements-resolved.txt"
printf '\nValidated environment: %s\nCUDA libraries: %s/lib\n' "$PY" "$CUDA_HOME"
