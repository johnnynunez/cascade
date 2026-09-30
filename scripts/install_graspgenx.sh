#!/usr/bin/env bash
# Private CUDA inference environment; does not alter Cascade/Isaac Python.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$REPO/.graspgenx-src"
ENV_DIR="$REPO/.graspgenx"
SOURCE_REF=b9429097728cb1c430dd78b92edf17ba318aad03
CHECKPOINT_REF=7c834043c11a11417e31d6d5ea9355801e40a2c1
GRIPPER_REF=19a03c00d19aeaf052d0f6801f0041982d676e8a
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
die() { printf '[graspgenx-install] ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && "$1" == --check ) ]] || die 'usage: install_graspgenx.sh [--check]'
check_only=0
[[ "${1:-}" != --check ]] || check_only=1

checkout() {
    local url="$1" path="$2" ref="$3"
    if [[ ! -e "$path" ]]; then
        [[ "$check_only" == 0 ]] || die "missing $path; run scripts/install_graspgenx.sh"
        mkdir -p "$path"
        git -C "$path" init -q
        git -C "$path" remote add origin "$url"
        GIT_LFS_SKIP_SMUDGE=1 git -C "$path" fetch --depth 1 origin "$ref"
        GIT_LFS_SKIP_SMUDGE=1 git -C "$path" checkout --detach FETCH_HEAD
    fi
    [[ "$(git -C "$path" rev-parse HEAD)" == "$ref" ]] || die "unexpected revision in $path; preserving existing source"
    [[ -z "$(git -C "$path" status --porcelain --untracked-files=no)" ]] || die "modified source in $path; preserving it"
    if [[ "$check_only" == 0 ]]; then git -C "$path" lfs pull; fi
}
command -v git >/dev/null || die 'git is required'
git lfs version >/dev/null || die 'git-lfs is required for model checkpoints'
checkout https://github.com/NVlabs/GraspGenX.git "$SOURCE" "$SOURCE_REF"
checkout https://huggingface.co/adithyamurali/GraspGenXModel "$SOURCE/ext/graspgenx_checkpoints" "$CHECKPOINT_REF"
checkout https://huggingface.co/datasets/adithyamurali/gripper_descriptions "$SOURCE/ext/gripper_descriptions" "$GRIPPER_REF"
if [[ "$check_only" == 0 ]]; then
    command -v uv >/dev/null || die 'uv is required (installed by scripts/install.sh)'
    [[ -x "$ENV_DIR/bin/python" ]] || uv venv --python 3.12 "$ENV_DIR"
    uv pip install --python "$ENV_DIR/bin/python" \
        'torch==2.14.0+cu130' 'torchvision==0.29.0+cu130' \
        --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python "$ENV_DIR/bin/python" -r "$REPO/scripts/requirements-graspgenx.txt"
fi
[[ -x "$ENV_DIR/bin/python" ]] || die 'missing private model environment; run scripts/install_graspgenx.sh'
export GRASPGENX_CHECKPOINT_DIR="$SOURCE/ext/graspgenx_checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="$SOURCE/ext/gripper_descriptions"
PYTHONPATH="$SOURCE" "$ENV_DIR/bin/python" -B - "$check_only" "$REPO/scripts/requirements-graspgenx.txt" <<'PY'
import os
import sys
import importlib.metadata as metadata
from pathlib import Path
assert sys.version_info[:2] == (3, 12), 'GraspGen-X environment requires Python 3.12'
requirements = Path(sys.argv[2]).read_text().splitlines()
for spec in ['torch==2.14.0+cu130', 'torchvision==0.29.0+cu130', *requirements]:
    if not spec.strip() or spec.startswith('#'):
        continue
    name, expected = spec.split('==')
    assert metadata.version(name) == expected, f'{name}: expected {expected}; rerun installer'
for name in ('release/gen/epoch_736.pth', 'release/dis/epoch_1056.pth'):
    p = Path(os.environ['GRASPGENX_CHECKPOINT_DIR']) / name
    assert p.stat().st_size > 100_000_000, f'missing checkpoint or LFS pointer: {p}'
if sys.argv[1] == '1':
    print('[graspgenx-install] pinned inference environment and checkpoint files present')
    sys.exit(0)
import torch
assert torch.cuda.is_available(), 'GraspGen-X requires working CUDA; CPU fallback is disabled'
from graspgenx.serving.zmq_server import GraspGenXZMQServer
print(f'[graspgenx-install] imports OK; {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}')
print('[graspgenx-install] Inference is checked during launch; physical acceptance remains separate.')
PY
