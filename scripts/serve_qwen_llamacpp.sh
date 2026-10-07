#!/usr/bin/env bash
# Install and serve pinned Qwen Q4 with a project-owned CUDA llama.cpp build.
# --setup-only downloads/prepares; --check verifies without writes or downloads.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$REPO}"
MODEL_DIR="$MODEL_ROOT/models/qwen3.8-27b"
# Desktop/reinstall entry points recover the explicit local paths saved during
# installation. Environment overrides always win; no personal path is assumed.
recorded_setting() {
    "$PY" -B - "$REPO" "$1" <<'PYEOF'
import json, pathlib, sys
repo = pathlib.Path(sys.argv[1]).resolve()
try:
    record = json.loads((repo / "runs/.install/install.json").read_text())
    value = record.get("model_environment", {}).get(sys.argv[2], "") if record.get("repo") == str(repo) else ""
    print(value if isinstance(value, str) else "")
except (OSError, ValueError, AttributeError):
    print("")
PYEOF
}
CASCADE_QWEN_MODEL="${CASCADE_QWEN_MODEL-$(recorded_setting CASCADE_QWEN_MODEL)}"
CASCADE_QWEN_MMPROJ="${CASCADE_QWEN_MMPROJ-$(recorded_setting CASCADE_QWEN_MMPROJ)}"
LLAMA_SERVER="${LLAMA_SERVER-$(recorded_setting LLAMA_SERVER)}"
LLAMA_DIR="${LLAMA_DIR-$(recorded_setting LLAMA_DIR)}"
MODEL_FILE="${CASCADE_QWEN_MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MMPROJ_FILE="${CASCADE_QWEN_MMPROJ:-$MODEL_DIR/mmproj-BF16.gguf}"
MODEL_MANIFEST="$REPO/deploy/brev/profiles/qwen3.8-27b-q4.json"
LLAMA_REF=4695f001fece1660d8bb1b3748f50726ddcc100b
LLAMA_DIR="${LLAMA_DIR:-$REPO/.llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"
# CUDA architecture of the GPU that runs the brain. The DGX Spark's GB10 is
# sm_121; the x86 test rig's RTX PRO 6000 Blackwell is sm_120. A value hardcoded
# for one host makes the owned build (and its receipt check) wrong on the other,
# so the architecture is read from the visible GPU unless CASCADE_LLAMA_CUDA_ARCH
# names it explicitly. Only the owned source build and its receipt depend on it.
cuda_arch() {
    if [[ -n "${CASCADE_LLAMA_CUDA_ARCH:-}" ]]; then
        [[ "$CASCADE_LLAMA_CUDA_ARCH" =~ ^[0-9]{2,3}$ ]] || {
            printf 'CASCADE_LLAMA_CUDA_ARCH must be a CUDA architecture number such as 120 or 121.\n' >&2; return 1; }
        printf '%s' "$CASCADE_LLAMA_CUDA_ARCH"
        return 0
    fi
    local selector=() cap
    [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] && selector=(--id="${CUDA_VISIBLE_DEVICES%%,*}")
    cap="$(nvidia-smi "${selector[@]}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
    [[ "$cap" =~ ^[0-9]{2,3}$ ]] || {
        printf 'Cannot read the CUDA compute capability from nvidia-smi; set CASCADE_LLAMA_CUDA_ARCH (120 for RTX PRO 6000 Blackwell, 121 for the GB10 Spark).\n' >&2
        return 1; }
    printf '%s' "$cap"
}
MODE=serve
case "${1:-}" in
    --check) MODE=check; shift ;;
    --setup-only) MODE=setup; shift ;;
    "") ;;
    *) printf 'Usage: serve_qwen_llamacpp.sh [--setup-only|--check]\n' >&2; exit 2 ;;
esac
[[ $# == 0 ]] || { printf 'Unexpected arguments\n' >&2; exit 2; }

check_assets() {
    if [[ -z "$CASCADE_QWEN_MODEL" && -z "$CASCADE_QWEN_MMPROJ" ]]; then
        "$PY" -B "$REPO/deploy/brev/fetch_models.py" --root "$MODEL_ROOT" --manifest "$MODEL_MANIFEST" --check
        return
    fi
    "$PY" -B - "$MODEL_FILE" "$MMPROJ_FILE" <<'PYEOF'
import pathlib, sys
for value, key in zip(sys.argv[1:], ("CASCADE_QWEN_MODEL", "CASCADE_QWEN_MMPROJ")):
    if not value:
        continue
    path = pathlib.Path(value)
    try:
        with path.open("rb") as stream:
            valid = stream.read(4) == b"GGUF" and path.stat().st_size >= 24
    except OSError:
        valid = False
    if not valid:
        raise SystemExit(f"Missing or invalid explicit local GGUF: {path}. Correct {key}, or unset both model overrides and rerun --setup-only for pinned downloads.")
PYEOF
}

check_source() {
    [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" == "$LLAMA_REF" && -z "$(git -C "$LLAMA_DIR" status --porcelain --untracked-files=no)" ]] || {
        printf 'Isolated llama.cpp source differs from its pinned revision; preserve it and repair explicitly.\n' >&2; return 1;
    }
}

runtime_receipt() {
    local arch
    arch="$(cuda_arch)" || return 1
    "$PY" -B - "$LLAMA_DIR" "$LLAMA_REF" "$1" "$arch" <<'PYEOF'
import hashlib, json, pathlib, platform, sys
root = pathlib.Path(sys.argv[1])
receipt = root / "build/.cascade-runtime.json"
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
files = sorted({root / "build/bin/llama-server", *(p.resolve() for p in (root / "build/bin").glob("*.so*"))})
data = {"source_revision": sys.argv[2], "architecture": platform.machine(),
        "cuda_architecture": sys.argv[4], "files": {str(p.relative_to(root)): digest(p) for p in files}}
if sys.argv[3] == "record":
    receipt.write_text(json.dumps(data, indent=2) + "\n")
else:
    try:
        saved = json.loads(receipt.read_text())
    except (OSError, ValueError):
        raise SystemExit("Missing llama.cpp build receipt; rerun --setup-only to build the pinned runtime.")
    if saved.get("cuda_architecture") != data["cuda_architecture"]:
        raise SystemExit(f"llama.cpp runtime was built for CUDA architecture {saved.get('cuda_architecture')!r} but this GPU "
                         f"needs {data['cuda_architecture']!r}; rebuild with --setup-only on this host or set CASCADE_LLAMA_CUDA_ARCH.")
    if saved != data:
        raise SystemExit("llama.cpp runtime differs from its pinned build receipt; preserve it and repair explicitly.")
PYEOF
}

check_server() {
    [[ -x "$LLAMA_SERVER" ]] || { printf 'Missing llama-server; run scripts/serve_qwen_llamacpp.sh --setup-only.\n' >&2; return 1; }
    if [[ "$LLAMA_DIR" == "$REPO/.llama.cpp" && "$LLAMA_SERVER" == "$LLAMA_DIR/build/bin/llama-server" ]]; then
        check_source
        runtime_receipt check
    fi
    local help
    help="$("$LLAMA_SERVER" --help 2>&1)" || return 1
    for flag in --alias --chat-template-kwargs --reasoning; do
        [[ "$help" == *"$flag"* ]] || { printf 'llama-server lacks %s; select a compatible LLAMA_SERVER.\n' "$flag" >&2; return 1; }
    done
    [[ -z "$MMPROJ_FILE" || "$help" == *--mmproj* ]] || { printf 'llama-server lacks --mmproj; select a compatible LLAMA_SERVER.\n' >&2; return 1; }
}

if [[ "$MODE" == setup && -z "$CASCADE_QWEN_MODEL" && -z "$CASCADE_QWEN_MMPROJ" ]]; then
    set -- --root "$MODEL_ROOT" --manifest "$MODEL_MANIFEST"
    [[ -z "${CASCADE_MODEL_MIRROR_URL:-}" ]] || set -- "$@" --model-mirror-url "$CASCADE_MODEL_MIRROR_URL"
    "$PY" -B "$REPO/deploy/brev/fetch_models.py" "$@"
fi
check_assets
if [[ "$MODE" == check ]]; then
    check_server
    printf 'Qwen Q4, vision projector and llama-server are prepared; no runtime proof performed.\n'
    exit 0
fi

if [[ ! -x "$LLAMA_SERVER" || ( "$LLAMA_DIR" == "$REPO/.llama.cpp" && "$LLAMA_SERVER" == "$LLAMA_DIR/build/bin/llama-server" && ! -f "$LLAMA_DIR/build/.cascade-runtime.json" ) ]]; then
    # A fresh source install stays in this checkout. Never update a host build.
    [[ "$LLAMA_SERVER" == "$LLAMA_DIR/build/bin/llama-server" && "$LLAMA_DIR" == "$REPO/.llama.cpp" ]] || {
        printf 'llama-server is missing in an existing/custom path; preserve it and select LLAMA_DIR or LLAMA_SERVER explicitly.\n' >&2; exit 1;
    }
    export PATH="$REPO/.venv/bin:/usr/local/cuda/bin:$PATH"
    command -v cmake >/dev/null && command -v ninja >/dev/null && command -v nvcc >/dev/null && command -v c++ >/dev/null || {
        printf 'Building llama.cpp requires the installed CUDA toolkit (nvcc), a C++ compiler, and installer-provided cmake/ninja.\n' >&2; exit 1;
    }
    if [[ ! -e "$LLAMA_DIR" ]]; then
        git init "$LLAMA_DIR"
        git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp
        git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_REF"
        git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
    fi
    check_source
    arch="$(cuda_arch)" || exit 1
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -G Ninja -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$arch" -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF
    cmake --build "$LLAMA_DIR/build" --config Release -j"${JOBS:-4}" --target llama-server
    runtime_receipt record
fi
check_server
[[ "$MODE" != setup ]] || exit 0

# The default projector is installed with the Q4 model for camera inspection.
set --
if [[ -n "$MMPROJ_FILE" ]]; then set -- --mmproj "$MMPROJ_FILE"; fi
exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" "$@" \
    --alias Qwen/Qwen3.8-27B --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" --n-gpu-layers 999 --flash-attn on --parallel 1 --jinja \
    --reasoning off --chat-template-kwargs '{"enable_thinking":false}'
