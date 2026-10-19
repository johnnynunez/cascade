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
LLAMA_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab   # llama.cpp v0.7.0; KEEP IN SYNC with serve_nemotron_bolt_llamacpp.sh
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
    # selector may stay empty: bash 3.2 + set -u needs the ${A[@]+"${A[@]}"} form.
    local selector=() cap
    [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] && selector=(--id="${CUDA_VISIBLE_DEVICES%%,*}")
    cap="$(nvidia-smi ${selector[@]+"${selector[@]}"} --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
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
die() { printf 'serve_qwen_llamacpp: %s\n' "$1" >&2; exit "${2:-1}"; }
# No inherited variable configures the server. llama-server v0.7.0 also takes
# every flag from LLAMA_ARG_<FLAG> / LLAMA_ARG_NO_<FLAG> (common/arg.cpp; a flag
# on argv wins), accepts LLAMA_API_KEY beside the key file, and with
# AIP_MODE=PREDICTION serves on AIP_HTTP_PORT (argv cannot override it) with a
# route that reaches every endpoint (tools/server/server-http.cpp). Inherited,
# LLAMA_ARG_SLOT_SAVE_PATH opens slot save/restore/erase and LLAMA_ARG_ENDPOINT_PROPS
# POST /props. LLAMA_SERVER_SLOTS_DEBUG puts every slot's prompt text in GET /slots
# (no warm start here needs it). Everything this server needs is a flag below and
# the script sets none of these itself, so all of them go. A bash without
# `compgen` refuses rather than serve unscrubbed. KEEP IN SYNC with
# serve_nemotron_bolt_llamacpp.sh and deploy/runtime/brain.py LLAMA_SERVER_ENV
# (tests/test_llama_server_inputs.py pins all three).
names="$(compgen -e)" || die "cannot list the environment (compgen failed); refusing to start llama-server unscrubbed"
for name in $names; do
    case "$name" in LLAMA_ARG_*|LLAMA_API_KEY|AIP_*|LLAMA_SERVER_SLOTS_DEBUG) unset "$name" ;; esac
done
# Not a variable, the same effect: before env and argv, llama-server v0.7.0
# applies /etc/llama.cpp/config.ini, then ${XDG_CONFIG_HOME:-$HOME/.config}/
# llama.cpp/config.ini (common/arg.cpp common_params_apply_system_config). Each
# takes ANY server option, no allowlist (common/preset.cpp): `tools = all` serves
# exec_shell_command at /tools, `api-prefix = /v1` moves every route (/slots too)
# inside the OpenShell sandbox's /v1/**. The user file follows the caller's
# XDG_CONFIG_HOME or HOME, so the server gets an empty private config home under
# the state directory instead; the system file cannot be turned off, so it
# refuses the start. CASCADE_LLAMA_CONFIG_INI=1 serves with both, with a warning.
# KEEP IN SYNC with serve_nemotron_bolt_llamacpp.sh.
LLAMA_SYSTEM_CONFIG=/etc/llama.cpp/config.ini
LLAMA_CONFIG_INI="${CASCADE_LLAMA_CONFIG_INI:-0}"
case "$LLAMA_CONFIG_INI" in 0|1) ;; *) die "CASCADE_LLAMA_CONFIG_INI must be 0|1 (got '$LLAMA_CONFIG_INI')" 2 ;; esac
check_config_files() {
    local user="${XDG_CONFIG_HOME:-${HOME:-}/.config}/llama.cpp/config.ini"
    if [[ "$LLAMA_CONFIG_INI" == 1 ]]; then
        printf 'serve_qwen_llamacpp: WARNING: CASCADE_LLAMA_CONFIG_INI=1: llama-server applies %s and %s, if present, before argv (any option: tools, api-prefix, endpoints, keys)\n' \
            "$LLAMA_SYSTEM_CONFIG" "$user" >&2
        return 0
    fi
    [[ ! -e "$LLAMA_SYSTEM_CONFIG" && ! -L "$LLAMA_SYSTEM_CONFIG" ]] \
        || die "$LLAMA_SYSTEM_CONFIG exists: llama-server applies it to every start before argv, any option (tools = all, api-prefix = /v1). Remove it, or set CASCADE_LLAMA_CONFIG_INI=1 to serve with it"
}
# A directory of this user (mkdir -p fails on anything else that exists there),
# never a link elsewhere, mode 0700 even when it existed before.
private_dir() {
    mkdir -p -m 700 "$1" && [[ ! -L "$1" ]] \
        && [[ "$(stat -c %u "$1" 2>/dev/null || stat -f %u "$1")" == "$(id -u)" ]] && chmod 700 "$1"
}
# llama-server's config home: private and empty, so it holds no llama.cpp/config.ini.
private_config_home() {
    local entries
    private_dir "$1" 2>/dev/null && entries="$(ls -A "$1")" \
        || die "cannot make $1 a private directory (llama-server's config home)"
    [[ -z "$entries" ]] || die "$1 must stay empty (llama-server would read llama.cpp/config.ini there); move its contents away"
}

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
    check_config_files
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
check_config_files
config_home=""
if [[ "$LLAMA_CONFIG_INI" == 0 ]]; then
    config_home="${XDG_STATE_HOME:-${HOME:?set HOME or XDG_STATE_HOME}/.local/state}/cascade/llama-server-config"
    private_config_home "$config_home"
fi

# The default projector is installed with the Q4 model for camera inspection.
set --
if [[ -n "$MMPROJ_FILE" ]]; then set -- --mmproj "$MMPROJ_FILE"; fi
# Authenticated brain (NemoClaw / OpenShell routes: PORT=8081 ALIAS=qwen3.8-27b
# API_KEY_FILE=...). llama-server reads the key from the file itself, so argv
# (world-readable in /proc) carries a path, never the key; the file must be
# private and non-empty. Those routes also need a bare ALIAS: OpenClaw splits
# provider/model on the first '/'.
if [[ -n "${API_KEY_FILE:-}" ]]; then
    key_file="${API_KEY_FILE/#\~/$HOME}"
    [[ -f "$key_file" ]] || { echo "API_KEY_FILE $key_file is missing" >&2; exit 1; }
    key_mode="$(stat -c %a "$key_file" 2>/dev/null || stat -f %Lp "$key_file")"
    (( (8#$key_mode & 8#077) == 0 )) || { echo "API_KEY_FILE $key_file is mode $key_mode; chmod 600 it first" >&2; exit 1; }
    [[ -n "$(tr -d '[:space:]' < "$key_file")" ]] || { echo "API_KEY_FILE $key_file is empty" >&2; exit 1; }
    set -- "$@" --api-key-file "$key_file"
fi
[[ -z "$config_home" ]] || export XDG_CONFIG_HOME="$config_home"
exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" "$@" \
    --alias "${ALIAS:-Qwen/Qwen3.8-27B}" --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" --n-gpu-layers 999 --flash-attn on --parallel 1 --jinja \
    --reasoning off --chat-template-kwargs '{"enable_thinking":false}'
