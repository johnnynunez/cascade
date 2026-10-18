#!/usr/bin/env bash
# Serve the cascade brain: NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB (IQ2 QAT,
# Mamba-2/Transformer hybrid LatentMoE VLM) with vLLM 0.32.0 from a pinned uv
# venv, the way serve_qwen_vllm.sh serves Qwen (no container).
# Measured on the DGX Spark (GB10), docs/evidence/brain-nemotron-bolt-2026-10-18:
# the agent and perception both run on it at GPU_UTIL 0.45.
#
# Same contract as serve_qwen_vllm.sh, so every route (OpenShell, NemoClaw,
# stdio OpenClaw) is engine-agnostic: OpenAI /v1, a BARE served id
# `nemotron-3.5-bolt` (OpenClaw splits provider/model on '/'), 64k context,
# thinking off, tool calls through vLLM's qwen3_coder parser.
#
#   scripts/serve_nemotron_bolt_vllm.sh                   # 127.0.0.1:8080, no key
#   PORT=8081 API_KEY_FILE=~/.cascade/nemoclaw/llama.key scripts/serve_nemotron_bolt_vllm.sh
#                                                         # authenticated brain (OpenShell / NemoClaw)
#   scripts/serve_nemotron_bolt_vllm.sh --setup-only      # venv + kernels + pinned weights + parser hash
#   scripts/serve_nemotron_bolt_vllm.sh --dry-run         # print the vllm command; run nothing
#
# Knobs: HOST (127.0.0.1), PORT (8080), API_KEY_FILE, GPU_UTIL (0.45), CTX (65536),
# MAX_SEQS (4), SPEC=none|dflash (none), NSPEC (7), SERVED_NAME (nemotron-3.5-bolt),
# PREFIX_MATCH (64; 0 = vLLM's own choice), MEM_FLOOR_GB (3), VENV
# (~/.venvs/vllm-0.32.0, shared with serve_qwen_vllm.sh), CACHE_DIR (kernel
# caches, ~/.cache/cascade/nemotron-bolt), MAX_JOBS (4), EXTRA_ARGS.
#
# Pinned: vLLM 0.32.0 with its b12x extra (b12x 1.5.1) and the FlashInfer
# kernels `vllm download-kernels` installs, both checkpoints by revision, and
# the model card's ultra_v3 reasoning parser plugin by sha256. Weights load
# offline from the HF cache once fetched.
#
# Security: the key is read from a 0600 file into VLLM_API_KEY, never argv.
# A bind beyond loopback without a key is refused.
#
# Thinking: off by default (--default-chat-template-kwargs), but the card's
# ultra_v3 parser reads ONLY the request's chat_template_kwargs: a
# NON-streaming request without {"enable_thinking": false} gets content null
# and its tool calls unparsed in reasoning_content. Every cascade client of
# this brain sends the kwarg (perception, demo_proof, OpenClaw provider params).
#
# Prefix caching (the agent's 13-18k-token prompts): the attention block is a
# multiple of PREFIX_MATCH (vLLM rounds it up to hold a Mamba page) and hits
# land every PREFIX_MATCH tokens up to the previous prompt's tail, where vLLM
# keeps a Mamba prefill checkpoint. The LiLiCorr DFlash drafter (SPEC=dflash)
# adds a sliding-window KV group, which vLLM can only hit on whole blocks
# (~4.5k tokens): measured beside Isaac Newton, a follow-up agent call waited
# 6.5-16 s with the drafter and 1-3 s without it, so SPEC=none is the default.
#
# Spark notes: eager safetensors load (mmap reads crawl on GB10); MemAvailable
# is watched during startup and vLLM is killed below MEM_FLOOR_GB so the host
# never freezes; run it under `systemd-run --user -p MemoryMax=infinity`.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
GPU_UTIL="${GPU_UTIL:-0.45}"
CTX="${CTX:-65536}"
MAX_SEQS="${MAX_SEQS:-4}"
SPEC="${SPEC:-none}"
NSPEC="${NSPEC:-7}"
SERVED_NAME="${SERVED_NAME:-nemotron-3.5-bolt}"
PREFIX_MATCH="${PREFIX_MATCH:-64}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-3}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/cascade/nemotron-bolt}"

VLLM_VERSION="0.32.0"   # KEEP IN SYNC with serve_qwen_vllm.sh: one venv serves both brains
VENV="${VENV:-$HOME/.venvs/vllm-$VLLM_VERSION}"
TARGET_REPO="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
TARGET_REV="c6cc2e8ffa353403213d6b9b18aaf8eb684d705a"
DRAFT_REPO="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
DRAFT_REV="abd478065678ee83a93821c4b3f512da24e4cecb"
PARSER="ultra_v3_reasoning_parser.py"
PARSER_SHA256="14e03de50a523c1b6b4542c22d38ec11cadc515a95752e5a4c8246967f620857"

die() { echo "serve_nemotron_bolt_vllm: $1" >&2; exit "${2:-1}"; }
log() { echo "[serve_nemotron_bolt] $*" >&2; }

MODE=serve
for arg in "$@"; do
    case "$arg" in
        --setup-only) MODE=setup ;;
        --dry-run) MODE=dry ;;
        *) die "unknown argument: $arg (usage: [--setup-only|--dry-run])" 2 ;;
    esac
done

case "$SPEC" in dflash|none) ;; *) die "SPEC must be none|dflash (got '$SPEC')" 2 ;; esac
[[ "$SERVED_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || die "SERVED_NAME must be a bare id (no '/'): $SERVED_NAME" 2
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( PORT >= 1 && PORT <= 65535 )) || die "PORT must be a TCP port (got '$PORT')" 2
for knob in CTX MAX_SEQS NSPEC PREFIX_MATCH MEM_FLOOR_GB; do
    [[ "${!knob}" =~ ^[0-9]+$ ]] || die "$knob must be a whole number (got '${!knob}')" 2
done
[[ "$GPU_UTIL" =~ ^0?\.[0-9]+$ ]] || die "GPU_UTIL must be a fraction such as 0.45 (got '$GPU_UTIL')" 2

# ── key: a private file, never argv ──────────────────────────────────────
if [ -n "${API_KEY_FILE:-}" ]; then
    key_file="${API_KEY_FILE/#\~/$HOME}"
    [ -f "$key_file" ] || die "API_KEY_FILE $key_file is missing"
    mode="$(stat -c %a "$key_file" 2>/dev/null || stat -f %Lp "$key_file")"
    [ $(( 8#$mode & 8#077 )) -eq 0 ] || die "API_KEY_FILE $key_file is mode $mode; chmod 600 it first"
    VLLM_API_KEY="$(tr -d '[:space:]' < "$key_file")"
    [ -n "$VLLM_API_KEY" ] || die "API_KEY_FILE $key_file is empty"
    export VLLM_API_KEY
else
    unset VLLM_API_KEY
    case "$HOST" in
        127.*|localhost) ;;
        *) die "HOST=$HOST without API_KEY_FILE would serve an unauthenticated brain to the network; set API_KEY_FILE (0600) or keep HOST=127.0.0.1" ;;
    esac
fi

# ── venv (uv only): vLLM + b12x, then its precompiled kernels ────────────
install_venv() {
    command -v uv >/dev/null || die "uv is required to create $VENV"
    log "creating vLLM $VLLM_VERSION venv at $VENV"
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" --torch-backend=auto "vllm[b12x]==$VLLM_VERSION" \
        "transformers>=5.16.1,<5.20.0" "huggingface_hub[hf_xet]"
    "$VENV/bin/vllm" download-kernels \
        || log "download-kernels failed: the first start JIT-compiles kernels (MAX_JOBS=${MAX_JOBS:-4})"
}

# ── models at pinned revisions: the local cache first, the hub on --setup-only
fetch() {
    local path
    path="$(HF_HUB_OFFLINE=1 "$VENV/bin/hf" download "$1" --revision "$2" --quiet 2>/dev/null | tail -n 1)" || path=""
    if [ -z "$path" ] && [ "$MODE" = setup ]; then
        path="$("$VENV/bin/hf" download "$1" --revision "$2" --quiet | tail -n 1)"
    fi
    echo "$path"
}
check_parser() {
    local digest
    digest="$(sha256sum "$1/$PARSER" 2>/dev/null | cut -d' ' -f1)" || digest=""
    [ "$digest" = "$PARSER_SHA256" ] || die "$1/$PARSER sha256 ${digest:-missing} != pinned $PARSER_SHA256; rerun --setup-only"
}

if [ "$MODE" = dry ]; then
    TARGET="<$TARGET_REPO@$TARGET_REV>"
    DRAFT="<$DRAFT_REPO@$DRAFT_REV>"
else
    if [ ! -x "$VENV/bin/vllm" ]; then
        [ "$MODE" = setup ] || die "no vLLM in $VENV; run scripts/serve_nemotron_bolt_vllm.sh --setup-only"
        install_venv
    fi
    TARGET="$(fetch "$TARGET_REPO" "$TARGET_REV")"
    [ -n "$TARGET" ] || die "weights missing: $TARGET_REPO@$TARGET_REV; run scripts/serve_nemotron_bolt_vllm.sh --setup-only"
    check_parser "$TARGET"
    DRAFT=""
    if [ "$SPEC" = dflash ] || [ "$MODE" = setup ]; then
        DRAFT="$(fetch "$DRAFT_REPO" "$DRAFT_REV")"
        [ -n "$DRAFT" ] || die "weights missing: $DRAFT_REPO@$DRAFT_REV; run scripts/serve_nemotron_bolt_vllm.sh --setup-only"
    fi
    if [ "$MODE" = setup ]; then
        log "ready: vLLM $VLLM_VERSION in $VENV; $TARGET_REPO@${TARGET_REV:0:8}; $DRAFT_REPO@${DRAFT_REV:0:8}; $PARSER sha256 ok"
        exit 0
    fi
fi

SPEC_ARGS=()
if [ "$SPEC" = dflash ]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NSPEC,\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASHINFER\",\"draft_load_config\":{\"load_format\":\"auto\"}}")
fi
CACHE_ARGS=()
if [ "$PREFIX_MATCH" != 0 ]; then
    CACHE_ARGS=(--block-size "$PREFIX_MATCH" --prefix-match-unit "$PREFIX_MATCH")
fi
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
EXTRA=(${EXTRA_ARGS:-})
SERVE=("$VENV/bin/vllm" serve "$TARGET"
    --served-model-name "$SERVED_NAME"
    --host "$HOST" --port "$PORT"
    --trust-remote-code --load-format auto --safetensors-load-strategy eager
    --max-model-len "$CTX" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL"
    --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align
    ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}
    --kv-cache-dtype fp8 --linear-backend b12x --moe-backend b12x
    --attention-backend FLASHINFER --mamba-backend flashinfer
    --mamba-ssm-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 --seed 42
    --limit-mm-per-prompt '{"video":0,"image":1}'
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --reasoning-parser-plugin "$TARGET/$PARSER" --reasoning-parser ultra_v3
    --default-chat-template-kwargs '{"enable_thinking": false}'
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
    ${EXTRA[@]+"${EXTRA[@]}"})

if [ "$MODE" = dry ]; then
    printf '%q ' "${SERVE[@]}"
    printf '\n'
    exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
# b12x/CuTe/FlashInfer compile caches in a directory this brain owns: a cache
# left root-owned by a container run (~/.cache/b12x) fails the engine at its
# first kernel with EACCES. The weights stay in the HF cache resolved above.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="$CACHE_DIR"
mkdir -p "$CACHE_DIR"
export MAX_JOBS="${MAX_JOBS:-4}"
# FlashInfer and b12x JIT kernels need nvcc even from a minimal environment (a unit).
if ! command -v nvcc >/dev/null; then
    for cuda in "${CUDA_HOME:-}" /usr/local/cuda; do
        if [ -n "$cuda" ] && [ -x "$cuda/bin/nvcc" ]; then
            export CUDA_HOME="$cuda" PATH="$cuda/bin:$PATH"
            break
        fi
    done
    command -v nvcc >/dev/null || die "nvcc not found (set CUDA_HOME): vLLM's FlashInfer/b12x kernels need the CUDA toolkit"
fi

# torch.compile's CPU code for the Nemotron VL image processor targets the host
# CPU (-march=armv9-a+sve2... on GB10); Ubuntu's default g++ 11 rejects that and
# the engine dies at its multimodal warm-up. Hand inductor a compiler that
# accepts it (CXX wins when set).
if [ -z "${CXX:-}" ] && [ "$(uname -m)" = aarch64 ]; then
    probe="$(mktemp -d)"
    printf 'int main(){return 0;}\n' > "$probe/t.cpp"
    for cxx in g++ g++-15 g++-14 g++-13 clang++; do
        if command -v "$cxx" >/dev/null && "$cxx" -march=armv9-a+sve2 -c "$probe/t.cpp" -o "$probe/t.o" 2>/dev/null; then
            export CXX="$cxx"
            break
        fi
    done
    rm -rf "$probe"
    [ -n "${CXX:-}" ] || die "no C++ compiler accepts -march=armv9-a (torch inductor needs one): install g++-13+ or clang, or set CXX"
fi

# vLLM in the background so a signal (a stopping unit) reaches it at once and
# the MemAvailable guard runs during startup.
"${SERVE[@]}" &
VLLM_PID=$!
trap 'kill -TERM "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; exit 143' TERM
trap 'kill -INT "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; exit 130' INT
log "vLLM $VLLM_VERSION pid $VLLM_PID (port $HOST:$PORT, util $GPU_UTIL, ctx $CTX, seqs $MAX_SEQS, spec $SPEC, prefix match ${PREFIX_MATCH/#0/vllm default}${VLLM_API_KEY:+, key set})"
floor_kb=$(( MEM_FLOOR_GB * 1024 * 1024 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" 2>/dev/null || true)" = 200 ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        wait "$VLLM_PID" || true
        die "vLLM exited during startup (log above)"
    fi
    avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo "$floor_kb")"
    if [ "$avail_kb" -lt "$floor_kb" ]; then
        kill -KILL "$VLLM_PID" 2>/dev/null || true
        die "MemAvailable $(( avail_kb / 1024 )) MiB fell below MEM_FLOOR_GB=$MEM_FLOOR_GB during startup; killed the brain" 3
    fi
    sleep 2
done
log "ready: $SERVED_NAME on http://$HOST:$PORT/v1"
wait "$VLLM_PID"
