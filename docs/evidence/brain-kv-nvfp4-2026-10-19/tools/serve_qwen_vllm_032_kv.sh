#!/usr/bin/env bash
# Serve the cascade brain with vLLM: Qwen3.8-27B NVFP4 + DFlash2 speculative
# decoding, measured on the DGX Spark (GB10, 2026-10-11).
#
#   single stream, tok/s     prose   code   tool plan
#   llama.cpp Q4_K_XL        10.6    10.6   10.6
#   this (DFlash2, 7 draft)  18.8    29.8   28.7     (acceptance length ~3.5)
#
# Same contract as serve_qwen_llamacpp.sh, so every route (stdio OpenClaw,
# NemoClaw, the standalone OpenShell profile) is engine-agnostic: OpenAI /v1,
# a BARE served id `qwen3.8-27b` (OpenClaw splits provider/model on '/'),
# 64k context, thinking off, and tool calls through vLLM's qwen3_coder parser
# (llama.cpp's schema->grammar conversion rejected an unanchored tool
# `pattern` with a 400 and the chat died before replying).
#
#   scripts/serve_qwen_vllm.sh                      # 127.0.0.1:8080, no key (local stdio profile)
#   PORT=8081 API_KEY_FILE=~/.cascade/brain.key scripts/serve_qwen_vllm.sh
#                                                   # authenticated brain (NemoClaw / OpenShell routes)
#   scripts/serve_qwen_vllm.sh --setup-only         # venv + pinned models, then exit
#
# Knobs: HOST (127.0.0.1), PORT (8080), API_KEY_FILE, GPU_UTIL (0.55; 0.40 next
# to a second model), CTX (65536), SPEC=dflash|mtp|none, NSPEC (7),
# SERVED_NAME, VENV (~/.venvs/vllm-0.32.0-cu130), KV_DTYPE=nvfp4|fp8|auto (nvfp4),
# MAX_JOBS (4), EXTRA_ARGS.
#
# KV cache: NVFP4 by default (vLLM 0.32.0: written by FlashInfer's slot-mapping
# writer, prefill on fa2, decode on XQA on SM12x, vLLM #46963 and #60452), a
# quarter of BF16's bytes per token. It needs the FLASHINFER attention backend;
# vLLM refuses it before loading the weights for head size 64, attention
# sinks and DCP. KV_DTYPE=auto is the BF16 cache 0.31.0 served.
#
# Security: the key is read from a 0600 file into VLLM_API_KEY, never argv
# (argv is world-readable in /proc). A bind beyond loopback without a key is
# refused: that would hand the GPU to anyone on the network.
#
# Spark notes (each one cost an hour):
# - `--safetensors-load-strategy eager`: mmap reads on GB10 run at ~5 MB/s at
#   100 % CPU (27 min to load); eager reads take 50-90 s.
# - `vllm download-kernels` (flashinfer-jit-cache cu130 with an sm121a build,
#   flashinfer-cubin): without it FlashInfer JIT-compiles the CUTLASS FP4 GEMM
#   with parallel nvcc next to the loaded model and the kernel OOM killer takes
#   the engine. MAX_JOBS caps any JIT that remains.
# - GPU memory counts against the caller's cgroup on unified memory: run it
#   under `systemd-run --user -p MemoryMax=infinity` from a capped shell.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
GPU_UTIL="${GPU_UTIL:-0.55}"
CTX="${CTX:-65536}"
SPEC="${SPEC:-dflash}"
NSPEC="${NSPEC:-7}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
KV_DTYPE="${KV_DTYPE:-nvfp4}"
VLLM_VERSION="0.32.0"   # DFlash2 landed in vLLM PR #52816
# torch cu130 (the build 0.31.0 ran on here, and FlashInfer's prebuilt kernels'
# CUDA), not uv's `--torch-backend=auto` pick (cu132 on this driver). A venv per
# vLLM version and torch build: an upgrade builds a fresh one beside the old
# instead of mutating it. (The Bolt brain does not use it: it runs in the model
# card's container, see serve_nemotron_bolt_vllm.sh.)
TORCH_BACKEND="cu130"
VENV="${VENV:-$HOME/.venvs/vllm-$VLLM_VERSION-$TORCH_BACKEND}"
# The target keeps BF16 lm_head/embeddings (the drafter reuses them) and vision.
TARGET_REPO="Inferact/Qwen3.8-27B-NVFP4"
TARGET_REV="6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
DRAFT_REPO="z-lab/Qwen3.8-27B-DFlash2"
DRAFT_REV="50307d4c4cde6860d4eee73e2547cd786fe8e8a4"

die() { echo "serve_qwen_vllm: $1" >&2; exit "${2:-1}"; }

SETUP_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --setup-only) SETUP_ONLY=1 ;;
        *) die "unknown argument: $arg" 2 ;;
    esac
done

case "$SPEC" in dflash|mtp|none) ;; *) die "SPEC must be dflash|mtp|none (got '$SPEC')" 2 ;; esac
case "$KV_DTYPE" in
    nvfp4) KV_ARGS=(--kv-cache-dtype nvfp4 --attention-backend FLASHINFER) ;;
    fp8)   KV_ARGS=(--kv-cache-dtype fp8) ;;
    auto)  KV_ARGS=() ;;
    *) die "KV_DTYPE must be nvfp4|fp8|auto (got '$KV_DTYPE')" 2 ;;
esac
[[ "$SERVED_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || die "SERVED_NAME must be a bare id (no '/'): $SERVED_NAME" 2

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
        127.*|localhost|::1) ;;
        *) die "HOST=$HOST without API_KEY_FILE would serve an unauthenticated brain to the network; set API_KEY_FILE (0600) or keep HOST=127.0.0.1" ;;
    esac
fi

# ── venv (uv only) ───────────────────────────────────────────────────────
if [ ! -x "$VENV/bin/vllm" ]; then
    command -v uv >/dev/null || die "uv is required to create $VENV"
    echo "[+] creating vLLM $VLLM_VERSION venv at $VENV" >&2
    uv venv "$VENV" --python 3.12
    # vLLM 0.32.0 ships CUDA 13 arm64 wheels (b12x extra: its SM12x kernels).
    uv pip install --python "$VENV/bin/python" --torch-backend="$TORCH_BACKEND" "vllm[b12x]==$VLLM_VERSION" \
        "transformers>=5.16.1,<5.20.0" "huggingface_hub[hf_xet]"
    # Prebuilt FlashInfer kernels (sm121a included) instead of a JIT next to the model.
    "$VENV/bin/vllm" download-kernels \
        || echo "[!] precompiled kernels not installed: the first start JIT-compiles them (MAX_JOBS=${MAX_JOBS:-4})" >&2
fi

# ── models at pinned revisions ───────────────────────────────────────────
# The local cache first, with no network (booths have no reliable internet,
# BOOTH_RUNBOOK.md); the hub only on a cache miss.
fetch() {
    local path
    path="$(HF_HUB_OFFLINE=1 "$VENV/bin/hf" download "$1" --revision "$2" --quiet 2>/dev/null | tail -n 1)" || path=""
    if [ -z "$path" ]; then
        path="$("$VENV/bin/hf" download "$1" --revision "$2" --quiet | tail -n 1)"
    fi
    echo "$path"
}
TARGET="$(fetch "$TARGET_REPO" "$TARGET_REV")"
[ -n "$TARGET" ] || die "could not resolve $TARGET_REPO@$TARGET_REV"
DRAFT=""
if [ "$SPEC" = dflash ] || [ "$SETUP_ONLY" = 1 ]; then
    DRAFT="$(fetch "$DRAFT_REPO" "$DRAFT_REV")"
    [ -n "$DRAFT" ] || die "could not resolve $DRAFT_REPO@$DRAFT_REV"
fi
if [ "$SETUP_ONLY" = 1 ]; then
    echo "[+] ready: vLLM $VLLM_VERSION in $VENV; $TARGET_REPO@${TARGET_REV:0:8}; $DRAFT_REPO@${DRAFT_REV:0:8}" >&2
    exit 0
fi

case "$SPEC" in
    dflash) SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$NSPEC}") ;;
    mtp)    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC}") ;;
    none)   SPEC_ARGS=() ;;
esac
# SPEC_ARGS is empty for none: expand it as ${A[@]+"${A[@]}"} below, since
# bash 3.2 (macOS /bin/bash) under set -u calls a plain "${A[@]}" of an
# empty array unbound.

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}"
export MAX_JOBS="${MAX_JOBS:-4}"

# vLLM registers its FlashInfer FP4 GEMM op only when FlashInfer is usable,
# and without flashinfer-cubin it deems FlashInfer unusable unless nvcc is on
# PATH. From a minimal environment (a systemd unit, cron) the NVFP4 engine
# then dies after minutes of loading with "'vllm' object has no attribute
# 'flashinfer_mm_fp4'". Put the toolkit on PATH, or fail now.
if ! command -v nvcc >/dev/null; then
    cuda_bin="${CUDA_HOME:-/usr/local/cuda}/bin"
    [ -x "$cuda_bin/nvcc" ] && export PATH="$cuda_bin:$PATH"
fi
command -v nvcc >/dev/null || die "nvcc not found (PATH or \$CUDA_HOME/bin): vLLM would disable FlashInfer and the NVFP4 model would fail after loading; set CUDA_HOME"

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
# Two names: the bare id the OpenClaw routes need first (it is the one vLLM
# reports), then the repo-wide id configs/llm/local_qwen.yaml and
# install_support.py ask for (vLLM rejects a request for a name it does not
# serve; llama.cpp ignores the name).
NAMES=("$SERVED_NAME")
[ "$SERVED_NAME" = "Qwen/Qwen3.8-27B" ] || NAMES+=("Qwen/Qwen3.8-27B")
exec "$VENV/bin/vllm" serve "$TARGET" \
    --served-model-name "${NAMES[@]}" \
    --host "$HOST" --port "$PORT" \
    --max-model-len "$CTX" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --safetensors-load-strategy eager \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    ${KV_ARGS[@]+"${KV_ARGS[@]}"} \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    ${EXTRA_ARGS:-}
