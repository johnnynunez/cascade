#!/usr/bin/env bash
# Serve Qwen3.6-27B locally on the DGX Spark (GB10) with llama.cpp.
#
# Qwen3.6 ships a native Multi-Token-Prediction (MTP) head; recent llama.cpp
# builds use it for self-speculative decoding on supported GGUFs, which is the
# single biggest single-stream speed lever on Spark (see
# https://github.com/ggml-org/llama.cpp/discussions/16578 and
# https://vlaicu.io/posts/dgx-llamacpp-playbook/). NVIDIA's Spark playbook:
# https://build.nvidia.com/spark/llama-cpp
#
# Usage:
#   ./serve_qwen_llamacpp.sh            # build (first run), download, serve
#   PORT=8080 QUANT=Q4_K_M ./serve_qwen_llamacpp.sh
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
PORT="${PORT:-8080}"
QUANT="${QUANT:-Q4_K_M}"
HF_REPO="${HF_REPO:-unsloth/Qwen3.6-27B-GGUF}"
CTX="${CTX:-32768}"

# 1. Build llama.cpp with CUDA for GB10 (sm_121) if not present.
if [ ! -x "$LLAMA_DIR/build/bin/llama-server" ]; then
    echo "[+] building llama.cpp in $LLAMA_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" 2>/dev/null || true
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server
fi

# 2. Download the GGUF (needs `pip install -U "huggingface_hub[cli]"`).
mkdir -p "$MODEL_DIR"
GGUF_PATH=$(ls "$MODEL_DIR"/Qwen3.6-27B*"$QUANT"*.gguf 2>/dev/null | head -1 || true)
if [ -z "$GGUF_PATH" ]; then
    echo "[+] downloading $HF_REPO ($QUANT) to $MODEL_DIR"
    # `hf` may not be on PATH; the demo venv ships it
    HF_BIN=$(command -v hf || echo "$HOME/Projects/demo/.demo/bin/hf")
    "$HF_BIN" download "$HF_REPO" --include "*${QUANT}*.gguf" --local-dir "$MODEL_DIR"
    # NOTE: under `set -euo pipefail` a bare ls-glob pipeline kills the
    # script when one glob has no match; find is match-count agnostic.
    GGUF_PATH=$(find "$MODEL_DIR" -name "*${QUANT}*.gguf" | head -1)
fi
echo "[+] model: $GGUF_PATH"

# 3. Serve. --host 0.0.0.0 so other machines on the venue LAN can use it too.
#    NOTE: flags evolve quickly; on older builds drop unknown ones. Check
#    `llama-server --help | grep -i -E "spec|draft|mtp"` for your build's
#    speculative-decoding switches.
# Vision: Qwen3.6 is a unified VLM; the mmproj projector GGUF enables image
# input (the wrc_demo VLM-grounding second filter depends on it).
MMPROJ_PATH=$(find "$MODEL_DIR" -name "mmproj-BF16.gguf" | head -1)
if [ -z "$MMPROJ_PATH" ]; then
    "$HF_BIN" download "$HF_REPO" --include "mmproj-BF16.gguf" --local-dir "$MODEL_DIR" || true
    MMPROJ_PATH=$(find "$MODEL_DIR" -name "mmproj-BF16.gguf" | head -1)
fi
MMPROJ_ARGS=()
[ -n "$MMPROJ_PATH" ] && MMPROJ_ARGS=(--mmproj "$MMPROJ_PATH")

exec "$LLAMA_DIR/build/bin/llama-server" \
    --model "$GGUF_PATH" \
    "${MMPROJ_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT" \
    --ctx-size "$CTX" \
    --n-gpu-layers 999 \
    --flash-attn on \
    --parallel 2 \
    --jinja
# --parallel 2: the agent LLM tier and the VLM-grounding second filter share
# this server; with one slot a grounding call starves behind a long chat
# completion and times out.
