#!/usr/bin/env bash
# Serve Qwen3.6-27B on the DGX Spark with vLLM + MTP speculative decoding.
#
# vLLM >= 0.21 supports Qwen3.6's built-in MTP draft head; on Spark this is
# "the single biggest single-stream lever" (https://vlaicu.io/posts/dgx-vllm/).
# vLLM aarch64+Blackwell wheels: follow NVIDIA's Spark playbook if the plain
# pip install has no GB10 kernels (https://build.nvidia.com/spark).
#
# Usage:  PORT=8080 ./serve_qwen_vllm.sh
set -euo pipefail

PORT="${PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
VENV="${VENV:-$HOME/.venvs/vllm}"

if [ ! -x "$VENV/bin/vllm" ]; then
    echo "[+] creating vLLM venv at $VENV"
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" "vllm>=0.21"
fi

# MTP speculative decoding: Qwen3.6 exposes its MTP head as draft tokens.
# num_speculative_tokens=2..3 is the sweet spot reported for GB10.
exec "$VENV/bin/vllm" serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 32768 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --gpu-memory-utilization 0.80
