#!/usr/bin/env bash
# Serve NVIDIA Cosmos3-Edge for the wrc_demo agent (vLLM-Omni, OpenAI-compatible).
#
# Cosmos3-Edge is the 4B member of the Cosmos3 omnimodal world-model family
# (Mixture-of-Transformers: autoregressive text tower + diffusion tower for
# image/video/action). We use the REASONER side: it is trained for Physical AI
# spatial/temporal reasoning, takes video at ~4 fps, carries a 256K context and
# is natively multimodal -- no separate mmproj projector like the Qwen3-VL
# GGUF path needs.
#
# BF16 weights are ~8.6 GB, so it coexists with Isaac Sim on a single
# RTX PRO 6000 far more comfortably than Qwen3-VL Q4_K_M (~18 GB + projector).
#
# Usage:
#   bash scripts/serve_cosmos3_edge.sh            # docker (recommended)
#   BACKEND=native bash scripts/serve_cosmos3_edge.sh
#
# Then run the demo against it:
#   python -m wrc_demo.apps.demo --llm cosmos3_edge --arm isaac \
#       --cameras isaac,isaac_side --interactive
set -euo pipefail

MODEL="${MODEL:-nvidia/Cosmos3-Edge}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
BACKEND="${BACKEND:-docker}"
# Isaac Sim already owns most of the GPU; cap vLLM so the two coexist.
GPU_FRACTION="${GPU_FRACTION:-0.25}"
MAX_LEN="${MAX_LEN:-32768}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/models/Cosmos3-Edge}"

echo "[cosmos3] model=$MODEL port=$PORT backend=$BACKEND gpu_fraction=$GPU_FRACTION"

if [[ "$BACKEND" == "docker" ]]; then
  # The release-tested image ships the Cosmos3 model code + omni runtime.
  IMAGE="${IMAGE:-vllm/vllm-omni:cosmos3}"
  echo "[cosmos3] pulling $IMAGE (first run downloads several GB)"
  docker pull "$IMAGE"

  # Mount a local snapshot when present so the container does not re-download.
  MOUNT_ARGS=()
  SERVE_TARGET="$MODEL"
  if [[ -d "$LOCAL_DIR" ]]; then
    echo "[cosmos3] using local weights at $LOCAL_DIR"
    MOUNT_ARGS+=(-v "$LOCAL_DIR:/models/Cosmos3-Edge:ro")
    SERVE_TARGET="/models/Cosmos3-Edge"
  fi

  exec docker run --rm --gpus all \
    --ipc=host \
    -p "${PORT}:${PORT}" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "${MOUNT_ARGS[@]}" \
    "$IMAGE" \
    vllm serve "$SERVE_TARGET" \
      --omni \
      --served-model-name "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --init-timeout 1800 \
      --gpu-memory-utilization "$GPU_FRACTION" \
      --max-model-len "$MAX_LEN"
fi

# Native path: needs a venv with a vllm-omni build that knows cosmos3_edge.
VENV="${VENV:-$HOME/Projects/demo/.cosmos3}"
if [[ ! -d "$VENV" ]]; then
  echo "[cosmos3] creating venv at $VENV"
  uv venv --python 3.12 "$VENV"
  # vllm-omni is the Cosmos3-capable distribution; plain vllm lacks the
  # cosmos3_edge architecture and will fail with an unknown-model error.
  VIRTUAL_ENV="$VENV" uv pip install --python "$VENV/bin/python" \
    "vllm-omni" "huggingface_hub[cli]"
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --omni \
  --host "$HOST" --port "$PORT" \
  --init-timeout 1800 \
  --gpu-memory-utilization "$GPU_FRACTION" \
  --max-model-len "$MAX_LEN"
