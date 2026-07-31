#!/usr/bin/env bash
# Serve NVIDIA Cosmos3-Edge (Reasoner tower) on the DGX Spark with vLLM as an
# OpenAI-compatible endpoint — an alternative local brain to Qwen3.6
# (scripts/serve_qwen_llamacpp.sh). 2.44B-param MoT omni model; the text
# Reasoner is what an MCP host / OpenClaw needs for chat + tool calling.
#
# Usage:
#   ./serve_cosmos_vllm.sh              # create venv (first run), export, serve
#   PORT=8082 CTX=32768 ./serve_cosmos_vllm.sh
#
# Verified working on GB10 2026-07-21. Day-one tooling for this model is
# rough; this script encodes every pitfall in launch order:
#   1. arch `cosmos3_edge` needs transformers GIT MAIN (>=5.15.0.dev0) —
#      the 5.14.x release does not know it.
#   2. Do NOT install cosmos-framework in this venv: it pins transformers
#      4.x, which vLLM >= 0.24 refuses. It is not needed to serve.
#   3. The HF repo ships a diffusers layout (weights under transformer/,
#      vision_encoder/) that vLLM's loader can't see ("Cannot find any model
#      weights") — fixed by a one-time re-export to a standard HF folder.
#   4. vLLM's compile warmup asserts ("Shape 2049 out of considered ranges")
#      on the transformers-backend fallback — serve with --enforce-eager.
#   5. transformers' get_rope_index crashes on the degenerate video grid vLLM
#      feeds during multimodal warmup (`video_grid_thw[:, 0]` IndexError) —
#      a 1-line guard is patched into the installed modeling file below.
#      Remove once upstream https://github.com/huggingface/transformers fixes it.
#   6. JIT kernel builds need the `ninja` BINARY on the serve process PATH.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/vllm}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
HF_REPO="${HF_REPO:-nvidia/Cosmos3-Edge}"
EXPORT_DIR="${EXPORT_DIR:-$MODEL_DIR/Cosmos3-Edge-hf}"
SERVED_NAME="${SERVED_NAME:-cosmos3-edge}"   # KEEP IN SYNC with configs/llm/local_cosmos.yaml
PORT="${PORT:-8082}"
CTX="${CTX:-32768}"
GPU_FRAC="${GPU_FRAC:-0.20}"   # unified memory on Spark is shared with Isaac + Qwen

# 1+2. venv with version-matched stack (vllm 0.25.x + transformers git main).
if [ ! -x "$VENV/bin/vllm" ]; then
    echo "[+] creating vLLM venv at $VENV"
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" "vllm==0.25.*" ninja
    uv pip install --python "$VENV/bin/python" \
        "git+https://github.com/huggingface/transformers.git"
fi
"$VENV/bin/python" - <<'PY'
from transformers import AutoConfig
c = AutoConfig.from_pretrained("nvidia/Cosmos3-Edge")
assert c.architectures == ["Cosmos3EdgeForConditionalGeneration"], c.architectures
print("[+] transformers knows cosmos3_edge")
PY

# 5. Guard the no-video warmup path in the installed modeling file
#    (idempotent; no-op once upstream ships the fix).
"$VENV/bin/python" - <<'PY'
import pathlib, transformers.models.cosmos3_edge.modeling_cosmos3_edge as m
p = pathlib.Path(m.__file__)
src = p.read_text()
bad = "if video_grid_thw is not None:\n            video_grid_thw = torch.repeat_interleave("
good = ("if video_grid_thw is not None and video_grid_thw.ndim == 2 and video_grid_thw.numel() > 0:\n"
        "            video_grid_thw = torch.repeat_interleave(")
if bad in src:
    p.write_text(src.replace(bad, good, 1))
    print("[+] patched get_rope_index no-video guard:", p)
elif "video_grid_thw.ndim == 2" in src:
    print("[+] modeling guard already present")
else:
    raise SystemExit(f"[!] modeling code changed upstream — re-check the warmup crash before serving ({p})")
PY

# 3. One-time re-export: diffusers layout -> standard HF folder vLLM can load.
if [ ! -f "$EXPORT_DIR/config.json" ]; then
    echo "[+] exporting $HF_REPO Reasoner to $EXPORT_DIR (one-time)"
    HF_REPO="$HF_REPO" EXPORT_DIR="$EXPORT_DIR" "$VENV/bin/python" - <<'PY'
import os, torch
from transformers import AutoProcessor
from transformers.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
repo, out = os.environ["HF_REPO"], os.environ["EXPORT_DIR"]
m = Cosmos3EdgeForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16)
print(f"[+] loaded {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
m.save_pretrained(out)
AutoProcessor.from_pretrained(repo).save_pretrained(out)
PY
fi
echo "[+] model: $EXPORT_DIR"

# 4+6. Serve. Thinking mode is ON by default (the chat template exposes
# enable_thinking) — expect ~1-2 min agent turns; plan booth pacing around it.
export PATH="$VENV/bin:$PATH"
exec "$VENV/bin/vllm" serve "$EXPORT_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$CTX" \
    --gpu-memory-utilization "$GPU_FRAC" \
    --enforce-eager
