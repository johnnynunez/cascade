#!/usr/bin/env bash
# Serve NVIDIA Cosmos3-Edge (Reasoner tower) on the DGX Spark with SGLang as
# an OpenAI-compatible endpoint -- an alternative local brain to the vLLM
# path (scripts/serve_cosmos_vllm.sh). Same model, same chat template, same
# XML tool-call shape agent/cosmos3.py parses; only the serving engine
# differs, so configs/llm/local_cosmos_sglang.yaml still declares
# `type: cosmos3`. Point OpenClaw or this repo's own orchestrator at
# whichever engine you brought up -- they are interchangeable brains.
#
# Usage:
#   ./serve_cosmos_sglang.sh              # create venv (first run), export, serve
#   PORT=8083 CTX=32768 ./serve_cosmos_sglang.sh
#
# UNVERIFIED on this rig -- unlike serve_cosmos_vllm.sh (verified on GB10
# 2026-07-21), this script has not had a booth rehearsal against it yet.
# It mirrors the vLLM script's day-one pitfalls (steps 1-3 are identical
# because they are properties of the CHECKPOINT, not the serving engine) and
# adds SGLang's own launch flags. Update this header once it has a verified
# run; treat every "assume" below as the first thing to check if serving
# fails.
#   1. arch `cosmos3_edge` needs transformers GIT MAIN (>=5.15.0.dev0) --
#      same requirement as vLLM, SGLang's HF fallback path uses the same
#      `transformers` modeling code.
#   2. Do NOT install cosmos-framework in this venv: it pins transformers
#      4.x, which recent SGLang releases also refuse. Not needed to serve.
#   3. The HF repo ships a diffusers layout (weights under transformer/,
#      vision_encoder/) neither loader can read directly -- reuses the same
#      one-time re-export as the vLLM script (EXPORT_DIR is shared, so run
#      serve_cosmos_vllm.sh once first and this script skips the re-export).
#   4. SGLang's CUDA-graph capture is the eager-mode analogue of vLLM's
#      warmup crash -- `--disable-cuda-graph` avoids replaying the same
#      "degenerate video grid" shape assertion until upstream fixes it.
#   5. Same transformers `get_rope_index` guard as the vLLM script (patched
#      idempotently below) -- SGLang's HF-backend fallback hits the same
#      crash on the no-video warmup path.
#   6. JIT kernel builds need the `ninja` BINARY on the serve process PATH,
#      same as vLLM.
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/sglang}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
HF_REPO="${HF_REPO:-nvidia/Cosmos3-Edge}"
EXPORT_DIR="${EXPORT_DIR:-$MODEL_DIR/Cosmos3-Edge-hf}"   # shared with serve_cosmos_vllm.sh
SERVED_NAME="${SERVED_NAME:-cosmos3-edge}"   # KEEP IN SYNC with configs/llm/local_cosmos_sglang.yaml
PORT="${PORT:-8083}"                          # deliberately != vLLM's 8082, see the yaml note
CTX="${CTX:-32768}"
GPU_FRAC="${GPU_FRAC:-0.20}"   # unified memory on Spark is shared with Isaac + Qwen/vLLM

# 1+2. venv with version-matched stack (sglang + transformers git main).
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sglang" 2>/dev/null; then
    echo "[+] creating SGLang venv at $VENV"
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" "sglang[all]" ninja
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
#    (idempotent; no-op once upstream ships the fix; identical patch to
#    serve_cosmos_vllm.sh since both engines fall back to the same HF code).
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
    raise SystemExit(f"[!] modeling code changed upstream -- re-check the warmup crash before serving ({p})")
PY

# 3. One-time re-export: diffusers layout -> standard HF folder either loader
#    can read. Shared with the vLLM script via EXPORT_DIR -- if you already
#    ran serve_cosmos_vllm.sh once, this is a no-op.
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
# enable_thinking) -- expect ~1-2 min agent turns; plan booth pacing around it.
export PATH="$VENV/bin:$PATH"
exec "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$EXPORT_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --context-length "$CTX" \
    --mem-fraction-static "$GPU_FRAC" \
    --trust-remote-code \
    --disable-cuda-graph
