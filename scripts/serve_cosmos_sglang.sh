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
#   API_KEY_FILE=~/.cascade/nemoclaw/llama.key ./serve_cosmos_sglang.sh   # create venv (first run), export, serve
#   PORT=8083 CTX=32768 API_KEY_FILE=... ./serve_cosmos_sglang.sh
#
# Keyed, scrubbed, loopback (#409 made it the OpenShell Cosmos agent's brain,
# `launch.sh --brain cosmos-sglang`): it serves only with API_KEY_FILE, a 0600
# file holding one bearer key; the key never reaches argv, and SGLang logs at
# warning (its INFO start line prints every argument, the key included). Use
# the brain key file every local brain uses, ~/.cascade/nemoclaw/llama.key:
# launch.sh's brain checks and the agent's brain provider, openclaw_demo.sh and
# the local_cosmos_sglang profile send it with no variable set
# (CASCADE_MODEL_API_KEY_FILE or CASCADE_BRAIN_KEY_FILE name another). It binds
# 127.0.0.1 only, drops the inherited variables that make SGLang add a
# listener, load code, move a route or swap its HTTP server, and runs the
# SGLang pinned below (SGLANG_VERSION: the version that list and
# docs/OPENSHELL.md's route audit were read from). No
# --trust-remote-code: Cosmos3-Edge is native in SGLang (srt/models/
# cosmos3_edge.py) and in transformers (models/cosmos3_edge, 5.19.0 too), and
# nvidia/Cosmos3-Edge ships no code (no .py, no auto_map at a9d944e2), so
# nothing remote would run; without the flag a repo that started shipping code
# fails to load instead of executing it. The vLLM route serves the same export
# with trust_remote_code=False (docs/evidence/cosmos-vllm-032-nvfp4-2026-10-19).
#
# UNVERIFIED on this rig -- unlike serve_cosmos_vllm.sh (verified on GB10
# 2026-07-21), this script has not had a booth rehearsal against it yet.
# It mirrors the vLLM script's day-one pitfalls (steps 1-3 are identical
# because they are properties of the CHECKPOINT, not the serving engine) and
# adds SGLang's own launch flags. Update this header once it has a verified
# run; treat every "assume" below as the first thing to check if serving
# fails.
#   1. arch `cosmos3_edge` is native in transformers 5.19.0 (SGLang 0.5.22's
#      own pin, and the vLLM route's, scripts/cosmos_serving.py): the venv
#      installs exactly that, never git main over the pin. SGLang's HF
#      fallback path uses the same `transformers` modeling code.
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
# The audited release: KEEP IN SYNC with the scrub below and docs/OPENSHELL.md.
SGLANG_VERSION=0.5.22
# SGLang 0.5.22's transformers pin; KEEP IN SYNC with scripts/cosmos_serving.py
# (TRANSFORMERS_VERSION, MODEL_REVISION): the checkpoint revision the vLLM
# route exports, which ships no code (no .py file, no auto_map in config.json).
TRANSFORMERS_VERSION=5.19.0
MODEL_REVISION=a9d944e2c6a1bf9f48b92ad16348e70c5f1836ba

die() { printf 'serve_cosmos_sglang: %s\n' "$1" >&2; exit "${2:-1}"; }

# No inherited variable opens anything on the server. SGLang 0.5.22 takes these
# from the environment (sglang/srt/environ.py, entrypoints/http_server.py,
# debug_utils/dumper.py): SGLANG_GRPC_PORT starts a native gRPC listener beside
# HTTP that the HTTP key does not cover (SGLang refuses it with --api-key; the
# caller must not decide whether the brain starts); SGLANG_RUST_SERVER swaps in
# the embedded Rust HTTP server; SGLANG_OLLAMA_*_ROUTE and AIP_PREDICT_ROUTE
# move the Ollama routes and /vertex_generate (raw /generate parameters)
# anywhere, /v1/** included; DUMPER_SERVER_PORT starts a keyless control server
# on 0.0.0.0 (=reuse adds /dumper/{method}), the other DUMPER_* dump tensors or
# load code; SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES lets x-override-* headers
# set request fields (priority, routing, the PD bootstrap host and port);
# EXA_API_KEY adds a web search tool behind /v1/responses (the brain would
# fetch the web for the caller); SGLANG_EXTERNAL_MODEL_PACKAGE imports a
# package that replaces the whole model registry (srt/models/registry.py),
# SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE one of multimodal processors
# (managers/tokenizer_manager.py) and SGLANG_EXTERNAL_MM_MODEL_ARCH names its
# architecture; SGLANG_LOGGING_CONFIG_PATH swaps the logging setup for a
# dictConfig file (utils/common.py): its `()` factories import code, and an
# INFO handler would write the start line, the key included;
# SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE replaces the torch.distributed init
# method (distributed/bootstrap.py): `env://` or `tcp://HOST:PORT` puts the
# rendezvous store (torch's TCPStore, which listens on every interface) on a
# port the caller picks, and HOST may be another machine. The host, the
# port, the key and the log level are flags below. A bash without `compgen`
# refuses rather than serve unscrubbed.
names="$(compgen -e)" || die "cannot list the environment (compgen failed); refusing to start SGLang unscrubbed"
for name in $names; do
    case "$name" in SGLANG_GRPC_PORT|SGLANG_RUST_SERVER|SGLANG_OLLAMA_*|AIP_*|DUMPER_*|SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES|EXA_API_KEY|SGLANG_EXTERNAL_*|SGLANG_LOGGING_CONFIG_PATH|SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE) unset "$name" ;; esac
done

# The key: a private file holding one key, never argv (checked before anything
# is installed or exported).
[[ -n "${API_KEY_FILE:-}" ]] \
    || die "API_KEY_FILE is required: a 0600 file holding the brain's bearer key; launch.sh and openclaw_demo.sh send ~/.cascade/nemoclaw/llama.key unless CASCADE_MODEL_API_KEY_FILE or CASCADE_BRAIN_KEY_FILE name another"
key_file="${API_KEY_FILE/#\~/$HOME}"
[[ -f "$key_file" ]] || die "API_KEY_FILE $key_file is missing"
key_mode="$(stat -c %a "$key_file" 2>/dev/null || stat -f %Lp "$key_file")"
(( (8#$key_mode & 8#077) == 0 )) || die "API_KEY_FILE $key_file is mode $key_mode; chmod 600 it first"
(( $(wc -w < "$key_file") == 1 )) || die "API_KEY_FILE $key_file must hold one key (SGLang takes exactly one)"

# 1+2. venv with the pinned stack (sglang + its own transformers pin).
if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c "import sglang" 2>/dev/null; then
    echo "[+] creating SGLang $SGLANG_VERSION venv at $VENV"
    uv venv "$VENV" --python 3.12
    # (0.5.22's `all` extra adds diffusion, fastokens, http2 and tracing: none
    # serves this model.)
    uv pip install --python "$VENV/bin/python" "sglang==$SGLANG_VERSION" "transformers==$TRANSFORMERS_VERSION" ninja
fi
installed="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("sglang"))')" \
    || die "cannot read the SGLang version in $VENV"
[[ "$installed" == "$SGLANG_VERSION" ]] \
    || die "$VENV holds SGLang $installed; this script's environment scrub and route audit are for SGLang $SGLANG_VERSION: choose another VENV"
installed="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("transformers"))')" \
    || die "cannot read the transformers version in $VENV"
[[ "$installed" == "$TRANSFORMERS_VERSION" ]] \
    || die "$VENV holds transformers $installed; Cosmos3-Edge loads without remote code on transformers $TRANSFORMERS_VERSION (SGLang $SGLANG_VERSION's pin): choose another VENV"
HF_REPO="$HF_REPO" MODEL_REVISION="$MODEL_REVISION" "$VENV/bin/python" - <<'PY'
import os
from transformers import AutoConfig
c = AutoConfig.from_pretrained(os.environ["HF_REPO"], revision=os.environ["MODEL_REVISION"])
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
    HF_REPO="$HF_REPO" MODEL_REVISION="$MODEL_REVISION" EXPORT_DIR="$EXPORT_DIR" "$VENV/bin/python" - <<'PY'
import os, torch
from transformers import AutoProcessor
from transformers.models.cosmos3_edge import Cosmos3EdgeForConditionalGeneration
repo, rev, out = os.environ["HF_REPO"], os.environ["MODEL_REVISION"], os.environ["EXPORT_DIR"]
m = Cosmos3EdgeForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.bfloat16)
print(f"[+] loaded {sum(p.numel() for p in m.parameters())/1e9:.2f}B params")
m.save_pretrained(out)
AutoProcessor.from_pretrained(repo, revision=rev).save_pretrained(out)
PY
fi
echo "[+] model: $EXPORT_DIR"

# 4+6. Serve. Thinking mode is ON by default (the chat template exposes
# enable_thinking) -- expect ~1-2 min agent turns; plan booth pacing around it.
# SGLang 0.5.22 takes its key only as `--api-key KEY` (no key file, no
# variable), and argv is readable by every local user (/proc/<pid>/cmdline): the
# launcher reads the file inside the server process and runs
# sglang.launch_server as `python -m` does (the same sys.argv and __main__ for
# its spawned workers). SGLang logs its resolved arguments, the key included,
# at INFO (entrypoints/engine.py `server_args=`), so it runs at
# `--log-level warning` (the HTTP log follows it): a log kept by nohup or the
# journal never holds the key. /get_server_info still returns the arguments to
# key holders.
export PATH="$VENV/bin:$PATH"
launch='import runpy, sys
with open(sys.argv[1]) as key_file:
    key = key_file.read().strip()
sys.argv[1:] = [*sys.argv[2:], "--api-key", key]
runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)'
exec "$VENV/bin/python" -c "$launch" "$key_file" \
    --model-path "$EXPORT_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --context-length "$CTX" \
    --mem-fraction-static "$GPU_FRAC" \
    --disable-cuda-graph \
    --log-level warning
