#!/usr/bin/env bash
# Serve the cascade brain NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB (IQ2 QAT,
# Mamba-2/Transformer hybrid LatentMoE VLM) with vLLM, in the model card's
# container (nightly 77154ca6 + b12x 1.5.1). SPEC=dflash adds the LiLiCorr
# DFlash drafter (7 speculative tokens).
#
# OPT-IN, NOT THE DEFAULT. The default Bolt server is llama.cpp v0.7.0
# (scripts/serve_nemotron_bolt_llamacpp.sh). WARNING: every vLLM build tried
# for Bolt faulted the GPU beside Isaac Sim (Newton) under agent + perception
# load on the DGX Spark: kernel `NVRM: Xid 31 ... VLLM::EngineCor ... MMU Fault
# FAULT_PDE ACCESS_TYPE_VIRT_WRITE`, then "illegal memory access" and
# EngineDeadError (docs/evidence/brain-nemotron-bolt-2026-10-18, section 9):
#   12:39  vLLM 0.32.0 uv venv, torch cu132  mixed replay beside Isaac
#   13:49  vLLM 0.32.0, torch cu132          e2e pick beside Isaac (host hard-reset 13:56)
#   15:37  vLLM 0.32.0, torch cu130          e2e pick beside Isaac
#   16:20  this container (nightly 77154ca6) soak beside Isaac, after a passing e2e
# and a host hard-reset at ~17:07 (no kernel log) 26 min into a soak of vLLM
# 0.32.0 cu130 with CUDA graphs off beside Isaac. Alone, Bolt on vLLM survived
# 55 min of stress (0 Xid). Do not run it beside Isaac Sim on a host that also
# drives a real robot.
#
# Same contract as serve_qwen_vllm.sh, so every route (OpenShell, NemoClaw,
# stdio OpenClaw) is engine-agnostic: OpenAI /v1, a BARE served id
# `nemotron-3.5-bolt` (OpenClaw splits provider/model on '/'), 64k context,
# thinking off, tool calls through vLLM's qwen3_coder parser.
#
#   scripts/serve_nemotron_bolt_vllm.sh                   # 127.0.0.1:8080, no key
#   PORT=8081 API_KEY_FILE=~/.cascade/nemoclaw/llama.key scripts/serve_nemotron_bolt_vllm.sh
#                                                         # authenticated brain (OpenShell / NemoClaw)
#   scripts/serve_nemotron_bolt_vllm.sh --setup-only      # pinned image + b12x, pinned weights, parser hash
#   scripts/serve_nemotron_bolt_vllm.sh --dry-run         # print the docker command; run nothing
#
# Knobs: HOST (127.0.0.1), PORT (8080), API_KEY_FILE, GPU_UTIL (0.45), CTX (65536),
# MAX_SEQS (4), SPEC=none|dflash (none), NSPEC (7), SERVED_NAME (nemotron-3.5-bolt),
# PREFIX_MATCH (64; 0 = vLLM's own choice), KV_DTYPE=fp8|nvfp4 (fp8), MEM_FLOOR_GB (3),
# CACHE_DIR, HF_HUB_CACHE, CONTAINER, EXTRA_ARGS.
#
# KV cache: FP8 by default; KV_DTYPE=nvfp4 is supported. The pinned nightly 77154ca6 carries
# vLLM #46963 (NVFP4 KV on SM12x through FlashInfer) and #60452 (XQA decode of it on SM12x),
# and Bolt's attention layers (head size 128, 2 KV heads) qualify. On an RTX PRO 6000
# (docs/evidence/brain-kv-nvfp4-2026-10-19), NVFP4 gave 28% more KV tokens, but capacity
# was never the limit (26x a 64k request at GPU_UTIL 0.45). It cost 20% on a 54k-token
# prefill and 1.4% decode, and scored GSM8K the same: so FP8 stays the default. The Mamba
# state stays BF16 (--mamba-*-cache-dtype).
#
# Pinned: the vendor image by digest (vLLM 0.32.0rc1 nightly 77154ca6), b12x by
# wheel hash (baked into a local image by --setup-only, so serving needs no
# network), both checkpoints by revision, and the card's ultra_v3 reasoning
# parser plugin by sha256. Weights load offline from the host's HF cache.
#
# Security: the key is read from a 0600 file into VLLM_API_KEY and handed to
# the container by NAME (`docker run -e VLLM_API_KEY`), never on argv. A bind
# beyond loopback without a key is refused. The container runs as this user.
#
# Thinking: off by default (--default-chat-template-kwargs), but the card's
# ultra_v3 parser reads ONLY the request's chat_template_kwargs: a
# NON-streaming request without {"enable_thinking": false} gets content null
# and its tool calls unparsed in reasoning_content. Every cascade client of
# this brain sends the kwarg (perception, demo_proof, OpenClaw provider params).
#
# Prefix caching (the agent's 13-18k-token prompts): PREFIX_MATCH=64 sets the
# attention block to a multiple of 64 (vLLM rounds it up to hold a Mamba page)
# and the match unit to 64; without the drafter a follow-up agent call then hits
# every 64 tokens up to the previous prompt's tail, where vLLM keeps a Mamba
# prefill checkpoint. The drafter (SPEC=dflash) adds a sliding-window KV group
# that vLLM hits only on whole blocks (~4.5k tokens): beside Isaac Newton a
# follow-up agent call waited 6.5-16 s with it and 1.2-2.8 s without it, so
# SPEC=none is the default; the drafter's gain is decode speed (evidence).
#
# Images per prompt: vLLM's default (999), as the Qwen brain, not the card's 1.
# OpenClaw puts a tool's images in the conversation and keeps them: one
# get_observation carries a frame per camera, and at 1 every later agent call
# in that chat was refused (HTTP 400; e2e beside Isaac Newton, 2026-10-18). The
# memory profile is unchanged (one maximum-size image fills the encoder budget).
#
# Spark notes: GPU via CDI (`--device nvidia.com/gpu=all`); eager safetensors
# load (mmap reads crawl on GB10); MemAvailable is watched during startup and
# the container is killed below MEM_FLOOR_GB so the host never freezes; run it
# under `systemd-run --user -p MemoryMax=infinity` from a capped shell.
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
KV_DTYPE="${KV_DTYPE:-fp8}"
MEM_FLOOR_GB="${MEM_FLOOR_GB:-3}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/cascade/nemotron-bolt}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
CONTAINER="${CONTAINER:-cascade-brain-nemotron-bolt-$PORT}"

BASE_IMAGE="vllm/vllm-openai@sha256:a7deff2c216def98c193352aca50fb03a2ee29231355b3033dc3789009f6d23e"
BASE_TAG="nightly-77154ca6116ae34cbfe314beef1016bde9b63989"   # the card's tag; the digest is what runs
B12X_VERSION="1.5.1"
B12X_SHA256="8731e23571d8d45db26d78e7699ec0b23e7ba20b52487adf8c1063c8d8b53a3d"
IMAGE="cascade/nemotron-bolt-vllm:77154ca6-b12x$B12X_VERSION"
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

case "$SPEC" in none|dflash) ;; *) die "SPEC must be none|dflash (got '$SPEC')" 2 ;; esac
case "$KV_DTYPE" in fp8|nvfp4) ;; *) die "KV_DTYPE must be fp8|nvfp4 (got '$KV_DTYPE')" 2 ;; esac
[[ "$SERVED_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || die "SERVED_NAME must be a bare id (no '/'): $SERVED_NAME" 2
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( PORT >= 1 && PORT <= 65535 )) || die "PORT must be a TCP port (got '$PORT')" 2
for knob in CTX MAX_SEQS NSPEC PREFIX_MATCH MEM_FLOOR_GB; do
    [[ "${!knob}" =~ ^[0-9]+$ ]] || die "$knob must be a whole number (got '${!knob}')" 2
done
[[ "$GPU_UTIL" =~ ^0?\.[0-9]+$ ]] || die "GPU_UTIL must be a fraction such as 0.45 (got '$GPU_UTIL')" 2
[[ "$CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "CONTAINER must be a docker container name (got '$CONTAINER')" 2
[[ "$MODE" == setup ]] || log "WARNING: opt-in route. Bolt on vLLM faulted the GPU (Xid 31) beside Isaac Sim on the" \
    "DGX Spark; the default Bolt server is scripts/serve_nemotron_bolt_llamacpp.sh (see this script's header)."

# ── key: a private file, never argv ──────────────────────────────────────
KEY_ENV=()
if [ -n "${API_KEY_FILE:-}" ]; then
    key_file="${API_KEY_FILE/#\~/$HOME}"
    [ -f "$key_file" ] || die "API_KEY_FILE $key_file is missing"
    mode="$(stat -c %a "$key_file" 2>/dev/null || stat -f %Lp "$key_file")"
    [ $(( 8#$mode & 8#077 )) -eq 0 ] || die "API_KEY_FILE $key_file is mode $mode; chmod 600 it first"
    VLLM_API_KEY="$(tr -d '[:space:]' < "$key_file")"
    [ -n "$VLLM_API_KEY" ] || die "API_KEY_FILE $key_file is empty"
    export VLLM_API_KEY
    KEY_ENV=(-e VLLM_API_KEY)   # docker copies the value from this environment
else
    unset VLLM_API_KEY
    case "$HOST" in
        127.*|localhost) ;;
        *) die "HOST=$HOST without API_KEY_FILE would serve an unauthenticated brain to the network; set API_KEY_FILE (0600) or keep HOST=127.0.0.1" ;;
    esac
fi

snapshot() { printf '%s/models--%s/snapshots/%s' "$HF_HUB_CACHE" "${1//\//--}" "$2"; }
TARGET_DIR="$(snapshot "$TARGET_REPO" "$TARGET_REV")"
DRAFT_DIR="$(snapshot "$DRAFT_REPO" "$DRAFT_REV")"
in_container() { printf '/hf/%s' "${1#"$HF_HUB_CACHE"/}"; }

check_parser() {
    local digest
    digest="$(sha256sum "$TARGET_DIR/$PARSER" 2>/dev/null | cut -d' ' -f1)" || digest=""
    [ "$digest" = "$PARSER_SHA256" ] || die "$TARGET_DIR/$PARSER sha256 ${digest:-missing} != pinned $PARSER_SHA256; rerun --setup-only"
}

# ── --setup-only: image, b12x, weights (network), parser hash ───────────
if [ "$MODE" = setup ]; then
    command -v docker >/dev/null || die "docker is required"
    docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        log "building $IMAGE: $BASE_IMAGE + b12x $B12X_VERSION (wheel hash pinned)"
        docker build -t "$IMAGE" \
            --label "org.cascade.base=$BASE_IMAGE" --label "org.cascade.b12x=$B12X_VERSION" - <<EOF
FROM $BASE_IMAGE
RUN printf 'b12x==$B12X_VERSION --hash=sha256:$B12X_SHA256\n' > /tmp/b12x.txt \\
    && pip install --no-cache-dir --no-deps --require-hashes --root-user-action=ignore -r /tmp/b12x.txt \\
    && rm /tmp/b12x.txt && python3 -c "import b12x"
EOF
    fi
    mkdir -p "$HF_HUB_CACHE" "$CACHE_DIR"
    for pin in "$TARGET_REPO@$TARGET_REV" "$DRAFT_REPO@$DRAFT_REV"; do
        log "fetching ${pin%@*} at ${pin#*@} (skips files already in $HF_HUB_CACHE)"
        docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_CACHE=/hf \
            -v "$HF_HUB_CACHE:/hf" --entrypoint hf "$IMAGE" \
            download "${pin%@*}" --revision "${pin#*@}" --quiet >/dev/null
    done
    check_parser
    log "ready: $IMAGE; $TARGET_REPO@${TARGET_REV:0:8}; $DRAFT_REPO@${DRAFT_REV:0:8}; $PARSER sha256 ok"
    exit 0
fi

# ── the server ───────────────────────────────────────────────────────────
SPEC_ARGS=()
if [ "$SPEC" = dflash ]; then
    SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$(in_container "$DRAFT_DIR")\",\"num_speculative_tokens\":$NSPEC,\"rejection_sample_method\":\"standard\",\"attention_backend\":\"FLASHINFER\",\"draft_load_config\":{\"load_format\":\"auto\"}}")
fi
CACHE_ARGS=()
if [ "$PREFIX_MATCH" != 0 ]; then
    # Attention blocks become a multiple of PREFIX_MATCH (vLLM rounds up to the
    # Mamba page) and hits land every PREFIX_MATCH tokens; with the drafter, on
    # whole 4,544-token blocks instead of the 17,984-token LCM (header).
    CACHE_ARGS=(--block-size "$PREFIX_MATCH" --prefix-match-unit "$PREFIX_MATCH")
fi
# shellcheck disable=SC2206  # EXTRA_ARGS is intentionally word-split
EXTRA=(${EXTRA_ARGS:-})
DOCKER=(docker run -d --rm --name "$CONTAINER"
    --device nvidia.com/gpu=all --ipc=host
    --user "$(id -u):$(id -g)" -w /cache
    -p "$HOST:$PORT:8000"
    ${KEY_ENV[@]+"${KEY_ENV[@]}"}
    -e HOME=/cache -e USER=cascade -e LOGNAME=cascade
    -e HF_HUB_CACHE=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e VLLM_USE_V2_MODEL_RUNNER=1 -e TRITON_CACHE_DIR=/cache/.triton
    -e TORCHINDUCTOR_CACHE_DIR=/cache/.cache/torchinductor -e MAX_JOBS=4
    -v "$HF_HUB_CACHE:/hf:ro" -v "$CACHE_DIR:/cache"
    --entrypoint vllm "$IMAGE"
    serve "$(in_container "$TARGET_DIR")"
    --served-model-name "$SERVED_NAME"
    --host 0.0.0.0 --port 8000
    --trust-remote-code --load-format auto --safetensors-load-strategy eager
    --max-model-len "$CTX" --max-num-seqs "$MAX_SEQS" --gpu-memory-utilization "$GPU_UTIL"
    --enable-chunked-prefill --enable-prefix-caching
    ${CACHE_ARGS[@]+"${CACHE_ARGS[@]}"}
    --kv-cache-dtype "$KV_DTYPE" --linear-backend b12x --moe-backend b12x
    --attention-backend FLASHINFER --mamba-backend flashinfer
    --mamba-ssm-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 --seed 42
    --limit-mm-per-prompt '{"video":0}'
    --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --reasoning-parser-plugin "$(in_container "$TARGET_DIR")/$PARSER" --reasoning-parser ultra_v3
    --default-chat-template-kwargs '{"enable_thinking": false}'
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"}
    ${EXTRA[@]+"${EXTRA[@]}"})

if [ "$MODE" = dry ]; then
    printf '%q ' "${DOCKER[@]}"
    printf '\n'
    exit 0
fi

command -v docker >/dev/null || die "docker is required"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE is missing; run scripts/serve_nemotron_bolt_vllm.sh --setup-only"
WEIGHTS=("$TARGET_DIR")
[ "$SPEC" != dflash ] || WEIGHTS+=("$DRAFT_DIR")
for dir in "${WEIGHTS[@]}"; do
    [ -f "$dir/config.json" ] || die "weights missing at $dir; run scripts/serve_nemotron_bolt_vllm.sh --setup-only"
done
check_parser
if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" = true ]; then
    die "container $CONTAINER is already running (reuse it, or stop it: docker stop $CONTAINER)"
fi
docker rm "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$CACHE_DIR"

STARTED=0
stop_container() {
    # stop_owned (process_owner) escalates to SIGKILL after 10 s: stop inside that.
    [ "$STARTED" = 0 ] || docker stop --time 8 "$CONTAINER" >/dev/null 2>&1 || true
}
trap stop_container EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
"${DOCKER[@]}" >/dev/null
STARTED=1
log "container $CONTAINER started (port $HOST:$PORT, util $GPU_UTIL, ctx $CTX, seqs $MAX_SEQS, spec $SPEC, kv $KV_DTYPE, prefix match ${PREFIX_MATCH/#0/vllm default}${VLLM_API_KEY:+, key set})"
docker logs -f "$CONTAINER" 2>&1 &
floor_kb=$(( MEM_FLOOR_GB * 1024 * 1024 ))
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" 2>/dev/null || true)" = 200 ]; do
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" = true ] \
        || die "the vLLM container exited during startup (log above)"
    avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo "$floor_kb")"
    if [ "$avail_kb" -lt "$floor_kb" ]; then
        docker kill "$CONTAINER" >/dev/null 2>&1 || true
        die "MemAvailable $(( avail_kb / 1024 )) MiB fell below MEM_FLOOR_GB=$MEM_FLOOR_GB during startup; killed the brain" 3
    fi
    sleep 2
done
log "ready: $SERVED_NAME on http://$HOST:$PORT/v1"
# In the background: bash runs a trap only between foreground commands, and
# `wait` returns at once on a signal, so a stopping unit stops the container.
docker wait "$CONTAINER" >/dev/null 2>&1 &
wait $! || true
STARTED=0
die "the vLLM container stopped"
