#!/usr/bin/env bash
# Install and serve the DEFAULT cascade brain, NVIDIA Nemotron 3.5 Bolt
# 74B-A9B 24GB (GGUF, IQ2 QAT), with the LiLiCorr DFlash drafter on a
# project-owned CUDA llama.cpp build (agent + perception, served id
# nemotron-3.5-bolt). scripts/serve_nemotron_bolt_vllm.sh is the opt-in vLLM
# route: every vLLM build tried for Bolt faulted the GPU (Xid 31) beside Isaac
# Sim; this server ran the e2e pick and a 30-minute soak beside Isaac Newton
# with none (docs/evidence/brain-nemotron-bolt-2026-10-18).
# --setup-only downloads/prepares; --check verifies without writes or downloads.
#
# Pinned: llama.cpp v0.7.0 (620d2225, Nemotron 3.5 Bolt #30520 and LiLiCorr
# speculative decoding #30506; the same build serve_qwen_llamacpp.sh uses, in
# the same $REPO/.llama.cpp), the model, projector and drafter by sha256
# (deploy/brev/profiles/nemotron-3.5-bolt-24gb.json, 24.8 GB).
#
# Two slots (PARALLEL=2, CTX tokens each): one for the agent, one for
# perception. With one slot every background inventory waited behind the
# agent's prefill and a REFER took 27 s (docs/evidence/brain-nemotron-bolt-
# 2026-10-18). llama.cpp keeps each slot's prompt, so a follow-up agent call
# reuses its whole earlier prompt (1-4 s to the first token beside Isaac).
#
# The drafter is on by default (SPEC=dflash): beside Isaac it decodes 41 tok/s
# against 18 without it, and here (unlike vLLM) it costs no prefix hits.
#
# The weights are read once at startup (`--load-mode none`, v0.7.0's --no-mmap)
# instead of memory-mapped: nothing stays mapped behind the GPU copy on the
# Spark's unified memory, and nothing is paged in while serving.
#
# Chat template: scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja, the
# GGUF's own template with OpenClaw's per-session system text (after its
# prompt-cache boundary: date, directives, the Runtime line with the session
# ids) moved after the tool block. The GGUF's template renders the whole system
# text first, so every new chat diverged ~2.4k tokens in and re-prefilled all
# ~13.6k tokens (25.9 s to the first token beside Isaac); now it reuses the
# system text + tools from a context checkpoint. Other prompts render as before.
#
# Warm start (WARM_START=1): scripts/brain_warm_start.py runs beside the server.
# Once the server answers /health (its own readiness, never delayed) it
# prefills what the previous run recorded into every slot: the OpenClaw prefix
# (system text + tools, up to OpenClaw's cache boundary), then what its last two
# sessions shared after the boundary (date, directives; up to the session ids).
# The first chat after a start then costs what any new chat costs instead of a
# full prefill (13.6 s alone); on another day it parts at the date and resumes
# at the boundary. Raw token prompts, no chat and no user turn, each bounded at
# 120 s (~13 s alone); a request arriving meanwhile waits for them. While
# serving it records the server's own tokens of the latest OpenClaw requests
# (0600, WARM_PREFIX_FILE), so it never builds a prefix and follows any change
# of tools or prompt at the next chat. For that the server runs with
# LLAMA_SERVER_SLOTS_DEBUG=1 and --slot-save-path WARM_PREFIX_FILE's
# directory/bolt-slots-PORT (0700): once per new prefix one slot is saved there
# (~0.1 s of server time), its token list read and the file deleted.
# Both widen the API for every holder of the key (docs/OPENSHELL.md, Brain warm
# start): GET /slots returns each slot's last prompt text (agent chats and
# perception), POST /slots/N?action=save|restore|erase works, and a request
# reusing part of a slot logs the 2 tokens where it diverges. llama.cpp v0.7.0
# reads SLOTS_DEBUG once at start, so it cannot be on only while the helper
# records. Hence the warm start runs only behind API_KEY_FILE (on a port without
# a key any local process could use them; the server binds 127.0.0.1 either
# way), and the OpenShell sandbox reaches only /v1/* of this brain
# (scripts/openshell_rig.py brain_profile). A warm start that cannot run (no
# key, no helper, a llama-server without --slot-save-path, a slot directory that
# cannot be private) is turned off with a warning; the brain serves without it,
# and without any LLAMA_SERVER_SLOTS_DEBUG the caller exported.
#
# Knobs: PORT (8080), API_KEY_FILE, CTX (65536 per slot), PARALLEL (2),
# SPEC (dflash | none: the LiLiCorr drafter or none), NSPEC (7, drafted tokens),
# ALIAS (nemotron-3.5-bolt), CHAT_TEMPLATE_FILE (the template above; empty =
# the GGUF's own), WARM_START (1 | 0), WARM_PREFIX_FILE (default
# ${XDG_STATE_HOME:-~/.local/state}/cascade/bolt-openclaw-prefix-$PORT.json),
# MODEL_ROOT, LLAMA_DIR / LLAMA_SERVER, CASCADE_LLAMA_CUDA_ARCH, JOBS,
# CASCADE_MODEL_MIRROR_URL.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$REPO}"
MODEL_DIR="$MODEL_ROOT/models/nemotron-3.5-bolt"
MODEL_FILE="$MODEL_DIR/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
MMPROJ_FILE="$MODEL_DIR/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"
DRAFT_FILE="$MODEL_DIR/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf"
MODEL_MANIFEST="$REPO/deploy/brev/profiles/nemotron-3.5-bolt-24gb.json"
LLAMA_REF=620d2225cc436ab0b001e3200e590e3b286ef9ab   # v0.7.0; KEEP IN SYNC with serve_qwen_llamacpp.sh
LLAMA_DIR="${LLAMA_DIR:-$REPO/.llama.cpp}"
LLAMA_SERVER="${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}"
PORT="${PORT:-8080}"
CTX="${CTX:-65536}"
PARALLEL="${PARALLEL:-2}"
SPEC="${SPEC:-dflash}"
NSPEC="${NSPEC:-7}"
ALIAS="${ALIAS:-nemotron-3.5-bolt}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE-$REPO/scripts/chat_templates/nemotron-3.5-bolt-openclaw.jinja}"
WARM_START="${WARM_START:-1}"
WARM_HELPER="$REPO/scripts/brain_warm_start.py"

die() { printf 'serve_nemotron_bolt_llamacpp: %s\n' "$1" >&2; exit "${2:-1}"; }
# The warm start only saves time: when it cannot run safely, the brain serves without it.
warm_off() {
    printf 'serve_nemotron_bolt_llamacpp: WARNING: warm start off (%s); serving as with WARM_START=0\n' "$1" >&2
    WARM_START=0
}

MODE=serve
case "${1:-}" in
    --check) MODE=check; shift ;;
    --setup-only) MODE=setup; shift ;;
    "") ;;
    *) printf 'Usage: serve_nemotron_bolt_llamacpp.sh [--setup-only|--check]\n' >&2; exit 2 ;;
esac
[[ $# == 0 ]] || die "unexpected arguments" 2
for knob in PORT CTX PARALLEL NSPEC; do
    [[ "${!knob}" =~ ^[1-9][0-9]*$ ]] || die "$knob must be a positive whole number (got '${!knob}')" 2
done
(( PORT <= 65535 )) || die "PORT must be a TCP port (got '$PORT')" 2
case "$SPEC" in dflash|none) ;; *) die "SPEC must be dflash|none (got '$SPEC')" 2 ;; esac
[[ "$ALIAS" =~ ^[A-Za-z0-9][A-Za-z0-9._:-]*$ ]] || die "ALIAS must be a bare id (no '/'): $ALIAS" 2
case "$WARM_START" in 0|1) ;; *) die "WARM_START must be 0|1 (got '$WARM_START')" 2 ;; esac
WARM_PREFIX_FILE="${WARM_PREFIX_FILE:-${XDG_STATE_HOME:-$HOME/.local/state}/cascade/bolt-openclaw-prefix-$PORT.json}"

# CUDA architecture of the GPU that runs the brain (GB10 = 121, RTX PRO 6000 = 120).
cuda_arch() {
    if [[ -n "${CASCADE_LLAMA_CUDA_ARCH:-}" ]]; then
        [[ "$CASCADE_LLAMA_CUDA_ARCH" =~ ^[0-9]{2,3}$ ]] || {
            printf 'CASCADE_LLAMA_CUDA_ARCH must be a CUDA architecture number such as 120 or 121.\n' >&2; return 1; }
        printf '%s' "$CASCADE_LLAMA_CUDA_ARCH"
        return 0
    fi
    local selector=() cap
    [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]] && selector=(--id="${CUDA_VISIBLE_DEVICES%%,*}")
    cap="$(nvidia-smi ${selector[@]+"${selector[@]}"} --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
    [[ "$cap" =~ ^[0-9]{2,3}$ ]] || {
        printf 'Cannot read the CUDA compute capability from nvidia-smi; set CASCADE_LLAMA_CUDA_ARCH.\n' >&2; return 1; }
    printf '%s' "$cap"
}

check_assets() {
    "$PY" -B "$REPO/deploy/brev/fetch_models.py" --root "$MODEL_ROOT" --manifest "$MODEL_MANIFEST" --check
}

check_template() {
    [[ -z "$CHAT_TEMPLATE_FILE" || -f "$CHAT_TEMPLATE_FILE" ]] || {
        printf 'CHAT_TEMPLATE_FILE %s is missing (empty = the GGUF'"'"'s own template).\n' "$CHAT_TEMPLATE_FILE" >&2; return 1; }
}

check_warm() {
    [[ "$WARM_START" == 0 || -n "${API_KEY_FILE:-}" ]] \
        || warm_off "it needs API_KEY_FILE: it opens GET /slots prompt text and slot save/restore/erase to every caller"
    [[ "$WARM_START" == 0 || -f "$WARM_HELPER" ]] || warm_off "$WARM_HELPER is missing"
}

check_source() {
    [[ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" == "$LLAMA_REF" && -z "$(git -C "$LLAMA_DIR" status --porcelain --untracked-files=no)" ]] || {
        printf 'Isolated llama.cpp source differs from its pinned revision (v0.7.0); preserve it and repair explicitly.\n' >&2; return 1;
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
    [[ -x "$LLAMA_SERVER" ]] || { printf 'Missing llama-server; run scripts/serve_nemotron_bolt_llamacpp.sh --setup-only.\n' >&2; return 1; }
    if [[ "$LLAMA_DIR" == "$REPO/.llama.cpp" && "$LLAMA_SERVER" == "$LLAMA_DIR/build/bin/llama-server" ]]; then
        check_source
        runtime_receipt check
    fi
    local help
    help="$("$LLAMA_SERVER" --help 2>&1)" || return 1
    local flags=(--alias --chat-template-kwargs --reasoning --mmproj --spec-draft-model draft-dflash --api-key-file --load-mode)
    [[ -z "$CHAT_TEMPLATE_FILE" ]] || flags+=(--chat-template-file)
    for flag in "${flags[@]}"; do
        [[ "$help" == *"$flag"* ]] || { printf 'llama-server lacks %s; select a compatible LLAMA_SERVER (v0.7.0).\n' "$flag" >&2; return 1; }
    done
    [[ "$WARM_START" == 0 || "$help" == *--slot-save-path* ]] || warm_off "llama-server lacks --slot-save-path"
}

if [[ "$MODE" == setup ]]; then
    set -- --root "$MODEL_ROOT" --manifest "$MODEL_MANIFEST"
    [[ -z "${CASCADE_MODEL_MIRROR_URL:-}" ]] || set -- "$@" --model-mirror-url "$CASCADE_MODEL_MIRROR_URL"
    "$PY" -B "$REPO/deploy/brev/fetch_models.py" "$@"
fi
check_assets
check_template
check_warm
if [[ "$MODE" == check ]]; then
    check_server
    printf 'Nemotron 3.5 Bolt GGUF, vision projector, LiLiCorr drafter and llama-server are prepared; no runtime proof performed.\n'
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

# Authenticated brain (OpenShell routes: PORT=8081 API_KEY_FILE=...). llama-server
# reads the key from the file itself, so argv carries a path, never the key.
set --
key_file=""
if [[ -n "${API_KEY_FILE:-}" ]]; then
    key_file="${API_KEY_FILE/#\~/$HOME}"
    [[ -f "$key_file" ]] || die "API_KEY_FILE $key_file is missing"
    key_mode="$(stat -c %a "$key_file" 2>/dev/null || stat -f %Lp "$key_file")"
    (( (8#$key_mode & 8#077) == 0 )) || die "API_KEY_FILE $key_file is mode $key_mode; chmod 600 it first"
    [[ -n "$(tr -d '[:space:]' < "$key_file")" ]] || die "API_KEY_FILE $key_file is empty"
    set -- --api-key-file "$key_file"
fi
if [[ "$SPEC" == dflash ]]; then
    set -- "$@" --spec-type draft-dflash --spec-draft-model "$DRAFT_FILE" --spec-draft-n-max "$NSPEC" --spec-draft-ngl 999
fi
[[ -z "$CHAT_TEMPLATE_FILE" ]] || set -- "$@" --chat-template-file "$CHAT_TEMPLATE_FILE"
# Slot files hold the slots' state: a directory of this user (mkdir -p fails on
# anything else that exists there), never a link elsewhere, mode 0700 even when
# it existed before.
private_dir() {
    mkdir -p -m 700 "$1" && [[ ! -L "$1" ]] \
        && [[ "$(stat -c %u "$1" 2>/dev/null || stat -f %u "$1")" == "$(id -u)" ]] && chmod 700 "$1"
}
warm_slots="$(dirname "$WARM_PREFIX_FILE")/bolt-slots-$PORT"
if [[ "$WARM_START" == 1 ]] && ! private_dir "$warm_slots" 2>/dev/null; then
    warm_off "cannot make $warm_slots a private directory"
fi
if [[ "$WARM_START" == 1 ]]; then
    # Watches this PID, which `exec` hands to llama-server; ends with it.
    warm=(--port "$PORT" --server-pid "$$" --slots "$PARALLEL" --prefix-file "$WARM_PREFIX_FILE"
          --slot-save-dir "$warm_slots" --chat-template-file "$CHAT_TEMPLATE_FILE" --model-file "$MODEL_FILE")
    [[ -z "$key_file" ]] || warm+=(--api-key-file "$key_file")
    "$PY" -B "$WARM_HELPER" "${warm[@]}" &
    # On whatever the caller exported: the helper reads the slots' prompts.
    export LLAMA_SERVER_SLOTS_DEBUG=1
    set -- "$@" --slot-save-path "$warm_slots"
else
    # Never inherited: with it GET /slots shows every slot's prompt text, here
    # possibly to any local process (no key).
    unset LLAMA_SERVER_SLOTS_DEBUG
fi
# The model card's llama.cpp recipe (q8_0 KV; DFlash with the LiLiCorr GGUF
# drafter when SPEC=dflash), loopback only, thinking off, one context of CTX
# tokens per slot.
exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" --mmproj "$MMPROJ_FILE" "$@" \
    --alias "$ALIAS" --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$(( CTX * PARALLEL ))" --parallel "$PARALLEL" -ngl 999 --load-mode none \
    --batch-size 2048 --ubatch-size 512 \
    --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja \
    --reasoning off --chat-template-kwargs '{"enable_thinking":false}'
