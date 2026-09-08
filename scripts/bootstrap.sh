#!/usr/bin/env bash
# One-shot bring-up for the cascade agentic booth: installs the OpenClaw
# CLI if missing, starts the Cosmos3-Edge (vLLM) brain, registers this
# repo's skills with OpenClaw over MCP, and opens the web chat -- the
# curl|bash entry point for a fresh DGX Spark / GB10 box.
#
#   curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash
#
# Or locally:
#   ./scripts/bootstrap.sh [--brain cosmos|cosmos-sglang|qwen|skip] [--arm ...] [--cameras ...] [--no-occupancy]
#
# GPU REQUIRED for the default brain: this composes two already-verified
# scripts in this repo rather than reinventing their logic --
#   scripts/serve_cosmos_vllm.sh   Cosmos3-Edge on vLLM (needs an NVIDIA GPU;
#                                  "Verified working on GB10 2026-07-21")
#   scripts/openclaw_demo.sh       MCP registration + OpenClaw gateway +
#                                  brain wiring ("Verified on OpenClaw
#                                  2026.7.1-2"); this is also what exposes
#                                  cascade's 30 skills to OpenClaw -- there
#                                  is no separate "install skills" step, the
#                                  `openclaw mcp add` + tool probe in step 3
#                                  below IS that step.
# Cannot be exercised end to end on a Mac / non-NVIDIA box: this script has
# only been checked here for syntax and idempotent no-ops (uv already
# installed, deps already present), never a live vLLM+OpenClaw run --
# that needs the actual rig.
set -euo pipefail

BRAIN="cosmos"
ARM="isaac"
CAMERAS="isaac,isaac_side"
WITH_OCCUPANCY=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --brain) BRAIN="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --cameras) CAMERAS="$2"; shift 2 ;;
        --no-occupancy) WITH_OCCUPANCY=0; shift ;;
        *) echo "unknown flag $1" >&2; exit 1 ;;
    esac
done

PY="${PY:-}"
if [[ -z "$PY" ]]; then
    # install_hermes.sh resolution order: repo venv, then ../.demo, then python3
    for cand in "$REPO/.venv/bin/python" "$(cd "$REPO/.." && pwd)/.demo/bin/python"; do
        [[ -x "$cand" ]] && { PY="$cand"; break; }
    done
fi
[[ -n "$PY" && -x "$PY" ]] || PY="$(command -v python3)"

echo "=== 1/4: base tooling (uv, OpenClaw CLI) ===" >&2
command -v uv >/dev/null 2>&1 || { curl -fsSL https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
if ! command -v openclaw >/dev/null 2>&1; then
    echo "[+] installing OpenClaw (https://openclaw.ai/install.sh, unattended)" >&2
    curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard
    export PATH="$HOME/.local/bin:$HOME/.openclaw/bin:$PATH"
    command -v openclaw >/dev/null 2>&1 || {
        echo "[!] openclaw not on PATH after install -- open a new shell (or source your profile) and re-run" >&2
        exit 1
    }
else
    # Upgrading an existing 2026.7.x install to OpenClaw 2.0 (v2026.8.x) is a
    # one-way state migration (flat files -> SQLite, agents.list ->
    # agents.entries). Back up first, migrate, then repair + validate --
    # docs/OPENCLAW_2.0_INTEGRATION_BRIEF.md has the full story.
    OC_VER="$(openclaw --version 2>/dev/null | head -1 || true)"
    case "$OC_VER" in
        *2026.7.*|*2026.6.*)
            echo "[+] OpenClaw $OC_VER -> 2.0 upgrade (backing up ~/.openclaw first)" >&2
            cp -a "$HOME/.openclaw" "$HOME/.openclaw.bak.$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
            openclaw update || echo "[!] openclaw update failed; continuing with $OC_VER" >&2
            openclaw doctor --fix >/dev/null 2>&1 || true
            openclaw config validate || true
            ;;
    esac
fi

if [[ "$WITH_OCCUPANCY" == "1" ]]; then
    echo "=== 2/4: occupancy bridge deps (Open3D, CPU/CUDA-agnostic) ===" >&2
    "$REPO/scripts/install_occupancy_backend.sh" --python "$PY"
    if ! pgrep -f serve_nvblox_bridge.py >/dev/null 2>&1; then
        echo "[+] starting occupancy bridge (background, log: /tmp/wrc-nvblox-bridge.log)" >&2
        PY="$PY" nohup "$REPO/scripts/serve_nvblox.sh" >/tmp/wrc-nvblox-bridge.log 2>&1 &
        disown
    fi
else
    echo "=== 2/4: occupancy bridge skipped (--no-occupancy) ===" >&2
fi

if [[ "$BRAIN" == "cosmos" || "$BRAIN" == "cosmos-sglang" ]]; then
    PORT=8082; SERVE_SCRIPT="serve_cosmos_vllm.sh"
    [[ "$BRAIN" == "cosmos-sglang" ]] && { PORT=8083; SERVE_SCRIPT="serve_cosmos_sglang.sh"; }
    echo "=== 3/4: Cosmos3-Edge brain ($BRAIN, port $PORT) ===" >&2
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[+] already serving on :$PORT" >&2
    else
        command -v nvidia-smi >/dev/null 2>&1 || {
            echo "[!] no NVIDIA GPU visible (nvidia-smi not found) -- Cosmos3-Edge needs CUDA." >&2
            echo "    Use --brain qwen (CPU-friendlier via llama.cpp) or --brain skip instead." >&2
            exit 1
        }
        echo "[+] launching $SERVE_SCRIPT in the background (log: /tmp/wrc-${BRAIN}.log)" >&2
        echo "    First run downloads + re-exports the model: can take a long time." >&2
        nohup "$REPO/scripts/$SERVE_SCRIPT" >"/tmp/wrc-${BRAIN}.log" 2>&1 &
        disown
        echo -n "[+] waiting for :$PORT to answer (up to 30 min, first-run download)..." >&2
        for _ in $(seq 1 360); do
            curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo " up." >&2; break; }
            echo -n "." >&2
            sleep 5
        done
        curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
            echo "" >&2
            echo "[!] $SERVE_SCRIPT never came up -- tail /tmp/wrc-${BRAIN}.log" >&2
            exit 1
        }
    fi
else
    echo "=== 3/4: brain=$BRAIN, no local server to start ===" >&2
fi

echo "=== 4/4: OpenClaw: register skills, wire brain, open chat ===" >&2
PY="$PY" "$REPO/scripts/openclaw_demo.sh" --brain "$BRAIN" --arm "$ARM" --cameras "$CAMERAS"
