#!/usr/bin/env bash
# Install the Hermes Agent CLI and register this robot with it.
#
#   ./scripts/install_hermes.sh                       # install + register (mock arm)
#   ./scripts/install_hermes.sh --portal              # also run the Portal login
#   ./scripts/install_hermes.sh --arm so101 --camera l515
#   ./scripts/install_hermes.sh --skip-install        # already have hermes
#
# Hermes is this project's default agent host: `scripts/setup_agents.py` lists it
# first, and one Nous Portal subscription covers both the host models and the
# `--llm hermes` brain profile, so it is the shortest path from a clone to a
# talking robot. Nothing here is required, though -- Codex, Claude Code and
# OpenClaw are equally supported (see setup_agents.py) and the framework runs
# with no agent host at all.
#
# HOST vs BRAIN, because they are different and this script sets up the HOST:
#   host  -- Hermes runs the agent loop, cascade is an MCP tool server. Hermes
#            owns the conversation; cascade's reflex/experience tiers are
#            bypassed (mcp_server.py pins llm='mock' so there is only one brain).
#   brain -- cascade runs its own loop and calls Portal for reasoning:
#            `export NOUS_API_KEY=... && python -m cascade.apps.demo --llm hermes`
#            Keeps the LLM-free tiers and the full trace. See configs/llm/hermes.yaml.
#
# PLATFORMS. The upstream installer supports Linux, macOS, WSL2 and Android
# (Termux); it fetches its own uv/Python 3.11/Node 22/ripgrep/ffmpeg, so the only
# hard prerequisite is git (plus curl and xz-utils on Linux). This works the same
# on a DGX Spark, a DGX Station, a Jetson (Orin/Thor), an RTX workstation or a
# CPU-only laptop -- the agent host is not the part that needs a GPU.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CAMERA="mock"
ARM="mock"
PORTAL=0
SKIP_INSTALL=0
SKIP_BROWSER=0
# Interpreter that has the cascade deps. Prefer an explicit PY, then a venv in
# the checkout, then the shared uv venv beside it, then whatever python3 is on
# PATH -- in that order so no single machine's layout is baked in.
if [[ -n "${PY:-}" ]]; then
    :
elif [[ -x "$REPO/.venv/bin/python" ]]; then
    PY="$REPO/.venv/bin/python"
elif [[ -x "$(cd "$REPO/.." && pwd)/.demo/bin/python" ]]; then
    PY="$(cd "$REPO/.." && pwd)/.demo/bin/python"
else
    PY="$(command -v python3 || true)"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --arm) ARM="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --portal) PORTAL=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --skip-browser) SKIP_BROWSER=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag $1 (try --help)"; exit 1 ;;
    esac
done

if [[ -z "$PY" || ! -x "$PY" ]]; then
    echo "!! no usable python found. Pass --python /path/to/python (it must be"
    echo "   the interpreter that has cascade installed)." >&2
    exit 1
fi

echo "[i] repo   $REPO"
echo "[i] python $PY"

# ── 1. the CLI ───────────────────────────────────────────────────────────
if [[ "$SKIP_INSTALL" == "0" ]]; then
    if command -v hermes >/dev/null 2>&1; then
        echo "[=] hermes already installed: $(command -v hermes)"
    else
        command -v git >/dev/null 2>&1 || {
            echo "!! git is required by the Hermes installer." >&2; exit 1; }
        if [[ "$(uname -s)" == "Linux" ]]; then
            for tool in curl xz; do
                command -v "$tool" >/dev/null 2>&1 || {
                    echo "!! $tool is required on Linux (apt install curl xz-utils)" >&2
                    exit 1; }
            done
        fi
        echo "[+] installing the Hermes Agent CLI"
        # --skip-browser avoids Playwright's apt step, which wants root. Offered
        # as a flag because a headless rig has no use for browser automation and
        # the step is the only part of the install that needs sudo.
        if [[ "$SKIP_BROWSER" == "1" ]]; then
            curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
        else
            curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
        fi
    fi
fi

# The installer writes ~/.local/bin/hermes, which a non-login shell often does
# not have on PATH yet.
export PATH="$HOME/.local/bin:$PATH"
command -v hermes >/dev/null 2>&1 || {
    echo "!! hermes is still not on PATH. Open a new shell (or source ~/.bashrc)"
    echo "   and re-run with --skip-install." >&2
    exit 1; }
echo "[=] hermes: $(command -v hermes)"

# ── 2. credentials ───────────────────────────────────────────────────────
if [[ "$PORTAL" == "1" ]]; then
    echo "[+] running the Nous Portal login (opens a browser)"
    hermes setup --portal
elif [[ -n "${NOUS_API_KEY:-}" ]]; then
    echo "[=] NOUS_API_KEY is set; Portal works without the OAuth login."
    echo "    The same key also drives the BRAIN profile: --llm hermes"
else
    echo "[i] no Nous credentials detected. Either:"
    echo "      $0 --portal          # OAuth login, 300+ models"
    echo "      export NOUS_API_KEY=...   # static key from portal.nousresearch.com"
    echo "    or point Hermes at any other provider with: hermes model"
fi

# ── 3. register the robot as a tool server ───────────────────────────────
echo "[+] registering the cascade MCP server (camera=$CAMERA arm=$ARM)"
"$PY" "$REPO/scripts/setup_agents.py" --host hermes --write \
    --camera "$CAMERA" --arm "$ARM" --python "$PY"

# ── 4. verify ────────────────────────────────────────────────────────────
echo "[+] hermes doctor"
hermes doctor || echo "[!] doctor reported problems (see above)"

cat <<EOF

Done. Next:

  hermes chat                       # talk to the robot through Hermes (HOST)
  ./scripts/hermes_demo.sh          # same, plus a connection test first

  # or keep cascade's own agent loop and use Portal only for reasoning (BRAIN):
  export NOUS_API_KEY=...
  $PY -m cascade.apps.demo --arm $ARM --camera $CAMERA --llm hermes \\
      --task "what is on the table?"

Registered with arm=$ARM. That is the MOCK arm unless you passed --arm;
re-run with --arm so101 (or rebot_rs) when you are at the hardware, and read
the safety notes in README.md first.
EOF
