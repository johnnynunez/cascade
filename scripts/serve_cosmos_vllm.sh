#!/usr/bin/env bash
# Local Cosmos3-Edge reasoner for OpenClaw via native OpenAI tool_calls.
# Setup and export use a dedicated VENV (default: this checkout/.cosmos).
# See --help / --dry-run; neither installs anything or needs an NVIDIA GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${COSMOS_PYTHON:-python3}" "$SCRIPT_DIR/cosmos_serving.py" "$@"
