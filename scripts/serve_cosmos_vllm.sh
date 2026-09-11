#!/usr/bin/env bash
# Local Cosmos3-Edge HF reasoner via ordinary vLLM, NOT vLLM-Omni.
# Native OpenAI tool_calls require real inference probes after launch.
# COSMOS_ENGINE=omni/vllm-omni fails closed: upstream Omni's full diffusion
# model is not a drop-in chat/tool-calling runtime for this reasoner export.
# Setup and export use a dedicated VENV (default: this checkout/.cosmos).
# See --help / --dry-run; neither installs anything or needs an NVIDIA GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${COSMOS_PYTHON:-python3}" "$SCRIPT_DIR/cosmos_serving.py" "$@"
