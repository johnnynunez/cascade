#!/usr/bin/env bash
# Local Cosmos3-Edge HF reasoner via ordinary vLLM (0.32.0), NOT vLLM-Omni.
# Native OpenAI tool_calls require real inference probes after launch.
# COSMOS_ENGINE=omni/vllm-omni fails closed: upstream Omni's full diffusion
# model is not a drop-in chat/tool-calling runtime for this reasoner export.
# Setup and export use a dedicated VENV, one per vLLM version (default:
# this checkout/.cosmos-vllm-0.32.0-cu130). KV_DTYPE=nvfp4 (default), fp8 or
# auto (BF16) selects the KV cache.
# See --help / --dry-run; neither installs anything or needs an NVIDIA GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${COSMOS_PYTHON:-python3}" "$SCRIPT_DIR/cosmos_serving.py" "$@"
