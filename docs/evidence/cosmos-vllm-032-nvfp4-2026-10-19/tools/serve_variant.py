#!/usr/bin/env python3
"""Serve the shipped Cosmos3-Edge command with other KV-cache/attention arguments (bench variants only).

    VENV=... EXPORT_DIR=... PORT=... serve_variant.py <scripts/cosmos_serving.py> [vllm args...]

Takes the script's serving_plan with KV_DTYPE=auto (no KV flags), appends the given vLLM arguments and execs vLLM
in the prepared venv with the script's serve environment. Run the script's --setup-only first: this skips its
stack checks, compatibility smokes and patches.
"""
import os
import runpy
import sys

helper = runpy.run_path(sys.argv[1])
plan = helper["serving_plan"](dict(os.environ, KV_DTYPE="auto"))
argv = plan["serve_command"] + sys.argv[2:]
print("[variant] " + " ".join(sys.argv[2:]), file=sys.stderr, flush=True)
os.execve(argv[0], argv, helper["serve_environment"](plan))
