#!/usr/bin/env bash
# Full LIBERO sweep: no-op floor + OpenVLA on every suite.
#
# The floor is re-run per suite with the CORRECT per-suite max_steps, so the
# two rows of each pair are directly comparable. It is cheap now (the first
# floor ran at a wrong 600-step cap and took 314 s; at 220 it is ~2 min).
#
# libero_90 is excluded: 90 tasks would be ~9x the runtime of any other suite
# and the papers report the four below.
set -uo pipefail

VENV=${CASCADE_BENCH_VENV:-python}
OUT=${CASCADE_BENCH_RESULTS:-./results}
mkdir -p "$OUT"
EPS="${EPS:-5}"

export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0

for suite in libero_spatial libero_object libero_goal libero_10; do
    for policy in noop openvla; do
        f="$OUT/${policy}_${suite}_ep${EPS}.json"
        if [ -f "$f" ]; then
            echo "[skip] $f exists"
            continue
        fi
        echo "=== $suite / $policy / ${EPS} eps ==="
        args=(--suite "$suite" --policy "$policy" --episodes "$EPS" --json "$f")
        # Each suite has its OWN fine-tuned checkpoint -- the spatial one only
        # carries the `libero_spatial` unnorm key, so reusing it elsewhere
        # would either crash or (worse) mis-scale every action.
        if [ "$policy" = openvla ]; then
            ckpt="${CASCADE_BENCH_MODELS:-$HOME/models}/openvla-7b-libero-${suite#libero_}"
            if [ ! -f "$ckpt/config.json" ]; then
                echo "[!] missing checkpoint $ckpt -- skipping"
                continue
            fi
            args+=(--model-path "$ckpt")
        fi
        stdbuf -oL "$VENV" -u ${CASCADE_BENCH_ROOT:-.}/run_libero.py "${args[@]}" 2>&1 \
            | stdbuf -oL grep -E "^  \[|^SUCCESS|unnorm_key"
    done
done
echo "SWEEP_COMPLETE"
