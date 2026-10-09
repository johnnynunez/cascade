#!/usr/bin/env bash
# B40 interleaved live A/B: on, off, on, off, ... (3 runs per arm; 5 if the first 3 disagree),
# one fresh process (= one fresh belief store) each. NOT YET RUN (CPU-only child).
# Needs the private Isaac 6.2 PhysX bridge on 45250 (bare scene, CASCADE_ISAAC_PIXEL_MASK=1) and GPU 0.
# Arm = memory.size_gate (on = B40, off = the store before it). Target: on -> 0 container views
# in prop beliefs (off: ~3 per run), bin belief {isaac: orange, isaac_side: yellow}, precision/recall
# not lower than off.
set -u
ITEM=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/fusion-size-gate
MODELS=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/same-colour-beliefs/live/models
GPU=${GPU:?set GPU to the GPU-0 UUID from nvidia-smi -L}
REPS=${REPS:-3}
cd "$MODELS" || exit 2
OUT=$ITEM/live/results
mkdir -p "$OUT"
for rep in $(seq 1 "$REPS"); do
  for arm in on off; do
    env -u PYTHONPATH PYTHONPATH=$ITEM/cascade/src CUDA_VISIBLE_DEVICES=$GPU \
      /home/johnny/Projects/demo/cascade/.venv/bin/python "$ITEM/live/b40_live_ab.py" \
      --size-gate $arm --frames 12 --models "$MODELS" \
      --json "$OUT/ab_${arm}_r${rep}.json" > "$OUT/ab_${arm}_r${rep}.log" 2>&1
    echo "r$rep $arm rc=$? $(tail -2 "$OUT/ab_${arm}_r${rep}.log" | head -1 | cut -c1-260)"
  done
done
echo AB_DONE
