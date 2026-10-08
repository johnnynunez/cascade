#!/usr/bin/env bash
# Interleaved A/B: on, off, on, off, ... (3 runs per arm), one fresh process each.
cd /home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/same-colour-beliefs/live/models
for rep in 1 2 3; do
  for arm in on off; do
    env -u PYTHONPATH PYTHONPATH=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b32-fusion-self-mask/cascade/src \
      CUDA_VISIBLE_DEVICES=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3 \
      /home/johnny/Projects/demo/cascade/.venv/bin/python \
      /home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b32-fusion-self-mask/live/b32_live_ab.py \
      --self-mask $arm --frames 12 \
      --json /home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b32-fusion-self-mask/live/results/ab_${arm}_r${rep}.json \
      > /home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b32-fusion-self-mask/live/results/ab_${arm}_r${rep}.log 2>&1
    echo "r$rep $arm rc=$? $(tail -2 /home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b32-fusion-self-mask/live/results/ab_${arm}_r${rep}.log | head -1 | cut -c1-220)"
  done
done
echo AB_DONE
