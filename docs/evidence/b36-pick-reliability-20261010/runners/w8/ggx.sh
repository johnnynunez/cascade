#!/usr/bin/env bash
# B36 wave-8 live: GraspGen-X sidecar on the private port 47702, GPU 0 (warm before it binds, B15a).
# Adapted from cascade-lab/HERMES_BACKLOG_20261009/live-w6/scratch/ggx.sh (port/unit/paths changed).
W8=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/w8
RIG=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/rig_w8main
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
bash $W8/guard.sh || exit 3
systemctl --user stop hermes-w8-b36-ggx 2>/dev/null; systemctl --user reset-failed hermes-w8-b36-ggx 2>/dev/null
systemd-run --user --unit=hermes-w8-b36-ggx -p MemoryMax=infinity \
  -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=$GPU0 \
  -E CASCADE_GRASPGENX_SOURCE=/home/johnny/Projects/demo/GraspGenX \
  -E CASCADE_GRASPGENX_PYTHON=/home/johnny/Projects/demo/.graspgenx/bin/python \
  /bin/bash -c "exec env -u PYTHONPATH $RIG/scripts/serve_graspgenx.sh '' 47702 > $W8/logs/ggx_47702.log 2>&1"
for i in $(seq 1 150); do ss -ltn | grep -q "127.0.0.1:47702 \|\*:47702 \|0.0.0.0:47702 " && break; sleep 2; done
ss -ltn | grep -q ":47702 " && echo "GGX READY $(date -Is)" || { echo "ggx not ready"; tail -5 $W8/logs/ggx_47702.log; exit 1; }
