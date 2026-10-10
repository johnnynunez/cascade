#!/usr/bin/env bash
# B36 wave-8 live: a FRESH Isaac stage (bare reBot scene, GPU 0) on the private bridge port 47701.
# usage: bridge.sh <tag>      (ENGINE=physx|newton, default physx). The bridge always runs from
# the 38f6d08 rig (as the 8-9 Oct series ran it from their main rig); only the MCP server differs.
W8=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/w8
RIG=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/rig_w8main
TAG=${1:?tag}; ENGINE=${ENGINE:-physx}
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
bash $W8/guard.sh || exit 3
systemctl --user stop hermes-w8-b36-bridge 2>/dev/null
systemctl --user reset-failed hermes-w8-b36-bridge 2>/dev/null
for i in $(seq 1 60); do ss -ltn | grep -q ":47701 " || break; sleep 1; done
if ss -ltn | grep -q ":47701 "; then echo "port 47701 still busy"; exit 1; fi
systemd-run --user --unit=hermes-w8-b36-bridge -p MemoryMax=infinity -E CASCADE_ISAAC_PIXEL_MASK=1 \
  -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=$GPU0 -p WorkingDirectory=$RIG \
  /bin/bash -c "exec env -u PYTHONPATH /home/johnny/Projects/demo/cascade/.venv/bin/python scripts/isaac_launch.py --python /home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release/python.sh -- scripts/isaac_bridge.py --port 47701 --usd assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda --engine $ENGINE > $W8/logs/bridge_$TAG.log 2>&1"
for i in $(seq 1 450); do grep -q "cameras ready" $W8/logs/bridge_$TAG.log 2>/dev/null && break; sleep 2; done
grep -q "cameras ready" $W8/logs/bridge_$TAG.log || { echo "bridge not ready ($TAG)"; exit 1; }
ss -ltn | grep -q ":47701 " || { echo "bridge ready but :47701 not listening ($TAG)"; exit 1; }
echo "BRIDGE READY $TAG $ENGINE $(date -Is) $(grep -m1 'GPU attestation' $W8/logs/bridge_$TAG.log | cut -c1-200)"
