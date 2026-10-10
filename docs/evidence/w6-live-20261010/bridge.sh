#!/usr/bin/env bash
# live-w6: a FRESH Isaac stage (bare reBot scene, PhysX, GPU 0) on the private bridge port 47000.
# usage: bridge.sh <tag>      (ENGINE=physx|newton, default physx)
# Adapted from HERMES_BACKLOG_20261007/b36-pick-reliability/live/fresh_rig2.sh (ports/units/paths changed).
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
WT=$LV/cascade
TAG=${1:?tag}; ENGINE=${ENGINE:-physx}
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
bash $LV/scratch/guard.sh || exit 3
systemctl --user stop hermes-live-w6-bridge 2>/dev/null
systemctl --user reset-failed hermes-live-w6-bridge 2>/dev/null
for i in $(seq 1 60); do ss -ltn | grep -q "127.0.0.1:47000 " || break; sleep 1; done
if ss -ltn | grep -q "127.0.0.1:47000 "; then echo "port 47000 still busy"; exit 1; fi
systemd-run --user --unit=hermes-live-w6-bridge -p MemoryMax=infinity -E CASCADE_ISAAC_PIXEL_MASK=1 \
  -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=$GPU0 -p WorkingDirectory=$WT \
  /bin/bash -c "exec env -u PYTHONPATH /home/johnny/Projects/demo/cascade/.venv/bin/python scripts/isaac_launch.py --python /home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release/python.sh -- scripts/isaac_bridge.py --port 47000 --usd assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda --engine $ENGINE > $LV/logs/bridge_$TAG.log 2>&1"
for i in $(seq 1 300); do grep -q "cameras ready" $LV/logs/bridge_$TAG.log 2>/dev/null && break; sleep 2; done
grep -q "cameras ready" $LV/logs/bridge_$TAG.log || { echo "bridge not ready ($TAG)"; exit 1; }
ss -ltn | grep -q "127.0.0.1:47000 " || { echo "bridge ready but :47000 not listening ($TAG)"; exit 1; }
echo "BRIDGE READY $TAG $ENGINE $(date -Is) $(grep -m1 'GPU attestation' $LV/logs/bridge_$TAG.log | cut -c1-200)"
