#!/usr/bin/env bash
# B36 A/B #4 (9 Oct): fresh Isaac stage + fresh MCP server per run.
# usage: fresh_rig2.sh <rig: m2|f2> <tag>   (ENGINE=physx|newton)
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
RIG=${1:?rig}; TAG=${2:?tag}; ENGINE=${ENGINE:-physx}
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
systemctl --user stop hermes-b36-mcp hermes-b36-isaac-bridge 2>/dev/null
systemctl --user reset-failed hermes-b36-mcp hermes-b36-isaac-bridge 2>/dev/null
for i in $(seq 1 30); do ss -ltn | grep -qE ":(18521|18740) " || break; sleep 1; done
systemd-run --user --unit=hermes-b36-isaac-bridge -p MemoryMax=infinity -E CASCADE_ISAAC_PIXEL_MASK=1 \
  -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=$GPU0 -p WorkingDirectory=$LV/rig_m2 \
  /bin/bash -c "exec /home/johnny/Projects/demo/cascade/.venv/bin/python scripts/isaac_launch.py --python /home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release/python.sh -- scripts/isaac_bridge.py --port 18521 --usd assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda --engine $ENGINE > $LV/bridge_$TAG.log 2>&1"
for i in $(seq 1 300); do grep -q "cameras ready" $LV/bridge_$TAG.log 2>/dev/null && break; sleep 2; done
grep -q "cameras ready" $LV/bridge_$TAG.log || { echo "bridge not ready"; exit 1; }
rm -rf $LV/run_$TAG
systemd-run --user --unit=hermes-b36-mcp -p MemoryMax=infinity /bin/bash -c "exec $LV/serve2.sh $RIG $TAG > $LV/mcp_server_$TAG.log 2>&1"
for i in $(seq 1 90); do ss -ltn | grep -q "172.17.0.1:18740 " && break; sleep 2; done
ss -ltn | grep -q "172.17.0.1:18740 " && echo "READY $RIG $TAG $ENGINE" || { echo "mcp not ready"; exit 1; }
