#!/usr/bin/env bash
# Fresh Isaac stage + fresh MCP server (B35 A/B: every arm starts from the same scene).
L=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/nemoclaw-runtime/live
RIG=$L/rig
TAG=${1:?tag}
systemctl --user stop cascade-nemoclaw-mcp hermes-b35-isaac-bridge 2>/dev/null
systemctl --user reset-failed cascade-nemoclaw-mcp hermes-b35-isaac-bridge 2>/dev/null
for i in $(seq 1 30); do ss -ltn | grep -q ":18521 " || break; sleep 1; done
systemd-run --user --unit=hermes-b35-isaac-bridge -p MemoryMax=infinity -E CASCADE_ISAAC_PIXEL_MASK=1 \
  -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3 -p WorkingDirectory=$RIG \
  /bin/bash -c "exec /home/johnny/Projects/demo/cascade/.venv/bin/python scripts/isaac_launch.py --python /home/johnny/Projects/demo/omni_isaac_sim/_build/linux-x86_64/release/python.sh -- scripts/isaac_bridge.py --port 18521 --usd assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda --engine physx > $L/bridge_$TAG.log 2>&1"
for i in $(seq 1 180); do grep -q "cameras ready" $L/bridge_$TAG.log 2>/dev/null && break; sleep 2; done
grep -q "cameras ready" $L/bridge_$TAG.log || { echo "bridge not ready"; exit 1; }
rm -rf $L/run_$TAG
systemd-run --user --unit=cascade-nemoclaw-mcp -p MemoryMax=infinity /bin/bash -c "exec $L/serve_isaac.sh $TAG > $L/mcp_server_$TAG.log 2>&1"
for i in $(seq 1 90); do ss -ltn | grep -q "172.17.0.1:18740 " && break; sleep 2; done
ss -ltn | grep -q "172.17.0.1:18740 " && echo "READY $TAG" || echo "mcp not ready"
