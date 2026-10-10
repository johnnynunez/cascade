#!/usr/bin/env bash
# B36 live A/B: CASCADE MCP server (HTTPS, host-side probe) from one arm's rig.
set -uo pipefail
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
ARM=${1:?arm}; TAG=${2:?tag}
RIG=$LV/rig_$ARM; RUN=$LV/run_$TAG; D=$HOME/.cascade/nemoclaw
mkdir -p "$RUN"
cd /home/johnny/Projects/demo/cascade/models
exec env -u PYTHONPATH -u CASCADE_MCP_TOKEN -u CASCADE_OCCUPANCY_PORT -u CASCADE_BRIDGE_PORT \
  PYTHONPATH="$RIG/src" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3 \
  CASCADE_ARM=isaac CASCADE_CAMERAS=isaac,isaac_side CASCADE_GRASPGENX_PORT=18522 \
  CASCADE_DETECTOR_MODEL=/home/johnny/Projects/demo/cascade/models/yoloe-11s-seg.pt \
  YOLO_OFFLINE=True ULTRALYTICS_OFFLINE=True CASCADE_OCCUPANCY=0 CASCADE_VIEW=0 CASCADE_STREAM=0 \
  CASCADE_RUN_DIR="$RUN" CASCADE_BELIEFS_PATH="$RUN/beliefs.json" CASCADE_GRASP_MEMORY_PATH="$RUN/grasp.json" \
  CASCADE_ENVELOPE_PATH="$RUN/envelope.json" CASCADE_GRASP_EVIDENCE_DIR="$RUN/grasp_evidence" \
  /home/johnny/Projects/demo/cascade/.venv/bin/python -m cascade.apps.mcp_server \
  --http "$(python3 -c "import json;print(json.load(open('$D/endpoint.json'))['host'])"):18740" \
  --tls-cert "$D/server.pem" --tls-key "$D/server.key" --token-file "$D/token"
