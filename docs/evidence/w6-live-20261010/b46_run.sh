#!/usr/bin/env bash
# live-w6 B46: one fresh Isaac stage + the REAL MCP server over stdio (spawned by the probe), lane on|off.
# usage: b46_run.sh <lane 0|1> <tag> [stop_after_s]
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
WT=$LV/cascade
LANE=${1:?lane}; TAG=${2:?tag}; STOP=${3:-}
GPU0=GPU-87c0fe81-bcfb-b636-9eec-e56fd88c33c3
bash $LV/scratch/bridge.sh "$TAG" || { rc=$?; echo "RIG_FAILED $TAG rc=$rc"; exit $rc; }
RUN=$LV/b46/run_$TAG
rm -rf "$RUN"; mkdir -p "$RUN"
EXTRA=()
[[ -n "$STOP" ]] && EXTRA+=(--stop-after "$STOP")
env -u PYTHONPATH -u CASCADE_MCP_TOKEN -u CASCADE_HUG_PORT -u CASCADE_STREAM_PORT -u CASCADE_VLA_PORT \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU0 \
  CASCADE_BRIDGE_PORT=47000 CASCADE_GRASPGENX_PORT=47001 CASCADE_OCCUPANCY_PORT=47002 CASCADE_OCCUPANCY=0 \
  CASCADE_DETECTOR_MODEL=$WT/models/yoloe-11s-seg.pt YOLO_OFFLINE=True ULTRALYTICS_OFFLINE=True CASCADE_VIEW=0 \
  CASCADE_BELIEFS_PATH=$RUN/beliefs.json CASCADE_GRASP_MEMORY_PATH=$RUN/grasp.json \
  CASCADE_ENVELOPE_PATH=$RUN/envelope.json CASCADE_GRASP_EVIDENCE_DIR=$RUN/grasp_evidence \
  timeout 900 /home/johnny/Projects/demo/cascade/.venv/bin/python $LV/scratch/live_lane_probe.py \
    --repo $WT --python /home/johnny/Projects/demo/cascade/.venv/bin/python --server-cwd $WT/models \
    --cameras isaac,isaac_side --arm isaac --object "pink cube" --destination "drop zone" --lane "$LANE" \
    --run-dir "$RUN" --timeout 600 --server-log $LV/logs/mcp_$TAG.log --raw-out $LV/b46/${TAG}_raw.json \
    "${EXTRA[@]}" > $LV/b46/$TAG.json 2> $LV/logs/probe_$TAG.err
rc=$?
echo "PROBE_DONE $TAG lane=$LANE stop=${STOP:-none} rc=$rc $(date -Is)"
systemctl --user stop hermes-live-w6-bridge 2>/dev/null
exit $rc
