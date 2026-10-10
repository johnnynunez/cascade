#!/usr/bin/env bash
# B36 wave-8 live: one run = guard + fresh Isaac stage + fresh stdio MCP server from <rig> + the B36 task.
# usage: run.sh <rig: w8main|w8head> <tag>   (ENGINE=physx|newton)
W8=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/w8
RIGN=${1:?rig}; TAG=${2:?tag}
RIG=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/rig_$RIGN
bash $W8/bridge.sh "$TAG" || { rc=$?; echo "RIG_FAILED $TAG rc=$rc"; systemctl --user stop hermes-w8-b36-bridge 2>/dev/null; exit $rc; }
RUN=$W8/runs/run_$TAG
rm -rf "$RUN"; mkdir -p "$RUN"
timeout 1200 /home/johnny/Projects/demo/cascade/.venv/bin/python $W8/probe.py --rig $RIG --run-dir $RUN \
  --server-cwd $W8/models --server-log $W8/logs/mcp_$TAG.log > $W8/runs/$TAG.json 2> $W8/logs/probe_$TAG.err
rc=$?
echo "PROBE_DONE $TAG rig=$RIGN engine=${ENGINE:-physx} rc=$rc $(date -Is)"
systemctl --user stop hermes-w8-b36-bridge 2>/dev/null
systemctl --user reset-failed hermes-w8-b36-bridge 2>/dev/null
exit $rc
