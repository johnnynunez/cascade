#!/usr/bin/env bash
# live-w6 B63/B44 (series 2): fresh Isaac stage, request-logging VLA stub on :47010 (vla arm), launcher-style
# registration + turn. usage: b63_run2.sh <tag> <vla|analytic> [proof|no] [fresh]
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
WT=$LV/cascade
TAG=${1:?tag}; ARM=${2:?arm}; PROOF=${3:-}; FRESH=${4:-}
bash $LV/scratch/bridge.sh "$TAG" || { rc=$?; echo "RIG_FAILED $TAG rc=$rc"; exit $rc; }
EXTRA=()
if [[ $ARM == vla ]]; then
  systemctl --user stop hermes-live-w6-vla 2>/dev/null; systemctl --user reset-failed hermes-live-w6-vla 2>/dev/null
  rm -f $LV/b63/vla_requests_$TAG.jsonl
  systemd-run --user --unit=hermes-live-w6-vla -p MemoryMax=infinity /bin/bash -c \
    "exec env -u PYTHONPATH PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES=-1 /home/johnny/Projects/demo/cascade/.venv/bin/python $LV/scratch/vla_stub_logged.py 47010 $LV/scratch/vla_chunks.json $LV/b63/vla_requests_$TAG.jsonl > $LV/logs/vla_stub_$TAG.log 2>&1"
  for i in $(seq 1 30); do ss -ltn | grep -q "127.0.0.1:47010 " && break; sleep 1; done
  ss -ltn | grep -q "127.0.0.1:47010 " || { echo "vla stub not listening"; exit 1; }
  echo "VLA STUB READY $(cat $LV/logs/vla_stub_$TAG.log)"
  EXTRA+=(--vla-port 47010)
fi
[[ $PROOF == proof ]] && EXTRA+=(--proof)
[[ $FRESH == fresh ]] && EXTRA+=(--fresh-memory)
env -u PYTHONPATH PYTHONPATH=$WT/src timeout 1500 /home/johnny/Projects/demo/cascade/.venv/bin/python \
  $LV/scratch/launcher_turn.py --tag "$TAG" "${EXTRA[@]}" > $LV/logs/turn_$TAG.out 2> $LV/logs/turn_$TAG.err
rc=$?
echo "TURN_DONE $TAG arm=$ARM proof=${PROOF:-no} fresh=${FRESH:-no} rc=$rc $(date -Is)"
systemctl --user stop hermes-live-w6-vla 2>/dev/null
systemctl --user stop hermes-live-w6-bridge 2>/dev/null
exit $rc
