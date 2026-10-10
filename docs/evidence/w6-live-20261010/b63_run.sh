#!/usr/bin/env bash
# live-w6 B63/B44: fresh Isaac stage, optional VLA stub on :47010, launcher-style registration + turn.
# usage: b63_run.sh <tag> <vla|analytic> [proof]
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
WT=$LV/cascade
TAG=${1:?tag}; ARM=${2:?vla|analytic}; PROOF=${3:-}
bash $LV/scratch/bridge.sh "$TAG" || { rc=$?; echo "RIG_FAILED $TAG rc=$rc"; exit $rc; }
EXTRA=()
if [[ $ARM == vla ]]; then
  systemctl --user stop hermes-live-w6-vla 2>/dev/null; systemctl --user reset-failed hermes-live-w6-vla 2>/dev/null
  systemd-run --user --unit=hermes-live-w6-vla -p MemoryMax=infinity /bin/bash -c \
    "exec env -u PYTHONPATH PYTHONPATH=$WT/src CUDA_VISIBLE_DEVICES=-1 /home/johnny/Projects/demo/cascade/.venv/bin/python $WT/scripts/serve_vla_stub.py --port 47010 --chunks $LV/scratch/vla_chunks.json > $LV/logs/vla_stub_$TAG.log 2>&1"
  for i in $(seq 1 30); do ss -ltn | grep -q "127.0.0.1:47010 " && break; sleep 1; done
  ss -ltn | grep -q "127.0.0.1:47010 " || { echo "vla stub not listening"; exit 1; }
  echo "VLA STUB READY $(cat $LV/logs/vla_stub_$TAG.log)"
  EXTRA+=(--vla-port 47010)
fi
[[ $PROOF == proof ]] && EXTRA+=(--proof)
env -u PYTHONPATH PYTHONPATH=$WT/src timeout 1500 /home/johnny/Projects/demo/cascade/.venv/bin/python \
  $LV/scratch/launcher_turn.py --tag "$TAG" "${EXTRA[@]}" > $LV/logs/turn_$TAG.out 2> $LV/logs/turn_$TAG.err
rc=$?
echo "TURN_DONE $TAG arm=$ARM proof=${PROOF:-no} rc=$rc $(date -Is)"
systemctl --user stop hermes-live-w6-vla 2>/dev/null
systemctl --user stop hermes-live-w6-bridge 2>/dev/null
exit $rc
