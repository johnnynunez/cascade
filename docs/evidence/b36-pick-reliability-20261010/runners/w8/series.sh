#!/usr/bin/env bash
# B36 wave-8 live series (bounded): main 38f6d08 (w8main) vs the merged branch head (w8head),
# fresh stage + fresh server per run, interleaved, PhysX x4 per arm; then Newton x2 on the head
# (the hold must NOT apply on Newton). GraspGen-X warm on 47702 throughout. A user job stops it.
W8=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/w8
LOG=$W8/logs/series.log
run() {  # engine rig tag
  export ENGINE=$1
  echo "== $3 $2 $1 $(date -Is)" >> $LOG
  bash $W8/run.sh "$2" "$3" >> $LOG 2>&1
  rc=$?
  if [[ $rc == 3 ]]; then echo "GUARD_STOP $3 $(date -Is)" >> $LOG; systemctl --user stop hermes-w8-b36-ggx; exit 3; fi
}
echo "SERIES_START $(date -Is)" >> $LOG
bash $W8/ggx.sh >> $LOG 2>&1 || { echo "GGX_FAILED $(date -Is)" >> $LOG; exit 1; }
for i in 1 2 3 4; do run physx w8main pm$i; run physx w8head ph$i; done
for i in 1 2; do run newton w8head nh$i; done
systemctl --user stop hermes-w8-b36-bridge hermes-w8-b36-ggx 2>/dev/null
echo "SERIES_END $(date -Is)" >> $LOG
