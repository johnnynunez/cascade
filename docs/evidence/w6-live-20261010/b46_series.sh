#!/usr/bin/env bash
# live-w6 B46 series: interleaved lane on/off (fresh stage + fresh server per run), then 1 stop run.
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
LOG=$LV/logs/series_b46.log
run() {  # lane tag [stop]
  echo "== $2 lane=$1 stop=${3:-none} $(date -Is)" >> $LOG
  bash $LV/scratch/b46_run.sh "$@" >> $LOG 2>&1
  rc=$?
  if [[ $rc == 3 ]]; then echo "GUARD_STOP $2 $(date -Is): user benchmark / foreign GPU-0 process -> series stopped" >> $LOG; exit 3; fi
}
for i in 1 2 3 4; do run 1 on$i; run 0 off$i; done
run 1 stop1 3
systemctl --user stop hermes-live-w6-bridge 2>/dev/null
echo "B46_SERIES_END $(date -Is)" >> $LOG
