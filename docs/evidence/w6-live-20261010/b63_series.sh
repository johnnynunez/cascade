#!/usr/bin/env bash
# live-w6 B63 + B44: vla arm (entry with CASCADE_GRASP_EXECUTOR=vla + stub), then analytic proof turns
# (entry without it) until one receipt is physics-verified (max 3).
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
LOG=$LV/logs/series_b63.log
run() {
  echo "== $1 $2 ${3:-} $(date -Is)" >> $LOG
  bash $LV/scratch/b63_run.sh "$@" >> $LOG 2>&1
  rc=$?
  if [[ $rc == 3 ]]; then echo "GUARD_STOP $1 $(date -Is)" >> $LOG; exit 3; fi
}
run vla1 vla
for i in 1 2 3; do
  run an$i analytic proof
  v=$(python3 -c "import json;print(json.load(open('$LV/b63/an$i/summary.json')).get('proof',{}).get('verified'))" 2>/dev/null)
  echo "an$i verified=$v" >> $LOG
  [[ "$v" == "True" ]] && break
done
systemctl --user stop hermes-live-w6-bridge hermes-live-w6-vla 2>/dev/null
echo "B63_SERIES_END $(date -Is)" >> $LOG
