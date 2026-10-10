#!/usr/bin/env bash
# live-w6 series 2: vla arm with the request-logging stub (same chunks), then analytic launcher proof turns with
# per-run memory stores until one receipt is physics-verified (max 5).
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261009/live-w6
LOG=$LV/logs/series_b63_2.log
run() {
  echo "== $* $(date -Is)" >> $LOG
  bash $LV/scratch/b63_run2.sh "$@" >> $LOG 2>&1
  rc=$?
  if [[ $rc == 3 ]]; then echo "GUARD_STOP $1 $(date -Is)" >> $LOG; exit 3; fi
}
run vla2 vla no fresh
for i in 4 5 6 7 8; do
  run an$i analytic proof fresh
  v=$(python3 -c "import json;print(json.load(open('$LV/b63/an$i/summary.json')).get('proof',{}).get('verified'))" 2>/dev/null)
  echo "an$i verified=$v" >> $LOG
  [[ "$v" == "True" ]] && break
done
systemctl --user stop hermes-live-w6-bridge hermes-live-w6-vla 2>/dev/null
echo "B63_SERIES2_END $(date -Is)" >> $LOG
