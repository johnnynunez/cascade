#!/usr/bin/env bash
# B36 A/B #5 (9 Oct): after A/B #4. PhysX only. r2 = recovery only (no hold) vs
# h15 = recovery + hold {physx: 0.15}; fresh stage + server per run, interleaved.
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
LOG=$LV/turns/ab5.log
while ! grep -q AB4_END $LV/turns/ab4.log 2>/dev/null; do sleep 20; done
run() {  # engine rig tag
  export ENGINE=$1
  echo "== $3 $1 $(date -Is)" >> $LOG
  bash $LV/fresh_rig2.sh $2 $3 >> $LOG 2>&1 || { echo "rig failed $3" >> $LOG; return; }
  (cd $LV && timeout 900 python3 probe_host.py get_observation \
     'pick_and_place:{"object":"pink cube","destination":"drop zone","material":"rigid"}' get_observation 'reset_scene' \
     > $LV/turns/$3.txt 2>&1)
  echo "done $3 rc=$? $(date -Is)" >> $LOG
}
for i in 1 2 3 4 5 6; do run physx r2 ab5pr$i; run physx h15 ab5ph$i; done
systemctl --user stop hermes-b36-mcp hermes-b36-isaac-bridge 2>/dev/null
echo "AB5_END $(date -Is)" >> $LOG
