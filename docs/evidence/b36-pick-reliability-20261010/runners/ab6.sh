#!/usr/bin/env bash
# B36 A/B #6 (9 Oct, learned GraspGen-X warm on :18522 as in A/B #2): M = origin/main 4e896c3 (rig_m2) vs F = final branch (rig_f2:
# recovery + PhysX-only post-contact hold). Fresh stage + server per run, interleaved,
# same task as A/B #2/#3. PhysX x6 per arm, then Newton x4 per arm.
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
LOG=$LV/turns/ab6.log
run() {  # engine rig tag
  export ENGINE=$1
  echo "== $3 $1 $(date -Is)" >> $LOG
  bash $LV/fresh_rig2.sh $2 $3 >> $LOG 2>&1 || { echo "rig failed $3" >> $LOG; return; }
  (cd $LV && timeout 900 python3 probe_host.py get_observation \
     'pick_and_place:{"object":"pink cube","destination":"drop zone","material":"rigid"}' get_observation 'reset_scene' \
     > $LV/turns/$3.txt 2>&1)
  echo "done $3 rc=$? $(date -Is)" >> $LOG
}
for i in 1 2 3 4 5 6; do run physx m2 ab6pm$i; run physx f2 ab6pf$i; done
for i in 1 2 3 4; do run newton m2 ab6nm$i; run newton f2 ab6nf$i; done
systemctl --user stop hermes-b36-mcp hermes-b36-isaac-bridge 2>/dev/null
echo "AB6_END $(date -Is)" >> $LOG
