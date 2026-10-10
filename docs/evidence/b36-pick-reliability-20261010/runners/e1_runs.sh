#!/usr/bin/env bash
# B36 E1: is the held wrist-roll deflection caused by the close squeeze?
# Same main code, same pick; material=fragile (close stage2 0.60) vs rigid (0.85).
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
while ! grep -q AB_END $LV/turns/ab.log 2>/dev/null; do sleep 20; done
for i in 1 2 3; do
  for mat in fragile rigid; do
    tag=e1${mat}$i
    echo "== $tag $(date -Is)" >> $LV/turns/e1.log
    bash $LV/fresh_rig.sh main $tag >> $LV/turns/e1.log 2>&1 || { echo "rig failed $tag" >> $LV/turns/e1.log; continue; }
    (cd $LV && timeout 900 python3 probe_host.py get_observation \
       "pick_and_place:{\"object\":\"pink cube\",\"destination\":\"drop zone\",\"material\":\"$mat\"}" get_observation 'reset_scene' \
       > $LV/turns/$tag.txt 2>&1)
    echo "done $tag rc=$? $(date -Is)" >> $LV/turns/e1.log
  done
done
systemctl --user stop hermes-b36-mcp 2>/dev/null
echo "E1_END $(date -Is)" >> $LV/turns/e1.log
