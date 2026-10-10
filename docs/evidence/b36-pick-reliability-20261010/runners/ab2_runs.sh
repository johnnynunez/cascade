#!/usr/bin/env bash
# B36 A/B #2 on Isaac (PhysX, bare reBot scene): M = origin/main 133876c vs
# H = branch (recovery + bounded post-contact hold). Fresh stage + fresh server per
# run, interleaved M H, same task. Starts after E1.
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
while ! grep -q E1_END $LV/turns/e1.log 2>/dev/null; do sleep 20; done
for i in 1 2 3 4 5 6; do
  for arm in main hold; do
    tag=ab2${arm}$i
    echo "== $tag $(date -Is)" >> $LV/turns/ab2.log
    bash $LV/fresh_rig.sh $arm $tag >> $LV/turns/ab2.log 2>&1 || { echo "rig failed $tag" >> $LV/turns/ab2.log; continue; }
    (cd $LV && timeout 900 python3 probe_host.py get_observation \
       'pick_and_place:{"object":"pink cube","destination":"drop zone","material":"rigid"}' get_observation 'reset_scene' \
       > $LV/turns/$tag.txt 2>&1)
    echo "done $tag rc=$? $(date -Is)" >> $LV/turns/ab2.log
  done
done
systemctl --user stop hermes-b36-mcp 2>/dev/null
echo "AB2_END $(date -Is)" >> $LV/turns/ab2.log
