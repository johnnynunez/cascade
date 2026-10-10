#!/usr/bin/env bash
# B36 A/B #3: same as A/B #2 but Isaac Sim + Newton (MJWarp). M = main 133876c vs
# H = branch (recovery + bounded hold). Fresh stage + server per run, interleaved.
LV=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live
while ! grep -q AB2_END $LV/turns/ab2.log 2>/dev/null; do sleep 20; done
# re-sync the branch rig (guards added after A/B #2 started; behaviour-neutral live)
rsync -a --exclude .git --exclude runs --exclude logs --exclude __pycache__ --exclude configs \
  $LV/../cascade/src/ $LV/rig_hold/src/
export ENGINE=newton
for i in 1 2 3 4; do
  for arm in main hold; do
    tag=nw${arm}$i
    echo "== $tag $(date -Is)" >> $LV/turns/ab3.log
    bash $LV/fresh_rig.sh $arm $tag >> $LV/turns/ab3.log 2>&1 || { echo "rig failed $tag" >> $LV/turns/ab3.log; continue; }
    (cd $LV && timeout 1200 python3 probe_host.py get_observation \
       'pick_and_place:{"object":"pink cube","destination":"drop zone","material":"rigid"}' get_observation 'reset_scene' \
       > $LV/turns/$tag.txt 2>&1)
    echo "done $tag rc=$? $(date -Is)" >> $LV/turns/ab3.log
  done
done
systemctl --user stop hermes-b36-mcp 2>/dev/null
echo "AB3_END $(date -Is)" >> $LV/turns/ab3.log
