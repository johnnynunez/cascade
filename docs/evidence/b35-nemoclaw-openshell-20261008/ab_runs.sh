#!/usr/bin/env bash
# B35 A/B on Isaac: host-direct MCP call vs sandboxed OpenClaw agent, same server
# code, fresh stage per run, alternating H S H S. Same pick, same scene.
L=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/nemoclaw-runtime/live
export PATH=/home/johnny/.nvm/versions/node/v26.5.0/bin:/home/johnny/.local/bin:$PATH
export NEMOCLAW_GATEWAY_PORT=18750
for tag in hostA2 sandboxB2 hostA3 sandboxB3; do
  echo "== $tag $(date -Is)" >> $L/turns/ab.log
  bash $L/fresh_rig.sh $tag >> $L/turns/ab.log 2>&1 || { echo "rig failed $tag" >> $L/turns/ab.log; continue; }
  case $tag in
    host*) (cd $L && timeout 600 python3 probe_host.py get_observation \
             'pick_and_place:{"object":"pink cube","destination":"drop zone","material":"rigid"}' 'reset_scene' \
             > $L/turns/$tag.txt 2>&1) ;;
    sandbox*) bash $L/proof_turns.sh b35-$tag >> $L/turns/ab.log 2>&1 ;;
  esac
  echo "done $tag $(date -Is)" >> $L/turns/ab.log
done
echo "AB_END $(date -Is)" >> $L/turns/ab.log
