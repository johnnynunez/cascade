#!/usr/bin/env bash
# B35 live proof: pick then reset, same chat session, agent inside the OpenShell sandbox.
export PATH=/home/johnny/.nvm/versions/node/v26.5.0/bin:/home/johnny/.local/bin:$PATH
export NEMOCLAW_GATEWAY_PORT=18750
L=/home/johnny/Projects/demo/cascade-lab/HERMES_BACKLOG_20261007/nemoclaw-runtime/live
S=${1:-b35-proof-2}
run_turn() {  # name message
  local t0; t0=$(date +%s.%N)
  timeout 900 nemoclaw cascade-robot agent --session-id "$S" --json -m "$2" > "$L/turns/$1.json" 2> "$L/turns/$1.err"
  echo "$1 rc=$? wall_s=$(python3 -c "import time;print(round(time.time()-$t0,1))")" >> "$L/turns/proof.log"
}
echo "session=$S start=$(date -Is)" >> "$L/turns/proof.log"
run_turn ${S}_t3_pick "Pick up the pink cube and place it in the box (the drop zone) using the cascade tools. Report whether the tool result says the pick and the place were confirmed."
run_turn ${S}_t4_reset "Now reset the scene to its starting state with the cascade tools, then read world_state and tell me where the pink cube is."
echo "end=$(date -Is)" >> "$L/turns/proof.log"
