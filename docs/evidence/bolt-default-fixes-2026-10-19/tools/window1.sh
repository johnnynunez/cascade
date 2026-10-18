#!/usr/bin/env bash
# GPU window 1 for PR perf/bolt-default-fixes (throwaway). Runs as systemd user unit bf-driver while the agent holds
# the shared brain-gpu lock. Bolt = a `git archive` snapshot of the branch (serve script + chat template) on :18499
# with a scratch key; Isaac (NEWTON) on :8641. Ports avoid every live / other-agent port.
#   1. S1 = the branch's default (cascade chat template): new-chat bench alone, perception variants (inventory),
#      refer for the schema variant, the 62-order tool eval
#   2. S0 = the same script with CHAT_TEMPLATE_FILE= (main's behaviour): new-chat bench alone
#   3. Isaac Newton up: new-chat bench beside Isaac for S0, then S1
# After every step: brain alive? kernel Xid? (the watcher unit stops everything at the first Xid.)
set -u
W=/home/spark/Projects/perception-bakeoff/out/bolt-default-fixes-2026-10-18
R=/home/spark/Projects/perception-bakeoff
S=/home/spark/.hermes/cache/scratch/boltfix
SNAP=$1
PY=$R/.venv/bin/python
PORT=18499
KEY=$S/secrets/brain.key
ST=$W/raw/status_w1.txt
U=bf-brain; UI=bf-isaac; UM=bf-memwatch; UX=bf-xidwatch; UD=bf-driver
T0=$(date +%s)
mkdir -p $W/raw $W/logs
log() { echo "$(date +%T) $*" | tee -a "$ST"; }
avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
export BRAIN_URL=http://127.0.0.1:$PORT/v1 BRAIN_MODEL=nemotron-3.5-bolt BRAIN_KEY_FILE=$KEY
export TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES=

cleanup() {
  systemctl --user stop "$UI" 2>/dev/null; systemctl --user stop "$U" 2>/dev/null; systemctl --user stop "$UM" 2>/dev/null
  log "cleanup: units stopped ($U $UI $UM)"
}
trap cleanup EXIT
xid_check() {
  local x; x=$(journalctl -k --since "@$T0" --no-pager 2>/dev/null | grep -E "NVRM: Xid" || true)
  if [ -n "$x" ]; then echo "$x" > $W/raw/xid_w1.txt; log "XID during $1 -> STOP"; exit 4; fi
}
alive() { systemctl --user is-active -q "$U" || { log "BRAIN DIED during $1"; exit 5; }; xid_check "$1"; }
step() {  # step <name> <timeout s> <cmd...>
  local name=$1 to=$2; shift 2
  log "START $name"; local t=$(date +%s)
  if timeout "$to" "$@" > $W/logs/w1-$name.log 2>&1; then log "OK $name ($(( $(date +%s) - t )) s)"
  else log "FAIL $name rc=$? ($(( $(date +%s) - t )) s; logs/w1-$name.log)"; fi
  alive "$name"
}
health() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health || true)" = 200 ] &&
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 -H "Authorization: Bearer $(cat $KEY)" http://127.0.0.1:$PORT/v1/models || true)" = 200 ]
}
brain_up() {  # brain_up <label> <CHAT_TEMPLATE_FILE value or __default__>
  local label=$1 tmpl=$2 envt=()
  [ "$tmpl" = __default__ ] || envt=(-E "CHAT_TEMPLATE_FILE=$tmpl")
  local t=$(date +%s.%N)
  systemd-run --user --collect --quiet --unit "$U" -p MemoryMax=infinity -p WorkingDirectory=$SNAP \
    -p StandardOutput=truncate:$W/logs/brain-$label.log -p StandardError=inherit \
    -E PATH=/home/spark/.local/bin:/usr/local/bin:/usr/bin:/bin -E HOME=/home/spark -E TMPDIR=$S/tmp \
    -E PORT=$PORT -E API_KEY_FILE=$KEY -E MODEL_ROOT=$S/gguf-root "${envt[@]}" \
    -E LLAMA_DIR=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0 \
    -E LLAMA_SERVER=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0/build/bin/llama-server \
    bash $SNAP/scripts/serve_nemotron_bolt_llamacpp.sh || { log "FAIL systemd-run brain $label"; exit 1; }
  systemd-run --user --collect --quiet --unit "$UM" bash $R/out/brain-comparison-2026-10-18/scripts/memwatch_cmp.sh "$U" $W/logs/memwatch-$label.log 3 || true
  for i in $(seq 1 450); do health && break; systemctl --user is-active -q "$U" || break; sleep 2; done
  health || { log "BRAIN $label NEVER READY"; exit 3; }
  log "READY $label after $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t}") s; MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
  sleep 5
}
brain_down() {
  systemctl --user stop "$U"; systemctl --user stop "$UM" 2>/dev/null
  for i in $(seq 1 60); do systemctl --user is-active -q "$U" || break; sleep 2; done
  sleep 10; log "brain stopped; MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
}

: > "$ST"
log "WINDOW 1 snap $SNAP ($(cat $SNAP/.commit 2>/dev/null)); load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
ld=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($ld >= 8)}" && { log "REFUSE: load $ld >= 8"; exit 2; }
systemd-run --user --collect --quiet --unit "$UX" bash $W/scripts/xidwatch_bf.sh "$ST" "$T0" "$UD" "$UI" "$U" || true

brain_up s1 __default__
step newchat_s1_alone 900 $PY $W/scripts/bench_newchat.py $W/raw/newchat_s1_alone.json 3
for v in v0 js20 js20dry js20drynb; do
  step perc_$v 1800 nice -n 10 $PY $W/scripts/perc_run.py $v inventory
done
step refer_js20 1800 nice -n 10 $PY $W/scripts/perc_run.py js20 refer
step tool_eval_s1 1800 env TAG=boltfix_s1 nice -n 10 $PY $R/agent_eval/tool_eval.py
brain_down

brain_up s0 ""
step newchat_s0_alone 900 $PY $W/scripts/bench_newchat.py $W/raw/newchat_s0_alone.json 3

a=$(avail_kb); ld=$(cut -d' ' -f1 /proc/loadavg)
if [ "$a" -lt $(( 28 * 1048576 )) ]; then log "SKIP isaac: MemAvailable $(( a / 1048576 )) GiB < 28"
elif ! python3 /home/spark/Projects/isaac/newton_prebundle.py 2>&1 | tail -1 | grep -q "NEWTON_PREBUNDLE CONSISTENT"; then log "SKIP isaac: prebundle not CONSISTENT"
elif awk "BEGIN{exit !($ld >= 8)}"; then log "SKIP isaac: load $ld >= 8"
else
  log "START isaac NEWTON (MemAvailable $(( a / 1048576 )) GiB, load $ld; prebundle CONSISTENT)"
  systemd-run --user --collect --quiet --unit "$UI" -p MemoryMax=infinity -p WorkingDirectory=$W \
    -p StandardOutput=truncate:$W/logs/isaac-w1.log -p StandardError=inherit \
    /home/spark/.hermes/cache/scratch/bolt-pr/clean.sh bash $W/scripts/run_isaac_bf.sh
  up=0
  for i in $(seq 1 120); do
    grep -q "serving cascade bridge on :8641" $W/logs/isaac-w1.log 2>/dev/null && { up=1; break; }
    systemctl --user is-active -q "$UI" || break; sleep 5
  done
  if [ $up = 1 ] && sleep 20 && /home/spark/Projects/cascade-wt/demo-video/.venv/bin/python $W/scripts/bridge_ping_bf.py $W/raw/isaac_ping_w1.json >> $ST 2>&1; then
    log "isaac up, engine newton; MemAvailable $(( $(avail_kb) / 1048576 )) GiB; load $(cut -d' ' -f1 /proc/loadavg)"
    sleep 30
    step newchat_s0_isaac 1200 $PY $W/scripts/bench_newchat.py $W/raw/newchat_s0_isaac.json 3
    brain_down
    brain_up s1b __default__
    step newchat_s1_isaac 1200 $PY $W/scripts/bench_newchat.py $W/raw/newchat_s1_isaac.json 3
    /home/spark/Projects/cascade-wt/demo-video/.venv/bin/python $W/scripts/bridge_ping_bf.py $W/raw/isaac_ping_w1_end.json >> $ST 2>&1 \
      && log "isaac still newton at the end" || log "WARN isaac ping failed at the end"
  else
    log "FAIL isaac never served / not newton"
  fi
  systemctl --user stop "$UI"; sleep 10; log "isaac stopped; MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
fi
xid_check end
brain_down
log "WINDOW 1 DONE"
