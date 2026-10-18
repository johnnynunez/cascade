#!/usr/bin/env bash
# GPU window 3 for PR perf/bolt-default-fixes (throwaway): the brain ALONE, no Isaac (controller rule after three hard
# host resets with a brain beside Isaac). Runs as systemd user unit bf-driver3 while the agent holds the shared
# brain-gpu lock. Each brain = a `git archive` snapshot (deploy scripts src) served by its own
# scripts/serve_nemotron_bolt_llamacpp.sh on :18499 with a scratch key.
#   1. main 3aa0a30 (before): bench_prefix2.py (the comparison's fixed new-chat bench, COLD=1), bench_newchat.py
#      (3 new-chat trials, 4 same-chat replays)
#   2. head (after): the same two benches, then perception with the production request (perc_prod.py: inventory, refer)
#   3. main again: bench_newchat.py only (order check for the same-chat follow-ups)
# After every step: brain alive? kernel Xid? (the watcher unit stops everything at the first Xid.)
set -u
W=/home/spark/Projects/perception-bakeoff/out/bolt-default-fixes-2026-10-18
R=/home/spark/Projects/perception-bakeoff
S=/home/spark/.hermes/cache/scratch/boltfix
SNAP_MAIN=$1; SNAP_HEAD=$2; TAG=$3
PY=$R/.venv/bin/python
PORT=18499
KEY=$S/secrets/brain.key
ST=$W/raw/status_w3.txt
U=bf-brain3; UM=bf-memwatch3; UX=bf-xidwatch3; UD=bf-driver3
T0=$(date +%s)
mkdir -p $W/raw $W/logs
log() { echo "$(date +%T) $*" | tee -a "$ST"; }
avail_kb() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
export BRAIN_URL=http://127.0.0.1:$PORT/v1 BRAIN_MODEL=nemotron-3.5-bolt BRAIN_KEY_FILE=$KEY
export TMPDIR=$S/tmp CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1

cleanup() {
  systemctl --user stop "$U" 2>/dev/null; systemctl --user stop "$UM" 2>/dev/null
  log "cleanup: units stopped ($U $UM)"
}
trap cleanup EXIT
xid_check() {
  local x; x=$(journalctl -k --since "@$T0" --no-pager 2>/dev/null | grep -E "NVRM: Xid" || true)
  if [ -n "$x" ]; then echo "$x" > $W/raw/xid_w3.txt; log "XID during $1 -> STOP"; exit 4; fi
}
alive() { systemctl --user is-active -q "$U" || { log "BRAIN DIED during $1"; exit 5; }; xid_check "$1"; }
step() {  # step <name> <timeout s> <cmd...>
  local name=$1 to=$2; shift 2
  log "START $name"; local t=$(date +%s)
  if timeout "$to" "$@" > $W/logs/w3-$name.log 2>&1; then log "OK $name ($(( $(date +%s) - t )) s)"
  else log "FAIL $name rc=$? ($(( $(date +%s) - t )) s; logs/w3-$name.log)"; fi
  alive "$name"
}
health() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health || true)" = 200 ] &&
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 -H "Authorization: Bearer $(cat $KEY)" http://127.0.0.1:$PORT/v1/models || true)" = 200 ]
}
brain_up() {  # brain_up <label> <snapshot dir>
  local label=$1 snap=$2
  local t=$(date +%s.%N)
  systemd-run --user --collect --quiet --unit "$U" -p MemoryMax=infinity -p WorkingDirectory=$snap \
    -p StandardOutput=truncate:$W/logs/brain-w3-$label.log -p StandardError=inherit \
    -E PATH=/home/spark/.local/bin:/usr/local/bin:/usr/bin:/bin -E HOME=/home/spark -E TMPDIR=$S/tmp \
    -E PORT=$PORT -E API_KEY_FILE=$KEY -E MODEL_ROOT=$S/gguf-root \
    -E LLAMA_DIR=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0 \
    -E LLAMA_SERVER=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0/build/bin/llama-server \
    bash $snap/scripts/serve_nemotron_bolt_llamacpp.sh || { log "FAIL systemd-run brain $label"; exit 1; }
  systemd-run --user --collect --quiet --unit "$UM" bash $R/out/brain-comparison-2026-10-18/scripts/memwatch_cmp.sh "$U" $W/logs/memwatch-w3-$label.log 3 || true
  for i in $(seq 1 450); do health && break; systemctl --user is-active -q "$U" || break; sleep 2; done
  health || { log "BRAIN $label NEVER READY"; exit 3; }
  log "READY $label ($(cat $snap/.commit)) after $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t}") s; MemAvailable $(( $(avail_kb) / 1048576 )) GiB; load $(cut -d' ' -f1-3 /proc/loadavg)"
  log "argv $label: $(pgrep -a llama-server | grep -o -- '--chat-template-file [^ ]*\|--parallel [0-9]*\|--load-mode [a-z]*\|--spec-type [a-z-]*' | tr '\n' ' ')"
  sleep 5
}
brain_down() {
  systemctl --user stop "$U"; systemctl --user stop "$UM" 2>/dev/null
  for i in $(seq 1 60); do systemctl --user is-active -q "$U" || break; sleep 2; done
  sleep 10; log "brain stopped; MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
}

: > "$ST"
log "WINDOW 3 (alone, no Isaac) main $SNAP_MAIN head $SNAP_HEAD; load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
ld=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($ld >= 8)}" && { log "REFUSE: load $ld >= 8"; exit 2; }
systemd-run --user --collect --quiet --unit "$UX" bash $W/scripts/xidwatch_bf.sh "$ST" "$T0" "$UD" "$U" || true

brain_up main $SNAP_MAIN
step prefix2_main 900 env COLD=1 $PY $R/out/brain-comparison-2026-10-18/scripts/bench_prefix2.py $W/raw/w3_prefix2_main.json
step newchat_main 900 $PY $W/scripts/bench_newchat.py $W/raw/w3_newchat_main.json 3 4
brain_down

brain_up head $SNAP_HEAD
step prefix2_head 900 env COLD=1 $PY $R/out/brain-comparison-2026-10-18/scripts/bench_prefix2.py $W/raw/w3_prefix2_head.json
step newchat_head 900 $PY $W/scripts/bench_newchat.py $W/raw/w3_newchat_head.json 3 4
step perc_${TAG} 1800 env PYTHONPATH=$SNAP_HEAD/src nice -n 10 $PY $W/scripts/perc_prod.py $TAG inventory
step refer_${TAG} 1800 env PYTHONPATH=$SNAP_HEAD/src nice -n 10 $PY $W/scripts/perc_prod.py $TAG refer
brain_down

brain_up main2 $SNAP_MAIN
step newchat_main2 900 $PY $W/scripts/bench_newchat.py $W/raw/w3_newchat_main2.json 3 4
xid_check end
brain_down
log "WINDOW 3 DONE"
