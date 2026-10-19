#!/usr/bin/env bash
# GPU window for PR perf/bolt-warm-start (throwaway): the Bolt brain ALONE (no Isaac: controller rule after three hard
# host resets with a brain beside Isaac). Runs as systemd user unit bw-driver while the agent holds the shared
# brain-gpu lock. Each brain = a `git archive` snapshot (deploy scripts src) served by its own
# scripts/serve_nemotron_bolt_llamacpp.sh on :18499 with a scratch key, llama.cpp v0.7.0 (620d2225), --load-mode none.
#   main x1 (window 3; main x3 + bench_newchat.py ran in window 2): restart, the FIRST chat after the start
#   head:    a start with no record (the first chat is cold and gets recorded), then x3: restart, warm-up, /slots,
#            the FIRST chat after the start; the third start then bench_newchat.py; then a chat sent at readiness
#            (during the warm-up), then 12 perception inventories before the first chat.
# After every step: brain alive? kernel Xid? (the watcher unit stops everything at the first Xid.)
set -u
S=/home/spark/.hermes/cache/scratch/boltwarm
R=/home/spark/Projects/perception-bakeoff
SNAP_MAIN=$1; SNAP_HEAD=$2
PY=$R/.venv/bin/python
PORT=18499
KEY=$S/secrets/brain.key
PREFIX=$S/state/prefix-$PORT.json
ST=$S/raw/status.txt
U=bw-brain; UM=bw-memwatch; UX=bw-xidwatch; UD=bw-driver
T0=$(date +%s)
mkdir -p $S/raw $S/logs $S/state
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
  if [ -n "$x" ]; then echo "$x" > $S/raw/xid.txt; log "XID during $1 -> STOP"; exit 4; fi
}
alive() { systemctl --user is-active -q "$U" || { log "BRAIN DIED during $1"; exit 5; }; xid_check "$1"; }
step() {  # step <name> <timeout s> <cmd...>
  local name=$1 to=$2; shift 2
  log "START $name"; local t=$(date +%s)
  if timeout "$to" "$@" > $S/logs/step-$name.log 2>&1; then log "OK $name ($(( $(date +%s) - t )) s): $(tail -c 600 $S/logs/step-$name.log | tr '\n' ' ' | cut -c1-600)"
  else log "FAIL $name rc=$? ($(( $(date +%s) - t )) s; logs/step-$name.log)"; fi
  alive "$name"
}
health() {
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 http://127.0.0.1:$PORT/health || true)" = 200 ] &&
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 -H "Authorization: Bearer $(cat $KEY)" http://127.0.0.1:$PORT/v1/models || true)" = 200 ]
}
brain_up() {  # brain_up <label> <snapshot dir> [poll s]
  local label=$1 snap=$2 poll=${3:-1} i
  local t=$(date +%s.%N)
  systemd-run --user --collect --quiet --unit "$U" -p MemoryMax=infinity -p WorkingDirectory=$snap \
    -p StandardOutput=truncate:$S/logs/brain-$label.log -p StandardError=inherit \
    -E PATH=/home/spark/.local/bin:/usr/local/bin:/usr/bin:/bin -E HOME=/home/spark -E TMPDIR=$S/tmp \
    -E PORT=$PORT -E API_KEY_FILE=$KEY -E MODEL_ROOT=$S/gguf-root -E WARM_PREFIX_FILE=$PREFIX \
    -E LLAMA_DIR=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0 \
    -E LLAMA_SERVER=/home/spark/.cache/cascade-bolt-gguf/llama.cpp-0.7.0/build/bin/llama-server \
    bash $snap/scripts/serve_nemotron_bolt_llamacpp.sh || { log "FAIL systemd-run brain $label"; exit 1; }
  systemd-run --user --collect --quiet --unit "$UM" bash $S/tools/memwatch_cmp.sh "$U" $S/logs/memwatch-$label.log 3 || true
  for i in $(seq 1 3000); do health && break; systemctl --user is-active -q "$U" || break; sleep "$poll"; done
  health || { log "BRAIN $label NEVER READY"; exit 3; }
  log "READY $label ($(cat $snap/.commit)) after $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t}") s; MemAvailable $(( $(avail_kb) / 1048576 )) GiB; load $(cut -d' ' -f1-3 /proc/loadavg)"
}
wait_helper() {  # wait_helper <label> <regex> <timeout s>: the warm-start helper's verdict in the brain log
  local label=$1 re=$2 to=$3 i
  for i in $(seq 1 $(( to * 2 ))); do grep -qE "$re" $S/logs/brain-$label.log && break; alive "wait_helper $label"; sleep 0.5; done
  log "helper $label: $(grep -F '[brain-warm]' $S/logs/brain-$label.log | tr '\n' '|' | cut -c1-700)"
}
brain_down() {
  local i
  systemctl --user stop "$U"; systemctl --user stop "$UM" 2>/dev/null
  for i in $(seq 1 60); do systemctl --user is-active -q "$U" || break; sleep 2; done
  sleep 8; log "brain stopped; MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
}
B="$PY $S/tools/bench_warm.py"

: > "$ST"
log "WINDOW (alone, no Isaac) main $SNAP_MAIN head $SNAP_HEAD; load $(cut -d' ' -f1-3 /proc/loadavg); MemAvailable $(( $(avail_kb) / 1048576 )) GiB"
ld=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($ld >= 8)}" && { log "REFUSE: load $ld >= 8"; exit 2; }
systemd-run --user --collect --quiet --unit "$UX" bash $S/tools/xidwatch_bf.sh "$ST" "$T0" "$UD" "$U" || true

# Window 6 (final head, first chats only; window 5 has main and the benches): main's first chat + bench_newchat.py, then head: a start with no record, a start whose
# first chat pairs two sessions, two starts with both warm-up steps, then bench_newchat.py (new chats, slot steal,
# follow-up replays) on a warmed head.
rec_wait() {  # rec_wait <label> <regex>: the helper's record line after the chats
  local n
  for n in $(seq 1 90); do grep -qE "$2" $S/logs/brain-$1.log && break; sleep 1; done
}
rm -f "$PREFIX"; rm -rf $S/state/bolt-slots-$PORT
brain_up rec $SNAP_HEAD
wait_helper rec 'no recorded OpenClaw prefix' 60
step first_rec 300 $B first $S/raw/first_head_rec.json
rec_wait rec 'recorded the latest|could not read'
log "helper rec: $(grep -F '[brain-warm]' $S/logs/brain-rec.log | tr '\n' '|' | cut -c1-700)"
[ -f "$PREFIX" ] && log "recorded $(stat -c '%s bytes, mode %a' $PREFIX); slot dir: $(ls -A $S/state/bolt-slots-$PORT | wc -l) files" || { log "NO RECORD"; exit 6; }
brain_down

brain_up head0 $SNAP_HEAD
wait_helper head0 'warmed|gave up|no recorded|another chat template|holds no|unreadable|unknown record' 200
step first_head0 300 $B first $S/raw/first_head0.json
rec_wait head0 'its sessions share|could not read'
log "helper head0: $(grep -F '[brain-warm]' $S/logs/brain-head0.log | tr '\n' '|' | cut -c1-900)"
brain_down

for n in 1 2; do
  brain_up head$n $SNAP_HEAD
  wait_helper head$n 'what the last OpenClaw sessions shared|gave up|no recorded|another chat template|holds no|unreadable|unknown record' 200
  step first_head$n 300 $B first $S/raw/first_head$n.json
  sleep 12; log "helper head$n: $(grep -F '[brain-warm]' $S/logs/brain-head$n.log | tr '\n' '|' | cut -c1-900)"
  brain_down
done

xid_check end
log "WINDOW DONE"
