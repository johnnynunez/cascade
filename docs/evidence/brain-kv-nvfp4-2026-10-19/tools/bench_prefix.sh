#!/usr/bin/env bash
# bench_prefix.sh <label> <serve script> <port> <gpu> : serve, KV lines, decode probe, prefix-reuse probe, stop.
set -u
D=/home/johnny/.hermes/cache/scratch/nvfp4
LABEL=$1; SCRIPT=$2; PORT=$3; GPU=$4
OUT=$D/results; LOG=$OUT/server-$LABEL.log
cd "$D"
echo "[bench] $(date +%T) $LABEL: KV_DTYPE=${KV_DTYPE:-} EXTRA_ARGS='${EXTRA_ARGS:-}'"
CUDA_VISIBLE_DEVICES=$GPU BENCH_GPU=$GPU PORT=$PORT GPU_UTIL=${BENCH_UTIL:-0.55} MAX_JOBS=8 "$D/$SCRIPT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; sleep 5; pkill -P $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 360); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ] && break
  kill -0 $SERVER 2>/dev/null || { echo "[bench] $LABEL: server exited during startup"; tail -40 "$LOG"; exit 3; }
  sleep 5
done
echo "[bench] $(date +%T) $LABEL healthy"
grep -aE "GPU KV cache size|kv cache group sizes|kv lcm block sizes|Setting attention block size" "$LOG" | sed 's/^/[log] /' | cut -c1-220
PY=/home/johnny/Projects/cascade/.venv/bin/python
$PY decode_probe.py --base "http://127.0.0.1:$PORT/v1" --key-file key.txt --model "$MODEL" --label "$LABEL" \
    --out "$OUT/decode.jsonl" --n 3 2>&1 | tail -1 | cut -c1-120
$PY prefix_probe.py --base "http://127.0.0.1:$PORT/v1" --model "$MODEL" --label "$LABEL" --out "$OUT/prefix.jsonl" 2>&1 | tail -8
echo "[bench] $(date +%T) $LABEL done"
