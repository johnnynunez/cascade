#!/usr/bin/env bash
# bench_tools.sh <label> <serve script> <port> <gpu> : serve, run the tool-call probe, stop.
set -u
D=/home/johnny/.hermes/cache/scratch/nvfp4
LABEL=$1; SCRIPT=$2; PORT=$3; GPU=$4
OUT=$D/results; LOG=$OUT/server-$LABEL.log
cd "$D"
CUDA_VISIBLE_DEVICES=$GPU BENCH_GPU=$GPU PORT=$PORT GPU_UTIL=${BENCH_UTIL:-0.55} MAX_JOBS=8 "$D/$SCRIPT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; sleep 5; pkill -P $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 360); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ] && break
  kill -0 $SERVER 2>/dev/null || { echo "[bench] $LABEL: server exited during startup"; tail -40 "$LOG"; exit 3; }
  sleep 5
done
echo "[bench] $(date +%T) $LABEL healthy"
/home/johnny/Projects/cascade/.venv/bin/python tool_probe.py --base "http://127.0.0.1:$PORT/v1" --model "$MODEL" --label "$LABEL" --out "$OUT/tools.jsonl" 2>&1 | tail -2
echo "[bench] $(date +%T) $LABEL done"
