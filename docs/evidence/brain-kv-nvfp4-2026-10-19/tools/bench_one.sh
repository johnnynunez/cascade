#!/usr/bin/env bash
# bench_one.sh <label> <serve script> <port> <gpu> [vllm extra args...]
# One brain configuration: serve, read the KV capacity from the log, decode probe, accuracy probes, agent probe, stop.
set -u
D=/home/johnny/.hermes/cache/scratch/nvfp4
LABEL=$1; SCRIPT=$2; PORT=$3; GPU=$4; shift 4
EXTRA="$*"
OUT=$D/results; mkdir -p "$OUT"
LOG=$OUT/server-$LABEL.log
cd "$D"
echo "[bench] $(date +%T) $LABEL: $SCRIPT port $PORT gpu $GPU extra '$EXTRA'"
CUDA_VISIBLE_DEVICES=$GPU BENCH_GPU=$GPU PORT=$PORT GPU_UTIL=${BENCH_UTIL:-0.55} EXTRA_ARGS="$EXTRA" MAX_JOBS=8 "$D/$SCRIPT" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; sleep 5; pkill -P $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
t0=$(date +%s)
for i in $(seq 1 360); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ] && break
  kill -0 $SERVER 2>/dev/null || { echo "[bench] $LABEL: server exited during startup"; tail -40 "$LOG"; exit 3; }
  sleep 5
done
echo "[bench] $(date +%T) $LABEL healthy after $(( $(date +%s) - t0 )) s"
grep -E "KV cache size|Maximum concurrency|kv_cache_dtype|Using .* backend|attention backend|XQA|nvfp4" "$LOG" | sed 's/^/[log] /' | head -20
PY=/home/johnny/Projects/cascade/.venv/bin/python
$PY decode_probe.py --base "http://127.0.0.1:$PORT/v1" --key-file key.txt --model "${MODEL:-qwen3.8-27b}" --label "$LABEL" \
    --out "$OUT/decode.jsonl" --n 3 2>&1 | tail -4
$PY brain_quality.py --base "http://127.0.0.1:$PORT/v1" --model "${MODEL:-qwen3.8-27b}" --label "$LABEL" --out "$OUT/quality.jsonl" \
    --gsm8k 200 --needle 9 --needle-tokens 56000 2>&1 | tail -3
cd /home/johnny/Projects/cascade-wt/nvfp4-kv
CASCADE_OCCUPANCY=0 MUJOCO_GL=egl CASCADE_GRASP_BACKEND=obb PYTHONUNBUFFERED=1 \
    $PY "$D/agent_probe.py" --profile "${PROFILE_PREFIX:-bench_qwen}_$PORT" --label "$LABEL" --out "$OUT/agent.jsonl" --reps 2 --workers 4 2>&1 | tail -2
grep -E "Avg prompt throughput|SpecDecoding metrics" "$LOG" | tail -2 | sed 's/^/[log] /'
echo "[bench] $(date +%T) $LABEL done"
