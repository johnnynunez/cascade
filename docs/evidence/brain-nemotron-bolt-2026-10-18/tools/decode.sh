#!/usr/bin/env bash
# decode.sh <port> <model> <label> [n]: wait for /health, then decode_probe.
set -euo pipefail
S=/home/spark/.hermes/cache/scratch/bolt-pr
PORT=$1; MODEL=$2; LABEL=$3; N=${4:-3}
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = 200 ] && break
  sleep 5
done
cd "$S"
exec /home/spark/.local/bin/uv run --no-project python "$S/decode_probe.py" --base "http://127.0.0.1:$PORT/v1" --key-file "$S/secrets/brain.key" --model "$MODEL" --label "$LABEL" --out "$S/results/decode-$LABEL.jsonl" --n "$N"
