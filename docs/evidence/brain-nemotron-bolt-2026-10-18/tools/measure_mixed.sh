#!/usr/bin/env bash
# measure_mixed.sh <port> <label> [replays]: wait for /health, then the agent replay with perception beside it.
set -euo pipefail
S=/home/spark/.hermes/cache/scratch/bolt-pr
PORT=$1; LABEL=$2; N=${3:-1}
cd "$S"
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = 200 ] && break
  sleep 5
done
[ "$code" = 200 ] || { echo "brain on :$PORT never healthy"; exit 3; }
echo "[measure_mixed] $(date +%T) healthy; $N mixed replays as $LABEL"
exec /home/spark/.local/bin/uv run --no-project python "$S/replay_mixed.py" --base "http://127.0.0.1:$PORT/v1" \
    --key-file "$S/secrets/brain.key" --replays "$N" --label "$LABEL" --out "$S/results/$LABEL.jsonl"
