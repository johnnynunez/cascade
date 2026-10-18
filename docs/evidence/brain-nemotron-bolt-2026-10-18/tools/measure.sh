#!/usr/bin/env bash
# measure.sh <port> <label> [replays] : wait for the brain's /health, then replay the trial's agent session.
set -euo pipefail
S=/home/spark/.hermes/cache/scratch/bolt-pr
PORT=$1; LABEL=$2; N=${3:-2}; KW=${4:-}
cd "$S"
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
  [ "$code" = 200 ] && break
  sleep 5
done
[ "$code" = 200 ] || { echo "brain on :$PORT never healthy"; exit 3; }
echo "[measure] $(date +%T) healthy; replaying $N sessions as $LABEL"
ARGS=(--base "http://127.0.0.1:$PORT/v1" --key-file "$S/secrets/brain.key" --model nemotron-3.5-bolt \
      --replays "$N" --label "$LABEL" --out "$S/results/$LABEL.jsonl")
[ -z "$KW" ] || ARGS+=(--kwargs "$KW")
exec /home/spark/.local/bin/uv run --no-project python "$S/replay_agent.py" "${ARGS[@]}"
