#!/usr/bin/env bash
# soak_load.sh <port> <label> <until_epoch>: the soak's brain load beside Isaac, after the e2e turns:
#   the agent replay one call at a time (2 sessions) -> decode probe -> mixed replays (agent + perception) until <until_epoch>.
# Every result goes to results/<label>-*.jsonl; logs/soak-<label>.post gets the kernel Xid lines and nvidia-smi at the end.
set -uo pipefail
S=/home/spark/.hermes/cache/scratch/bolt-pr
PORT=$1; LABEL=$2; UNTIL=$3
cd "$S"
health() { [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ]; }
health || { echo "[soak] brain on :$PORT not healthy at start"; exit 3; }
echo "[soak] $(date +%T) agent replay"
bash "$S/measure.sh" "$PORT" "$LABEL-isaac" 2 '{"enable_thinking": false}' || echo "[soak] measure rc=$?"
echo "[soak] $(date +%T) decode"
bash "$S/decode.sh" "$PORT" nemotron-3.5-bolt "$LABEL-isaac" 3 || echo "[soak] decode rc=$?"
n=0
while [ "$(date +%s)" -lt "$UNTIL" ] && health; do
    n=$((n + 1)); echo "[soak] $(date +%T) mixed replay $n"
    bash "$S/measure_mixed.sh" "$PORT" "$LABEL-mixed-isaac" 1 || echo "[soak] mixed rc=$?"
done
health && state=healthy || state=DEAD
echo "[soak] $(date +%T) end: brain $state after $n mixed replays"
{ echo "end $(date -Is) brain=$state mixed=$n"; journalctl -k -n 20 --no-pager | grep -i -E 'xid|nvrm'; nvidia-smi --query-gpu=name,memory.used --format=csv; } > "$S/logs/soak-$LABEL.post" 2>&1
[ "$state" = healthy ]
