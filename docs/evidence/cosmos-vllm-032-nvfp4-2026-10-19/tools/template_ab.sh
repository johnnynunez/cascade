#!/usr/bin/env bash
# template_ab.sh <venv> <export dir> <derived template> <out dir>
# Serve Cosmos3-Edge twice on vLLM (Transformers backend, BF16 KV, --tool-strict-level auto: no grammar for
# non-strict tools), once with the Hub chat template and once with the JSON-schema template, and run
# template_probe.py against each.
set -u
VENV=$1; EXPORT=$2; DERIVED=$3; OUT=$4
PORT=${PORT:-8183}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PROBE_PYTHON:-python3}
mkdir -p "$OUT"
export PATH="$VENV/bin:$PATH" HF_HUB_OFFLINE=1
for label in hub json; do
  template=$EXPORT/chat_template.jinja
  [ "$label" = json ] && template=$DERIVED
  setsid "$VENV/bin/vllm" serve "$EXPORT" --served-model-name cosmos3-edge --host 127.0.0.1 --port "$PORT" \
    --max-model-len 32768 --gpu-memory-utilization 0.20 --model-impl transformers --enforce-eager --max-num-seqs 2 \
    --mm-processor-kwargs '{"size": {"shortest_edge": 65536, "longest_edge": 1048576}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --tool-strict-level auto --reasoning-parser qwen3 \
    --chat-template "$template" > "$OUT/server-$label.log" 2>&1 &
  server=$!
  for _ in $(seq 1 120); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ] && break
    kill -0 $server 2>/dev/null || { echo "server $label exited"; tail -30 "$OUT/server-$label.log"; exit 3; }
    sleep 5
  done
  "$PY" "$HERE/template_probe.py" --base "http://127.0.0.1:$PORT/v1" --label "$label" \
    --catalog "${CATALOG:?set CATALOG to a tool catalog json}" --out "$OUT/template_ab.jsonl"
  kill -TERM -- -$server; sleep 8; kill -KILL -- -$server 2>/dev/null
done
