#!/usr/bin/env bash
# bench_one.sh <label> <out dir> <serve command...>
# Serve one Cosmos3-Edge configuration (its knobs come from the caller's environment), wait for /health, keep only
# the engine/KV lines of its log, run cosmos_probe.py and tests/test_cosmos_live.py against it, then stop it.
# PORT (8182) must match the serve command's port; PROBE_PYTHON is a CASCADE venv python (pytest, openai, jsonschema;
# PROBE_PYTHONPATH adds e.g. a Pillow --target dir for the image live tests). PROBE_ARGS go to cosmos_probe.py
# (e.g. "--probes decode --decode-runs 5"); LIVE=0 skips the live tests.
set -u
LABEL=$1; OUT=$2; shift 2
PORT=${PORT:-8182}
TOOLS=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$TOOLS/../../../.." && pwd)
PY=${PROBE_PYTHON:-$REPO/.venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/server-$LABEL.log
echo "[bench] $(date +%T) $LABEL: $*"
setsid "$@" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill -TERM -- -$SERVER 2>/dev/null; sleep 8; kill -KILL -- -$SERVER 2>/dev/null' EXIT
t0=$(date +%s)
for _ in $(seq 1 180); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)" = 200 ] && break
  kill -0 $SERVER 2>/dev/null || { echo "[bench] $LABEL: server exited during startup"; tail -40 "$LOG"; exit 3; }
  sleep 5
done
echo "[bench] $(date +%T) $LABEL healthy after $(( $(date +%s) - t0 )) s"
grep -a -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|kv_cache_dtype|[Nn][Vv][Ff][Pp]4|attention backend|Using [A-Za-z_]+ backend|AttentionBackend|block size|kv cache group|lcm block|XQA|xqa|Resolved architecture|\[cosmos\] KV cache" "$LOG" \
  | grep -v "non-default args" > "$OUT/server-$LABEL.kvlines.txt"
sed 's/^/[log] /' "$OUT/server-$LABEL.kvlines.txt" | head -30
PYTHONPATH=$REPO/src:$REPO${PROBE_PYTHONPATH:+:$PROBE_PYTHONPATH} "$PY" "$TOOLS/cosmos_probe.py" --base "http://127.0.0.1:$PORT/v1" --label "$LABEL" --out "$OUT" ${PROBE_ARGS:-}
[ "${LIVE:-1}" = 1 ] && COSMOS_TEST_BASE_URL="http://127.0.0.1:$PORT/v1" COSMOS_TEST_EVIDENCE_DIR="$OUT/live-$LABEL" PYTHONPATH=$REPO/src:$REPO${PROBE_PYTHONPATH:+:$PROBE_PYTHONPATH} \
  "$PY" -m pytest "$REPO/tests/test_cosmos_live.py" -q -p no:cacheprovider -rA 2>&1 | tail -12 | tee "$OUT/live-$LABEL.txt"
echo "[bench] $(date +%T) $LABEL done"
