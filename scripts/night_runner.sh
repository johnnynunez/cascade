#!/usr/bin/env bash
# Nightly persistent runner for the wrc_demo agentic dashboard.
# Keeps the demo REPL (with the :8090 MJPEG dashboard) alive all night and
# feeds it a rotating list of agent tasks so there's always motion to watch.
# The dashboard stays up on http://<LAN-IP>:8090/ the entire time.
set -u
cd /home/johnny/Projects/demo/wrc_demo/models
export PYTHONPATH=/home/johnny/Projects/demo/wrc_demo/src
PY=/home/johnny/Projects/demo/.demo/bin/python

# Guard 1: refuse to start a second dashboard. A stale instance still holding
# :8090 while pointed at a dead bridge is what spams "Broken pipe" -- never
# run two. If 8090 is taken, assume a healthy one is already up and exit.
if (exec 3<>/dev/tcp/127.0.0.1/8090) 2>/dev/null; then
  echo "[night_runner] :8090 already served -- not starting a second dashboard"
  exit 0
fi

# Guard 2: wait for the bridge (:8611) to be alive before starting, so the
# stream never opens against a dead bridge.
for _ in $(seq 1 60); do
  if (exec 3<>/dev/tcp/127.0.0.1/8611) 2>/dev/null; then break; fi
  echo "[night_runner] waiting for bridge :8611 ..."
  sleep 3
done
if ! (exec 3<>/dev/tcp/127.0.0.1/8611) 2>/dev/null; then
  echo "[night_runner] bridge :8611 not up after 180s -- aborting"
  exit 1
fi

# Rotating agent tasks: motion skills that WORK reliably (point/wave), plus
# perception/reasoning tasks. These keep Qwen busy and the arm moving.
TASKS=(
  "pick and place the pink cube in the box"
  "what objects are on the table and what colors are they?"
  "point at the pink cube then wave hello"
  "describe the scene and pick the object you would grab first"
  "point at the green cube"
)

# Feed tasks on stdin, one every ~105s, looping forever. Between tasks,
# reset props so a pick that emptied a cube into the bin starts fresh.
(
  i=0
  while true; do
    echo "${TASKS[$((i % ${#TASKS[@]}))]}"
    i=$((i+1))
    sleep 100
    # best-effort prop reset via the bridge (ignore errors if busy)
    "$PY" -c "from wrc_demo.sim.bridge_client import BridgeClient; c=BridgeClient(port=8611); c.connect(); c.request({'op':'reset_props'}); c.close()" 2>/dev/null || true
    sleep 5
  done
) | stdbuf -oL -eL "$PY" -m wrc_demo.apps.demo \
      --cameras isaac,isaac_side,isaac_wrist --arm isaac --llm local_qwen \
      --interactive --run-dir /tmp/wrc_night 2>&1
