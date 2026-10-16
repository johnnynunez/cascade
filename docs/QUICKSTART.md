# Opening the demo

Live presentation: use the [presenter card](BOOTH_RUNBOOK.md#9-simulated-kitchen-on-dgx-spark-presenter-card).
Prepared DGX Spark: run `python3 scripts/desktop.py launch --repo "$PWD"` from the
checkout and wait for READY plus three advancing cameras. Brev RTX PRO 6000
profile (PhysX, Qwen3.8-27B Q8_0): [Brev instructions](BREV.md). A Mac is a
development platform; its results do not certify Spark execution. Measured scope
per capability: [capability status](ROADMAP.md).

## 1. DGX Spark (event profile)

The profile is **DGX Spark/Linux + Isaac Sim 6.1 + PhysX CUDA + real GraspGen-X +
Qwen Q4 + OpenClaw**, with the dedicated `cascade-demo` chat profile. It installs
no Cosmos; occupancy/nvblox is disabled and JEv is a separate experiment.

1. Install with the single command in [DGX Spark setup](DGX_SPARK_SETUP.md#2-install) (it pins the source commit).
2. Start: `python3 scripts/desktop.py launch --repo "$PWD"` (also starts Qwen; Isaac is headless unless `--gui`, which needs its own rehearsal).
3. Wait for READY and advancing cameras. The Isaac runtime check prepares the read-only physics verifier and requires new RGB-D captures from every camera; a failure lands in `runtime-check.log` and stops startup ([readiness check](ARCHITECTURE.md#grasp-pipeline)).
4. Between visitors: **Start over**, **Reset** in the PAAI OpenClaw interface, or send **"Reset the scene."**; services keep running and startup tests are not repeated.

Lower-level restart, only while Qwen is running:
`./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge`.
It skips the startup manipulation tests (banner: `STARTED (UNVERIFIED: robot proof
skipped)`) and gives none of the desktop launcher's model supervision; see the
[presenter recovery limits](BOOTH_RUNBOOK.md#9-simulated-kitchen-on-dgx-spark-presenter-card)
and [quick recovery](DGX_SPARK_SETUP.md#ready-checks-and-quick-recovery).

## 2. Laptop or installed rig (MuJoCo + OpenClaw)

```bash
git clone https://github.com/johnnynunez/cascade && cd cascade
./run.sh mujoco --brain keep                  # use the auth already configured in OpenClaw
./run.sh mujoco --cameras mujoco_scene_two    # two-cube scene for the memory demo
./run.sh check isaac                          # report missing requirements; installs nothing
./run.sh down                                 # stop owned sidecars and stale MCP servers
```

The first run creates the venv, installs the mode's extras and robot meshes,
installs or updates the OpenClaw CLI, starts the sidecars (occupancy :5557,
GraspGen-X stub :5556) and registers the robot tools; later runs reuse all of it.
READY needs the full proof ([README → One click](../README.md#one-click)); the
banner names the occupancy backend, grasp planner (stub or model), tool count and
chat URL.

Then chat at `http://127.0.0.1:18789/` or from the terminal, one session keeping memory:

```bash
openclaw agent --session-id demo-1 -m "put both cubes in the drop zone, one at a time; call task_memory before each action"
openclaw agent --session-id demo-1 -m "how many did you move and how do you know?"
openclaw agent --session-id demo-1 -m "reset the scene"
```

The brain sees up to K=4 frames of its earlier actions (initial state, views after
each action, physics verdicts) beside the current view, so its answer cites those
verdicts (recorded development run: 2 physics-confirmed `pick_and_place`, 0
failures, ~100 s). "Reset the scene" or "start over" homes the arm, returns props
to spawn and clears the world model and task memory; Isaac requires each prop's
measured pose within the spawn tolerance.

- **Second stack beside a running one:** export `CASCADE_BRIDGE_PORT`, `CASCADE_OCCUPANCY_PORT` and `CASCADE_GRASPGENX_PORT` before launching; the launcher, the MCP registration and the runtime all use them. A malformed value stops startup naming the variable; empty means the default. `CASCADE_HUG_PORT`, `CASCADE_GRASPGENX_HOST` and `CASCADE_HUG_HOST` follow the same rules.
- **Pick a GPU:** export `CUDA_VISIBLE_DEVICES` before launching or running `scripts/setup_agents.py`; registration copies it (and `CUDA_DEVICE_ORDER`, `CASCADE_DEVICE`, `CASCADE_REQUIRE_CUDA`) unchanged. It does not attest the GPU of an already running server: restart that server.
- **Other hosts:** `scripts/setup_agents.py`, `scripts/hermes_demo.sh` and `scripts/openclaw_demo.sh` copy every runtime switch set in the shell (list: `src/cascade/apps/mcp_env.py`) into the host's entry, as the launcher does.

## 3. Reference rig: reBot / Isaac Sim with a local brain

Start each component once, in its own terminal (one brain server):

```bash
export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-x86_64/release
./run.sh isaac                          # bridge :8611 and the rest of the stack
$ISAACSIM_PATH/python.sh scripts/isaac_bridge.py     # or the bridge alone (Newton by default; --engine physx)
scripts/serve_qwen_llamacpp.sh          # Qwen -> :8080
scripts/serve_cosmos_vllm.sh            # or Cosmos3-Edge -> :8082
TORCH_CUDA_ARCH_LIST=12.0 scripts/serve_graspgenx.sh franka_panda 5556   # learned 6-DoF grasps
```

Connectivity check (works on macOS, which has no `ss`; not a readiness certificate):

```bash
python3 - <<'EOF'
import socket
for p in (8611, 8080, 8082, 5556, 5557, 18789):
    s = socket.socket(); s.settimeout(0.3)
    print(p, "up" if s.connect_ex(("127.0.0.1", p)) == 0 else "down"); s.close()
EOF
```

`./run.sh isaac --brain auto` uses the local server when it answers, else
OpenClaw's existing auth. Real arm instead of Isaac: `sudo ip link set can0 up
type can bitrate 1000000`, then `./scripts/launch.sh --sim none --arm rebot_rs
--cameras l515` (read the [safety notes](../README.md#safety-notes-for-a-live-rig) first).

## 4. CLI without a chat host

```bash
cd models     # the cwd of every launch route; uv still finds the project one level up
uv run python -m cascade.apps.demo --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view       # one task
uv run python -m cascade.apps.demo --cameras isaac,isaac_side --arm isaac --llm local_qwen --interactive
```

Ask in plain English with names from the scene: `What can you see on the table?` ·
`Could you put the green cube in the green square?` · `Please put the orange in the
open box.` · `Let's start over.` The CLI runs all three tiers (reflex → habit →
LLM); a chat host runs only its own. `--llm` takes `codex_astra`, `hermes`,
`local_qwen`, `local_cosmos`, `local_cosmos_sglang`, `anthropic`, `openai` or `mock`
([LLM backends](../README.md#llm-backends)). Object names come from the Qwen brain
at `CASCADE_PERCEPTION_URL` (default `http://127.0.0.1:8080/v1`; the rig's OpenShell
instance uses :8081); mock and MuJoCo cameras need no brain.

## Cameras and the dashboard

Perception always runs; the viewer binds no port until someone asks.

| tool | does |
|---|---|
| `describe_scene` / `get_observation` | fresh view; MCP attaches the image |
| `annotated_view` / `analyze_scene` | badges + 5 cm grid (badges may be stale) / detector estimates and depth diagnostics |
| `open_live_view` / `close_live_view` / `live_view_status` | browser dashboard; closing releases the port |
| `probe_point(u,v)` | a pixel's depth, base-frame position and workspace membership; no IK or grasp check |
| `task_memory` / `world_state` (MCP only) | up to K frames of earlier actions with verdicts / tracked objects and session state |

The dashboard (port 8090) closes after 15 minutes without a viewer and shows rgb,
depth and agent views, the world model, narration and a chat that drives the same
arm. `stream.mode` in `configs/demo.yaml`: `lazy` (default) · `eager` (bind at
startup; set by `booth.yaml`) · `off`; `CASCADE_STREAM` overrides it
(`CASCADE_STREAM=0` is the kill switch) and `CASCADE_BOOTH=1` enables booth mode. The
native MuJoCo window opens with `CASCADE_MJ_VIEW=1` (set by the launcher in sim
modes, under `mjpython` on macOS); with the display locked at startup it opens on
the first motion after wake. macOS has no cv2 camera window from the server: use the dashboard.

## Stopping the arm

- Dashboard: red **stop** button, from any browser on the LAN.
- MCP/OpenClaw: `emergency_stop` acts out of band; `reset_stop` clears it (`CASCADE_HIDE_TOOLS=reset_stop` makes that staff-only).
- Terminal: `Ctrl+C` soft-stops; press it again to exit.
- Cancelling a host turn mid-motion freezes the arm and leaves the e-stop engaged: `reset_stop` before the next motion.

## Troubleshooting

```bash
./run.sh check mujoco                                   # read-only preflight
tail -f runs/mcp_*/server.log                           # every host tools/call and its result
uv run python scripts/learn_from_runs.py --dry-run --report   # recent failures and causes; no notes or envelope writes
```

Chat host and MCP:

- **"Connection closed"** on a turn: the MCP server exited mid-call. Read `/tmp/openclaw/openclaw-<date>.log` (macOS: also `~/Library/Logs/DiagnosticReports/mjpython-*.ips`).
- **Pick cancelled after 60 s, then e-stop errors:** the host kept the default `requestTimeoutMs`; re-register with `./run.sh <mode>` (300 s) and call `reset_stop`.
- **Every turn 10 s slower:** a dead MCP entry in `~/.openclaw/openclaw.json`; the launcher prunes these at startup.
- **Stale tool list:** `openclaw gateway restart` (the launcher does it).
- **No tool calls with Cosmos:** the profile needs `type: cosmos3`, not `openai_compat` (XML tool calls).

Perception and display:

- **Port 8090 busy:** an earlier run is alive; use `CASCADE_STREAM_PORT=8097`.
- **`no frame yet`:** the rig is warming up; wait 2–3 s.
- **Objects without masks:** `models/sam2.1_l.pt` is missing; run `scripts/install.sh --profile spark` (`scripts/booth_up.sh` checks it).
- **`world_state` lists no objects on a real or Isaac camera:** the Qwen brain at `CASCADE_PERCEPTION_URL` is down or `CASCADE_PERCEPTION_KEY_FILE` is wrong.
