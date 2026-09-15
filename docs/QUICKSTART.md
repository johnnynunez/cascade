# Opening the demo

The original delivery profile uses **DGX Spark/Linux + Isaac Sim 6.1.0.0 +
Newton + Cosmos3-Edge + OpenClaw**. The launcher opens chat after the proof
completes; `--no-open` keeps it closed. The camera UI is separate and optional.
The Mac is a development platform; its results do not certify the delivery's
GPU execution. Spark/GPU cold-start certification remains pending.

For the **Brev RTX PRO 6000 profile with PhysX and Qwen3.8-27B Q8_0**, see
[the Brev instructions](BREV.md). They describe the tested deployment and its
remaining acceptance limits.
The performance and test counts below describe earlier development runs,
not a Spark or Brev certification.

---

## 0. One click for visitors

### DGX Spark

Once these changes are available at the distribution ref:

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash -s -- --accept-eula
```

For multiple machines, pin a proven commit in both the URL and `--ref`.
Spark cold-start certification remains pending until the required acceptance
run succeeds on a Spark machine.
From this checkout, use:

```bash
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula
# Prepare packages/models before the event without starting services:
bash scripts/install.sh --dir "$PWD" --profile spark --accept-eula --prepare-only
```

The installer requires explicit license acceptance and leaves drivers
unchanged. It isolates CASCADE, Isaac, and Cosmos, installs OpenClaw
2026.9.3 locally, and uses the `cascade-demo` profile independently of
personal configuration. A Cosmos failure does not trigger an OpenAI
fallback. For details and the pending acceptance gate, see
[SPARK_DELIVERY.md](SPARK_DELIVERY.md).

### Development on a Mac or an installed rig

```bash
git clone https://github.com/johnnynunez/cascade && cd cascade
./run.sh mujoco --brain keep     # use authentication already configured in OpenClaw
./run.sh mujoco --cameras mujoco_scene_two    # two-cube scene for the memory demo
./run.sh check isaac            # report missing requirements without installing or downloading
./run.sh down                   # stop owned sidecars and MCP servers from earlier sessions
```

The first run creates the venv, installs the mode's extras, downloads robot
meshes, installs or updates the OpenClaw CLI, starts the sidecars
(occupancy :5557, GraspGen-X stub :5556), and registers the robot tools.
It **checks the stack before reporting READY**: it builds the runtime with
the same environment as the server, lists the 41 tools, requests a simple
brain response, and performs pick and reset within one session.
The `proof.json` receipt must bind the model, session, and MCP process to
the physical result and restoration of the manipulated object. An open
port, a model response, or another session's trace is insufficient.
With `--no-robot-turn`, the status is STARTED / UNVERIFIED, never READY.
Subsequent runs reuse downloaded packages and assets.

The READY banner reports the actual occupancy backend, grasp planner
(stub or model), tool count, chat URL, and whether the MuJoCo window is
open. If the display was locked at startup, it explains why the window
was deferred; the window opens on the first movement after the display wakes.

Then open `http://127.0.0.1:18789/` and chat, or use the terminal:

```bash
openclaw agent exec "what do you see?"
openclaw agent exec "pick and place the red object"
openclaw agent exec "did it actually move?"
# One session retaining memory between turns, as the dashboard does:
openclaw agent -m "put both cubes in the drop zone, one at a time; call task_memory before each action" --session-id demo-1
openclaw agent -m "how many did you move and how do you know?" --session-id demo-1
openclaw agent -m "reset the scene" --session-id demo-1
```

Every host tool call is recorded in `runs/mcp_<pid>/server.log`.
OpenClaw itself shows only a failure counter.

### Memory demo with two cubes

A single pick-and-place is Markovian, so it does not visibly demonstrate
memory. With two props, the current view after the first pick alone does
not establish whether one object or neither has moved. Ask in chat:

> Put both cubes in the drop zone, one at a time. Before each action call
> task_memory to see what you already did, and when both are done tell me
> how many cubes you moved and how you know.

The brain receives up to K=4 frames from its earlier actions: the initial
state, views after actions, and physics verdicts, alongside the current
view. Its answer cites those verdicts. Recorded development result:
2 physics-confirmed `pick_and_place` actions, 0 failures, approximately 100 s.

Between visitors, ask **"reset the scene"** or "start over". The arm
returns home, props return to spawn, and the world model and task memory
are cleared. Isaac requires finite physical readback of each prop within
the spawn tolerance. The reflex works without an LLM in the CASCADE CLI;
in OpenClaw chat, the host model still chooses the tool.

---

## 0b. Without hardware, a GPU, servers, or OpenClaw

```bash
uv venv && uv pip install -e '.[dev,kinematics]' && source .venv/bin/activate
python -m cascade.apps.demo --arm so101_mock --camera mock_small \
    --task "pick and place the red object"
```

This runs the complete cascade on a kinematically simulated five-axis
SO-101. With `.[sim]` and `python scripts/fetch_robot_assets.py so101`,
use MuJoCo physics via `--arm so101_mujoco --camera mujoco_scene`.
The camera renders the same world as the arm, and the postcondition reads
the prop's physical pose: `postcondition: confirmed (channel: physics)`.
With `.[sim-warp]`, `--arm so101_mjwarp` uses the same MJCF in MuJoCo Warp
on the GPU. Its CPU path was approximately 650× slower than the C engine
in the recorded comparison; use it for developing that path, not presenting
the demo.

---

## 1. Reference rig: reBot / Isaac Sim with a local brain

Start each required component once, in its own terminal. Choose one of the
two brain servers shown below. These commands retain the original local
server profiles; use the Brev guide for the Qwen3.8 production profile.

```bash
export ISAACSIM_PATH=~/Projects/isaac/IsaacSim/_build/linux-x86_64/release
./run.sh isaac                          # start the bridge (:8611) and the rest of the stack
# Or start the bridge manually:
$ISAACSIM_PATH/python.sh scripts/isaac_bridge.py     # Newton by default; --engine physx

scripts/serve_qwen_llamacpp.sh          # Qwen3.6 -> :8080   (choose one brain server)
scripts/serve_cosmos_vllm.sh            # Cosmos3-Edge -> :8082
TORCH_CUDA_ARCH_LIST=12.0 scripts/serve_graspgenx.sh franka_panda 5556   # learned 6-DoF grasps
```

Check service reachability without relying on `ss`, which macOS does not
provide. This is a connectivity check, not a readiness certificate:

```bash
python - <<'EOF'
import socket
for p in (8611, 8080, 8082, 5556, 5557, 18789):
    s = socket.socket(); s.settimeout(0.3)
    print(p, "up" if s.connect_ex(("127.0.0.1", p)) == 0 else "down"); s.close()
EOF
```

`./run.sh isaac --brain auto` uses the local server when it responds;
otherwise, it uses authentication already configured in OpenClaw.

## 2. One terminal command without a chat host

```bash
cd models && python -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen \
    --task "pick and place the pink cube in the box" --no-view
```

> `cd models` is required for the YOLOE detector: it looks for
> `mobileclip_blt.ts` in the CWD. Mock/MuJoCo cameras do not require it.

Brain profiles: `--llm hermes` (Nous Portal, `NOUS_API_KEY`) |
`local_qwen` | `local_cosmos` | `local_cosmos_sglang` | `anthropic` |
`openai` | `mock` (wiring check without an LLM). The default is `auto`:
Hermes, Anthropic, or OpenAI according to the exported key; without a key,
it uses `mock`. The CLI runs all three layers: reflex → habit → LLM.
The chat host runs only its own layer.

## 3. Interactive terminal chat

```bash
cd models && python -m cascade.apps.demo \
    --cameras isaac,isaac_side --arm isaac --llm local_qwen --interactive
```

It accepts English and Spanish instructions. English examples:
`describe the scene` / `what do you see?` · `pick up the pink cube and put
it in the box` · `open the cameras` (returns a URL) · `throw the banana` ·
`reset the scene`.

---

## Cameras open on request

Perception **keeps running**: the rig supplies frames and the world model
stays current. The viewer does not bind a port until someone requests it.

| Skill | Behavior |
|---|---|
| `analyze_scene` | Answer "what do you see?" **without opening a viewer**: detections, depth quality, and description |
| `open_live_view` / `close_live_view` / `live_view_status` | Open, close, or inspect the browser dashboard; closing releases its port |
| `probe_point(u,v)` | Inspect a **cursor** location: object at the pixel, distance, and reachability |
| `annotated_view` | Numbered markers, a 5 cm grid, and the reachable region |
| `task_memory` (MCP only) | Up to K frames from earlier actions in this task, with verdicts |
| `world_state` (MCP only) | Objects, the held object, and the latest dispatch path |

The dashboard **closes automatically after 15 minutes** without a viewer.
Views include **rgb** (detector and HUD), **depth** (colormap,
minimum/median/maximum, and valid percentage), and **agent** (markers,
grid, and IK band). It also provides an analysis panel, world model,
narration, and chat controlling the same arm.

Set `stream.mode` in `configs/demo.yaml`; `CASCADE_STREAM` takes precedence:
`lazy` (default) · `eager` (bind at startup; selected by `booth.yaml`) ·
`off` (`CASCADE_STREAM=0`, kill switch). `CASCADE_BOOTH=1` enables booth mode.

The native **MuJoCo** window opens with `CASCADE_MJ_VIEW=1`, which the
launcher sets in simulation mode, using `mjpython` on macOS. If the
display is locked at startup, the launcher logs the reason for deferring
the window and opens it on the first movement after the display wakes.
Opening it while the display was locked caused server segmentation faults.
The server does not open a cv2 camera window on macOS because Cocoa
requires the main thread; use the dashboard.

---

## Stopping the arm

- Dashboard: red **stop** button.
- MCP/OpenClaw: `emergency_stop` acts out of band, without waiting for the
  movement to finish. `reset_stop` clears it; `CASCADE_HIDE_TOOLS=reset_stop`
  reserves that operation for staff.
- Terminal: `Ctrl+C` soft-stops; press it again to exit.
- Canceling a host turn during movement also freezes the arm and leaves
  the emergency stop engaged. Use `reset_stop` before the next movement.

---

## Troubleshooting

```bash
python -m pytest tests/ -q                    # historical run: 737 passed / 2 deselected / 0 skipped, ~4 min
python scripts/learn_from_runs.py --report    # recent failures and their causes
./run.sh check mujoco                         # read-only preflight
tail -f runs/mcp_*/server.log                 # each host tools/call and its result
```

- **A visitor turn fails with "Connection closed"** → the MCP server
  exited during the call. Check `~/Library/Logs/DiagnosticReports/mjpython-*.ips`
  on macOS and `/tmp/openclaw/openclaw-<date>.log`. A known, resolved cause
  was opening the MuJoCo window while the display was asleep.
- **A pick is canceled after 60 s and later calls fail with e-stop** → the
  host still has the default 60-second `requestTimeoutMs`; the launcher
  registers 300 seconds. Register again with `./run.sh <mode>` and use
  `reset_stop`.
- **Every turn takes an extra 10 s** → a dead MCP entry remains in
  `~/.openclaw/openclaw.json`; the launcher prunes these at startup.
- **Port 8090 is busy** → an earlier run is still active. Use
  `CASCADE_STREAM_PORT=8097`.
- **`no frame yet`** → the rig is still warming up; allow 2–3 seconds.
- **YOLOE cannot find its weights** → start from `models/`.
- **The agent does not call tools with Cosmos** → the profile must use
  `type: cosmos3`, rather than `openai_compat`, because it emits XML tool calls.
- **The tool list is stale after registration changes** → run `openclaw
  gateway restart`; the launcher does this automatically.
