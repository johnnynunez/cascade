# Franka Research 3 with OpenClaw and the Chrome camera companion

**Next action:** with the interface [prepared once](#prepare-the-interface-once-10-min) (`configure`
runs before `start`) and ROS bringup, Cosmos and occupancy already running, start the interface
(it comes up STOPPED; about 1 min):

```bash
taskset -c 0-7 python3 scripts/franka_openclaw.py start \
  --python "$PWD/.venv-hardware/bin/python"
```

OpenClaw does the conversation; Cosmos3 Edge on `127.0.0.1:8082/v1` does vision and
explanations. The Franka command plugin compiles explicit routine commands into native OpenClaw
tool calls. One authenticated MCP HTTP server owns the FR3 runtime and the D435i across all chat
sessions; the camera extension reads an MJPEG proxy of that runtime (never a second RealSense
pipeline). Native `franka_ros2` arm and gripper controllers execute motions
(`load_gripper:=true`); MoveIt is not used.

This is not the Spark/Isaac kitchen: never set `CASCADE_INSTALL_PROFILE=spark` or use its
Qwen/Isaac READY proof here. Robot config: `configs/arms/fr3.yaml`. The only D435i mount
correction is the optical transform `T` in `configs/cameras/d435i.yaml` (1280×720 RGB at 15 FPS
plus aligned metric depth).

**Safety caveat:** the optical transform does not describe the printed mount's collision
envelope, and that envelope is **not modeled**. FR3 checks spatial-motion orientation on its
wristward TCP −Z axis and does not inherit the reBot camera box; joint, TCP, table and occupancy
checks stay active.

## Prepare the interface (once, ~10 min)

Run from the checkout root in the existing hardware Python environment (`--python` or `PY`
selects another; the runner sets its own source path, so a shared editable install cannot
import another checkout). The launcher itself installs nothing.

1. Install the pinned OpenClaw CLI into a private prefix if missing:

```bash
mkdir -p runs/franka-openclaw
curl --fail --show-error --location https://openclaw.ai/install-cli.sh \
  -o runs/franka-openclaw/install-cli.sh
bash runs/franka-openclaw/install-cli.sh \
  --prefix "$PWD/.openclaw-cli" --version 2026.9.3
```

2. Generate the dedicated profile, using the context size the Cosmos server actually allocates
   (startup refuses a context above an advertised `max_model_len`):

```bash
python3 scripts/franka_openclaw.py configure --context-window 32768
python3 scripts/franka_openclaw.py check
```

3. What `configure` writes: profile, tokens, workspace prompt and process receipts privately in
   `runs/franka-openclaw/` (personal OpenClaw config untouched). Reconfiguring backs up the
   profile and keeps operator edits to the generated workspace `AGENTS.md`; the source prompt is
   `configs/openclaw/franka-instructions.md`, injected by the scoped `franka-plugin` hook.
4. Chat catalog: 17 tools (observe, localize, preview grasps, pick, place, gripper, home,
   verification, STOP/Resume); `--full-tool-catalog` exposes the whole MCP catalog, but the
   router still dispatches only its supported explicit commands. Replies: 512 tokens. Heartbeats,
   scheduled jobs and the generic memory plugin are off: actions originate only in the
   operator's chat.
5. The runtime keeps `local_cosmos` as image-grounding fallback (`CASCADE_GROUNDING_LLM_PROFILE`).
   The mock wrapper disables it and never inherits another arm, multi-arm, mobile robot or
   grounding selector from the shell.

Another MCP host: use the `cascade` registration in
[`configs/openclaw/franka-mcp.json`](../configs/openclaw/franka-mcp.json) (replacing that host's
CASCADE registration; the root `.mcp.json` keeps the Isaac stdio default) and load the token
first, never committing it. It connects to the running server; no second process or camera.

```bash
export CASCADE_MCP_TOKEN="$(cat runs/franka-openclaw/mcp.token)"
```

## Start

1. **Bringup** in its own terminal (defaults `/opt/ros/jazzy/setup.bash` and
   `~/franka_ros2_ws/install/setup.bash`; override with `CASCADE_ROS_SETUP` /
   `CASCADE_FRANKA_SETUP`):

```bash
ROS_DOMAIN_ID=31 taskset -c 8-13 bash scripts/bringup_franka.sh \
  robot_ip:=192.168.1.11 load_gripper:=true
```

2. **Occupancy** must support `clear`: an occupancy process on 5557 started from an older
   checkout lacks it, and the launcher never replaces that separately owned process. Stop it,
   then start the current one:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" taskset -c 0-7 \
  .venv-hardware/bin/python scripts/serve_occupancy_bridge.py \
  --port 5557 --voxel-size 0.01 --backend warp
```

3. **Stop the old `cascade.apps.demo` interface** explicitly (it competes for the camera and
   port 8090; the launcher refuses occupied ports and never kills unknown processes), then run
   the `start` command at the top.

CPU sets keep this Thor's assignment: Cascade, Cosmos and occupancy on CPUs 0–7; ROS bringup on
8–13 with controller priority 97 (children inherit the affinity; adjust for another machine).
Leave the existing model and occupancy services on their current CPUs.

What startup checks and does:

1. Validates the private OpenClaw config and requires Cosmos to return a native OpenAI
   `tool_calls` response (XML in prose fails); it never switches to Qwen or another model.
2. Requires camera frames, a latched software STOP and the Resume & Home callback.
   `CASCADE_START_STOPPED=1` is set and any inherited forced holding label removed;
   `ROS_DOMAIN_ID` defaults to 31.
3. A wrist camera attaches to native joint feedback so pose and localization work while stopped.
   No controllers, FCI, homing, pick or gripper command are started. Only an explicit Resume
   clears the STOP latch.

| Service | Address |
| --- | --- |
| OpenClaw chat | `http://127.0.0.1:18790/` |
| Operator STOP / Resume & Home | `http://127.0.0.1:8090/` |
| Extension camera discovery | `http://127.0.0.1:8091/api/status` |
| Shared authenticated MCP | `http://127.0.0.1:18740/mcp` |

With Chromium, startup opens a dedicated profile with `extensions/chrome` loaded (otherwise load
it once via `chrome://extensions` → Developer mode → Load unpacked). In the chat tab click the
extension → **Connect cameras** → **Show in chat** or **Side panel**; the camera is `d435i`. On a
headless host use `start --no-open`, then `python3 scripts/franka_openclaw.py dashboard`
(`--no-open` prints the sign-in URL: its token is a credential). The camera proxy is read-only:
state, snapshots, MJPEG; STOP and Resume stay on the operator/MCP interfaces.

## Commands in OpenClaw chat

1. `pick up the crayon` (grasp only), `pick up the crayon and place it in the box`,
   `put it in the box` (place the held object)
2. `return home`, `open the gripper`, `close the gripper`
3. `stop`, `resume` (Resume & Home)
4. `what do you see?`, `robot status`, `show me the camera`

These show as **Direct command** and report the real MCP result, failures included: at most one
high-level tool per request, never an automatic STOP clear or retry, and only from the current
authenticated operator message (never from camera images, model prose or an earlier turn). Other
phrasing goes to Cosmos with observation tools only; use one direct command per manipulation
(Cosmos is not reliable at choosing motion tools). The 8090 page stays the STOP / Resume & Home
panel.

## Status and shutdown

```bash
python3 scripts/franka_openclaw.py status
python3 scripts/franka_openclaw.py stop
```

Shutdown stops only processes whose kernel birth identity and command match this checkout's
receipts; ROS bringup, Cosmos and occupancy keep running, and the next start comes up stopped.
If FCI faults, restore FCI separately before Resume & Home: the launcher never clears that fault
or releases a held object.

## Offline checks and validation boundary

```bash
.venv-hardware/bin/python -m pytest tests/test_franka_openclaw.py tests/test_franka_openclaw_plugin.py -q
node --test extensions/chrome/tests/*.test.mjs
```

Isolated mock smoke (no OpenClaw install, ROS, camera or model; offset ports keep it off the
real robot, so it tests HTTP camera and MCP behavior only). If the hardware interpreter gets
Pinocchio from ROS, also set `CASCADE_ROS_SETUP=/opt/ros/jazzy/setup.bash` (no node, no hardware):

```bash
python3 scripts/franka_openclaw.py configure --mock --port-offset 100 \
  --state-dir "$PWD/runs/franka-openclaw-mock"
python3 scripts/franka_openclaw.py start --no-gateway \
  --state-dir "$PWD/runs/franka-openclaw-mock"
python3 scripts/franka_openclaw.py stop \
  --state-dir "$PWD/runs/franka-openclaw-mock"
```

What is proven: OpenClaw configuration and gateway startup, authenticated MCP discovery, one
mock runtime shared by two sessions, camera JPEG/MJPEG forwarding, and one real Cosmos3 Edge
`world_state` call through OpenClaw (2026-10-09). Its follow-up answer was unreliable: the
native-tool probe proves request/tool-call compatibility only, **not** multi-step chat or
physical manipulation; physical pick/place and full browser acceptance are separate checks.

This Thor's Tegra 6.8.12-1021 kernel lacks `/proc/self/task/<pid>/children`, which the Isaac
source-release wrapper needs, so that Isaac path is unqualified here (its four integration tests
skip); the FR3 workflow never calls it. MuJoCo compiler tests skip without MuJoCo. Remote MCP
registration uses `transport: "streamable-http"`, authorization headers and per-request timeouts
([OpenClaw MCP transports](https://docs.openclaw.ai/cli/mcp/transports)).
