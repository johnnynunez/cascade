# Franka Research 3 with OpenClaw and the Chrome camera companion

The hardware workflow uses OpenClaw for conversation and Cosmos3 Edge on
`127.0.0.1:8082/v1` for vision and explanations. Explicit routine commands are
compiled into native OpenClaw tool calls by the Franka command plugin. One authenticated MCP
HTTP server owns the FR3 runtime and D435i camera across all chat sessions.
The camera extension reads an MJPEG proxy of that same runtime; it never opens
another RealSense pipeline. Native `franka_ros2` arm and gripper controllers
execute motions, with `load_gripper:=true`. MoveIt is not used.

This is separate from the upstream Spark/Isaac kitchen installer. Do not set
`CASCADE_INSTALL_PROFILE=spark` or use its Qwen/Isaac READY proof for this rig.
The robot's configuration remains `configs/arms/fr3.yaml`; the sole D435i mount
correction remains the optical transform `T` in `configs/cameras/d435i.yaml`.
That profile requests 1280×720 RGB at 15 FPS and aligned metric depth.

The optical transform does not describe the printed mount's collision envelope.
FR3 uses its wristward TCP −Z axis for spatial-motion orientation checks and
does not inherit the reference reBot's camera box. The printed mount envelope
is not modeled; joint, TCP, table and occupancy checks remain active.

## Prepare the interface

Use the existing hardware Python environment. The examples run from the
checkout root. `--python` or `PY` can select another installed environment.
The runner sets its own source path, so a shared editable environment cannot
silently import another checkout.

Install the pinned OpenClaw CLI separately if it is missing. This follows the
upstream launcher's private-prefix installation, with no global CLI migration:

```bash
mkdir -p runs/franka-openclaw
curl --fail --show-error --location https://openclaw.ai/install-cli.sh \
  -o runs/franka-openclaw/install-cli.sh
bash runs/franka-openclaw/install-cli.sh \
  --prefix "$PWD/.openclaw-cli" --version 2026.9.3
```

The hardware launcher itself installs nothing. Generate its dedicated profile:

```bash
python3 scripts/franka_openclaw.py configure --context-window 32768
python3 scripts/franka_openclaw.py check
```

Choose the context size actually allocated by the Cosmos server; startup
refuses a configured context larger than an advertised `max_model_len`.
The profile, tokens, workspace prompt and process receipts live privately in
`runs/franka-openclaw/`. Existing personal OpenClaw configuration is untouched.
Reconfiguration backs up the previous profile and preserves operator edits to
the generated workspace `AGENTS.md`. The source prompt is
`configs/openclaw/franka-instructions.md`.
The scoped `franka-plugin` prompt hook supplies those instructions directly to
Cosmos while preserving the native tool schemas. The default chat catalog
contains 17 tools for observing, localizing, previewing grasps, picking, placing,
gripper control, home, verification and STOP/Resume. Advanced gestures, assembly
and memory tools remain available through the full MCP API. Configure with
`--full-tool-catalog` to expose that entire available catalog to OpenClaw too;
the Franka chat router still only dispatches its supported explicit commands.
Replies have a 512-token budget. Automatic
heartbeats, scheduled jobs and the generic memory plugin are disabled in this
robot-control profile; actions originate in the operator's chat.
The native runtime still uses `local_cosmos` for image-grounding fallback
through `CASCADE_GROUNDING_LLM_PROFILE`; OpenClaw remains the chat/tool host.
The mock wrapper disables that fallback and never inherits another arm,
multi-arm, mobile robot or grounding-model selector from the shell.

Type commands in **OpenClaw chat**. Supported examples include:

- `pick up the crayon` (grasp only)
- `pick up the crayon and place it in the box`
- `put it in the box` (place the held object)
- `return home`, `open the gripper`, `close the gripper`
- `stop`, `resume` (Resume & Home)
- `what do you see?`, `robot status`, `show me the camera`

The chat marks these as **Direct command** and reports the actual MCP result,
including failures. It dispatches at most one high-level tool per request; it
does not automatically clear STOP or retry a failed tool. Routine commands use
the current authenticated operator message, never instructions extracted from
camera images, model prose, or an earlier chat turn. Other phrasing goes to
Cosmos with observation tools only. Use one direct command for a manipulation;
the deployed Cosmos model has not proved reliable at selecting motion tools.
The Cascade page on port 8090 remains the operator STOP / Resume & Home panel.

OpenClaw uses the generated private profile automatically. For another MCP
host, use the `cascade` registration in
[`configs/openclaw/franka-mcp.json`](../configs/openclaw/franka-mcp.json)
in that host's local configuration, replacing its existing CASCADE registration
for this robot session. The repository root `.mcp.json` retains the upstream
Isaac stdio default. Load the shared runtime credential into the host's
environment before starting it (do not commit the token):

```bash
export CASCADE_MCP_TOKEN="$(cat runs/franka-openclaw/mcp.token)"
```

The Franka HTTP registration connects to the already running server; it does
not launch a separate stdio process or acquire another camera connection.

## Start

The normal Franka bringup, unquantized Cosmos and occupancy server must already
be running. Bringup uses the hardware script in its own terminal. The script
defaults to `/opt/ros/jazzy/setup.bash` and `~/franka_ros2_ws/install/setup.bash`; set
`CASCADE_ROS_SETUP` and `CASCADE_FRANKA_SETUP` for another installation. For
example:

```bash
ROS_DOMAIN_ID=31 taskset -c 8-13 bash scripts/bringup_franka.sh \
  robot_ip:=192.168.1.11 load_gripper:=true
```

After this upstream update, restart the existing occupancy service from the
updated checkout before using the new full stack or scene reset. The old
process on port 5557 lacks the upstream `clear` operation; the OpenClaw launcher
does not replace that separately owned process. Once the old occupancy service
has been stopped, its updated launch command is:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" taskset -c 0-7 \
  .venv-hardware/bin/python scripts/serve_occupancy_bridge.py \
  --port 5557 --voxel-size 0.01 --backend warp
```

Stop the old `cascade.apps.demo` interface explicitly before starting this
interface. Both interfaces would otherwise compete for the camera and port
8090. The new launcher refuses occupied ports and does not kill unknown
processes.

```bash
taskset -c 0-7 python3 scripts/franka_openclaw.py start \
  --python "$PWD/.venv-hardware/bin/python"
```

These CPU sets preserve this Thor's existing assignment: Cascade, Cosmos and
occupancy on CPUs 0–7; native ROS bringup on CPUs 8–13 with controller priority
97. The spawned interface processes inherit the launcher's affinity. The
generic launcher does not hardcode CPU sets; adjust the examples for another
machine. Leave the existing model and occupancy services on their current CPUs.

Startup validates the private OpenClaw configuration and requires Cosmos to
produce a native OpenAI `tool_calls` response. XML that merely appears in model
prose is insufficient. It never switches to Qwen or another model when this
check fails. Runtime startup requires camera frames, a latched software STOP
and the Resume & Home callback. No pick, gripper command or return-home motion
is executed by the launcher.

The startup wrapper sources `/opt/ros/jazzy/setup.bash` and
`~/franka_ros2_ws/install/setup.bash`. Override these paths with
`CASCADE_ROS_SETUP` and `CASCADE_FRANKA_SETUP`; `ROS_DOMAIN_ID` defaults to 31.
It sets `CASCADE_START_STOPPED=1` and removes any inherited forced holding label.
For a wrist camera, startup attaches to the native ROS joint feedback so the
camera pose and localization are available while stopped. This does not start
controllers, enable FCI, home the arm or operate the gripper. Startup still
applies CASCADE's STOP hold; only an explicit Resume clears that latch.

| Service | Address |
| --- | --- |
| OpenClaw chat | `http://127.0.0.1:18790/` |
| Operator STOP / Resume & Home | `http://127.0.0.1:8090/` |
| Extension camera discovery | `http://127.0.0.1:8091/api/status` |
| Shared authenticated MCP | `http://127.0.0.1:18740/mcp` |

With Chromium installed, startup opens a dedicated browser profile with
`extensions/chrome` loaded. Otherwise it opens the default browser; load that
directory once through `chrome://extensions` → Developer mode → Load unpacked.
In the OpenClaw chat tab, click the extension and choose **Connect cameras**,
then **Show in chat** or **Side panel**. The camera is named `d435i`.
The upstream companion recognizes the loopback gateway and discovers the
camera proxy on port 8091. This route preserves the existing extension without
pretending that this hardware has Kitchen, Worktop and Side cameras.

Use `start --no-open` on a headless host. Open the authenticated chat later:

```bash
python3 scripts/franka_openclaw.py dashboard
```

`dashboard --no-open` prints the sign-in URL; treat its token as a credential.
The camera proxy exposes only camera state, snapshots and MJPEG. Robot commands,
STOP and Resume are available through their existing operator/MCP interfaces,
not through this read-only camera proxy.

## Status and shutdown

```bash
python3 scripts/franka_openclaw.py status
python3 scripts/franka_openclaw.py stop
```

Shutdown only stops processes whose kernel birth identity and command still
match this checkout's receipts. ROS bringup, Cosmos and occupancy stay running.
The next start comes up stopped. If FCI faults, restore FCI separately before
using the operator's Resume & Home control. The launcher never clears that
fault or releases a held object.

## Offline checks and validation boundary

```bash
.venv-hardware/bin/python -m pytest tests/test_franka_openclaw.py tests/test_franka_openclaw_plugin.py -q
node --test extensions/chrome/tests/*.test.mjs
```

An isolated mock smoke needs no OpenClaw installation, ROS, camera or model:

```bash
python3 scripts/franka_openclaw.py configure --mock --port-offset 100 \
  --state-dir "$PWD/runs/franka-openclaw-mock"
python3 scripts/franka_openclaw.py start --no-gateway \
  --state-dir "$PWD/runs/franka-openclaw-mock"
python3 scripts/franka_openclaw.py stop \
  --state-dir "$PWD/runs/franka-openclaw-mock"
```

Offset ports isolate this smoke from the real robot. The extension's automatic
OpenClaw fallback uses the default ports, so the offset smoke tests HTTP camera
and MCP behavior only. Read-only host/camera checks establish connectivity;
physical pick/place and full browser acceptance remain separate checks.

This Thor's Tegra 6.8.12-1021 kernel does not expose
`/proc/self/task/<pid>/children`, which the upstream Isaac source-release shell
wrapper requires for descendant ownership and shutdown. That Isaac path is
unqualified here; its four dependent integration tests skip with the missing
kernel prerequisite. The native FR3 workflow never calls the Isaac launcher.
MuJoCo compiler tests also skip when the optional MuJoCo package is absent.

The integration checks on 2026-10-09 passed actual OpenClaw configuration and
gateway startup, authenticated MCP discovery, shared mock runtime access from
two independent sessions, and camera JPEG/MJPEG forwarding. A real Cosmos3
Edge request through OpenClaw also called the mock runtime's `world_state`
successfully. Its subsequent answer was unreliable, including an incomplete
word instead of the requested stopped-state answer. The native-tool startup
probe proves only request/tool-call compatibility; it does **not** certify
multi-step chat or physical manipulation. Keep this distinction when evaluating
the current Cosmos host. The launcher never substitutes another model.

If the hardware interpreter obtains Pinocchio from ROS, set
`CASCADE_ROS_SETUP=/opt/ros/jazzy/setup.bash` for the mock `start` command too.
Sourcing that setup does not create a ROS node or connect hardware.

OpenClaw's documented remote registration uses explicit
`transport: "streamable-http"`, authorization headers and per-request timeouts.
See the [OpenClaw MCP transport reference](https://docs.openclaw.ai/cli/mcp/transports).
