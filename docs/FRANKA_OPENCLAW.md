# Franka Research 3 on the unified launcher

**Next action:** with the ROS bringup and the brain already running ([Start](#start)), launch the
FR3 (it comes up with a latched STOP and observe-only; about 5 min):

```bash
taskset -c 0-7 ./scripts/launch.sh --sim none --arm fr3 --cameras d435i
```

The FR3 runs on the same route as every real arm (the reBot's): `scripts/launch.sh` with the agent
(OpenClaw 2026.9.3) inside an NVIDIA OpenShell sandbox ([OPENSHELL.md](OPENSHELL.md)), Nemotron 3.5
Bolt as the brain and the object identifier (`detector: type: brain`), and the GraspGen-X sidecar.
`--arm fr3` adds the Franka operator contract:

| Piece | Where |
|---|---|
| ROS bringup: native `franka_ros2` arm + gripper controllers (`load_gripper:=true`), no MoveIt | `scripts/bringup_franka.sh`, its own terminal; the launcher **checks** it, never starts it |
| Brain: Nemotron 3.5 Bolt on llama.cpp | `scripts/serve_nemotron_bolt_llamacpp.sh`, a user unit |
| Robot server: FR3 + D435i, software STOP latched at start | user unit `cascade-openshell-mcp`, 127.0.0.1:18741, ROS env via `scripts/ros_env.sh` |
| Operator STOP / Resume & Home | `http://127.0.0.1:8090/` (the robot server's live view) |
| Agent with the FR3 operator profile | sandbox `cascade-agent`, chat `http://127.0.0.1:18799/` |
| GraspGen-X | :5556 (`--graspgenx local` starts it; the default expects one running) |
| Occupancy (supports `clear`) | :5557, started by the launcher |

The FR3 operator profile (`deploy/openshell/franka-plugin`, `up --agent-profile franka`): the
command router compiles explicit routine commands into native tool calls, `instructions.md` is the
agent's `AGENTS.md`, the chat sees only the FR3 operator tools, and heartbeats, scheduled jobs and
the generic memory plugin are off: actions start only in the operator's chat. `ROS_DOMAIN_ID`
defaults to 31. Robot config: `configs/arms/fr3.yaml`; the only D435i mount correction is the
optical transform `T` in `configs/cameras/d435i.yaml` (1280×720 RGB at 15 FPS plus aligned depth).

**Safety caveat:** the optical transform does not describe the printed mount's collision envelope,
and that envelope is **not modeled**. FR3 checks spatial-motion orientation on its wristward TCP −Z
axis and does not inherit the reBot camera box; joint, TCP, table and occupancy checks stay active.

## Prepare the host (once)

1. Python: the launcher uses `$PY`, else `.venv/bin/python` (`PY=$PWD/.venv-hardware/bin/python`
   for the existing hardware env). On Jetson install NVIDIA's torch first; the launcher otherwise
   installs the PyPI build.
2. Brain, built for this GPU (`nvidia-smi` compute capability; Jetson Thor is `110`, set
   `CASCADE_LLAMA_CUDA_ARCH=110` if `nvidia-smi` cannot report it). Stop the old Cosmos server on
   :8082: it is no longer used and holds unified memory.

```bash
scripts/serve_nemotron_bolt_llamacpp.sh --setup-only
systemd-run --user --unit=cascade-brain --collect -p MemoryMax=infinity -p CPUAffinity=0-7 \
    -E PORT=8081 -E API_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key" \
    "$PWD/scripts/serve_nemotron_bolt_llamacpp.sh"
export CASCADE_NEMOTRON_BOLT_BASE_URL=http://127.0.0.1:8081/v1
export CASCADE_BRAIN_KEY_FILE="$HOME/.cascade/nemoclaw/llama.key"
```

   The vLLM route (`scripts/serve_nemotron_bolt_vllm.sh`, NVFP4 KV cache) is not for the Thor yet:
   its b12x kernels and NVFP4 KV path target SM120/SM121.
3. GraspGen-X: `scripts/install_graspgenx.sh` (CUDA 13 torch; needs JetPack 7's driver), then
   launch with `--graspgenx local`. `--graspgenx none` keeps the profile's analytic planner.
4. Docker for the sandbox (the launcher installs the pinned OpenShell 0.1.3 itself), then the
   preflight: `./scripts/launch.sh --sim none --arm fr3 --cameras d435i --check`.

## Start

1. **Bringup** in its own terminal (defaults `/opt/ros/jazzy/setup.bash` and
   `~/franka_ros2_ws/install/setup.bash`; override with `CASCADE_ROS_SETUP` / `CASCADE_FRANKA_SETUP`,
   which the launcher and the robot server's unit source too):

```bash
ROS_DOMAIN_ID=31 taskset -c 8-13 bash scripts/bringup_franka.sh \
  robot_ip:=192.168.1.11 load_gripper:=true
```

2. **Stop the old stack** (`python3 scripts/franka_openclaw.py stop`, any old `cascade.apps.demo`
   interface, an occupancy server from another checkout): the launcher refuses a busy :8090 and a
   foreign :5557 and never kills a process it did not start.
3. **Launch** (the command at the top; add `--graspgenx local` and `--occupancy warp` as needed).
   CPU sets keep this Thor's assignment: Cascade, brain and occupancy on 0–7 (the launcher pins the
   robot server's unit to its own `taskset`; `CASCADE_MCP_CPUS` overrides), ROS bringup on 8–13.
   In order, the launcher:
   1. sources the ROS setup, requires :8090 free and checks the bringup, read-only: joint states,
      the `fr3_arm_controller` trajectory action (controller ACTIVE) and the Franka Gripper actions
      (`scripts/franka_rig.py bringup`);
   2. starts occupancy and checks GraspGen-X inference;
   3. resolves the brain (`--brain auto` = Nemotron 3.5 Bolt, never a Cosmos left running; it also
      identifies objects, `CASCADE_PERCEPTION_*`) and builds the runtime once (camera, ROS feedback);
   4. starts the robot server with `CASCADE_START_STOPPED=1` and waits until its live view shows
      the D435i streaming, the STOP latched and Resume & Home available (`franka_rig.py ready`);
   5. brings the sandbox up **observe-only** with the franka profile, proves the brain answers from
      inside it, and opens the chat with the camera extension (`--no-open` skips the browser).
4. **Let the arm move** when the operator is ready: **Resume & Home** on `http://127.0.0.1:8090/`
   (clears the STOP after readiness checks, returns home without changing the jaws), then
   `scripts/openshell_rig.py motion unlock --for 20m` (re-locks itself; `motion lock` ends it).

The camera extension reads the robot server's live view (MJPEG on :8090), never a second
RealSense pipeline. If FCI faults, restore it separately before Resume & Home: nothing here clears
that fault or releases a held object.

## Commands in chat

1. `pick up the crayon` (grasp only), `pick up the crayon and place it in the box`,
   `put it in the box` (place the held object)
2. `return home`, `open the gripper`, `close the gripper`
3. `stop` (Resume is the operator page: the chat answers with its address and dispatches nothing)
4. `what do you see?`, `robot status`, `show me the camera`

These show as **Direct command** and report the real tool result, failures included: at most one
tool per request, never an automatic STOP clear or retry, and only from the current authenticated
operator message. Other phrasing goes to the brain with observation tools only. OpenShell refuses
`reset_stop` for the agent always and every motion tool while observe-only.

## Status and shutdown

```bash
scripts/openshell_rig.py status
./scripts/launch.sh --down
```

`--down` stops the chat forward, the sandbox, the robot server and the sidecars this launch
started; ROS bringup and the brain keep running, and the next start comes up stopped and
observe-only.

## Offline checks and validation boundary

```bash
./scripts/launch.sh --sim none --arm fr3 --cameras d435i --dry-run
.venv/bin/python -m pytest tests/test_launch_franka_route.py tests/test_franka_rig.py \
  tests/test_openshell_rig.py tests/test_franka_openclaw_plugin.py tests/test_franka_command_router.py -q
```

`--arm fr3_mock --cameras mock --graspgenx stub` rehearses the same route without ROS on any Linux
host with Docker and a brain. Validated without the robot: the launcher's FR3 path end to end with
its host boundaries doubled, the real robot server's latched-STOP contract, the agent image's
OpenClaw loading the FR3 plugin and routing webchat commands to the real robot server (mock FR3),
and FR3 picks in MuJoCo. Not validated: the Thor (aarch64, sm_110) builds, OpenShell on its Tegra
kernel, ROS bringup and DDS discovery from the robot server's unit, the D435i, and any physical
motion.

## Legacy route

`scripts/franka_openclaw.py` (a private OpenClaw CLI, the Cosmos3-Edge brain on :8082, MCP on
:18740, `configs/openclaw/franka-mcp.json` for other hosts) is kept until this route is accepted
on the robot. It loads the same plugin and instructions from `deploy/openshell/franka-plugin`.
