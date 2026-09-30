# Spark delivery details

Presenting the demo? Start with the [presenter card](PRESENTER_QUICKSTART.md).

Use [DGX Spark setup](DGX_SPARK_SETUP.md) for the install and start commands.
That guide names the current release candidate and distinguishes pending clean
installation/native-agent acceptance from earlier hardware measurements.
The default event flow installs Isaac Sim, GraspGen-X, Qwen Q4, the vision projector,
llama.cpp, OpenClaw and the verified kitchen. Cosmos is not installed.

## Installation identity

| Component | Project-owned location and identity |
| --- | --- |
| App | `.venv`, Python 3.12; Spark torch 2.14.0+cu130 and torchvision 0.29.0+cu130 |
| Isaac Sim | `.isaacsim`; `isaacsim[all,extscache]==6.1.0.0`, torch 2.11.0+cu130 and tinyobjloader 2.0.0rc13 |
| GraspGen-X | `.graspgenx`, Python 3.12 and torch 2.14.0+cu130; `.graspgenx-src` with pinned source, full generator/discriminator checkpoints and gripper assets |
| Qwen | `models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Vision | `models/qwen3.8-27b/mmproj-BF16.gguf` |
| llama.cpp | `.llama.cpp`, pinned source and local CUDA build receipt |
| OpenClaw | `.openclaw-cli/bin/openclaw`, version 2026.9.3; profile `cascade-demo` |
| Kitchen and props | `demo/cocina_asier.py`, verified room bundle in `demo/scene/cocina_asier/`, `demo/own_kitchen_props.py` and `demo/scene/props/`; source and bundle manifests verify their bytes |

The model manifest is
[`deploy/brev/profiles/qwen3.8-27b-q4.json`](../deploy/brev/profiles/qwen3.8-27b-q4.json).
It pins public repository revisions, file sizes and SHA-256 digests.
The model alias is `Qwen/Qwen3.8-27B`; reasoning is off.
The installer fetches the model and projector into the project and builds
its own aarch64 llama.cpp. It does not need a preinstalled model server.

The installer downloads the pinned owner-authorized Cocina Asier room archive
and verifies every extracted asset. The checkout supplies the original orange,
platter and task prop authoring code. It verifies those source files against
[`own_assets.json`](../demo/scene/own_assets.json). See the
[scene provenance notice](../demo/scene/NOTICE.md) and
[original prop dedication](../demo/scene/props/LICENSE.txt).

`--check` downloads nothing. It verifies the installed runtime, model
identities, robot assets and kitchen files. A passing check proves preparation.
Live acceptance is a separate step.

## Runtime and isolation

Spark uses the working Brev path: PhysX on CUDA, the `isaac_kitchen_gpu`
arm profile, the shipped kitchen scene and a 1/120 second physics step.
GraspGen-X runs its real CUDA diffusion model; startup checks an actual
inference before enabling robot tools. Occupancy/nvblox is disabled, and JEv
is outside the presenter installation and runtime. Startup checks the live engine, CUDA context,
scene hash and timestep before the OpenClaw proof.

The app, Isaac and model runtime stay separate. A selected complete Isaac
6.1 source release can be reused without package writes to its embedded
Python; the fresh install uses the managed runtime above.

`runs/.install/install.json` records explicit EULA consent, source identity
and selected runtime. `runs/.install/env.sh` restores that installation's
settings. The desktop launcher supervises the complete stack.

The dedicated OpenClaw profile uses native CASCADE tools, a local Qwen API
and its own workspace and memory under `runs/.launch/profile-cascade-demo/openclaw/`.
The gateway runs as a project-owned foreground process on loopback port
18790. It creates no systemd service. The personal gateway on 18789 is separate. Startup fails on an
unrelated occupied port; it does not adopt or stop that service.
Process receipts bind the checkout, profile, command, PID and process birth.
`./run.sh down` stops only processes belonging to the installation.
The Spark launch passes `--no-judge`: the optional keyframe judge in
`configs/demo.yaml` targets the personal gateway with a token from
`~/.openclaw`, so it never runs here. The physical audit is the gate.

## Natural English requests

The attendee profile presents six high-level tools: scene inspection, object
location, pick and place, reset, camera images and session status. Qwen chooses
the native call with `tool_choice: auto`. The adapter does not select intent
from keywords, narrow the catalog for an order or rewrite the attendee's message.
Advanced tools remain available outside this profile.

The startup proof uses the same natural English requests shown in the setup
guide. Scene answers must be grounded in current images, and genuine ambiguity
requires clarification. Failed and unverified tool results remain visible in
the English answer; the adapter never upgrades them to success.

## READY checks

The launcher invalidates old success before starting a new proof.
Both orders enter through OpenClaw in one session and one bound simulation:

1. Inspect the scene and move the green cube to the green square.
2. Verify the physical placement, reset every prop and inspect again.
3. Move the orange into the open box.
4. Verify the physical placement, reset every prop and inspect again.

The independent observer reads the live simulation. It requires real lift,
both finger contacts, release, settled support and the whole collider inside
the destination. The orange uses its measured convex collider and the box's
measured cavity and floor. Every camera must advance during motion and reset.
The proof retains native tool results, physics samples, scene hashes,
screenshots and reset readbacks. A failed native placement or reset, refuted
placement or failed audit prevents READY. `--no-robot-turn` remains
STARTED / UNVERIFIED.

The previous scene's installation acceptance needed two orange grasp attempts. Its placement passed the
independent audit, but the return-home substep reported `did not settle at home`.
The following native reset and final inspection passed. Reset and inspect
before another order if this warning appears. The native tool's center-only
postcondition remains unverified; the independent audit establishes containment,
release and support.

The previous scene's natural-language acceptance also passed inspection, green cube placement,
orange placement and same-world resets with native request/response traces.
An initial orange attempt failed at the pre-grasp pose and was reported plainly.
After a natural reset and cleanup of completed test sessions, a retry passed
the unchanged physical audit. This does not establish the cause of the first
failure. No controller or physics setting was changed for that retry.
These historical results do not establish acceptance of the original kitchen on Spark.

## Quick recovery during the demo

Use the in-session reset for scene/task state. `scripts/booth_reset.sh
--wipe-brain` is an older booth workflow: it deletes the default grasp-memory
file, while a running MCP server retains its in-memory copy and can save it
again. Spark profiles use their own memory paths. Deleting a file therefore
does not establish that a live session has reset, or explain a failed startup
proof without the failing trace. Restarting with the flags below does not
require deleting memory files.

If the stack is running and you only need to put the objects back, use
**Start over** in the Spark demo chat, **Reset** in the PAAI OpenClaw interface,
or send **"Reset the scene."** in the current conversation.
This keeps Isaac, the model and OpenClaw running;
it returns the arm home, restores the props, clears object beliefs and task
images, and captures a fresh observation. Learned grasp history, recent text
memory and chat history remain. It does not rerun the startup pick-and-place tests.
These buttons are unavailable while another order is running. Wait for the
reset result and check the camera views before the next order.

After a crash, add `--no-robot-turn --no-judge` to the usual launch command
to skip the startup manipulation tests and optional image judge. Preserve
the engine, scene and other flags used for the demo:

| Engine | Quick launch command |
| --- | --- |
| PhysX presenter | `./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge` |
| Newton, separately rehearsed | `./run.sh isaac --engine newton --graspgenx local --no-robot-turn --no-judge` |

The launcher reuses matching services that are still running and starts
missing ones. Simulator initialization, connection, scene identity, robot
tool and model checks still run. The resulting banner is
`STARTED (UNVERIFIED: robot proof skipped)`; a fast restart does not create
a new physical acceptance result. Qwen must already be running: this lower-level
command does not start or supervise it. The desktop entry does not accept these
skip flags, and its camera/chat surfaces require a verified proof to attach.
For the complete supervised interface, use
`python3 scripts/desktop.py launch --repo "$PWD"` (add `--gui` for visible Isaac).
That path starts Qwen and runs the normal READY proof when needed.

## Fresh-install acceptance

A release receipt must record the fresh source root, initially empty
dependency/model destinations, exact installer command, normal user HOME and
any reused download or package caches. A fresh installation may reuse the
user's uv cache; that does not make it a cache-free or private-HOME experiment.
Do not reuse an existing application/model environment or substitute a global
model path for the new checkout's destinations.
The install and a later read-only check must exit 0 before live acceptance.
Record actual Isaac build, physics engine/device, model files and kitchen
identity. Keep the unrelated-service inventory before and after the run.
The GraspGen-X check must validate pinned source, package versions and full
checkpoint bytes rather than just the presence of filenames or Git LFS pointers.
The live launch must then verify actual learned inference.

For a bandwidth-saving rehearsal, `CASCADE_MODEL_MIRROR_URL` can point the
model download code at a local HTTP mirror of the exact verified Q4 file.
The destination must start empty. The normal manifest still supplies the
public revision, size and checksum. The receipt must identify the mirror
and separately record a public metadata/ranged-download check.

Accept the release only after both native placement cases and resets pass. Store its
clean-install and acceptance receipts with the deployment evidence.
