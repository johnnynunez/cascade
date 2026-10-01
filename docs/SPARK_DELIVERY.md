# Spark delivery details

Current source and acceptance are indexed in
[project status, 1 October](PROJECT_STATUS_20261001.md). Runtime baseline `477c88f` contains
PRs #27 and #32–42 and passes the 3,205-test combined suite. The dated results
below retain their original pins and verdicts; they are not current-source
installation, campaign or restart claims.

Presenting the demo? Start with the [presenter card](PRESENTER_QUICKSTART.md).

Use [DGX Spark setup](DGX_SPARK_SETUP.md) for the install and start commands.
That guide pins the tested runtime source and reports the measured installation,
native-agent proof and browser scope.
The default event flow installs Isaac Sim, GraspGen-X, Qwen Q4, the vision projector,
llama.cpp, OpenClaw and the verified kitchen. Cosmos is not installed.

## Validated delivery, 30 September 2026

Source `9cf540279f09df8197eda6541311b41edb455243` completed installation and
the read-only check with exit 0 on an aarch64 NVIDIA GB10 Spark. Its first
desktop launch also exited 0, and `spark_verify.py` reported READY. The green
cube and orange both passed the independent placement, contact, camera and
reset audits. Installation took **12 min 21 s**; startup and proof took
**732.42 s**. These are single-run timings.

The checkout, Python environments, model source and model-file destinations
were newly created directories, with no symlinks to prior installations.
Host prerequisites already existed; the run used the normal user HOME and
shared download/uv caches. It is a fresh-destination installation, not an
empty-cache or fresh-OS measurement.

Real Chromium on the Spark loaded the shipped extension on an Xvfb virtual
display. Kitchen, Worktop and Side decoded at **1280 × 720**, with frame IDs
advancing from 10 to 13; chat was connected and Send/Start over were enabled.
This validates the real browser and extension, but not GNOME app-grid interaction
or a visible Isaac editor. The
[release receipt](../benchmark/results/spark_clean_delivery_20260930.json) and
[UI screenshot](../benchmark/results/images/spark_clean_three_cameras_20260930.png)
retain the measured scope. Neither this two-object proof nor earlier diagnostic
campaigns establish a success rate for arbitrary future requests.

After READY, the same installed stack passed a complete five-object round
through the visitor chat API. Every case used native Qwen tool selection,
the unchanged passive physics auditor, all three cameras and a native reset.
Installation and READY receipts remained unchanged throughout the campaign.

| Object | Destination | Physical placement and reset | Final center error |
| --- | --- | --- | --- |
| Green cube | Green square | PASS | 6.26 mm |
| Orange | Open box | PASS | 14.84 mm |
| Pink cube | Green square | PASS | 8.86 mm |
| Lemon | Open box | PASS | 21.02 mm |
| Tomato can | Green square | PASS | 14.28 mm |

The campaign took 20 min 6 s, including inspections, settling and resets.
Its [diagnostic](../benchmark/diagnostics/spark_native_campaign.py) submits
requests to the existing visitor session; it does not start another robot
controller. An earlier incomplete attempt is retained in the release receipt:
the diagnostic passed a string to a validator requiring `Path` and stopped
after a completed green-cube request and reset. That attempt is not counted
as accepted. The corrected campaign passed all five cases without changing
the installed runtime, model, physics or acceptance thresholds.

The stack was then stopped with `./run.sh down` and launched again using the
same desktop command. This second full launch reached **READY in 9 min 10 s**
with a new native session and MCP process. Both placement/reset audits passed;
final center errors were **8.31 mm** for the green cube and **10.22 mm** for
the orange. The Chromium page stayed open through the stop/start and recovered
connected chat, enabled controls and all three live cameras without a reload.
Frame IDs advanced from 51 to 55 in the
[post-restart screenshot check](../benchmark/results/images/spark_clean_three_cameras_after_restart_20260930.png).

Shutdown required SIGKILL for the owned Qwen process after its SIGTERM timeout.
All demo ports were released before the second launch; the personal OpenClaw
gateway on 18789 retained its PID and start time. After acceptance, only the
test Chromium and virtual display were closed, releasing the browser profile
for a desktop session. The clean installed demo remained running and its final
read-only check still reported READY. Both launches, the stop log and the final
service identities are retained in the release receipt.

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
scene hash and timestep before the OpenClaw proof. The installed defaults now
include six camera bridge iterations and the observed-finger gate. [Render-bound
frame history](isaac-render-frame-history.md) prevents a repeated renderer buffer
from acquiring a new capture timestamp. [Physical-time motion](ISAAC_MOTION_CLOCK.md)
uses 30 Hz nominal Isaac targets while preserving legacy 50 Hz safety edges.

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
measured cavity and floor. Every camera must advance during motion and reset, with sampled server capture
age at most two seconds. Client delivery age and observed gaps are separate
measurements; the audit does not prove continuous availability or absence of
unobserved neighbor contact.
The proof retains native tool results, physics samples, scene hashes,
screenshots and reset readbacks. A failed native placement or reset, refuted
placement or failed audit prevents READY. `--no-robot-turn` remains
STARTED / UNVERIFIED.

At the tested 30 September pin `9cf540279f09df8197eda6541311b41edb455243`,
the first launch recorded zero tool failures for both placement orders.
Qwen still said **"Placement unverified"** because that revision's skill
postcondition checked only the object's center. The independent physics audit separately
confirmed full containment, release and support for both objects, with final
center errors of **9.61 mm** for the green cube and **6.69 mm** for the orange.
READY comes from those independent checks; it does not rewrite the native
tool verdict or the chat response. Preserve that distinction when presenting
the result. Earlier diagnostic campaigns remain documented in
[GraspGen-X on Spark](GRASPGENX_SPARK.md).

## Development placement verdict, 1 October 2026

A separate direct-runtime campaign on the real GB10 Spark completed all five
kitchen objects: **5/5 placement verdicts were `confirmed`**, and every
external physical audit and reset passed. The
[placement-verdict receipt](../benchmark/results/spark_placement_verdict_20261001.json)
records this follow-up separately from the September release. This was one
five-object round, not a success-rate estimate for future requests.

Green cube and orange also passed a second direct-runtime campaign with the
final object/destination and multi-arm guards, again with external audits and
resets. All eight retained placement windows (seven successful calls and the
idle-cube negative control) reproduce their verdicts offline from the published
replay bundle. The final local suite passed 2,473 tests, plus 34 portable-bundle
tests; these software checks do not extend the live acceptance scope.

The new verifier applies to single-arm kitchen calls targeting the configured
green square or open box. After the motion, it collects fresh independent
physics readings and checks full collider containment, actual jaw release,
measured support and settling. A `confirmed` tool verdict covers that observed
final placement. It does not prove the preceding lift or transport, camera
evidence or reset; those remain separate checks in the external campaign.
Missing or inconsistent evidence remains `unverified`, and measured placement
failures are `refuted`. See the [runtime contract](ARCHITECTURE.md#the-execute-choke-point)
for the bounded observation window and per-call receipts.

This direct-runtime campaign does not establish a new native-chat or UI
acceptance, full desktop READY proof, fresh installation or deployment of the
final source. The September installation result remains historical. The setup guide now
selects runtime baseline `477c88f`; its current acceptance is recorded separately in the
[status index](PROJECT_STATUS_20261001.md).

## Native confirmation gate and failed follow-up, 1 October 2026

The additional offline gate in
[`spark_native_confirmed.py`](../benchmark/diagnostics/spark_native_confirmed.py)
requires each native placement to report a case-bound `confirmed` physics
postcondition and a successful return home. It also requires the existing full
placement/camera/reset audit. A successful final placement alone cannot satisfy
this gate. It reads saved evidence without issuing robot commands or rewriting
the native outcome. Trace attribution uses the owning MCP process, unique step,
exact arguments and observed turn intervals; trace rows do not contain native
OpenClaw run or visitor order IDs.
Native run IDs are reported as declared by the saved turn files. Swapping two
otherwise valid envelopes from the same session cannot independently be
detected from those IDs; physical case attribution still uses the exact
arguments and observed trace intervals.

Run it after the normal launcher proof, then again with the five-object native
campaign receipt:

```bash
python3 benchmark/diagnostics/spark_native_confirmed.py \
  --proof runs/.launch/profile-cascade-demo/proof.json \
  --output runs/native-confirmed-proof.json
python3 benchmark/diagnostics/spark_native_confirmed.py \
  --proof runs/.launch/profile-cascade-demo/proof.json \
  --campaign runs/native-campaign/campaign.json \
  --output runs/native-confirmed-campaign.json
```

Outputs must be new files. For an archived repository layout, pass
`--evidence-root /path/to/archive` and point `--proof` and `--campaign` inside
that archive. The report records hashes of every input and the checker source;
file modification times are not used as evidence.

A normal desktop launch from clean source `0fee56f0c4891f6d85362d5cc38f6b58879c370f`
passed the green-cube placement, native physics confirmation and full reset.
The orange failed both grasp attempts, with the final attempt reporting an air
grasp and a `refuted` placement. The launcher withheld READY and closed its
owned services. The saved
[negative gate report](../benchmark/results/spark_native_confirmed_negative_20261001.json)
accepts the completed green case and rejects the incomplete orange case and
proof. It is failure evidence, not a completed two-object acceptance.

A subsequent normal launch of the unchanged September source `9cf5402` also
failed the orange, this time ending with `did not settle at grasp pose`.
Therefore the old successful September run is historical evidence, not a
demonstration that restoration currently succeeds. Neither attempt completed
the new five-object native campaign, UI acceptance or restart acceptance.
The candidate reused the recorded baseline dependencies and assets in a
separate application environment; it was not a fresh dependency installation.
The grasp-memory file and failed receipts were preserved before diagnostic
instrumentation. Root cause remains unestablished by these two attempts.

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
