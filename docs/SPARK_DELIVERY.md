# Spark delivery details

Use [DGX Spark setup](DGX_SPARK_SETUP.md) for the install and start commands.
The default event flow installs Isaac Sim, Qwen Q4, the vision projector,
llama.cpp, OpenClaw and the verified kitchen. Cosmos is not installed.

## Installation identity

| Component | Project-owned location and identity |
| --- | --- |
| App | `.venv`, Python 3.12; Spark torch 2.14.0+cu130 and torchvision 0.29.0+cu130 |
| Isaac Sim | `.isaacsim`; `isaacsim[all,extscache]==6.1.0.0`, torch 2.11.0+cu130 and tinyobjloader 2.0.0rc13 |
| Qwen | `models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf` |
| Vision | `models/qwen3.8-27b/mmproj-BF16.gguf` |
| llama.cpp | `.llama.cpp`, pinned source and local CUDA build receipt |
| OpenClaw | `.openclaw-cli/bin/openclaw`, version 2026.9.3; profile `cascade-demo` |
| Kitchen | `demo/scene/assets/` and `demo/vendor-kitchen/`; all 166 manifest members verified |

The model manifest is
[`deploy/brev/profiles/qwen3.8-27b-q4.json`](../deploy/brev/profiles/qwen3.8-27b-q4.json).
It pins public repository revisions, file sizes and SHA-256 digests.
The model alias is `Qwen/Qwen3.8-27B`; reasoning is off.
The installer fetches the model and projector into the project and builds
its own aarch64 llama.cpp. It does not need a preinstalled model server.

The kitchen comes from the `kitchen-v1` release. Its archive is 705,250,534
bytes with SHA-256
`c94c2826180e295e4df16d799b5b3f581c9a2e60b08f0c34a99831ca1f80e35d`.
The installer checks the archive, then every member against
[`bundle_assets.json`](../deploy/brev/bundle_assets.json).
It rejects unexpected members, links and unsafe paths. The Lightwheel
CC BY-NC and orange asset notices remain with the installed files.
The release is an installer detail. Do not add the unpacked asset tree to Git.

`--check` downloads nothing. It verifies the installed runtime, model
identities, robot assets and kitchen files. A passing check proves preparation.
Live acceptance is a separate step.

## Runtime and isolation

Spark uses the working Brev path: PhysX on CUDA, the `isaac_kitchen_gpu`
arm profile, the shipped kitchen scene and a 1/120 second physics step.
Occupancy is disabled. Startup checks the live engine, CUDA context,
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

The tested orange case needed two grasp attempts. Its placement passed the
independent audit, but the return-home substep reported `did not settle at home`.
The following native reset and final inspection passed. Reset and inspect
before another order if this warning appears. The native tool's center-only
postcondition remains unverified; the independent audit establishes containment,
release and support.

## Clean-room acceptance

A release receipt must record a fresh source root, private HOME/cache tree,
empty dependency/model destinations and the exact installer command.
No inherited app environment or global model path counts as a fresh install.
The install and a later read-only check must exit 0 before live acceptance.
Record actual Isaac build, physics engine/device, model files and kitchen
identity. Keep the unrelated-service inventory before and after the run.

For a bandwidth-saving rehearsal, `CASCADE_MODEL_MIRROR_URL` can point the
model download code at a local HTTP mirror of the exact verified Q4 file.
The destination must start empty. The normal manifest still supplies the
public revision, size and checksum. The receipt must identify the mirror
and separately record a public metadata/ranged-download check.

Publish only after both clean-room cases and resets pass. Store the release's
clean-install and acceptance receipts with the deployment evidence.
