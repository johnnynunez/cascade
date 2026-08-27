# CASCADE 🦾 — Cascaded Agentic Skill Control with Adaptive Dispatch and Execution

<p align="center">
  <a href="https://github.com/johnnynunez/cascade/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/johnnynunez/cascade/ci.yml?branch=main&style=flat-square&label=ci" alt="CI status"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/docs-architecture-informational?style=flat-square" alt="Architecture docs"></a>
</p>

CASCADE is a **hardware-agnostic** framework for agentic manipulation:
cameras, arms, and LLM backends are all pluggable behind one curated skill
API, so the same 30 skills, safety harness, and traces work whether the
backend is a RealSense D455F/D435i or a generic UVC webcam, a real 6-DoF
arm over CAN or a simulated one in Isaac Sim, and a cloud LLM or a local
one. Its defining idea is the cascade itself: routine commands resolve on a
regex reflex or a learned habit tier and never touch the LLM, which only
gets called when both fail — behavior, safety, and tracing stay identical
regardless of which tier (or which hardware) acted.

The reference deployment drives a Seeed reBot DevArm B601 (RobStride build)
from an NVIDIA DGX Spark — that is *a* configuration this framework runs on,
not what it is. It is the agentic evolution of the
[reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)
baseline, designed after NVIDIA GEAR's
[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) (curated skill API +
multimodal traces + skill library) and
[Agentic-VLA](https://arxiv.org/abs/2605.22896) (task decomposition + VLM
advisor + experience memory).

[Architecture](docs/ARCHITECTURE.md) · [Quickstart](docs/QUICKSTART.md) · [Booth runbook](docs/BOOTH_RUNBOOK.md) · [Roadmap](docs/ROADMAP.md) · [Agent guide](CLAUDE.md)

```
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│       chat: Hermes / OpenClaw /        │ │    CLI: --task / --interactive REPL    │
│       Claude Code / Codex / ...        │ │    (offline, scripted, or mock LLM)    │
└────────────────────────────────────────┘ └────────────────────────────────────────┘
                     │                                          │
   MCP stdio: host's own LLM picks tools    AgentOrchestrator: reflex -> habit -> LLM
                     ▼                                          ▼
                     └────────────────────┬────────────────────┘
              21 traced skills, one surface for MCP + CLI (ASPIRE-style)
        ▼                ▼                ▼                ▼                ▼
 get_observation   grasp_object       move_home     emergency_stop    recall_memory
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  perception  │ │   grasping   │ │   control    │ │    safety    │ │    memory    │
│──────────────│ │──────────────│ │──────────────│ │──────────────│ │──────────────│
│ RS/UVC/Isaac │ │ GraspGen-X   │ │ FK/IK (pin)  │ │ harness gate │ │ episodic +   │
│ cams, YOLO   │ │ + OBB fallbk │ │ min-jerk CAN │ │ +occupancy   │ │ belief/habit │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Install

Everything runs through the shared uv venv — there is no bare `python` on
the rig and `python3` alone has no pytest.

```bash
# macOS / Linux / DGX Spark
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/install_occupancy_backend.sh | bash
```

That installs the optional occupancy/collision-map backend (pyzmq,
msgpack-numpy, Open3D — same code path on CPU and on an NVIDIA GPU, CUDA:0
auto-detected at runtime). It is one piece of a larger environment; the full
one-shot rig bring-up — OpenClaw CLI, the Cosmos3-Edge (vLLM) brain, and
registering this repo's skills over MCP — is:

```bash
curl -fsSL https://raw.githubusercontent.com/johnnynunez/cascade/main/scripts/bootstrap.sh | bash
```

`scripts/bootstrap.sh` needs an NVIDIA GPU for the default brain (fails
fast with a clear message if `nvidia-smi` isn't found — use `--brain qwen`
or `--brain skip` on a CPU-only box). For a from-source checkout instead:

```bash
git clone https://github.com/johnnynunez/cascade.git && cd cascade
PY=/home/johnny/Projects/demo/.demo/bin/python scripts/setup_env.sh
```

See the [installation notes](#setup-notes) below for pyrealsense2, YOLOE's
text encoder, and other rig-specific gotchas.

## Quick start

```bash
# offline wiring check: mock camera + mock arm + scripted LLM, no hardware.
# Routine commands run on the REFLEX fast path (no LLM); a livestream
# dashboard with N camera streams + robot narration prints its URL.
PYTHONPATH=src python -m cascade.apps.demo --task "pick and place pink object"

# tests (unit/integration; live-hardware tests deselected by default)
python -m pytest tests/ -q
python -m pytest tests/ -m hardware -q     # needs a RealSense camera (profile: l515) + can0 up (read-only)

# learned grasps (the default backend) need the GraspGen-X server running;
# without it grasping silently falls back to the analytic OBB planner
scripts/serve_graspgenx.sh

# real rig, N cameras (first = manipulation camera), cloud LLM fallback
sudo ip link set can0 up type can bitrate 1000000
python -m cascade.apps.demo --interactive \
    --cameras d455f,uvc4k --arm rebot_rs --llm anthropic

# two D455F (serials pinned in the profiles: with identical hardware an
# unpinned profile binds to whichever unit enumerates first). The live
# window shows one row per camera, RGB beside its depth colormap.
python -m cascade.apps.demo --interactive \
    --cameras d455f_wrist,d455f_scene --arm rebot_rs --llm anthropic

# Isaac Sim instead of hardware (same demo, simulated reBot):
#   1. inside Isaac Sim's python:  python.sh scripts/isaac_bridge.py --usd <rebot.usd>
#   2. then:
python -m cascade.apps.demo --cameras isaac --arm isaac --interactive

# real rig, local Qwen3.6 on the Spark (start the server first)
scripts/serve_qwen_llamacpp.sh          # or serve_qwen_vllm.sh (MTP spec decoding)
python -m cascade.apps.demo --task "..." --cameras d455f --arm rebot_rs --llm local_qwen

# talk to it through OpenClaw's web chat instead of the CLI loop
./scripts/openclaw_demo.sh              # registers skills, wires the Cosmos3-Edge brain, opens chat
```

## How it fits together

- **[Perception](src/cascade/perception/)** is camera-agnostic: every
  backend yields `Frame` objects with float32 *metric* depth aligned to
  color. Cameras without depth fall through a strategy chain (sensor →
  optional mono-depth plugin → table-plane ray-casting), so an RGB-only
  webcam still grasps tabletop objects.
- **[Safety](src/cascade/safety/)** is a fail-closed `SafetyHarness` that
  vets every streamed waypoint (joint limits, velocity caps, workspace AABB,
  table-plane clearance, keep-out zones, perception watchdog, e-stop latch,
  and an optional [nvblox-style occupancy map](src/cascade/perception/occupancy.py)),
  plus an ASPIRE-style multimodal trace (`trace.jsonl` + before/after
  keyframes) on every skill call.
- **[Grasping](src/cascade/grasping/)** uses learned 6-DoF grasps from a
  GraspGen-X ZMQ server, falling back to analytic 3D OBB grasps whenever the
  server is down. Candidates are re-ranked by a persisted grasp-outcome
  memory, then vetted against IK *and* the safety-harness geometry.
- **[Control](src/cascade/control/)** is self-contained Pinocchio FK/IK on
  the RS URDF, damped-least-squares with random restarts, min-jerk joint
  streaming with feedback-based settling over RobStride CAN.
- **[Memory](src/cascade/memory/)** is a 10–15 s episodic window plus an
  object-permanence belief store, so "the mug you saw 10 seconds ago" is
  still actionable after occlusion.
- The **[agent](src/cascade/agent/)** dispatches through three tiers —
  reflex (regex, no LLM) → experience (learned habits) → LLM — so routine
  commands never wait on the model, with a VLM advisor kicking in on failure.

cascade works with hosted and local [LLM backends](#llm-backends) and is
exposed as an [MCP server](#run-it-under-any-mcp-agent-platform) any
MCP-capable host can drive.

## LLM backends

| profile | backend | notes |
|---|---|---|
| `anthropic` | Claude (cloud) | `ANTHROPIC_API_KEY`; vision + tools |
| `local_qwen` | local Qwen via llama.cpp / vLLM on the Spark | OpenAI-compatible; MTP speculative decoding (~1.4–2.2× decode) |
| `local_cosmos` | NVIDIA Cosmos3-Edge Reasoner via vLLM on the Spark | `scripts/serve_cosmos_vllm.sh` (:8082); 2.44B MoT, thinking on by default — see the script header for the day-one serving pitfalls it works around |
| `local_cosmos_sglang` | same Cosmos3-Edge Reasoner via SGLang instead of vLLM | `scripts/serve_cosmos_sglang.sh` (:8083); same `type: cosmos3` client (chat template is a property of the checkpoint, not the engine) — **UNVERIFIED**, first booth run against it is the verification pass |
| `openai` | any OpenAI-compatible cloud endpoint | `OPENAI_API_KEY` |
| `mock` | scripted | tests / wiring checks |

> **Keep the profile in sync with the server:** `configs/llm/local_qwen.yaml`
> pins `model: Qwen/Qwen3.6-27B`, matching what both serve scripts load.
> llama.cpp ignores the request's model name (vision comes via the mmproj
> projector its script downloads); vLLM rejects a mismatch — if you change
> the script's `MODEL`/`HF_REPO`, update the profile's `model:` (and
> `supports_vision:` if the server lacks a vision path) to match.

## Run it under any MCP agent platform

The whole skill runtime is also exposed as an **MCP stdio server**
(`cascade/apps/mcp_server.py`) — so instead of the built-in loop, any
MCP-capable agent platform can drive the arm. The agent gets the same 30
safety-gated skills (only the loop-internal `task_done` is excluded) plus
seven gateway extras — `camera_snapshot` (returns a live JPEG the agent can
*see*), `world_state`, `live_view_url`, `robot_knowledge`,
`verify_last_action`, and `emergency_stop`/`reset_stop` — 37 tools total
(see [The 30 skills](#the-30-skills) below for what each one does). Safety
harness, tracing, memory and the always-on camera window are identical —
only the brain swaps.

One registrar for every host — prints what each platform needs, `--write`
applies the file edits (preserving unrelated entries):

```bash
python scripts/setup_agents.py                     # show all hosts
python scripts/setup_agents.py --host codex --write
python scripts/setup_agents.py --camera d455f --arm rebot_rs --write
```

| platform | mechanism | setup |
|---|---|---|
| **Hermes** | `~/.hermes/config.yaml` `mcp_servers` | `./scripts/hermes_demo.sh` (interactive: register + test + chat) |
| **Claude Code** | project `.mcp.json` (ships in this repo; interpreter path is machine-specific, and it pins the Isaac camera/arm profiles) | if your checkout lives elsewhere, regenerate with the profiles you want: `setup_agents.py --host claude --camera isaac,isaac_side --arm isaac --write` (add `--python <interpreter>` if your venv is not at `<checkout-parent>/.demo`); user-scope: `--host claude` prints the `claude mcp add` one-liner |
| **Claude Desktop** | `claude_desktop_config.json` | paste the JSON block from `setup_agents.py --host claude` |
| **Codex CLI** | `~/.codex/config.toml` `[mcp_servers.cascade]` | `setup_agents.py --host codex --write`, verify with `codex mcp list` |
| **OpenClaw** | native `mcp.servers` (2026+) or [mcporter](https://docs.openclaw.ai/cli/mcp) | `./scripts/bootstrap.sh` (installs the OpenClaw CLI + Cosmos3-Edge brain + registers skills, one shot) or `./scripts/openclaw_demo.sh` if OpenClaw and the brain are already running (register + local-brain provider + gateway + web-chat URL); `setup_agents.py --host openclaw` prints the `openclaw mcp add` one-liner + JSON block. OpenClaw blocks the `PYTHONPATH` env — the package must be editable-installed in the venv (the script handles it) |

The server pre-warms perception at startup (cameras + detector + world
model) while the ARM stays unpowered until the first motion command
(LazyArm) -- so "pick and place pink object" from the chat is a single
`pick_and_place` tool call that starts moving immediately. Stopping is
never queued behind a running motion: `emergency_stop` frames are handled
out-of-band by the stdin reader, Esc/cancellation in the host mid-motion
freezes the arm, first Ctrl+C on the server latches the e-stop (no
free-fall), and the dashboard STOP button works from any browser on the
LAN. For attendee-facing sessions, `CASCADE_HIDE_TOOLS=reset_stop` makes
clearing a stop staff-only. Env knobs:
`CASCADE_CAMERAS` (comma list, first = manipulation camera), `CASCADE_CAMERA`
(single-camera fallback), `CASCADE_ARM`, `CASCADE_DETECTOR_MODEL`,
`CASCADE_DETECT_CLASSES`, `CASCADE_HIDE_TOOLS`, `CASCADE_VIEW`, `CASCADE_PREWARM`,
`CASCADE_STREAM`, `CASCADE_STREAM_PORT`, `CASCADE_RUN_DIR` (trace dir), `DISPLAY`. The Isaac bridge
side has its own knobs (`CASCADE_USD`, `CASCADE_PHYSICS_DEVICE` — `cpu` is the
escape hatch for GPU-PhysX boot NaNs —, `CASCADE_BRIDGE_BIND`,
`CASCADE_BRIDGE_NO_TARGETS`, `CASCADE_COMPANION_EXTS`); see `scripts/isaac_bridge.py`.

## The 30 skills

One schema source (`TOOL_SPECS` in `src/cascade/skills/runtime.py`) feeds
every consumer — the built-in `AgentOrchestrator`, the OpenAI/Anthropic
LLM backends, and the MCP server — so this list is exactly what any brain,
built-in or external, can call. "moves arm" marks the 15 skills in
`_MOTION_SKILLS`, the only ones that pause `WorldWatcher` belief fusion
while they run.

**Perception (no motion)**

| skill | what it does |
|---|---|
| `get_observation` | Fresh camera frame: visible objects with 3D positions, remembered objects, robot state |
| `list_objects` | Every object the robot knows about, including out-of-view ones with last-known position + age |
| `describe_scene` | Instant text description from the live world model — no motion, no camera wait |
| `analyze_scene` | Full perception report: per-camera detections, depth quality, scene description, numbered object key |
| `annotated_view` | Rendered camera view with numbered object badges, a 5 cm base-frame grid, and the reachable region shaded |
| `count_objects` | Count known objects, optionally filtered ("red", "cube", "pink object") |
| `localize_object` | Precisely localize one named object: base-frame position + size |
| `probe_point` | Cursor: click a pixel, get distance, 3D position, which object it is, reachability, offset from the gripper |
| `locate_pixel` | Inverse of `probe_point`: given a tracked object, where is it in the image right now |
| `preview_grasp` | Plan a grasp and report the proposed waypoint (position, approach, confidence) **without** moving |
| `open_live_view` / `close_live_view` / `live_view_status` | Open/close/check the browser dashboard (closed by default — chat is the interface) |

**Manipulation (moves arm)**

| skill | what it does |
|---|---|
| `grasp_object` | Full pipeline on a named object: localize → plan top-down grasp → approach → material-aware close → lift → verify |
| `grasp_at_pixel` | Grasp whatever is at a pixel, without needing to name it |
| `pick_and_place` | Fast path: complete pick-and-place in one call |
| `place_at` | Place the held object at base-frame coordinates |
| `place_on_object` | Place the held object on/in another named object |
| `push_object` | Push a named object along the table (too wide to grasp, or to reposition it) |
| `move_relative` | Nudge the gripper a few centimeters (forward/back/left/right/up/down) |
| `open_gripper` / `close_gripper` | Open (drops what's held) / close with the default grip profile |
| `move_home` | Return to the home configuration; also clears the camera view |
| `halt_motion` | Stop the in-flight motion because it's no longer the right action (wrong object, scene changed, subgoal already met) |

**Social / gesture (moves arm)**

| skill | what it does |
|---|---|
| `point_at` | Point at a named object (hover the gripper above it) |
| `wave` | Wave at the audience, around the home pose |
| `handover` | Grasp (if needed), present at the handover pose, hold until `open_gripper` |
| `sort_by_color` | Sort every known object into per-color zones along the table edge |
| `throw` | Grab (if needed) and throw via a harness-vetted wind-up-and-release swing |

**Memory**

| skill | what it does |
|---|---|
| `recall_memory` | Recent events (~15 s) and, optionally, where a named object was last seen |

`task_done` (declare success/failure with a summary) is the 31st spec but
is loop-internal — excluded from the MCP tool list, since an external host
ends its own turns its own way.

## Safety notes for the live rig

- The safety harness fails closed; motions abort mid-stream on violation.
- Gripper open/close angles in `configs/arms/rebot_rs.yaml` were
  characterized on the DM build — **re-verify travel and stall torque on the
  RS gripper before the first grasp**.
- Do not run `motorbridge-gateway` / MotorBridge Studio while the demo runs
  (host-id 0xFD conflict on the CAN bus).
- Hand-eye extrinsics in the camera profiles are placeholders — calibrate
  on-site (the baseline repo's `collect_handeye_eih.py` output loads
  directly via `hand_eye_npz`).
- Strict top-down tool poses are only IK-reachable below z ≈ 0.15 m on the
  B601-RS (wrist limits); grasp heights + hover offsets are configured
  accordingly.

## Setup notes

Setup on this rig: `scripts/setup_env.sh` (installs into the shared `.demo`
uv venv). pyrealsense2 comes from the local
[librealsense fork](https://github.com/johnnynunez/librealsense) build --
shared by every RealSense profile (D455F, D435i, ...), not just one model.
Open-vocabulary text prompts (YOLOE/YOLO-World) additionally need
`uv pip install git+https://github.com/ultralytics/CLIP.git`; the closed-set
`yolo11n.pt` works without it.

YOLOE also needs a MobileCLIP text encoder — for the shipped
`yoloe-11s-seg.pt` that is `mobileclip_blt.ts` (~572 MB, too big for
GitHub, so it is gitignored; this checkout carries it at
`models/mobileclip_blt.ts`). Ultralytics resolves it **relative to the
working directory** and auto-downloads it there on first use — launch from
`models/` (or copy the `.ts` next to your CWD), otherwise every detection
silently vanishes.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module-by-module design and
  the decisions behind it
- [docs/BOOTH_RUNBOOK.md](docs/BOOTH_RUNBOOK.md) — the 15-minute hands-on
  booth session: script, safety rules, fallback ladders, reset procedure
  (`scripts/booth_up.sh` / `scripts/booth_reset.sh`)
- [docs/ROADMAP.md](docs/ROADMAP.md) — VLA policy backend, GraspGen-X
  follow-ups, NuRec sim2real, skill-library growth
- [CLAUDE.md](CLAUDE.md) — working guide for AI coding agents (commands,
  invariants, gotchas)

## Naming note

The GitHub repo, Python package/import path, CLI entry points
(`cascade`, `cascade-record`, `cascade-mcp`, `cascade-view`), and env var
prefix (`CASCADE_*`) are all **cascade** (github.com/johnnynunez/cascade) —
renamed together from the original `wrc_demo`/`WRC_*` naming so the codebase
has one consistent identifier throughout. If you have an older checkout or
external tool config (Hermes/OpenClaw/Claude/Codex MCP registration,
`~/.wrc_demo/` persisted grasp/experience memory) still pointing at the old
name, re-run `scripts/setup_agents.py` for the former and copy
`~/.wrc_demo/*` to `~/.cascade/` for the latter.
