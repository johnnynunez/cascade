# wrc_demo — agentic grasping for the reBot DevArm

[![CI](https://github.com/johnnynunez/wrc_demo/actions/workflows/ci.yml/badge.svg)](https://github.com/johnnynunez/wrc_demo/actions/workflows/ci.yml)

Camera-agnostic, LLM-orchestrated tabletop manipulation on the Seeed reBot
DevArm B601 (RobStride build), driven from an NVIDIA DGX Spark. This is the
agentic evolution of the
[reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)
baseline, designed after NVIDIA GEAR's
[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) (curated skill API +
multimodal traces + skill library) and
[Agentic-VLA](https://arxiv.org/abs/2605.22896) (task decomposition + VLM
advisor + experience memory).

```
              ┌───────────────────────────── agent ─────────────────────────────┐
task ────────▶│ reflex ▸ habit ▸ LLM — decompose → tool call → verify → recover │
              │ routine commands never wait on the model; VLM advisor on failure│
              └──────────────────────────┬──────────────────────────────────────┘
                 curated skill API — 20 traced skills (ASPIRE-style)
   ┌────────────┬────────────┬───────────┴────────┬──────────────┬─────────────┐
   ▼            ▼            ▼                    ▼              ▼             ▼
get_observation localize   grasp_object        place_at/on    push_object  recall_memory
   │            │            │                    │              │             │
┌──┴────────────┴──┐   ┌─────┴──────┐      ┌──────┴───────┐ ┌────┴─────┐ ┌─────┴────────┐
│ perception       │   │ grasping   │      │ control      │ │ safety   │ │ memory       │
│ camera-agnostic  │   │ GraspGen-X │      │ FK/IK (pin)  │ │ harness  │ │ 15 s episodic│
│ Frame(+depth_m)  │   │ ▸OBB fallbk│      │ min-jerk     │ │ gates    │ │ + beliefs +  │
│ depth chain      │   │ material → │      │ streaming    │ │ every    │ │ habit/grasp  │
│ open-vocab YOLO  │   │ force prof │      │ RobStride CAN│ │ waypoint │ │ memory       │
└──────────────────┘   └────────────┘      └──────────────┘ └──────────┘ └──────────────┘
```

## The four challenges, addressed

1. **Depth** — every camera backend yields `Frame` objects with float32
   *metric* depth aligned to color (kills the L515 0.25 mm vs D4xx 1 mm scale
   bug class). Cameras without depth fall through a strategy chain:
   sensor → optional mono-depth plugin → table-plane ray-casting (known
   plane ⇒ whole-image depth; the YOLO detection mask is applied later at
   grounding to lift object pixels to 3D), so an RGB-only webcam still
   grasps tabletop objects. `src/wrc_demo/perception/depth_provider.py`.
2. **Harness** — a fail-closed `SafetyHarness` vets *every streamed waypoint*
   (joint limits, velocity caps, workspace AABB, table-plane clearance with a
   grasp-exemption cylinder, keep-out zones, perception watchdog, e-stop
   latch), and every skill call writes an ASPIRE-style multimodal trace
   (`trace.jsonl` + before/after keyframes). `src/wrc_demo/safety/harness.py`.
3. **Grasping / force** — learned 6-DoF grasps from a GraspGen-X ZMQ server
   (`grasp.backend: graspgenx`, the default — `scripts/serve_graspgenx.sh`)
   with automatic fallback to analytic 3D OBB grasps planned in the base
   frame (short-axis jaw opening, height-fraction grasp depth) whenever the
   server is down. Candidates are re-ranked by a persisted per-object
   grasp-outcome memory, then vetted against IK *and* the safety-harness
   geometry before execution. Material-aware grip profiles
   (rigid/fragile/soft/deformable/slippery/heavy) map to two-stage
   stall-aware closes; the LLM passes the material hint it infers.
   `src/wrc_demo/grasping/`.
4. **Kinematics** — self-contained Pinocchio FK/IK on the RS URDF (ships in
   `assets/`), damped-least-squares with random restarts, min-jerk joint
   streaming with feedback-based settling. Live joint positions on the RS
   arm come from RobStride `mechPos` (0x7019) param reads — the only reliable
   feedback path on this hardware. `src/wrc_demo/control/`.

Plus **memory**: a 10–15 s episodic window (events + JPEG thumbnails +
optional TurboQuant-compressed embeddings, arXiv:2504.19874) and an
object-permanence belief store, so "the mug you saw 10 seconds ago" is still
actionable after occlusion. `src/wrc_demo/memory/`.

## Quick start

```bash
# offline wiring check: mock camera + mock arm + scripted LLM, no hardware.
# Routine commands run on the REFLEX fast path (no LLM); a livestream
# dashboard with N camera streams + robot narration prints its URL.
PYTHONPATH=src python -m wrc_demo.apps.demo --task "pick and place pink object"

# tests (unit/integration; live-hardware tests deselected by default)
python -m pytest tests/ -q
python -m pytest tests/ -m hardware -q     # needs L515 + can0 up (read-only)

# learned grasps (the default backend) need the GraspGen-X server running;
# without it grasping silently falls back to the analytic OBB planner
scripts/serve_graspgenx.sh

# real rig, N cameras (first = manipulation camera), cloud LLM fallback
sudo ip link set can0 up type can bitrate 1000000
python -m wrc_demo.apps.demo --interactive \
    --cameras l515,uvc4k --arm rebot_rs --llm anthropic

# Isaac Sim instead of hardware (same demo, simulated reBot):
#   1. inside Isaac Sim's python:  python.sh scripts/isaac_bridge.py --usd <rebot.usd>
#   2. then:
python -m wrc_demo.apps.demo --cameras isaac --arm isaac --interactive

# real rig, local Qwen3.6 on the Spark (start the server first)
scripts/serve_qwen_llamacpp.sh          # or serve_qwen_vllm.sh (MTP spec decoding)
python -m wrc_demo.apps.demo --task "..." --cameras l515 --arm rebot_rs --llm local_qwen
```

Setup on this rig: `scripts/setup_env.sh` (installs into the shared `.demo`
uv venv). pyrealsense2 comes from the local
[librealsense L515 fork](https://github.com/johnnynunez/librealsense) build.
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

## LLM backends

| profile | backend | notes |
|---|---|---|
| `anthropic` | Claude (cloud) | `ANTHROPIC_API_KEY`; vision + tools |
| `local_qwen` | local Qwen via llama.cpp / vLLM on the Spark | OpenAI-compatible; MTP speculative decoding (~1.4–2.2× decode) |
| `openai` | any OpenAI-compatible cloud endpoint | `OPENAI_API_KEY` |
| `mock` | scripted | tests / wiring checks |

> **Keep the profile in sync with the server:** `configs/llm/local_qwen.yaml`
> currently pins `model: Qwen3VL-30B-A3B-Instruct-Q4_K_M` while both serve
> scripts fetch **Qwen3.6-27B**. llama.cpp ignores the request's model name
> (and its script wires vision via the mmproj projector, so
> `supports_vision: true` still holds there); vLLM rejects a mismatched
> model name and has no vision wiring. Set the profile's `model:` (and
> `supports_vision:` if your server lacks a vision projector) to whatever
> the server actually loads.

## Run it under any MCP agent platform

The whole skill runtime is also exposed as an **MCP stdio server**
(`wrc_demo/apps/mcp_server.py`) — so instead of the built-in loop, any
MCP-capable agent platform can drive the arm. The agent gets the same 20
safety-gated skills (only the loop-internal `task_done` is excluded) plus
five gateway extras — `camera_snapshot` (returns a live JPEG the agent can
*see*), `world_state`, `live_view_url`, and `emergency_stop`/`reset_stop` —
25 tools total. Safety harness, tracing,
memory and the always-on camera window are identical — only the brain swaps.

One registrar for every host — prints what each platform needs, `--write`
applies the file edits (preserving unrelated entries):

```bash
python scripts/setup_agents.py                     # show all hosts
python scripts/setup_agents.py --host codex --write
python scripts/setup_agents.py --camera l515 --arm rebot_rs --write
```

| platform | mechanism | setup |
|---|---|---|
| **Hermes** | `~/.hermes/config.yaml` `mcp_servers` | `./scripts/hermes_demo.sh` (interactive: register + test + chat) |
| **Claude Code** | project `.mcp.json` (ships in this repo, pinned to the demo rig's venv paths and the Isaac profiles) | regenerate for your machine first: `setup_agents.py --host claude --python <venv python> --write`; user-scope: `--host claude` prints the `claude mcp add` one-liner |
| **Claude Desktop** | `claude_desktop_config.json` | paste the JSON block from `setup_agents.py --host claude` |
| **Codex CLI** | `~/.codex/config.toml` `[mcp_servers.wrc-demo]` | `setup_agents.py --host codex --write`, verify with `codex mcp list` |
| **OpenClaw** | native `mcp.servers` (2026+) or [mcporter](https://docs.openclaw.ai/cli/mcp) | `setup_agents.py --host openclaw` prints the `openclaw mcp set` one-liner + JSON block |

The server pre-warms perception at startup (cameras + detector + world
model) while the ARM stays unpowered until the first motion command
(LazyArm) -- so "pick and place pink object" from the chat is a single
`pick_and_place` tool call that starts moving immediately. Env knobs:
`WRC_CAMERAS` (comma list, first = manipulation camera), `WRC_CAMERA`
(single-camera fallback), `WRC_ARM`, `WRC_DETECTOR_MODEL`,
`WRC_DETECT_CLASSES`, `WRC_VIEW`, `WRC_PREWARM`, `WRC_STREAM`,
`WRC_STREAM_PORT`, `WRC_RUN_DIR` (trace dir), `DISPLAY`. The Isaac bridge
side has its own knobs (`WRC_USD`, `WRC_PHYSICS_DEVICE` — `cpu` is the
escape hatch for GPU-PhysX boot NaNs —, `WRC_BRIDGE_BIND`,
`WRC_BRIDGE_NO_TARGETS`, `WRC_COMPANION_EXTS`); see `scripts/isaac_bridge.py`.

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

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module-by-module design and
  the decisions behind it
- [docs/ROADMAP.md](docs/ROADMAP.md) — VLA policy backend, GraspGen-X
  follow-ups, NuRec sim2real, skill-library growth
- [CLAUDE.md](CLAUDE.md) — working guide for AI coding agents (commands,
  invariants, gotchas)
