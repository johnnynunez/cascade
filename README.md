# wrc_demo — agentic grasping for the reBot DevArm

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
              ┌────────────────────────── agent (LLM) ──────────────────────────┐
task ────────▶│ decompose → tool call → verify → recover (advisor after failure)│
              └──────────────────────────┬──────────────────────────────────────┘
                            curated skill API (traced, ASPIRE-style)
   ┌────────────┬────────────┬───────────┴────────┬──────────────┬─────────────┐
   ▼            ▼            ▼                    ▼              ▼             ▼
get_observation localize   grasp_object        place_at/on    push_object  recall_memory
   │            │            │                    │              │             │
┌──┴────────────┴──┐   ┌─────┴──────┐      ┌──────┴───────┐ ┌────┴─────┐ ┌─────┴────────┐
│ perception       │   │ grasping   │      │ control      │ │ safety   │ │ memory       │
│ camera-agnostic  │   │ OBB grasp  │      │ FK/IK (pin)  │ │ harness  │ │ 15 s episodic│
│ Frame(+depth_m)  │   │ material → │      │ min-jerk     │ │ gates    │ │ + beliefs +  │
│ depth chain      │   │ force      │      │ streaming    │ │ every    │ │ TurboQuant   │
│ open-vocab YOLO  │   │ profiles   │      │ RobStride CAN│ │ waypoint │ │ embeddings   │
└──────────────────┘   └────────────┘      └──────────────┘ └──────────┘ └──────────────┘
```

## The four challenges, addressed

1. **Depth** — every camera backend yields `Frame` objects with float32
   *metric* depth aligned to color (kills the L515 0.25 mm vs D4xx 1 mm scale
   bug class). Cameras without depth fall through a strategy chain:
   sensor → optional mono-depth plugin → table-plane ray-casting (SAM/YOLO
   mask + known plane ⇒ 3D), so an RGB-only webcam still grasps tabletop
   objects. `src/wrc_demo/perception/depth_provider.py`.
2. **Harness** — a fail-closed `SafetyHarness` vets *every streamed waypoint*
   (joint limits, velocity caps, workspace AABB, table-plane clearance with a
   grasp-exemption cylinder, keep-out zones, perception watchdog, e-stop
   latch), and every skill call writes an ASPIRE-style multimodal trace
   (`trace.jsonl` + before/after keyframes). `src/wrc_demo/safety/harness.py`.
3. **Grasping / force** — 3D OBB grasps planned in the base frame (short-axis
   jaw opening, height-fraction grasp depth), plus material-aware grip
   profiles (rigid/fragile/soft/deformable/slippery/heavy) mapped to
   two-stage stall-aware closes; the LLM passes the material hint it infers.
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
# offline wiring check: mock camera + mock arm + scripted LLM, no hardware
PYTHONPATH=src python -m wrc_demo.apps.demo --task "look at the table"

# tests (48 unit/integration; live-hardware tests deselected by default)
python -m pytest tests/ -q
python -m pytest tests/ -m hardware -q     # needs L515 + can0 up (read-only)

# real rig, cloud LLM
sudo ip link set can0 up type can bitrate 1000000
python -m wrc_demo.apps.demo --task "put the red cube in the bowl" \
    --camera l515 --arm rebot_rs --llm anthropic

# real rig, local Qwen3.6 on the Spark (start the server first)
scripts/serve_qwen_llamacpp.sh          # or serve_qwen_vllm.sh (MTP spec decoding)
python -m wrc_demo.apps.demo --task "..." --camera l515 --arm rebot_rs --llm local_qwen
```

Setup on this rig: `scripts/setup_env.sh` (installs into the shared `.demo`
uv venv). pyrealsense2 comes from the local
[librealsense L515 fork](https://github.com/johnnynunez/librealsense) build.
Open-vocabulary text prompts (YOLOE/YOLO-World) additionally need
`uv pip install git+https://github.com/ultralytics/CLIP.git`; the closed-set
`yolo11n.pt` works without it.

## LLM backends

| profile | backend | notes |
|---|---|---|
| `anthropic` | Claude (cloud) | `ANTHROPIC_API_KEY`; vision + tools |
| `local_qwen` | Qwen3.6-27B via llama.cpp / vLLM on the Spark | OpenAI-compatible; MTP speculative decoding (~1.4–2.2× decode) |
| `openai` | any OpenAI-compatible cloud endpoint | `OPENAI_API_KEY` |
| `mock` | scripted | tests / wiring checks |

## Run it under any MCP agent platform

The whole skill runtime is also exposed as an **MCP stdio server**
(`wrc_demo/apps/mcp_server.py`) — so instead of the built-in loop, any
MCP-capable agent platform can drive the arm. The agent gets the same 12
safety-gated skills plus `camera_snapshot` (returns a live JPEG the agent
can *see*) and `emergency_stop`/`reset_stop`. Safety harness, tracing,
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
| **Claude Code** | project `.mcp.json` (ships in this repo) | open Claude Code here — zero setup; user-scope: `setup_agents.py --host claude` prints the `claude mcp add` one-liner |
| **Claude Desktop** | `claude_desktop_config.json` | paste the JSON block from `setup_agents.py --host claude` |
| **Codex CLI** | `~/.codex/config.toml` `[mcp_servers.wrc-demo]` | `setup_agents.py --host codex --write`, verify with `codex mcp list` |
| **OpenClaw** | native `mcp.servers` (2026+) or [mcporter](https://docs.openclaw.ai/cli/mcp) | `setup_agents.py --host openclaw` prints the `openclaw mcp set` one-liner + JSON block |

The server attaches hardware lazily: `initialize`/`tools/list` work with the
robot powered off, so agents can inspect the toolbox anytime. Env knobs:
`WRC_CAMERA`, `WRC_ARM`, `WRC_DETECTOR_MODEL`, `WRC_DETECT_CLASSES`,
`WRC_VIEW`, `DISPLAY`.

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
- [docs/ROADMAP.md](docs/ROADMAP.md) — VLA policy backend, GraspNet, NuRec
  sim2real, skill-library growth
