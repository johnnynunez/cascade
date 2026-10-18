# NVIDIA technology inventory, 2026-10-18

Summary: CASCADE `main` at `4d32eeb5` uses **34 NVIDIA technologies**. Six are on by default in the
shipped Linux/Spark launch: **Isaac Sim 6.1** (Kit, RTX cameras through Replicator), **CUDA PhysX**,
**GraspGen-X**, **OpenShell**, **CUDA** and the **DGX Spark (GB10)**. Warp is the default occupancy
backend when nvblox is absent. 19 are opt-in, each with its own validation scope; five build or
supply the robot and scene assets; three are tooling. Two more were evaluated and not adopted, and
eight are references or plans with no dependency.

| tier | count | technologies |
|---|---|---|
| Core (shipped Linux/Spark launch) | 6 | Isaac Sim 6.1 + Replicator RTX cameras, CUDA PhysX, GraspGen-X, OpenShell, CUDA, DGX Spark |
| Default backend | 1 | Warp (occupancy TSDF/EDT when nvblox is not installed) |
| Opt-in (merged, own validation scope) | 19 | Newton, Isaac Lab 3.0, MuJoCo Warp, OVRTX + ovstage, nvblox, cuVSLAM, cuMotion, Cosmos3-Edge, NemoClaw, H2 policy, Isaac Sim 6.2 (internal), Isaac Lab-Arena, Brev, NGC Isaac Sim container, NVENC, Isaac Sim RTSP streaming (off), Jetson Thor, NVFP4, RTX PRO 6000 Blackwell |
| Assets | 5 | urdf-usd-converter, Isaac Sim asset transformer (SimReady rules), vMaterials (MDL), Isaac Sim public assets, IsaacGymEnvs Factory assets |
| Tooling | 3 | NVIDIA Container Toolkit, Kit Tracy profiler extension, NVML / `nvidia-smi` |
| Evaluated, not adopted | 2 | Nemotron 3.5 Super VL, Cosmos3-Edge as detector |
| Reference or plan only | 8 | ASPIRE (API re-implemented), Isaac ROS, cuRobo, GR00T, AGILE, TensorRT, newton-assets / mujoco-usd-converter (H2 Newton route), NuRec (roadmap) |

## Method

1. Source: `main` at `4d32eeb5f905ef2ce9918129939621cfbe23995b` (2026-10-18 11:36 +0200), clean tree.
2. Scan: `tools/scan_nvidia.py` counts the tracked text files that name each technology, split into
   code (`src/ scripts/ configs/ deploy/ extensions/ demo/`), tests (`tests/ benchmark/`) and docs
   (`docs/` without `docs/evidence/`, `README.md`, `AGENTS.md`). 1736 text files. A second, wider
   discovery pass read all 2623 tracked text files (including `assets/`, `docs/evidence/` and
   `uv.lock`) for every `NVIDIA <name>` phrase and about 150 NVIDIA product, library and extension
   names (`isaacsim.*`, `omni.*`, `nvidia-*` packages); it found the asset, tooling and Thor rows.
   The six open draft PRs were diffed against `main` for the same names.
3. Classification: every hit cluster was read (owning doc, entry points, installers, profiles). A
   name in a comment or a link is not counted as use.
4. Versions come from the pins in `pyproject.toml`, the installers and the owning docs. Status and
   evidence are quoted from the owning doc and [ROADMAP](../../ROADMAP.md), **not re-measured**: no GPU
   job ran for this report.

Tiers:
- `core`: started by `./run.sh` / `scripts/launch.sh` on Linux or the Spark installer with no extra
  flag.
- `default backend`: picked automatically when the preferred backend is missing.
- `opt-in`: merged and selected by a profile, flag, extra or env var; its results do not certify the
  core path, and vice versa.
- `evaluated`: scored in a recorded comparison and not selected.
- `reference`: cited for design or comparison; nothing imports it.

## Simulation and physics

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **Isaac Sim** (Kit, USD stage, timeline) | 6.1.0 (`isaacsim[all,extscache]==6.1.0.0`, Python 3.12) | the kitchen and tabletop simulator behind the ZMQ bridge (:8611) | `scripts/isaac_bridge.py`, `scripts/install_isaac.sh` | core | Spark fresh install 12 min 21 s; native visitor chat **5/5** objects with placement and reset checks (30 Sep 2026, [DGX Spark setup](../../DGX_SPARK_SETUP.md)) |
| **Isaac Sim 6.2** (internal build) | 6.2 | H2 humanoid owner and the MicroDuck native bridge | `scripts/isaac_h2_bridge.py`, `scripts/isaac_microduck_bridge.py`, `src/cascade/sim/h2_physx.py` | opt-in | H2 binding gate passed in simulation, 7 Oct 2026 ([H2](../../HUMANOID_H2.md)) |
| **PhysX** (GPU/CUDA) | bundled with Isaac Sim | default engine of the Spark launcher and Brev; H2 runs PhysX first (its policy was trained there) | `--engine physx` | core | as Isaac Sim |
| **Newton** | 1.6.1rc1 inside Isaac Sim 6.2; 1.6.0 standalone | `--engine newton` in the Isaac bridge (:8681); MicroDuck BAM actuators; Factory screw/thread-contact fixture | `isaacsim.physics.newton` in `scripts/isaac_bridge.py`, `src/cascade/control/newton_bam.py`, `src/cascade/sim/microduck_newton.py`, `scripts/validate_screw_contact.py` | opt-in | Newton launcher READY with GraspGen-X; green cube and orange passed the full audit (2.3 / 5.6 mm), x86_64 only ([Newton](../../NEWTON_ENGINE.md)) |
| **Isaac Lab** | 3.0 (separate venv) | agentic MicroDuck demos: one Isaac Lab env = one Newton world per robot; the admitted MicroDuck USD comes from Isaac Lab #8265 | `src/cascade/sim/microduck_lab_world.py`, `scripts/microduck_agentic/` | opt-in | MicroDuck status **candidate**; walking and turning not physically admitted ([MicroDuck](../../MICRODUCK.md)) |
| **Warp** | `warp-lang>=1.16` (occupancy), `>=1.7` (sim-warp), 1.17.0 in the Newton env | occupancy TSDF + exact EDT kernels on CPU or CUDA; Isaac camera readback buffers; base of Newton and MJWarp | `src/cascade/perception/occupancy_backends.py` (`WarpTsdfBackend`), `_warp_tsdf_kernels.py`, `scripts/isaac_camera_readback.py` | default backend | bridge order `auto: nvblox > warp > voxel` |
| **MuJoCo Warp** (MJWarp) | `mujoco-warp>=3.1`; 3.12.0 in the Newton env | `engine: warp` for MJCF arms; Newton's solver (contacts 32768, 132272 constraint rows) | `src/cascade/control/mujoco_arm.py`, `configs/arms/so101_mjwarp.yaml` | opt-in | ~650x slower than the C engine on CPU (README); no GPU number recorded |

## Rendering and sensors

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **RTX cameras + Omniverse Replicator** | bundled with Isaac Sim | RGB, metric depth (`distance_to_image_plane`) and instance masks from Isaac render products | `isaacsim.sensors.experimental.rtx`, `omni.replicator.core` in `scripts/isaac_bridge.py`, `src/cascade/sim/h2_physx.py`, `src/cascade/sim/microduck_newton.py` | core | three advancing 1280x720 camera views on the Spark (30 Sep 2026) |
| **OVRTX + ovstage** | `ovrtx==0.5.0.377615`, `ovstage==0.2.0.377349` (`.[ovrtx]`); 0.5.1 in the MicroDuck Isaac Lab venv | RTX RGB-D without Kit: static USD camera, `OvrtxRenderer.render(SceneSnapshot)`, live Isaac producer (`CASCADE_CAMERA_RENDERER=ovrtx`), MicroDuck 1080p replays | `src/cascade/sim/ovrtx_renderer.py`, `src/cascade/perception/ovrtx_camera.py` | opt-in | analytic RGB-D passed on RTX PRO 6000 and GB10; one `pick_and_place` with live OVRTX cameras + cuMotion + GPU PhysX (2 Oct 2026, [OVRTX](../../OVRTX_RENDERER.md)) |
| **NVENC** | host FFmpeg `h264_nvenc` | Brev visitor camera video, one encoder for all cameras | `deploy/brev/visitor_video.py`, `deploy/brev/streaming/camera_streams.py` | opt-in | Brev only ([Brev](../../BREV.md)) |
| **Isaac Sim RTSP streaming** | `isaacsim.streaming.rtsp` | camera streams from Isaac on Brev | `deploy/brev/streaming/` | opt-in, **off** | the RTSP/WebRTC profile is off and unaccepted |

## Perception, mapping and localization

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **nvblox** (`nvblox_torch`) | v0.0.10 source `c457c3fc`, private CUDA 13.2.2, PyTorch 2.14.1+cu132 | GPU TSDF + ESDF clearance map behind the occupancy bridge (:5557); the reBot's required always-on map when installed | `src/cascade/perception/occupancy_backends.py` (`NvbloxBackend`), `scripts/serve_occupancy_bridge.py`, `scripts/install_nvblox_source.sh`, `scripts/check_nvblox.py` | opt-in (off in the Spark presenter profile) | 15 upstream tests on both GPUs; a 5 cm cube's clearances within **10 mm** at 10 mm voxels; streamed check 0.18 ms/step on the Spark ([nvblox](../../NVBLOX.md)) |
| **cuVSLAM** | 17.0.0 (`pycuvslam`) | optional RGB-D SLAM in the `spatial` domain, rigid camera-to-base registration | `src/cascade/spatial/cuvslam.py`, `cuvslam_worker.py`, `cuvslam_uncertainty.py` | opt-in | 12-frame synthetic replay 0.01178 m translation error; 12 live simulated RGB-D captures tracked; pose uncertainty uncalibrated ([spatial](../../SPATIAL_PROVIDERS.md)) |

## Grasping and motion planning

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **GraspGen-X** (NVlabs) | source `b9429097`, checkpoints `7c834043`, grippers `19a03c00` | learned 6-DoF grasps (diffusion, 400 samples in the kitchen) from a local server (:5556); required in the Spark profile, stubs rejected | `scripts/install_graspgenx.sh`, `scripts/serve_graspgenx.sh`, `src/cascade/grasping/graspgenx_backend.py` | core | the Spark 5/5 campaign above ([GraspGen-X on Spark](../../GRASPGENX_SPARK.md)) |
| **cuMotion** | 1.1.0 (`ac115fdb`), separate env, not redistributed | `TrajectoryOptimizer` for every arm motion in the cuMotion profiles, through `SafeArm` | `src/cascade/planning/cumotion.py`, `cascade.apps.plan_motion`, `configs/arms/isaac_cumotion.yaml`, `isaac_kitchen_cumotion.yaml` | opt-in | one complete Isaac tabletop `pick_and_place`; planning smokes on x86_64 and Spark; no success rate ([cuMotion](../../CUMOTION.md)) |

## Models and agent runtime

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **OpenShell** | 0.1.3 (sha256-checked) | sandbox for the OpenClaw agent on every Linux launch; L7 proxy swaps placeholder secrets, blocks `reset_stop` and motion while observe-only | `scripts/openshell_rig.py`, `scripts/launch_openshell.sh`, `deploy/openshell/` | core | sandboxed agent: physics-confirmed pick and verified reset on 2 of 3 fresh Isaac stages (B35, NemoClaw route) ([OpenShell](../../OPENSHELL.md)) |
| **NemoClaw** | v0.0.124 on OpenShell 0.0.116 | the older agent route: TLS MCP from a private CA, `inference.local` | `scripts/nemoclaw_mcp.py` | opt-in | B35 above |
| **Cosmos3-Edge Reasoner** (2.44B MoT) | `cosmos3-edge` served by vLLM (:8082) or SGLang (:8083, unverified) | alternative local brain (`--brain cosmos`), XML tool calls | `configs/llm/local_cosmos.yaml` (Jetson Thor), `local_cosmos_sglang.yaml`, `src/cascade/agent/cosmos3.py`, `scripts/cosmos_serving.py` | opt-in | vision and explanations on the Franka FR3 / Jetson Thor rig ([Franka](../../FRANKA_OPENCLAW.md)); replaces Qwen on the kitchen only after the same acceptance gates ([ROADMAP](../../ROADMAP.md)) |
| **H2 policy + Isaac assets** | `Velocity-H2-History-v0` (TorchScript, 255 -> 14), `Isaac/Robots/Unitree/H2/H2.usda`, digests in `configs/h2/bundle.json` | the H2 whole-body walking policy and its training stage | `src/cascade/control/h2_policy_contract.py`, `src/cascade/sim/h2_physx.py` | opt-in | candidate, gate 1 passed in simulation |
| **NVFP4** | `Inferact/Qwen3.8-27B-NVFP4` | the vLLM brain option (FP4 GEMM from CUTLASS via FlashInfer, needs `nvcc`) | `scripts/serve_qwen_vllm.sh` | opt-in | the booth's object naming was measured on it ([Booth runbook](../../BOOTH_RUNBOOK.md)); the Spark install serves the llama.cpp Q4 GGUF instead |

## Platforms, deployment and CUDA

| technology | version / pin | what CASCADE uses it for | entry points | tier | best recorded evidence |
|---|---|---|---|---|---|
| **DGX Spark** (GB10, aarch64) | install pinned to `ff8d58be` | event machine: the PAAI kitchen demo | `scripts/install.sh --profile spark`, `scripts/bootstrap.sh`, `scripts/spark_verify.py` | core | READY in 12 min 12 s; second launch 9 min 10 s |
| **CUDA** | `uv.lock`: torch 2.13.0 (via the `perception` extra) with CUDA 13.0 runtime, cuBLAS 13.1, cuDNN 9.20, NCCL 2.29, cuSPARSE(Lt), cuFFT, cuRAND, cuSOLVER, cuFile, NVRTC, nvJitLink, CUPTI, NVSHMEM, NVTX, CUDA Python bindings 13.3.1; private CUDA 13.2.2 for nvblox | every GPU component above; `device: auto` probes CUDA first | `src/cascade/device.py` | core | n/a |
| **RTX PRO 6000 Blackwell** | dual-GPU local PC; Brev `g7e.2xlarge` (Server Edition, 97,887 MiB) | development and x86_64 validation while the Sparks are disconnected | n/a | opt-in | one Qwen + Isaac run used 36,590 MiB with video off |
| **Brev** | `brev create ... --type g7e.2xlarge` | x86_64 cloud deployment of the kitchen | `deploy/brev/deploy.sh` | opt-in | own acceptance boundary; no Spark default applies |
| **NGC Isaac Sim container** | `nvcr.io/nvidia/isaac-sim:6.1.0@sha256:0c16dd67...` | base image of the Brev deployment | `deploy/brev/container/Dockerfile` | opt-in | Brev only |
| **Jetson Thor** | Tegra 6.8.12 kernel | host of the Franka FR3 rig: CASCADE, Cosmos3 Edge and the occupancy bridge pinned to CPUs 0-7, ROS bringup beside them; `device: auto` also covers Orin/Thor builds | `scripts/franka_openclaw.py`, `configs/arms/fr3.yaml`, `configs/llm/local_cosmos.yaml` | opt-in | rig bring-up documented in [Franka](../../FRANKA_OPENCLAW.md); no success rate |
| **Isaac Lab-Arena** | inspected source `c8d04e21` | strict importer for Arena results and a policy adapter | `src/cascade/eval/arena.py`, `benchmark/arena/` | opt-in | CPU tests; no benchmark task completed ([adapters](../../../benchmark/arena/README.md)) |

## Assets

| technology | version / pin | what CASCADE uses it for | where |
|---|---|---|---|
| **urdf-usd-converter** (newton-physics) | 0.3.0 `@554f3dc` | builds the reBot B601 USD from Seeed's URDF | `assets/usd/RS-rebot-dev-arm/docs/VALIDATION.md` |
| **Isaac Sim asset transformer** + SimReady rules | develop rules >= 1.7.12 | restructures the reBot USD (payload on the root prim, mimic-joint gains per SimReady) | `assets/usd/RS-rebot-dev-arm/scripts/run_transformer_361.py`, `prep_asset.py` |
| **vMaterials** (MDL) | n/a | the reBot's `ABS_Hard_Leather.mdl` material | `assets/usd/RS-rebot-dev-arm/Textures/` |
| **Isaac Sim public assets** (S3) | Isaac 6.0 / 6.1 asset roots | H2 USD and policy files, Isaac Lab materials and skies, the MicroDuck USD folder from Isaac Lab | `assets/h2/bundle/`, `configs/h2/bundle.json`, `assets/microduck/` |
| **IsaacGymEnvs Factory assets** | pinned commit (`manifest.json`) | nut, bolt, peg, hole and gear plate for the fastening runtime | `assets/factory/nut_bolt/` |

## Tooling

| technology | what CASCADE uses it for | where |
|---|---|---|
| **NVIDIA Container Toolkit** | GPU access for the Brev bridge containers (`nvidia-container-runtime` is checked at setup) | `deploy/brev/host_setup.py` |
| **Kit Tracy profiler** (`omni.kit.profiler.tracy` 1.2.1) | the Isaac bridge profiling captures | `docs/evidence/isaac-profile08/`, `isaac-profile09/` |
| **NVML / `nvidia-smi`** | GPU identity, memory and process receipts in installs and benchmarks (`nvidia-ml-py` is locked) | `benchmark/results/`, `scripts/`, `deploy/` |

## Scan counts (files naming the technology)

| technology | code | tests | docs |
|---|---|---|---|
| CUDA | 98 | 137 | 22 |
| Newton | 77 | 103 | 18 |
| Isaac Sim / Kit | 71 | 71 | 17 |
| PhysX | 52 | 109 | 19 |
| GraspGen-X | 40 | 77 | 21 |
| Cosmos | 31 | 28 | 8 |
| OVRTX / ovstage | 28 | 13 | 9 |
| nvblox | 28 | 68 | 18 |
| MuJoCo Warp | 25 | 26 | 6 |
| Brev | 25 | 15 | 9 |
| Warp | 24 | 17 | 5 |
| ASPIRE | 20 | 31 | 4 |
| DGX Spark / GB10 | 20 | 19 | 16 |
| Jetson | 17 | 1 | 4 |
| Isaac Lab | 16 | 9 | 6 |
| cuMotion | 15 | 20 | 13 |
| OpenShell | 14 | 12 | 10 |
| NemoClaw | 13 | 9 | 7 |
| NVENC | 7 | 0 | 1 |
| Omniverse Replicator | 6 | 0 | 0 |
| cuVSLAM | 5 | 4 | 7 |
| NGC (nvcr.io) | 2 | 0 | 0 |
| cuRobo | 2 | 0 | 0 |
| Isaac Lab-Arena | 1 | 3 | 1 |
| Isaac ROS | 1 | 0 | 3 |
| Nemotron | 1 | 0 | 0 |
| NVFP4 | 1 | 1 | 2 |
| TensorRT | 0 | 3 | 0 |
| GR00T | 0 | 0 | 1 |

The code hits for cuRobo, Isaac ROS and Nemotron are comments; the TensorRT hits are receipts that
record `"tensorrt": false`.

## Evaluated, not adopted

From the [perception bake-off](../perception-bakeoff-20261015/REPORT.md) (47 frames, 157 objects):

| model | inventory id_lenient | refer hit | why not |
|---|---|---|---|
| Nemotron 3.5 Super VL NVFP4 (thinking off) | 0.78 | 0.85 | below Qwen3.8-27B (0.83 / 0.94) |
| Cosmos3-Edge (thinking off) | 0.33 | 0.82 | precise (0.93) but identifies a third of the objects |

## Design references only (no dependency)

1. **ASPIRE** (NVIDIA GEAR): CASCADE re-implements its API surface (`get_observation`,
   `localize_object`, `generate_grasps`, `select_grasp`, `move_to_pose`), per-call trace and
   skill-note loop in `src/cascade/agent/aspire.py`; no ASPIRE package is imported
   ([architecture](../../ARCHITECTURE.md#design-position)).
2. **Isaac ROS**: the nvblox multi-camera tutorial and URDF body masking are cited; no ROS package.
3. **cuRobo**: robot-sphere carving is cited in `src/cascade/perception/robot_mask.py`.
4. **GR00T** (Isaac ROS G1 integration, data config): research table in
   [robot modularity](../../ROBOT_MODULARITY.md#research-directions-and-external-benchmarks).
5. **AGILE**: the H2 policy's training framework; its code is not public, so the policy is frozen.
6. **TensorRT**: not used; GraspGen-X runs as PyTorch (`tensorrt: false` in the launch receipts).
7. **newton-assets / mujoco-usd-converter**: the closed-loop H2 USD (newton-assets PR #53) is the
   documented Newton route for H2; nothing loads it yet ([H2](../../HUMANOID_H2.md)).
8. **NuRec**: on the [roadmap](../../ROADMAP.md) to reconstruct the demo tabletop as a photoreal twin;
   not started.

## Pending in open draft PRs (not on `main`)

| PR | NVIDIA technology | change to this inventory if merged |
|---|---|---|
| [#375](https://github.com/johnnynunez/cascade/pull/375) | **Nemotron 3.5 Bolt** 74B-A9B (vLLM) | becomes the default brain for the agent and perception: Nemotron moves from `evaluated` to `core`, Qwen stays `--brain qwen` |
| [#376](https://github.com/johnnynunez/cascade/pull/376) | ASPIRE | demo port of the ASPIRE note and retry loop (A1-A8); still a re-implementation, no package dependency |

## Not NVIDIA (often assumed)

Qwen3.8-27B (the event brain), SAM 2.1-L, OpenClaw, HUG (`grasp.backend: hug`), vLLM, SGLang,
llama.cpp, FlashInfer (it ships CUTLASS kernels), OpenAI Triton (locked with torch), MuJoCo C engine,
Pinocchio, LIBERO, VAB, Tracy itself, the Franka ROS 2 controllers.

## Gaps seen

- **Three version lines run side by side:** Isaac Sim 6.1 on the event path and the internal 6.2 for
  H2 and MicroDuck; Newton 1.6.1rc1 inside 6.2 and 1.6.0 standalone; OVRTX 0.5.0 in the extra and
  0.5.1 in the MicroDuck venv. A result on one does not certify another.
- **nvblox** is a local source build: upstream caps Torch at 2.9.1, so the wrapper is installed with
  `--no-deps`, and the release has no ARM64 wheel. It is off in the Spark presenter profile.
- **cuMotion** is under the Isaac ROS Software License and must be installed by the user.
- **Cosmos3-Edge** waits on native OpenClaw tool calling; the SGLang route is unverified.
- **Agentic MicroDuck demos on `main`** take the pose from the simulator; cuVSLAM is not wired there.
- **No profiled numbers per technology:** latency, throughput and VRAM per component are not in this
  report (only the few figures the owning docs record).

## Limits

- One commit of `main` plus the six open draft PRs. Other branches and worktrees (for example the
  onboard-only MicroDuck runs) are not included.
- The scan finds names. Classification comes from reading the docs and entry points, so a
  technology used only through a dependency of a dependency can be missed.
- Status and evidence are as documented by the owning page; nothing was re-run for this report.

## Re-run

```bash
python3 docs/evidence/nvidia-stack-20261018/tools/scan_nvidia.py
```

Stdlib only, read-only, a few seconds. It prints the scan-count table for the checked-out commit;
the tiers and the per-area tables are a manual reading of the hits.
