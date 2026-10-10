# GraspGen-X on DGX Spark

For current installation and acceptance use [Spark setup](DGX_SPARK_SETUP.md)
and [project status](PROJECT_STATUS_20261001.md). The presenter requires real
CUDA inference; optional-profile OBB fallback does not apply to that path.
[Planning search](grasp-planning-search.md) and [memory ranking](grasp-memory-ranking.md)
run after model responses without relaxing geometric vetoes.

The Spark profile uses the **real GraspGen-X CUDA model**. It starts a local
server, checks a diffusion inference, and requires learned candidates during
manipulation. An absent server, protocol stub or inference failure is reported
as an error. Analytic OBB is available only when explicitly selected.

## Install and start

For a new Spark, use the complete pinned installer in
[DGX Spark setup](DGX_SPARK_SETUP.md). It installs the model automatically.
To prepare or check only GraspGen-X in an existing current checkout:

```sh
bash scripts/install_graspgenx.sh
bash scripts/install_graspgenx.sh --check
python3 scripts/desktop.py launch --repo "$PWD" --gui
```

The normal Spark installer performs the first step automatically. It creates
`.graspgenx` and `.graspgenx-src` inside this checkout, independently of the
Cascade and Isaac Python environments. Source, checkpoints and gripper assets
are pinned to commits in the installer; the inference requirements are in
`scripts/requirements-graspgenx.txt`. This inference-only environment avoids
the upstream training dependency constraint `torch<2.7`, which cannot provide
the CUDA 13 ARM64 stack used here. The upstream model code is unchanged.

Spark defaults to `--graspgenx local`. The server binds loopback port 5556.
The launcher checks the model metadata and finite grasps tagged `diff` before
creating the robot runtime. The check is retained during quick restart:

```sh
./run.sh isaac --engine physx --no-robot-turn --no-judge
```

The quick command expects Qwen to be running (the desktop start manages it).
It skips the physical startup proof and prints **STARTED
(UNVERIFIED: robot proof skipped)**. Use it after rehearsing the normal READY
startup. It is a lower-level stack restart, not a replacement for the desktop's
verified camera/chat attachment. The desktop `--gui` option shows Isaac. See the
[presenter card](PRESENTER_QUICKSTART.md) for scene reset and the demo script.

## What was broken

1. The Spark launcher selected `none` and never started the configured model.
   The former serving script also depended on a particular sibling directory
   layout. Installation and launch now own a reproducible local sidecar.
2. CUDA perception produced a CUDA tensor; the client called `np.asarray` on
   it, failing before a model request was sent. The segmented cloud now crosses
   to host memory explicitly at the msgpack transport boundary. Width geometry
   remains on the input tensor's device; model inference runs on CUDA.
3. Sweep boxes were described relative to the TCP, but returned poses
   were shifted from a virtual gripper base by 98 mm. Only one direction of
   that frame conversion was applied. Both conditioning offsets and returned
   poses now use the same transform, including the fingertip depth. The reBot
   URDF also puts `gripper_end` at the distal fingertip: its 45 mm pad volume
   extends behind the TCP, so the volume center is -22.5 mm along approach.
   The old zero-centered box invented 22.5 mm of finger beyond the real tip
   and produced shallow, empty grasps on the orange. The geometry input is
   corrected; learned output poses are not replaced by analytic grasps.
4. A single timeout permanently selected OBB. Optional profiles now retry after
   a short cooldown. The required Spark profile fails visibly and retries on
   the next command without substituting analytic candidates.

The kitchen requests 400 diffusion samples, filters by learned score and a
near-vertical approach compatible with the countertop, then applies the same
IK and motion checks as before. It does not generate replacement OBB grasps
when those candidates fail. The upstream `graspmoe` planner remains an explicit
configuration choice for other profiles; its internally generated OBB branches
are distinct from pure diffusion and carry separate provenance.
Candidate validation also checks the vertical carry lift before closing: a
tilted pose that is reachable at pickup can be unreachable at carry height.

## Evidence and diagnostics

Hardware inference was exercised on an actual **NVIDIA GB10, AArch64,
compute capability 12.1**, driver 580.178.04, PyTorch 2.14.0+cu130 and CUDA 13.
The released generator and discriminator run in FP32. The first 100-sample
diffusion request took 2.36 s; a warm request took 0.89 s. These are individual
measurements, not restart-time or throughput guarantees.

```sh
.venv/bin/python scripts/check_graspgenx.py --output runs/graspgenx-check.json
```

The JSON records model metadata, GPU identity when served by Cascade's wrapper,
diffusion sample count and latency. Physical acceptance additionally records
the actual backend, branch counts and the unchanged contact, placement,
stability, camera and reset audits. Inference readiness alone is not a successful
pick-and-place. The first complete GB10 campaign passed **5/5** objects on
PhysX (orange, green cube, lemon, pink cube and tomato can), each on its first
physical attempt, with 400 diffusion candidates and all reset audits passing.
Source hashes stayed unchanged. The [receipt](../benchmark/results/graspgenx_spark_20260930.json)
contains per-object metrics and hardware provenance. This is one round, not
a statistical reliability guarantee. The full native Qwen/OpenClaw launcher
also reached **READY** on GB10 with PhysX and real GraspGen-X: green cube and
orange passed physical placement, contact, camera and reset checks. The orange
needed a second physical attempt in this chat run. See the
[launcher receipt](../benchmark/results/graspgenx_spark_launch_20260930.json)
and [actual three-camera recording, replayed at 4×](../benchmark/results/videos/graspgenx_spark_ready_20260930.mp4).
The run used independently installed GraspGen-X with existing Isaac/app packages
and a separately started Qwen server; it does not establish a fresh full desktop
installation. The [local Newton launcher proof](NEWTON_ENGINE.md) also reached READY.

Quick restart was exercised on that reused stack with Qwen and the bridges
still running. `--no-robot-turn --no-judge` skipped the physical proof and printed
the expected UNVERIFIED status; this was a warm reuse test, not a timed crash
recovery measurement. [nvblox remained disabled](NVBLOX.md) in these runs.

The pinned [fresh Spark installation](DGX_SPARK_SETUP.md#validated-release-and-scope)
now independently passed its first complete desktop launch with PhysX, real
GraspGen-X and Qwen, including both native placement/reset cases. The
[release receipt](../benchmark/results/spark_clean_delivery_20260930.json)
distinguishes new component/model destinations from reused download caches and
records the real Chromium/extension check under Xvfb. That two-object startup
proof is separate from the five-object campaign above.

For an already running server use `--graspgenx external`. Set
`CASCADE_GRASPGENX_PORT` for a different local port; it is propagated to the
runtime and OpenClaw MCP process. `CASCADE_GRASPGENX_HOST` names a server on
another machine for the runtime (a hostname or IPv4 address); the launcher's
`local` / `external` readiness checks dial that same host, print which
endpoint they check, and stop the launch on a malformed value (B70).
`--graspgenx none` explicitly selects analytic
OBB. Spark rejects `--graspgenx stub`; a stub cannot validate the model.

For a diagnostic native launch, set `CASCADE_GRASP_EVIDENCE_DIR` to an absolute
directory before starting the existing launcher. It passes this opt-in variable
to the registered MCP process. Each physical grasp attempt, including retries,
gets a JSON receipt and a hashed NPZ containing the localized point cloud and
raw GGX poses/scores. The receipt retains localization/capture provenance,
candidates before and after the existing memory prior, its z nudge, material
profile, jaw datum, selected IK targets, and the original attempt exception.
Isaac feedback and joint/gripper commands already used by the controller are
tagged by stage; the diagnostic adds no robot or camera reads or commands.

Events are buffered during execution and written after the attempt returns or
raises. Enabling capture adds copying and post-attempt filesystem overhead; it
does not certify unchanged timing or physical success. Keep and hash the prior
memory before the run rather than resetting it to seek a passing trial. Source
hashes and Git HEAD are explicitly observations of disk at receipt flush, not
an attestation of already imported bytecode; use a clean pinned checkout for
comparisons. `logging_ok: false`, `logging_errors`, and `dropped_events` expose
incomplete evidence. If the directory itself is unwritable, stderr carries the
failure receipt and terminal event; the original skill result/exception remains
unchanged. With the variable absent, no evidence directory is created.

The complete local regression after these changes passed **2,264 tests**
(46 skipped, 2 deselected; 263.23 s), including the portable deployment
packaging tests. The 130 focused model/launcher/carry-lift tests also passed.
