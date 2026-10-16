# GraspGen-X on DGX Spark

**Next action:** for a new Spark, run the pinned installer in [DGX Spark setup](DGX_SPARK_SETUP.md)
(it installs the model). To prepare or check only GraspGen-X in an existing checkout (~15 min
first install):

```sh
bash scripts/install_graspgenx.sh
bash scripts/install_graspgenx.sh --check
python3 scripts/desktop.py launch --repo "$PWD" --gui
```

The Spark profile uses the **real GraspGen-X CUDA model**: it starts a local server on loopback
5556 (`--graspgenx local`, the Spark default), checks the model metadata and finite grasps tagged
`diff` before creating the robot runtime, and requires learned candidates during manipulation.
An absent server, a protocol stub or an inference failure is an error; analytic OBB runs only
when explicitly selected. [Planning search and memory ranking](ARCHITECTURE.md#grasp-pipeline)
run after the model responds without relaxing geometric vetoes. Capability status:
[ROADMAP](ROADMAP.md).

## Install and options

1. The installer creates `.graspgenx` and `.graspgenx-src` in the checkout, separate from the
   Cascade and Isaac environments. Source, checkpoints and gripper assets are pinned in the
   installer; inference requirements are `scripts/requirements-graspgenx.txt`. This
   inference-only env avoids upstream's training pin `torch<2.7` (no CUDA 13 ARM64 build); the
   upstream model code is unchanged.
2. Quick restart keeps the inference check but skips the physical proof (expects Qwen running;
   prints **STARTED (UNVERIFIED: robot proof skipped)**; a lower-level restart, not the desktop's
   verified camera/chat attachment). Use it only after rehearsing the normal READY start:

```sh
./run.sh isaac --engine physx --no-robot-turn --no-judge
```

3. Already running server: `--graspgenx external`. `CASCADE_GRASPGENX_PORT` moves the local port
   (propagated to the runtime and the OpenClaw MCP process); `CASCADE_GRASPGENX_HOST` names a
   server on another machine (hostname or IPv4); `local` / `external` readiness checks dial that
   host, print the endpoint and stop on a malformed value (B70).
4. `--graspgenx none` explicitly selects analytic OBB. Spark rejects `--graspgenx stub` (a stub
   cannot validate the model). The desktop `--gui` option shows Isaac; scene reset and the demo
   script are on the [presenter card](BOOTH_RUNBOOK.md#9-simulated-kitchen-on-dgx-spark-presenter-card).
5. Kitchen sampling: 400 diffusion samples, filtered by learned score and a near-vertical
   approach, then the usual IK and motion checks, plus a vertical carry-lift check before closing
   (a tilted pose reachable at pickup can be unreachable at carry height). No replacement OBB
   grasps when candidates fail. Upstream `graspmoe` stays an explicit choice for other profiles;
   its internal OBB branches carry separate provenance.

## What was fixed for Spark

1. The launcher selected `none` and the serving script assumed a sibling layout: install and
   launch now own a reproducible local sidecar.
2. CUDA perception produced a CUDA tensor that the client passed to `np.asarray`: the segmented
   cloud now crosses to host memory at the msgpack boundary (width geometry stays on device).
3. Sweep boxes were TCP-relative while returned poses were shifted 98 mm from a virtual gripper
   base, converted in one direction only. Both now use the same transform. The reBot URDF puts
   `gripper_end` at the distal fingertip, so the 45 mm pad volume is centred -22.5 mm along the
   approach (the old zero-centred box invented 22.5 mm of finger and made shallow, empty grasps).
4. One timeout permanently selected OBB: optional profiles now retry after a short cooldown; the
   required Spark profile fails visibly and retries on the next command, never substituting
   analytic candidates.

## Evidence and diagnostics

```sh
.venv/bin/python scripts/check_graspgenx.py --output runs/graspgenx-check.json
```

The JSON records model metadata, GPU identity (when served by Cascade's wrapper), sample count
and latency. Inference readiness alone is not a successful pick-and-place.

1. **GB10 inference** (AArch64, compute capability 12.1, driver 580.178.04, PyTorch
   2.14.0+cu130, CUDA 13, FP32): first 100-sample request 2.36 s, warm 0.89 s (single
   measurements).
2. **Five-object campaign** on PhysX: **5/5** (orange, green cube, lemon, pink cube, tomato can),
   each on its first physical attempt with 400 candidates and passing reset audits; one round, not
   a reliability guarantee ([receipt](../benchmark/results/graspgenx_spark_20260930.json)).
3. **Native launcher READY** with PhysX and real GraspGen-X: green cube and orange passed
   placement, contact, camera and reset checks (orange on its second attempt)
   ([receipt](../benchmark/results/graspgenx_spark_launch_20260930.json),
   [three-camera recording at 4×](../benchmark/results/videos/graspgenx_spark_ready_20260930.mp4)).
   Reused Isaac/app packages and a separate Qwen: not a fresh install; the fresh install is in
   [Spark setup](DGX_SPARK_SETUP.md#validated-release-and-scope). [nvblox stayed off](NVBLOX.md);
   the [Newton launcher](NEWTON_ENGINE.md) also reached READY.
4. **Per-attempt evidence (opt-in):** set `CASCADE_GRASP_EVIDENCE_DIR` to an absolute directory
   before launching; it reaches the MCP process. Each physical attempt (retries included) gets a
   JSON receipt and a hashed NPZ (localized cloud, raw GGX poses/scores, candidates before and
   after the memory prior and its z nudge, material profile, jaw datum, IK targets, the original
   exception, stage-tagged Isaac feedback and commands). It adds no robot or camera reads or
   commands; events are written after the attempt, adding copy and filesystem overhead.
5. **Reading that evidence:** hash the prior memory before the run instead of resetting it to
   seek a pass; source hashes and Git HEAD are disk observations at flush (use a clean pinned
   checkout); `logging_ok: false`, `logging_errors` and `dropped_events` flag incomplete evidence;
   an unwritable directory reports on stderr and never changes the skill result. Without the
   variable no directory is created.
