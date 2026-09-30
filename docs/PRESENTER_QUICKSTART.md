# Build-a-Claw presenter card

**Use PhysX with the real GraspGen-X server on the presentation Spark.**
GraspGen-X deployment and physical acceptance are being validated in PR #23.
Run these commands from the installed Cascade folder on the presentation
machine. This card assumes the Spark installation profile; installation is
covered in [DGX Spark setup](DGX_SPARK_SETUP.md).

## 1. Start before the audience arrives

```sh
./run.sh isaac --engine physx --graspgenx local --no-judge
```

1. Leave out `--headless` to show the Isaac Sim window. A running headless
   bridge is reused; stop that demo with `./run.sh down` before changing mode.
2. Wait for **READY**. The startup proof moves the green cube and orange,
   checks their placement, and resets the scene.
3. Check the camera views and the runtime's `grasp_planner=graspgenx (learned 6-DoF)` banner.
4. Open the existing demo chat. Use the same conversation for commands and reset.

Use the version rehearsed on that machine. The local workstation acceptance
in [Newton validation](NEWTON_ENGINE.md) is not a substitute for rehearsal
on either presentation Spark.

## 2. Run the short demo

Send one request at a time; wait for its result:

1. **"What can you see on the table?"**
2. **"Could you put the green cube in the green square?"**
3. **"Let's start over."**
4. **"Please put the orange in the open box."**
5. **"Let's start over."**

For a failure demonstration, move an object in Isaac only while the arm is
idle and keep physics playing. Ask for a new scene description before the
next manipulation. Rehearse that exact intervention; arbitrary object moves
are not covered by the two startup placement tests. If the agent reports
failure or an unverified placement, show that result as reported.

## 3. Recover during the presentation

| Situation | Action |
| --- | --- |
| Stack works; objects need restoring | Click **Start over** in the Spark chat, **Reset** in PAAI OpenClaw, or send **"Reset the scene."** |
| An order is still running | Wait for its result before resetting; the reset buttons are disabled while busy. |
| Stack crashed | Run the quick restart below, preserving any custom scene flags. |
| Reset fails or placement is unverified | Check the cameras and the reported error before issuing another movement. |

```sh
./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge
```

Quick restart skips the startup manipulation proof and optional image judge.
Service startup, scene identity, model and tool checks still run. Expect
**STARTED (UNVERIFIED: robot proof skipped)**, not a new READY certificate.
There is no fixed restart time: simulator initialization can still take minutes.

The in-session reset restores props, returns the arm home, clears object
beliefs and task images, and obtains a fresh observation. Learned grasp history,
recent text memory and chat history remain. Deleting a memory file is a
different operation: a live server can retain and rewrite its in-memory copy.
Use scene reset between visitors. See [memory and recovery details](SPARK_DELIVERY.md#quick-recovery-during-the-demo).

## 4. Which components should be active?

| Component | Presenter choice | What another selection means |
| --- | --- | --- |
| Physics | **PhysX**, the Spark default | Newton is an explicit opt-in under separate validation; do not switch engines during the event. |
| Grasps | **GraspGen-X**, learned inference on the Spark | A protocol stub or an analytic fallback does not validate this setup. |
| Isaac window | **Visible**; omit `--headless` | Headless runs still produce camera images but do not show the editor. |

**Why did GraspGen-X fall back to OBB?** The previous Spark launcher omitted
the learned-model server. A failed startup probe or inference also selected
OBB for the rest of the runtime. The corrected installation and launch must
start and verify real inference; a GraspGen-X **stub** is only an analytic
protocol stand-in. Hardware validation results will be recorded with the PR.

**Should Newton work by default?** The Spark launcher defaults to PhysX.
Newton requires `--engine newton` and a validated Isaac/Cascade combination.
Its state-buffer compatibility handling and current acceptance results are
documented in [Newton validation](NEWTON_ENGINE.md).

**Next: run the normal startup command on the presentation machine and check READY.**
