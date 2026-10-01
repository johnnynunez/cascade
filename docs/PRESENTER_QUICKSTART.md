# Build-a-Claw presenter card

Use the [current source and acceptance status](PROJECT_STATUS_20261001.md)
before rehearsal. The setup guide pins runtime baseline `28061a6`; its software
suite and [two-object native proof](SPARK_DELIVERY.md#current-two-object-proof-1-october-2026)
pass, while campaign and restart require their own receipts.
The historical successes below do not mark a new checkout READY.

**Use PhysX with the real GraspGen-X server on the presentation Spark.**
The historical September build `9cf5402` passed a fresh-destination
GB10 installation and its first desktop READY proof: green cube and orange,
independent physical checks, resets and three cameras. Real Chromium and the
camera extension also passed with a virtual display. GNOME app-grid interaction
and the visible Isaac editor remain untested in this run. The
[release receipt](../benchmark/results/spark_clean_delivery_20260930.json)
records the scope; individual successful runs do not guarantee every future pick.
The same stack subsequently passed all five kitchen objects through native
visitor chat, with independent physical placement checks and resets.
A full stop and second launch also reached READY in **9 min 10 s**; the
already-open browser recovered chat and all three cameras automatically.
Run these commands from the installed Cascade folder on the presentation
machine. The desktop entry starts Qwen as well as the robot stack; `run.sh`
expects Qwen to be running already. This card assumes the Spark installation profile; installation is
covered in [DGX Spark setup](DGX_SPARK_SETUP.md).

## 1. Start before the audience arrives

```sh
python3 scripts/desktop.py launch --repo "$PWD"
```

1. Isaac defaults to headless and still provides the three cameras. Optional
   `--gui` shows the editor but needs a separate rehearsal on the presentation
   machine. A running bridge is reused; stop that demo with `./run.sh down`
   before changing display mode.
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
| Isaac/robot stack crashed; Qwen still runs | Use the normal desktop launch for the complete UI; the lower-level restart below skips proof but has UI limits. |
| Qwen also stopped | Use the desktop start above, or start Qwen separately as described below before quick restart. |
| Reset fails or placement is unverified | Check the cameras and the reported error before issuing another movement. |

```sh
./run.sh isaac --engine physx --graspgenx local --no-robot-turn --no-judge
```

Quick restart skips the startup manipulation proof and optional image judge.
Service startup, scene identity, model and tool checks still run. Expect
**STARTED (UNVERIFIED: robot proof skipped)**, not a new READY certificate.
There is no fixed restart time: simulator initialization can still take minutes.
This is not a complete restart of the desktop interface: its camera/chat
attachment requires a verified proof, and the desktop command does not accept
these skip flags. Use the normal desktop launch to restore the complete UI.

If the model server also stopped, the desktop start restores the whole stack
and runs its full proof. To skip that proof, run
`bash scripts/serve_qwen_llamacpp.sh` in a separate terminal, wait for its server
to report ready, then use the quick command above. Leave that terminal open;
stop this manually started model with Ctrl-C before returning to the supervised
desktop start. Do not start a second model on an occupied port 8080.

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
| Isaac window | **Headless**, as validated | Optional desktop `--gui` shows the editor; rehearse that setting separately. |
| Occupancy/nvblox | **Disabled** in the Spark presenter profile | CUDA mapping and payload experiments have separate acceptance; see [nvblox status](NVBLOX.md). |
| Jev/Kev | **Evaluation code merged; not an active presenter backend** | [Ten-outcome replay](JEV_DECISIONS.md): Kev, Qwen and rules each 10/10; no advantage over rules demonstrated. Official TypeSafe Jev was not evaluated. |

**Why did GraspGen-X fall back to OBB?** The previous Spark launcher omitted
the learned-model server. An older failed startup probe or inference also selected
OBB for the rest of the runtime. The corrected installation and launch must
start and verify real inference; a GraspGen-X **stub** is only an analytic
protocol stand-in. The required Spark profile now reports inference failure;
it does not replace learned candidates with OBB. See the
[GraspGen-X evidence](GRASPGENX_SPARK.md#evidence-and-diagnostics).

**Should Newton work by default?** The Spark launcher defaults to PhysX.
Newton requires `--engine newton` and a validated Isaac/Cascade combination.
Its state-buffer compatibility handling and current acceptance results are
documented in [Newton validation](NEWTON_ENGINE.md).

Rehearse the normal startup, two placements, resets and recovery on the
presentation machine with the exact version and display setting used at the event.
