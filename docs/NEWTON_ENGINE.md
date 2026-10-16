# Newton physics engine: setup and validation

**Next action:** start a clean Newton bridge, then run the kitchen acceptance campaign from this
checkout (25 cases; roughly an hour, estimated from the 289 s two-case launcher recording below):

```sh
uv run python benchmark/diagnostics/kitchen_acceptance.py \
  --port 8681 --engine newton --rounds 5 --output runs/acceptance-newton5
```

The exit status is nonzero for any failed case. Results for one engine, asset or architecture do
not certify another; see [profile defaults and current acceptance](ROADMAP.md).

## Select and configure the engine

1. `scripts/isaac_bridge.py` defaults to `--engine newton`; the Spark launcher defaults to PhysX,
   so select Newton explicitly with `./run.sh isaac --engine newton` (or `launch.sh --engine
   newton`).
2. Substeps: `--newton-substeps` (default 5). The bridge refuses a count whose parity hits
   [Isaac's Newton stepping bug](#4b-isaacs-newton-stepping-bug-use-an-odd-substep-count) (odd with
   the CUDA graph on, even with it off).
3. Contact buffers: MJWarp contacts 32768 (recorded peak 30,955) and 132272 constraint rows, set
   before initialization and re-checked on the live stage; the bridge fails if they are smaller.
   `--newton-contacts mujoco` is a diagnostic in which MJWarp detects its own contacts.
4. Hydroelastic contact is a separate opt-in (volumetric SDF on both colliders,
   `newton:hydroelasticEnabled` and the runtime `HydroelasticConfig` switch), not enabled by
   changing `contact_ke` or `impratio`, and not validated by the receipts here (see the
   [Isaac Sim 6.1 contact reference](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact.html)
   and [nut-and-bolt walkthrough](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/hydroelastic_contact_walkthrough.html)).
5. Resets and verification read and write engine state; editing USD does not update a running
   Newton scene ([data flow](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/new_physics_engine.html)).
   Keep simulation time separate from wall time: rendering can make a fixed wall-clock delay too
   short for jaw motion ([simulation fundamentals](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/physics/simulation_fundamentals.html)).

## Real reBot asset on the internal 6.2 build: start-up, probe battery, wrist camera (7 October 2026)

Rig: x86 test rig, GPU 0, Isaac Sim 6.2.0-alpha.19 (`omni_isaac_sim` develop `48b2d951`), Newton
1.6.1rc1, Warp 1.17.0; bare reBot scene (`assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda`, two
0.05 × 0.05 × 0.08 m boxes `pink_cube`/`green_cube`), bridge with `CASCADE_ISAAC_PIXEL_MASK=1`.
[Evidence](evidence/newton-probe-62-20261007/REPORT.md).

1. **Start-up deadlock (fixed).** `--engine newton --headless` never served on 6.2: the full Newton
   experience (`isaacsim.exp.full.newton.kit`) waits for the viewport's first frame, which
   `SimulationApp(disable_viewport_updates=True)` prevents. Explicit-experience launches now keep
   the viewport on until `is_app_ready()` (`_disable_viewport_updates_once_ready`); PhysX is
   unchanged.
2. **Probe battery** (`scripts/physics_probe.py`): props come from the bridge's `_PROP_SPAWNS`,
   teleports use the engine-aware `place_prop` op, each grasp records `tcp_err_m`,
   `shoved_on_descent_m` and `lift_target_dz_m`. Settle / drop / grasp / push for both boxes:
   **Newton 8/8, PhysX 8/8** (Newton 8/8 on the 6.1 baseline too).
3. **"Green box grasp fails on Newton" was the probe**: at r = 0.34 m a lift 0.12 m above the grasp
   is outside the IK envelope on both engines. The probe now lifts to the highest reachable of
   +0.12/0.10/0.08/0.06 m and reports an unreachable lift as `skipped`, never a physics failure.
4. **Wrist camera, validated mid-descent** against physics-truth corners: PhysX centroid error
   0.83 px static / 0.69 px moving (IoU 0.995 / 0.997), Newton 2.63 / 2.73 px (IoU 0.982 / 0.984),
   depth exact on both; per-frame extrinsics are time-aligned. Not covered: the real D435i
   hand-eye calibration.
5. **Open: Newton camera cadence on 6.2.** Each `app.update()` advances three physics steps on
   the Newton experience (one on PhysX), so a render can carry an intermediate step with no matching
   state; it is refused (`render frame has no unique matching state history`, fail-closed). While
   the arm streams, 34–40 % of camera refreshes publish no frame on 6.2 (2.8 % on 6.1). Candidate
   fixes, unmeasured: record state per physics step, or one physics step per update.

## Kitchen acceptance (2026-09-30)

A completed motion routine is not placement acceptance. `benchmark/diagnostics/kitchen_acceptance.py`
runs all five props through the real skills, then applies the passive GPU audit: bilateral grasp
forces, full destination entry while held, final footprint, support, release, stability, upright
cubes/can, current cameras and all five reset poses. Each campaign records source hashes and rejects
source changes during the run. The harness requires CUDA perception, matching the Spark launcher.

Corrections the audit forced (no acceptance bound was relaxed):

1. **Placement height** from measured joints and the calibrated pickup support plane (a partial
   point cloud can miss the bottom), plus the destination support height and 15 mm clearance (6 mm
   pressed a lemon into the floor; the box floor is 4 mm above the counter). Explicit
   `place_at(z=...)` keeps its height; without valid held geometry the configured release height is
   the fallback; profiles without a calibrated plane keep the observed point-cloud bound.
2. **Transport**: own 140 mm clearance and vertical pre-lift. Before retreat the runtime waits for
   ≥ 98 % measured jaw opening (8 s limit); closing waits for measured travel or a stable jaw (8 s
   per stage). Missing or still-moving feedback stops motion and returns failure.
3. **PhysX** authors 64 position iterations for kitchen props before play (16 and 32 left a resting
   pose at 0.260 rad/s against the 0.2 rad/s bound).
4. **Rounded grasps** slice and take width percentiles on CUDA; only compact section geometry
   crosses to the host planner. Slip authority follows the
   [coherent observation contract](ARCHITECTURE.md#grasp-pipeline); cached compensation alone
   cannot authorize release.
5. These historical trials used truth-assisted held-object XY/slip compensation and were not
   perception-only; the passive witness never fed the controller.

The completed 15 mm / 64-iteration baseline has **75/75 passing cases**:

| Engine at 120 Hz | Substeps | Cases | Resets |
| --- | --- | --- | --- |
| Newton / CUDA graph | 3 | 25/25 | 25/25 |
| Newton / CUDA graph | 5 | 25/25 | 25/25 |
| PhysX / CUDA | n/a | 25/25 | 25/25 |

Five rounds of green cube, orange, pink cube, lemon and tomato can per engine, using the **analytic
OBB fallback** (validates the physics corrections, not learned GraspGen-X)
([final baseline](../benchmark/results/newton_kitchen_20260930_final.json);
[cold-start failures](../benchmark/results/newton_kitchen_20260930_cold_start.json) on the 2 s
camera-age bound, archived). A full Newton launcher run then reached **READY** with Qwen issuing
native OpenClaw tools and the diffusion-only GraspGen-X server; green cube and orange passed the
full audit (center errors 2.3 mm and 5.6 mm) on the local x86_64 workstation, separate from GB10
validation ([receipt](../benchmark/results/newton_graspgenx_launch_20260930.json),
[three-camera replay](../benchmark/results/videos/newton_graspgenx_ready_20260930.mp4)). The Spark
learned-planner follow-up is in [GRASPGENX_SPARK](GRASPGENX_SPARK.md).

## Kitchen contact tuning (2026-09-29)

The Spark kitchen (`demo/scene/kitchen_config.json`: green cube to the green square, orange into
the open box) uses the same skills, launcher and acceptance criteria on both engines. Contact and
servo settings below are Newton-specific; `ok=True` results from this stage are diagnostics,
superseded by the acceptance above.

| problem measured under Newton | cause | fix |
|---|---|---|
| contact buffer overflow, 7396 shapes | `convexDecomposition` on the gripper ran Newton's CoACD: 6452 hulls, 4-5 mm p99 error on the fingers | the 28 convex hulls from Seeed's vendor reBot MJCF (`assets/newton/rebot_gripper_hulls.usda`, `scripts/build_newton_gripper_hulls.py`); peak ~490 contacts |
| held cube creeps, fruit slips | pyramidal cone, point contacts, pad friction 1.0 | Menagerie options (elliptic cone), pad solref/solimp/priority, pad friction 1.3 (the PhysX material) |
| orange dropped short of the box in 7 of 9 rounds | at the vendor model's impratio 10 it slides 5-15 mm through the pads during the carry | `impratio` 100: slip 0.3-2.7 mm, 6/6 in the box (3 and 5 substeps) |
| green cube lands on its side in 5 of 7 rounds | Isaac's default contact (ke 1e4, kd 100) is MuJoCo solref (0.02, 0.5): under-damped, the 7 cm release bounces it over | Newton's own default (ke 2500, kd 100 = solref 0.02/1.0) for every collider without a material: 8/8 upright, 0.3-0.6 cm from the centre |
| scene ran at 0.75x, prop resets silently undone | Isaac's `NewtonStage.simulate` state-buffer bug with an even substep count (below) | odd substep count (5) until Isaac ships the fix |
| NaN after every case, while parking | the asset's PhysX drive gains (joint2: 85,944 N·m/rad) saturate MuJoCo's 36 / 14 N·m effort limits: a relay, 0.1 rad limit cycle, 0.37 rad past the joint limit | the vendor model's gains for the same motors (900/60, 120/10): holds within 0.024 rad (PhysX 0.07) |
| fingers slam shut at 3.4 m/s | Newton ignores the joint velocity limit | finger target rate-limited to the asset's 0.243 m/s (0.14 m/s measured; PhysX ~0.18) |
| a reset launched the cube at 17 m/s, 145 rad/s, then NaN | `_newton_teleport` wrote the identity as w,x,y,z; Newton stores x,y,z,w (a 180 degree turn), and body_q disagreed with joint_q for one step | identity in Newton's layout, body_q/body_qd kept consistent in both state buffers |
| `CASCADE_REQUIRE_CUDA=1` always failed | Newton's SimulationView reports no CUDA context handle | attestation reads the live Newton stage: Warp device, context, array devices, MJWarp (not the MuJoCo CPU backend) |
| proof witness refused Newton | contact sensor and convex support were PhysX APIs | MJWarp constraint forces per jaw (`contacts.force`, filled by `SolverMuJoCo.update_contacts`) and the model's shape source; the audit takes each engine's own channel and nothing else |

Two engine-independent fixes from the same stage: a visibly rounded object is grasped at its widest
section, 10 mm inside the pads (`obb_grasp._widest_section_z`; `grasp.depth_fraction: 0.15` put the
pads 16-18 mm above a sphere's centre), and a gated least-squares sphere fit
(`grounding.refine_sphere_center`, CUDA twin `cuda_math.refine_sphere`) re-centres only clouds that
are unmistakably spheres. Hull source, commit, SHA-256 and MIT attribution:
[`assets/newton/PROVENANCE.md`](../assets/newton/PROVENANCE.md).

## Bridge and simulator failures (fixed)

1. **Collision groups are inverted between engines.** PhysX `filteredGroups` is a deny-list; Newton
   assigns every shape a group and generates no contacts across groups (props free-fell from boot).
   The arm-base/tabletop groups are authored only under PhysX.
2. **`set_velocities` differs per engine** (PhysX `(N, 6)`; Newton `(linear, angular)`, each
   `(N, 3)`). `_zero_prop_velocity()` tries both and returns whether it worked, so a silent no-op
   cannot pass as success.
3. **MJWarp silently discards contacts past its cap** (default 200: `Number of Newton contacts
   (1015) exceeded MJWarp limit (200). Increase nconmax.`), so props sank and grasps failed to hold.
   The bridge now sets the buffers before initialization and verifies them live (above).
4. **`settle_timeout_s` was hard-coded to 2.0 s**; Newton needed 3.6 s for a 0.17 rad step.
   `ArmBase.settle_timeout_s` is configurable; `configs/arms/isaac.yaml` sets 6.0.
5. **Newton ignores the PhysX anti-tunnelling knobs** (`maxLinearVelocity`, `enableCCD`,
   `enableSpeculativeCCD`; the bridge still authors them for PhysX). A cube ejected at 7.5 m/s went
   through the 3 cm table at one substep (threshold ~1.8 m/s); each substep is a fresh collision
   check, so substepping is the lever Newton respects.

### 4b. Isaac's Newton stepping bug: use an odd substep count

`isaacsim.physics.newton`'s `NewtonStage.simulate()` swaps `state_0` and `state_1` after every
substep and, with the CUDA graph on (the default), copies back only on the last substep. The
captured graph always reads the buffer that was `state_0` at capture, so the loop is correct only
for an odd substep count with the graph on (even with it off). With 4 substeps physics ran at 0.75x
and every write through the supported tensor API (`RigidPrim.set_world_poses`, prop resets) was
overwritten on the next frame. Measured with Isaac's kit test runner on a free-falling cube
(`test_newton_substeps`): graph on, N=2 at 0.500x and N=4 at 0.750x. The same loop is in 6.0.1,
6.1.0 GA and 7.0.0a1; an upstream fix is under review. Until it ships, the bridge refuses the wrong
parity and defaults to 5.

### Contact softness, resets and placement

1. Released green cube: Newton's default contact (ke 2500, kd 100, critically damped) instead of
   Isaac's under-damped ke 1e4; ke 1e4 with kd 200 still tipped it 4/4. The bridge sets
   `contact_ke/kd` before play (`isaac_materials.NEWTON_DEFAULT_CONTACT`); gripper pads keep their
   raw `mjc:solref` and win by priority.
2. Orange slip during the carry is set by `impratio` (10 → 100), not grasp height.
3. **Reset**: the compatibility reset writes `joint_q`, `joint_qd`, `body_q` and `body_qd`
   consistently in both buffers, resolving the prop by `body_label` and its unique free joint
   (`joint_child`, `joint_type`, `joint_q_start`/`joint_qd_start` offsets; seven coordinates in
   x/y/z/w order, six velocities). It validates the target, both buffers, world-root parent and
   identity joint frames first; incompatible topology is an explicit reset failure.
4. **Initial-state sweeps**: `place_cube` lost its pose write on the affected stepping
   configuration, so six "initial states" were one state measured six times (the 3 cm start-pose
   guard was wider than the 2–3 cm spacing). The `place_prop` op routes through `_newton_teleport`
   (worst error 0.00 cm) and the guard is now 5 mm.
5. **Perception, not physics** (both engines): a constant 1.6 cm localization bias (identical on
   PhysX and Newton) from a shadow-shifted mask (+8.1 px, 0.69 cm) and density-weighted box fitting
   pulled 18.3 mm toward the camera; the arm aimed precisely at a wrong target. Height-filtering the
   table points alone is not a fix (1.63 → 1.57 cm).

Earlier blocker claims did not survive re-testing: offscreen render products return real RGB-D
under the Newton experience (cam0, side and wrist), and NaN at grasp contact was the contact-cap
bug.

## Debugging notes

### The gripper is NOT broken: that result was measurement contamination

A "fingers push each other" result (2.35 mm per finger, right finger stopping at 54.0 mm of
71.5 mm) came from `scripts/night_runner.sh`, respawned by a watchdog cron, issuing commands during
the measurement. With it stopped, on the same USD:

| engine | error per finger | reaches full 71.5 mm travel |
|---|---|---|
| MuJoCo (MJCF direct) | 0.002 mm | yes |
| Isaac + **Newton** | **0.002 mm** | yes |
| Isaac + PhysX | 0.088 mm | yes (71.4 mm) |

Three independent engines agree (engine agreement is the repo's gold metric). The asymmetric
travel (50.0 vs 71.5 mm) comes from the manufacturer URDF; each finger is a single convex STL, no
CoACD. Guard: `benchmark/rig/ablation.py` holds `/tmp/wrc_measuring.lock` during a run and
`~/.hermes/scripts/wrc_watchdog.sh` stands down while it exists (locks older than 6 h ignored).

Before trusting any scene reading:

1. Check `ps aux | grep [n]ight_runner`; the night runner drives picks in the background.
2. Keep a bridge `exec` block under ~200 steps: exceeding the 30 s timeout leaves the sim mid-step
   and every later reading is NaN.
3. Never write `art.set_dof_position_targets(...)` raw from an exec block (it NaNs the Newton
   articulation); use the bridge's `set_joints` op.
4. Once the arm is NaN the scene is unrecoverable (`reset_props` does not fix it): restart the
   bridge and re-verify.
