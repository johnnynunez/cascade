# Newton physics engine: setup and validation

`scripts/isaac_bridge.py` defaults to `--engine newton`. The Spark launcher
defaults to PhysX; select Newton explicitly with `./run.sh isaac --engine newton`.

Historical result from 2026-07-31 (before strict whole-object acceptance): `pick_and_place` succeeds in 25.6 s with the
cube physics-confirmed inside the bin (31.9 cm displacement), postcondition
`confirmed via physics`, articulation finite afterwards, 260 tests green.

## Independent placement corrections (2026-09-30)

A completed motion routine is not placement acceptance. The diagnostic
`benchmark/diagnostics/kitchen_acceptance.py` runs all five props through the
real skills, then applies the existing passive GPU audit. It checks actual
bilateral grasp forces, full destination entry while held, complete final
footprint, support, release, stability, upright cubes/can, current cameras and
all five reset poses. Its exit status is nonzero for any failed case. Each
campaign records source hashes and rejects source changes during the run.

Independent observations reproduced four failures despite the earlier
contact tuning:

- The orange arrived centered and held by both jaws, then rolled about 4 cm
  after release and protruded beyond the box interior.
- The pink cube toppled even when the green cube passed under the same
  contact settings. Lowering a single fixed drop height did not fix every prop.
- The can could still oscillate after release: the previous fixed wall-clock
  dwell let retraction begin before Newton's rate-limited jaws fully opened.
- At 120 Hz, both three and five substeps missed the green cube: lifting
  started while the jaws were still closing. A successful 60 Hz case was
  insufficient to validate the launcher configuration.

The kitchen profile now estimates the TCP-to-bottom distance from measured
joints and the calibrated pickup support plane before lift. A partial point
cloud can miss the bottom by several millimeters: using its lowest visible
point pressed the lemon against the box floor and made it jump when released.
At placement the runtime adds the measured offset to the destination support
height and 15 mm clearance. A 6 mm margin still pressed a lemon into the floor
after its orientation changed during carry; the 15 mm candidate passed three
isolated lemon trials with three substeps, but needs the full matrix.
The box floor is 4 mm above the counter. This vertical calculation uses
proprioception and calibration. Profiles without a
calibrated pickup plane retain the observed point-cloud bound.
Explicit `place_at(z=...)` commands retain their requested height. If no valid
held geometry exists, the configured release height remains the fallback.
The production runtime's existing truth-assisted held-object XY/slip
compensation remains enabled. These trials are not perception-only; the
passive witness itself never feeds the controller.

Transport has its own 140 mm clearance and vertical pre-lift, independent of
the final descent. Before retreat the runtime waits for at least 98% measured
jaw opening, with an 8 s limit; missing confirmation stops retreat and home
motion and returns failure. Closing also waits for measured travel or a
stable jaw position before lifting, with an 8 s limit per stage; missing or
still-moving feedback returns failure. Physical grasp acceptance remains a
separate check after the lift.

With the calibrated pickup plane, all five props passed one round at 120 Hz
on Newton with three substeps, Newton with five substeps, and PhysX:
`runs/support-plane-cuda-{nw3,nw5,physx}-120` (15/15 cases, including resets).
The later 6 mm-clearance campaign passed 25/25 on Newton with five substeps.
Newton with three substeps passed 15 cases before lemon round four failed
whole-object containment. PhysX passed five cases before lemon round two
failed the unchanged 0.2 rad/s angular-speed bound (measured 0.262 rad/s).
Increasing clearance alone did not fix PhysX. These receipts are retained in
`runs/acceptance-calibrated-{nw3,nw5,physx}-120`; they do not establish
acceptance of the current 15 mm candidate. The complete matrix and launcher
proof remain pending. No acceptance bounds have been relaxed.

A controlled replay of the failed PhysX resting pose measured 0.260 rad/s
with 16 and 32 position iterations, 0.044 with 64, and 0.260 again after
restoring 16. The bridge now authors 64 position iterations for the kitchen
props before play; clean-start acceptance of this change is pending.

The final harness requires CUDA perception, matching the Spark launcher.
Earlier CPU-perception diagnostics are not counted as that profile: its
existing can-center refinement is CUDA-only, and a CPU run reached release
with the can tilted about 17 degrees. The first strict run also caught an
attempt to convert a CUDA point cloud directly to NumPy in the new rounded
grasp planner. Horizontal slicing and width percentiles now run on CUDA;
only compact section geometry crosses to the host planner. Three CUDA
parity tests cover rounded, cylindrical and sparse clouds.

The complete local regression suite passes **2250 tests**, with 46 skips and
2 deselected tests (255.83 s). It includes launcher delivery and failure
propagation when jaw opening cannot be confirmed. The motion-recorder fixture
reports its recorded gripper state; conservative fallback heights remain in
place for unknown held geometry. Full physical campaign and launcher results
must still be recorded before declaring acceptance.

Run a campaign from this checkout against a matching clean bridge:

```sh
.venv/bin/python benchmark/diagnostics/kitchen_acceptance.py \
  --port 8681 --engine newton --rounds 5 --output runs/acceptance-newton5
```

## Earlier kitchen contact tuning (2026-09-29)

The Spark kitchen (`demo/scene/kitchen_config.json`: green cube to the green
square, orange into the open box) has a Newton path using the same skills,
launcher and physical acceptance criteria as PhysX. PhysX stays the Spark
default; `launch.sh --engine newton` is an explicit opt-in. Contact and servo
settings are Newton-specific; the perception and grasp fixes affect both
engines. Full physical acceptance of the combined changes is still pending.
Historical `ok=True` skill results and center distances below are diagnostics,
not proof of complete containment, support, release or a successful reset.

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

Two earlier failures were engine-independent planner/perception issues.
The shared fixes apply to both engines:

- **The orange was pinched above its equator.** `grasp.depth_fraction: 0.15`
  suits boxes; on a sphere it puts the pads 16-18 mm above the centre, where
  the squeeze pushes the fruit out. PhysX happened to hold it; plain MuJoCo
  loses it with any friction up to 2.0 and condim 3/4/6. The planner now
  grasps a visibly rounded object at its widest section, 10 mm inside the
  pads (`obb_grasp._widest_section_z`); boxes and upright cans are unchanged.
- **The orange's centre was 13 mm off.** The view-ray re-centring cannot see
  how deep a ball goes. A gated least-squares sphere fit
  (`grounding.refine_sphere_center`, CUDA twin in `cuda_math.refine_sphere`)
  moves only the x/y centre, only for clouds that are unmistakably spheres:
  0.7 mm (PhysX) and 0.5 mm (Newton) on the real kitchen clouds; the cubes,
  lemon and can are rejected and keep their centre.

The hull source, exact commit, SHA-256 and MIT attribution are recorded in
[`assets/newton/PROVENANCE.md`](../assets/newton/PROVENANCE.md). The original
Menagerie commit attribution was not reproducible and has been corrected.

## Historical blocker investigations

The bridge previously defaulted to PhysX and cited two reasons, both dated
2026-07-18. Re-tested against the current build, neither survived.

**"Offscreen render products return garbage under the newton kit experience;
only viewport capture works. Fine for physics work, useless for RGB-D."**

False on this build. All three cameras return real RGB-D through the same code
path:

| camera | distinct colours | valid depth |
|---|---|---|
| cam0 | 19131 | 85.2% |
| side | 11748 | 47.1% |
| wrist | 2225 | 93.2% |

Confirmed by eye on a captured frame: arm, bin, both cubes, correct shadows.

**"Newton NaNs at GRASP CONTACT (fingers closing on an object)."**

Not reproducible. Closing the jaws onto the cube for 120 steps leaves the
articulation finite. The NaN that looked like this was really bug 3 below.

## Historical bridge and simulator failures

### 1. Collision-group semantics are inverted between engines

The bridge authors two `UsdPhysics.CollisionGroup`s so the arm base and the
tabletop (both static, both at z=0) do not generate contacts and NaN PhysX at
boot.

- **PhysX**: `filteredGroups` is an explicit deny-list. A collider in *no*
  group still collides with everything, so dynamic props fall onto the table
  normally.
- **Newton**: every shape ends up assigned to a group, and shapes in
  *different* groups generate no contact pairs at all.

Measured: 405 registered contact pairs, **zero cross-group**. The 5 furniture
shapes were isolated from the other 227, so both cubes free-fell to
z = -227849 m at 6244 m/s with x/y untouched — pure free fall, straight from
boot.

Fix: author the groups only under PhysX. Newton resolves the static overlap
without them.

### 2. `set_velocities` has a different signature per engine

```
PhysX   set_velocities(v)                 v shaped (N, 6)
Newton  set_velocities(linear, angular)   each shaped (N, 3)
```

The call sat inside a bare `except Exception: pass`, so under Newton the
velocity was **never zeroed**. A prop that picked up speed kept it and
tunnelled on the next step. Every settle attempt failed the same way, and the
prop-reset path was marked "PhysX-only" for the same reason.

Fix: `_zero_prop_velocity()` tries both signatures and *returns whether it
worked*, so a silent no-op cannot masquerade as success.

### 3. MJWarp silently discards contacts past its cap

The headline bug. Authoring `newton:solver:nconmax` on the prim is not enough:
the MJWarp solver instantiates with its own default of **200** and drops every
contact beyond it, printing

```
Number of Newton contacts (1015) exceeded MJWarp limit (200). Increase nconmax.
```

once per step — 4311 times in a single boot. Dropped contacts is exactly the
observed symptom: props resting on the table for a while and then sinking
through it, fingers closing on an object without holding it.

Fix: `_raise_newton_contact_cap()` sets `solver_cfg.nconmax = 8192` on the live
solver. It runs twice — once after boot and again just before `_settle_props`,
because the solver only exists after physics has been created and stepped.

Before / after:

```
settle attempt 1..4 + hard-clamp      ->  props settled cleanly (attempt 1)
prop z = -592.375                     ->  prop z = 0.040
4311 discarded-contact errors         ->  0
```

### 4. `settle_timeout_s` was hard-coded to 2.0 s

Newton bleeds off the last of the tracking error more slowly than PhysX. A
0.17 rad step measured **3.6 s** to come inside `settle_tol`, so
`stream_to()` returned False and every grasp reported *"did not settle at
pregrasp pose"* on a pose the arm was reaching correctly.

Fix: `ArmBase.settle_timeout_s` is now configurable; `configs/arms/isaac.yaml`
sets 6.0.

### 4. Newton ignores the PhysX anti-tunnelling knobs

The headline bug for manipulation, and the one that took longest to find.

Symptom: `pick_and_place` would lift the cube, then the cube would vanish and
every later episode in a sweep would report an identical `0.0 cm`.

Instrumenting a real grasp caught the moment:

```
t+29.97  cube z= 0.025  speed 0.000   fingers on target
t+30.14  cube z= 0.029  speed 0.296   finger error +12.8 mm  <- jaws pop open
t+31.42  cube z=-0.188  speed 7.481                          <- through the table
```

The diagnosis hinges on what was ruled out:

| candidate | measurement | verdict |
|---|---|---|
| contact cap | peak 1695 vs cap 4600, buffer 6985 | not it |
| depenetration impulse | deepest penetration 0.000 mm | not it |
| transport inertia | max abs(dq) 0.075 rad/s at ejection | not it |
| grasp close | 120 steps on a parked cube, 0.000 m/s | not it |

A body passing *through* a static collider while everything moves slowly is a
**missed** contact, not a mis-resolved one. The scene runs `num_substeps=1` at
1/60 s, so a body only needs **~1.8 m/s to clear the 3 cm table slab between
two collision checks** — and the props ship with

```
physxRigidBody:maxLinearVelocity     = inf
physxRigidBody:enableCCD             = False
physxRigidBody:enableSpeculativeCCD  = False
```

Authoring those three attributes does nothing: **Newton does not read them.**
Its model exposes `particle_max_velocity` and no rigid-body speed limit at
all. (Verified by dumping the model's attributes — the USD values are present
and correct on the prim, and the cube still tunnels.)

The lever Newton *does* respect is substepping: each substep is a fresh
collision check, so N substeps multiply the tunnelling threshold by N.

Fix: `num_substeps 1 -> 4` (threshold ~1.8 -> ~7.2 m/s, just above the
measured 7.5 m/s ejection) plus `contact_margin 0.01 -> 0.02` so the solver
sees an approaching body before it overlaps, and `njmax 1200 -> 32768` (the
other half of the pair the asset's evidence package documents). The count is
now 5: 4 hits Isaac's stepping bug (next section).

Result: cube survives, `pick_and_place` succeeds in 23.2 s, postcondition
`confirmed via physics`, arm finite.

### 4b. Isaac's Newton stepping bug: use an odd substep count

`isaacsim.physics.newton`'s `NewtonStage.simulate()` swaps `state_0` and
`state_1` after every substep and, with the CUDA graph on (the default),
copies the result back only on the last substep. The captured graph always
reads the buffer that was `state_0` at capture, so the loop is only right for
an odd substep count with the graph on (an even one with it off). With 4
substeps every frame replayed one substep: physics ran at 0.75x, and every
write through the supported tensor API (`RigidPrim.set_world_poses`, the prop
resets) landed in the buffer the graph did not read and was overwritten on the
next frame. That was misread for a while as "MuJoCo is reduced-coordinate".

Measured with Isaac's own kit test runner on a contact-free falling cube
(`test_newton_substeps`, local development build): graph on, N=2 runs
at 0.500x and N=4 at 0.750x; graph off, N=1 and N=3 put `state_0` on a
different buffer every frame. The same loop is in 6.0.1, 6.1.0 GA and
7.0.0a1. An upstream fix is under review: it keeps `state_0` on one buffer and
passes the seven parity/graph regression cases. Until that ships, the
bridge refuses a substep count with the wrong parity for the graph setting and
defaults to 5.

### 4c. Contact softness and friction impedance for the kitchen props

With the stepping bug out of the way two kitchen failures remained, both from
contact parameters rather than geometry. Each was isolated by changing one
parameter on a live bridge (same code, same grasp heights, both GPUs, 3 or 5
substeps) and then re-measured from code on fresh bridges.

- **Released green cube lands on its side.** It is released 7 cm above the
  counter. Isaac's `NewtonConfig` default contact for every collider without
  an authored material is ke 1e4, kd 100, which Newton's MuJoCo solver
  converts to solref (0.02, 0.5): under-damped. The cube bounced over in 5 of
  7 rounds (0.8 m/s, 12-15 rad/s at the impact). Newton's own `ShapeConfig`
  default, ke 2500, kd 100 (solref 0.02/1.0, critically damped): 8/8 upright,
  0.3-0.6 cm from the square's centre. Critical damping alone is not it: ke 1e4
  with kd 200 (solref 0.01/1.0) tipped it 4/4. The bridge now sets
  `contact_ke/kd` before play (`isaac_materials.NEWTON_DEFAULT_CONTACT`); the
  gripper pads keep their raw `mjc:solref` and win every pad contact by
  priority.
- **The orange slides out during the carry.** Pinned grasp heights (15.2,
  17.5, 21.5 mm) showed the grasp height is not the cause. In the gripper
  frame the orange slid 5-15 mm during the 30 cm carry at the vendor model's
  `impratio` 10. `impratio` weights the friction constraints against the
  normal ones; at 100 the slip is 0.3-2.7 mm and 6/6 rounds land in the box,
  under both contact settings above. (Why softening the props' contact made
  the slip worse at impratio 10 is not explained: the pad-orange contacts use
  the pad's solref.)

Those tuning measurements precede the stricter independent acceptance above;
center distances alone do not establish containment or support.

### 5. Reset prop identity and both state buffers

The earlier diagnosis that `RigidPrim.set_world_poses` cannot reset a resting
MuJoCo body was incorrect. Isaac's tensor backend writes the free joint's
coordinates. The state-buffer bug described above could overwrite that write
on the following frame; reduced-coordinate dynamics alone do not cause it.

The compatibility reset writes `joint_q`, `joint_qd`, `body_q` and `body_qd`
consistently in both buffers. A free joint has seven coordinates (position
and quaternion in x/y/z/w order) and six velocity degrees of freedom. Each
array therefore needs its own offset.

`joint_q_start` and `joint_qd_start` are valid topology tables. Repeated
entries are expected for fixed joints, which have zero degrees of freedom.
The bridge now resolves the prop by `body_label`, finds its unique free joint
through `joint_child` and `joint_type`, and reads those offsets. Matching the
old position was ambiguous for coincident props and failed if a bad step left
the old state non-finite.

Before writing, the reset validates the target, both buffers, world-root
parent and identity joint frames. The five live kitchen props satisfy this
frame contract (checked on 2026-09-30). Incompatible topology is an explicit
reset failure. Unit tests cover duplicate positions, fixed joints, non-finite
old poses, invalid targets and atomic validation of both buffers. Physical
reset acceptance is recorded separately by the kitchen acceptance harness.

### 6. A sweep over initial states that wasn't (FIXED)

Same root cause, worse consequence. `place_cube` in the ablation harness did
`reset_props` (correct) and then a `RigidPrim.set_world_poses` to the episode's
requested position — a write lost on the affected Newton stepping configuration. Measured:

```
asked ->  actual                 error
(0.170,0.150) -> (0.170,0.150)    0.05 cm
(0.175,0.130) -> (0.170,0.150)    2.06 cm  <-- did not land
(0.165,0.170) -> (0.170,0.150)    2.06 cm  <-- did not land
(0.180,0.140) -> (0.170,0.150)    1.41 cm  <-- did not land
(0.160,0.160) -> (0.170,0.150)    1.41 cm  <-- did not land
(0.172,0.120) -> (0.170,0.150)    3.01 cm  <-- did not land
```

Every episode ran at the **spawn**. The sweep reported six independent initial
states while measuring one state six times; the between-episode variance was
noise, not coverage.

The start-pose guard did not catch it because its tolerance (3 cm) was **wider
than the spacing between the states themselves** (2–3 cm). A guard that cannot
distinguish state *i* from state *j* is not guarding anything — it is now
5 mm.

Fix: a `place_prop` bridge op that routes through `_newton_teleport`. After it,
all six land exactly:

```
worst placement error: 0.00 cm
```

### 7. The remaining grasp failure is PERCEPTION, not physics (both engines)

With the instrumentation bugs closed, the sweep failed 6/6 honestly — so the
next question was why. It is not Newton, and it is not the gripper.

Isolation chain, each step measured rather than argued:

1. The cube is **knocked away during the approach**: `air grasp: gripper closed
   fully, object not held`, with the cube displaced 7.9 cm before the jaws
   close.
2. At first contact the gripper is **1.8 cm off-centre** on a cube whose
   half-width is 2.5 cm — a finger catches the edge and shoves it.
3. Who is wrong, perception or kinematics?

   | | position |
   |---|---|
   | physics truth | (+0.1686, +0.1510) |
   | perception | (+0.1810, +0.1410) → **1.60 cm error** |
   | gripper landed | (+0.1806, +0.1406) → **0.01 cm from the belief** |

   The arm aims *precisely* at a wrong target.
4. The bias is **constant**: dx=+1.25 cm, dy=−1.00 cm, std ≤ 0.05 cm across
   four table positions.
5. It is **identical under PhysX** (1.61 cm) and Newton (1.60 cm) — so it is
   engine-independent and predates the Newton switch entirely.
6. The extrinsics in `configs/cameras/isaac.yaml` match the bridge's printed
   values byte for byte, so it is not a stale calibration.
7. **Mechanism — two independent contributions, both measured.**

   `_localize` does not deproject one pixel: it builds a point cloud from the
   detection mask and takes the centre of an oriented box over it. Two things
   go wrong, in different directions.

   **(a) The mask sits low, because of the cube's shadow.** In pixels:

   | | u | v |
   |---|---|---|
   | true cube projection | 663…748 | 156…253 |
   | detector bbox | 663…748 | **143…264** |

   Horizontally it is exact (mask centroid off by **+0.4 px**); vertically the
   box runs 13 px high and 11 px low, and the mask centroid sits **+8.1 px**
   below the true one. At 0.93 m that is **0.69 cm** — 42 % of the error.

   **(b) The fitted box centre is pulled toward the camera by 18.3 mm.** The
   cloud *does* wrap the cube (59.5 mm of depth span for a 5 cm object), so
   this is not a one-face shell. It is density: near faces get many more
   pixels per unit area than far ones, so the box centre is dragged forward.
   The fitted extent, **7.2 × 7.3 × 4.3 cm**, is inflated laterally by the
   shadow/table pixels and compressed in z.

   Height-filtering the table points alone is **not** a fix: it moves the error
   only 1.63 → 1.57 cm (kept fraction 0.96), because the table skirt is a
   symptom of the same bad mask rather than the main term.

> **Retraction.** An earlier version of this section blamed *surface-vs-centre*
> depth alone: measured range 0.8950 m vs a true 0.9336 m (38.6 mm ≈ half a
> diagonal), with a +2.5 cm push along the ray cutting the error 3.17 → 1.15 cm.
> That test was real but it measured the **wrong pipeline** — a single-pixel
> deprojection I wrote, not `_localize`. The tell was that the real pipeline is
> *more* accurate (1.6 cm) than my naive round-trip (3.2 cm). The shipped
> pipeline's cloud genuinely wraps the object; the depth-direction bias is
> density-weighting, not a missing far face, and it comes with a separate
> shadow-driven mask shift. A blind "+2.5 cm along the ray" would have
> overshot.

## Debugging notes

### The gripper is NOT broken — that result was measurement contamination

A "the fingers push each other" investigation produced this, which looked like
a real asset bug and is **wrong**:

```
per finger 2.35 mm off target, sum of the two errors 0.01 mm
right finger tops out at 54.0 mm against a commanded 71.5 mm
```

With `scripts/night_runner.sh` killed and the `cascade-watchdog` cron paused,
the same measurement on the same USD:

| engine | error per finger | reaches full 71.5 mm travel |
|---|---|---|
| MuJoCo (MJCF direct) | 0.002 mm | yes |
| Isaac + **Newton** | **0.002 mm** | yes |
| Isaac + PhysX | 0.088 mm | yes (71.4 mm) |

Three independent engines agree — the repo's own "engine agreement is the gold
metric" criterion. The drives, the limits and the collision meshes are fine.

The tell was in the settle trace, and it should have stopped the investigation
much earlier:

```
 3.0 s   err  -0.016 / +0.030 mm     <- converging normally
 5.2 s   err +19.61  / +27.16 mm     <- something GRABS the gripper
32.1 s   err  -5.37  / -16.85 mm     <- oscillating
```

A constant mechanical offset does not jump from 0.03 mm to 27 mm. That is
another process issuing commands: `night_runner.sh`, respawned by the watchdog
cron I had re-enabled myself a few hours earlier.

Claims retracted along the way, each refuted by its own data:

- *"finite-stiffness PD response under inertial load"* — deviation does **not**
  scale with speed; the FAST sweep was the *smallest* (1.29 mm vs 3.49 mm slow).
- *"`frictionloss=0.2` static friction"* — friction must reverse sign with
  sweep direction. It does not.
- *"CoACD decomposition inflates the collision hulls"* — **CoACD is not used
  anywhere in this pipeline.** It appears in one line of one evidence README.
  Each finger has a single convex STL (216 triangles), not a decomposition.
- *"the asymmetric travel (50.0 vs 71.5 mm) is a bug"* — it comes from the
  **manufacturer URDF** and the asset's own evidence validates both fingers
  against those limits to ~1e-08 m.

**Guard added.** `benchmark/rig/ablation.py` now takes `/tmp/wrc_measuring.lock`
for the duration of a run and `~/.hermes/scripts/wrc_watchdog.sh` stands down
while it exists (ignoring locks older than 6 h). Pausing the cron by hand was
not enough — it gets re-enabled and forgotten, and this is the *second*
investigation it has corrupted.


- **`scripts/night_runner.sh` may be running.** It drives the arm through picks
  in the background and corrupted several measurements here. Check
  `ps aux | grep [n]ight_runner` before trusting any scene reading.
- **A bridge `exec` block that exceeds the 30 s timeout leaves the sim
  mid-step**, and every later reading comes back NaN. Keep exec blocks under
  ~200 steps or split them across calls.
- **Do not write `art.set_dof_position_targets(...)` raw from an exec block** —
  it NaNs the Newton articulation. Use the bridge's `set_joints` op, which is
  the path the skills take.
- **Once the arm is NaN the scene is unrecoverable.** `reset_props` does not
  fix it; restart the bridge and re-verify before measuring anything.
