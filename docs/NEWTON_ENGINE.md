# Newton is the default physics engine

`scripts/isaac_bridge.py` defaults to `--engine newton`. Pass `--engine physx`
for the old path.

Verified end to end on 2026-07-31: `pick_and_place` succeeds in 25.6 s with the
cube physics-confirmed inside the bin (31.9 cm displacement), postcondition
`confirmed via physics`, articulation finite afterwards, 260 tests green.

## The blockers that were not real

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

## The four bugs that were real — all ours, not Newton's

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
other half of the pair the asset's evidence package documents).

Result: cube survives, `pick_and_place` succeeds in 23.2 s, postcondition
`confirmed via physics`, arm finite.

### 5. Teleporting a resting prop needs `state.joint_q` (FIXED)

A teleport of a **resting** body silently did nothing under Newton. The write
appeared to succeed and one step later the body was back at its old pose.

The reason is in Newton's own source
(`newton/_src/solvers/mujoco/solver_mujoco.py`, `reset_state` docstring):

> Because MuJoCo is a reduced-coordinate solver, `state.body_q` /
> `state.body_qd` are **derived from the joint coordinates by forward
> kinematics** on the next `step()`; the corresponding `BODY_Q` / `BODY_QD`
> flags **are not actionable here and are ignored**.

and `step()` calls `_update_mjc_data(..., state_in)` every step, which pushes
the joint coordinates into `mjw_data.qpos`. Everything else is downstream:

| write target | result |
|---|---|
| `RigidPrim.set_world_poses` | reverted after 1 step |
| `state.body_q` (one buffer) | reverted after 1 step |
| `state.body_q` (both buffers) | body **ejected at 72 m/s** |
| `mjw_data.qpos` | reverted after 1 step |
| **`state.joint_q`** | **sticks, \|vel\| 0.000, stable** |

Two traps in the fix:

1. A free body has **7 coordinates** in `joint_q` (3 pos + 4 quat) but **6
   dofs** in `joint_qd`. Reusing the coordinate index for the velocity slice
   zeroes the wrong entries, leaves the body's real velocity intact, and it
   flies off — that failure read `[12.5, -22.7]` after 40 steps.
2. `model.joint_q_start` is unreliable on this build (its entries repeat), so
   `_newton_teleport` locates the slice by **matching the prop's current
   position** instead of trusting the offsets table.

Verified end to end:

```
1. fresh scene      : [0.170, 0.150, 0.040]
2. after pick       : [0.194, 0.119, 0.025]
3. after reset_props: [0.170, 0.150, 0.040]   <- 0.2 mm from spawn
```

### 6. A sweep over initial states that wasn't (FIXED)

Same root cause, worse consequence. `place_cube` in the ablation harness did
`reset_props` (correct) and then a `RigidPrim.set_world_poses` to the episode's
requested position — a **silent no-op** under Newton. Measured:

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
