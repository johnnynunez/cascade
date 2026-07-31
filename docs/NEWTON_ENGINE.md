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

## Debugging notes

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
