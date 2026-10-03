# Explicit destination region for the shared MuJoCo C world

Status: local Stage2 candidate; static/contract tests only so far. The original
two-pick task has not yet been run on this candidate. Stage1 withdrawal and
occupied-point refusal remain frozen at22cac914; its physical evidence does
not validate this new selector or observer.

The SO-101 MuJoCo profile declares a90×80mm semantic table area in robot-base
coordinates: x[.15,.24], y[-.16,-.08]. This lies within its workspace on the
explicitly declared horizontal support. The profile's measured top-down
reachable band is approximately r=.12–.28m; corners of the rectangle still
need actual IK and complete-path checks. It is an area request, not a promise
that every point or payload fits. `place_at(x,y,z)` remains exact and never
selects a neighboring point. Only the default `pick_and_place` destination
uses this region. The Warp profile explicitly disables this C-world channel.

The selector erodes semantic borders by the measured collision footprint,
then packs deterministically from the remaining edges. It does not use object
names or a fixed list of slots. A35mm footprint leaves55mm of center range
along the90mm side. Adding20mm to **both border erosions** would leave15mm,
which is not the implemented contract:20mm is the existing physical clearance
between objects/robot, separate from the semantic border on an infinite plane.
Tilt, different extents, IK or collision geometry may make the area unusable.
No6cm acceptance radius is used by the selector.
Edge-packed candidates can put a predicted footprint exactly on the semantic
boundary; actual placement error may therefore refute containment. That is a
possible physical failure, not a reason to accept an outside footprint.

Each candidate uses the same ordinary placement IK (extracted without changing
its yaw order or ceilings), the measured held transform in detached scratch,
full carry collision checks, an empty-tool release escape, and the checked home
route. The candidate must also preserve measured object separation and contain
its predicted oriented footprint. All candidates share a3-second planning
budget, generation and unchanged live-state binding. This budget cannot renew
per candidate. The ordinary skill then repeats its actual-state preflight and
post-descent release checks; selecting a point grants no release authority.
No feasible point returns a safe holding/refusal result without home or a
physical search retry. Results expose the region, chosen point and rejections.

## Independent passive evidence

`_MjcEngine.step(n)` remains the sole simulation writer. Immediately before its
last existing step it copies qpos/qvel/time; after the batch it passes those
inputs and the native solved results to `PlacementHistory`. The observer never
calls a step or a forward on live data. MuJoCo computes forward dynamics before
integration, so contact/FK/velocity inputs cannot simply be stamped as final
state. Euler, implicit and implicitfast have the admitted single-stage phase;
RK4 is rejected. [MuJoCo simulation loop](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop).

The journal explicitly separates solve input time/state from integrated final
time/state. Detached FK is compared with native solved `geom_xpos`; its final
footprint and velocity are checked separately. It records the last contact
interval of **each batch**, not every substep. Contact frames and solved
force/torque retain native values and collider IDs; a contact is visited once,
even when both existing coincident `floor` and `demo_floor` are present.
[Contact fields](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjcontact),
[force extraction](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce).

The read channel requires its exact world/data objects, complete compiled MJB
fingerprint plus descriptor/SDK, epoch, monotonically advancing owner clock,
complete current-state signature and explicit effective collider registry.
A channel exception remains sticky until a real world-reset boundary; a reset
changes epoch and clears history/release/prefix state. No old sample is relabeled.
Capacity/numerical warnings make the channel unavailable. Reads do not create a
world, connect a lazy arm, refresh timestamps or advance a paused simulation.

A confirmation needs at least6 distinct batch samples spanning at least0.6
physical seconds, gaps at most0.25s, after a measured jaw transition to>=.98
open. The window ends at the last solve of the current state; it may accumulate
during withdrawal and home, so it does not require extra physics after home.
Every sampled full collision footprint must be inside the region, with a
positive upward solved support reaction, no active arm contact (including zero-force
active constraints), and no other loaded contact. The
sampled object must remain within5mm pairwise displacement, below.02m/s and
.2rad/s. The integrated final footprint, velocity and jaw state are also checked.
These thresholds do not replace the original unchanged6cm end-task assertion.
Missing or inconsistent evidence is unverified; failed measured predicates are
refuted. The configured area cannot fall back to a center-only confirmation.

Previously confirmed objects are checked again and sampled intervening
support/contact/containment/velocity failures are retained. This preserves the
already placed prefix across a later object's approach, placement and home at
the stated batch sampling resolution. It does not claim continuous support.
The verifier, rather than the actor, owns this prefix record. A top-level
`place_at` request outside the region may explicitly replace that object's old
area goal only after the ordinary point verifier confirms it on physics and a
new, same-epoch passive window confirms release/support/rest after the request.
The old disturbance and request remain in a bounded retirement record and in
the ordinary result/trace. Nested default placement, failed/unknown point
results, old samples and reset_stop do not retire anything. This ledger change
grants no motion authority and does not alter place_at coordinates or verdicts.
Repeated passive reads cannot clear a target's prior disturbance. Retirement
audits and fingerprints run outside the short harness stop-state lock; only the
final ledger mutation holds it, with world-to-harness lock order. A separate
monotonic cancellation token preserves estop-then-reset history without changing
the existing halt-generation semantics. Stop never waits for the world/audit
lock, and a cancellation before the commit keeps the old obligation intact.

## Planned next validation, not yet executed

1. Two bounded MuJoCo C CPU microfixtures establish contact sign, solve/final
   phase, reset invalidation and no extra steps. Synthetic malformed records
   remain separately labeled. No robot controller or renderer in these cases.
2. Run the original unmodified
   `test_memory_frames_record_two_physics_confirmed_picks` on MuJoCo3.14.0 with
   the same26 pinned SO-101 assets, private stores and software Mesa EGL.
   Preserve trace, full independent journal receipts, sampled phases, final
   images, source/asset/package/protected-store hashes and terminal cleanup.
   Check both full region verdicts, final prefix, and the original6cm assertion.
3. Retain the separate exact-point occupied-destination/refusal and reset tests;
   they explicitly disable optional area selection. No success is inferred
   from this preparatory document, static IK or the old Stage1 episode.
