# Reach envelope of the reBot B601-RS (B45, measured 2026-10-09)

Seeed's feedback after the 8 October session: *"very limited AREA constraints
for skills, the object has to be really close to the arm"*. This page records
what limits the reach, how it was measured, and the opt-in profiles that use
the measurement. Nothing here is validated in physics or on hardware yet (see
[Live validation owed](#live-validation-owed)).

## What limited the reach

`configs/demo.yaml` ships the TCP workspace box x 0.10..0.50 m, y ±0.30 m,
drawn from a top-down-only IK probe, and the analytic (OBB) planner emits only
straight-down grasps. Measured with the runtime's own IK seed (`home_q`), IK
margin (0.025 rad > harness `joint_margin` 0.02), flip twin, pregrasp offset
(0.04 m) and harness vet, **top-down grasps reach only r ≤ 0.447 m from the base
at 2 cm grasp height, 0.430 m at 5 cm, 0.414 m at 7 cm and 0.378 m at 10 cm**.
So the far part of the shipped box was already out of reach for top-down
grasps: the limit was the approach family as much as the box. (On the y = 0
line top-down reaches x = 0.425 / 0.425 / 0.400 / 0.375 m.)

## The study

`scripts/reachability_study.py` (CPU only, ~5 min with 32 workers on an idle 128-core box, 17 min under a loaded one) evaluates a
TCP grid over the table — x 0..0.80 m, y ±0.80 m, 2.5 cm step, grasp heights
0.02 / 0.05 / 0.07 / 0.10 m (the heights the shipped configs' objects are
grasped at, listed in the JSON's `z_sources`) — for each approach family and
four jaw rolls (0/45/90/135°; the selector adds the 180° flip twin):

| family | approach |
| --- | --- |
| `topdown` | straight down |
| `out15` / `out30` / `out45` | leaning 15/30/45° away from the base (in the vertical plane through the base→point line) |
| `about_radial15/30/45` | leaning 15/30/45° sideways (rotation about the base→point line), either side |
| `diag30` / `diag45` | leaning 30/45° halfway between out and sideways, either side |
| `side` | horizontal, pointing away from the base |

Every candidate goes through the real `select_grasp` (pregrasp IK seeded from
`home_q`, grasp IK seeded from the pregrasp, flip twin) and the harness half
of `skill_grasp_object`'s candidate vet in its order: `vet_pose(pregrasp)`
without exemption, `vet_segment(home → pregrasp)` over the streamed waypoint
profile, and the seven descent samples under the grasp exemption cylinder.
Live-scene gates (occupancy, observed fingers, cuMotion spheres) are not
modelled: they only refuse more. Three passes: the box opened (where each
family reaches at all), the shipped box (what reaches today), and the derived
opt-in box (verifying the envelope under the limit that will gate motion).
Results: [`evidence/reach-envelope-20261009/rebot_reachability.json`](evidence/reach-envelope-20261009/rebot_reachability.json)
(provenance: URDF sha256, settings, script sha256; every reachable point with
its per-roll mask and the lowest joint origin along its descent).

Maximum reach r (m) from the base, box opened, any roll:

| family | z 0.02 | z 0.05 | z 0.07 | z 0.10 |
| --- | --- | --- | --- | --- |
| topdown | 0.447 | 0.430 | 0.414 | 0.378 |
| out15 | 0.540 | 0.526 | 0.513 | 0.495 |
| out30 | 0.621 | 0.615 | 0.605 | 0.594 |
| out45 | 0.693 | 0.688 | 0.683 | 0.673 |
| about_radial30 | 0.430 | 0.407 | 0.388 | 0.340 |
| about_radial45 | 0.391 | 0.364 | 0.329 | none |
| diag30 | 0.571 | 0.555 | 0.548 | 0.530 |
| diag45 | 0.621 | 0.613 | 0.605 | 0.594 |
| side | 0.746 | 0.763 | 0.772 | 0.779 |

- Leaning **out** is what extends reach; leaning **sideways** adds nothing (it
  reaches less than top-down) — so a tilt "about the radial axis" in the
  literal sense is not useful on this arm.
- `side` reaches furthest but only far out (r ≥ 0.51–0.59 m): it is the outer
  ring, not a general approach.
- Inside the shipped box, angled approaches already reach 137–203 more grid
  points per height than top-down (pass 2) — GraspGen-X, which proposes
  angled grasps, can use those today; the analytic planner could not.

## The opt-in envelope

`derive_envelope` grows the shipped box only over grid cells where an
approach the opt-in planner emits (`topdown`, `out30`, `out45`, `side`, with
the planner's horizontal jaw, roll 0) was measured reachable **at every one
of the four grasp heights**. It returns the largest box that contains the
shipped box and whose every grid sample outside the shipped box is admitted —
an inscribed box, not a bounding box, so no unmeasured corner is admitted;
edges sit on the outermost admitted samples. The near-base edge (x 0.10) and
the z range are kept: neither IK nor the harness models self-collision near
the base.

Result: **x 0.10..0.55 m, y −0.50..0.50 m** (shipped: x 0.10..0.50, y
±0.30). Under that box (pass 3): 2833 grid points reachable with the planner's
families and roll vs 953 top-down points in the shipped box today (+1880,
454–500 per grasp height), none lost, none outside the box, and every joint
origin of those solutions stays ≥ 49 mm above the table along the descent.

The opt-in profiles carry exactly that box and the matching tilts, and change
nothing else (pinned by `tests/test_reach_envelope.py`):

- `rebot_rs_reach` (real arm), `isaac_reach` (Isaac Sim), `mock_reach`
  (offline);
- `safety.workspace: {min: [0.10, -0.50, -0.01], max: [0.55, 0.50, 0.55]}` —
  the new limit for that profile, documented as measured; every other harness
  gate (joint margin, velocity cap, table clearance, keep-outs, occupancy,
  the IK margin invariant) is the parent profile's, unchanged in kind;
- `grasp.angled_approach_tilts_deg: [30, 45, 90]` — the analytic planner
  (`obb_grasp.plan_grasps_from_fix`) appends tilted versions of every top-down
  footprint candidate, ranked behind every top-down and rim candidate (top-down
  stays first wherever it reaches). The jaw keeps the top-down candidate's
  horizontal closing axis; the approach leans perpendicular to it, toward the
  side away from the base. A malformed list fails the grasp.

The default profiles (`rebot_rs`, `rebot_rs_mb`, `mock`, `isaac*`) are
unchanged; `angled_approach_tilts_deg: []` in `demo.yaml` is the old planner.
The localization filter ("outside the active arm workspace") follows the
harness box, so the opt-in profiles also consider objects in the new region.

## Limits of the measurement (not claimed)

- **Kinematic and harness reachability only.** No contact physics, no gripper
  housing or finger collision with the table or the object (the harness's link
  proxies are joint origins), no grasp success rate. A reachable pose is not a
  held object.
- **The planner leans perpendicular to the jaw.** For an object whose only
  fitting jaw axis points at the robot (narrow side facing the base), the lean
  is sideways, which adds no reach: such an object beyond top-down reach is
  refused before any motion (pinned by a mock-stack test). Square objects
  always have a jaw within 45° of tangential; the `diag*` rows above are the
  45°-yawed case.
- `side` admits the outer ring (r up to ~0.75 m) and the box's outer corners;
  it is the least plausible family physically (horizontal sweep at grasp
  height, housing near the table at low heights).
- Learned candidates (GraspGen-X, HUG) are not filtered by family: in the
  opt-in profiles they gain the larger box, nothing else.
- Place targets, drop zones and `topdown_z_max` are unchanged; this is about
  picking.
- Real rig: the URDF is the model; table height, base mount and camera field
  of view on the physical setup were not measured here.

## Live validation owed

Parent (Isaac, GPU 0, bare reBot scene, PhysX): pick the pink cube placed in
the newly admitted region with `isaac_reach` (B) against the same placements
with `isaac` (A, which must refuse them: outside the workspace, or no
executable grasp where top-down cannot reach). The six placements are
pre-checked offline (`tests/test_reach_envelope.py::LIVE_POSITIONS`): (0.47, 0)
inside the shipped box but beyond top-down reach, then (0.53, 0),
(0.53, 0.25), (0.35, 0.40), (0.20, −0.42), (0.45, −0.40) in the new region;
cube axis-aligned, `place_prop` at z 0.04. Run with the analytic planner
(`--backend obb`, isolates the tilts) and once with GraspGen-X if its server is
up. The harness (`live/reach_ab.py` in the B45 item folder) was rehearsed on
the mock stack only: `mock` refused all six, `mock_reach` grasped all six with
30°/45° tilts — kinematic, not physics. What to read per run: `ok`, the
verifier's postcondition, the chosen tilt, the cube pose after the grasp, and
any harness refusal; a held cube is the claim, a reachable pose is not.

Re-run the study after any change to the URDF, `home_q`, the pregrasp offset,
the exemption radius, the joint margin or the IK margin (the evidence test
pins the URDF hash and these settings):

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python scripts/reachability_study.py \
        --out docs/evidence/reach-envelope-20261009/rebot_reachability.json
