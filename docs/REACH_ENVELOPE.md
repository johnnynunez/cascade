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

## Gripper clearance of the tilted candidates (B73, measured 2026-10-10)

The harness vets every candidate with link proxies -- joint origins -- and
during the descent exempts every proxy inside the grasp cylinder (radius
`exempt_radius_m`, floor table_z − 0.06), which a top-down descent needs.
A leaning approach puts the housing, the wrist motors and the fingers near
the table and the object instead, where the proxies cannot see them. The
`*_reach` profiles therefore also set `grasp.angled_clearance_vet: true`
(false everywhere else; `grasping/gripper_clearance.py`):

- **Geometry:** the convex hull of every connected component of the URDF's
  collision meshes from `link3` (forearm) onward -- `link3`..`link6`, the
  housing `gripper_end` and both fingers -- built offline
  (`scripts/build_gripper_clearance_hulls.py`, SciPy) into
  `assets/grasp_geometry/rebot_rs_clearance_hulls.json`, bound to the URDF
  and STL sha256. Hulls, not raw meshes: the lowest point of a hull is one of
  its vertices and every hull vertex is a mesh vertex, so the table test is
  exact for the mesh; components, not whole links: a whole-finger hull spans
  the slide carriage behind the palm, which reaches past the jaw centreline,
  and cut the measured finger-to-cube clearance of a 5 cm cube from 20 mm
  to 2 mm. Not bounding boxes: their corners lie outside the mesh, so they
  are exact for neither test. Placed by the IK model's own FK with the
  fingers at the commanded pre-grasp opening (fully open = `max_width_m`
  0.09 m on these profiles).
- **Where:** every pose of the executed approach, pregrasp (s = 0) to grasp
  (s = 1) along the joint-space line, bisected until no hull vertex moves
  more than 1 mm between two vetted poses (57–113 poses for the 4 cm
  approach on the study grid).
- **Against:** the support plane (`safety.table_z`) -- lowest hull vertex --
  and the target's observed box (upright, yawed to the planner's jaw axis,
  enclosing the OBB footprint, down to the support when the cloud does not
  resolve the bottom) -- a separating-axis lower bound over the box's axes
  and the hulls' face normals, never optimistic. The jaw gap needs no
  exemption: no hull is there.
- **Margin:** `width_pad_m / 2` = 7.5 mm, the clearance the planner already
  gives each open jaw against the observed object; between vetted poses the
  approach keeps ≥ 7.0 mm (margin − half the 1 mm step).
- **Decision:** below the margin the candidate is refused and the next one
  tried (its flip twin is vetted on its own). Each decision, both
  clearances, the binding part and the approach fraction are a
  `gripper_clearance` grasp-evidence event; the refusal reason also reaches
  the selector's `no executable grasp` message. Unloadable or mismatched
  hulls, or a check that raises, refuse every tilted candidate with the
  reason (fail closed). Top-down and learned candidates are not vetted here.

Measured on the B45 grid (`scripts/angled_clearance_study.py` →
[`evidence/angled-clearance-20261010/rebot_angled_clearance.json`](evidence/angled-clearance-20261010/rebot_angled_clearance.json)):
every point of the envelope pass that the planner's tilts reached (roll 0,
`rebot_rs_reach`), through the real selector with the runtime's vet order
(harness, then clearance), with an upright object box under the TCP for each
grasp height (the objects B45's heights stand for: 4.3 cm lemon box,
3.75 × 3.75 × 6 cm, 5 × 5 × 8 cm, 5 × 5 × 12 cm), surviving / reachable:

| tilt | z 0.02 | z 0.05 | z 0.07 | z 0.10 | lowest clearance (part) |
| --- | --- | --- | --- | --- | --- |
| 30° (`out30`) | 655 / 655 | 654 / 654 | 648 / 648 | 634 / 634 | +19.9 mm at z 0.02 (fingers) |
| 45° (`out45`) | 649 / 649 | 662 / 662 | 675 / 675 | 692 / 692 | +19.5 mm at z 0.10 (fingers) |
| 90° (`side`) | not reachable | 0 / 47 | 81 / 81 | 189 / 189 | -3.1 mm at z 0.05 (link5) |

- **The side approach loses most, at low objects:** all 47 side grasps at
  5 cm (r 0.57–0.64 m) are refused -- the wrist motor `link5` passes
  3.1 mm *below* the table on the approach (the forearm `link3` 8 mm above
  it) while every joint origin passes the harness. At 7 cm `link5` clears
  the table by 16.9 mm; at 10 cm the binding clearance is the open fingers'
  20 mm beside the object. Side is not reachable at 2 cm at all.
- **30° and 45° lose nothing, also near the base (r from 0.125 m) and at
  2 cm:** their lowest point is always the fingertips (19.8 mm above the
  table at a 2 cm grasp: 0.2 mm under the TCP), and the closest they come to
  the object box is 19.5 mm (the open fingers beside it, (0.09 − object
  width) / 2, or the housing's front edge at 45°). Geometry says why: the
  housing's front face is 73 mm behind the fingertips and reaches 41 mm off
  the approach axis, so it drops below the tips only beyond atan(73/41) ≈ 61°
  of tilt.
- **Not tuned:** the survivors are identical for any margin from 0 to 15 mm
  (5539 of 5586); at 20 mm they fall to 1936 / 2025 / 189 (the finger-to-box
  clearance is 20 mm for 5 cm objects).
- **The envelope is unchanged:** every one of its 703 admitted cells keeps a
  surviving approach at every grasp height (the refused side points are all
  reached by a 30°/45° candidate that ranks ahead of side), so the `*_reach`
  box stays as derived. On the study grid the vet therefore changes no
  analytic pick; it removes side fallbacks that would have hit the table
  when the 30°/45° candidates fail for other reasons (occupancy, observed
  fingers, cuMotion, an obstacle), and it vets the observed box everywhere.

## Limits of the measurement (not claimed)

- **Kinematic and harness reachability only.** No contact physics and no
  grasp success rate. A reachable pose is not a held object. The B73 vet
  adds the forearm-to-fingers collision hulls against the table and the
  target's observed box on the pregrasp → grasp approach of the planner's
  tilted candidates; it does not cover the home → pregrasp transit (harness
  proxies only), other objects or obstacles, learned (GraspGen-X / HUG)
  candidates, perception error beyond the margin, or the real rig's table
  height.
- **The planner leans perpendicular to the jaw.** For an object whose only
  fitting jaw axis points at the robot (narrow side facing the base), the lean
  is sideways, which adds no reach: such an object beyond top-down reach is
  refused before any motion (pinned by a mock-stack test). Square objects
  always have a jaw within 45° of tangential; the `diag*` rows above are the
  45°-yawed case.
- `side` admits the outer ring (r up to ~0.75 m) and the box's outer corners;
  it is the least plausible family physically (horizontal sweep at grasp
  height). Its wrist at low heights is now measured and vetted (B73 above:
  refused at 5 cm, clear by 16.9 mm at 7 cm).
- Learned candidates (GraspGen-X, HUG) are not filtered by family: in the
  opt-in profiles they gain the larger box, nothing else (the B73 clearance
  vet does not apply to them).
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

B73 (the clearance vet, on in `isaac_reach`) owes the same runs on a tree
that contains it, with `CASCADE_GRASP_EVIDENCE_DIR` set: every receipt then
carries a `gripper_clearance` event per vetted tilted candidate. Expected
offline: the chosen 30°/45° candidates are admitted with ≥ 19.5 mm to the
table and the cube box, so the picks equal B45's. Read per run the event's
decision, both clearances and the binding part next to the outcome, and
whether any contact other than the finger pads reaches the cube or the table
before the close (the vet's claim). A low-prop side check (a ≤ 6 cm tall
prop at r ≈ 0.6 m where a 30°/45° candidate is refused for another reason)
is not reachable with the analytic ranking on the bare scene; the vet's side
refusal is pinned offline only.

Re-run the study after any change to the URDF, `home_q`, the pregrasp offset,
the exemption radius, the joint margin or the IK margin (the evidence test
pins the URDF hash and these settings):

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python scripts/reachability_study.py \
        --out docs/evidence/reach-envelope-20261009/rebot_reachability.json

and then the clearance hulls (after a URDF/mesh change; SciPy) and the B73
measurement (its evidence test pins the URDF, hulls and B45 evidence hashes):

    PYTHONPATH=src python scripts/build_gripper_clearance_hulls.py
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1 python scripts/angled_clearance_study.py \
        --out docs/evidence/angled-clearance-20261010/rebot_angled_clearance.json
