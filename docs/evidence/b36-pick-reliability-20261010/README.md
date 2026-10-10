# B36 — pick reliability on the bare Isaac reBot scene (live, 8–10 Oct 2026)

**Item.** B36: fresh-stage picks of the pink cube on the bare Isaac reBot scene failed with three signatures
independent of the caller (sandboxed agent, MCP host probe, launcher turn):

- **S1** — attempt 1 ends `did not settle at grasp lift pose` with the cube **lifted in the jaws**; every retry then
  refuses `already holding 'pink cube'` until the budget is spent; the cube ends held at the home pose (~34 cm from
  the target) and the scene reset fails `did not settle at home`.
- **S2** — `did not settle above the place target`: the cube is carried to ~3–4 cm from the drop zone, still held.
- **S3** — a drop in the carry after a verified grip: the skill reports ok, the physics postcondition refutes it
  (the cube lies ~12 cm short of the drop zone, typically at about [0.20, -0.05]).

**Shipped (PR, merged tree on 38f6d08).**

1. Held-object recovery (S1 is a logic bug: a held object must never count as a failed attempt). After a failed
   attempt the persistence loops (`pick_and_place`, `_grasp_with_persistence` behind handover / sort_by_color /
   rearrange) reconcile the held flag with the jaws; the requested object in the jaws is the grasp
   (`grasp_recovered_after`, `grip_verified: null`), another object stops the loop.
2. The grasp's bounded post-contact hold on PhysX (`gripper.hold_squeeze_frac: {physx: 0.05}` in `isaac.yaml`,
   inherited by `isaac_cumotion` / `isaac_reach`, `null` in the kitchen profiles), with the post-lift air-grasp check
   judging the hold opening.

## Method

- Bare reBot USD (`assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda`), Isaac Sim source build, GPU 0 only.
  **A fresh Isaac stage and a fresh MCP server per run**; arms interleaved; the bridge of a series always ran from that
  series' main tree (only the server code differed).
- Same task every run: `get_observation` → `pick_and_place {object: pink cube, destination: drop zone, material:
  rigid}` → `get_observation` → `reset_scene`. 8–9 Oct: HTTPS MCP server + host probe (`probe_host.py`);
  10 Oct (W8): stdio MCP server spawned by `w8/probe.py`, same calls.
- Grasp planner: learned GraspGen-X (warm sidecar) unless noted; A/B4 and smoke4 ran with the analytic OBB fallback
  because the GraspGen-X server was down (server log: `grasp_planner=obb (graspgenx down)`).
- Every run's `trace.jsonl` (pick result + physics postcondition) and first grasp-evidence receipt (`close_hold`,
  `grip_verification`, lift wrist roll, attempt exceptions) were read by `build_evidence.py` → `runs.json`
  (sha256 of every input in `manifest.json`); `render_tables.py` renders the tables below from `runs.json`.
- Classification: `confirmed` = physics postcondition confirmed; S1 = stage grasp + "already holding"; S2 = stage
  place + "did not settle above the place target"; S3 = grip verified, self-reported ok, physics refuted, cube moved
  ≥ 5 cm.

| series (runner) | date | engine | compared |
| --- | --- | --- | --- |
| A/B1 (`ab_runs.sh`) | 8 Oct | PhysX | main 133876c vs recovery + `contact_lift: pregrasp` vs recovery + `contact_lift: preserve_rotation`, 4 each |
| E1 (`e1_runs.sh`) | 8 Oct | PhysX | main 133876c, `material` fragile (stage-2 0.60) vs rigid (0.85), 3 each |
| A/B2 (`ab2_runs.sh`) | 8 Oct | PhysX | main 133876c vs recovery + hold 0.05 (all engines), 6 each |
| A/B3 (`ab3_newton.sh`) | 8 Oct | Newton | the same two, 4 each (complete: nwhold4 finished 20:38; only its bridge log tail was lost at the 22:40 reboot) |
| A/B4 (`ab4.sh`) | 9 Oct | PhysX | main 4e896c3 vs branch 45c8945 — GraspGen-X down (OBB fallback); stopped after 3 + 3 |
| A/B5 (`ab5.sh`) | — | — | never ran |
| A/B6 (`ab6.sh`) | 9 Oct | PhysX 6 + 6, Newton 4 + 4 | main 4e896c3 vs branch 45c8945 (hold `{physx: 0.05}`: none on Newton) |
| W8 (`w8/series.sh`) | 10 Oct | PhysX 4 + 4, Newton 2 | main 38f6d08 vs the merged branch head (this PR's code) |

## Results (rendered from `runs.json`)

### Per arm

| series | engine | planner | arm (code) | confirmed | failures by signature |
| --- | --- | --- | --- | --- | --- |
| A/B1 | physx | graspgenx | main (133876c) | 2/4 | S1 1, S3 1 |
| A/B1 | physx | graspgenx | recovery+contact_lift=pregrasp (wip-8oct) | 2/4 | S2 1, S3 1 |
| A/B1 | physx | graspgenx | recovery+contact_lift=preserve_rotation (wip-8oct) | 2/4 | S2 1, S3 1 |
| E1 | physx | graspgenx | main material=fragile (133876c) | 2/3 | S3 1 |
| E1 | physx | graspgenx | main material=rigid (133876c) | 0/3 | S1 1, S2 1, S3 1 |
| A/B2 | physx | graspgenx | main (133876c) | 2/6 | S1 1, S3 3 |
| A/B2 | physx | graspgenx | recovery+hold(all engines) (wip-8oct) | 6/6 | -- |
| A/B3 | newton | graspgenx | main (133876c) | 4/4 | -- |
| A/B3 | newton | graspgenx | recovery+hold(all engines) (wip-8oct) | 1/4 | S3 3 |
| A/B4 | physx | obb | branch (45c8945) | 0/3 | S3 3 |
| A/B4 | physx | obb | main (4e896c3) | 0/3 | S1 3 |
| A/B6 | newton | graspgenx | branch (45c8945) | 4/4 | -- |
| A/B6 | newton | graspgenx | main (4e896c3) | 4/4 | -- |
| A/B6 | physx | graspgenx | branch (45c8945) | 5/6 | S2 1 |
| A/B6 | physx | graspgenx | main (4e896c3) | 3/6 | S1 1, S3 2 |
| smoke | physx | obb | recovery+hold (wip-8oct) | 0/1 | S3 1 |
| W8 | newton | graspgenx | branch (ce0d1f8) | 2/2 | -- |
| W8 | physx | graspgenx | branch (ce0d1f8) | 4/4 | -- |
| W8 | physx | graspgenx | main (38f6d08) | 4/4 | -- |
| W8-smoke | physx | graspgenx | branch (ce0d1f8) | 1/1 | -- |

### Lift wrist-roll error (first lift of each run, |q6 - target|, settle_tol 0.045)

| engine | close | lifts | median rad | max rad | >= 0.045 |
| --- | --- | --- | --- | --- | --- |
| newton | hold applied | 2 | 0.0000 | 0.0000 | 0 |
| newton | stage-2 0.85, no hold | 16 | 0.0000 | 0.0000 | 0 |
| physx | fragile 0.60 | 3 | 0.0169 | 0.0258 | 0 |
| physx | hold applied | 20 | 0.0109 | 0.0147 | 0 |
| physx | stage-2 0.85, no hold | 35 | 0.0449 | 0.0500 | 9 |

### Every run

| run | series | engine | arm | planner | outcome | grip verified | hold | contact -> hold open | width after lift | lift q6 err | recovered | final xyz (m) | err to target (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main1 | A/B1 | physx | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044935 | -- | [0.182, -0.146, 0.047] | 0.024 |
| main2 | A/B1 | physx | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044899 | -- | [0.170, -0.151, 0.047] | 0.021 |
| main3 | A/B1 | physx | main (133876c) | graspgenx | S1 | None | -- | -- | -- | 0.048949 | -- | [0.474, 0.002, 0.431] | 0.341 |
| main4 | A/B1 | physx | main (133876c) | graspgenx | S3 | True | -- | -- | 0.480 | 0.042841 | -- | [0.199, -0.060, 0.040] | 0.111 |
| rec1 | A/B1 | physx | recovery+contact_lift=pregrasp (wip-8oct) | graspgenx | S3 | True | -- | -- | 0.471 | 0.042296 | -- | [0.218, -0.046, 0.040] | 0.130 |
| rec2 | A/B1 | physx | recovery+contact_lift=pregrasp (wip-8oct) | graspgenx | S2 | True | -- | -- | 0.492 | 0.044311 | -- | [0.162, -0.197, 0.136] | 0.033 |
| rec3 | A/B1 | physx | recovery+contact_lift=pregrasp (wip-8oct) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044908 | -- | [0.172, -0.170, 0.040] | 0.008 |
| rec4 | A/B1 | physx | recovery+contact_lift=pregrasp (wip-8oct) | graspgenx | confirmed | True | -- | -- | 0.498 | 0.044998 | -- | [0.171, -0.168, 0.040] | 0.009 |
| b361 | A/B1 | physx | recovery+contact_lift=preserve_rotation (wip-8oct) | graspgenx | S2 | None | -- | -- | -- | 0.048923 | yes | [0.153, -0.198, 0.164] | 0.039 |
| b362 | A/B1 | physx | recovery+contact_lift=preserve_rotation (wip-8oct) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044900 | -- | [0.167, -0.166, 0.040] | 0.013 |
| b363 | A/B1 | physx | recovery+contact_lift=preserve_rotation (wip-8oct) | graspgenx | confirmed | True | -- | -- | 0.497 | 0.044967 | -- | [0.196, -0.153, 0.041] | 0.023 |
| b364 | A/B1 | physx | recovery+contact_lift=preserve_rotation (wip-8oct) | graspgenx | S3 | True | -- | -- | 0.481 | 0.043047 | -- | [0.205, -0.051, 0.040] | 0.121 |
| e1fragile1 | E1 | physx | main material=fragile (133876c) | graspgenx | confirmed | True | -- | -- | 0.535 | 0.016947 | -- | [0.189, -0.153, 0.046] | 0.019 |
| e1fragile2 | E1 | physx | main material=fragile (133876c) | graspgenx | confirmed | True | -- | -- | 0.524 | 0.015636 | -- | [0.208, -0.149, 0.040] | 0.036 |
| e1fragile3 | E1 | physx | main material=fragile (133876c) | graspgenx | S3 | True | -- | -- | 0.623 | 0.025812 | -- | [0.239, -0.047, 0.040] | 0.136 |
| e1rigid1 | E1 | physx | main material=rigid (133876c) | graspgenx | S1 | None | -- | -- | -- | 0.049002 | -- | [0.474, 0.002, 0.431] | 0.341 |
| e1rigid2 | E1 | physx | main material=rigid (133876c) | graspgenx | S2 | True | -- | -- | 0.492 | 0.044301 | -- | [0.167, -0.195, 0.139] | 0.028 |
| e1rigid3 | E1 | physx | main material=rigid (133876c) | graspgenx | S3 | True | -- | -- | 0.493 | 0.044696 | -- | [0.200, -0.047, 0.040] | 0.124 |
| ab2main1 | A/B2 | physx | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.492 | 0.044312 | -- | [0.207, -0.161, 0.040] | 0.028 |
| ab2main2 | A/B2 | physx | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044912 | -- | [0.183, -0.150, 0.047] | 0.020 |
| ab2main3 | A/B2 | physx | main (133876c) | graspgenx | S3 | True | -- | -- | 0.497 | 0.044943 | -- | [0.205, -0.048, 0.040] | 0.124 |
| ab2main4 | A/B2 | physx | main (133876c) | graspgenx | S1 | None | -- | -- | -- | 0.047034 | -- | [0.485, 0.002, 0.433] | 0.350 |
| ab2main5 | A/B2 | physx | main (133876c) | graspgenx | S3 | True | -- | -- | 0.489 | 0.043263 | -- | [0.188, -0.046, 0.040] | 0.124 |
| ab2main6 | A/B2 | physx | main (133876c) | graspgenx | S3 | True | -- | -- | 0.492 | 0.044317 | -- | [0.299, -0.055, 0.040] | 0.165 |
| ab2hold1 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.529 | 0.010431 | -- | [0.178, -0.173, 0.040] | 0.003 |
| ab2hold2 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.492 -> 0.442 | 0.529 | 0.010954 | -- | [0.208, -0.148, 0.040] | 0.035 |
| ab2hold3 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.548 -> 0.498 | 0.600 | 0.012138 | -- | [0.178, -0.174, 0.040] | 0.004 |
| ab2hold4 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.549 -> 0.499 | 0.600 | 0.012027 | -- | [0.183, -0.150, 0.047] | 0.020 |
| ab2hold5 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.530 | 0.010610 | -- | [0.179, -0.172, 0.040] | 0.002 |
| ab2hold6 | A/B2 | physx | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | applied | 0.510 -> 0.460 | 0.540 | 0.010014 | -- | [0.183, -0.149, 0.047] | 0.021 |
| nwmain1 | A/B3 | newton | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000000 | -- | [0.187, -0.154, 0.025] | 0.017 |
| nwmain2 | A/B3 | newton | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000000 | -- | [0.187, -0.154, 0.025] | 0.017 |
| nwmain3 | A/B3 | newton | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.539 | 0.000001 | -- | [0.200, -0.154, 0.033] | 0.025 |
| nwmain4 | A/B3 | newton | main (133876c) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000000 | -- | [0.186, -0.155, 0.025] | 0.016 |
| nwhold1 | A/B3 | newton | recovery+hold(all engines) (wip-8oct) | graspgenx | S3 | True | applied | 0.514 -> 0.464 | 0.517 | 0.000001 | -- | [0.222, 0.100, 0.025] | 0.274 |
| nwhold2 | A/B3 | newton | recovery+hold(all engines) (wip-8oct) | graspgenx | confirmed | True | not applied: no jaw stall observed | -- | 0.521 | 0.000000 | -- | [0.196, -0.149, 0.025] | 0.026 |
| nwhold3 | A/B3 | newton | recovery+hold(all engines) (wip-8oct) | graspgenx | S3 | True | applied | 0.514 -> 0.464 | 0.517 | 0.000001 | -- | [0.226, 0.092, 0.025] | 0.266 |
| nwhold4 | A/B3 | newton | recovery+hold(all engines) (wip-8oct) | graspgenx | S3 | True | not applied: no jaw stall observed | -- | 0.549 | 0.000000 | -- | [0.209, -0.004, 0.025] | 0.169 |
| ab4pf1 | A/B4 | physx | branch (45c8945) | obb | S3 | True | applied | 0.576 -> 0.526 | 0.657 | 0.014672 | -- | [0.194, -0.046, 0.025] | 0.125 |
| ab4pf2 | A/B4 | physx | branch (45c8945) | obb | S3 | True | applied | 0.585 -> 0.535 | 0.658 | 0.013823 | -- | [0.247, -0.050, 0.040] | 0.138 |
| ab4pf3 | A/B4 | physx | branch (45c8945) | obb | S3 | True | applied | 0.593 -> 0.543 | 0.660 | 0.013094 | -- | [0.244, -0.052, 0.040] | 0.134 |
| ab4pm1 | A/B4 | physx | main (4e896c3) | obb | S1 | None | -- | -- | -- | 0.048241 | -- | [0.485, 0.002, 0.443] | 0.350 |
| ab4pm2 | A/B4 | physx | main (4e896c3) | obb | S1 | None | -- | -- | -- | 0.049415 | -- | [0.484, 0.004, 0.443] | 0.350 |
| ab4pm3 | A/B4 | physx | main (4e896c3) | obb | S1 | None | -- | -- | -- | 0.049951 | -- | [0.483, 0.004, 0.443] | 0.350 |
| ab6nf1 | A/B6 | newton | branch (45c8945) | graspgenx | confirmed | True | -- | -- | 0.539 | 0.000001 | -- | [0.192, -0.154, 0.029] | 0.020 |
| ab6nf2 | A/B6 | newton | branch (45c8945) | graspgenx | confirmed | True | -- | -- | 0.539 | 0.000001 | -- | [0.196, -0.155, 0.032] | 0.022 |
| ab6nf3 | A/B6 | newton | branch (45c8945) | graspgenx | confirmed | True | -- | -- | 0.511 | 0.000007 | -- | [0.192, -0.155, 0.025] | 0.020 |
| ab6nf4 | A/B6 | newton | branch (45c8945) | graspgenx | confirmed | True | -- | -- | 0.511 | 0.000002 | -- | [0.196, -0.140, 0.040] | 0.034 |
| ab6nm1 | A/B6 | newton | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.517 | 0.000000 | -- | [0.186, -0.154, 0.025] | 0.017 |
| ab6nm2 | A/B6 | newton | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000000 | -- | [0.188, -0.154, 0.025] | 0.018 |
| ab6nm3 | A/B6 | newton | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.507 | 0.000001 | -- | [0.201, -0.159, 0.025] | 0.024 |
| ab6nm4 | A/B6 | newton | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.539 | 0.000001 | -- | [0.194, -0.147, 0.040] | 0.026 |
| ab6pf1 | A/B6 | physx | branch (45c8945) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.530 | 0.010359 | -- | [0.181, -0.153, 0.047] | 0.017 |
| ab6pf2 | A/B6 | physx | branch (45c8945) | graspgenx | confirmed | True | applied | 0.541 -> 0.491 | 0.588 | 0.011651 | -- | [0.182, -0.150, 0.047] | 0.020 |
| ab6pf3 | A/B6 | physx | branch (45c8945) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.531 | 0.010575 | -- | [0.181, -0.148, 0.047] | 0.022 |
| ab6pf4 | A/B6 | physx | branch (45c8945) | graspgenx | S2 | None | not applied: no jaw stall observed | -- | -- | 0.048920 | yes | [0.153, -0.197, 0.166] | 0.038 |
| ab6pf5 | A/B6 | physx | branch (45c8945) | graspgenx | confirmed | True | applied | 0.491 -> 0.441 | 0.529 | 0.010962 | -- | [0.207, -0.148, 0.040] | 0.035 |
| ab6pf6 | A/B6 | physx | branch (45c8945) | graspgenx | confirmed | True | applied | 0.543 -> 0.493 | 0.596 | 0.012179 | -- | [0.183, -0.148, 0.047] | 0.022 |
| ab6pm1 | A/B6 | physx | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044909 | -- | [0.185, -0.148, 0.025] | 0.022 |
| ab6pm2 | A/B6 | physx | main (4e896c3) | graspgenx | S3 | True | -- | -- | 0.497 | 0.044467 | -- | [0.207, -0.051, 0.040] | 0.122 |
| ab6pm3 | A/B6 | physx | main (4e896c3) | graspgenx | S1 | None | -- | -- | -- | 0.049218 | -- | [0.475, 0.003, 0.431] | 0.342 |
| ab6pm4 | A/B6 | physx | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.480 | 0.043496 | -- | [0.202, -0.155, 0.047] | 0.027 |
| ab6pm5 | A/B6 | physx | main (4e896c3) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044918 | -- | [0.169, -0.148, 0.047] | 0.024 |
| ab6pm6 | A/B6 | physx | main (4e896c3) | graspgenx | S3 | True | -- | -- | 0.492 | 0.044308 | -- | [0.238, -0.051, 0.025] | 0.133 |
| smoke4 | smoke | physx | recovery+hold (wip-8oct) | obb | S3 | True | applied | 0.581 -> 0.531 | 0.655 | 0.013976 | -- | [0.247, -0.048, 0.040] | 0.139 |
| nh1 | W8 | newton | branch (ce0d1f8) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000001 | -- | [0.187, -0.154, 0.025] | 0.017 |
| nh2 | W8 | newton | branch (ce0d1f8) | graspgenx | confirmed | True | -- | -- | 0.516 | 0.000000 | -- | [0.191, -0.158, 0.025] | 0.016 |
| ph1 | W8 | physx | branch (ce0d1f8) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.530 | 0.010588 | -- | [0.180, -0.149, 0.047] | 0.021 |
| ph2 | W8 | physx | branch (ce0d1f8) | graspgenx | confirmed | True | applied | 0.497 -> 0.447 | 0.533 | 0.010847 | -- | [0.181, -0.148, 0.047] | 0.022 |
| ph3 | W8 | physx | branch (ce0d1f8) | graspgenx | confirmed | True | applied | 0.495 -> 0.445 | 0.529 | 0.010579 | -- | [0.181, -0.148, 0.047] | 0.022 |
| ph4 | W8 | physx | branch (ce0d1f8) | graspgenx | confirmed | True | applied | 0.497 -> 0.447 | 0.530 | 0.010394 | -- | [0.181, -0.148, 0.047] | 0.022 |
| pm1 | W8 | physx | main (38f6d08) | graspgenx | confirmed | True | -- | -- | 0.496 | 0.044895 | -- | [0.177, -0.170, 0.040] | 0.003 |
| pm2 | W8 | physx | main (38f6d08) | graspgenx | confirmed | True | -- | -- | 0.492 | 0.044316 | -- | [0.180, -0.148, 0.040] | 0.021 |
| pm3 | W8 | physx | main (38f6d08) | graspgenx | confirmed | True | -- | -- | 0.497 | 0.044972 | -- | [0.182, -0.145, 0.047] | 0.025 |
| pm4 | W8 | physx | main (38f6d08) | graspgenx | confirmed | True | -- | -- | 0.498 | 0.044995 | -- | [0.175, -0.154, 0.040] | 0.017 |
| smoke2 | W8-smoke | physx | branch (ce0d1f8) | graspgenx | confirmed | True | applied | 0.496 -> 0.446 | 0.528 | 0.010373 | -- | [0.181, -0.149, 0.047] | 0.021 |

## Reading

- **S1 is gone wherever the recovery ran**: 0 of 39 recovery-arm runs, against every main series (A/B1 1/4,
  E1 1/6, A/B2 1/6, A/B4 3/3, A/B6 1/6, W8 0/4). When the recovery triggered live (b361, ab6pf4) the cube
  was carried to the drop zone and the place then failed S2: still a failed pick, but without four wasted retries
  and with an honest "still holding" state.
- **The hold, PhysX, GraspGen-X grasps:** main 5/12 vs branch 11/12 confirmed over A/B2 + A/B6 (main S1 2, S3 5;
  branch S2 1). The hold applied in 11 of those 12 branch runs and all 11 were confirmed; in the 12th (ab6pf4) the
  jaws chattered ±0.003 around 0.544, wider than the stall detector's 0.002 band, no hold applied, the lift did not
  settle, and the recovered cube failed S2. On the merged head (W8, 10 Oct: main 38f6d08 vs this branch, 4 + 4 interleaved) both arms
  confirmed 4/4: main had no failure that day (its four lifts sat at 0.0443-0.0450 rad, a hair under the
  tolerance), the branch's hold applied 4/4 (0.0104-0.0108 rad). Pooled over A/B2, A/B6 and W8: **main 9/16 vs
  branch 15/16** (two-sided Fisher exact p = 0.037; A/B2 + A/B6 alone 5/12 vs 11/12, p = 0.027). With the hold
  applied, 16 of 16 PhysX GraspGen-X picks were confirmed (smoke2 included).
- **Mechanism.** With the full stage-2 squeeze, PhysX left wrist roll a median 0.045 rad from its lift target —
  right on settle_tol 0.045. 9 of 35 first lifts were at or over it, and those 9 are exactly the 7 S1 runs plus the 2
  recoveries (b361, ab6pf4): every lift over the tolerance failed to settle, every lift under it settled at the lift pose. With the
  hold the median was 0.011 rad (max 0.015, 20 lifts); the fragile 0.60 close 0.017 (3). Newton shows no deflection
  (0.000 rad, 18 lifts), which is why it needs no hold. Drops in carry (S3) follow the same split on PhysX with
  GraspGen-X grasps: 9 of 32 rigid runs without a hold applied (1 of 3 fragile), 0 of 16 with it (with OBB grasps
  4 of 4 with it, see below).
- **Newton:** main 8/8 and the branch without a hold 4/4, + 2/2 on the merged head (W8: no hold commanded, as
  configured) = 14/14; the hold arm 1/4 (the hold applied in nwhold1/3, both dropped the cube 7–8 cm from its start;
  in nwhold2/4 no stall was seen, nwhold2 confirmed, nwhold4 dropped). Hence `{physx: 0.05}`, no Newton entry.
- **OBB fallback (A/B4 + smoke4):** 0/3 main (S1 ×3) vs 0/3 branch (S3 ×3) + smoke4 S3: with analytic grasps the
  hold let the lift settle but the cube dropped in the carry. No claim for the fallback.
- **Air-grasp check after the hold:** all 22 recorded live grasps with the hold applied still verify when judged
  against the hold opening, as this PR does (width after lift − (hold opening + 0.03): min 0.050 on PhysX, 0.023 on
  Newton); an object lost in the lift leaves the jaws at the hold opening and fails the check.
- **Not shipped:** `contact_lift` (A/B1: 2/4 for both variants, same as main). A wider stall band (it would have
  caught ab6pf4) is unmeasured — follow-up.

## Not claimed

- Small samples: 16 + 16 PhysX GraspGen-X picks over three days and three mains (133876c, 4e896c3, 38f6d08), one
  scene, one object (pink cube), one destination; the W8 head check alone (4/4 vs 4/4) does not discriminate. No
  kitchen, no real reBot, no OBB-fallback benefit.
- W8 ran the stdio MCP server spawned by `w8/probe.py`, the 8-9 Oct series the HTTPS server + host probe; the
  calls, scene, task and bridge were the same.
- The recovery converts S1 into a placed cube only when the place then settles; both live triggers ended S2.
- The hold is a position-target workaround for a bridge without a force bound, not a force controller.

## Reproduce

- `build_evidence.py <item>/live <out>` then `render_tables.py <out>/runs.json` (inputs: the run directories under
  `cascade-lab/HERMES_BACKLOG_20261007/b36-pick-reliability/live/`, sha256 in `manifest.json`).
- Runners as they ran: `runners/` (8–9 Oct: `ab_runs.sh`, `e1_runs.sh`, `ab2_runs.sh`, `ab3_newton.sh`, `ab4.sh`,
  `ab6.sh`, `fresh_rig.sh`, `fresh_rig2.sh`, `serve.sh`, `serve2.sh`, `probe_host.py`; 10 Oct: `w8/`).
