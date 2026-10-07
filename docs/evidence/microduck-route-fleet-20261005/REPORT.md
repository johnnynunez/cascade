# MicroDuck long routes: one shared Isaac 6.2/Newton world vs one world per robot (2026-10-05)

> Evidence copy (CASCADE `docs/evidence/microduck-route-fleet-20261005/`) of the harness in
> `cascade-lab/HERMES_MICRODUCK_ROUTE_20261005/` (`shared_lib.py`, `route_walk.py`, `fleet_walk.py` +
> `fleet_one.py`, `compose_route.py`, `compose_fleet.py`, copied here). Per-run folders hold the
> experiment receipts, walk summaries, contact sheets and (for g2d) the owner timing summary; raw
> physics/timing rows and videos stay in the lab directory. Source: origin/main 90f9d65 + PR #219
> (hulls) + PR #220 (long walk) + the route-fleet commits (PR #222). Asset: Isaac Lab USD bundle
`isaaclab-usd-allcollisions-v1` (receipt ba7cdd9e…), complete collision hulls, `rough_walk_e`
(5aa423bd…), BAM `official_infer_nominal_no_delay`, heading hold kp 4 / ki 2, SDK recipe
`isaac62_48b2d951`, GPU 1 (`GPU-4c811d02…`), plus GPU 0 for half of the 12-world fleet.
Diagnostic evidence; not physical admission. No verifier deadline was relaxed.

| run | setup | result |
|---|---|---|
| `g1-rwe-1duck-5m` | 1 robot, shared owner, line layout, 1080p overview | **CONFIRMED**: body 4.996 m, net world 4.780 m, lateral +0.06 m, yaw +0.10 rad, 39.5 s sim, 97.7 s wall, no fall, settled 4e-5 m. Owner 10.6 ms / 5 ms step. Video `g1-rwe-1duck-5m.mp4`. |
| `g2a-FAILED-poll10ms` | 12 robots, one world, client polls 10/20 ms | all 12 fail ≤13 s wall (clock did not advance / stale): 12×2 clients at 100 Hz starve the owner. |
| `g2b-FAILED-poll100ms` | same, client polls 0.1 s | all 12 fail at ~16 s: capture attempts 400–460 ms (13 JPEG encodes per capture + physics row on the same attempt). |
| `g2c-profile-12ducks` | same + `--profile-phases` | attempt median 81 ms, p95 316, max 463, 8 >400 ms; camera.overview 253 ms of which capture 67 ms. |
| `g2d-rwe-12ducks-5m` | single encode + offset rows, 12 concurrent walks | owner max attempt 227 ms, **0 >400 ms** in 3,960; walks still fail at 0.75–0.86 m on stale feedback / RPC deadline (RPC path, not the owner loop). |
| `g2e-…-3waves` | same, 4 robots walking at a time | wave 1 fails at 2.4–2.7 m, same errors; wave 2 partly at admission. Terminated. |
| `f1-rwe-12worlds` | 12 worlds, ONE client process | 2/12 confirmed: the single driver's GIL (12 × 150 polls/s) and `--max-steps` reached mid-walk. |
| `f2-rwe-12worlds` | 12 worlds, one client process each, routes 5/4.5/5.5/4/5/5.5/3.5/5/4.5/5.5/5/4 m | **10/12 CONFIRMED** (body 3.995–5.497 m, net 3.82–5.26 m, lateral 0.05–0.07 m, yaw ≤0.09 rad, 28–44 s sim, no fall). duck06 (3.5 m) and duck11 (4.0 m) REFUTED after completing: did not settle (0.11 / 0.48 rad/s, 0.10 m/s residual in the window). Video `f2-rwe-12worlds.mp4`. |

Verifier `body_displacement_m` integrates body-frame increments (gait sway counts); net
world displacement is reported separately (≈95.6 % of the body figure here).

Twelve robots in ONE world remain blocked by the owner's per-step Python cost (≈70–100 ms wall
per 5 ms physics step with twelve robots; solve 0.3 ms) and RPC/GIL contention with concurrent
clients, under the unchanged 0.5 s state-age / RPC and 0.4 s progress limits. Candidate fixes:
batch validation/BAM across robots, serve RPCs outside the owner's GIL.

Not done: the Qwen agent layer on this fleet (the bounded receipt view it needs is PR #221).
