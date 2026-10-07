# H2 (PhysX owner): standing/settle measurement, geometric skills, chat-host (MCP) command — 2026-10-07

Diagnostic, **not physical admission**. Same owner as the first episodes
(`scripts/isaac_h2_bridge.py`, Isaac Sim 6.2 alpha build, PhysX, NVIDIA `Velocity-H2-History-v0`,
GPU 0 of the x86 rig), candidate profile `h2_velocity_candidate` now extended with
`walk_distance`/`turn` (`distance_control`, `turn_control`, values derived from the first episodes —
see the YAML comment), independent `BasePostconditionChecker` with the unchanged candidate limits
(stop: 0.08 m/s, 0.20 rad/s, 0.03 m, 0.05 rad in the last 0.4 s of the settle budget).
Harness: `cascade-lab/HERMES_AUDIT_20261007/h2-owner/h2_geometric.py`; raw rows and hashes in
[manifest.json](manifest.json), aggregates in [summary.json](summary.json).

## Run geo1 — RobotRuntime → SafeBase (Python call path)

| phase | command | execution | measured | verifier |
| --- | --- | --- | --- | --- |
| A standing (no command, 3.8 s sim right after readiness) | — | — | speed p95 0.036 m/s; **ω p95 0.23, max 0.34 rad/s**, decaying 0.34 → 0.04 over 3.5 s; 9 cm of pelvis path | (no verdict: measurement) |
| B `walk_velocity` 0.3 m/s × 3 s | admitted | ok | **+0.643 m**, yaw −0.04 rad | **confirmed** (settle ω 0.117) |
| B settle, reader sampled 4.7 s sim after the skill returned | — | — | speed p95 0.008, **ω max 0.06 rad/s** from the first sample | (measurement) |
| C1 `walk_distance` +0.5 m | admitted, stopped at goal | ok | **+0.452 m** (2.28 s) | **refuted**: settle ω 0.63 rad/s, drift 0.168 rad |
| C2 `turn` +0.8 rad | admitted, stopped at goal | ok | **+0.771 rad**, path 0.185 m (< 0.35 veto) | **refuted**: settle ω 0.289, drift 0.083 rad |
| C3 `walk_distance` −0.5 m | admitted, stopped at goal | ok | **−0.453 m** (2.11 s) | **refuted**: settle ω 0.369, drift 0.111 rad |
| D `walk_velocity` via MCP | refused: `backend stop is latched` | — | — | unverified (correct: the previous runtime's close latched the stop; travel refused) |

Owner: 6469 steps, 647 frames, no fall (pelvis 0.92 m, tilt 0.005 rad at the end), ended by the harness.

## Run geo2-mcp — chat-host path only (`python -m cascade.apps.mcp_server`, `CASCADE_ROBOT=h2`, JSON-RPC over stdio)

13 tools listed (`locomotion.walk_velocity/walk_distance/turn/stop_navigation/get_base_state/…`,
`emergency_stop`, `reset_stop`). `reset_stop` before the lazy runtime build is refused
(`runtime starting or input closed; reset refused`) — fail-closed, and no latch existed on this fresh owner.

| command (MCP) | execution | measured | verifier |
| --- | --- | --- | --- |
| `walk_velocity` 0.3 × 3 s | ok | **+0.63 m** | **confirmed** (settle ω 0.088) |
| `turn` 0.6 rad | ok | **+0.574 rad**, path 0.122 m | **refuted**: settle ω 0.205 (limit 0.20), drift 0.06 rad (limit 0.05) |
| `walk_velocity` −0.3 × 2 s | ok | **−0.41 m** | **confirmed** (settle ω 0.104) |

Owner: 3515 steps, 352 frames, no fall; final state `latched: false`, `ready`.

## What the settle refutations are (measured, not interpreted away)

The verifier samples for `settle_timeout_s = 4 s` of **wall** time after the command ends and judges the
LAST 0.4 s (sim) of that window. At this owner's rate (≈0.47× real time) 4 s wall ≈ 1.9 s sim. The
verifier's own after-phase samples (geo1 `trace.jsonl`), max |ω| per 0.25 s after the command end:

- `walk_velocity` (duration-ended, confirmed): 0.0s 0.42 / 0.25s 0.33 / 0.5s 0.36 / 0.75s 0.52 / 1.0s 0.28 / 1.25s 0.15 / 1.5s 0.12 / 1.75s 0.08
- `walk_distance` +0.5 (stopped at goal, refuted): 0.0s 0.91 / 0.25s 0.62 / 0.5s 0.73 / 0.75s 0.84 / 1.0s 0.83 / 1.25s 0.74 / 1.5s 0.63 / 1.75s 0.56
- `turn` 0.8 (stopped at goal, refuted): 0.0s 0.60 / 0.25s 0.43 / 0.5s 0.34 / 0.75s 0.32 / 1.0s 0.32 / 1.25s 0.30 / 1.5s 0.29 / 1.75s 0.23
- `walk_distance` −0.5 (stopped at goal, refuted): 0.0s 1.23 / 0.25s 0.46 / 0.5s 0.42 / 0.75s 0.43 / 1.0s 0.39 / 1.25s 0.39 / 1.5s 0.37 / 1.75s 0.34

So: the H2 policy, cut to zero twist at goal arrival (mid-step), keeps a yaw oscillation of 0.3–0.9 rad/s for
more than the ~1.9 s of sim the budget covers, while a duration-ended `walk_velocity` is below 0.12 rad/s by
then. Standing right after readiness shows the same decay shape (0.34 → 0.04 rad/s over 3.5 s), and the
reader sampled right after the geo1 B skill returned (≥ 1.9 s sim after the stop) saw ω ≤ 0.06 rad/s from the
first sample: **the robot does come to rest; it needs ~2.5–3.5 s of sim to pass the candidate 0.20 rad/s, and
the settle budget is spent in wall seconds, so the verdict depends on the owner's sim rate.** Nothing in the
profile was changed to make these pass; the refutations stand. The decision whether the candidate
`settle_timeout_s`/`max_wall_duration_s` should be sized from this measurement (e.g. ≥ 3.5 s **sim**, i.e.
~8 s wall at this rate), or the geometric skills should hand over to a standing phase before the verdict, is
recorded as open in `docs/HUMANOID_H2.md`.

## Not claimed

No physical admission; no Newton run; the geometric bounds are candidates derived from one day of
episodes; no verifier threshold was relaxed; x86 numbers (sim rate ≈0.47× RT), never a Spark number.
