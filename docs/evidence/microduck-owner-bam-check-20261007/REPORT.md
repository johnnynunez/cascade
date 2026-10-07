# Shared owner per-step cost: one device-side finiteness check for the BAM outputs (2026-10-07)

Same-day A/B on the x86 test rig (Isaac Sim 6.2 alpha, Newton 1.6.1rc1, GPU 1), twelve
MicroDucks in ONE world, `rough_walk_e`, Isaac Lab USD + complete hulls, route profile,
twelve closed-loop `walk_distance 5 m` clients, one walk per robot
(`HERMES_AUDIT_20261007/harness/route_walk.py`, the route harness pointed at the worktree
under test), owner with `--profile-phases`. Arm M is `origin/main` 30a61fd (the tree after
the 7 October merge train, i.e. host snapshot + reader rebind already in); arm E is M plus
this change. Nothing else differs (same scene model identity, same clients, same card).

| metric | M = main 30a61fd | E = M + this change |
| --- | --- | --- |
| attempts (solved steps) | 2943 | 3175 |
| median ms / step | 48.82 | 46.78 |
| p95 ms / step | 126.0 | 123.9 |
| mean ms / step | 62.59 | 58.99 |
| phase bam.before_step | 13.91 | 11.79 |
| phase bam.verify (new) | — | 0.08 |
| bam.before_step + bam.verify | 13.91 | 11.87 |
| phase completed.validate | 23.6 | 23.77 |
| phase policy.prepare | 4.34 | 4.19 |
| phase publication | 2.91 | 2.87 |
| phase solve | 0.25 | 0.24 |
| per-robot bam.before_step | 1.004 | 0.849 |
| sparse record.physics (median ms, count) | (27.7, 295) | (26.5, 319) |
| sparse camera.overview (median ms, count) | (84.1, 148) | (83.6, 159) |

- **What changed (E):** the twelve `NewtonBamAdapter`s no longer `.numpy()` their five
  output arrays each (`external_torque` before `compute`; `_forces`, `motor_torque`,
  `effective_vin`, `friction_budget` after) — 60 full device syncs per 5 ms step. Each
  adapter launches two Warp kernels that mark a per-(adapter, array) slot of one cohort
  flag array (`BamOutputCheck`, `control/newton_bam.py`) when any element is non-finite;
  the shared tick does ONE `.numpy()` of that array (`bam.verify`) immediately before
  `owner.step()` and refuses the step naming the robot and array. Every check still runs
  for every adapter; a check of another stage/step or an unregistered adapter is refused;
  the single-robot path is unchanged (CPU tests on the real pinned BAM kernels:
  `tests/test_newton_bam.py` 124 passed, plus the shared-scene/stepper suites).
- **Measured effect:** `bam.before_step` 13.91 → 11.79 ms, plus `bam.verify`
  0.08 ms: **13.91 → 11.87 ms per step** (per robot
  1.004 → 0.849); step median 48.82 → 46.78 ms (−4 %),
  p95 126.0 → 123.9. The other phases are unchanged within noise. The sixty
  syncs were therefore worth about 2 ms of the 13.9; what remains in `bam.before_step`
  (~11.8 ms, ~1 ms per robot) is the Warp launch overhead (`gather_external_torque`,
  `compute`, `publish_dof_friction`, scatter, `update_state`, now the two flag kernels)
  and the host-side binding/model-sync checks on the snapshot copies.
- **Not changed, not admission:** all 12 client walks remain `unverified` in M and
  `unverified` in E (stale feedback / RPC wall-time deadline against the unchanged 0.5 s
  state-age, 0.4 s progress and 0.5 s RPC limits); median measured distance per walk
  0.435 m (M) / 0.494 m (E) of the 5 m asked. No limit, physics,
  asset or policy changed; the GPU physics step is still ~7.6 ms per 5 ms step. Both
  owners ended by the driver's lifecycle SIGTERM after the clients finished
  (`end_reason: signal/lifecycle shutdown`, exit 143), as in the earlier arms.
- **Next (unchanged list, re-ranked by this measurement):** the deep copies and per-robot
  validation inside `completed.validate` (23.6 ms, of which ~7.6 ms is the GPU wait),
  the per-adapter Warp launch count in `bam.before_step` (batch the twelve adapters'
  launches into cohort kernels), the sparse `record.physics` / `camera.overview` writes
  behind the p95 tail, RPC serving off the owner's GIL, and the physics budget per asset.

Raw rows (owner `timing.jsonl`, `receipt.json`, driver logs) stay in
`cascade-lab/HERMES_AUDIT_20261007/harness/` — paths, sizes and SHA-256 in
[manifest.json](manifest.json); the per-arm aggregates are
[timing-summary-M-main.json](timing-summary-M-main.json) and
[timing-summary-E-bam-check.json](timing-summary-E-bam-check.json)
(`harness/timing_summary.py`).
