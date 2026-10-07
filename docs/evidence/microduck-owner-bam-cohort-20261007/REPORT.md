# Shared owner per-step cost: cohort Warp launches for the twelve BAM adapters (2026-10-07)

Same-session A/B on the x86 test rig (Isaac Sim 6.2 alpha, Newton 1.6.1rc1, GPU 1 shared
with the idle `cascade-qwen` llama-server), twelve MicroDucks in ONE world, `rough_walk_e`,
Isaac Lab USD + complete hulls, route profile, twelve closed-loop `walk_distance 5 m`
clients, one walk per robot (`HERMES_AUDIT_20261007/harness/route_walk.py` pointed at the
worktree under test), owner with `--profile-phases`, arms run back to back.
Arm F is `b7536cd` (main after #238's output check, i.e. slice 1); arm G is `8703a39`, the
head of #243, whose only parent is `b7536cd`. Nothing else differs (same scene model
identity, clients, card and harness). The G receipt records
`bam_cohort: {"adapters": 12, "path": "cohort", "reason": null}`.

| metric | F = b7536cd (per-adapter) | G = 8703a39 (cohort) |
| --- | --- | --- |
| attempts (solved steps) | 4130 | 3200 |
| median ms / step | 46.79 | **42.43** |
| p95 ms / step | 124.7 | 119.7 |
| mean ms / step | 59.90 | 54.72 |
| phase `bam.before_step` | 12.11 | **6.57** |
| phase `bam.verify` | 0.08 | 0.08 |
| phase `completed.validate` | 23.87 | 24.54 |
| phase `policy.prepare` | 4.25 | 4.41 |
| phase `publication` | 2.92 | 3.10 |
| phase `solve` | 0.25 | 0.24 |
| nested `support.decode` | 11.67 | 11.85 |
| sparse `record.physics` (median ms, count) | (27.5, 414) | (27.9, 321) |
| sparse `camera.overview` (median ms, count) | (84.3, 207) | (83.9, 161) |

- **What changed (G):** the twelve `NewtonBamAdapter`s are driven by ONE pinned `DriveBam`
  over 14·12 DOFs (`env_dof_stride=14`) and ONE pinned bridge (`BamCohort`,
  `control/newton_bam.py`): 6 `wp.launch` + 9 `wp.copy` per step instead of 72 + 108
  (counted on the real kernels in `tests/test_newton_bam.py`), with every per-adapter
  check, the single output-check read and the receipt preserved.
- **Measured effect:** `bam.before_step` **12.11 → 6.57 ms** (−5.5 ms, −46 %); step median
  **46.79 → 42.43 ms** (−9.3 %), p95 124.7 → 119.7, mean 59.9 → 54.7. The other phases are
  unchanged within noise (`completed.validate` 23.9 / 24.5, of which ~7.6 ms is the GPU
  physics wait inside `support.decode`). Arm F reproduces the slice-1 arm E of the same
  day (46.78 ms, `bam.before_step` 11.87 ms incl. verify), so the baseline is stable.
  What remains inside the cohort span (~6.5 ms) is the twelve adapters' host-side
  binding/model-sync checks on the snapshot copies, not launches.
- **Not changed, not admission:** all twelve client walks stay `unverified` in both arms
  (RPC wall-time deadline 7/6, stale feedback 2/4, generation change 2/1, clock not
  advancing 1/1) against the unchanged 0.5 s state-age, 0.4 s progress and 0.5 s RPC
  limits; median measured distance per walk 0.644 m (F) / 0.476 m (G) of the 5 m asked —
  per-walk distance is set by when each client trips a deadline, not by this change. No
  limit, physics, asset or policy changed. Both owners ended by the driver's lifecycle
  SIGTERM after the clients finished (`end_reason: signal/lifecycle shutdown`). The owner
  is still ~8.5× slower than real time for twelve robots.
- **Next, re-ranked by this measurement:** `completed.validate` (24 ms: the GPU wait plus
  per-robot deep copies/validation), RPC serving off the owner's GIL (ledger B12a, in
  progress), the remaining host-side binding checks inside the cohort span, the sparse
  `record.physics` / `camera.overview` writes behind the p95 tail, and the physics budget
  per asset (hull/solver) decided with the yaw-drift gate.

Raw rows (owner `timing.jsonl`, `receipt.json`, driver logs) stay in
`cascade-lab/HERMES_AUDIT_20261007/harness/` — paths, sizes and SHA-256 in
[manifest.json](manifest.json); per-arm aggregates in
[timing-summary-F-precohort.json](timing-summary-F-precohort.json) and
[timing-summary-G-cohort.json](timing-summary-G-cohort.json) (`harness/timing_summary.py`).
