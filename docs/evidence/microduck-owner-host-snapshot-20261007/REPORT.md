# Shared owner per-step cost: one host snapshot for the BAM checks (2026-10-07)

Same-day A/B on the x86 test rig (Isaac Sim 6.2 alpha, Newton 1.6.1rc1, GPU 1), twelve
MicroDucks in ONE world, `rough_walk_e`, Isaac Lab USD + complete hulls, route profile,
twelve closed-loop `walk_distance 5 m` clients (`HERMES_AUDIT_20261007/harness/route_walk.py`,
a copy of the route harness pointed at the worktree under test). Owner profiled with
`--profile-phases`; arm C adds the new `--profile-sync-solve` diagnostic so the GPU
time of the physics step is attributed to `solve` instead of to the first host read.

| metric | A baseline (PR #222 head) | B + host snapshot | C = B + sync-solve diagnostic |
| --- | --- | --- | --- |
| attempts (solved steps) | 1808 | 2868 | 2100 |
| median ms / step | 76.24 | 61.3 | 62.53 |
| p95 ms / step | 167.0 | 147.6 | 145.3 |
| phase bam.before_step | 24.31 | 13.61 | 13.77 |
| phase completed.validate | 29.86 | 29.68 | 22.51 |
| phase policy.prepare | 8.66 | 8.47 | 8.78 |
| phase publication | 5.37 | 5.39 | 5.32 |
| phase solve | 0.3 | 0.28 | 7.59 |
| nested support.decode | 11.75 | 11.87 | 4.39 |
| nested native.capture | 2.48 | 2.33 | 2.36 |
| per-robot bam.before_step | 1.872 | 0.987 | 0.999 |

- **What changed (B):** `NewtonBamAdapter.before_step(dt, snapshot=…)` runs every one of
  its checks against ONE per-step host copy of the model/solver/state arrays
  (`BamHostSnapshot`, captured by `SharedMicroduckStepper.tick` after the policy commits
  and before the first actuation) instead of each of the twelve adapters re-reading the
  same ~25 whole-model Warp arrays with `.numpy()` (each a device sync). No check was
  removed or weakened: CPU tests show identical efforts with and without the snapshot,
  a stale/foreign snapshot is refused, and model drift / NaN state / DOF-map changes
  still fail closed through the snapshot path.
- **Measured effect:** `bam.before_step` 24.31 → 13.61 ms per step
  (1.872 → 0.987 ms per robot); the step median
  76.24 → 61.3 ms. Every other phase is unchanged within noise.
- **What C shows:** the physics step itself costs **7.59 ms of GPU time per 5 ms step**
  for twelve robots (hidden inside `support.decode` = 11.87 ms in A/B; the pure
  contact decode is 4.39 ms). The owner cannot reach real time while the
  physics alone takes 1.5× the step, independently of the Python work.
- **Not changed, not admission:** all twelve client walks (one per robot) in A and B remain `unverified`
  (deadline failures: stale feedback / RPC wall-time), as in the retained route evidence.
  The 0.5 s state-age, 0.4 s progress and 0.5 s RPC limits were not relaxed. The remaining
  owner cost after B: `completed.validate` 22.51 ms (decode 4.39,
  native capture 2.36, per-robot validation and deep copies), `bam.before_step`
  13.61 (per-adapter output reads and ~48 Warp launches), `policy.prepare`
  8.47, `publication` 5.39.
- Raw rows, receipts and launch records stay in `cascade-lab/HERMES_AUDIT_20261007/harness/`
  (paths and SHA-256 in `manifest.json`).
