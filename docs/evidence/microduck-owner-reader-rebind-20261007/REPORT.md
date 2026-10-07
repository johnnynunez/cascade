# Shared owner per-step cost: cheaper per-robot reads, binds and reader polls (2026-10-07)

Same-day A/B on the x86 test rig (Isaac Sim 6.2 alpha, Newton 1.6.1rc1, GPU 1), twelve
MicroDucks in ONE world, `rough_walk_e`, Isaac Lab USD + complete hulls, route profile,
twelve closed-loop `walk_distance 5 m` clients (`HERMES_AUDIT_20261007/harness/route_walk.py`,
a copy of the route harness pointed at the worktree under test), owner with
`--profile-phases`. Arm B is the host-snapshot build of
[microduck-owner-host-snapshot-20261007](../microduck-owner-host-snapshot-20261007/REPORT.md);
arm D adds this change. Nothing else differs (same scene model identity, same clients).

| metric | B host snapshot | D = B + this change |
| --- | --- | --- |
| attempts (solved steps) | 2868 | 3175 |
| median ms / step | 61.3 | 48.55 |
| p95 ms / step | 147.6 | 126.3 |
| mean ms / step | 80.91 | 62.44 |
| phase completed.validate | 29.68 | 23.75 |
| phase policy.prepare | 8.47 | 4.22 |
| phase publication | 5.39 | 2.99 |
| phase bam.before_step | 13.61 | 13.86 |
| phase solve | 0.28 | 0.25 |
| nested support.decode | 11.87 | 11.48 |
| nested native.capture | 2.33 | 2.24 |
| per-robot completed.validate | 0.571 | 0.278 |
| per-robot policy.prepare | 0.647 | 0.346 |
| per-robot publication | 0.424 | 0.246 |
| per-robot bam.before_step | 0.987 | 0.997 |

- **Why these four changes:** an in-process `cProfile` of the B owner
  (`--profile-cprofile`, new diagnostic flag; [top 30](cprofile-top30-P2.md)) over 1000
  completed steps showed, besides the GPU waits, pure-Python work repeated per robot or
  per reader poll on data that is immutable within a step:
  `RobotBinding.model_identity_sha256` recomputed `asdict + json + sha256` on every access
  (23 036 calls, 5.7 M `_asdict_inner` calls); `SceneLayout.robot_support` rebuilt the
  shared `SupportObservation` with `dataclasses.replace` for each of the twelve robots,
  re-running `__post_init__` over every solved contact (11 999 calls); every reader poll
  (`state()`, about one per robot per step from the twelve clients, served on the owner's
  GIL) walked the ~90 solved contacts again in `BaseState.as_dict`; and `identifier()`
  scanned every shape path, joint and body name character by character in a generator
  (34 M iterations).
- **What changed (D):** the binding digest is a `cached_property` of the frozen binding;
  `SupportObservation.rebound` re-validates only the two identifiers it changes and keeps
  the shared, already validated contact tuple (non-plain records keep the `replace` path);
  `BaseState.as_dict` memoizes the plain copy of the contacts once per shared contact
  tuple (a cell shared by the rebound copies) and still hands every reader fresh
  containers; `identifier()` uses a compiled regex with the same rule. No check was
  removed: the same inputs are rejected, every reader reply is byte-identical in content
  and still isolated from the others (tests in `tests/test_microduck_shared_support_cache.py`).
- **Measured effect:** step median 61.3 → 48.55 ms (−21 %), p95 147.6 → 126.3;
  `completed.validate` 29.68 → 23.75 (per robot 0.571 → 0.278), `policy.prepare`
  8.47 → 4.22 (0.647 → 0.346), `publication` 5.39 → 2.99 (0.424 → 0.246);
  `bam.before_step` unchanged (13.61 → 13.86, noise). The reader polls are not a phase:
  their GIL time shows up as inflation of whichever phase is running, which is why phases
  that this change touches only indirectly also shrink.
- **Not changed, not admission:** all twelve client walks remain `unverified` in B and D
  (stale feedback / RPC wall-time deadline); the 0.5 s state-age, 0.4 s progress and 0.5 s
  RPC limits were not relaxed. The GPU physics step is still 7.6 ms per 5 ms step.
- **Remaining host cost after D** (per 5 ms step, twelve robots): `completed.validate`
  23.75 (of which GPU wait ≈ 7.6, decode 4.4, native capture 2.2, deep copies);
  `bam.before_step` 13.86 — the profile counts ~230 Warp device→host copies per step, of
  which the twelve adapters' five output reads each (`external_torque`, `forces`,
  `motor_torque`, `effective_vin`, `friction_budget`) are 60 device syncs: the next slice
  is a cohort-level batched check (one device-side finiteness reduction, one read);
  `policy.prepare` 4.22; `publication` 2.99; the sparse `record.physics` (≈28 ms every
  tenth step) and `camera.overview` (≈83 ms every twentieth) that make the p95 tail.
- Raw rows, receipts, launch records and the cProfile dump stay in
  `cascade-lab/HERMES_AUDIT_20261007/harness/` (paths and SHA-256 in `manifest.json`).
