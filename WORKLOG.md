# Development notes

Use this file for a short plan and unresolved handoff for the current change.
Describe completed changes and validation in the commit or pull request;
maintain capability limits in the relevant document below. Do not append full
command transcripts, repeated suite totals or a second copy of an evidence
receipt here.

## Inter-arm gate measured against MuJoCo (7 October 2026, ROADMAP #8)

Landed: `multi_arm_scene_xml` (two prefixed SO-101s in one world at the
profiles' `base_pose`s), `tests/test_multi_arm_physics.py` (chain = MuJoCo
joint anchors to 1e-5 m; declared `link_radii_m` envelope every collision
geom; the gate's surface clearance is a lower bound on `mj_geomDistance` over
2000 random pose pairs; the centreline gate approved 12 overlapping pairs),
per-link radii in `so101.yaml`, fail-closed radii plumbing in the harness and
`_wire_neighbors`, margin 0.03 m on the dual profiles. Handoff, in order:
(1) prefix-aware `MujocoArm` + `MujocoWorld` sharing so `so101_left`/`so101_right`
can be `type: mujoco` in that scene and `test_arm_rig`'s rig test runs on
physics (joint names `left/shoulder_pan`, one world for both arms and the
camera); (2) the envelope is loose where a slim link faces the neighbour
(inward-yaw pose: 0.13 m physics, 0.032 m gate) — a per-geom sphere model
placed by Pinocchio frames would recover that workspace while keeping the
lower-bound property; measure before changing the margin; (3) the physics
test takes ~6 s — keep it CPU-only and seeded.

## Unitree H2 vertical — owner and first episodes (7 October 2026)

Decision: Unitree H2, PhysX first (NVIDIA's public H2 USD + `Velocity-H2-History-v0`,
pinned in `configs/h2/bundle.json`; Newton route = newton-assets PR #53). Landed:
the contract module (`control/h2_policy_contract.py`), the PhysX owner
(`scripts/isaac_h2_bridge.py`, `sim/h2_physx.py`, `sim/h2_stepper.py`,
`sim/h2_identity.py`), the `kind` field on the MOBILE wire (`microduck`/`h2`),
the candidate profiles and `docs/HUMANOID_H2.md`. Measured: the reference loop
walks 0.63 m/2 s; the CASCADE-owned episode confirmed 3 of 5 `walk_velocity`
commands under the candidate verifier (two refuted on the post-command settle
window). Handoff: measure the settle behaviour (how long the policy keeps
stepping after a zero twist) before touching any candidate limit; add
`distance_control`/`turn_control` for `walk_distance`/`turn`; MCP-driven
episodes; a Newton experiment with the same policy is a separate, labelled run.

## Start here

- [README](README.md): project overview and entry points.
- [Architecture](docs/ARCHITECTURE.md): components and runtime flow.
- [Agent guide](AGENTS.md): repository rules, testing and safety contracts.
- [Project status](docs/PROJECT_STATUS_20261003.md): dated, source-bound
  capabilities, retained failures and pending physical admission.
- [Roadmap](docs/ROADMAP.md): longer-term direction, distinct from verified status.

## Implementation and evidence guides

| Area | Maintained documentation |
| --- | --- |
| Manipulation and assembly | [Integration](docs/MANIPULATION_ASSEMBLY_20261002.md), [fastening](docs/FACTORY_FASTENING_RUNTIME.md) |
| MuJoCo placement | [Withdrawal](docs/MUJOCO_RELEASE_WITHDRAWAL.md), [region](docs/MUJOCO_DESTINATION_REGION_V2.md), [post-release planning](docs/MUJOCO_POSTRELEASE_WITHDRAWAL.md) |
| Kitchen and cuMotion | [cuMotion](docs/CUMOTION.md), [observer measurements](docs/KITCHEN_OBSERVER_LATENCY.md), [motion clocks](docs/ISAAC_MOTION_CLOCK.md) |
| Mobile and composed robots | [MicroDuck](docs/MICRODUCK.md), [robot domains](docs/ROBOT_MODULARITY.md), [embodiment](docs/EMBODIMENT.md) |
| RGB-D and spatial sensing | [Observations](docs/OBSERVED_RGBD.md), [checker accuracy](docs/RGBD_CHECKER_ACCURACY.md), [ground rendering](docs/RGBD_GROUND_NATIVE.md), [spatial providers](docs/SPATIAL_PROVIDERS.md) |
| Benchmark adapters | [Arena preflight](docs/ARENA_NATIVE_PREFLIGHT.md), [VAB preflight](docs/VAB_NATIVE_PREFLIGHT.md) |

CPU contract tests, passive preparation, native readiness and task acceptance
are different results. Keep each claim tied to its source and recipe. A later
software fix does not turn an earlier physical failure into a pass; diagnostic
records and process closure do not establish motion, support or placement.
Existing receipts remain under [benchmark/results](benchmark/results/) and
[docs/evidence](docs/evidence/).
Do not delete a real regression test merely because another test exercises
some of the same code.

## Working practice

1. Record the concrete problem, affected capability and intended check briefly.
2. Reuse existing fixtures and parameterize parallel cases; retain distinct
   faults, real transport boundaries and safety assertions.
3. Run the affected tests and required checks. Report their scope without
   adding overlapping suite counts or inferring physical admission.
4. Put stable interface/usage changes in the relevant guide. Close this note
   when the change lands instead of growing a chronological transcript.

## Historical worklogs

The previous 16 worklogs are preserved in Git, without another archive copy.
Browse the [last complete snapshot](https://github.com/johnnynunez/cascade/tree/9b08094d44af6f05fe713d20b825d3e1a3851b1c)
or retrieve an individual file locally:

```bash
git show 9b08094d44af6f05fe713d20b825d3e1a3851b1c:WORKLOG.md
git show 9b08094d44af6f05fe713d20b825d3e1a3851b1c:WORKLOG_MUJOCO_REGION.md
git log --all -- WORKLOG.md 'WORKLOG_*.md'
```

Old “next” steps, temporary publication freezes and per-episode authorizations
are historical context, not current instructions. Historical receipt hashes
continue to identify their original source; those receipts were not rewritten
by this cleanup.

## Current change

Shared-owner per-step cost (twelve MicroDucks, one world, closed-loop agents):
`BamHostSnapshot` lets the twelve `NewtonBamAdapter`s run their unchanged checks
against one per-step host copy of the model/solver/state arrays (measured
`bam.before_step` 24.3 → 13.6 ms, step median 76.2 → 61.3 ms, all walks still
`unverified`; `docs/evidence/microduck-owner-host-snapshot-20261007/`). The new
`--profile-sync-solve` diagnostic shows 7.6 ms of GPU physics per 5 ms step.
Second slice, from an in-process `--profile-cprofile` of that owner: cached
binding digest, metadata-only `SupportObservation.rebound`, memoized plain
contacts for the twelve clients' `state()` polls, regex `identifier()` —
step median 61.3 → 48.55 ms, `bam.before_step` unchanged
(`docs/evidence/microduck-owner-reader-rebind-20261007/`).
Third slice: `BamOutputCheck` replaces the twelve adapters' five per-step host
reads of their own drive outputs (60 device syncs) with device-side finiteness
flags and ONE read per step in `SharedMicroduckStepper.tick`, right before the
solve (`bam.verify` span); every check and its wording are preserved, the
single-robot path is unchanged. Measured 13.9 → 11.9 ms
(`docs/evidence/microduck-owner-bam-check-20261007/`).
Fourth slice: `BamCohort` drives the twelve adapters with ONE pinned `DriveBam`
over 14·N DOFs and ONE pinned bridge over the concatenated DOF indices (the
adapters' arrays become live row views, so `set_targets`/`reset`/`telemetry` are
unchanged per adapter): one Warp launch per stage for the fleet — counted on the
real kernels, 72 → 6 launches and 108 → 9 copies per step — with every adapter's
host-side check run first and the cohort-level marks keeping the output check's
single read. Only delay-free profiles (every twelve-duck run so far) are exact
under the pinned kernels' block-seeded delay RNG, so `cohort()` declines others
and the owner keeps the per-adapter path (receipted). CPU tests on the real kernels
and the software doubles establish the mechanism; the x86 A/B is not yet measured.

Unresolved handoff: measure the cohort build on the x86 rig (`--profile-phases`, twelve
robots, route harness; `bam.before_step` is now the one cohort span, compare
`bam.before_step + bam.verify` against the slice-1 build, 11.87 ms) and add the numbers to
MICRODUCK.md / ROADMAP / an evidence dir. Inside the cohort span what remains is host
work: the twelve adapters' `_check_live` binding/model-sync checks on the snapshot copies
(a cohort-level check over the stacked DOFs is the next candidate, keeping every refusal).
Then the deep copies of `completed.validate` and `policy.prepare`, the sparse
`record.physics` / `camera.overview` writes behind the p95 tail, the physics
budget per asset (full-vertex hulls, 7.6 ms GPU per 5 ms step) and RPC serving
on the owner's GIL are what stand between the owner and twelve closed-loop
clients meeting the unchanged deadlines.
