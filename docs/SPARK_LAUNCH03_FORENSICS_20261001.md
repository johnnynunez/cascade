# Spark launch03: physical green case passes, camera freshness fails

The normal native launch at `7cb4b26cad6bbdb6ccddb8835b94b9e8ebec2068` combined
the reviewed physics-time executor with the headless viewport setting. It used
the existing camera cadence option `CASCADE_ISAAC_CAM_EVERY=12`, 1280×720 images
and 1/120 s physics. Memory was retained and the motion, settling, safety,
freshness and native turn budgets were unchanged.

The launch correctly ended **NOT READY**. Its first case completed one native
`pick_and_place` action for the green cube, returned home and reset the scene.
The native result had `ok=true`, `verified=true` and a physics-confirmed
post-placement verdict. The independent witness also passed physical pick,
support, release, destination containment and reset checks. Camera freshness
failed, so the complete case failed and the orange case never started.

| Measurement | Observed result |
| --- | --- |
| Grasp attempt wall duration | 105.93 s |
| Full native pick-and-place action | 254.74 s |
| Passive full-pick lift | 103.387 mm |
| Bilateral contact samples during full-pick lift | 730 |
| Pregrasp command window, 375 targets | 18.683–18.717 physical seconds |
| Descent command window, 100 targets | 4.950–4.983 physical seconds |
| Fresh stable handoff windows | 0.100 physical seconds, seven samples |
| Stale camera samples | 38 of 2,119, identically for all three cameras |
| Maximum measured camera age | 2.906825 s; allowed maximum remains 2 s |

Physical command windows are bounded by adjacent feedback samples, not exact
command-application timestamps. The conservative executor did not compress the
trajectory or catch up after delays. Its post-acknowledgement wait and the
bridge's two-physics-step updates made nominal 50 Hz transmission about 20 Hz
in this observed run, extending the physical duration.

All camera records remained available, robot-bound and monotonic. Three stale
samples occurred before or after the grasp; the other 35 occurred during reset
and subsequent native turns. Cache captures were typically 24 physics steps
apart: 0.2 physical seconds. Around the worst stale sample the measured local
simulation/wall ratio fell to approximately 0.066, making that cadence too slow
to meet the existing two-second age limit. The initial idle camera check did
not establish freshness under this load.

The selected grasp and preserved memory prior differ from launch02. The lemon
also moved up to 4.53 mm during descent/close/lift, without an identified
contacting link. These observations do not establish collision-free handling,
an identical-grasp comparison, five-object reliability or successful orange
handling.

The [receipt](../benchmark/results/spark_native_launch03_forensics_20261001.json)
records hashes and analysis paths. The retained archive contains the original
proof, owner/process binding, exact MCP trace, physical witness, grasp JSON/NPZ,
camera images and preparation records. Because the first case failed before a
checkpoint, the original `proof.json` still has no process/model fields. It was
not rewritten: the forensic binding uses the observed attempt PID, exact trace
hash and matching retained launch-owner process receipt. No OpenClaw private
configuration or session archive is included.
