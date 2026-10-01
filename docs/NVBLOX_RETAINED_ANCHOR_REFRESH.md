# Retained-payload capture refresh

This recovery contract has passed offline tests and passive validation in a new
owned environment. Physical held-camera recovery and the five-object normal
campaign remain pending. The original scene on port 8691 remains unchanged; the
held scene on 8692 was paused without actuator commands after a separate captured
state and process-identity review.

`isolated-clock-02/camera-held-01` refused home with an unknown attached-object
sample after a frozen-camera barrier discarded the earlier background anchors.
The archived timestamps establish that history was discarded; that run did not
save a comparative ESDF replay, so this is not proof of the sole cause of the
home refusal.

For a confirmed attachment, `_reset_camera_frames(scene_changed=False)` now
invalidates the cached map before reading the producer clock. It preserves and
replays only measured depth history with the same camera set, source, robot,
clock, producer epoch, and prop identities. Captures carry the existing physics
clock UUID as `proprioception.producer_epoch`; no extra state RPC or new UUID is
introduced. Missing epoch metadata prevents this recovery mode.

Historical replay contributes measured background but never counts as a fresh
camera commit. Every mapping camera must integrate and query successfully above
the common capture floor, with the same attachment and observed payload surface,
before the barrier completes. The epoch binding persists after completion. The
public reset preserves the halt generation through home, jaw opening, and the
prop reset. Any capture-refresh failure leaves geometry fenced. No unknown-clearance or 30 mm
clearance rule is relaxed, and no contact exemption is added.

A real prop reset still invokes `begin_scene_reset()` and discards all history.
That invalidation dominates subsequent retries: it cannot be changed into an
unchanged-scene refresh after a failure. Existing per-prop history floors remain
active during retained-anchor replay. As with ordinary historical depth fusion,
a shared producer epoch and absence of a prop reset do not prove that every
unheld object in the world remained stationary.

The held-camera diagnostic records the retained anchor markers, replay count,
producer epoch, and three fresh map commits before home and the later real prop
reset. Its fault remains a `FrozenProducer` stream injection, not a physical
camera disconnection. Cleanup sends no motion, jaw, or reset commands.

The new owned environment uses ports 8693/25561 on GPU 1, with
960×540 cameras, physics dt 1/120 s, capture every six bridge loop iterations,
viewport disabled, 4 Hz camera consumers, and nominal 50 Hz target sampling.
The query ROI is explicitly `[.10, -.30, -.01]` to `[.58, .30, .55]`; the safety
workspace remains bounded by X=.50, clearance remains 30 mm, and mapper RPC,
barrier, and motion wall deadlines remain 500 ms, 5 s, and 120 s. The state RPC
retains its 1 s deadline. No reset, jaw command, or trajectory has been sent in
this environment. Passive observation uses the real runtime and mapper with
bridge writes blocked, including during cleanup. Source, listener ownership,
producer clock, and all three camera identities are checked.

The frozen source `8b6b865` passed 2,795 tests, with 43 skipped and three deselected.
The first complete run had four telemetry-fixture failures because its test
double lacked the real halt-generation guard. Binding the real guard in that
fixture fixed those failures without changing production code. Independent
review covered 185 focused tests; the final anchor/evidence selection passed
42 tests. The PR integration preserves the tested production and test files.

The first passive attempt failed before interval sampling because the diagnostic
passed an unsupported timeout argument to `LazyArm`. It sent no actuator
commands. The corrected reader uses the already materialized Isaac backend with
the existing 1 s state deadline. The second interval passed its observation
checks: 20.158 wall seconds, 3.45 simulation seconds, and RTF 0.1711. All three
cameras had 117 physical-snapshot samples, maximum frame age 0.787 s, and no
samples over 2 s. This does not mean every camera had integrated into the map
within 2 s: maximum sampled map-commit ages were 2.712 s for cam0, 1.685 s for
side, and 1.920 s for proof.

That loaded interval did not establish enough motion margin. Roughly scaling
the previous observed 80.475 s pregrasp by the RTF change suggested 130.17 s,
above the unchanged 120 s limit. This is a workload estimate, not a trajectory
prediction. No physical diagnostic was started on that basis. The owned 8692
bridge was subsequently paused with a single reviewed pidfd SIGSTOP after
observing the raised can with bilateral contacts and saving three RGBD captures.
The first two pause preparations failed without a signal; their receipts remain
archived. The successful third pause preserves the process and has no automatic
resume.

The third passive interval, with only that held bridge paused, also passed its
observation checks: RTF 0.2306, 123 samples per camera, maximum frame age
0.636 s, and no frame samples over 2 s. Sampled map-commit ages remained a
separate limitation: cam0 reached 2.200 s in two samples, while side and proof
peaked at 1.286 s and 1.273 s. Minimum RTF over three- and five-second windows
was 0.1821 and 0.2021. Scaling the previous pregrasp gives 96.63 s at the global
RTF, 110.23 s at the five-second minimum, and 122.35 s at the three-second
minimum. The last estimate exceeds 120 s, so this result alone does not
establish a conservative margin for the physical test. The environment received only observation commands. It was then closed
normally, after a final snapshot confirmed home, both jaws open, all three
camera contact lists empty, and all five props within 1 mm of their spawn
positions. Exact bridge and mapper PIDs each received one SIGTERM through a
pidfd; the launcher exited with its child. No parking, jaw, or prop-reset
command was sent, and no stronger termination signal was needed. All ten passive artifacts and 254 frozen source hashes were
verified. The [compact evidence receipt](../benchmark/results/nvblox-retained-anchor-refresh-20261001.json)
retains the earlier failures, separate frame/commit measurements, and exact
source and artifact references.

During publication preparation, the retained-scenes checkout briefly changed
on disk from `87c55bd` to `d1c6a68` between 04:49:42 and 04:53:13 on 2026-10-01
(Europe/Madrid). It was restored to `87c55bd`, with the commits preserved in a
separate publication worktree. This NV agent sent zero RPC or actuator commands to the retained scenes
during that interval; simulator physics continued advancing. The isolated environment
remained frozen at `8b6b865`.
