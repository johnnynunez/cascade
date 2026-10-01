# Retained-payload capture refresh

The implementation is merged through PR #27. The source-specific trials below
remain historical; later release and held-observation changes do not upgrade
any failed trial. Use [project status](PROJECT_STATUS_20261001.md) for the current
MAIN recovery/campaign stage.

This recovery contract has passed offline tests and passive validation in a new
owned environment. A later single held-camera recovery passed as recorded below; the five-object
normal campaign remains pending. The original scene on port 8691 remains unchanged; the
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


A later environment, `isolated-clock-04`, used frozen source `1efa8e9` with the
ordinary Isaac profile set to 30 Hz target sampling. Safety preserves every
original 50 Hz edge together with the actual targets and intervening subedges.
Both retained halt-generation checks remain in planned moves. The complete
suite passed 2,829 tests, with 43 skipped and three deselected; independent
review also ran 96 focused tests. The PR integration has identical production
and test files. Rendering, camera rates, ROI, clearance, unknown-payload policy,
and deadlines were unchanged. Its first passive interval passed, with RTF
0.2287 and maximum age 0.628 s across all three cameras; sampled cam0 map-commit
age still reached 2.190 s. This passive evidence is distinct from the physical
run below.

One `camera-held-01` diagnostic then passed. The can rose about 99.9 mm with
observed bilateral contact. Freezing only the consumed side-camera producer
caused the intended 5.002 s barrier failure; the pending public reset refused
after 5.263 s without actuator calls. Restoring that producer let recovery
retain all three original measured anchors, replay six measured frames, and
commit all three new captures above the common floor under the same producer
epoch. Home then completed, followed by one jaw-open command and one real
five-prop reset. The later scene reset discarded old history and produced three
new map commits. Final feedback was home with open jaws, no held object, and a
fresh map without pending errors. Cleanup issued no commands.

The five motion requests sent 45, 225, 60, 120, and 90 targets, taking 10.411,
49.201, 12.858, 43.717, and 29.667 wall seconds. Each was below the unchanged
120 s per-motion limit. The complete grasp phase took 140.377 s, including
perception and multiple motions; it is not one motion budget. All 206 recorded
return-home snapshots before release showed bilateral contacts, with the can
at least 98.7 mm above its spawn height. Total diagnostic actuation was 540
joint targets, four jaw commands, and one prop reset.

The physical diagnostic's `GpuProofObserver` records continuous proof-camera
frames only. Its 1,217 snapshots contained two frames older than 2 s at
sequences 5 and 6, peaking at 3.548 s before the first actuator call. Across the
931 fully enclosed joint-motion snapshots, maximum proof-frame age was
0.590 s. Continuous cam0 and side ages are unavailable for this physical run;
the generic three-camera summary consequently fails for missing fields.
Three successful source commits at each recovery boundary do not prove
continuous three-camera freshness. The normal campaign's separate observer
must audit all three cameras.

Ultralytics also downloaded the previously absent `mobileclip_blt.ts` during
setup despite the offline environment flags. The 599,764,649-byte artifact
matched the historical cached model exactly (SHA256
`a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54`).
The startup freeze included the two `.pt` models but not this `.ts` file; its
hash is explicitly post-download evidence. Future campaign preparation includes
that model before launch. All 17 physical artifacts, 242 diagnostic sources,
and 257 startup source entries were checked unchanged. The
[held-recovery receipt](../benchmark/results/nvblox-held-camera-anchor-refresh-20261001.json)
links the complete physical evidence and these scope limits. No normal
five-object result is claimed.


## First normal five-object campaign: stopped after orange

The next ordinary campaign used the same frozen `1efa8e9` runtime, nominal
30 Hz target sampling, retained 50 Hz safety checks, three cameras and unchanged
30 mm clearance, 500 ms mapper RPC, 5 s capture barrier and 120 s motion budget.
The normal five-object acceptance has **not passed**: one orange case was attempted
and failed; green cube, pink cube, lemon and tomato can were not attempted because
`--fail-fast` stopped after the first case and its one explicit reset.

The physical witness independently passed the orange pickup and placement: two
jaw contacts during the lift, full convex footprint inside the box, actual jaws
open, settled support at the destination. The object rose 122.64 mm and moved
280.06 mm horizontally. This placement result does not make the whole skill pass.
The post-release withdrawal aborted after a native masked-depth mapper request
timed out at 500 ms. The subsequent reset refused home during planning because
its starting TCP had 27 mm clearance, below the unchanged 30 mm minimum. The
nearest obstacle has not been attributed from that error alone. The recorded
payload query was inherited from an earlier phase and is not the reset clearance.

All three camera age series passed: 1,544 samples per camera, maximum age
0.759687 s, no samples above 2 s, 560 distinct sampled captures and maximum
observed capture gap 0.830530 s. All 261 frozen source and model hashes, including
MobileCLIP, matched after completion. Seven motion requests completed; the
longest took 73.808 s. The failing retreat request lasted 0.567 s. This trace
records SafeArm requests, so it does not establish the exact number of backend
targets sent during that failing request. Cleanup closed resources without
movement; no external retry, opening, prop reset or parking was added.

The failed scene was initially preserved at the post-release pose. A later
read-only capture saved three RGBD frames and three identical current ESDF
queries, costing 35.74/36.36/29.36 ms for the existing map. Those later queries do
not establish the cause of the earlier integration timeout or the freshness of
all fused geometry. After a final state capture, only that scene's owned bridge
and mapper were terminated normally, without withdrawal, home, opening or prop
reset. This administrative close did not recover the failed trial or create a
restorable checkpoint. The subsequent [release recovery implementation](NVBLOX_RELEASE_RETREAT.md)
retains authority in its original runtime; the separate [mapper reset correction](NVBLOX_MAPPER_RESET.md)
is included in candidate `7bdaf5a`. Neither change retroactively completes this
first five-object campaign.

Evidence: `benchmark/results/nvblox-normal-five-first-failure-20261001.json`;
full local receipt SHA256 `45c0c91dab7a13d3e3a380004bfac57e92ef1dc7603cbf45ce88311ce5a2ceab`
binds 34 artifacts and the 261 source/model files. The later read-only receipt is
`5a3490f63cba03e4b86f58afbcc627adb0d4a3d417744f18a25e74b7032b44d2`.
