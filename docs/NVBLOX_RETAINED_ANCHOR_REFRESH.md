# Retained-payload capture refresh

This change is an offline-tested recovery contract, not a completed physical
acceptance result. The retained scenes on Isaac ports 8691 and 8692 remain
untouched. The five-object normal campaign remains pending.

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

The next proposed environment uses new owned ports 8693/25561 on GPU 1, with
960×540 cameras, physics dt 1/120 s, capture every six bridge loop iterations,
viewport disabled, 4 Hz camera consumers, and nominal 50 Hz target sampling.
The query ROI is explicitly `[.10, -.30, -.01]` to `[.58, .30, .55]`; the safety
workspace remains bounded by X=.50, clearance remains 30 mm, and mapper RPC,
barrier, and motion wall deadlines remain 500 ms, 5 s, and 120 s. The state RPC
retains its 1 s deadline. The existing held scenes
also consume GPU compute. Source/ownership verification and passive measurements
of the clock, all three cameras, and the real mapper must be reviewed before any
actuation in that new environment. No new environment has been started by this
patch.
