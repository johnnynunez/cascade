# Isaac verifier readiness before native proof

The native launcher's existing lazy runtime check now prepares Isaac's
read-only verification channel before starting the manipulation proof. It
executes the existing `TruthPoseReader` probe once on a dedicated bridge
connection. This initializes the bridge's shared `RigidPrim` views using the
existing `reset_xform_properties=False` and `prepare_contact_sensors=False`
flags. It never connects, resumes or materializes the lazy arm.

An appended completion record must agree with the probe's physical clock.
Every inventoried dynamic body must have a unique live physical pose/path;
legacy USD fallback alone is insufficient. Robot identity, joint convention,
physical base pose and units are checked. These measured poses are not returned
as action authority. Later effect verification and held-object observation keep
their existing fresh reads and deadlines.

After completion, every configured Isaac camera must deliver RGBD with a
matching endpoint, robot and producer epoch, and a render reference bound to
its captured state. Capture time and physics step must be newer than the probe.
Repeated whole packets can be observed while waiting but cannot satisfy this
barrier. The startup report records server age and a conservative age bound
that adds local elapsed time for the clock request and validation. Both must
fit the existing two-second freshness limit. No camera is retimestamped.

The complete readiness phase, including connection, stream lock acquisition,
probe and frame admission, has a ten-second wall limit matching the existing
bridge client's default transport budget. A failed or timed-out probe is never
resubmitted. It aborts startup, leaves the error in `runtime-check.log`, tears
down the temporary runtime through the existing `finally`, and cannot print
`runtime builds` or proceed to the native proof. Other arm backends skip this
check. This is the native Isaac proof contract, including custom Isaac arm
profiles: the scene must contain at least one uniquely identified dynamic body,
and every configured proof camera must be an Isaac RGBD stream on that same
bridge. An empty scene, a mixed camera/backend rig or an Isaac configuration
without the required live atomic physics channel fails explicitly. Such setups
need their own verification contract before using this native proof path; this
does not claim untested simulator-backend parity.

## Evidence and limits

The local x86 runs on `dc566892` and `ed29a59` each recorded stale frames before
the first grasp actuator command. In run 02, the first stale witness timestamp
was 5.702 ms after the grasp-attempt recorder began; in run 03 it was 4.017 ms
afterward. The maximum server ages were 2.268394 s and 2.395272 s respectively.
The exact source calls `effects.snapshot()` before entering that grasp recorder.
Its first truth read executes the probe on Isaac's main thread after a camera
publication. This order is consistent with a cold verifier blocking subsequent
camera publication. It does not isolate import, stage traversal, view creation
or other queued work as the exclusive cause. Both failed runs remain failures.

CPU regression tests execute the native probe with SDK doubles, check that it
reuses views without authoring USD, and demonstrate that a later truth read
returns the new physical pose. Additional tests reject incomplete probes,
missing physical bodies, foreign clocks/epochs, malformed or duplicate camera
bindings, stale/repeated frames, transport failures and a contended stream lock.
The actual launcher block is exercised to ensure failure precedes its success
marker and still shuts down the temporary runtime. These tests establish the
software contract, not live startup or physical acceptance. A new source-bound
live run is required to measure those outcomes.
