# Check the observed closing stroke before commanding it

This closing preflight and the earlier approach gate are merged and enabled
by default in the installed Spark profile. The reports below are source-bound
historical evidence. [Project status](PROJECT_STATUS_20261001.md) distinguishes
software validation from the current native proof, five-object campaign and
restart acceptance.

Native launch05 completed both audited pick/place/reset cases, but the pink
cube moved about 19.8 mm during the orange grasp's closing phase. The existing
gate covered the open approach. The target-only contact sensor cannot directly
identify contact with the neighbor; recorded timing and offline collision-mesh
geometry support the closing right finger as the cause.

With the [mask alignment correction](detector-mask-letterbox.md), the saved
orange observation contains a non-target surface point at pixel (223,751)
inside the selected pose's closing envelope. No object name, simulator pose,
bounding-box exemption, mask dilation or point-count threshold is used by the
new veto. Every observed point outside the exact target mask remains an obstacle.
Target points also remain obstacles throughout home, pregrasp and descent.

The gate rejects a candidate if either complete finger envelope, extruded over
its full mechanical stroke, intersects those non-target points at the grasp
pose. Each finger's interval is independent, so asynchronous closing is included.
The same fixed 0.1 mm geometric expansion extends the checked volume; it is not
a new assertion that closing under contact tracks within that amount.

The validated PhysX path uses the [nominal-plus-derived envelope](physx-finger-envelope.md)
for both approach and closing checks. It preserves the exact target-mask
exemption only during closure; target points remain checked during approach.
The added components are a geometric correction, not a new 30 mm finger-distance
rule, contact margin or relaxation of the existing opening feedback bound.
The historical results below are not reruns with this later artifact.

Before **each** closing-stage command, the runtime reads current joint and
individual-finger feedback, validates the source/robot/epoch/physical-clock
contract, rejects contradictory snapshots from the same physics step, and
rechecks the entire closing envelope at that measured pose. The second stage
may start partially closed. Cancellation is checked before and after the read
and geometry check, and the original halt generation reaches the gripper command.
A failed first veto does not create a provisional held object. After a command
has been attempted, a later failure retains the provisional marker because an
object may physically be held. There is no automatic retry or recovery motion.

This is a **pre-command veto**, not continuous control or a demonstrated brake
for an in-progress closing command. Rebinding the robot pose does not refresh
the captured scene or prove neighbors stayed still. The scope remains observed
surfaces, calibrated moving fingers, discrete approach samples, and a stationary
TCP assumption for the geometric closing extrusion. The palm, remaining arm,
lift, transport, occlusions and motion induced during closure are not certified.

Offline replay of the historical candidates reproduces both original selections
before applying the new closing check. With corrected masks, it finds alternatives
for green and orange while keeping all unassigned boundary pixels as obstacles.
Those candidates were produced from historical localization, so this establishes
neither the result of new GGX requests nor physical acceptance. Full runtime
validation and a new physical campaign are required before deployment.

## Native launch06: green passes, orange planning rejected

The normal Spark installation of `b28ebc3` passed its asset, dependency and
installation checks. The runtime passed 2,658 tests (43 skipped, two deselected)
and 242 independent focused tests. The installed defaults supplied camera
cadence 6, physics step 1/120 s and the observed-finger gate; the driver removed
inherited overrides for those settings. Existing grasp memory was preserved.

The [retained launch06 receipt](../benchmark/results/spark-observed-finger-native06-20261001.json)
records a complete green-cube native placement and reset, passing external
physics audit and strict native confirmation. Both closing preflights ran.
Sampled neighboring props moved less than 0.000102 mm during home, pregrasp,
descent, close and lift. All three cameras had 1,858 samples with server ages
below two seconds; two deliveries per camera exceeded two seconds during reset
(maximum 2.046437 s). These sampled measurements do not establish continuously
fresh delivery or collision-free motion between observations.

Orange did not complete. GGX returned 400 learned poses: 167 passed the existing
score threshold, ten passed the vertical-approach filter, and only one passed
both. The observed-surface gate rejected its pregrasp path in 1.50 s, before
that grasp attempt sent any joint or gripper targets. Saved-input replay
reproduces the rejection. The symmetric orientation also fails; a historical
candidate passes on this observation but was not used for actuation. Preserved
memory adjusted ranking and offsets rather than supplying a cached pose.

The launcher withheld READY and closed its owned services. The strict checker
accepts green and rejects the incomplete orange proof. The retained Chromium
window stayed disconnected with controls disabled, without reloading; this is
an offline-state check, not successful UI or restart acceptance. The archive
contains 82 hash-verified files. Five-object native acceptance remains pending.
