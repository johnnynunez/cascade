# Check the observed closing stroke before commanding it

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

The gate now rejects a candidate if either complete finger mesh, extruded over
its full mechanical stroke, intersects those non-target points at the grasp
pose. Each finger's interval is independent, so asynchronous closing is included.
The same fixed 0.1 mm geometric expansion extends the checked volume; it is not
a new assertion that closing under contact tracks within that amount.

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
