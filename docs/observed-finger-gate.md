# Observed-surface finger veto (experimental)

`CASCADE_OBSERVED_FINGER_GATE=1` enables this gate for the calibrated Isaac
reBot grasp path. The normal MCP launcher forwards it explicitly. It is off
by default and requires the `grasp-safety` extra (SciPy; also present with the
occupancy extra). Other backends fail closed if this option is requested.
This change has offline evidence; it does **not** have physical acceptance.

The native04 orange attempt localized the object within about 1 mm, but the
selected descent pushed the orange and its neighbor before closing. An offline
mesh analysis found that even its ideal joint path intersected the neighbor.
GGX received the segmented object cloud; that alone could not veto the neighbor.

The new check backprojects every valid, non-robot depth pixel from the same
calibrated frame. It requires that frame's exact robot mask, real target mask,
camera endpoint, robot identity and producer epoch, matched to current atomic
physics feedback. The target mask only classifies a conflict: target, neighbors
and support remain in the checked scene. Invalid/missing masks, depth, epoch or
calibration reject the attempt. It does not synthesize hidden geometry or free
space. Eye-in-hand transforms must come from the capture; no live FK fallback.

Detection, segmentation and localization retain their CUDA execution. Copying a
mask to the host and building the observed-scene KD-tree is an explicit CPU
safety boundary, like the existing FK/harness. This is not an entirely CUDA
pipeline claim. No new camera read or segmentation call is introduced. Logging
only copies available observations; the new safety gate adds one current arm
read immediately before the first opening command, after selection.

## Geometry and opening interval

The artifact covers **only the complete collision meshes of the two finger
links**, `gripper_left` and `gripper_right`. It does not cover `gripper_end`
(palm), the rest of the arm, carried objects or the closing/lift/place phases.
The ordinary safety harness remains in force for those motions.

`scripts/build_gripper_scene_geometry.py` consumes the hashed URDF and both
collision STLs. It applies mesh scale, collision origin and joint origin, and
builds a convex hull for every connected component (8 per finger). No small or
degenerate component is silently omitted. Coverage of all 17,246 triangles and
8,649 unique vertices per finger is certified by containment of every vertex
in the component's unit-normal halfspaces. Convexification can cause additional
rejection of concavities. The runtime verifies source hashes and calibrated
joint names, limits and TCP frame. Runtime assets remain subject to the normal
source/asset manifest and deployment checks.

For each finger separately, the checked stroke interval spans its measured
position and the authored open limit, expanded by **0.0001 m at both ends**.
Measurements beyond the authored limit are never clipped. Every component is
extruded conservatively across that whole interval before planning and before
the first opening command. Halfspace offsets and bounding boxes use the same
interval; the KD-tree's circumscribed radius encloses the tested volume.

The fixed 0.1 mm expansion is an empirical tracking envelope, not a universal
noise bound. The retained [individual-jaw receipt](evidence/observed-finger-gate/open-jaw-tracking.json)
measured a maximum 70.531 micrometers during the green native04 open phases;
its target contact count was zero, which does not establish absence of contact
with other bodies. The orange descent's millimeter-scale interaction was
excluded from this calibration. Passive05 also demonstrated nanometer-scale
variation across the authored upper limit. Feedback outside the fixed interval
aborts; the bound never grows automatically to accommodate a collision.

The contract is “opening remains inside a geometrically checked interval,” not
“the open command has completed.” Current feedback checks both independent jaw
positions, finite arm q, clock progression and unchanged endpoint/robot/epoch.
Contradictory q or jaw positions on the same physics step reject the snapshot.

## Planning and execution

Before opening, the gate vets current→home, home→pregrasp and descent. It uses
the shared nominal profile: actual 30 Hz targets/edges, original complete
50 Hz edges including intersecting lookahead, and virtual subedges. Both edge
endpoints are checked; this is discrete sampling, **not continuous collision
certification**. Durations receive the same velocity stretch as SafeArm.

Each actual home/pregrasp/descent stream repeats preflight using the executor's
current feedback and existing bounded rebind. Home must settle successfully;
the old ignored best-effort home is retained only when this option is off.
The feedback hook consumes the executor's existing state reads and rejects an
opening-interval departure or measured surface intersection before its next
target. It preserves physics-clock pacing, post-ACK anchoring, deadlines and
the existing safety approvals. Detection at a measured pose may occur after
a deviation; prior profile checks are the preventive part.

The halt generation is captured before localization/planning and checked before
open, each motion segment and each close stage. A halt cannot be erased by a
later `begin_motion`. A callback failure cannot retry without its guards.
An opt-in failed grasp returns with `home_skipped`; automatic retry/re-home
would have no validated recovery geometry. The open-finger gate stops at the
explicit close phase; closure verification is unchanged.

The scene remains the original observation. Checking a newer q or epoch does
not refresh its geometry or prove neighbors stayed still. This gate cannot
certify occluded surfaces, dynamic obstacles, all robot colliders or continuous
trajectory safety.

## Offline evidence

The [independent replay](evidence/observed-finger-gate/observed-gate-replay05.json)
checks native04's recorded joint paths against **new passive05** depth and
production detector/SAM masks, with no ground-truth object geometry. Every
historical actuator target matches the reconstructed nominal profile within
2.3e-16 rad. Across 10 object/view/round combinations, home/pregrasp had no
observed conflict; all six green descents passed this gate and all four orange
descents were rejected. This is a counterfactual replay on a new observed
initial scene, not a recreation of unavailable native04 full-frame depth.

The replay preserves the evaluated module and geometry hashes. Maximum local
host build time was 0.329 s; maximum phase check was 0.0813 s. These are local
CPU measurements, not Spark timing or physical validation. Source scripts and
input hashes accompany the receipts. A new reviewed physical trial with the
explicit option remains required.
