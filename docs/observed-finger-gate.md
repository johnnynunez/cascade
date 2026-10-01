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

The [independent final-source replay](evidence/observed-finger-gate/observed-gate-replay05-014f016.json)
checks native04's recorded joint paths against **new passive05** depth and
production YOLOE segmentation masks, with no ground-truth object geometry. Every
historical actuator target matches the reconstructed nominal profile within
2.3e-16 rad. Across 10 object/view/round combinations, home/pregrasp had no
observed conflict; all six green descents passed this gate and all four orange
descents were rejected. This is a counterfactual replay on a new observed
initial scene, not a recreation of unavailable native04 full-frame depth.

The [selection replay](evidence/observed-finger-gate/observed-selection-replay05b-014f016.json)
also uses the exact runtime vet functions, real selector and safety harness.
Without the gate it reproduces historical selected joints within 5e-16 rad.
With the gate, green retains candidate 0 in all six views; orange rejects
candidate 0 and selects candidate 1 in all four views. Candidates come from
native04 and the observed scene from passive05; no new planner or robot call
was made. The earlier preliminary replay is retained separately.

The receipts preserve evaluated source/module/geometry and input hashes.
Reported CPU timings are local host measurements, not Spark timing or physical
validation. A new reviewed physical trial with the explicit option remains
required.

The runtime source is frozen at `014f01699800bb9da7b3127e2517047f55890bdf`:
158 focused tests, 146 independent reviewer tests, and the full suite
**2617 passed, 43 skipped, 2 deselected** in 311.05 s. Full-suite HEAD was
identical before/after and the checkout remained clean. The log SHA-256 is
`89eee92cb7080373d2d8483c46f6693c799619dc27c02f04d3d54a939cd41453`.
This final evidence update changes documentation only.

## Native launch05: two cases pass; closing remains outside this gate

On Spark, source `c5baad5` completed the normal native OpenClaw launch with the
explicit gate enabled and camera cadence six. Green cube → green square and
orange → open box both passed the existing physical placement, bilateral lift,
release, home and reset audits. The independent source/owner/trace checker also
passed both cases. Their native `pick_and_place` calls took approximately
239 and 250 seconds respectively; the tool and motion deadlines were unchanged.
The orange selection rejected four candidates before motion.

This is **not complete acceptance**. During the orange closing phase, after the
protected open approach, the neighboring pink cube moved approximately 19.8 mm.
The placement audit does not check this condition. The open approach had left
that neighbor stationary; attribution of the later displacement and a closing
veto are under investigation. No claim of whole-gripper collision freedom follows
from the passing placement audit. The five-object campaign, persistent default
configuration and restart acceptance remain pending.

All three cameras advanced during each case. Across the complete sampled cases,
maximum age at the physics snapshot was 1.785 s for green and 1.944 s for orange.
Transport-inclusive age is a different measurement: one green sample and two
orange samples exceeded two seconds, with maxima 2.002 s and 2.069 s. These
numbers do not change the recorded audit result, but prevent a claim that every
delivered sample was younger than two seconds.

The shipped Chromium page and extension on Xvfb displayed all three live
1280×720 cameras with advancing frame IDs and enabled send/reset controls.
This checks the actual browser, not a GNOME login or a visible Isaac editor.
The normal owned stack was stopped after evidence retention; the browser window
was preserved for a future reconnect check. No reconnect pass is claimed yet.

The [compact native receipt](../benchmark/results/spark-observed-finger-native05-20261001.json)
contains the source, exact trace and proof hashes, per-case checks, both camera
age definitions, UI observations, known closing limitation, and hashes of all
99 retained files. The original passing proof remains unchanged; its archive is
41,509,551 bytes with SHA-256
`b898660f36695e7688b8ed64e63411f1cc8118e87a5dc1b0b2baed308f7a89c0`.
