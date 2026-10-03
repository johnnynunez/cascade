# MuJoCo release withdrawal — 2026-10-03

The SO-101 C-runtime grasp ceiling also capped the old post-release retreat.
Opening therefore left the tool beside the placed cube; returning home struck
it. The unmodified two-pick test reproduced a 70.582591 mm final blue-cube
error against its 60 mm limit. The 24 installer assets were byte-identical to
the locally materialized assets; changing only MuJoCo 3.14 to 3.12 reproduced
the failure within 3 µm. This is a motion/geometry defect, not an asset-version
explanation or a reason to change the verifier.

The optional `mj_release_tool_body` capability names a collision subtree in
the connected MuJoCo C model. Its adapter previews transport with the measured
payload transform in detached `MjData`, then plans an empty-tool escape before
release. It rechecks actual geometry after transport and after opening. No live
qpos, body pose, gain, contact or gravity is modified for validation.
The profile also names the fixed articulation root and complete joint mapping,
plus the horizontal support-plane pose. `floor` is required; the existing
generated wrapper's coincident `demo_floor` is optional and validated equally
when present. Both remain in the physical scene. Any other collidable static
geometry, moved root or mocap model is rejected; this is not a general obstacle
planner. The plain `scene.xml` with only `floor` passes model admission.

An IK lift is considered only if the complete pose is feasible. Otherwise the
adapter considers the bounded joint-first/joint-last corners derived from the
declared home configuration, through the existing waypoint and safety checks.
Orientation may change after opening. During the initial escape, native convex
collider distances require no new penetration; existing intended tool/payload
contact may only separate. The endpoint and home route additionally require
the existing 20 mm safety clearance around conservative collision AABBs. A
bounding sphere and vertical fall interval cover the released object's possible
orientation and fall for that subsequent route. This is sampled planning, not
a continuous collision certificate or a bound on arbitrary bouncing motion.

The same pose/geometry is read again before the actual retreat and each home
segment. Actual jaw opening must reach 98%; the original halt generation is
retained for opening and streaming. Cancellation, missing geometry, non-finite
distances, unsupported collision types or an occupied transport path refuse the
operation. A refusal before carry preserves the grasp and skips recovery/home.
Pending empty-tool paths are bound to their arm and cannot carry a new object.
The common SDK-independent harness retains this debt before opening, including
when opening or withdrawal raises. Its thread-local adapter scope admits one
exact gripper or joint command; later samples belong to that same transmission,
not to another nested command. Direct SafeArm calls, raw two-stage closing and
other threads cannot borrow the scope. Reads and priority stop remain available.
Only checked home, or explicit verified reset recovery, removes the debt.
An explicit scene reset may create a new context only after the stop is
released, with an empty/open tool, the same backend/model bindings and fresh
geometry. It does not renew the retained plan or reuse its target. Only actual
spawn coordinates, zero free-body velocities and a fresh observation permit
clearing that arm's pending state. Failed reset/camera checks retain it.
A cancellation after the final observation also retains it. The earlier
174-test snapshot is preserved with its missing-actuation-barrier limitation;
it is not used to claim this final fencing behavior.
The three-second budget covers each bounded geometry calculation. The initial
carry preview is discarded; a fresh calculation follows actual descent before
opening, so normal transport time does not renew or exhaust a motion permit.
No arm name is recognized in the skill, and no SDK import becomes mandatory.
Warp has no shared C-world observation channel and fails closed when this
explicit capability is requested; this checkpoint does not admit Warp parity.

The [source-bound receipt](../benchmark/results/mujoco_release_withdrawal_20261003.json)
records **213 passing tests in 78.26 s**, including actual CPU contact dynamics
and rendered observations. The first cube finishes 1.865 mm from the destination
after home. Its measured displacement during home remains below 1 µm; the second occupied-point attempt leaves both objects exactly at their
pre-carry poses. Waypoint-end contact samples found no loaded arm/red-cube
contact during home; they do not cover every solver substep. Source files and
protected user stores remained unchanged throughout the run. Rendering used
Mesa EGL/llvmpipe; CUDA was disabled.

The original two-pick success test and its 60 mm requirement remain unchanged.
The corrected first placement exposes a second issue: requesting the same
exact point again is occupied. This checkpoint refuses that attempt before
transport. A separate explicit destination-area contract is needed to select
a different free location without silently changing `place_at(x,y,z)`.
The earlier late refusal and its misleading legacy proximity confirmation
remain recorded in `physical-01`; they are not rewritten as successful trials.

Primary API semantics: [MuJoCo `mj_geomDistance`](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)
provides signed native collider distance. The adapter requires native CCD and
rejects unsupported geometry/contact-pair overrides. All forward kinematics
operate on separate data; the ordinary driver remains the only physics writer.
