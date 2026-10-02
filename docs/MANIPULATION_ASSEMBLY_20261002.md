# Manipulation and physical assembly — 2 October 2026

[PR #65](https://github.com/johnnynunez/cascade/pull/65), branch
`feat/manipulation-assembly-runtime`, based on `1271d52d09feaa0edb4652d39a9ab21d69e4a686`.
Production integration at `c07d9225f93ebe7cc1be5d65079ebcc141b60117` passed the
full software suite. The native combined manipulation episode used runtime
`f2b2186317a6eb6278bf7a57bf039c912e64dc18`; the later assembly commit changes
separate Newton experiment files. The existing canonical checkout and services
from other sessions were left untouched. These results are local branch evidence.

## OVRTX and cuMotion in normal manipulation

The optional `isaac_cumotion` profile now wires `RuntimeMotionPlanner` into the
ordinary arm/skill path. Its model, joint names/signs, base and tool must match
the selected arm. SafeArm vets each curve and the Isaac executor streams the
native shape with uniform slowing, physical-clock feedback and existing safety
checks. Contact paths require a linear TCP corridor. Missing SDKs, bad bindings,
unsafe trajectories, cancellation and measured start drift reject execution.

OVRTX now supplies live bridge cameras from completed physics-body tensors.
A separate renderer process owns a private USD scene. Pixels, depth, robot and
prop masks, camera pose, joint state and contacts are bound to the same physical
snapshot. Exact duplicate frames retain their complete identity. No live Kit
USD stage is edited to prepare the renderer's scene.

The joint native episode used PhysX GPU, SDK cuMotion 1.1.0, OVRTX, real required
GraspGen-X and the observed-finger gate. Occupancy was explicitly disabled.
The green object completed grasp, lift, transport, placement and return home
in **126.19 seconds**: eight task curves, 529 actuator targets and 2,545 checked
safety edges; one additional native park curve succeeded during shutdown.
There were no retained held/provisional/contact/carry states.

Independent observations after the task recorded 24 advancing physical samples
with open jaws and stable position; 12 further samples after park confirmed
the same result. Final object position was `(0.19958, -0.15699, 0.04000)` m,
23.51 mm from the configured drop-zone center. The support conclusion uses
geometry and stationary body poses, not a table force sensor. Source hashes
were unchanged during execution; the producer's loaded-source hashes matched.

![Actual OVRTX video: start, grasp and final placement](../benchmark/results/images/ovrtx-runtime-green-20261002.png)

- [OVRTX implementation, native evidence and failure history](OVRTX_RENDERER.md)
- [cuMotion integration, configuration and safety contracts](CUMOTION.md)
- [OVRTX receipt](../benchmark/results/ovrtx-runtime-manipulation-20261002.json)
  and [cuMotion receipt](../benchmark/results/cumotion-runtime-20261002.json)
- [Local captured video](../runs/ovrtx-native-05/pick-place-green-01/side.mp4)

The retained pink-object run is a failure: it lifted the object about 40.5 mm,
but joint 6 had 0.048762 rad of following error against the unchanged 0.045 rad
settling tolerance. Shutdown correctly skipped park/open with a possible load.
Only its owned simulation process was then terminated; the jaws were not opened
aloft. Other retained failures include target-mask border rejection and native
contact trajectory/stability vetoes. One green-object success does not establish
success for every object, the full kitchen campaign, Spark or hardware.

The retained-load shutdown guard is specific to backends that explicitly
preserve drive state when disconnecting. Isaac only closes its client transport;
hardware retains the existing park-before-disconnect policy. This avoids a
Feetech regression found during review: skipping park and then disabling torque
would leave the arm and payload unsupported. The capability is checked per arm,
including mixed rigs, and inspecting an unused LazyArm never connects it.

## Physical threading, seating and retention

A separate SO-101 experiment drives a mounted motorized hex socket around a
free Factory M20 nut on a fixed bolt. Original threaded meshes use Newton SDF
collision. Motion comes from solved contacts and the robot/spindle actuators;
there is no helical constraint, nut attachment, injected nut wrench or object
pose assignment after initialization.

The final fixture adds a 3 mm annular spacer and six passive compliant socket
inserts. The controller completed **15.365321 turns and 38.154768 mm of axial
advance**, observed loaded nut/spacer contact for 0.5 seconds at the illustrative
0.05 N·m motor limit, then commanded exactly zero motor torque and retained the
seat for two seconds. The arm servos and mounted socket remained engaged;
this is zero spindle-motor torque, not tool withdrawal or an unpowered robot.
Thread/tool contact covered 99.4097% of angular travel
when measured at every physics substep. Tool/fixture and tool/mount interference
were absent. Before seating, the maximum pitch residual was 25.64 µm; including
the loaded seating phase it was 0.258536 mm against the predeclared 0.3 mm bound.

The same final geometry with zero motor drive advanced only **0.73 µm** and
correctly refuted the commanded threading task. A half-timestep short probe
confirmed threading with 100% contact; it was not a second full seating run.
The earlier bare-head stall, before adding the spacer, remains a failed seating
experiment because the nut had not reached a real support surface.

This is a mounted-tool, initially engaged simulation fixture. Autonomous tool
pickup, screwdriver-bit engagement, hardware tightening and calibrated preload
are not established. Ordinary `turn_screw` still returns physical verification
as unknown; it now rejects failed engagement IK and incomplete commanded strokes.

- [Implementation, pinned assets and reproduction commands](FACTORY_THREAD_CONTACT.md)
- [Measured seating receipt](evidence/factory-thread-contact/final-seating.json)
- [Local 39-second solver-state replay](../../screw-contact-evidence/final-seating/threading.mp4)

Both videos were inspected at the beginning, middle and end. The fastening
video renders saved solver poses without advancing the physical simulation.

## Trial 12 and its follow-up

Historical trial 12 on `d1d54dcf078b3efefd9542d46947c77bd29afe7e` remains failed.
Green-cube placement passed its complete physical audit. The orange failed
before actuation when the shared **3-second route-preflight deadline** expired;
the five-object campaign was not completed. A separate terminal ownership check
also lacked `gateway_child`. Neither failure is rewritten by later experiments.

The route change moves existing endpoint occlusion rejections before expensive
route sampling. Every accepted candidate still needs every original check;
ranking, collision geometry and the 3-second budget are unchanged. An offline
replay of the retained frame selected the same candidate in 2.833772 seconds
before and 1.831435 seconds after. Both replay versions passed on this host:
this measures saved work, not reproduction of the old native timeout.

Fresh skill-path tests use isolated port 8733 and independent physical/camera
auditing, not the launcher's LLM/gateway acceptance. At 1280×720 the orange
passed route selection and moved toward pregrasp, but streaming exhausted the
unchanged 120-second motion wall budget. The measured RTF was about 0.082,
while all three cameras remained fresh. The orange was untouched; reset then
failed because the timed-out client had closed its socket. The first attempt
also retains a separate startup clock-RPC failure before any actuation.

At **640×360**, orange and green cube subsequently passed independent physical
pick/place/home/reset audits, with 103.921/100.528 mm lifts and 9.724/6.076 mm
final XY errors. Each settled for 0.5 physical seconds across 31 samples.
The diagnostic first warmed existing read-only readiness and exact-label
perception; no warmup motion occurred and task images were freshly acquired.
The earlier cold-perception rejection remains recorded. Required learned
GraspGen-X ran on CUDA; occupancy was disabled. The launch command enabled the
observed-finger gate, but this diagnostic did not retain a separate runtime
flag/hook receipt. That declaration is distinct from the independently measured
bilateral contact and physical outcome.

Their skill durations were **483.72 and 438.83 seconds**, both above the real
host's 300-second budget. This source-stable, two-case diagnostic used a mock
LLM and direct skills. It does not establish real-host/MCP strict proof2,
five-object campaign, restart or original-resolution acceptance. The
[640 audit](evidence/manipulation-assembly/trial12-native-640.json) binds the
receipts and inspected before/placed/reset stills; no MP4 was captured there.

See [the full historical/follow-up account](LOCAL_RTX_VALIDATION.md#native-trial-12-and-route-preflight-follow-up)
and the [audit receipts](evidence/manipulation-assembly/).

## Software verification and evidence boundaries

- Full suite at `c07d922`: **3,797 passed, 48 skipped, four deselected** in
  312.57 seconds, with tracked files unchanged during the run.
- The later fixture-import lint cleanup passed its 12 trajectory tests.
  Changed Python files pass Ruff `F,E9`; unrelated pre-existing lint findings
  are outside this result.
- Fastening checks passed 59 tests across threading, seating and `turn_screw`.
  Native SDK/physics results above are separate from unit-test doubles and skips.
- Initial PR CI passed the main suite on all three platforms but failed portable
  bundle checks because the new OVRTX helper/identity files were omitted.
  The corrected inventory also includes the optional cuMotion XRDF/provenance;
  [62 packaging checks passed](evidence/manipulation-assembly/portable-bundle.json),
  including SDK-free profile resolution and rejection of missing dependencies.
- The shutdown correction passed 119 focused checks. Actual Feetech driver
  register I/O against an in-memory bus confirms park targets precede torque-off;
  Isaac, lazy Isaac and mixed-rig cases preserve their distinct teardown rules.
  This is software regression evidence, not a new hardware execution.
- New simulators used owned launchers and private ports. Termination verifies
  process birth identity and preserves receipts; it does not close other sessions.

The [regression receipt](evidence/manipulation-assembly/regression.json), native
component receipts and asset manifests identify their actual tested sources.
Raw captures and videos remain at the local paths recorded in those manifests.
The changes and compact evidence are published in PR #65. No merge or remote
deployment is claimed by this report.
