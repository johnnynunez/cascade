# Isaac camera frames bound to render history

The first visitor case in correction08 completed its motion and reset, but the
camera audit failed: one reset-phase sample contained all three cameras with a
server age of 2.021012694 seconds (client delivery age 2.120163994 seconds).
The two-second threshold remains unchanged. The six-iteration cache schedule
allowed a slow main-thread job to delay the next publication.

Publishing more often alone is insufficient. The SDK can return a prior rendered
buffer after `app.update()`. Pairing that image with current joints, wrist pose,
or a new timestamp would conceal its age and corrupt the robot mask.

## Producer contract

The sensor tick rate and physics step remain unchanged. The bridge requests
readback on its existing `cam_every` iterations, and when at least 0.5 wall
seconds have passed since the **start** of the previous request. Outer updates
and nested reset updates share this anchor. There are no catch-up captures,
extra physics steps, sleeps, renderer toggles, or changed motion deadlines.

Each relevant existing `app.update()` records a bounded private history entry:

- The SDK's raw rational current-time key and a separate simulation time/step.
- A monotonic timestamp taken **before** that update, conservatively including
  the cost of rendering. It is not an exposure timestamp.
- Joints and individual jaws measured after the update, and the wrist camera
  transform actually authorized before it. The latter is not recomputed from
  later joint feedback.
- The producer epoch. Stop/Play, regressing or invalid clocks discard camera
  packets/history and renew this epoch.

Each camera reads its own public SDK `rpFabricTime` and
`IsaacReadSimulationTime` annotators before and after copying RGB, depth and
segmentation. Only an unchanged token with a unique matching history entry can
publish. Generic `ReferenceTime` is unsuitable: with multiple render products,
it advanced while a tick-limited camera's arrays remained unchanged.

The SDK Double-time history path truncates to integer microseconds; render
products can express nanoseconds. Matching uses that exact integer truncation
at the **original** SDK denominator, never a reduced Fraction denominator or
an arbitrary epsilon. Other history resolutions require exact rational equality.
Simulation time is a separate coherence check, never a fallback to current state.

Repeated tokens preserve the entire previous packet, including its timestamp.
Contradictory snapshots poison the key instead of replacing it. Missing or
ambiguous history invalidates that camera's packet; no fresh joint read repairs
an old image. Failures in one camera do not prevent validation of the others.
The wire frame carries `render_reference`, and the client preserves it in
`Frame.capture` and opt-in grasp evidence.

The portable bundle includes and requires `isaac_frame_history.py`. Kitchen
source fingerprints are regenerated with the standard manifest tool.

## Isolated renderer evidence

These were new synthetic Kit processes on GPU1, with no robot, bridge, ports,
external scene, or actuator calls. Controls 01 and 02 preserved native failures
from render-product toggling and a default metadata getter; production uses
neither mechanism. Control03 showed that generic `ReferenceTime` was not a
camera-buffer identity. Control04 used public `rpFabricTime`, cameras created
independently at 20 Hz and continuous ticks, and kept both enabled throughout.

Control04 retained 220 RGB/depth/segmentation archives, whose 660 channel hashes
were independently verified. Its 72 static-marker samples matched their unique
history entries. For 144 dynamic samples, projecting the authored cube's eight
corners using its measured historical physical pose and historical camera pose
reproduced depth and segmentation bounds exactly under the explicit pixel-center
convention. Of those, 138 samples had measured motion and rejected both adjacent
physics-substep hypotheses; six stationary startup samples were inconclusive.
RGB color-threshold bounds differed by up to two pixels and are not claimed to
be exact geometric measurements. Both render modes matched; production retains
the existing tick rate.

[Evidence identities](evidence/isaac-render-history-controls.json) bind the scripts,
receipts and independent review. The raw archives remain in the retained local
renderer-controls directory; this file does not claim that a new camera has
been validated on the kitchen or on a moving articulation.

## Validation limits

Unit tests cover delayed/repeated/contradictory frames, malformed clocks,
regression and epochs, conservative timestamps, per-camera failures, unchanged
reset step counts, no catch-up and isolated bundle loading. Synthetic controls
establish the tested render/history relationship, not a kitchen performance or
continuous freshness guarantee. A blocked update can still exceed two seconds;
its real age must remain visible and fail the existing audit.

The candidate also includes the separately reviewed truth-reader correction:
constructing a classic `RigidPrim` for observation sets
`reset_xform_properties=False` and `prepare_contact_sensors=False`, avoiding USD
authoring during observation. Camera history does not change motion, placement,
retry, or safety thresholds. Kitchen validation, five-object acceptance and an
acceptance restart are still pending for this candidate.

## nvblox payload masks

With contact tracking enabled, the completed-update snapshot also owns the
bilateral-contact paths, the set of scene-prop paths and any contact-read error.
Mask publication uses that historical evidence alongside the matching pixels,
joints and camera transform. A later release or new attachment cannot change
the mask of an older render. A historical contact error leaves the frame without
a usable robot/payload mask, even if the current sensor has recovered.

Focused integration tests cover both directions of attachment changes, a
historical contact failure and a newer failure while an older valid packet is
retained. These tests do not establish contact timing or camera performance in
the new kitchen scene; those remain part of physical validation.
