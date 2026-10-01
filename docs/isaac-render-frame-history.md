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
seconds have passed since the **start** of the previous scheduled cycle. Each
cycle requests all cameras. A camera remains pending until it publishes a new
bound packet: duplicate tokens and readback failures keep it eligible on the
next existing update, without another half-second cooldown. Completed cameras
are not polled or copied again for another camera's retry; the next nominal or
wall cycle requests all cameras again. Outer updates and nested reset updates
share the anchor and pending set. There are no catch-up captures,
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

## Publication scheduling regression (1 October 2026)

Campaign correction05 on `0e23870` stopped after its orange camera audit failed.
Placement and reset were physically confirmed. Sample 1989 contained three
correctly bound camera packets aged 2.071459983 seconds at the server; client
delivery age was 2.165724130 seconds. This was during the subsequent
`world_state` response, 31.099 seconds after the reset order finished. The
broader witness phase named RESET also includes world-state and scene queries.

The previous scheduler restarted its half-second cooldown even when a poll
found only repeated render tokens. A deterministic test reproduces this defect:
a slow main-thread job, an unchanged render token, and a new token available
on the following existing update. The old source delays that publication;
pending-camera scheduling reads it without adding a Kit update. The actual
campaign witness did not retain the render token available at each poll, so
this is a reproduced mechanism compatible with the failure, not proof of its
sole cause. The long witness RPC also includes observation/encoding/transport
work that was not separately timed. The campaign remains failed.

Regression tests exercise the real scheduler/loop and the real packet producer
with deterministic SDK boundaries. They cover independent camera completion,
duplicate packet identity and timestamps, missing or contradictory history,
changing tokens during readback, nested reset scheduling, and epoch clearing.
The kitchen source manifest changes with the bridge. The renderer history
helper, collider geometry, camera settings, physics steps, sensor tick rate,
and two-second acceptance threshold do not change. A blocked main thread or
renderer can still exceed that threshold; physical performance and acceptance
on the new source require a new run.

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
retry, or safety thresholds. Kitchen validation, five-object acceptance and restart are separate stages;
[current project status](PROJECT_STATUS_20261001.md) records their source pins.

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

## Exact SDK double encodings

The first ARM admission of `b31b021` failed closed after 137 complete frame
observations. A separate passive diagnostic failed after 409 observations and
captured the cause: history step 2058 stores SDK time `17150000 / 1000000` and
simulation time `17.15`, while all four render products report
`17149999999 / 1000000000`, with simulation annotator `17.149999999`.
The diagnostic response retains only its final 8000 characters; seven complete
history entries and the four render tokens survive. No native manipulation was
started after either passive failure.

Both integers are reproduced by truncating the corresponding multiplication
of the same IEEE double. The resolver additionally admits this representation
only for the observed raw precisions (SDK microseconds and render nanoseconds),
when **both** forward encodings of a stored simulation time match the raw keys.
An alias also requires the annotator to equal the render rational converted to
double, as the SDK multitick `getSimulationTimeAt` implementation specifies.
All ordinary and alias candidate identities are combined before ambiguity and
coherence checks. Multiple candidates, poisoned entries, incorrect raw SDK
values, epochs, or alias annotators remain rejected. There is no nearest-time
search, numeric epsilon, interpolation, or replacement with current state.

The retained ARM failure replays from four rejections to four bindings to the
original step 2058, preserving its original timestamp and state; all 220 earlier
synthetic control rows remain bound. The receipt in
`docs/evidence/isaac-render-time-encoding.json` links the failed runs and replay.
This offline result does not itself establish corrected ARM availability or
native campaign acceptance; those remain separate candidate validations.

## Subsequent ARM admission and integration

The retained 09d passive admission on `aedfba1` exercised both exact SDK encodings
on all three cameras: 551 samples per camera, six alias observations each,
367 duplicate packets and 183 advances. Maximum delivered age was 0.431 seconds.
This was a read-only camera admission, not a manipulation campaign. The earlier
09c run retained its aggregate coverage failure despite 600 valid packets;
no failure was relabeled as success.

The combined MAIN producer uses the same resolver/readback contract and adds
historical nvblox contact-mask state when that mode is enabled. Later passive
admission may combine source-identical alias coverage with a current-source
whole-packet observation; each component and its limitations must remain
explicit. [Project status](PROJECT_STATUS_20261001.md) identifies the accepted
source and distinguishes passive admission from native physical results.
