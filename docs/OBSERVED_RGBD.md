# A camera-only RGB-D domain

An explicit `mobile_rgbd` sensor provider connects the native mobile bridge's
registered RGB-D capture to `RobotRuntime` and MCP. A robot profile containing
only this sensor domain builds no arm, locomotion controller, grasp planner or
map. Its resource advertises `rgbd`, has no controller/writer, and remains
`admission: unvalidated`. A valid observation is not physical task acceptance.

This first producer is the existing Newton backend's **static, world-mounted
overview camera**. It is not a camera attached to MicroDuck's head, and does not
add a tactile driver, visual SLAM, collision map or navigation policy. Other
camera producers need their own observed identity, capture and calibration
contract; a generic legacy `CameraBase` receipt counter cannot supply it.

## Producer and identity

Add `--camera-rgbd` to an explicitly owned invocation of
[`isaac_microduck_bridge.py`](../scripts/isaac_microduck_bridge.py). All existing
bundle, policy, solver, wall-time and shutdown requirements still apply. There
is no new default profile or simulator launch in the sensor reader.

The option attaches `distance_to_image_plane` alongside `rgb` to the same
experimental `CameraSensor` render product. Camera initialization reads the
actual USD perspective optics, resolution and optical-to-world transform. The
producer accepts a meter stage, centered undistorted perspective optics, and
static USD attributes only. Even a single authored time sample is rejected:
it can override a different default without `ValueMightBeTimeVarying()` being
true. Calibration is read again around every capture and must retain the
startup digest.

The effective model identity includes the RGB-D calibration record and its
SHA-256, render product, annotator and new source bytes. Previously admitted
model hashes cannot be reused. A future native run must separately validate
this recipe before anyone describes its render or physical behavior as
accepted. Current validation consists of CPU transport/capture doubles and
an OpenUSD-only geometry check; no RGB-D GPU episode is claimed here.

On the producer's main thread, rendering holds the physical step/time fixed.
Each AOV read is bracketed by the same render product's `rpFabricTime` and
`IsaacReadSimulationTime`. Both channel references must agree and match the
completed step. Changed references, absent depth, changed calibration and
uncontrolled physics advancement reject the capture. Signal checkpoints fence
the SDK reads, including an SDK callback that absorbs a signal exception.

These references describe the render product during readback. The installed
SDK's RGB/depth arrays provide no independent per-pixel timestamp. The software
checks do not establish native asynchronous AOV alignment on their own.

## Configure the passive consumer

Use the **new** run's `model-identity.json` and `BRIDGE_LISTENING.json` after
independently checking its source, model and lifecycle evidence. The following
example generates an explicitly pinned camera-only profile. It does not grant
physical admission or start a producer:

```python
import json
from pathlib import Path
import yaml

run = Path("/absolute/path/to/new-owned-rgbd-run")
model = json.loads((run / "model-identity.json").read_text())
ready = json.loads((run / "BRIDGE_LISTENING.json").read_text())
hello = ready["hello"]
assert model["model_identity_sha256"] == hello["model_identity_sha256"]
calibration = model["recipe"]["native"]["rgbd_camera"]
profile = {key: hello[key] for key in (
    "robot_id", "source", "epoch", "engine", "device", "asset_sha256",
    "policy_sha256", "model_identity_sha256", "physics_dt", "policy_dt")}
profile.update(support_contract=model["support_contract"],
               bridge_host="127.0.0.1", bridge_port=ready["port"], timeout_s=0.2)
robot = dict(version=1, robot_id=hello["robot_id"], domains={"sensing": {
    "kind": "sensors", "max_age_s": 0.5, "read_timeout_s": 0.25,
    "providers": [{"id": "overview", "kind": "mobile_rgbd", "profile": profile,
                   "camera": "overview", "max_pixels": 640 * 480,
                   "calibration_sha256": calibration["calibration_sha256"]}]}})
with Path("configs/robots/overview_rgbd.yaml").open("x") as output:
    yaml.safe_dump(robot, output, sort_keys=False)
```

Run the ordinary MCP server with `CASCADE_ROBOT=overview_rgbd`. Discovery and
`list_resources` do not connect. `sensing.read_sensor(sensor_id="overview")`
opens an independent `hello(role="reader")` socket and requests the completed
capture. Only `hello` and `frame` are sent; reading cannot solve physics, acquire
control or construct an actuator. Configure wall-time and age budgets explicitly
for the intended producer cadence. There is no fallback to old pixels or to a
different camera when those limits fail.

The same admitted capture can feed the [spatial observation domain](RGBD_SPATIAL_OBSERVATIONS.md)
through its exact returned `capture_sha256`, epoch and sequence. Sharing it does
not acquire another frame or refresh its original age.

The initial native producer requires its fixed world camera. Do not label its output as
a robot-mounted sensor: `world_from_camera` maps the optical frame to `world`,
not to a head, torso or robot base. A moving mount requires a new producer
contract for its measured pose and a distinct calibration policy.

## Opt-in moving-capture contract

`RgbdFrameCache` and the passive reader also support calibration **version 3**,
which replaces `world_from_camera` with `rig_frame_id`, `rig_from_camera`,
`mount_position_error_m` and `mount_angular_error_rad`. All other calibration
fields, including the named world frame, remain explicit. `rig_from_camera` is
a rigid column-vector transform from optical coordinates into the named rig;
the two error bounds are nonnegative meters/radians or `null` when unknown.
The calibration hash and effective model bind this fixed mount. It is not a
changing joint transform.

For this calibration, `publish(..., capture_pose=...)` requires a pose record
with exactly `epoch`, `step`, `sim_time_s`, `model_identity_sha256`,
`world_frame_id`, `world_from_rig`, `position_error_m`, `angular_error_rad` and
`render_reference`. The identity and render reference must match the completed
RGB/depth capture. This emits packet version 2. The reader checks these bindings
again and the sensor envelope retains a typed `capture_pose`; its computed
`world_from_camera = world_from_rig @ rig_from_camera` belongs only to that
capture. Unknown bounds stay unknown; known bounds include the rotating lever
arm. Neither a new pose nor a new receipt can rejuvenate old pixels.

The existing static calibration version 2 / packet version 1 remains unchanged
and refuses an injected dynamic pose. The shipped Newton launcher still uses
that static producer. The new moving path has CPU transport/geometry evidence
only; no Fabric pose adapter, moving native camera, articulated mount or physical
calibration has been admitted. Such an adapter must observe the actual rig pose
at the same render completion, not read USD defaults or interpolate an unrelated
controller sample. No camera, depth, map or navigation acceptance transfers
from the earlier static captures.

## Packet semantics and failure behavior

- Color is lossless RGB8. The legacy JPEG/BGR camera path keeps its existing wire
  schema and remains available independently from the same cached capture.
- Depth is aligned float32 little-endian **optical-axis distance in meters**,
  not Euclidean range. Zero explicitly marks invalid depth. Far-clip/nonfinite
  native pixels become zero; negative depth is an error. No depth estimator is
  invoked to fill absent data.
- Pixel intrinsics and the rigid `world_from_camera` matrix come from the
  capture's pinned calibration. Optical axes are X right, Y down, Z forward;
  the USD camera's Y-up/-Z-forward convention is converted explicitly.
- Native calibration version 2 declares `pixel_center_offset_uv: [0.5, 0.5]`.
  USD K has the raster boundary as its origin: depth array element `[v,u]`
  corresponds to `[u+0.5,v+0.5]`. K itself is unchanged. The offset belongs to the
  calibration hash, effective model recipe and immutable sensor payload. The
  native decoder rejects version 1, which omitted this convention; it never
  relabels an old packet or reuses its model admission. Generic legacy RGB-D
  payloads remain byte-compatible and retain their integer-center convention.
- Robot, source, engine, device, asset, policy, model, epoch and camera must
  match. The complete calibration must hash to the configured pin on every read.
- Producer capture age includes render/encoding time. The client adds its RPC
  round trip and elapsed local time. Reconnection retains epoch and replay
  watermarks. Stale or repeated captures cannot obtain new authority by acquiring
  a new receipt timestamp, and a new episode requires a new reader.
- Pixel count, encoded size and zlib output are bounded before image allocation.
  Truncated streams, trailing compressed streams, malformed matrices/nonfinite
  depth, missing capability and exceeded deadlines fail closed.
- Captures expose immutable byte buffers and deep-frozen metadata with defensive
  copies. The sensor hub still enforces its original packet/history limits.

Each opt-in native capture is also written as `frames/overview_<step>.rgbd.json`
beside the existing JPEG and frame log. These are retained evidence, not a replay
service: loading a historical packet does not make its original clock fresh.

A bounded native run on the preceding source captured five RGB-D frames and
admitted four through TCP/Hub/MCP. Offline ground projection exposed a half-pixel
mismatch: 0.416–0.512 mm Z residual at integer indices, below 0.1 micrometre at
pixel centers with the same recorded K, pose and depth. The version-2 correction
is covered by CPU tests of retained numeric samples and the real MCP path, not a
new native render or per-AOV synchronization test. A fresh native recipe/model
must be admitted for this changed producer; the historical packets stay intact.
The new extrinsic fields are emitted only for calibrated RGB-D payloads; older
`RgbdPayload` serialization without extrinsics remains unchanged.

The older mobile IMU/RGB `calibration_id` remains an optional declared reference;
it is not upgraded into independently observed calibration by this change. The
new RGB-D provider deliberately uses `calibration_sha256`, which checks the
producer's complete calibration record.
