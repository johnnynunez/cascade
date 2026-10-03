# Retained RGB-D surface observations

An opt-in spatial domain consumes a capture already admitted by `SensorHub`.
It projects a selected integer pixel with that capture's depth, intrinsics and
optical-to-world transform, then records a `LandmarkObservation` in the existing
`SpatialMemory`. The ordinary composed runtime and MCP expose this path without
an arm, locomotion controller or a second camera connection.

The result is an **observed surface point**. The caller supplies the label and
observation ID as annotations. They are not a verified object class, tracked
identity, object center or object pose. Confidence and geometric uncertainty
are explicitly unknown. The domain supplies no occupancy grid, free-space
inference, SLAM, route planner or `go_to`; remembered points never admit motion.

## Configure and use

First configure the calibrated `mobile_rgbd` provider described in
[Observed RGB-D](OBSERVED_RGBD.md), using a new producer's actual model and
calibration identity. Add this entry to that robot profile's `domains` mapping:

```yaml
spatial:
  kind: spatial
  rgbd:
    sensor_domain: sensing
    sensor_id: overview
    map_id: room_observations
    world_frame_id: world
    capacity: 128
```

`sensor_domain` names an explicit `sensors` domain in the same robot profile;
`sensor_id` names its declared RGB-D provider. Declaration order is immaterial.
The provider's robot identity and modality must match, and its calibration must
have an explicit SHA-256. The world frame must exactly equal the capture's frame.
Neither the builder nor catalog discovery opens a camera or constructs an arm.
The new memory resource has no controller or writer; physical sources remain
`admission: unvalidated`. Its tools declare the sensor resource dependency.

Use the same MCP server session for both calls:

1. Call `sensing.read_sensor` with `{"sensor_id":"overview"}`. Besides the
   original observation, the result includes `capture_sha256`, which hashes its
   complete immutable envelope and payload, including original receipt and age.
2. Call `spatial.annotate_pixel` with the returned epoch, sequence and digest:

   ```json
   {
     "epoch": "<exact returned epoch>",
     "sequence": 42,
     "capture_sha256": "<exact returned SHA-256>",
     "pixel": [320, 240],
     "observation_id": "selected_surface_1",
     "label": "possible cup"
   }
   ```

   The sequence and pixel above are illustrative, not admitted fixture values.
   Select an in-bounds pixel from the returned capture with nonzero depth.
3. Call `spatial.recall` with that exact epoch and the annotation label. Results
   are explicitly historical search hints. They retain their capture stamps and
   report current local age and whether that age remains within the sensor's
   configured bound in a separate `capture_ages` mapping keyed by observation ID.
   Entries and their content digests stay unchanged. A stale hint does not
   become a fresh observation on recall.

An annotation contains the camera-frame point, world-frame point, capture stamp,
transform used, capture SHA, source/model identity, epoch, sequence, calibration
digest, pixel, depth, K and full extrinsic matrix. Pixel coordinates are ordered
`[u,v]` and depth is optical-axis distance, in meters. Projection solves
`K p = [u,v,1] * depth`; the capture's column-vector transform maps `p` into the
named world frame. No Euclidean-range conversion or pixel rounding is implicit.

## Fan-out and lifetime

`SensorHub.retained` selects an exact admitted capture by sensor, epoch, sequence
and SHA. It returns the same immutable object after checking its descriptor
binding and current age. It never calls the provider, changes the replay
watermark, refreshes a receipt, or substitutes the newest frame. Multiple spatial
consumers can therefore use one admitted capture. A direct repeated provider
read remains a replay error, including attempts to restamp an old capture.

The hub's count and byte limits still govern capture retention. An evicted,
stale, foreign-epoch or differently hashed capture cannot be annotated. The
consumer checks age again after projection and before inserting memory. Each
annotation resolves its own capture-bound `FrameTree`, so two retained captures
may be consumed in reverse order without using a later transform for the older
image. The original capture time is preserved in both observation and transform.

This first consumer requires a fixed world camera: dimensions, K, frame IDs and
extrinsics must remain equal throughout the domain's epoch. A changed calibration
or epoch requires a new domain. A camera attached to a moving robot needs a
separate measured-pose contract; these points do not provide that contract.

The memory stores at most `capacity` annotations (1–1024) and rejects duplicate
observation IDs or a full store. Returned dictionaries are defensive copies.
Stop/reset does not clear identity, replay or annotation history. Closing the
spatial consumer does not close a shared provider; the sensor domain owns that
lifetime. The synthetic replay spatial profile and its four original tools
remain available independently.

## Validation limits

CPU tests exercise actual loopback TCP capture delivery and ordinary MCP
dispatch, passive catalog construction, exact point geometry, shared and
reverse-order captures, age/epoch/model/calibration failures, invalid pixels and
zero depth, changed transforms, bounded memory, defensive copies and unchanged
replay rejection. No simulator, native RGB-D episode, hardware camera, object
recognition or physical navigation is validated by these tests. The producer's
native AOV alignment and model admission limits in [Observed RGB-D](OBSERVED_RGBD.md)
continue to apply.
