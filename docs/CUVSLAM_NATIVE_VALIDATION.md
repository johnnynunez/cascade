# Native cuVSLAM RGB-D provider validation — 2026-10-04

The optional provider built and ran against cuVSLAM **17.0.0+b405f13** on
Linux x86_64, Python 3.12.13, CUDA Toolkit 13.4.92, driver 615.71.09 and one
RTX PRO 6000 Blackwell. The provider code was `ac9594e5a2ff80b3d2f93105160af7adeee3535f`.
The SDK source was `b405f132b8fb1d861a570f3aea64c2c5d4b59525`.

The [complete receipt](evidence/spatial/cuvslam-native-20261004.json) records
the installed extension and linked-library SHA-256 values, the image-generator
and validation-script hashes, calibration, every capture identity and all
returned poses. The native worker uses the production `CuVslamSpatialDomain`
and `SensorHub`; the SDK and its process are not mocked.

## Build recovery

The interrupted build failed because cuNLS `sparse_matrix.cu` used
`thrust::make_tuple` without including its defining `<thrust/tuple.h>`.
The [retained one-line patch](evidence/spatial/cunls-thrust-tuple.patch) supplies
that header. It applies to the cuNLS `Release_07_13_2026` archive selected by
the pinned SDK, SHA-256
`23b2917ae3903e6a688edb1652e40202d314527cd7fa9db68c762f0429375f77`.
No dependency version or algorithm changed; cuNLS remained enabled.
The complete resumed `cuvslam` target and Python binding both built successfully.

For a fresh isolated build, use the pinned SDK checkout and a Python 3.12
virtual environment. Set these paths to the corresponding directories:

```bash
export CASCADE_CHECKOUT=/absolute/path/to/cascade
export CUVSLAM_CHECKOUT=/absolute/path/to/cuVSLAM
export CUVSLAM_NATIVE_BUILD=/absolute/path/to/native-build
cmake -S "$CUVSLAM_CHECKOUT" -B "$CUVSLAM_NATIVE_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DUSE_CUNLS=ON -DCUVSLAM_BUILD_SHARED_LIB=ON -DUSE_RERUN=OFF
patch -p1 -d "$CUVSLAM_NATIVE_BUILD/_deps/cunls-src" \
  < "$CASCADE_CHECKOUT/docs/evidence/spatial/cunls-thrust-tuple.patch"
cmake --build "$CUVSLAM_NATIVE_BUILD" --target cuvslam --parallel 8
CUVSLAM_BUILD_DIR="$CUVSLAM_NATIVE_BUILD" uv pip install "$CUVSLAM_CHECKOUT/python"
uv pip install -e "$CASCADE_CHECKOUT" scipy pytest
CUDA_VISIBLE_DEVICES=0 python "$CASCADE_CHECKOUT/scripts/validate_cuvslam_rgbd.py" \
  --sdk-checkout "$CUVSLAM_CHECKOUT" --output /absolute/path/to/native-replay.json
```

The architecture above targets the measured x86 Blackwell GPU; it is not a
Spark/Jetson binary or build result. Apply the dependency patch only once.
The SDK's own CMake also applies its existing minimum-CMake compatibility edit.
Original and resumed build logs remain in the local
`cascade-lab/SPATIAL_CUVSLAM_20261003/` workspace.

## Measured exercise and limits

The SDK's deterministic upstream image generator supplied **12 synthetic
640 × 480 RGB-D frames**, depicting a camera moving toward a textured plane.
All 12 production provider calls returned finite native estimates. Final
translation error against the generator's known displacement was **0.01178 m**
(the explicit smoke-test gate is 0.1 m). Warmup took 0.1984 s and the slowest
tracking call took 0.03074 s in this run. These are local diagnostic timings,
not a throughput benchmark or deadline qualification.

Every result retained unknown position/angular uncertainty and
`physical_admission: false`. A stop invalidated the map epoch, a subsequent
read was rejected, and both the owned worker and sensor hub closed successfully.
No actuator or simulator ran. This one analytic plane does not validate real
camera calibration, tracking robustness, loop closure, world registration,
obstacle maps, base pose, controller tracking or navigation admission.

The focused CPU spatial regressions also passed: **86 passed in 1.70 s**, across
`test_spatial_cuvslam.py`, `test_spatial_rgbd.py`, `test_spatial_providers.py` and
`test_spatial_mcp.py`. The local log is `pytest-spatial-20261004.log`.
