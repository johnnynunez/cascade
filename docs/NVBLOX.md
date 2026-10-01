# nvblox and the kitchen cameras

Current integration: PR #27 is merged, including payload/contact and release
recovery, retained anchors, mapper replacement on clear and historical renderer
state. PR #49 adds bounded synchronization of actual jaw opening and newer
camera captures before establishing release geometry. See [current source and acceptance](PROJECT_STATUS_20261001.md) for the
MAIN pin and separate RTX diagnostic/campaign results. The measurements below
retain their original dates and sources; nvblox remains disabled in the normal
Spark presenter profile.

On 1 October, the isolated RTX environment 12 on `7e02de7` passed a native
orange release/recovery diagnostic: a labeled mapper refusal after opening
prevented withdrawal, then one explicit reset resumed the original withdrawal
before returning home and resetting all props. Three newer camera map commits
and all sampled camera ages passed. See the [source-bound receipt](evidence/nvblox-environment-12/release.json).
The subsequent five-object campaign failed on the first orange case before
release: the same observed attachment could not be confirmed. The planned
native reset passed; the remaining four cases were not attempted. Witnesses show real contact loss and a fall during horizontal transport, before
any release command. The mechanical cause remains unresolved. The [campaign receipt](evidence/nvblox-environment-12/campaign.json)
retains the failure, reset and camera measurements. Neither run establishes
five-object or presenter acceptance.

The [retained carry attachment guard](NVBLOX_CARRY_ATTACHMENT.md) checks the
original post-close attachment in existing Isaac state feedback. Lost or
unavailable evidence stops further transport and preserves a terminal failure;
it does not authorize an automatic open, reset, retry or return home. This
guard does not correct mechanical slip and is not enabled by occupancy-off
presenter runs. [Local RTX validation](LOCAL_RTX_VALIDATION.md) records the
subsequent software checks, administrative closure and separate PC diagnostics.

Environment 13 on `f9cb6b8e` passed passive admission and read-only truth readiness,
then failed its first orange route preflight before any actuator command. No
reset or transport followed. Localization retained a roughly 104-second-old
image while a missing model downloaded; independently fresh camera witnesses
did not make that analyzed image fresh. The subsequent three-second route
budget began after grasp generation. See the
[separate NV result](LOCAL_RTX_VALIDATION.md#separate-nv-result) for the retained
failure, timing limits and independent review.

Environment 14 on `0a27bda2` passed passive admission and native read-only
readiness with local model files already present. Its first orange localization
then exceeded the five-second image lifetime (9.5 seconds of analysis) and
stopped before actuation, without home or reset. Witness cameras remained
fresh; this did not validate the expired analysis image. The environment was
closed administratively with its failure retained. See the
[NV14 result and detector follow-up](LOCAL_RTX_VALIDATION.md#detector-reuse-and-nvblox-environment-14).

## Status (2026-09-30)

**Real nvblox CUDA mapping now runs on both RTX PRO 6000 Blackwell and DGX
Spark GB10. It is still disabled in the validated Spark presenter profile.**
The published PR #23 GraspGen-X acceptance used `occupancy=disabled`.

Both machines compiled nvblox v0.0.10 source commit
`c457c3fc01003bec6eba3ec1c61e6bf84bc3f51f`, using private CUDA **13.2.2**
(nvcc 13.2.86) and **PyTorch 2.14.1+cu132**. Native targets were SM120 and
SM121 respectively. System CUDA, NVIDIA drivers, Isaac Sim and the app's
Python environment were left in place. The release's source identifies its
Python package as `0.0.10.dev1`; receipts include the source SHA.

The source compatibility patch fixes Blackwell architecture parsing, uses
C++20 for the PyTorch wrapper, and resolves the ARM64 cuDNN library supplied
by PyTorch. Upstream Python metadata caps Torch at 2.9.1, so this isolated
build installs the wrapper with `--no-deps` after installing its mapping
dependencies. This is a tested local compatibility build, not an upstream
support claim. The [upstream release](https://github.com/nvidia-isaac/nvblox/releases/tag/v0.0.10)
has no ARM64 wheel; that did not prevent a working Spark source build.

## Reproduce the build

On Linux x86_64 or Spark aarch64, with a C++ compiler, `git`, `uv` and an
NVIDIA driver already installed:

```bash
bash scripts/install_nvblox_source.sh
# Optional: --prefix /path/to/private/nvblox --jobs 4

export LD_LIBRARY_PATH="$PWD/.nvblox/cuda-13.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.nvblox/venv/bin/python scripts/check_nvblox.py --output runs/nvblox-check.json
.nvblox/venv/bin/python scripts/serve_occupancy_bridge.py --backend nvblox --port 5557
```

The installer extracts NVIDIA's checksummed CUDA redistributions into its
own prefix, builds the pinned source, then requires actual depth integration,
ESDF queries and map clearing on the GPU. It does not enable the map in the
presenter launcher. Use explicit `--backend nvblox` when evaluating it:
`auto` can select a different implementation if nvblox fails to initialize.

## Hardware validation

- Both GPUs passed 15 upstream query/frame-integration tests.
- The real Cascade ZMQ bridge reconstructed a synthetic 5 cm cube above a
  table. Five known clearances differed by at most **10 mm** with 10 mm voxels.
  After 20 depth frames showing the cube removed, its old position changed
  from 0 to 40 mm clearance. This verifies actual ray carving, not an import
  test or a CPU stub.
- The production adapter needed two fixes: the ESDF query requires **Nx4
  `(x, y, z, radius)`** input despite the Python Nx3 docstring, and flat SDF
  masks cannot index a 3-D grid. The probe now reports `cuda:0` correctly.
- A concurrent startup probe and map refresh could corrupt the shared ZMQ
  REQ socket's send/receive state. Requests are now serialized; a real delayed
  REP peer reproduces and tests that race.

Machine-specific JSON receipts are in [benchmark/results](../benchmark/results/),
with names beginning `nvblox-rtx-` and `nvblox-spark-`. CUDA integration is
measured independently of robot manipulation acceptance.

## Does another camera improve accuracy?

The same archived idle Isaac RGB-D frames were replayed on both GPUs. The
metric samples 245 known surface locations on each of the two cubes, covering
five faces per cube. Ground truth is used only for evaluation. Depth is
robot-masked and subsampled to 640x360; each camera is integrated five times.
Both GPUs produced the same geometric results.

| Voxel | Cameras | Observed samples / 490 | Mean surface residual on observed samples | Samples within one voxel / 490 |
| --- | --- | --- | --- | --- |
| 10 mm | Main | 443 | 3.612 mm | 423 |
| 10 mm | Main + side | 490 | 4.516 mm | 470 |
| 10 mm | Main + side + proof | 490 | 4.421 mm | 474 |
| 5 mm | Main | 383 | 1.971 mm | 360 |
| 5 mm | Main + side | 458 | 2.314 mm | 428 |
| 5 mm | Main + side + proof | 469 | 2.455 mm | 426 |

**The second camera improved coverage substantially. The third camera gave
only a small change on these cubes, and did not consistently reduce residuals.**
A lower residual for one camera does not mean it reconstructed the scene
better: it missed more samples. Likewise, means at 5 and 10 mm use different
observed subsets, so they are not a controlled precision improvement.

Processing cost rises with resolution and camera count. In these runs, the
median warm depth integration for the three-camera sequence was 2.93 ms/frame
on RTX and 18.91 ms/frame on Spark at 10 mm; at 5 mm it was 29.99 and 98.47
ms/frame. The corresponding full workspace grid queries took 4.23 / 6.41 ms
and 19.87 / 31.85 ms. These are component measurements during the recorded
runs, not an end-to-end latency or throughput guarantee.

The input [capture](../benchmark/fixtures/nvblox/kitchen-idle-01/) includes
camera calibration, robot masks and SHA256 receipts. To replay it:

```bash
.nvblox/venv/bin/python benchmark/diagnostics/nvblox_camera_benchmark.py \
  --repo . --capture benchmark/fixtures/nvblox/kitchen-idle-01 \
  --voxel 0.01 --output runs/nvblox-cameras.json
```

To capture another idle scene, use
`benchmark/diagnostics/capture_nvblox_scene.py --port 8611 --output DIR` with
the app interpreter and a bridge providing validated robot pixel masks.

## Integration and limitations

In the current camera profiles, `isaac` and `isaac_side` feed geometric
fusion, while `isaac_proof` has `fuse_beliefs: false`. Three viewer streams
therefore do not automatically mean three map inputs. The replay above
explicitly integrates all selected cameras; it does not alter those profiles.

Cascade consumes the distance grid as a clearance gate for TCP and sampled
arm-link positions and attached payload samples. It does **not** refine object
localization or GraspGen-X input geometry. The merged planned-motion path vets
configured candidate routes against the map; this is not an arbitrary obstacle
avoidance planner or evidence that an unseen route is free.
These measurements establish mapping coverage, not more accurate grasping or
placement. No manipulation accuracy improvement has been demonstrated.

The first live Spark manipulation trial exposed an integration issue: the
carried cube remained in the world map and blocked the arm's own carry path.
An experimental follow-up separates payload geometry using measured bilateral
jaw contact and capture-time segmentation. Its interaction with retained map
observations, intentional grasp contact and reset is still under physical
validation. The presenter default remains unchanged until that acceptance
passes.

The ordinary non-payload world-map consumer can report unknown cells as no
known obstacle and an unavailable cache as no data. That policy is not proof of
free space. Payload/recovery modes additionally reject unknown samples and
require fresh, correctly bound integration commits. Known robot-mask failures
fail closed even if the cache has aged out. A mapping
benchmark, especially an idle scene, cannot establish collision safety during
motion or around unseen geometry.

For the upstream model and multi-camera requirements, see the
[nvblox interface](https://nvidia-isaac.github.io/nvblox/v0.0.10/pages/core_library_interface.html)
and NVIDIA's [multi-camera tutorial](https://nvidia-isaac-ros.github.io/concepts/scene_reconstruction/nvblox/tutorials/tutorial_multi_realsense.html).
