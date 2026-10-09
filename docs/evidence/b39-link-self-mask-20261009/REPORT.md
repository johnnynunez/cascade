# B39: the robot self-mask from link geometry (9 Oct 2026, CPU only)

**Question.** B32a gates both fusion paths on the render self-mask (`Frame.robot_mask`). Only Isaac frames carry
one, so on the real rig the arm's upper link is still fused as an object whenever it stands outside the base
cylinder and an open-vocabulary detector names it. Can the stack draw the same mask from what every rig has (the
URDF collision geometry, the measured joints and the camera calibration), accurately and cheaply enough to run on
every fused frame?

**Answer (offline).** Yes, against an independent per-triangle rasterisation of the reBot RS collision meshes.
Measured on the x86 test rig's CPU; no Isaac, no GPU, no camera. **The comparison against the Isaac render mask is
still owed** (see "Live validation owed" below): the render mask comes from the USD visual geometry, not from the
URDF collision meshes, so that number can differ from the one here.

## Mechanism (`src/cascade/perception/link_mask.py`)

- Each link's collision mesh is split into link-frame cells of `cell_m` (shipped 5 cm). Each cell becomes the 3D
  convex hull of its triangles, so the union of the projected pieces contains the projected mesh by construction.
  Box, cylinder, sphere and capsule primitives use a polyhedron that contains them. A mesh that cannot be read (a
  missing file, a git-lfs pointer that was never fetched, an unsupported format) becomes a capsule to the link's
  child joints, and the reason is recorded.
- Per frame, the links are posed by Pinocchio FK at the joint sample NEAREST the frame's capture time. The pieces
  are clipped at the camera's near plane, projected, filled as convex polygons and dilated by `dilate_px`
  (shipped 2 px).
- Without a joint sample within `max_skew_s` (0.15 s) of the frame, NO mask is built and the reason is counted
  (`no_joint_state` / `stale_joint_state`). Fusion then behaves exactly as without the flag.
- A frame that carries a render mask is never touched, and the arm is not even read.
- The mask goes into a fusion-local copy of the frame. The occupancy map keeps its own body masking.

## Results (`bench_cpu.py`, `cpu_bench.json`)

The reBot RS, 4 poses (home, handover, reach, low), the two demo cameras (`isaac` cam0 and `isaac_side`, Isaac
optics, 1280 × 720). The truth is the RS STLs read by the script, posed by Pinocchio directly, with each triangle
filled by its own call.

| variant | coverage (0 px / 2 px) | bloat 0 px | IoU 2 px | ms / frame |
| --- | --- | --- | --- | --- |
| **cell 5 cm (shipped)** | ≥ 0.999 / **1.0** | 0.8–5.2 % | **0.889–0.951** | **3.1–5.1** |
| cell 2 cm | ≥ 0.999 / 1.0 | 0.06–1.7 % | 0.917–0.959 | 15.7–23.0 |
| one hull per link | ≥ 0.9995 / 1.0 | 4.5–33 % | 0.715–0.919 | 1.2–1.8 |
| cell 5 cm, no scipy | as shipped | as shipped | as shipped | 14.3–30.6 |

- Coverage below 1 at 0 px is rasterisation at the silhouette's edge. The shipped 2 px removes it in every row.
- Most of the shipped bloat at 2 px is the dilation itself. At 0 px the cells add 0.8–5.2 %; one hull per link
  adds up to 33 % on the side camera, which sees the L of the upper link.
- Re-run on the rebased branch (`4d0947b` + B39), same numbers; timing 3.3–4.8 ms per frame.
- `pytest tests/test_link_self_mask.py`: 73 tests. They include the RS check (coverage 1.0, bloat ≤ 0.20 and
  IoU ≥ 0.82 per pose and camera) and the synthetic arm through `WorldWatcher._tick`, `get_observation` and
  `_reobserve`.
- Subdividing one mesh past 2 × 10^6 triangles is refused, and the link becomes a capsule with the reason
  recorded. A millimetre mesh without its URDF `scale` would otherwise exhaust memory. The largest RS mesh needs
  1.7 × 10^5 at 5 cm.

## Live validation owed (the parent, on Isaac)

`scripts/compare_link_self_mask.py`. `record` runs the real runtime with the link mask on, against a private
bridge with `CASCADE_ISAAC_PIXEL_MASK=1`. It saves every fused frame with its render self-mask, its capture-time
joint snapshot, the runtime link mask (nearest joint sample, the real-rig path) and the detector's masks. `score`
re-builds the link mask offline at the capture-time joints and reports IoU, coverage and bloat per camera. It also
reports how often fusion's gate drops the same detections under both masks.

The target before calling the link mask equivalent to the render mask on Isaac: gate agreement on every detection,
render coverage ≥ 0.99. IoU is reported but not targeted (USD visual vs URDF collision geometry).

## Not claimed

- **Isaac and hardware.** No live Isaac comparison yet and no physical-camera measurement. The real-rig accuracy
  also depends on the hand-eye calibration and on joint-encoder offsets, which no CPU test can see.
- **Fast motion.** The mask is posed at the nearest joint sample within 0.15 s. A fast-moving arm can be off by
  its joint speed × that skew. Fusion is paused during motion skills, so this matters only for motion outside them.
- **Detection masks.** Unchanged. A detection that is mostly the arm is dropped. A prop's pixels that the dilated
  silhouette covers are not lifted to 3D, as with the render mask.
