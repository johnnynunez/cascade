# B31 on Isaac: same-colour twins and the open-vocabulary A/B (8 Oct 2026)

Live check of PR #251 (`BeliefStore.update_frame`, per-frame instance association) on the real demo
perception stack. The question: does it keep two IDENTICAL props apart, and does it change the
open-vocabulary phantom result that label-agnostic fusion was introduced for? Measured on the x86 test rig
(RTX PRO 6000 GPU 0). These are not Spark numbers.

## Setup

- Source: branch `feat/backlog-same-colour-beliefs` at `a5e20c5` (bridge and runtime from the same tree).
- Isaac Sim `6.2.0-alpha.19+develop.0.48b2d951` (internal build), **PhysX**, CUDA physics, 60 Hz, private
  bridge on port 18521 (`CASCADE_ISAAC_PIXEL_MASK=1`), bare reBot scene: `pink_cube` (0.17, 0.15), `green_cube`
  (0.30, 0.16), the bin at (0.18, -0.17). The props are 0.05 × 0.05 × 0.08 m boxes.
- Perception: YOLOE prompt-free (`yoloe-11s-seg-pf.pt`, the shipped `detector:` block), cameras `isaac` +
  `isaac_side`, `load_demo_config(cameras=["isaac","isaac_side"], arm="isaac", llm="mock")`, the real
  `build_runtime` → WorldWatcher + `get_observation` path. Nothing moves the arm.
- Metric: `scripts/eval_detector.evaluate` (the metric behind `benchmark/results/perception_open_vocab.json`),
  ground truth from PhysX (`TruthPoseReader`). For the twins it adds, per frame, the beliefs within 10 cm of
  either twin and the YOLOE detections on the pair in that frame.
- Arms: `memory.instance_association: true` (new) vs `false` (the pre-B31 per-detection store). There is one
  fresh process, runtime and belief store per run.
- Twins: `green_cube` is recoloured to the pink cube's exact displayColor (`exec`) and placed (`place_prop`)
  at `pink + (0, sep)`, 5–9 cm centre distance (0–4 cm gap). This gives two identical props.
- Harness: [`b31_live_eval.py`](b31_live_eval.py). Compact rows: [`live_summary.json`](live_summary.json).
  The raw per-run JSON and the bridge log stay in `cascade-lab` ([`manifest.json`](manifest.json), sha256).

### Harness friction found on the way

On this revision `load_demo_config` does not apply `CASCADE_BRIDGE_PORT` / `CASCADE_OCCUPANCY_PORT`. Only
`GraspGenXPlanner` reads `CASCADE_GRASPGENX_PORT`, so the resolved config still named 8611/5556/5557. The
harness rewrites every port in the resolved config and refuses to start if a shared port remains.

## Results

Same-colour twins: 10 runs per arm (separations 5, 6, 7, 8, 9 cm; 6 and 8 cm repeated), 4–8 frames each.

| arm | runs with two beliefs on the pair | twin xy error (mean of run means / worst) | recall |
| --- | --- | --- | --- |
| instance association (new) | **9 / 10** (every frame of those runs) | 0.7–2.0 cm / 3.05 cm | 0.83–1.00 |
| per-detection (old) | **0 / 10** (one belief in every frame) | 3.0–3.7 cm / 4.85 cm | 0.667 |

- Where it was recorded (8 runs, 16 frames per arm), YOLOE returned **2–3 detections on the pair in every
  frame**: one per cube, sometimes a second name boxed on one cube. So the old store's single belief was the
  STORE merging two detections, not the detector.
- The third (alias) box stayed in its cube's belief under the new store, as one instance. It added no belief.
- The one unresolved run with the new store was the first run at 6 cm, before detection counting was added.
  Two later 6 cm runs with the new store resolved (2 beliefs in every frame).

Default 3-prop scene: 2 runs per arm, 12 frames each.

| arm | precision | recall | phantom rate | count per frame |
| --- | --- | --- | --- | --- |
| instance association (new) | 0.821 / 0.804 | 1.0 / 1.0 | 17.9 % / 19.6 % | 3,3 → 5 |
| per-detection (old) | 0.821 / 0.821 | 1.0 / 1.0 | 17.9 % / 17.9 % | 3,3 → 5 |

The two stores are indistinguishable on the open-vocabulary scene: both end with 5 beliefs for 3 props. In
both arms the extra beliefs are the same two, and only one of them is a phantom by the metric:

- **The phantom (all of the 17.9–19.6 %)** is a gray "biplane" / "fighter jet" about 0.42 m above the table.
  The side camera sees the arm's own upper link at the top edge of its image, and 98–99 % of that detection's
  pixels are robot pixels in the bridge's render mask. Fusion never consults that mask, and the link is outside
  the workspace filter's 0.12 m base cylinder.
- **The duplicate** is a yellow "building block" 2–4 cm from the bin. The side camera renders the orange bin pale
  yellow and names it "yellow"; the top camera names it "orange". The 2026-09-10 colour rule keeps differently
  named observations apart, so the bin becomes two beliefs. The metric scores the duplicate as a hit, since it
  lies within `HIT_M` = 6 cm of the bin, but it makes the count wrong.

The diagnosis comes from a per-detection log of both cameras over 24 rounds, on the same bridge, with
annotated crops ([`b32_phantom_diag.py`](b32_phantom_diag.py), [`diag_summary.json`](diag_summary.json)):

- [`side_cam_phantom_robot_link.jpg`](side_cam_phantom_robot_link.jpg): the "biplane" mask (green) lies on the
  robot's render mask (red).
- [`side_cam_bin_named_yellow.jpg`](side_cam_bin_named_yellow.jpg): the bin as the side camera renders it.

Both causes are on `main` and independent of B31; they are tracked as backlog B32. The
documented 0 % row (`perception_open_vocab.json`, 2026-08-08, before the colour rule) does not reproduce on
this build for EITHER store.

## Not claimed

- No physical (real-camera) measurement, no Spark measurement, no Newton measurement (PhysX only).
- Twins separated along one axis (y) in one scene with one prop shape. Touching or occluding props remain a
  detector question.
- The 17.9–19.6 % phantom rate (the arm's own link) and the bin duplicate occur on this build with both
  stores. B31 neither causes nor fixes them.
