# B32b/B32c on Isaac: one bin from two cameras that name it "orange" and "yellow" (8 Oct 2026)

**Question.** The bare scene ends with 4 beliefs for 3 props. The extra belief is a duplicate of the bin, because the top camera
`isaac` names the bin "orange" (median H 22) and the side camera `isaac_side` names it "yellow" (H 23). B32b (PR #261)
made colour identity per camera. Does it make the bin one belief live, without merging a yellow prop that sits in or
next to the orange bin?

**Answer.** With B32b alone, only partly: the bin was one belief in 5 of 7 runs. The overlap gate's box took in YOLOE's
table pixels, so the two live views scored a median IoU of 0.744 against a 0.75 threshold. With B32c (`90a9f0c`), the
box ignores each cloud's lowest centimetre. The same decisions then score 0.870–0.945, and the bin is one belief in
**5 of 5 runs, in all 60 frames**, with precision and recall unchanged at 1.0. Yellow 5 cm props in or next to the bin
were never merged with it, under either box (overlap ≤ 0.118). Measured on the x86 test rig (RTX PRO 6000, GPU 0).
These are not Spark numbers.

## Setup

- **Simulator.** Isaac Sim `6.2.0-alpha.19+develop.0.48b2d951` (internal build), **PhysX** GPU, private bridge on
  port 18521 with `CASCADE_ISAAC_PIXEL_MASK=1`. The B32a self-mask gate is on in every run.
- **Scene.** The bare reBot scene: `pink_cube` (0.17, 0.15), `green_cube` (0.30, 0.16), and the open bin at
  (0.18, -0.17). The props are reset before each run.
- **Perception.** YOLOE prompt-free, cameras `isaac` + `isaac_side` at 1280×720, and the real `build_runtime` →
  WorldWatcher + `get_observation` path. Nothing moves the arm.
- **Metric.** `scripts/eval_detector.evaluate` over 12 frames, with PhysX truth.
- **Arms.** `memory.per_camera_colour: true` (B32b, plus B32c from `90a9f0c`) vs `false` (the one-name rule of
  2026-09-10). Runs are interleaved; each is one fresh process with a fresh belief store.
- **Harnesses.**
  - [`b32b_live_ab.py`](b32b_live_ab.py): the A/B.
  - [`b32b_live_diag.py`](b32b_live_diag.py): adversarial scenes, plus logs of every neighbour-colour decision with
    both clouds and every observation → belief assignment.
  - [`analyse_samples.py`](analyse_samples.py): offline comparison of box metrics on the logged clouds.
- **Files.** Compact rows are in [`ab_summary.json`](ab_summary.json). Raw JSON, logs and clouds stay in `cascade-lab`
  ([`manifest.json`](manifest.json), sha256).

## Results

| arm | source | runs | bin = one belief | final beliefs | precision | recall | localisation mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| one-name rule (`false`) | c32ffcb | 3 | 0/3 | 4 / 4 / 4 | 1.0 / 0.80 / 1.0 | 1.0 | 1.68–1.93 cm |
| B32b, whole-cloud box | c32ffcb | 7 | 5/7 | 3 ×5, 4 ×2 | 1.0 ×7 | 1.0 | 1.55–1.88 cm |
| one-name rule (`false`) | 90a9f0c | 5 | 0/5 | 4 ×5 | 1.0 ×5 | 1.0 | 1.81–2.49 cm |
| **B32b + B32c, floor-trimmed box** | **90a9f0c** | **5** | **5/5 (60/60 frames)** | **3 ×5** | **1.0 ×5** | **1.0** | **1.61–1.65 cm** |

- **The 0.80 precision.** It comes from one one-name run with 9 phantom belief-frames. The harness does not log which
  belief they were; that run's duplicate yellow bin ended 5.6 cm off truth, the farthest of any bin belief.
- **Per-camera names.** In every B32c `on` run, the bin belief carries
  `source_colors = {isaac: orange, isaac_side: yellow}`.

### Why B32b alone was a coin flip

Run with `memory.neighbour_colour_iou: 1.0`, every side-camera view of the bin is logged as a decision sample:
86 decisions over 4 runs.

**Measured overlap (live).**
- **Whole-cloud 2–98 % box:** 0.708–0.926, median 0.744.
- **With each cloud's lowest centimetre dropped:** 0.870–0.945, median 0.921.

**Cause.**
- X and Z agree to 0.96–0.99. The loss is one axis (Y).
- The masks include table pixels at the object's own lowest height: a median 35 % of the side camera's bin cloud and
  12 % of the top camera's.
- The top camera's box ran to y = -0.28, which is 3.5 cm past the near wall.

**Relative vs absolute.** Dropping points by a fixed table height gives the same picture (min 0.869, median 0.925).
B32c uses the cloud's own low point instead, so a bin on a shelf behaves the same; a test pins this.

**The decision sticks.** The first decision is the one that counts. Once the side camera's view becomes its own
belief, later views match that belief by name, and no merge undoes it.

### What must stay apart

The adversarial scenes recolour `green_cube` yellow (RGB 0.90/0.80/0.10) and place it at three positions:

| scene | whole-cloud box (c32ffcb, 1 run) | floor-trimmed box (90a9f0c, 2 runs) | merged with the bin? |
| --- | --- | --- | --- |
| in the bin, centre (0.18, -0.17) | cube's own belief | cube's own belief | no |
| in the bin, corner (0.15, -0.20) | IoU 0.062 | IoU 0.072–0.100 | no |
| next to the bin (0.30, -0.17) | IoU 0.067–0.073 | IoU 0.067–0.075 | no |

- **All adversarial samples.** Over the 6 cube-sized samples logged with `--iou 1.0` (the decision population of the
  fixture receipt), the cube vs bin overlap is ≤ 0.118 with the whole-cloud box and ≤ 0.077 with the floor-trimmed box.
- **Centre scene.** There, the neighbour decisions are the side camera's views of the bin (0.72–0.75 whole-cloud,
  0.86–0.94 floor-trimmed). They pass the identity check against the orange bin, but the same-name yellow cube wins the
  match. See "Same-colour containment" below.

- **Ray-cast worst case.** The worst synthetic case, a 3.5 cm cube wholly inside the bin seen as a bleeding sliver,
  scores ≤ 0.634 with the floor-trimmed box (≤ 0.684 before). The threshold stays 0.75.

## Found on the way, not changed here

- **Same-colour containment.** With the yellow cube at the centre of the bin, the side camera's bin-sized "yellow" view
  (3 per run) is fused into the **yellow cube's** belief, because the names match and the cube's centre is closer than
  the bin's.
  - Behaviour is identical with `per_camera_colour: false`: this is the 2026-09-10 proximity rule, not B32b/B32c.
  - The cube belief's extent is overwritten for those ticks, then restored by the next top-camera frame.
  - Final cube error: 1.4 cm in both arms, vs 0.9 cm for a free cube.
  - Follow-up: a size-consistency check in the fusion gate (0.23 m bin view vs 0.09 m cube belief).
- **Bin position biased toward a cube in its corner.** With a cube in the corner, the bin belief sits nearer the cube
  than the bin centre (recall 0.72–0.86 in both arms). The top camera's bin mask takes in the cube. This is
  perception, not identity.
- **Brown "doormat" belief.** In some scenes with the cube moved, the top camera also reports a brown "doormat" near
  the bin, 3–11 cm off it, in both arms.

## What this is not

- No kitchen-scene run and no Newton run (this rig ran PhysX).
- No change to detection, colour bands, the threshold, radii, the workspace filter, safety or motion.
- One rig, one scene, 5 runs per arm. It is a floor, not a guarantee.
