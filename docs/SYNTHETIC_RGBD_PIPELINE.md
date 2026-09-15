# Synthetic-Augmented RGB-D → 3D Object Localization

Analysis of the proposed pipeline (diagram dated 2026-07-31) against what this
repository already provides, using **measurements from the rig**.

---

## The measured problem

`scripts/eval_detector.py`, 12 frames, a **completely static scene**, and
Isaac physics ground truth (`TruthPoseReader` → PhysX, independent of
perception):

```
GROUND TRUTH (3 objects)
   pink_cube   [0.17,  0.15, 0.04]
   green_cube  [0.30,  0.16, 0.04]
   bin         [0.18, -0.17, 0.03]

count per frame : [3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
stability       : FLICKER
modal count     : WRONG  (4, should be 3)

precision       : 76.6%
recall          : 100.0%
phantom rate    : 23.4%   (11 of 47 detections)
localization    : mean 2.4 cm, worst 3.9 cm
```

Three findings need separate treatment:

1. **Recall 100%.** Every real object is detected.
2. **Precision 76.6% — a persistent phantom.** The detector sees four objects
   where there are three, frame after frame. This is a *stable* false positive
   that enters the world model and persists. It also appeared in the UI as
   the fourth badge in the first session's `annotated_view`.
3. **Flicker.** The first frame reports three objects; the rest report four.
   The same scene produces different answers, so "how many cubes are there?"
   has no stable answer.

The **mean localization error of 2.4 cm** suggests block C works well:
2D→3D lifting is already effective. The problem is upstream, in the masks the
detector produces.

A generic open-vocabulary detector (YOLOE with text prompts) has never seen
*this* table, *these* cubes, or *this* lighting. It is operating zero-shot in
a domain it was not trained for, with a measured phantom rate of 23.4%.

**This is the gap the proposed pipeline addresses.**

---

## The proposal, block by block

| Block | Function | Status in this repository |
|---|---|---|
| **A** RGB-D bootstrapping | Eye-in-hand captures → SAM-assisted labeling → fine-tune → relabel → promote to train/validation sets | ❌ Not implemented |
| **B** View synthesis | Lift objects to point clouds, render again with different pitch/yaw, compose new labeled scenes | ❌ Not implemented |
| **C** YOLO-Seg + 2D→3D lifting | Mask + depth + intrinsics → centroid in the base frame | ✅ **Implemented** |
| **D** Centroid for robotics | Feed the centroid into grasp planning | ✅ **Implemented** |

**Block C is already complete** in `perception/grounding.py`:
`mask_to_points_cam()` backprojects mask pixels using an interquantile depth
band (the diagram's *lift objects to point clouds* step), and
`oriented_bbox()` computes the center and axes with PCA. `probe.py` adds
`deproject()` with median depth sampling.

The proposal therefore builds on the existing pipeline by **supplying it
with a detector that produces fewer false positives**.

---

## The value of the self-labeling loop (block A)

The key to reducing cost is that **the robot generates its own labels**:

```
eye-in-hand captures → SAM labels (RGB only) → bootstrap fine-tuning
      ↑                                                ↓
      └──────── review ← generate new labels ──────────┘
```

Each iteration improves labeling, reducing the human review needed on the
next pass. This applies the Voyager/ASPIRE approach to perception and fits
the existing outer loop: `scripts/learn_from_runs.py` and the nightly cron job.

**The earlier sessions had already accumulated 80 keyframes**, with arm poses
and associated beliefs. These provide a starting point for block A without
capturing new data.

The repository also has **physics ground truth**: `TruthPoseReader` reports
the actual position of each prop. This can turn block A's manual
"SELECTED SUBSET / review" step into an automatic filter: reject a label if
its 3D centroid is not within 6 cm of a real prop. In simulation, this closes
loop A **without human review**.

---

## Where the diagram needs changes

**1. Block B (synthetic composition) has the weakest return for its cost on
this rig and should come last.**

The pipeline proposes it to increase viewpoint diversity. This rig already
has **Isaac Sim**, which can move the camera, vary lighting, randomize prop
poses, and produce *physically correct renders with ground-truth labels*.
Composing point clouds can introduce inconsistent lighting and cutout edges
that the model learns as shortcuts. Native domain randomization is preferable
when the simulator is already available.

Block B makes sense for a **physical-only domain** without a digital twin,
such as a D435i/L515 on the physical arm. For the simulated demo, use Isaac.

**2. "Get centroid for robotic tasks" (D) is the diagram's weakest step.**

A centroid is enough for pointing, but grasping needs more information.
Measurements in this repository showed that a partial point cloud's centroid
is biased toward its visible face. GraspGenX addresses grasp orientation,
which an OBB alone cannot determine. The pipeline should **provide GraspGenX
with better masks**.

**3. The verification loop is missing.** The diagram ends at training and
deployment without defining how to establish whether the new model improves
on the old one. `TruthPoseReader` makes this measurable: compare precision
and recall against physics before and after training. Promotion into the
training set needs that measurement.

---

## Recommended implementation order

**1. Establish the metric — ✅ DONE.** `scripts/eval_detector.py` measures
precision, recall, flicker, and localization against physics. The baseline
is recorded above and provides the comparison needed for label promotion.

```bash
cd models && PYTHONPATH=../src python ../scripts/eval_detector.py \
    --frames 12 --json /tmp/det_baseline.json
```

**2. Automate labeling in simulation (estimated 1–2 days).** Randomize props
in Isaac, capture RGB-D, masks, and actual poses, and **filter every label
against physics ground truth**. This implements block A with automated
review, using information that would require human annotation on a real rig.

**3. Fine-tune YOLOE on that dataset and repeat step 1.** The measurable
target is **precision rising from 76.6% to >95%**, with `count_stable = true`
over 12 frames. If the measurements do not improve, reject the approach.

**4. Connect it to the existing nightly cron job.** `learn_from_runs.py`
was already scheduled for 03:00 on the rig. Extend it to accumulate labeled
keyframes and trigger fine-tuning when enough new examples are available.
This closes block A's loop without manual intervention.

**5. Reserve block B for the physical arm**, where no digital twin is available.

---

## Assessment

The proposal addresses a **measured problem in this repository**: a 23.4%
phantom detection rate. Block C is already implemented and tested, supporting
that part of the design.

Prioritize **A**. Physics ground truth makes its labeling loop cheaper on this
rig than the diagram assumes.
