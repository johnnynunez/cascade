# Perception and Execution: what to change, and what the measurements say

Status: research note, 2026-08-08. No code changed yet.

This note answers four questions raised together:

1. is the perception stack (L515 to YOLOE to mask to cloud to BeliefStore) still the right shape?
2. does YOLOE need replacing, specifically by DINOv3?
3. does TurboVLA fill the open `VLAExecutor` slot?
4. does Qwen-3D change the perception architecture?

It starts from what this repo has already measured, because two of those
questions are already partly answered by numbers in the tree.

---

## 1. What we already measured

`scripts/eval_detector.py`, 12 frames, fully static scene, ground truth from
Isaac PhysX via `TruthPoseReader` (independent of perception):

```
recall          : 100.0%     never misses a real object
precision       :  76.6%     sees 4 objects where there are 3
phantom rate    :  23.4%     11 of 47 detections
localization    : mean 2.4 cm, worst 3.9 cm
count stability : FLICKER    frame 1 sees 3, the rest see 4
```

Two readings that decide everything downstream:

**YOLOE is not blind. It hallucinates.** Recall is perfect. The failure is a
*stable* false positive that survives into the world model and reached the UI
as a fourth badge in `annotated_view`.

**The largest measured win in this repo did not come from the detector.**
From commit `1337263`:

> one perception change moved success 4/10 to 10/10. The entire
> verification-and-retry stack had moved it 4 to 5 to 6

That change was `_recentre_by_size()` in `grounding.py`: the fitted OBB centre
was biased 1.6 cm toward the camera. Geometry downstream of the detector, not
the detector.

Any proposal to swap perception has to beat that baseline: does it move a
measured number, and is it cheaper than the alternative that also moves it?

---

## 2. DINOv3 as a YOLOE replacement

**It cannot be a drop-in replacement, because it has no text head.**

DINOv3 (Meta FAIR, `arXiv:2508.10104`) is a self-supervised backbone. It emits
a class token, register tokens and dense patch tokens. It does not map
"pink cube" to a box. The `Detector` ABC in `perception/detector.py` is:

```python
def detect(self, frame, classes: list[str] | None = None) -> list[Detection]
def set_classes(self, classes: list[str]) -> None
```

`set_classes(["pink cube"])` cannot be honoured by DINOv3 in any meaningful
sense. Substituting it directly would silently break open-vocabulary
prompting, which is a booth-facing capability: any visitor object localizes
by name today.

That is the constraint. It does not mean DINOv3 is the wrong tool. It means
the interface it fits is a different one.

### The shape that does fit the measured failure

Recall is already 100%. We do not need a better proposer. We need a
**verifier** that kills a stable false positive.

DINOv3's headline property is exactly that discriminator: dense features whose
cosine similarity separates object identity, without fine-tuning. So:

```
YOLOE (open-vocab proposer, ~30 ms)  ->  N+1 candidate masks, recall 100%
        |
        v
DINOv3 patch-feature check against enrolled object prototypes
        |
        v
reject candidates whose features match no enrolled object  ->  phantom filtered
```

This preserves the property we have (never miss a real object), targets the
property we lack (precision), and keeps the text interface intact because
YOLOE still does the naming.

Enrollment cost is near zero here, and this is the part the rig makes cheap:
`TruthPoseReader` gives every prop's real pose, so prototype crops can be
harvested and verified automatically in sim, with no human labelling.

### Honest costs

- **Gated weights.** HF marks the models `gated: manual`; the GitHub repo
  requires an access request answered by email. Not a blocker, but it is not
  `pip install` either.
- **DINOv3 License, not Apache.** Redistribution carries the agreement, and
  publications using it must acknowledge it. Worth knowing before it appears
  in a paper or a shipped demo.
- **New heavy dependency.** The shared venv has neither `transformers` nor
  `timm`. DINOv3 needs `transformers>=4.56.0` (or `timm>=1.0.20`). Per the
  rig rules this goes in through `uv`, and never into an embedded interpreter.
- **A second model in the hot path.** A ViT-S/16 forward per frame is not
  free. The 3 Hz WorldWatcher budget has room, but this must be measured, not
  assumed.

### The alternative that also fixes it

`docs/SYNTHETIC_RGBD_PIPELINE.md` already proposes auto-labelling in sim
filtered by physics truth, then fine-tuning YOLOE. Target: precision 76.6% to
>95%. That needs no new dependency, no gated weights, and no second model at
runtime.

**These two are competing hypotheses for the same measured number.** That is
the good news: `eval_detector.py` can score both, so this is decidable rather
than arguable.

---

## 3. TurboVLA

`arXiv:2607.27205` (HUST). Reformulates the usual `V -> L -> A` VLA pathway as
a direct `V + L -> A` mapping: separate encoders, lightweight bidirectional
interaction, compact action-chunk decoder, no LLM as the perception-to-action
interface.

```
LIBERO   97.7% mean success
params   0.2B
latency  31.2 ms   (32 Hz)
VRAM     0.9 GB    (RTX 4090)
```

Apache-2.0, checkpoints on HF, training and eval code released.

**Why it fits here.** The ROADMAP already specifies the slot:

> VLA policy backend: add a `VLAExecutor` behind the skill API so
> `grasp_object` can be served either by the deterministic OBB pipeline or by
> a language-conditioned policy; the agent layer stays unchanged.

TurboVLA is a better candidate for that slot than the LingBot-VLA-v2 currently
named there, for one rig-specific reason the ROADMAP itself flags: LingBot
needs flash-attn built for aarch64 + Blackwell, or two hardcoded attention
implementations patched to SDPA. A 0.2B model at 0.9 GB has far less surface
to break on GB10 (sm_121).

**How to test it cheaply.** `benchmark/libero/run_wrc.py` already drives
LIBERO with this repo's real runtime. TurboVLA reports on LIBERO. The
comparison is therefore available without building anything new.

**Caveat.** 97.7% is their evaluation on their stack. This repo learned the
hard way (`0e045a7`, `54ebd4a`) that a LIBERO success signal can score
episodes the robot never ran. Reproduce before believing, including our own
harness.

---

## 4. Qwen-3D

ECCV 2026, CMU. A geometry-aware LMM on Qwen2.5-VL: RGB-D features unprojected
to world space and voxel-pooled (~5 cm), 3D RoPE over `(t,x,y,z)`, and a
query-based mask decoder.

Its diagnosis names this repo's architecture directly. It lists three
bottlenecked interfaces between language and dense geometry, and the second is
what we run:

- decoding coordinates as text tokens
- **selection over proposals from an external detector**  (YOLOE here)
- compressing grounding into a single `<REF>` token

Out-of-domain results, which is the regime our phantom lives in:

| model | Acc@25 (Locate-3D / ScanNet++) |
|---|---|
| UniVLG (specialist) | 32.3 |
| Video-3D LLM | 33.2 |
| **Qwen-3D-3B** | **55.7** |

Their claim is that decoding directly from VLM features generalizes better
out of domain than proposal-selection pipelines. Our phantom is an
out-of-domain failure: YOLOE never saw this table, these cubes, this lighting.

**Why it is not a near-term swap.** It is 3B/7B with no published latency,
trained on ScanNet room scenes rather than 0.9 m tabletops. Only ~50M
trainable parameters (LoRA + mask decoder) makes adaptation plausible, but it
is still a fine-tune. It is the architectural answer to "is the bounding box
the constraint", not a weekend change.

---

## 5. Where the nine sources agree

VoLo (NVIDIA + UMich), Pigey (Princeton + Together), ASPIRE (NVIDIA GEAR),
VIA, RPent/Harness-VLA, HumanCLAW (Meta + NTU), Agentic-VLA, TurboVLA,
Qwen-3D.

The convergent finding is that **orchestration, not the module, is usually the
bottleneck**:

- Pigey: 12.8% to 53.3% on LIBERO-PRO with **frozen policy weights**; 16.7% to
  97.3% on a real FR3. They name the gap directly.
- VIA: 96.7% on three LIBERO-Goal tasks with an off-the-shelf agent and no
  robot-specific fine-tuning, given a readable interface.
- ASPIRE: the trace-exposing execution engine alone lifted success 14% to 62%.
- VoLo: monitor, halt, redirect over an interruptible VLA, with SAM3 as a
  *tool* the VLM calls rather than as closed-loop perception.

This repo's own measurement is the dissenting data point, and it is the more
useful one: here the bottleneck **was** a module (a 1.6 cm sensor-model bias),
and the orchestration layers were measurably not the constraint. Commit
`1337263` publishes that against our own prior thesis.

The synthesis: orchestration wins when the modules are sound; module fixes win
when a module is quietly wrong. Only measurement distinguishes the two, which
is an argument for `eval_detector.py` and `TruthPoseReader` over any
architectural preference.

### Two things worth flagging

**A likely citation error.** `docs/ARCHITECTURE.md` describes Agentic-VLA as
"ICML 2026, LLM sub-goal decomposition, zero-shot VLM exploration critic,
embedding-indexed experience memory". `arXiv:2605.22896` is Jin and Zhang, on
*online adaptation* of VLAs with adaptive reward synthesis. Either a different
paper is meant or the description is wrong. Fix before it reaches a
publication.

**VoLo uses SAM3 as a tool, not as perception.** Relevant to the question of
whether SAM replaces YOLOE: in the closest published system, it does not. SAM
segments, it does not name; it needs a prompt from something upstream.

---

## 6. Proposed order

Every step has a falsifiable target measured with tooling that already exists.

**Phase 0. Re-establish the baseline.** Re-run `eval_detector.py` on the
current tree. The 76.6% predates `_recentre_by_size()` and other changes;
every later comparison needs a current number.

```bash
cd models && PYTHONPATH=../src python ../scripts/eval_detector.py \
    --frames 12 --json /tmp/det_baseline.json
```

**Phase 1. DINOv3 as a phantom verifier.** Not a swap. YOLOE keeps proposing
and naming; DINOv3 rejects candidates matching no enrolled prototype.
Prototypes harvested in sim and auto-filtered against `TruthPoseReader`.

Target: **precision 76.6% to >95%, recall stays 100%, `count_stable` true over
12 frames.** Also record added latency per frame; if it does not fit the 3 Hz
budget it belongs on the failure path instead.

**Phase 2. The competing hypothesis, same metric.** Auto-labelled sim data,
fine-tune YOLOE, re-measure. Same target. Whichever wins on precision per unit
of added complexity is the one that ships. If both work, YOLOE fine-tuning
wins on dependency cost; if neither moves precision, the phantom is not a
detector problem and we look upstream at masks.

**Phase 3. TurboVLA behind `VLAExecutor`.** Reproduce their LIBERO number in
our harness first, then wire it behind the skill API so `grasp_object` can be
served by either backend. Target: match or beat the deterministic OBB pipeline
on the rig ablation with no regression in the harness-gated safety path.

**Phase 4. Qwen-3D prototype, offline.** Score it with `eval_detector.py` on
recorded frames only, no runtime integration. It answers a research question
(does direct mask decoding beat proposal selection on our OOD tabletop) and
should not touch the hot path until it has a number.

---

## 7. What this note recommends against

- **Replacing YOLOE with DINOv3 outright.** It has no text head; open-vocab
  naming would be lost, and recall is not the failing metric.
- **Putting any VLM in the 3 Hz perception loop.** The warm world model is
  what makes command resolution a lookup instead of a round trip. A 1 to 3 s
  model call destroys it. `vlm_ground.py` already has the correct placement:
  the failure path.
- **Adopting an architecture before measuring the current one.** The most
  valuable result in this tree so far was produced by scoring perception
  against physics truth and following the number, including when it refuted
  the repo's own published thesis.

---

## Sources

| work | venue / id | relevance here |
|---|---|---|
| DINOv3 | Meta FAIR, `2508.10104` | dense features; verifier, not detector |
| TurboVLA | `2607.27205` | 0.2B, 31.2 ms, 0.9 GB; `VLAExecutor` candidate |
| Qwen-3D | ECCV 2026, CMU | names proposal-selection as the bottleneck |
| VoLo | NVIDIA + UMich | physical orchestration; SAM3 as a tool |
| Pigey | Princeton + Together AI | orchestration gap, frozen weights |
| ASPIRE | NVIDIA GEAR | trace-exposing engine; skill library |
| VIA | `2607.11119` | the interface is the missing piece |
| RPent / Harness-VLA | `2607.08448` | service-oriented agent composition |
| HumanCLAW | Meta + NTU | skill harness plus verifier before the body |
| Agentic-VLA | `2605.22896` | online adaptation; see citation flag above |
