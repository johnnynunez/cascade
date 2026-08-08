# Perception and evaluation for an open-vocabulary booth agent

Status: research note, 2026-08-08. Evidence-first. Nothing here is a plan to
adopt something because it is new; every recommendation is tied either to a
number measured in this repo or to a claim traced back to its source.

Scope of the two questions this answers:

1. What do the current agentic-robotics systems actually use for perception?
2. What would a credible, paper-grade evaluation of wrc_demo look like?

---

## 1. What each system uses for perception

This is the table that matters, because the answer is not what the framing of
"they use VLMs" suggests. Sources: paper abstracts, project pages, and where
possible the code itself.

| system | perception | identity | 3D | notes |
|---|---|---|---|---|
| **VoLo** (NVIDIA + UMich) | SAM3 as a **tool the VLM calls** | VLM | grasp/place primitives | not in a closed loop |
| **RPent / Harness-VLA** | `segment` tool -> optional SAM3 service, **plus `back_project`** | VLM | pixel -> world via depth map | SAM3 is optional; falls back to image inspection |
| **ASPIRE** (NVIDIA GEAR) | **Depth + SAM3 + grasp sampler**, ground truth explicitly forbidden | SAM3 prompt registry | depth + `sample_grasp_pose` | verified from the PDF, see section 4 |
| **VIA** | screenshots of a 3D browser interface | frontier model | the interface renders it | no detector at all; **no privileged state** |
| **Pigey** | frozen policy + VLM orchestration, cameras fixed | VLM | policy-internal | weights unchanged; no paper yet |
| **HumanCLAW** | egocentric view -> VLM skill harness | VLM | half-physics sim | verifier gates every decision |
| **TurboVLA** | its own vision encoder | implicit in policy | implicit | `V + L -> A`, no LLM in the middle |
| **Qwen-3D** | **is** the perception: RGB-D -> world-space tokens | mask decoder | native, voxel-pooled ~5 cm | no published latency |
| **Agentic-VLA** | inherits from the base VLA | policy | policy | about online adaptation, not perception |
| **Waddle Labs** | no public technical detail | ? | ? | landing page only, treat as unknown |

ASPIRE's stack is the most explicit, taken from its own architecture figure and
agent instructions: `Depth`, `SAM3`, `Grasp`, all emitting multimodal traces
into a skill library. Its disambiguation skill is worth copying directly:

> Detect all instances, sort by the axis implied by the qualifier (X for
> front/back, Y for left/right in robot frame), then select by keyword index.

plus a per-object SAM3 prompt registry and confidence filtering that rejects
low-score centroids. wrc_demo has the equivalent axis map already
(`_SPATIAL_AXES` in `grounding.py`), which is convergent evidence the approach
is right.

### The finding

**Not one of these systems runs a continuous open-vocabulary detector as its
perception loop.** The dominant pattern is:

> a VLM reasons over raw frames, and calls a segmenter *on demand* when it
> needs a precise mask; 3D comes from back-projecting depth at points the VLM
> chose.

RPent is the clearest evidence because the code is on this machine. Its LIBERO
toolkit exposes 15 tools, and the perception-relevant ones are:

```
segment          prompt or point -> SAM3 service, OPTIONAL
back_project     (row, col) or a region -> world XYZ from a precomputed map
view_camera_meta
```

`segment` degrades gracefully when `SAM3_SERVER_URL` is unset:

```python
return {
    "error": "segmentation service not configured",
    "fallback": "Use manual visual localization and back_project.",
}
```

That is a system where **segmentation is a convenience, not the perception
substrate**. The substrate is depth back-projection at VLM-chosen pixels.

### What this means for wrc_demo

wrc_demo already has both halves of that pattern and did not get credit for it
in its own docs:

| RPent / VoLo concept | wrc_demo equivalent | status |
|---|---|---|
| `back_project` | `probe.py` `PointProbe` + `deproject()` | present |
| `segment` on demand | `vlm_ground.py` on the failure path | present |
| annotated interface | `visual_interface.py` (VIA-style) | present |
| trace-exposing engine | `TraceLogger` + `agent/aspire.py` | present |
| verifier before acting | `PostconditionChecker`, `SafetyHarness` | present |

The genuine architectural difference is that wrc_demo *also* runs a 3 Hz
always-on detector feeding a `BeliefStore`. None of the surveyed systems do
that. It is worth keeping deliberately rather than by accident:

- **for**: object memory across occlusion and out-of-view, colour queries
  ("the pink one") that resolve without a model call, and a warm world model
  that makes command resolution a lookup. VoLo's own text calls memory and
  state tracking one of its four capability suites, and its agent needs a
  scene history to do it. wrc_demo gets that continuously and for free.
- **against**: it is the component that produced every phantom measured here,
  and it costs a detector forward pass per frame regardless of whether anyone
  asked a question.

Both halves of that trade were measured today (section 3).

---

## 2. Perception is three separable jobs

The request was "identifier, where the objects are, and memory over time".
Those are three jobs and the survey shows they are best served by three
different mechanisms:

```
   WHAT IS IT            WHERE IS IT           WHAT WAS IT DOING
   (identity)            (geometry)            (memory)
   open-vocab labels     depth back-projection  BeliefStore over time
   + VLM on failure      + OBB / GraspGen       + aliases + last_seen
```

Conflating them is what produced the phantom problem: a *labeller* was being
used as an *object counter*. The measured fix was to stop asking the label to
carry geometric truth (`WorkspaceFilter`) and stop asking geometry to carry
identity (label-agnostic fusion).

DINOv3 fits this decomposition as a **fourth** mechanism, identity
*verification* without naming, and it is the only role in which it can be used
here: it has no text head, so it cannot replace an open-vocabulary detector,
and enrolling prototypes per object is impossible in a booth where the objects
are unknown. The usable form is class-agnostic duplicate rejection (are these
two masks the same physical thing?), which needs no enrollment.

**That is a hypothesis, not a recommendation.** It only earns its place if it
beats the geometric fusion already landed, measured on the same metric.

---

## 3. What is actually measured in this repo today

Every number below came from `scripts/eval_detector.py` against Isaac PhysX
ground truth (`TruthPoseReader`), 12 frames, static scene, 3 real props.

| configuration | precision | phantoms | stability | recall |
|---|---|---|---|---|
| closed vocabulary (8 words) | 76.6% | 23.4% | FLICKER | 100% |
| open vocabulary, both fixes disabled | 51.8% | 48.2% | FLICKER | 100% |
| open vocabulary, geometric gate disabled only | 42.0% | 58.0% | FLICKER | 80.6% |
| **open vocabulary + workspace filter + label-agnostic fusion** | **100%** | **0%** | **STABLE** | **100%** |

Artifacts: `benchmark/results/perception_nofilter.json` (gate disabled),
`benchmark/results/perception_open_vocab.json` (final). Reproduce the ablation
with `scripts/ablate_workspace_filter.py`, which overrides the thresholds at
runtime rather than editing the shipped config.
The 51.8% row was measured live during development and its JSON was not kept;
the 42.0% row is the reproducible ablation. The two differ because the 51.8%
run predated label-agnostic fusion, so they are not the same ablation and are
listed separately rather than conflated.

Note the recall column: with the geometric gate off, recall drops to 80.6%.
Scenery detections do not merely add phantoms, they also displace real objects
through the fusion step, so the gate protects recall as well as precision.

Open vocabulary alone made things worse, and the reason is instructive: with
4585 concepts the detector correctly names the robot arm (`amplifier`,
`transformer`, `computer tower`) and the scene itself (`studio shot`). Those
are not hallucinations. A closed vocabulary hid them by having no word for
them.

Caveat that must travel with these numbers: this is a 3-prop Isaac scene, not
a booth. It is a floor, not a guarantee.

---

## 4. Evaluation: what a credible claim requires

This is the part the request was most specific about, and the literature is
unusually clear here.

### LIBERO alone is no longer a defensible benchmark

**LIBERO-PRO** (arXiv:2510.03827, Zhou et al.) perturbs four dimensions:
manipulated objects, initial states, task instructions, environments. Its
headline result:

> models achieve over 90% accuracy under the standard LIBERO evaluation, [but]
> their performance collapses to 0.0% under our generalized setting

They document models grasping when the target object has been replaced with an
irrelevant item, and producing unchanged outputs given corrupted instructions
or garbage tokens. Any number reported on stock LIBERO is now read as a
memorization check unless paired with a perturbed setting.

The systems in this survey have already moved:

| system | reports on |
|---|---|
| Harness-VLA | LIBERO-Pro (+38.6 pp), RoboCasa365 (+25.4 pp), RoboTwin C2R (58.4%) |
| ASPIRE | LIBERO-Pro (+77%), Robosuite bimanual (+72%), BEHAVIOR-1K (+32%), LIBERO-Pro Long 31% vs 4% |
| Pigey | LIBERO-PRO 12.8% -> 53.3%, real FR3 16.7% -> 97.3% |
| VoLo | RoboVoLo, its own 126-task benchmark |
| TurboVLA | LIBERO (97.7%) plus RoboTwin 2.0 |

**So the credible target for wrc_demo is LIBERO-Pro, not LIBERO.**

### The harness is already honest, and that is the asset

Before comparing against anyone, the measuring instrument has to be sound.
This repo has already been through that fire, and the evidence is in the tree:

- `f9c16dc` flagged the LIBERO table as uncitable
- `0e045a7` fixed the cause: success was read from `arm._last_term`, never
  reset between episodes, so episodes the robot never ran scored 100%
- `54ebd4a` traced a 0/38 to harness misconfiguration rather than model failure
- `1337263` re-ran the whole ablation after a perception fix and published a
  result that **refuted the repo's own headline thesis**

And critically, there is a **no-op floor** already measured:

```
policy=noop   libero_10      0/50
policy=noop   libero_goal    0/50
policy=noop   libero_object  0/50
policy=noop   libero_spatial 0/50
```

A zero-action policy scores 0.0% on 200 episodes. That is the single most
important control for the failure mode LIBERO-PRO describes, and it is already
in `benchmark/results/`. Very few papers publish it.

Current recorded numbers (5 episodes/task, `max_steps` 520), **all with oracle
perception**, now stamped as such in every results file:

| suite | bare | loop | verified | OpenVLA reference |
|---|---|---|---|---|
| libero_spatial | 6/50 | 5/50 | 18/50 | 43/50 |
| libero_object | 20/50 | 10/50 | 20/50 | 45/50 |
| libero_goal | 9/50 | 4/50 | 10/50 | 37/50 |
| libero_10 | 0/50 | 0/50 | 0/50 | 31/50 |

The OpenVLA column is not oracle-fed: that policy runs directly against the
env and never reads the belief store. So the two columns are not measuring the
same thing, and the gap between them is not a like-for-like deficit.

These are sober and well below the reference policy. Publishing them as-is,
with the no-op floor and the oracle label beside them, is more credible than a
headline number without controls. **The honest framing is that wrc_demo
currently measures orchestration on a Franka in LIBERO, with perception held
perfect.**

### What a paper-grade claim needs here

1. **State what is being compared.** wrc_demo is an orchestrator, so its
   comparison class is Pigey / VoLo / ASPIRE / Harness-VLA (orchestration over
   frozen skills), not OpenVLA or TurboVLA (policies).
2. **Report the no-op floor next to every table.** Already available.
3. **Report on LIBERO-Pro**, and say plainly if a suite scores 0.
4. **Report n and the interval.** At 5 episodes/task, 9/50 vs 10/50 is one
   episode. `1337263` already makes this point internally; it must survive
   into anything published.
5. **Separate the embodiment claim.** LIBERO is a 7-DoF Franka; the booth arm
   is a 6-DoF B601-RS. Skills and harness are identical, the embodiment is not.
6. **Publish the perception metric separately**, against physics truth, since
   that is where this repo has an unusually strong instrument that most papers
   lack.

### BLOCKING: the current LIBERO numbers use oracle perception

Before any of the tables above are compared to Pigey, ASPIRE or Harness-VLA,
this has to be stated plainly, because it decides whether the comparison is
legitimate at all.

`benchmark/libero/run_wrc.py` does not run perception during LIBERO episodes.
It injects ground-truth object poses straight from the simulator into the
belief store (`seed_beliefs()`, line 319), republished continuously in a
background loop:

```python
p = object_pos(name)          # inner.sim.data.body_xpos -- simulator truth
rt.beliefs.update(label=name, position=np.asarray(p, float), conf=0.99, ...)
```

The docstring is candid that this is a perception substitute and argues it
keeps the experiment about the orchestration loop. It also correctly notes
that verification still reads physics again after the motion, so a skill
cannot confirm itself. Both points are fair.

**But the stated justification does not survive checking.** The docstring says:

> Pigey and ASPIRE do the same thing -- they supply perception externally
> (Gemini Robotics ER) and study the loop around it.

This was checked against the full ASPIRE paper (arXiv:2607.00272, downloaded
and text-extracted, 11 pages + appendix). It says the **opposite**, in its own
agent instructions, verbatim:

```
## FORBIDDEN APIs
**These APIs access simulator ground truth and are STRICTLY FORBIDDEN in all
fix code, debug scripts, and skill implementations.** Using them invalidates
benchmark results, as they don't transfer to real robots.

[FORBIDDEN] env.handle.env.sim   - no MuJoCo sim object access
[FORBIDDEN] sim.data.body_xpos   - no ground-truth object positions
```

`sim.data.body_xpos` is the exact call `seed_beliefs()` uses. ASPIRE also
forbids reading `.bddl` / `.xml` / `.urdf` assets to infer geometry, and states
the rule of thumb: *"If a real robot with a camera could do it, it's allowed.
If it reads the physics engine's internal state, it's forbidden."*

What ASPIRE uses instead is a segmentation call from pixels:

```
masks = sam3(rgb, "bowl")   # returns >= 2 masks
```

For Pigey: its project page has no arXiv entry yet and does not mention Gemini,
Robotics ER, ground truth, oracle or privileged state anywhere. Its stated
protocol is "hold the robot, cameras, scenes, demonstrations, and policy
weights fixed; change only the inference-time process", which reads as
camera-based. Unverified either way, but nothing supports the claim.

And VIA states the opposite of privileged access in its abstract:

> the agent receives no robot-specific fine-tuning and **no access to
> privileged state information**

So the justification is not merely unverified. For ASPIRE it is **refuted by
the source**: doing this would, in their words, invalidate benchmark results.

**Consequences, and none of them are optional:**

1. The bare / loop / verified tables above measure **orchestration given
   perfect perception**. They are not comparable to any number from a system
   that perceives its own scene.
2. Publishing them next to Pigey's 53.3% on LIBERO-PRO without this caveat
   would be the exact failure mode LIBERO-PRO was written to expose: a number
   that looks like task competence but encodes a shortcut.

### What the camera-mode number actually is

`--perception camera` now exists and was run. The result, stated plainly:

```
libero_spatial task 0, 1 episode, skill_only, --perception camera
  -> 0/1
  -> grasp failed: localize 'black bowl' failed:
     no detections for ['black bowl', 'bowl']
```

Two bugs had to be fixed first, and both were themselves privileged-information
leaks worth recording:

- the harness passed the MuJoCo body name (`akita_black_bowl_2_main`) to the
  detector. No camera can emit that string; it is scene structure. Camera mode
  now converts it to what a person would say (`black bowl`) via `_human_label`.
- the detector was pinned to `mock` unconditionally, so even camera mode ran
  no perception at all.

With both fixed, the failure is genuine and it is a **domain gap, not a bug**.
Probing the detector directly on a LIBERO `agentview` frame, across all four
channel/flip orderings, it produces 38 to 44 detections at conf >= 0.05:

```
BGR + flipped:  first-aid kit 0.863, studio shot 0.818, robot 0.752,
                weight scale 0.746, paperweight 0.542, disco ball 0.498, ...
```

It sees plenty. It never once says `bowl`, `plate` or `ramekin`: zero
occurrences across every ordering. YOLOE's prompt-free vocabulary was not
trained on LIBERO's rendered kitchen assets, and an explicit `bowl` prompt
returns nothing either.

So the honest statement is:

> With its own perception, wrc_demo scores 0 on LIBERO because the open-
> vocabulary detector does not recognise LIBERO's rendered objects. This is a
> perception domain gap, measured, not an orchestration result.

That is a real and publishable finding, and it is the number that would be
comparable. It also explains why ASPIRE pairs its agent with SAM3 rather than
a detector vocabulary: SAM3 segments what it is pointed at instead of needing
the class to exist in a fixed concept list.

3. The remaining options are: adopt a segmenter for benchmark scenes (ASPIRE's
   choice), keep the oracle but label it in every table (now enforced in the
   JSON), or report the camera-mode zero with this explanation attached.

---

## 5. Serving: vllm-omni

Verified from the v0.26.0 release notes (2026-08-03):

- **Cosmos3 Edge and Distilled** are explicitly in the expanded model coverage
  (#5001, #5035, #5411), with "Edge/Distilled checkpoints and presets,
  CPU-offload paths, NPU recipes, ModelOpt FP8 loading" (#5313, #5596, #5076)
- rebased onto vLLM 0.26.0 (#5443)

wrc_demo currently serves Cosmos3-Edge through `scripts/serve_cosmos_vllm.sh`
on `vllm==0.25.*`, and that script encodes **six** workarounds: transformers
git main for the `cosmos3_edge` arch, a manual re-export out of the diffusers
layout, `--enforce-eager` to dodge a compile-warmup assert, a patched
`get_rope_index`, a ninja-on-PATH requirement, and a warning not to install
cosmos-framework.

First-class Cosmos3-Edge support upstream plausibly removes most of that. That
is a real maintenance win for a booth machine.

Two cautions before switching:

- neither `vllm` nor `vllm-omni` is installed in the shared venv today; the
  serving venv is separate (`~/.venvs/vllm`), which is the right pattern and
  should stay
- the release's headline memory figures were measured on **Ascend 910B3**, and
  the notes say explicitly to treat them as reference results for that
  environment, not universal guarantees. GB10 / sm_121 is not that environment
- there is a breaking change (GGUF diffusion moved to a plugin) that does not
  affect this use but signals the release is not a drop-in

Recommended as a **bounded experiment**: install into a fresh venv, serve
Cosmos3-Edge, compare against the current 6-workaround script on startup
success, tokens/s and VRAM. Keep the old script until the new one wins.

---

## 6. Proposed work, in order, each with a falsifiable target

**Phase 1. Publish the perception result properly.**
Re-run `eval_detector.py` across more scenes (props moved, added, removed) and
report mean/worst rather than a single static scene. Target: precision stays
>95% with 5+ props and at least one non-YCB object.

**Phase 2. LIBERO-Pro harness.**
Extend `benchmark/libero/run_wrc.py` to the LIBERO-PRO perturbation axes.
Keep the no-op floor for every axis. Target: a table with bare / verified /
no-op columns on all four perturbation dimensions. This is the deliverable
that makes any comparison to Pigey, ASPIRE or Harness-VLA legitimate.

**Phase 3. Close the orchestration gap where the papers agree it is.**
The convergent finding across Pigey (12.8 -> 53.3 frozen), VIA (96.7 with no
fine-tuning), ASPIRE (14 -> 62 from trace exposure) is that the win is in
monitor / halt / redirect, not in the module. wrc_demo has the pieces; what it
lacks is VoLo's **interruptible** tool contract: skills currently run to
completion. Target: a skill can be halted mid-execution on a monitor signal,
measured as reduced time-to-recovery on a deliberately perturbed episode.

**Phase 4. TurboVLA behind `VLAExecutor`.**
Reproduce their LIBERO number in this harness before believing it, then wire
it as an alternative backend for `grasp_object`. Target: match the analytic
OBB pipeline on the rig ablation with no regression in the harness-gated
safety path.

**Phase 5. DINOv3 duplicate rejection, only if Phase 1 leaves duplicates.**
Class-agnostic, no enrollment. Target: beat the geometric fusion already
landed on the same metric. If it does not, drop it.

**Not recommended:** replacing YOLOE with DINOv3 (no text head, kills open
vocabulary), putting Qwen-3D in the 3 Hz loop (3B/7B, no published latency,
trained on ScanNet rooms not 0.9 m tabletops), or reporting any headline
number on stock LIBERO.

---

## 7. Correction to carry into any publication

`docs/ARCHITECTURE.md` describes Agentic-VLA as "ICML 2026, LLM sub-goal
decomposition, zero-shot VLM exploration critic, embedding-indexed experience
memory". arXiv:2605.22896 is Jin and Zhang, *Agentic-VLA: Efficient Online
Adaptation for Vision-Language-Action Models*, about adaptive reward synthesis
and online adaptation. Either a different paper is meant or the description is
wrong. This must be resolved before it reaches a submission.

---

## Sources

| work | id | verified how |
|---|---|---|
| LIBERO-PRO | arXiv:2510.03827 | abstract |
| Harness-VLA / RPent | arXiv:2607.08448 | abstract + **local code** at ~/Projects/RPent |
| ASPIRE | arXiv:2607.00272 | abstract + project page |
| VoLo | chicychen.github.io/VoLo | project page |
| Pigey | lianegalanti.github.io/Pigey | project page |
| VIA | arXiv:2607.11119 | abstract |
| HumanCLAW | human-claw.github.io | project page |
| TurboVLA | arXiv:2607.27205 | abstract + GitHub repo metadata |
| Qwen-3D | qwen-3d.github.io | project page + repo listing |
| Agentic-VLA | arXiv:2605.22896 | abstract (see correction above) |
| Waddle Labs | waddlelabs.ai | landing page only, no technical detail |
| vLLM-Omni | v0.26.0 release notes | GitHub API |
