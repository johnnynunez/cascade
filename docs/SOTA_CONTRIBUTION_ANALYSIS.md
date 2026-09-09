# Where cascade can actually advance the state of the art

Status: 2026-08-08. Written after downloading and text-extracting the six
papers (`~/research/sota/papers/*.pdf`) and cloning the available code
(`~/research/sota/code/`, plus RPent at `~/Projects/RPent`). Every claim below
cites a line that can be checked in those files.

This note answers one question: given that these systems exist, what can this
repo contribute that is not already done better elsewhere?

---

## 1. The three-way agreement nobody in this repo had checked

All three agentic-manipulation systems that publish an agent contract forbid
privileged simulator state, in near-identical words.

**ASPIRE** (`2607.00272`), in the agent instructions shipped with the paper:

```
## FORBIDDEN APIs
**These APIs access simulator ground truth and are STRICTLY FORBIDDEN in all
fix code, debug scripts, and skill implementations.** Using them invalidates
benchmark results, as they don't transfer to real robots.

[FORBIDDEN] env.handle.env.sim   - no MuJoCo sim object access
[FORBIDDEN] sim.data.body_xpos   - no ground-truth object positions
```

**Harness-VLA / RPent** (`2607.08448`, line 1862), under "Perception isolation":

> No ground-truth poses or simulator internals; localize from RGB-D and world
> maps.

**VIA** (`2607.11119`, line 169):

> The agent observes only what the interface renders and has no access to
> privileged simulator state. For simplicity, we also do not provide any
> perception-assisting APIs such as segmentation functions, so visual
> understanding is solely the job of the agent and its underlying model.

cascade's LIBERO harness was doing exactly the forbidden thing until this
session (see `SOTA_PERCEPTION_AND_EVALUATION.md`). That is now behind
`--perception oracle|camera`, and every historical result file is stamped.

---

## 2. What each system's evaluation actually is

Read from the PDFs rather than the project pages, because the protocol is
where comparability lives.

| system | benchmark | protocol | n |
|---|---|---|---|
| **ASPIRE** | LIBERO-Pro (Object/Goal/Spatial x Pos/Task) | learns on seeds 51-65, evaluates one generated program on seeds **1-50** | 10 tasks x 50 seeds |
| **Harness-VLA** | LIBERO-Pro, RoboCasa365, RoboTwin C2R | 100 trials per suite (10 tasks x 10 seeds) | 100/suite |
| **VIA** | 3 tasks (2 from LIBERO-Goal, 1 robosuite) | 10 seeds per task, varying initial placements | **30** |
| **Pigey** | LIBERO-PRO + real FR3 | not published (no paper yet) | unknown |
| **TurboVLA** | LIBERO + RoboTwin 2.0 | standard LIBERO | standard |

Two things worth internalising:

**VIA's headline 96.7% is 29/30.** Three tasks, ten seeds. It is an existence
proof that a general agent can drive a robot through a good interface, not a
broad competence claim. Anyone comparing against it should say so.

**ASPIRE's protocol is the strict one**: one program per task, generated on a
disjoint learning split, then frozen and run on 50 held-out seeds. That is the
bar to match if the goal is a credible LIBERO-Pro number.

---

## 3. The gap: nobody measures perception separately

Searched all five papers for any separate perception metric scored against
physics truth:

| paper | "perception precision/recall", "phantom", "false positive rate" | "ground-truth pose / localization error / perception ablation" |
|---|---|---|
| ASPIRE `2607.00272` | 0 | 0 |
| Harness-VLA `2607.08448` | 0 | 1 (a *prohibition*, not a metric) |
| VIA `2607.11119` | 0 | 0 |
| LIBERO-PRO `2510.03827` | 0 | 0 |
| Agentic-VLA `2605.22896` | 0 | 0 |

And for an execution floor:

| paper | "no-op policy", "zero action", "null policy" |
|---|---|
| all four | **0** |

**Nobody publishes a no-op floor. Nobody isolates perception error from
orchestration error.** Every one of these systems reports end-to-end task
success and attributes failures qualitatively.

This repo already has both instruments, built before this survey:

- `scripts/eval_detector.py` scores precision / recall / flicker /
  localization against Isaac PhysX poses via `TruthPoseReader`. The poses come
  from the physics engine, independently of perception, which is exactly what
  makes it a measurement rather than a held-out split the detector may have
  memorised.
- `benchmark/results/noop_*.json`: a zero-action policy scoring **0/50 on all
  four LIBERO suites**, 200 episodes.

That combination is the contribution. Not a better VLA, not a better
orchestrator: **a way to tell which layer is failing**, on a benchmark where
everyone else reports a single end-to-end number.

### Why this matters beyond bookkeeping

This repo has already used the instrument to overturn its own published claim.
Commit `1337263`:

> one perception change moved success 4/10 to 10/10. The entire
> verification-and-retry stack had moved it 4 to 5 to 6

A 1.6 cm sensor-model bias in `_recentre_by_size` was responsible for what had
been written up as an orchestration finding. With only end-to-end numbers, that
error is invisible: the verification layer *looked* like it was working,
because it was converting a perception failure into an honest failure report.

Every system in section 2 is exposed to that same confound and has no
instrument to detect it.

---

## 4. Two ideas worth taking, with attribution

### VIA's interface, which this repo half-built already

VIA's central claim is that frontier agents are already capable robot
controllers *given a readable interface*, and it deliberately withholds
perception APIs to prove the interface is doing the work. Its interface is a
navigable RGB-D point cloud plus click-to-teleport, rotate-by-gizmo, and
`execute_waypoint`.

cascade has `visual_interface.py` (VIA-style annotated frame) and `probe.py`
(the Anthropic cursor ablation), but its agent still acts through
`grasp_object(label)`. The genuinely VIA-shaped missing piece is
**click-to-act**: let the agent name a pixel and have the system resolve it to
a 3D waypoint, instead of requiring a label the detector must first recognise.

That directly addresses the measured failure in section 5: on LIBERO the
detector never emits `bowl`, so a label-based interface cannot express the
task. A pixel-based one can, and `PointProbe.probe(u, v)` already does the
back-projection.

### ASPIRE's disambiguation rule, which converges with existing code

ASPIRE's skill library contains:

> Detect all instances, sort by the axis implied by the qualifier (X for
> front/back, Y for left/right in robot frame), then select by keyword index.

cascade had the axis map (`_SPATIAL_AXES`) but nothing parsed the user's
phrase into it. `perception/reference.py` (added this session) closes that,
covering VoLo's four reference types: spatial, ordinal, size, negation.

---

## 5. The measured obstacle, stated plainly

`--perception camera` on LIBERO scores **0**, and the cause is a perception
domain gap, not orchestration:

```
BGR + flipped:  first-aid kit 0.863, studio shot 0.818, robot 0.752,
                weight scale 0.746, paperweight 0.542, ...
```

38 to 44 detections per frame at conf >= 0.05, across all four channel/flip
orderings, and **zero occurrences of `bowl`, `plate` or `ramekin`**. YOLOE's
prompt-free vocabulary does not cover LIBERO's rendered kitchen assets, and an
explicit `bowl` prompt returns nothing either.

This is precisely why ASPIRE, VoLo and RPent all pair their agent with SAM3
rather than a detector vocabulary: a segmenter localises what it is pointed at,
without needing the class to exist in a fixed concept list.

---

## 6. What to build, in order

**1. Pixel-addressed acting (VIA). BUILT, AND IT FOUND ITS OWN LIMIT.**
`perception/pixel_target.py` + `grasp_at_pixel` let the agent act on a pixel
with no class name anywhere in the path. Depth-connectivity segmentation, zero
new dependencies, 13 tests.

Measured on a real LIBERO agentview frame against MuJoCo body poses:

| probe | region | extent | error vs truth |
|---|---|---|---|
| bowl pixel (217, 135) | 10000 px (**cap reached**) | 102 cm | **34 cm** |
| plate pixel (201, 191) | 10000 px (**cap reached**) | 83 cm | **22 cm** |

Both hit the region cap, which means the flood fill escaped onto the table.
Depth connectivity cannot separate an object from the surface it rests on: the
depth step across the contact line is smaller than the segmentation threshold.

Before concluding that, the camera convention was eliminated as a cause by
sweeping all four flip matrices x image flip against ground truth. `flip_yz` +
vertical flip won by a wide margin (16.7 cm vs 46-203 cm for the others), so
the convention is right and the residual error is the method, not the framing.

The code now **raises** on a capped fill rather than returning a 34 cm-wrong
pose, because that pose would drive the arm into the table. Raising the cap
would make the wrong answer bigger, not righter.

So the honest status of idea 1: the addressing mechanism is correct and
useful, and it works for objects with a depth gap all round. The zero-
dependency segmenter behind it is not sufficient for tabletop scenes. That is
a negative result with a number attached, and it directly motivates step 2
rather than leaving it as a preference.

**2. SAM3 as an on-demand tool (VoLo / ASPIRE / RPent).** Now the load-bearing
step, not an alternative: it is the piece that separates touching objects by
appearance, which is exactly the information depth alone lacks. It slots
behind the same `fix_from_pixel` interface, so `grasp_at_pixel` and all its
tests keep working unchanged.
Target: bowl-pixel error under 6 cm on the same frame, same measurement.

**3. Publish the layer-attribution result.** Run ASPIRE's protocol (10 tasks x
50 held-out seeds) in three conditions: oracle perception, camera perception,
no-op floor. The deliverable is not a headline success rate; it is the first
published decomposition of how much of an agentic manipulation score is
perception and how much is orchestration.

**4. Only then, LIBERO-Pro perturbations.** With the decomposition in hand,
the perturbation axes become interpretable: an Object-perturbation collapse
with oracle perception intact means the orchestrator memorised, which is
exactly LIBERO-PRO's thesis, but demonstrated rather than asserted.

---

## 7. Honest limits of this note

- Pigey has no paper; its protocol is unknown and it should not be quoted as a
  precise baseline.
- ASPIRE's numeric tables did not survive `pdftotext` cleanly (column layout).
  The prose claims quoted here are reliable; specific per-cell numbers should
  be read from the PDF before publication.
- Waddle Labs publishes no technical detail at all.
- `2605.22896` (Agentic-VLA) remains inconsistent with the description in
  `docs/ARCHITECTURE.md`; that citation still needs fixing.

## Sources on disk

```
~/research/sota/papers/2605.22896.{pdf,txt}   Agentic-VLA
~/research/sota/papers/2607.00272.{pdf,txt}   ASPIRE
~/research/sota/papers/2607.08448.{pdf,txt}   Harness-VLA / RPent
~/research/sota/papers/2607.11119.{pdf,txt}   VIA
~/research/sota/papers/2607.27205.{pdf,txt}   TurboVLA
~/research/sota/papers/2510.03827.{pdf,txt}   LIBERO-PRO
~/research/sota/code/TurboVLA/                Apache-2.0
~/research/sota/code/qwen3d/
~/Projects/RPent/                             Harness-VLA implementation
```
