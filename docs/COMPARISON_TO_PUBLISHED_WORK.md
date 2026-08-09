# wrc_demo against the published agentic-manipulation results

Status: 2026-08-09. Every number attributed to another system is copied from
the paper PDF in `~/research/sota/papers/`, with the line it came from. Every
number attributed to wrc_demo was produced on this machine by
`benchmark/libero/run_wrc.py` and is stored in `benchmark/results/`.

Read the "What is NOT comparable" section before quoting anything here. Three
of the four axes do not line up yet, and pretending otherwise is exactly the
failure this repo has been trying to avoid.

---

## 1. The benchmark most of them report is LIBERO-Pro, not LIBERO

This is the single most important fact for positioning wrc_demo, and it took
reading the PDFs to find it.

- **Harness-VLA** (`2607.08448`, Table 3) reports LIBERO-**Pro**: 10 tasks x
  10 seeds = 100 trials per cell, under instruction-redirection (T) and
  position-swap (S) perturbations.
- **ASPIRE** (`2607.00272`, Fig. 4a and 3.3) also reports LIBERO-**Pro**,
  learning on a debug seed split and evaluating on held-out seeds.
- **LIBERO-PRO** (`2510.03827`) is the paper that introduced it, precisely
  because standard LIBERO scores are saturated and misleading.

wrc_demo currently runs **standard LIBERO**. `~/bench/LIBERO-PRO` is being
cloned; until it runs, no cell in this document is a like-for-like comparison
against the LIBERO-Pro columns.

## 2. What the gap looks like on standard LIBERO

Harness-VLA Table 2, success rate (%), 100 trials per suite (10 tasks x 10
seeds) -- the same protocol size as the wrc_demo runs:

| method | Spatial | Object | Goal | LIBERO-10 |
|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 |
| NORA | 85.6 | 89.4 | 80.0 | 63.0 |
| pi-0 | 96.8 | 98.8 | 95.8 | 85.2 |
| piRLinf | 99.0 | 96.0 | 97.0 | 89.0 |
| AtomVLA | 96.4 | 99.6 | 97.6 | 94.4 |
| Harness VLA (CC) | 97.0 | **100.0** | 94.0 | 93.0 |
| **wrc_demo (oracle, verified)** | **0.0** | not run | not run | not run |
| **wrc_demo (camera, verified)** | **0.0** | not run | not run | not run |
| no-op floor | 0.0 | 0.0 | 0.0 | 0.0 |

wrc_demo scores zero. The decomposition in `LAYER_ATTRIBUTION_LIBERO.md` says
why, and it is not a mystery: LIBERO's `On()` predicate needs the object
within **3 cm** of the destination, and median placement error is **1.8 cm
against its own aim point** but **12.4 cm to the actual destination**, because
gross failures (15-30 cm) and destination-shoving dominate the tail.

Stating it plainly: as a manipulation system on this benchmark, wrc_demo is
not competitive with a fine-tuned VLA, and nothing here should be read as
claiming otherwise.

## 3. Where the numbers do favour this repo

### 3.1 wrc_demo on LIBERO-Pro, measured

Run on this machine with LIBERO-Pro's own suites, 100 episodes per cell
(10 tasks x 10 seeds), the same protocol Harness-VLA Table 3 uses.
`libero_spatial_lan` is the instruction-redirection cell (Spat-T),
`libero_spatial_swap` is the position-swap cell (Spat-S).

| method | Spat-T | Spat-S |
|---|---|---|
| OpenVLA | 0.0 | 0.0 |
| pi-0 | 0.0 | 0.0 |
| NORA | 0.0 | 0.0 |
| MolmoAct | 0.0 | 0.0 |
| pi-0.5 | 1.0 | 20.0 |
| AtomVLA | 1.0 | 16.0 |
| Cap-X | 14.0 | 12.0 |
| RATS | 31.0 | 29.0 |
| Harness VLA (CC) | **94.0** | **80.0** |
| **wrc_demo (verified, oracle)** | **0.0** | **0.0** |

On success rate wrc_demo sits with OpenVLA, pi-0, NORA and MolmoAct, which
also score 0.0 on these cells. The cause is known and is not robustness: place
precision misses LIBERO's 3 cm predicate (see `LAYER_ATTRIBUTION_LIBERO.md`).

Note the oracle caveat still applies: wrc_demo's row uses seeded poses, which
the papers forbid. It is not a like-for-like perception setup, and a camera
row would be worse, not better.

### 3.2 The measurement none of them report

Harness-VLA Table 3, LIBERO-Pro, aggregate success (%):

| method | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | Overall |
|---|---|---|---|---|---|---|---|
| OpenVLA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| pi-0 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.3 |
| pi-0.5 | 1.0 | 20.0 | 1.0 | 17.0 | 2.0 | 38.0 | 11.0 |
| NORA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Cap-X | 14.0 | 12.0 | 18.0 | 22.0 | 17.0 | 26.0 | 18.2 |
| RATS | 31.0 | 29.0 | 63.0 | 61.0 | 36.0 | 43.0 | 43.8 |

Models scoring 85-100% on standard LIBERO collapse to 0.0% under
perturbation. That collapse is the entire argument of the LIBERO-PRO paper,
and it is why the agentic systems exist.

wrc_demo attacks that collapse from a different angle: not "score higher
under perturbation" but "know when you failed".

| system | reports no-op floor | separates perception vs orchestration error |
|---|---|---|
| ASPIRE | no | no |
| Harness-VLA | no | no |
| VIA | no | no |
| Agentic-VLA | no | no |
| LIBERO-PRO | no | no |
| **wrc_demo** | **yes, 0/50** | **yes** |

False success claims per 100 episodes, oracle perception:

| suite | bare | verified | reduction |
|---|---|---|---|
| LIBERO standard, spatial | 48 | 5 | **90%** |
| LIBERO-Pro Spat-T (lan) | 50 | 5 | **90%** |
| LIBERO-Pro Spat-S (swap) | 51 | 2 | **96%** |

The 90% replicates exactly on a different benchmark under instruction
redirection, which makes it an effect rather than a single-suite number. Under
the bare path perturbation makes things slightly worse (48 -> 50 -> 51) while
the verified path holds or improves.

### 3.3 The swap suite found a real hole, and closing it is the result

The swap numbers above are POST-FIX. The first measurement was 13/100 verified
false claims against 5/100 everywhere else, and that regression is worth more
than the headline because of what it exposed. Traced to
`libero_spatial_swap` task 3:

```
skill aimed at        [ 0.016, -0.267]
true destination      [-0.009, -0.290]     3.4 cm away
object landed         4.2 cm from the true destination
postcondition         "0.8 cm from the requested drop point (physics)" -> CONFIRMED
LIBERO predicate      needs < 3 cm         -> FAILED
```

Two independent defects, both invisible on unperturbed suites:

1. **The check scored against the skill's own aim point.** It answered "did
   the object reach where I aimed?", not "did it reach the destination?". A
   stale aim confirmed itself. Placement was excellent (0.8 cm); the aim was
   3.4 cm off because the swap moved the destination after the belief was
   seeded and nothing re-checked the target before release.

2. **The threshold was looser than the benchmark's own predicate.** After
   fixing (1) it still confirmed: the check used `SAME_PLACE_M * 2` = 10 cm
   while LIBERO's `On()` requires 3 cm. One constant was answering two
   different questions, "did it move at all" (where being generous is right)
   and "did it arrive" (where it is not).

Fixes: `place_on_object` re-localizes the destination immediately before
committing to a drop point, `_check_relocated` scores against a named
destination's CURRENT pose, and `DEST_TOLERANCE_M = 0.03` is now separate from
`SAME_PLACE_M` and matched to the predicate.

Result: **13 -> 2** verified false claims, now the best of the three suites.
A verifier looser than the task's success criterion cannot catch a near miss,
which is the failure mode that matters: the arm does something plausible and
slightly wrong, then reports success.

None of the surveyed papers reports this number, so none of them would have
detected either defect.

## 4. What is NOT comparable, and why

Four axes. Only one currently lines up.

| axis | published work | wrc_demo | comparable? |
|---|---|---|---|
| robot | Franka Panda, 7-DoF | Franka Panda, 7-DoF (`configs/arms/libero_panda.yaml`) | **yes** |
| benchmark | LIBERO-Pro | LIBERO-Pro (Spat-T, Spat-S) + standard | **yes, for those two cells** |
| protocol | 10 tasks x 10 seeds = 100 trials | 10 tasks x 10 seeds = 100 trials | **yes** |
| action space | OSC_POSE (7-D deltas) | JOINT_POSITION + own IK | **no** |
| perception | RGB-D + SAM/GraspNet, ground truth forbidden | oracle poses, or camera that fails to localize | **no** |

On the robot axis the harness now loads a Panda profile rather than patching
the reBot's `mock` profile at runtime, so DoF, joint limits, gripper width and
home pose all describe the machine actually being driven.

On perception: ASPIRE forbids `sim.data.body_xpos` by name, Harness-VLA states
"No ground-truth poses or simulator internals", VIA states "no access to
privileged state information". Every oracle row here uses exactly what those
three forbid, and is stamped `perception: oracle` in its JSON for that reason.

## 5. What would make this a real comparison

In dependency order:

1. **Run LIBERO-Pro.** Same suites, same 10x10 protocol, with the
   perturbations. Without this there is no shared x-axis with Table 3.
2. **Switch to OSC_POSE**, or state the difference every time a number is
   quoted. Placement error is a property of the controller, and this repo
   currently uses a different one from every paper cited here.
3. **Close the camera path.** The camera rows are 0 CLAIMS, not 0 successes:
   episodes die at `localize failed: no detections`. Either pixel addressing
   driven by a real agent (built, unused because the harness calls skills
   directly with `llm=mock`) or a text-promptable segmenter.
4. **Fix the gross place failures.** Median aim error is already 1.8 cm; the
   15-30 cm tail is what keeps success at zero.

Until at least 1 and 3 land, the honest one-line summary is:

> wrc_demo is not yet comparable to published agentic-manipulation results on
> success rate. What it has that they do not is a measured no-op floor and a
> measured separation of perception error from orchestration error, including
> a 90% reduction in false success claims from its verification layer.
