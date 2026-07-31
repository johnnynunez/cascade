# Benchmarks

Everything needed to measure this repo against the manipulation benchmarks the
literature uses, plus the ablation that measures wrc_demo's own layers.

```
benchmark/
  paths.py            all locations, overridable by env var
  libero/             LIBERO experiments
    backend.py          LiberoArm + LiberoCamera (ArmBase / CameraBase impls)
    kinematics_mj.py    MujocoKinematics (FK/IK off the Panda MJCF)
    run_wrc.py          *** wrc_demo's own runtime driving LIBERO ***
    run_baseline.py     OpenVLA-7B reference numbers
    orchestration_gap.py  frozen primitive, with and without verification
    sweep.sh            all four suites
  rig/
    ablation.py         wrc_demo layer ablation on the Isaac B601-RS rig
  report/
    comparison_table.py console tables + Wilson intervals
    make_pdf.py         paper-style PDF
    report.py           OpenVLA aggregate vs published
  diagnostics/          11 scripts that found the bridge-degradation bug
  results/              measurements (JSON)
```

## Setup

LIBERO pins `numpy<2` and `transformers==4.40.1`, which conflict with the demo
stack, so it needs **its own venv**:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ~/bench/LIBERO
uv venv --python 3.10 ~/.venvs/libero
uv pip install --python ~/.venvs/libero/bin/python \
    -r ~/bench/LIBERO/requirements.txt robosuite==1.4.1 mujoco==3.2.3

export WRC_BENCH_LIBERO=~/bench/LIBERO
export WRC_BENCH_MODELS=~/models          # openvla-7b-libero-* checkpoints
export WRC_BENCH_VENV=~/.venvs/libero/bin/python
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
```

Four known setup traps, all of which cost time:

1. **LIBERO's first import calls `input()`** and dies with `EOFError` when
   headless. Write `~/.libero/config.yaml` by hand instead.
2. **`torch.load` of the init states** needs `weights_only=False` on torch
   2.13+ (patch `libero/libero/benchmark/__init__.py`).
3. **Two EGL vendors** (nvidia + mesa) give `EGLError`. Pin
   `MUJOCO_EGL_DEVICE_ID=0`.
4. **One checkpoint per suite.** `dataset_statistics.json` for the spatial
   checkpoint contains only the `libero_spatial` key; reusing it on another
   suite mis-scales every action *silently*.

## Running

### wrc_demo on LIBERO

```bash
$WRC_BENCH_VENV benchmark/libero/run_wrc.py \
    --suite libero_spatial --tasks 10 --episodes 5 \
    --conditions skill_only,verified
```

This runs **wrc_demo's real runtime**: `SkillRuntime.execute` and every
`skill_*` method, `SafeArm` + `SafetyHarness` per-waypoint approval, the grasp
pipeline, `PostconditionChecker`, `BeliefStore`, `TraceLogger`. Only three
backends are adapted:

| wrc_demo contract | LIBERO implementation | validated |
|---|---|---|
| `ArmBase` | `LiberoArm` (7-DoF Franka, stepped sim) | drives the arm |
| `Kinematics` | `MujocoKinematics` | **FK 0.0 mm, IK 0.1 mm** |
| `CameraBase` | `LiberoCamera` (agentview RGB-D) | real intrinsics |

The Panda is 7-DoF where the B601-RS is 6. That is a property of the robot, not
a rewrite: `ArmBase.n_joints` is a class attribute precisely so backends can
differ.

**Honest framing:** this measures wrc_demo's orchestration and verification on
a Franka in LIBERO, *not* wrc_demo on its own arm. Skills, harness and
verification are identical; the embodiment is not.

### OpenVLA reference

```bash
bash benchmark/libero/sweep.sh              # all four suites
$WRC_BENCH_VENV benchmark/report/report.py  # table vs published
```

### wrc_demo layer ablation (Isaac rig)

Needs the Isaac bridge on `:8611`.

```bash
cd models && PYTHONPATH=../src python ../benchmark/rig/ablation.py \
    --conditions skill_only,verify_only,verify_retry --states 10
```

**Restart the bridge first.** See `docs/BRIDGE_DEGRADATION.md`: a bridge that
has served hundreds of episodes degrades and its verification channel starts
reporting stale poses.

### Reports

```bash
python benchmark/report/comparison_table.py   # console
python benchmark/report/make_pdf.py           # PDF
```

## Results so far

### wrc_demo layer ablation (10 initial states, physics-judged)

Measured on a **freshly restarted bridge** — see the note below.

| | condition | success | 95% CI | self-claimed | **false claims** |
|---|---|---|---|---|---|
| A | skill only | 4/10 | [17%, 69%] | 8/10 | **4** |
| B | + verification | 5/10 | [24%, 76%] | 5/10 | **0** |
| C | + retry | 6/10 | [31%, 83%] | 6/10 | **0** |

Task success climbs 4 → 5 → 6, but the intervals overlap heavily at n=10, so
that ordering is not evidence. Read the success column as flat.

The self-report column needs no statistics. **The bare skill claimed success 8
times and achieved it 4** — wrong about its own outcome in 40% of episodes.
Independent verification takes that to **zero** in both verified conditions
without changing what the robot physically does.

That is the point of the layer: a robot that fails and says so can be retried
or escalated; a robot that fails and reports success corrupts the belief
store, the skill library that learns from traces, and any operator reading the
log.

**Instrument warning.** An earlier run of this same ablation showed 3 false
claims in condition C. They were not the robot — the bridge had been up for
hours and its truth channel was returning stale poses. On a fresh bridge those
3 became 0. Chasing them found two real bugs, both fixed and now covered by
`tests/test_truth_channel.py`; see `docs/BRIDGE_DEGRADATION.md`. This run alone
still produced **four impossible poses** (up to 3171 m of "displacement"),
which the new sanity guard rejects.

### LIBERO (n=50 per cell)

| suite | no-op | frozen primitive | + self-retry | + verification | OpenVLA | published |
|---|---|---|---|---|---|---|
| spatial | 0% | 12% | 10% | **36%** | 86% | 84.7% |
| object | 0% | 40% | 20% | 40% | 90% | 88.4% |
| goal | 0% | 18% | 8% | 20% | 74% | 79.2% |
| libero_10 | 0% | 0% | 0% | 0% | 62% | 53.7% |
| **total** | 0% | 18% | **10%** | 24% | 78% | |

OpenVLA lands inside our Wilson interval on **4/4** suites, so the harness
reproduces a published baseline — that check is what exposed a
gripper-convention bug scoring a working policy at 0%.

The aggregate verification gap (+6.5 pts) is **not** significant; the intervals
overlap. The effect is local to `libero_spatial`. What *did* replicate on all
four suites: **retrying on self-report scores below not retrying at all**
(18% → 10%). The `reached` flag is computed by the same code that executed the
motion, so retries fire on the wrong episodes.

## Diagnostics

`diagnostics/` holds the scripts that isolated the bridge-degradation bug. They
are kept because the *method* is reusable: each one kills a specific
hypothesis, and together they show how a "retry resets the scene" symptom
turned out to be a resource leak in the verification probe.
