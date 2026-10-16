# Benchmarks

**Next action:** create LIBERO's own venv (it pins `numpy<2` and `transformers==4.40.1`, which
conflict with the demo stack; about 10 min):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO ~/bench/LIBERO
uv venv --python 3.10 ~/.venvs/libero
uv pip install --python ~/.venvs/libero/bin/python \
    -r ~/bench/LIBERO/requirements.txt robosuite==1.4.1 mujoco==3.2.3

export CASCADE_BENCH_LIBERO=~/bench/LIBERO
export CASCADE_BENCH_MODELS=~/models          # openvla-7b-libero-* checkpoints
export CASCADE_BENCH_VENV=~/.venvs/libero/bin/python
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
```

Results are scoped to their recorded source, dataset, hardware and execution path; the
[capability status](ROADMAP.md) separates software tests, native proofs, campaigns and restart.
All locations are in `benchmark/paths.py` and overridable by env var.

```
benchmark/
  paths.py            all locations, overridable by env var
  libero/             LIBERO experiments
    backend.py          LiberoArm + LiberoCamera (ArmBase / CameraBase impls)
    kinematics_mj.py    MujocoKinematics (FK/IK off the Panda MJCF)
    run_wrc.py          *** cascade's own runtime driving LIBERO ***
    run_baseline.py     OpenVLA-7B reference numbers
    orchestration_gap.py  frozen primitive, with and without verification
    sweep.sh            all four suites
  rig/
    ablation.py         cascade layer ablation on the Isaac B601-RS rig
  report/
    comparison_table.py console tables + Wilson intervals
    make_pdf.py         paper-style PDF
    report.py           OpenVLA aggregate vs published
  diagnostics/          source-bound bridge, mapping and acceptance diagnostics
  results/              measurements (JSON)
```

Setup traps:

1. **LIBERO's first import calls `input()`** and dies with `EOFError` headless: write
   `~/.libero/config.yaml` by hand.
2. **`torch.load` of the init states** needs `weights_only=False` on torch 2.13+ (patch
   `libero/libero/benchmark/__init__.py`).
3. **Two EGL vendors** (nvidia + mesa) give `EGLError`: pin `MUJOCO_EGL_DEVICE_ID=0`.
4. **One checkpoint per suite:** the spatial checkpoint's `dataset_statistics.json` has only the
   `libero_spatial` key; reusing it on another suite mis-scales every action *silently*.

## Run

1. **cascade on LIBERO** (cascade's real runtime: `SkillRuntime.execute`, every `skill_*`,
   `SafeArm` + `SafetyHarness`, the grasp pipeline, `PostconditionChecker`, `BeliefStore`,
   `TraceLogger`; only three backends adapted):

   ```bash
   $CASCADE_BENCH_VENV benchmark/libero/run_wrc.py \
       --suite libero_spatial --tasks 10 --episodes 5 \
       --conditions skill_only,verified
   ```

2. **OpenVLA reference:** `bash benchmark/libero/sweep.sh` (all four suites), then
   `$CASCADE_BENCH_VENV benchmark/report/report.py` (table vs published).
3. **Layer ablation on the Isaac rig** (bridge on `:8611`; **restart the bridge first**: one that
   has served hundreds of episodes reports stale poses):

   ```bash
   cd models && PYTHONPATH=../src uv run python ../benchmark/rig/ablation.py \
       --conditions skill_only,verify_only,verify_retry --states 10
   ```

4. **Reports:** `uv run python benchmark/report/comparison_table.py` (console),
   `uv run python benchmark/report/make_pdf.py` (PDF).

| cascade contract | LIBERO implementation | validated |
|---|---|---|
| `ArmBase` | `LiberoArm` (7-DoF Franka, stepped sim) | drives the arm |
| `Kinematics` | `MujocoKinematics` | **FK 0.0 mm, IK 0.1 mm** |
| `CameraBase` | `LiberoCamera` (agentview RGB-D) | real intrinsics |

LIBERO measures cascade's orchestration and verification on a 7-DoF Franka, *not* cascade on its
own 6-DoF arm (`ArmBase.n_joints` is a class attribute so backends can differ).

## Results

### cascade layer ablation (10 initial states, physics-judged, fresh bridge)

After fixing a 1.6 cm perception bias (`_recentre_by_size`, commit `c129161`):

| | condition | success | self-claimed | **false claims** | mean s |
|---|---|---|---|---|---|
| A | skill only | 10/10 | 10/10 | **0** | 23.5 |
| B | + verification | 9/10 | 9/10 | **0** | 26.6 |
| C | + retry | 9/10 | 9/10 | **0** | 28.2 |

1. Before the fix the same ablation scored 4/10, 5/10, 6/10 (overlapping intervals) with **4**
   false claims for the bare skill (claimed 8, achieved 4) and 0 with verification.
2. With perception fixed the bare skill also reports 0 false claims: the lies were a symptom of a
   misread failed grasp, not of running unverified. One perception change (4 → 10/10) did more
   than the whole verification-and-retry stack.
3. Verification is now slightly negative on success (10 → 9, one episode, not significant) and
   costs 3–5 s per episode. It is insurance whose payout depends on how broken the rest of the
   stack is, not a substitute for fixing the root cause.
4. Instrument warning: an earlier run's 3 false claims in C came from a long-running bridge
   returning stale poses; two real bugs were fixed (`tests/test_truth_channel.py`) and a sanity
   guard now rejects impossible poses (up to 3171 m of "displacement" were seen).

### LIBERO (n=50 per cell)

> **The primitive columns are stale (inflated by a fixed harness bug) and need a re-run.** Success
> was read from `arm._last_term`, which was never reset between episodes, and once `_terminated`
> is set `LiberoArm._step` returns `True` without stepping, so the confirmation loop confirmed
> nothing. Success now comes from LIBERO's own `_check_success()` (isolated in
> `benchmark/diagnostics/last_term_leak.py`). **The OpenVLA column is unaffected**:
> `libero/run_baseline.py` drives `env.step()` directly.

| suite | no-op | frozen primitive | + self-retry | + verification | OpenVLA | published |
|---|---|---|---|---|---|---|
| spatial | 0% | 12% | 10% | **36%** | 86% | 84.7% |
| object | 0% | 40% | 20% | 40% | 90% | 88.4% |
| goal | 0% | 18% | 8% | 20% | 74% | 79.2% |
| libero_10 | 0% | 0% | 0% | 0% | 62% | 53.7% |
| **total** | 0% | 18% | **10%** | 24% | 78% | |

OpenVLA lands inside the Wilson interval on **4/4** suites, so the harness reproduces a published
baseline (that check exposed a gripper-convention bug). The aggregate verification gap (+6.5 pts)
is not significant. What replicated on all four suites: **retrying on self-report scores below not
retrying** (18% → 10%), because the `reached` flag comes from the code that executed the motion.
The Isaac perception fix does not touch LIBERO (0 calls to `_recentre_by_size`; LIBERO pins a mock
detector).

## LIBERO layer attribution (measured 2026-08-09)

`benchmark/libero/run_wrc.py` on `libero_spatial` (10 tasks x 10 seeds): the
no-op floor is 0/50 and every condition scores 0/100. With oracle perception the
postcondition check cuts false success claims from 48 to 5
(`benchmark/results/proto_spatial_oracle.json`); in camera mode the
open-vocabulary detector of that run emitted 38-44 detections per LIBERO frame
and never named the bowl, so those rows measure a detector domain gap.

## Diagnostics

`benchmark/diagnostics/` keeps the scripts that isolated the bridge-degradation bug (each kills one
hypothesis; a "retry resets the scene" symptom was a resource leak in the verification probe) and
the source-bound mapping and acceptance diagnostics.
