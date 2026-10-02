# Official VelStand policy: full-environment startup diagnostic

On 2026-10-02, the complete mjlab environment reproduced negligible motion at a +0.1 m/s command. A preceding +0.3 m/s phase did **not** sustain useful +0.1 m/s locomotion in this diagnostic. This does not establish a universal policy defect. [Upstream issue #59](https://github.com/pollen-robotics/microduck_rl/issues/59) reports sustained motion at +0.15 m/s; our +0.1 experiment neither reproduces nor refutes that specific claim.

The private environment used [MicroDuck RL source `8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec`](https://github.com/pollen-robotics/microduck_rl/tree/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec), its unchanged frozen lock, mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, Warp 1.12.0 and BAM `62bd8ce12154340be97e06f7f41a0ca8f116d967`. The original [published `velstand.onnx`](https://huggingface.co/pollen-robotics/microduck-policies/blob/main/velstand.onnx) retained SHA256 `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`: it was neither replaced nor re-exported. Its static `obs[1,61] → actions[1,14]` interface ran separately for every world through ONNX Runtime's CPU provider.

Two fresh 32-world episodes used seed 2001. Their initial actor, joint state and root state matched exactly. Episode A commanded zero for 2 seconds, then +0.1 for 10 seconds. Episode B commanded zero for 2 seconds, +0.3 for 5, +0.1 for 10, then zero for 2. These total 1,550 controls and 6,200 physics solves per world, with 20 ms control and 5 ms physics steps.

| Phase | Mean measured body-frame vx | Worlds above 0.05 m/s |
|---|---:|---:|
| Cold +0.1 | 0.000077 m/s | 0/32 |
| Cold +0.3 | 0.143978 m/s | 32/32 |
| +0.1 following +0.3 | 0.001737 m/s | 0/32 |
| Final zero | −0.000265 m/s | 0/32 |

Means exclude each phase's first second. The 0.05 m/s threshold only describes motion; it is **not tracking acceptance**. The +0.3 phase substantially undertracks. Following the reduction to +0.1, mean speed fell from 0.0442 in the first second to 0.00328 in the next. Over the scored phase, one world still averaged 0.048689 m/s, while the median across worlds was 0.000028 m/s.

The `play=False` recipe retained BAM, model, solver, domain randomization, noise, encoder bias and observation delays. Evaluation explicitly disabled pushes, topples, prone initialization and their curricula; fixed pose commands; scheduled twist through the real command manager; selected final remaining curricula; and disabled automatic resets. No post-reset root-state writes or external kicks were introduced. Every tick checked effective command/actor binding and finite 61/14 arrays; metadata checked observation/joint order and action offsets. No termination or nonfinite sample occurred.

The retained constructor failure preceded reset/inference: disabling heading mode also required clearing its inherited heading range. Both attempts, effective recipes, traces, harness hashes and import origins remain in the evidence bundle. Source and policy hashes were unchanged. The owned PID exited successfully and its resource scope became inactive. This source recipe is not cryptographic proof of the checkpoint's historical training configuration; production limits and acceptance were unchanged.

The recorded execution command was `bash /home/johnny/Projects/demo/cascade-lab/MICRODUCK_CODEX/full-env-startup-20261002/run.sh`; that retained launcher fixes the private interpreter, GPU UUID, caches, 24 GiB memory limit, 400% CPU quota and deadline. `full_env_startup.py` contains the explicit seed/phase parameters. `installation-record.md` records reconstruction from the source pin and frozen lock. `fullenv-diagnostic-summary.json` provides artifact paths and SHA256 hashes. Preserve this evidence directory before any reproduction because filenames are fixed.

The [checked-in evidence summary](../../benchmark/results/microduck_fullenv_startup_20261002.json) also binds the independent completed-run audit: no inconsistencies, 241 source files unchanged, and metric recalculation within `5.9e-8` m/s.
