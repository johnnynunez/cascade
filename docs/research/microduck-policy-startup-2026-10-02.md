# MicroDuck policy startup evidence — 2 October 2026

Public evidence supports a distinction between starting a slow walk from rest
and sustaining an established gait. It does not establish a remedy for the
current candidate or explain its negative-command failures. This note records
source research, not new physical experiments or locomotion admission.

A subsequent [local full-environment diagnostic](microduck-fullenv-startup-2026-10-02.md)
tests cold +0.1 and the sequence +0.3 → +0.1 with the official policy. It
retains negligible low-speed motion in both cases; the report separates those
measurements from the upstream author's different +0.15 sustained-motion claim.

The deployed reference remains Pollen's `velstand.onnx`, SHA-256
`1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`.
The [published file](https://huggingface.co/pollen-robotics/microduck-policies/blob/d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0/velstand.onnx)
retains that digest. The [v6 update](https://huggingface.co/pollen-robotics/microduck-policies/commit/d30a21695085d6685d8c19ce63521a7b8e069b2f)
changes SitStand's seated-head stability; it does not replace VelStand.

The open [MicroDuckRL issue #59](https://github.com/pollen-robotics/microduck_rl/issues/59),
filed on 26 September, reports shipped `seed-v5` policies at RL revision
`cb70b79`. Its push-free VelStand results in 32 environments are:

| Requested vx | Mean forward speed | Environments walking |
|---|---:|---:|
| +0.15 m/s | 0.002 m/s | 0/32 |
| +0.20 m/s | 0.021 m/s | 1/32 |
| +0.30 m/s | 0.141 m/s | Not reported |

The author reports the symptom in `Mjlab-Velocity-Flat-MicroDuck`,
`infer_policy.py` and `robotd --sim`, and reports that an already moving robot
can continue more slowly. These are the author's measurements, not our
reproduction. The issue does not provide a signed-negative-command battery,
reconstruct the historical VelStand training run, or demonstrate a fix.

There is a concrete evaluation confound in the
[MicroDuck factory](https://github.com/pollen-robotics/microduck_rl/blob/cb70b792312d559a4da09064d92009079671815f/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py#L355):
play mode applies velocity pushes every 0.5–1 s. Also, mjlab 1.3.0's
[base command configuration](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/tasks/velocity/velocity_env_cfg.py#L168)
assigns a forward-only fraction of 0.2; the
[sampler](https://github.com/mujocolab/mjlab/blob/v1.3.0/src/mjlab/tasks/velocity/mdp/velocity_command.py#L82)
maps that group's vx through `abs().clamp(min=0.3)`. Pinning a range alone
therefore does not guarantee the intended signed command. These mechanisms
must be distinguished from the effective command recorded in each episode.
Their causal contribution to learned startup behavior remains unproven.

The independent Duckbatch evaluation
[pins the same teacher digest](https://github.com/craigm26/duckbatch/blob/main/scripts/fetch_teachers.sh#L4).
Its [walking evaluator](https://github.com/craigm26/duckbatch/blob/main/src/duckbatch/sim.py#L253)
uses the full mjlab actor observations but retains ordinary pushes and sampled
command histories, then averages upright tracking error. Those aggregate
results do not establish slow or negative startup from rest.

Acceptance must measure startup and sustained motion separately, preserve
signed per-command results, and bind the effective actor inputs, model and
perturbation settings. A successful sustained-gait example cannot repair a
failed startup result. The runtime must keep its requested velocity and safety
caps: this finding does not authorize an initial kick, an unreported faster
command, physical-state writes, changed gains or replacement weights.
