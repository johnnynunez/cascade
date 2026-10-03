# MicroDuck catalogue review and concrete CASCADE transfers

Reviewed 2026-10-02. This is a selective primary-source audit, not a claim to have
reviewed every project in the catalogue. The pinned discovery index is
[awesome-microduck@3ddaa0e](https://github.com/joeynyc/awesome-microduck/tree/3ddaa0eab900f9d53041de3d7de21e4a7fb31167)
(CC0). It separates source projects and weight artifacts and publishes a policy
registry plus a separate revocation registry. Both registry files were read at
that same commit; the revocation list was empty. Catalogue inclusion establishes
neither compatibility nor physical acceptance.

| Primary material actually inspected | Pin and transfer to CASCADE |
|---|---|
| [Official policy manifest](https://github.com/pollen-robotics/microduck/blob/1fa84386f07884e27866411bc1ba166977bced95/docs/policy-manifest.md), `duck-control/src/{policy,obs,sim}.rs`, `robotd/src/control.rs` | `1fa84386f07884e27866411bc1ba166977bced95`. Keep explicit observation/action/command contracts, policy kinds and entry poses. Our selected feed-forward policies use 61 observations and 14 actions at 50 Hz; recurrent variants are not interchangeable. Head joints already belong to the policy; mouth is separate. |
| [Official RL inference](https://github.com/pollen-robotics/microduck_rl/blob/8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec/scripts/infer_policy.py) and VelStand training configuration | `8d0db74916a4f833d1d9b95d6a1d7f4d13b9d5ec`. Separate the exact inference recipe from randomized training and hardware-daemon settings. The audited default inference uses BAM kp 200, 7.4 V, sag 0.1, minimum 6 V, no current cap/delay, stiff friction and voltage-derived torque limit. This motivates an explicitly named recipe, not arbitrary gain tuning. |
| [Official weights](https://huggingface.co/pollen-robotics/microduck-policies/tree/d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0) | `d5a8b55033e157f1af2ed6bd5c1e435b770a8ee0`; VelStand SHA `1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45`. Preserve this default and all failed low-speed receipts. |
| [Policy golden vectors](https://huggingface.co/datasets/craigm26/microduck-policy-golden-vectors/tree/69baf2d4fe73c19d8e91da1f36fac55da10c289e) | `69baf2d4fe73c19d8e91da1f36fac55da10c289e`, Apache-2.0. Four actual alpha-walking vectors passed CASCADE observation/inference comparison: maximum observation error 1.49e-8, action error 1.192e-7. This validates the numerical interface, not locomotion. |
| [rough-walk-e model card and manifest](https://huggingface.co/RemiFabre/microduck-rough-walk-e/tree/fa7b27eeb5610d3b351362f4bd71691ee8be3d7d) | `fa7b27eeb5610d3b351362f4bd71691ee8be3d7d`, policy SHA `5aa423bd693e431b19e2ead77f99cbae6184e40a529eb2f7c1b4f85bb7f57040`, 793772 bytes. Apache-2.0 weights, explicitly simulation-only upstream. Same selected feed-forward layout. Added named opt-in admission with exact bytes and separate native evidence; no default replacement. |
| [Newton IsaacLab port runner/BAM notes](https://github.com/kabilankb/isaaclab-microduck/tree/4310fe050b7a1b01e7b2f4bada103dea81d71fb2) and [Duckbench README](https://github.com/craigm26/duckbench/tree/3a0f2f6e9d5670a446678f99d665b778b236c2cb) | Different actuator/current/solver recipes cannot be treated as identical plants. CASCADE retains its separately pinned native IsaacLab BAM implementation and force-channel validation. External benchmark success is not imported as CASCADE acceptance. |
| [Locomotion diagnosis README/reward patch](https://github.com/Qi-hub-dot/microduck-locomotion-diagnosis/tree/2a6366211e6752b12eeb771202c698cb30001575) | Useful hypotheses about rewards for standing rather than motion. No license was found in the inspected snapshot, so code was not copied. Hypotheses do not prove the cause of our runtime failures. |

The upstream [VelStand dead-zone discussion](https://github.com/pollen-robotics/microduck_rl/issues/59)
was treated as a diagnostic lead. Actual local comparisons used the reviewed
CPU inference implementation and saved solved body trajectories. VelStand
still advanced only about 12.6 mm during a three-second +0.3 command; an eight-second
reference began progressing after roughly three seconds but exceeded the
existing heading bound by 45 mm. Extending the CASCADE action deadline would not
by itself admit that episode.

Rough-e generated prompt forward/reverse motion in the same CPU reference.
The explicit geometric skill `walk_distance` closes the loop on measured signed
displacement; it does not claim that an internal 0.3 policy command achieves
0.3 m/s. Fifty-millimetre CPU trials violated the original heading gate. Thirty-
millimetre native trials reached their targets, while their original wall rest
budget exposed residual motion. Solver-only graph capture subsequently allowed
one reverse trial to pass all unchanged independent gates; forward rest and
turn translation still failed. This remains a candidate, not general gait
admission. Separate final stop success does not revise an earlier failed task.

The subsequent [native distance report](MICRODUCK_DISTANCE_CANDIDATE.md) adds
same-solve read reuse, four confirmed fresh-start ±30 mm repetitions and a
priority stop during measured motion. It also retains the composed reverse
contact veto and all failed turns; this does not establish general gait admission.

Code transfers are in `control/microduck_policy.py`,
`sim/microduck_policy_admission.py`, `sim/microduck_solver_graph.py`,
`safety/base_harness.py` and `agent/base_effects.py`. The exact native BAM source
remains [IsaacLab PR8161@28aa1fca](https://github.com/isaac-sim/IsaacLab/pull/8161/commits/28aa1fca5843208ff9a67935695a4d5376e44d50).
The [candidate policy manifest](../assets/microduck/policy-candidates.json) records
the alternative byte/contract admission. The graph option captures only the reviewed SDK solver; host BAM, ownership,
signal barriers, contact verification and actual-frame capture remain live.

Voice leads discovered in the index include `digows/microduck-twin`,
`andreagenovese/quacksat` and `osolmaz/microquack`. Their implementations were not
reviewed in this locomotion audit. They were handed to the conversation owner
as leads, not adopted dependencies. A voice integration must keep media separate
from the policy-owned head joints and route movement through the same runtime
cancellation/admission boundary. No direct servo-writing voice loop was added.

[Research and trial artifact hashes](evidence/robot-modularity/microduck-ecosystem-research.json)
bind the inspected material and retained experiments. Local source downloads and hashes are retained in
`LOCOMOTION_NEXT/research/download-manifest.json` and
`LOCOMOTION_NEXT/research/awesome-microduck/download.json`; alternative-policy
bytes have a separate LFS/hash receipt. CPU references, native failed and positive
cases, independent raw observations and videos remain in their original
`LOCOMOTION_NEXT/reference-*` and `native-*` directories. See the [native signal
closure report](MICRODUCK_NATIVE_SIGNAL_CLOSURE.md) for lifecycle scope. Code, model geometry and weights have
separate licenses; the official model README's BY-SA-NC wording is not replaced
by the Apache-2.0 code or weight licenses.
