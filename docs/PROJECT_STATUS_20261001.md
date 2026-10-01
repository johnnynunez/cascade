# Project status — 1 October 2026

This is the current source and acceptance index. Dated experiment reports remain
historical evidence; their successful cases and failures are not rewritten when
a later source is merged. See [Spark setup](DGX_SPARK_SETUP.md) for installation,
[Spark delivery](SPARK_DELIVERY.md) for native proof and campaign contracts, and
[nvblox](NVBLOX.md) for the separate mapping experiments.

## Source and software validation

The validated runtime baseline `477c88fe40092fdaad99c0f777a6f1c3b9a41224`
merges the reviewed integration
`391436063c5dd98b0686202db02cefc9bdf294d1` with the same product tree
`ff43ff758ddf39dac8e9661fe083fb5ac7943bcd`. The final suite passed **3,205 tests**,
with 43 skipped and four deselected, in 331.79 seconds. The 1,056 tracked files
were compared with the immutable test-source manifest. These are software
results, not a new physical acceptance or a fresh-install measurement.

The previous integration run retained one failing test: it expected the learned
candidate to remain first in an optional learned/analytic list, although the
legacy selector already sorted that list by quality. The test now checks model
recovery by candidate identity, required-mode isolation and optional-mode quality
order. Production was unchanged by that follow-up. The earlier failure remains
in the validation history.

| Merged change | Current contract and documentation |
| --- | --- |
| [PR #30](https://github.com/johnnynunez/cascade/pull/30) | Read-only [Jev/Kev evaluation](JEV_DECISIONS.md); no active presenter backend or demonstrated advantage over recorded-status rules. |
| [PR #32](https://github.com/johnnynunez/cascade/pull/32) | Optional grasp-attempt evidence retains inputs, model responses and motion phases without changing the normal control path. [Forensic example](SPARK_LAUNCH02_FORENSICS_20261001.md). |
| [PR #33](https://github.com/johnnynunez/cascade/pull/33) | [Strict native confirmation](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026) binds saved native cases to placement, home, camera and reset evidence. |
| [PR #34](https://github.com/johnnynunez/cascade/pull/34) | [Isaac physical-time motion](ISAAC_MOTION_CLOCK.md), bounded RPCs and settling; 30 Hz nominal Isaac targets retain the full legacy 50 Hz safety edges. |
| [PR #35](https://github.com/johnnynunez/cascade/pull/35) | Headless viewport updates are disabled. [Render comparison](SPARK_RENDER_COMPARISON_20261001.md) is an idle benchmark, not loaded-motion throughput. |
| [PRs #36–37](https://github.com/johnnynunez/cascade/pull/37) | [Observed finger approach](observed-finger-gate.md), [mask alignment](detector-mask-letterbox.md), full-stroke [closing preflight](observed-finger-closing.md), and Spark defaults. |
| [PR #38](https://github.com/johnnynunez/cascade/pull/38) | [Bounded planning search](grasp-planning-search.md): at most three batches/eight seconds, one saved scene and prior, before motion. |
| [PR #39](https://github.com/johnnynunez/cascade/pull/39) | [Native visitor turn budget](native-visitor-turn-budget.md): 300 seconds plus CLI return margin; cancellation still stops motion. |
| [PR #40](https://github.com/johnnynunez/cascade/pull/40) | [Render-bound frame history](isaac-render-frame-history.md), including exact SDK time encodings observed on ARM. |
| [PR #27](https://github.com/johnnynunez/cascade/pull/27) | Multicamera payload mapping, retained contact/release episodes, scoped recovery, [anchor refresh](NVBLOX_RETAINED_ANCHOR_REFRESH.md), [mapper reset](NVBLOX_MAPPER_RESET.md) and [read-only truth construction](NVBLOX_TRUTH_READONLY.md). |
| [PR #41](https://github.com/johnnynunez/cascade/pull/41) | A deterministic recorder-clock test replaces an assumption about host scheduling overhead; recorder production is unchanged. |
| [PR #42](https://github.com/johnnynunez/cascade/pull/42) | [Preserved grasp-memory ranking](grasp-memory-ranking.md) and [coherent held-object observations](HELD_OBJECT_OBSERVATION.md); a cached aiming estimate cannot authorize slip release. |

## Current physical acceptance

**The new MAIN acceptance is incomplete.** Spark final-MAIN-02 and the isolated
RTX nvblox environment 09 are separate runs. Neither a completed five-object
campaign nor same-version restart acceptance is claimed here until its own
bound receipts pass. Preparation, software tests and passive observations do
not replace those stages.

| Source/run | Measured result and boundary |
| --- | --- |
| September `9cf5402` delivery | A fresh-destination GB10 installation, two desktop proofs, a five-object visitor round and restart passed on that source. Download caches were reused. [Historical receipt](../benchmark/results/spark_clean_delivery_20260930.json). Later orange failures remain recorded; this is not the acceptance result for current MAIN. |
| MAIN `6b5dad6`, Spark final-MAIN-01 | Green completed placement, return home and reset under the existing server-camera-age audit. Orange failed with an air grasp and no observed lift. The overall native proof failed; no five-object campaign or restart followed. [Ranking diagnosis](grasp-memory-ranking.md). |
| MAIN `6b5dad6`, nvblox environment 08 | A slip verdict opened the gripper although the last measured sample still showed bilateral contact and a lifted orange. The intended post-release fault was never reached. The exact legacy offset branch was not logged. [Retained failure](HELD_OBJECT_OBSERVATION.md). |
| MAIN `477c88f`, Spark final-MAIN-02 | Green completed placement, return home and reset. Orange stopped during descent when finger feedback left its checked opening interval, before any close command. The overall proof failed; no five-object campaign or restart followed. Offline diagnosis is in progress. Outcome memory is preserved. |
| MAIN `477c88f`, nvblox environment 09 | Passive admission passed: three cameras, no actuator writes; mapping warmup is reported separately. The subsequent release diagnostic stopped during descent, retreated and returned home before any close, fault injection or release episode. Aggregate failure is retained; no five-object campaign followed. Diagnosis is in progress. |

The camera auditor requires each sampled camera's **server snapshot age ≤2 s**
and advancing, correctly bound captures. Client delivery age is also recorded
where available but is not that gate. For example, the successful green case
on `6b5dad6` had one reset delivery at 2.122 s while server age was 1.991 s.
An observed gap between two captures does not prove that no intermediate frame
was produced while a request was blocked. No report here claims continuous
camera availability from sparse samples.

A two-object READY proof, strict native confirmation, five-object campaign,
installation check and same-version restart are distinct results. An overall
failure stays a failure even when its grasp, retreat or placement subchecks pass.

## Profiles, clocks and architectures

Defaults belong to entry points and profiles; do not apply a setting from one
row universally.

| Entry point/profile | Relevant defaults and scope |
| --- | --- |
| Installed Spark desktop/profile | PhysX CUDA; `isaac_kitchen_gpu`; required real GraspGen-X; Qwen Q4 with vision; three cameras; physics `dt=1/120`; camera cadence six bridge iterations; observed-finger gate enabled. Explicit supported environment overrides take precedence. Occupancy is disabled. |
| Isaac arm driver/profile | `motion_rate_hz=30` nominal targets; an explicit call rate overrides the profile. The safety profile retains complete legacy 50 Hz edges and checks actual command edges. Physical pacing and acknowledgements can lower observed target throughput. |
| Base hardware arm API | Existing 50 Hz default. Isaac timing is not imposed on serial, CAN or ROS2 drivers. |
| Raw `scripts/isaac_bridge.py` | Engine default Newton, timestep `1/60`, camera cadence two bridge iterations unless overridden. The installed Spark launcher deliberately selects different defaults. |
| Isolated RTX nvblox diagnostics | Explicit PhysX/CUDA and mapper configuration, independent ports/ownership, three real camera sources and unchanged clearance/unknown-space rules. These runs do not enable nvblox in the presenter profile. |

Camera cadence counts **bridge iterations**, not physics steps or measured image
Hz. The producer polls at the configured cadence or after 0.5 wall seconds, then
publishes only a render token that resolves to matching recorded physics state.
Repeated renderer output retains the entire previous packet and timestamp.
`rpFabricTime`, captured q/jaws/transforms and producer epoch travel together.
An unmatched or ambiguous history does not get relabeled as fresh.

The RTX workstation is Linux x86_64; DGX Spark GB10 is Linux aarch64. Isaac,
Torch/CUDA and learned-model environments remain separate. x86 simulation or
mapper evidence does not certify ARM deployment, and neither certifies a real
robot. [Newton results](NEWTON_ENGINE.md) are a separate engine/asset matrix.

## Research and separate implementation work

[Jev/Kev decision experiments](JEV_DECISIONS.md) remain offline research. The
captured ten-outcome replay found 10/10 for Kev 4B, Qwen and recorded-status
rules, with no demonstrated decision-quality benefit over the rule baseline.
That comparison has no motion authority and is not integrated into presenter
startup. Official TypeSafe Jev was not evaluated without its API credential.

[NVIDIA ovrtx](ROADMAP.md#optional-sensor-backend-nvidia-ovrtx-2026-10-01) is being
implemented in a separate branch as an optional camera and rendering backend.
No ovrtx dependency or acceptance is included in runtime baseline `477c88f`.
Available x86_64 and aarch64 SDK packages do not by themselves establish Cascade
execution on either host. Cosmos, mobile-base navigation and new hardware validation retain their
separate roadmap scopes. Locally installed development skills are authoring
tools, not runtime dependencies or substitutes for source-bound physical tests.

## Operator evidence sequence

Use the normal [desktop launch](DGX_SPARK_SETUP.md#4-start-and-prove-the-demo)
from the intended installed checkout. A successful installation check means
PREPARED; a skip-proof launch means STARTED / UNVERIFIED. After a complete native
proof, run the saved-evidence [strict checker](SPARK_DELIVERY.md#native-confirmation-gate-and-failed-follow-up-1-october-2026).
Run a five-object native campaign in that same verified session, fail fast on a
failed case, and preserve its receipts before considering restart acceptance.
Never delete outcome memory to make a new test look like a clean success.
