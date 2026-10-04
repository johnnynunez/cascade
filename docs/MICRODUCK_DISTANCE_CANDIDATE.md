# MicroDuck: bounded native distance candidate

Four fresh-start native episodes originally recorded confirmed signed 30 mm
objectives: two forward and two reverse. A later
[retained-observation audit](evidence/voice-reply-recovery-20261004/guarded-continuation.json)
finds post-completion heading drift of −0.18149 and −0.18552 rad in the two
forward cases, beyond their unchanged 0.08 rad bound. Their original receipts
predate the post-completion veto described below and remain unmodified;
they do not admit the current contract. The two reverse traces do not violate
that heading bound, which alone does not re-admit them. A separate priority
stop interrupted an independently observed moving robot and subsequently
confirmed physical rest. These are local Newton
results on flat ground, not general locomotion admission. A subsequent single
fresh −0.1-rad turn also confirmed; the larger measured ±0.2-rad turns still fail
the translation-path criterion, and reverse motion after a forward episode produced
a forbidden ankle-shell contact. The original velocity-tracking failures remain.

The [compact evidence record](evidence/robot-modularity/microduck-distance-candidate.json)
contains exact metrics, epochs, source/asset hashes and raw artifact references.
The [ecosystem review](MICRODUCK_ECOSYSTEM_REVIEW_20261002.md) explains the primary
sources and policy selection. All measurements below were retained on
`d9f47660ad8069820e617cb02fdeff7d5b706e74` or the explicitly identified earlier
sources; publication does not relabel them as a new native run.

The later [aggregate composition](MICRODUCK_LOCOMOTION_COMPOSITION.md) restores
the complete implementation on the modular/RGB-D runtime. It has CPU integration
evidence and requires a new native identity; these earlier results are not
reassigned to that source.

## Implemented contract

The 2026-10-03 [distance-outcome regression receipt](../benchmark/results/mobile_distance_outcome_20261003.json)
records a software defect reproduced on `cbe9873`: 30 mm of admitted motion,
followed by 50 mm of coasting and a quiet suffix, could confirm an 80 mm final
displacement for a 30 mm request. Positive credit was correctly clipped at
completion, but that also hid the intervening excessive motion from the goal
check. The corrected `walk_distance` verifier keeps that positive interval and
separately checks every valid observed prefix from completion through the last
observation, using the same baseline, heading integration and existing limits.
Overshoot, retreat and lateral/heading drift retain a veto even after returning
to the goal. Valid late or cancelled in-flight observations can veto; they cannot
complete a missing objective or establish rest. Existing channel, support,
execution and cancellation failures remain failures.

The exact review probe changes only the excessive-coast case from confirmed to
refuted; its healthy, inert, late-only and forbidden-contact controls retain
their verdicts. The seven-file CPU selection passed 566 tests in 48.88 seconds,
including 27 new synthetic regressions. No policy, profile limit, actuator or
native recipe changed. No new physical episode was run and the historical
native results below were not re-admitted by this software check.

`walk_distance(distance_m=...)` is a geometric objective exposed only by a
configured mobile profile. SafeBase uses a first completed post-ACK observation
as its baseline and integrates measured displacement in the body heading frame.
The independent checker separately compares that signed displacement with the
caller’s distance. Neither layer derives success from policy command × time.
Progress after the physical deadline cannot complete the controller objective.

The optional [distance profile](../configs/bases/microduck_distance_candidate.yaml)
uses internal policy commands ±0.3 with a 5 mm controller stopping tolerance.
That input is not a claim of ±0.3 m/s measured velocity. It bounds distance,
lateral/heading drift, posture, support, freshness, cancellation and duration.
Its 100 mm configuration ceiling is a software bound; the repeated 30 mm
cases below retain their original source and verifier scope. Ordinary
`walk_velocity`, the official VelStand default and the original diagnostic
profile remain unchanged.

The separate [slow native profile](../configs/bases/microduck_distance_native_slow.yaml)
sets a 25-second host action budget from measured compute cost: approximately
7.512 wall seconds per physical second, a three-second physical command cap,
two 0.5-second RPC budgets and a one-second scheduling allowance, rounded up.
The original eight-second profile and its failures remain. The verifier’s
29-second total budget and 1,451-sample capacity cover that host duration.
Physical action duration stays at three seconds; rest still requires a
0.2-second physical window within three wall seconds. No freshness, progress,
support, drift, rest-speed or posture threshold was relaxed.

The existing independent progress contract combines its 0.8–1.25 ratios with
a 15 mm absolute translation tolerance. Turns instead use
`max_lateral_drift_m=0.02` as their 20 mm translation-path bound; all archived
native MCP profiles carry that exact value, without a turn-specific override.
The controller’s tighter 5 mm target gate remains separate. Rest limits remain 0.02 m/s, 0.1 rad/s, 5 mm translation
drift and 0.03 rad yaw drift. Only exact registered soles may support rest;
walking permits a known flight phase, never an unavailable force channel.
Any solved external non-sole contact retains its veto, including a zero-force row.

## Explicit policy and compute recipe

The named `rough_walk_e` admission binds repository pin
`fa7b27eeb5610d3b351362f4bd71691ee8be3d7d` and the exact 793,772-byte ONNX SHA
`5aa423bd693e431b19e2ead77f99cbae6184e40a529eb2f7c1b4f85bb7f57040`.
It uses the reviewed feed-forward 61-observation/14-action, 50 Hz contract;
mouth control remains outside it and policy-owned head joints have no second
writer. Caller-provided hashes cannot authorize an arbitrary checkpoint. Weight,
code and robot-geometry licenses remain separate.

`official_infer_nominal_no_delay` explicitly selects the audited inference BAM
parameters: kp 200, 7.4 V, sag 0.1, minimum 6 V, no current cap or delay, stiff
friction and the voltage-derived 0.9634015946604082 Nm limit. The native BAM
implementation remains IsaacLab PR8161 at `28aa1fca5843208ff9a67935695a4d5376e44d50`.
This establishes parameter fidelity, not equivalence of the CPU and Newton plants.

`--solver-cuda-graph` is optional and pins the inspected SDK NewtonStage source.
It enables capture after model preparation, warms once and performs one physical
solve per call. BAM, command/stop fencing, signal checkpoints, force extraction
and rendering remain on the host. Buffer/model/timestep/mode changes fail closed.
In a separate 240-solve profile, total tick time fell from 6.621 to 3.133 seconds.
Nested phase timings overlap, and asynchronous solver cost may appear in reads;
the 0.102-second solver-call sum is not total GPU execution time.

`--reuse-solved-read` additionally requires that bound graph mode. It reuses only
a detached observation of the same completed solve, rechecking the model, clock
and buffer references. Every consumer receives a copy; no timestamp is refreshed
and no extra physical step is published. Cache invalidation precedes each solve
and teardown. This backend requires its owned single main-thread state writer:
reference guards do not detect arbitrary foreign in-place writes into unchanged
GPU buffers. Reset or another state writer cannot be attached to this recipe.
Standing traces across 812 matched solves differed by at most 7.08e-8 m in
position; this is observed standing parity, not general trajectory equivalence.

## Native results and retained failures

Latest model identity:
`0227a46cfa2882922cbcba70b52b056b9daf0c0bb07a122e28d5adea0080e0da`.
Physics used actual dt `0.004999999888241291` seconds, with one policy evaluation
per four completed solves. Each repetition used a fresh process/epoch and a
1.5-second physical standing preparation. The actual mobile MCP runtime called
the skill; a separate passive truth channel supplied the independent verdict.

| Retained episode | Requested / measured forward mm | Lateral mm | Heading rad | Rest max m/s / rad/s | Verdict |
|---|---:|---:|---:|---:|---|
| `native-rough-reuse-forward01` | +30 / +26.576 | −1.679 | −0.04524 | 0.00716 / 0.08283 | confirmed |
| `native-rough-reuse-reverse02` | −30 / −25.959 | +10.067 | +0.02032 | 0.00263 / 0.02064 | confirmed |
| `native-rough-reuse-forward03` | +30 / +26.208 | −2.462 | −0.04281 | 0.00728 / 0.08399 | confirmed |
| `native-rough-reuse-reverse04` | −30 / −25.358 | +10.035 | +0.01972 | 0.00040 / 0.00251 | confirmed |

All four passed the complete configured independent motion/support/rest checks,
with no fallen state. Each also confirmed a subsequent emergency-stop rest
window. Native force probes matched the separate SDK force API with zero
reported difference. Foundation transient non-sole rows before 0.09 physical
seconds are retained; none remained after the 1.5-second preparation.

The combined `native-rough-reuse-distance01` passed forward motion but halted
reverse motion at approximately −21.55 mm. At step 661 two left ankle-shell /
ground solved rows carried zero normal/vector force. They are not evidence of
load-bearing ankle support, but they violate the unchanged any-forbidden-contact
contract. Failed controller calls can omit their ACK from the final receipt;
the independent result then remains **unverified**, rather than borrowing an ACK.

Earlier rough-e trials without the compute improvements reached distance targets
but failed rest. Graph-only forward rest also failed. All measured ±0.2-rad turns
in these campaigns exceeded the 20 mm permitted translation path; later rest
does not repair that failure. Even separate ±0.1-rad CPU reference trials had
34.94/18.88 mm translation paths: the positive turn exceeds this bound, while
the negative turn does not. That diagnostic justified the single native probe
below; it did not itself establish native admission. Longer-distance and longer-duration references,
the original ±0.1 m/s failures, and the earlier forced-shutdown failure remain
separate negative evidence. No command remapping or threshold adjustment makes
them successful.

## Stop during movement

`native-rough-reuse-interrupt01` requested +30 mm. A pinned passive truth reader
observed an active generation-2 command at step 335, then triggered stop at step
353 after 5.198 mm planar movement: +2.373 mm world X and −4.625 mm world Y.
This is not 5 mm of forward travel. The client received the generation-3 stop
ACK 7.334 ms after sending it. The pending walk returned `cancelled by stop` and
remained unverified; a cancelled objective is not a successful walk.

The last nonzero policy input used observation step 350. Observation step 354
committed generation-3 zero travel intent, first solved at step 355. All later
policy commits retained zero intent. Balancing joint targets continued, as the
standing contract requires. The post-ACK trajectory still travelled 14.818 mm;
an ACK does not mean instantaneous motor-off or zero inertia. Independent rest
then confirmed a 0.205-second physical window: maximum speed 0.00153 m/s and
0.01353 rad/s, translation drift 0.155 mm and yaw drift 0.00217 rad.

## Single negative-turn follow-up

The preregistered `turn(angle_rad=-0.1)` in
`native-rough-reuse-turn-negative03` confirmed measured yaw −0.0712782 rad
within the unchanged 0.04-rad angle tolerance. Its measured path was 15.4787 mm
against the existing **20 mm** bound, with lateral displacement −13.9175 mm.
Rest confirmed a 0.205-second window with maximum speed 0.0008721 m/s and
0.0072991 rad/s. Subsequent emergency-stop rest also confirmed. This is one
small negative turn, not evidence for positive or larger turns.

Source remained `d9f4766`, model identity remained `0227a46c…0e0da`, and the
episode epoch was `f23ea0c575f74a9a8a02bdb778c14e74`. All 44 native solves in
the generation-3 action interval, steps 330–373, had known support records and
no forbidden external robot contact. The run retained 850 consecutive solves,
43 force probes with zero reported difference, and 43 real frames. Those force
probes compare two APIs of the same solver to check extraction/sign/frame
mapping; they are not independent physics engines. The separate passive
postcondition reader does not trust actor-reported progress. Process closure
retained native exit 143, launcher exit 1 and inactive scope, with unchanged source.

Attempts 01 and 02 failed during read-only preparation, before any turn command:
state RPCs reached the 0.5-second limit at physical steps 42 and 62. Both retained
zero travel intent and clean closure. A separate passive 240-solve profile then
measured one capture at 1.5886 seconds; later captures took 53–75 ms. This
supports explicit startup preparation but does not establish the internal cause
of those two failures.

The successful harness prepared a model/epoch-pinned passive reader **before**
constructing MCP, within the existing 25-second preparation budget and unchanged
0.5-second read/freshness bounds. Its 47 observations covered 1.525 physical
seconds in 4.859 wall seconds. No read failed in this attempt; the preparation
protocol records any failure and closes its socket before another bounded
read-only handshake. It never retries an action or postcondition read. All
thresholds and native recipe bytes remained unchanged. The
[follow-up receipt](evidence/robot-modularity/microduck-negative-turn-probe.json)
binds all three attempts, preregistrations, profile timings, exact values, raw
samples and the inspected video. Earlier failed turns are unchanged.

## Reproduction and validation boundary

Use the reviewed installation, bundle and controller-limit procedure in
[MicroDuck runtime](MICRODUCK.md). The measured variant adds these explicit
bridge options to that complete command:

```sh
--policy-profile rough_walk_e --policy /reviewed/path/policy.onnx \
--policy-sha256 5aa423bd693e431b19e2ead77f99cbae6184e40a529eb2f7c1b4f85bb7f57040 \
--bam-profile official_infer_nominal_no_delay \
--solver-cuda-graph --reuse-solved-read
```

Review a fresh foundation’s canonical recipe before pinning its identity and
exact support registry into the MCP profile. Use the matching slow controller
limits and `microduck_distance_native_slow` profile; a hello digest alone is not
admission. The archived `campaign.json`, copied harness and `mcp-profile.json`
retain the complete actual invocation and resolved configuration for each case.
The local artifact root is `LOCOMOTION_NEXT/` beside the publication checkout.

Saved native RGB frames and their manifests bind MP4s to the same physical
episodes, using actual simulation timestamps. Start/middle/end frames were
visually inspected; the robot is upright and the captures are valid. Images
support inspection but do not replace measured contact or motion evidence.
Successful runs ended with a pre-SDK shutdown receipt, child exit 143, launcher
exit 1 and an inactive owned scope, without SIGKILL. `SimulationApp.close` exited
the process rather than returning to Python; its return is not claimed.

The publication tree preserves all 22 native recipe source files byte-for-byte.
It also retains the newer composed-runtime admission fixes from its PR76 base.
The focused software suite passed 1,508 cases with eight explicit optional
dependency/source skips in 129.97 seconds. A separate compatible CPU converter
environment passed 77 cases with three external-asset skips in 4.78 seconds.
All selected F/E9 checks passed; successful suite snapshots and protected shared
stores remained unchanged. The initial six outdated contact-fixture failures,
74 missing-OpenUSD setup errors and private-environment USD packaging aborts are
retained in the evidence record, with their fixes and exact dependency scope.
No hardware, uneven-ground, real-time or global-navigation admission is inferred.
