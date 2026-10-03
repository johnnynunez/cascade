# Fastening: ordinary runtime and the contact fixture

Audit date: 2026-10-03. The physical experiment and initial runtime audit use
`5f50b8b67600123f02488b6c721e685ac051b10e`. This correction is based on coordinator
`2067019b8d7397110d0f139f11e116ff5f1bd9b6` in an isolated clone. It neither launches
the fixture nor adds physical fastening admission.

## Ordinary command result

[`skill_turn_screw`](../src/cascade/skills/runtime.py) localizes a label, moves to
an engagement pose and ratchets the wrist with gripper opening/closing. It
counts commanded wrist travel, not measured fastener turns. IK reaching the
engagement pose does not establish tool contact. The routine already reported
nested `physical_verification.status=unverified` and `seating_verified=false`.

The missing connection was [`POSTCONDITIONS`](../src/cascade/agent/effects.py):
without a `turn_screw` entry, `verify()` returned `None` and `annotate_result()`
left `ok=true` without a standard verdict. The dispatcher recorded `-> ok` in
memory and credited the operating envelope; the reflex orchestrator could
record successful completion and train experience memory using that flag.

The correction registers the `threaded` postcondition. No admitted independent
fastener observer exists in this runtime, so the result remains `unverified`.
Successful wrist execution is retained as `execution_ok=true`; `ok=false` and
`verified=false` describe the unconfirmed requested fastening effect. Failed
execution retains its original error and `execution_ok=false`. Bare runtimes,
missing checker verdicts and verifier exceptions cannot restore success credit.
XYZ displacement, jaw closure and the actor's own claimed evidence do not prove
threading or seating. This special handling is limited to `turn_screw`; other
skills retain their existing result convention. It does not add a general
legacy-LLM `task_done` adjudicator.

Validation: 154 CPU tests passed in 20.88 s across the new regression file,
ordinary wrist/error tests and existing verifier/learning/threading/seating
tests. All 15 new cases pass; on the unchanged coordinator base, 14 fail and
the existing-convention control passes. The first extended run retained one
static guard failure (153 passed): the added fallback moved annotation beyond
its 900-character inspection window. Shortening a comment fixed that guard
without changing its assertion or execution logic. The final run retained
identical hashes for 649 source/config/test files and protected memory. Ruff
F/E9 and diff checks pass. See the
[receipt](../benchmark/results/fastening_postcondition_20261003.json).

## What the native fixture actually establishes

The [Factory contact report](FACTORY_THREAD_CONTACT.md) and
[final receipt](evidence/factory-thread-contact/final-seating.json) describe a
SO-101 with mounted powered socket, six passive spring inserts, a dynamic M20
nut, fixed bolt and fixed 3 mm annular spacer. The nut begins threaded and the
tool begins engaged. Contact drives the nut; there is no ideal helix or nut pose
write during motion. This establishes neither tool pickup nor autonomous
engagement, preload calibration, hardware fastening, nor a general thread family.

The final run measured 15.3653213767 nut turns, 38.1547678262 mm axial advancement,
0.2585356155 mm maximum pitch residual and 99.4097212681% angular contact coverage.
Actual spacer-face contact plus a 0.05 Nm spindle limit preceded motor-off
retention. The run lasted 39 simulated seconds and 322.091112238 wall seconds.
The no-drive control had only 0.0003952571 turns and refuted threading; its
top-level `confirmed` denotes successful detection of that negative control.
The bare-head torque stall without shoulder contact remains a seating failure.
The 1200 Hz control confirms a shorter threading interval, not full seating.

These source files at the audited runtime still exactly match the positive
native receipt. SHA-256 bindings:

| Path | SHA-256 |
| --- | --- |
| `src/cascade/sim/newton_screw_contact.py` | `d0ad84027d28ff8dcfb9c7036c55fa5d2888adf2a2bb53d9b4a3a705e46ccec3` |
| `src/cascade/sim/newton_screw_seating.py` | `3f3465319cada30a37612abffe38a5fa5abdca89ff08e13fa32475e9e9320883` |
| `src/cascade/sim/threading_verification.py` | `23a0fe46271251f087962bedbe055157f52463ed9fb69415597a24cd16979c22` |
| `src/cascade/sim/seating_verification.py` | `01845faf010d48a818de320297fbba33548998c4a2ecca7bd43523faa8f0a565` |
| `scripts/validate_screw_seating.py` | `595068912afd8f1e14d01d3b926b4fcaf2a5f312a3040d5b5def0649b5286041` |
| `docs/evidence/factory-thread-contact/final-seating.json` | `f40a6f30396b1048b476beaf65447bfde1fcdc18b23fc8949fbf6da10c056d9f` |

The [artifact manifest](evidence/factory-thread-contact/artifacts.json) binds the
full per-substep samples, body-state replay and native video. This audit compared
the above source/receipt hashes; it did not rerun physics or rehash every video.

## Next implementation: one observed turn through the ordinary runtime

An explicit optional Factory fastening profile is the smallest physical
integration. Its initial condition must declare mounted socket and pre-engaged
nut. It must not silently send a reBot command to a different SO-101 simulator.

1. Wrap `ThreadingScene.command/step/observe` in a dedicated single-writer
   controller and passive observation transport. Keep the existing measured
   nut-height follower and torque-limited spindle. Add bounded command lifetime,
   generation/admission ACK, priority stop and owned-process closure. Stop ACK
   is not rest proof: observe the nut/tool afterward. No reset, pose correction
   or rendering call may advance the passive read channel.
   This also requires explicit model-specific command admission for joint and
   speed/workspace/table/self-contact limits, spindle effort, target travel and
   command lifetime. Check the lease and generation before each substep/control
   write, not just around the scene's ten-substep outer call. A zero spindle
   command alone cannot certify rest.
   `RobotRuntime` does not supply those physical checks, and the existing scene
   follower is not already a `SafeArm` backend.
2. Add a fastening domain/profile through
   [`apps/robot_runtime.py`](../src/cascade/apps/robot_runtime.py), with explicit
   command endpoint ownership for the complete arm and spindle. Existing
   [`RobotRuntime`](../src/cascade/robotics/runtime.py) supplies namespaced tool
   dispatch, cancellation and task-result gating. Declare only implemented
   tools; do not fake an ArmBase backend with pickup/grasp capabilities. Reject
   a second legacy arm writer sharing that endpoint.
3. Bind the controller and separate reader to source, model/config and asset
   digests, exact robot/fixture/nut/tool IDs, epoch, generation and admission
   solve/time. Publish immutable per-solve pose/quaternion/contact records with
   original capture time. Reject resets, clock gaps, stale/missing data,
   overflow and identity changes. The current `ThreadSample` contains body
   IDs, epoch and physics clock but does not itself provide all these transport
   and admission guarantees; add the envelope rather than implying it does.
   The current private contact buffer also needs an explicit capacity/overflow
   contract for lossless substep transport.
4. Admit the pre-engaged condition using measured tool/thread contact and axis
   alignment, then take the independent post-admission baseline. Feed the full
   substep interval to `verify_threading`, never actor turn counters, archived
   samples or a coarse OR of contacts. Preserve pitch/radial/tilt/fixture,
   sampling and 80% contact-rotation gates. First request: one actual tightening
   turn, followed by independently verified stop, with video from that episode.
5. Return the verifier's standard postcondition to the domain. Static nut with
   spinning tool, no-drive, lost engagement, wrong pitch, epoch changes and
   interrupted motion must not confirm. Keep seating false unless separately
   verified through real spacer contact, torque dwell and motor-off retention.

Full seating requires a separate explicit goal: the existing final fixture
travels about 15 turns, while ordinary wrist `turn_screw` allows at most 6.
Do not reinterpret one turn as seating or silently extend that cap. The measured
322 s wall cost also exceeds the ordinary 300 s MCP host limit; any full-seat
profile needs a declared computational budget and bounded physical lifetime
before a new native episode. All of this integration remains pending; the
postcondition correction alone establishes no new physical capability.
