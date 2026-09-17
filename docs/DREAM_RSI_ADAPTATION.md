# Scoped retry evidence admission

A later successful action is not necessarily a repair of an earlier failure.
CASCADE's ASPIRE trace-to-library path now requires matching context and a
measured confirmation before writing a new guidance note. This is an
admission safeguard, not a policy optimizer or a change to robot control.

## Admission contract

[`agent/aspire.py`](../src/cascade/agent/aspire.py) examines at most the next
six trace rows after a failed call (`REPAIR_WINDOW`). A candidate must pass
all of these checks:

| Check | Required evidence |
|---|---|
| Result | Literal `ok: true`; `verified`, if present, must also be literal `true` |
| Postcondition | `status: confirmed`, the same skill, and its registered postcondition kind |
| Measurement | A `physics`, `belief`, `gripper` or `visual_diff` channel, nonempty evidence text and nonempty measurements; command-only `arm` confirmation is rejected |
| Goal | Equality of the recorded identity-bearing arguments listed below |
| Context | Explicit matching resolved arm identity and pre-call held-object label |
| Boundary | No intervening `reset_scene`, `task_done` or result with `task_complete: true` |

Goal comparison preserves the recorded values of `object`, `label`, `query`,
`destination`, `arm`, `spatial_hint`, `camera`, `x`, `y`, `z`, `direction` and
`distance_m`. It does not infer that differently named targets are the same.
Other parameters can differ, and their recorded delta is included in the note.
For `place_at`, `place_on_object`, `throw` and `handover`, the held-object label
must be nonempty on both calls; equal missing possession is not enough.

Examples of the admission decision (illustrative, not robot measurements):

- Failed red-cube grasp, followed by a physics-confirmed red-cube grasp on the
  same arm with matching pre-call possession: eligible.
- Failed red-cube grasp, followed by a successful blue-cube grasp: rejected.
- Same request but a different arm, different held subject, absent context,
  unverified outcome or contradictory `verified: false`: rejected.
- A matching success after an explicit scene reset or task end: rejected.

## Trace context and generated notes

[`SkillRuntime.execute()`](../src/cascade/skills/runtime.py) records context
before the skill body runs, separately from its arguments. For example, a
placement can record `{"arm": "left", "held_object": "red cube"}` even when
it clears `held_object` after opening the gripper. These are recorded labels,
not independently sensed object-instance identities.

Omitted, default and primary selectors resolve through the arm registry before
logging. Logging does not probe a lazy arm backend or power hardware. Skill
arguments and routing behavior are unchanged.

[`TraceLogger.record()`](../src/cascade/agent/trace.py) keeps `context` optional
for caller compatibility. Reading legacy traces remains supported, but absent
or incomplete arm/possession context cannot produce new notes. Do not fill in
missing historical context by guessing the robot or object.

`diagnose()` retains both contexts and the confirmation receipt. `distil()`
rechecks their compatibility before writing a note with the observed error,
argument delta and evidence. The note calls this a recorded association: it
must not claim that the parameter change caused success, invent a
re-observation or prescribe fixed workspace values from another robot.

`harvest()` retains its existing per-call deduplication by `(skill, signature)`.
Storage still overwrites the existing file when a note has the same title-derived
slug; there is no migration, bulk deletion or historical revalidation. Retrieval
still matches keywords, not arm/scene compatibility. Recorded scope is guidance
for the reader, not an enforced retrieval filter or cross-task promotion gate.

## Inspect and harvest offline

Run from the checkout root with the development environment installed. Choose
a directory whose immediate children are reviewed sessions containing
`trace.jsonl`; keep unrelated robot/deployment histories separate. The example
uses `runs/`, the CLI default:

```bash
# Inspect first: no library notes or persisted envelope updates.
.venv/bin/python scripts/learn_from_runs.py --runs runs --dry-run --report

# After reviewing the selected sessions, write notes and envelope updates.
.venv/bin/python scripts/learn_from_runs.py --runs runs \
  --library skills_library --envelope ~/.cascade/envelope.json --json
```

The script also ingests outcomes into the operating envelope, a separate
learning path that this admission change does not modify. A dry run may create
an empty library directory; `--report` alone is not read-only. Do not combine
`--dry-run` with `--export-md` when avoiding file output, because that option
explicitly writes an export.

The built-in orchestrator retrieves keyword-matched library notes at task
start. Harvesting stays between sessions; this script neither runs a robot nor
replays the recorded actions in a simulator.

## Verify the contract

```bash
.venv/bin/python -m pytest tests/test_agentic_upgrades.py \
  tests/test_aspire_admission.py tests/test_llm_and_library.py \
  tests/test_arm_rig.py tests/test_verifier_crash.py -q

# Broader non-hardware regression suite.
.venv/bin/python -m pytest tests/ -q
```

The admission tests use synthetic trace fixtures and real dispatch, checker,
logger and library integration with data-only physical callbacks. They cover
negative admission cases, resolved aliases, pre-placement possession,
explicit boundaries and retention of evidence. A backend-probe sentinel checks
that logging alone never accesses lazy hardware. These tests are not physical
trials or measured improvements in robot success rate.

A MuJoCo pick-and-place/reset run can separately check that manipulation still
works and produces scoped traces. It does not establish that this gate learns
better policies, and it does not certify the Isaac/CUDA/OpenClaw deployment.

## Boundaries and research provenance

- No changes to model weights, motion commands, IK, safety limits or the
  postcondition evaluator. Experience-memory and operating-envelope admission
  remain separate and unchanged.
- The gate trusts historical verifier receipts structurally; it does not
  re-evaluate their measurements or establish equality of hidden state,
  scene, object instance, model revision or calibration.
- Held-object context is the runtime's existing global label, not a new
  per-arm possession model. Hosts without explicit task/reset records still
  lack proven episode boundaries; a run directory alone is not one.
- Notes are guidance, not transferable controllers. Matching one retry does
  not prove causality, cross-task generalization or better future performance.

Dream-RSI motivates restricting claims to recorded continuations. Its paper
evaluates exploration controllers with fixed underlying models and execution
interfaces (Section 3, pages 4–6); the experiments concern algorithm
engineering, mathematics and GPU kernels, not robotics.[1] Its semantic-guidance
comparison is specific to ConvDiv (Section 5.1, page 11), not evidence that
repair memory is generally harmful. The pinned repository has no released
implementation or reproduction scripts.[2]

CASCADE does not implement Dream-RSI's policy optimizer. Further work would
need explicit state/episode lineage, held-out episodes and independently
measured new trials before claiming online improvement.

[1] [Dream-RSI paper, pinned revision](https://github.com/zhengkid/Dream-RSI/blob/4149ea9181ab1db80f85717ffda2c9f0f130e85b/papers/Dream-RSI.pdf)
[2] [Dream-RSI repository, same revision](https://github.com/zhengkid/Dream-RSI/tree/4149ea9181ab1db80f85717ffda2c9f0f130e85b)
