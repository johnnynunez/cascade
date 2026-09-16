# Dream-RSI-inspired evidence admission

## Scope

This is a first prerequisite for replay-based improvement, not an implementation of Dream-RSI's policy optimizer. Dream-RSI evaluates exploration controllers against recorded discovery outcomes while keeping the underlying model, evaluator and execution interfaces fixed (Section 3, pages 4 to 6).[1]

Its replay is restricted to recorded continuations. It does not predict new robot outcomes, and the published experiments concern algorithm engineering, mathematics and GPU kernels rather than robotics.[1] The pinned upstream repository has not released its implementation or reproduction scripts.[2]

## What changed in CASCADE

The existing `agent/aspire.py` path associates a failed skill with a later retry, then writes a reusable library entry. Previously, a later `ok: true` for the same skill name was enough, even if the postcondition was unverified or the target had changed.

Newly harvested retry associations now require:

- A literal `ok: true` tool result plus a `confirmed` postcondition, with no contradictory `verified` flag.
- A matching skill and postcondition kind.
- A measured `physics`, `belief`, `gripper` or `visual_diff` channel with evidence and measurements. Command-only `arm` confirmation is not sufficient.
- Matching recorded task parameters, including object/label, destination, spatial hint, coordinates, direction and distance when present.
- Explicit matching resolved arm identity and pre-call held-object context. Placement, throw and handover require a nonempty held-object label on both sides.
- No intervening `reset_scene`, `task_done` or `task_complete: true` result within the existing repair window.

`SkillRuntime.execute()` records `context.arm` and `context.held_object` before the skill body runs. Omitted/default/primary arm selectors resolve to the same registry identity; logging never probes the arm backend. Skill kwargs and routing behavior are unchanged. `TraceLogger.record()` accepts optional context for compatibility, but traces without explicit routing and possession context cannot produce new notes.

`Diagnosis` retains the actual postcondition and both contexts, and `distil()` rechecks their compatibility before writing. Generated notes include the scope and measured confirmation. They no longer inject fixed reBot workspace advice, claim unrecorded re-observation, or describe a later success as proof of a causal repair.

This removes unsupported interpretation from newly generated guidance. It is not proof that a parameter change caused the successful retry. The paper's comparison against semantic guidance is limited to its ConvDiv experiment (Section 5.1, page 11); it is not evidence that repair memory is generally harmful.[1]

## Unchanged behavior

- No changes to Qwen weights, prompts, control commands, IK, collision thresholds or motion limits.
- No changes to the postcondition evaluator.
- Operating-envelope and experience-memory admission are separate paths and are not modified here.
- No new simulator execution, automatic policy deployment or robot motion.
- No deletion or migration of existing library entries. This admission rule applies to newly distilled repairs; manually authored and older notes remain untouched.

## Verification

Run the focused regression suite in the development environment:

```bash
python -m pytest tests/test_agentic_upgrades.py tests/test_aspire_admission.py \
  tests/test_llm_and_library.py tests/test_arm_rig.py tests/test_verifier_crash.py -q
```

The tests use explicitly synthetic trace fixtures and real dispatch/checker/logger integration with data-only physical callbacks. They cover missing/unverified/refuted postconditions, contradictory verification flags, changed task parameters, resolved arm aliases, changed held subjects, explicit task/reset boundaries, mismatched verifier records, command-only confirmation, and preservation of measured evidence. Placement coverage verifies that the trace retains the subject even when the skill clears it. A backend-probe sentinel checks that logging alone never accesses lazy hardware. These are not robot trials or performance measurements.

## Limits and the next research step

Matching recorded parameters and labels does not establish equality of hidden physical state, object instance, scene, model revision or calibration. Held-object context is the runtime's recorded possession label, not independently sensed identity or a new per-arm possession model. The gate trusts the recorded verifier receipt; it does not re-evaluate historical measurements. Legacy traces without context are excluded, not retroactively repaired. Existing library notes are not migrated or deleted.

Only explicit task-end/reset rows stop matching. Hosts and fast paths that do not emit task markers still lack a proven episode boundary; a run directory alone does not establish one. Do not merge unrelated robot/deployment histories and treat the resulting association as a transferable policy. Full episode/state lineage remains future work.

A faithful next step would record explicit decision/state lineage and evaluate bounded stop/continue policies on supported historical prefixes. Unsupported actions or branches must remain unknown. Candidate selection would need episode-level held-out data and fresh, independently graded trials with fixed safety and evaluation rules. A better score on the same replay history is not a guarantee of better online behavior.[1]

Sources:
[1] https://github.com/zhengkid/Dream-RSI/blob/4149ea9181ab1db80f85717ffda2c9f0f130e85b/papers/Dream-RSI.pdf (Dream-RSI paper)
[2] https://github.com/zhengkid/Dream-RSI/tree/4149ea9181ab1db80f85717ffda2c9f0f130e85b (Dream-RSI repository)
