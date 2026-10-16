# Scoped retry evidence admission

**Next action:** inspect what the ASPIRE harvest would admit from your recorded runs, without
writing notes (seconds; `runs/` is the CLI default and must hold reviewed sessions with
`trace.jsonl`):

```bash
uv run python scripts/learn_from_runs.py --runs runs --dry-run --report
```

After reviewing the selected sessions, write notes and envelope updates:

```bash
uv run python scripts/learn_from_runs.py --runs runs \
  --library skills_library --envelope ~/.cascade/envelope.json --json
```

A later success is not necessarily a repair of an earlier failure: the ASPIRE trace-to-library path
writes a guidance note only with matching context and a measured confirmation. This is an admission
safeguard, not a policy optimizer or a change to robot control. Keep unrelated robot or deployment
histories in separate directories. A dry run may create an empty library directory (`--report`
alone is not read-only), and `--export-md` always writes its export. The script also feeds the
operating envelope (a separate path this gate does not modify); it never runs a robot or replays
actions.

## Admission contract

[`agent/aspire.py`](../src/cascade/agent/aspire.py) examines at most the next six trace rows after
a failed call (`REPAIR_WINDOW`). A candidate must pass all of these:

| Check | Required evidence |
|---|---|
| Result | Literal `ok: true`; `verified`, if present, must also be literal `true` |
| Postcondition | `status: confirmed`, the same skill, and its registered postcondition kind |
| Measurement | A `physics`, `belief`, `gripper` or `visual_diff` channel, nonempty evidence text and nonempty measurements; command-only `arm` confirmation is rejected |
| Goal | Equality of the recorded identity-bearing arguments listed below |
| Context | Explicit matching resolved arm identity and pre-call held-object label |
| Boundary | No intervening `reset_scene`, `task_done` or result with `task_complete: true` |

1. Goal comparison uses the recorded `object`, `label`, `query`, `destination`, `arm`,
   `spatial_hint`, `camera`, `x`, `y`, `z`, `direction` and `distance_m`; differently named
   targets are never inferred equal. Other parameters may differ; their delta goes into the note.
2. For `place_at`, `place_on_object`, `throw` and `handover` the held-object label must be
   nonempty on both calls.
3. Examples: a failed then physics-confirmed red-cube grasp on the same arm with matching
   possession is eligible; a blue-cube success after a red-cube failure, a different arm or held
   subject, absent context, an unverified or `verified: false` outcome, or a success after a scene
   reset or task end is rejected.

## Trace context and generated notes

1. [`SkillRuntime.execute()`](../src/cascade/skills/runtime.py) records context (e.g. `{"arm":
   "left", "held_object": "red cube"}`) before the skill body runs, separately from arguments;
   these are recorded labels, not sensed object identities. Selectors resolve through the arm
   registry without probing a lazy backend or powering hardware.
2. [`TraceLogger.record()`](../src/cascade/agent/trace.py) keeps `context` optional; legacy traces
   still load, but absent or incomplete arm/possession context cannot produce notes. Never fill in
   missing historical context by guessing.
3. `diagnose()` keeps both contexts and the confirmation receipt; `distil()` rechecks compatibility
   and writes the observed error, argument delta and evidence as a recorded association (never a
   causal claim, an invented re-observation or another robot's workspace values).
4. `harvest()` folds every eligible run into the note for its `(skill, signature)`; no migration,
   bulk deletion or revalidation. Retrieval matches keywords, not arm/scene compatibility.

## Cross-task promotion gate

Upstream ASPIRE (`aspire/sim/cap/skills/library.py`) promotes a distilled skill only once it recurs
beyond a single task; CASCADE applies the distinct-task reading:

| Field (front matter of `skills_library/<slug>.md`) | Meaning |
|---|---|
| `occurrences` | Distinct run directories that contributed a confirmed retry |
| `source_tasks` | Distinct recorded task strings (whitespace/case-insensitive) |
| `source_runs` | The contributing run names; a run already listed is never counted again |
| `status` | `promoted` when `len(source_tasks) >= min_tasks` (default `PROMOTION_MIN_TASKS = 2`), else `candidate` |

1. **Promoted** notes are what `aspire.retrieve()` injects into the tier-3 context at task start,
   prefixed with their evidence (`[promoted: recurred in 2 distinct tasks, 3 confirmed runs]`).
2. **Candidate** notes stay on disk and are listed by `SkillLibrary.entries()` / `records()` /
   `relevant()`, but never reach the agent context (a run without a recorded task adds an
   occurrence but no task).
3. **Legacy files** without front matter are one occurrence from an unknown task (candidates); the
   unknown task never counts, and re-harvesting that run cannot promote them.
4. **Idempotent:** re-running over the same runs changes nothing (`learned: 0`); counters persist.
5. **Explicit relaxation only:** `SkillLibrary(root, min_tasks=1)` via `memory.skill_min_tasks: 1`
   in `configs/demo.yaml` or `learn_from_runs.py --min-tasks 1`, tagged `[admitted by explicit
   min_tasks=1 ...; not cross-task validated]`. Retrieval recomputes `status` with the configured
   threshold; values below 1 are rejected.

Promotion counts distinct task strings, not distinct scenes, objects or calibrations, and is not
evidence that a note causes success.

## Verify the contract

```bash
uv run --extra dev --extra kinematics pytest tests/test_agentic_upgrades.py \
  tests/test_aspire_admission.py tests/test_aspire_promotion.py \
  tests/test_llm_and_library.py tests/test_arm_rig.py tests/test_verifier_crash.py -q
```

Synthetic trace fixtures with real dispatch, checker, logger and library integration cover negative
admission cases, resolved aliases, pre-placement possession, boundaries, evidence retention and the
promotion gate; a backend-probe sentinel checks that logging never touches lazy hardware. These are
not physical trials or measured success-rate improvements; the broader non-hardware suite is CI's
`uv run pytest tests/ -q` after its `uv sync` with every extra (`.github/workflows/ci.yml`).

## Boundaries and research provenance

1. No changes to model weights, motion commands, IK, safety limits or the postcondition evaluator;
   experience memory (tier-2 recipes) and operating-envelope admission are separate and do not read
   this gate's receipts.
2. Historical verifier receipts are trusted structurally, not re-measured; hidden state, scene,
   object instance, model revision and calibration are not proven equal.
3. Held-object context is the runtime's global label, not a per-arm possession model; a run
   directory alone is not a proven episode boundary.
4. Notes are guidance, not controllers; two distinct tasks is upstream's recurrence threshold, not
   proof of causality or generalization.
5. Dream-RSI motivates restricting claims to recorded continuations; its experiments (algorithm
   engineering, mathematics, GPU kernels) are not robotics, its semantic-guidance result is specific
   to ConvDiv, and its pinned repository has no released implementation.[1][2] CASCADE does not
   implement its policy optimizer; online improvement would need episode lineage, held-out episodes
   and independently measured new trials.

[1] [Dream-RSI paper, pinned revision](https://github.com/zhengkid/Dream-RSI/blob/4149ea9181ab1db80f85717ffda2c9f0f130e85b/papers/Dream-RSI.pdf)
[2] [Dream-RSI repository, same revision](https://github.com/zhengkid/Dream-RSI/tree/4149ea9181ab1db80f85717ffda2c9f0f130e85b)
