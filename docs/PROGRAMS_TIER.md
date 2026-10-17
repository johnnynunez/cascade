# Programs tier (Waddle "programs", ROADMAP follow-up #8)

**Next action:** run the tier's tests on the mock stack (about 1 min):

```bash
uv run --extra dev --extra kinematics pytest tests/test_programs_tier.py tests/test_programs_mcp.py -q
```

Enable it with `CASCADE_PROGRAMS=1` (or `agent.programs: true`); the default is
`agent.programs: false`.

Status: implemented behind the opt-in flag (2026-10-08) and exposed to MCP chat hosts as
`list_programs` / `run_program` with opt-in embedding retrieval (2026-10-09, B42). Measured on the
mock stack with scripted brains and the dependency-free `hash` embedder only; nothing ran on a
physical rig or in Isaac.

## Decision: both, under one admission rule

Waddle's tiers are primitives → skills → programs ("full task-specific policies composed from
skills, written fresh per instruction"). CASCADE had nothing above a skill: a task was a
reflex/habit replay (tiers 1–2) or a turn-by-turn LLM loop (tier 3). A program can be **authored**
or **distilled**, and the path never decides trust:

| path | when | what is produced |
| --- | --- | --- |
| authored | the program tier is consulted for a task no reflex/habit plan covers: one LLM text turn writes a program for THIS instruction (Waddle's "written fresh per instruction") | a validated program, executed now |
| distilled | a tier-3 LLM run ends in a VERIFIED success (same trigger as a tier-2 recipe) | the run's motion steps as a program: coordinates → perception queries (`memory/recipes.py`), labels → parameters |

**One admission rule for both:** authorship is never evidence.

1. A program enters the library (`runs/programs.jsonl`) as a **candidate** only from an execution
   in which every registered effect was CONFIRMED by an independent channel (the per-task effects
   ledger, not the step's own result) and no motion step lacked a postcondition.
2. It is **offered for reuse** only once **promoted**: verified in at least `PROMOTION_MIN_TASKS`
   (2) distinct tasks (cascade's own rule, as in `skills/library.py`) and while its
   verified executions outnumber its failed ones.
3. Authored and distilled programs with the same structure fold into one record (one signature);
   the origin is metadata (`origins: [authored, distilled]`).
4. Distillation alone never saves the first run's LLM turns; authoring alone throws verified work
   away and invites trusting a program because its author wrote it.

## What a program is, and what it is not

A declarative JSON object, not code:

```json
{"version": 1,
 "name": "place-beside",
 "description": "put an object a few cm in front of another object",
 "params": {"object": "the object to move", "anchor": "the object to place it beside"},
 "steps": [
   {"tool": "grasp_object", "args": {"label": {"$param": "object"}}},
   {"tool": "place_at", "args": {"$target": {"query": "localize_object",
     "label": {"$param": "anchor"}, "offset_m": [-0.06, 0.0], "args": ["x", "y"]}}}
 ]}
```

1. **Registered tools only**, each checked against its `TOOL_SPECS` schema. Refused as steps:
   `task_done` (a program never declares its own success), `reset_scene`, `halt_motion`,
   `grasp_at_pixel` / `probe_point` (scene-bound pixels), `recall_step` and the live-view tools.
   The MCP stop channel (`emergency_stop`, `reset_stop`) is not in `TOOL_SPECS`, so a program can
   neither stop nor reset stop permission.
2. **No stored raw coordinates:** `place_at`'s `x`/`y`/`z` come only from a `$target` perception
   query (`localize_object(label) + offset_m`, offset within the recipe anchor radius, 0.40 m in
   xy). **Parameters are labels:** `{"$param": name}` fills only free-form strings.
3. **Bounded:** at most 12 steps, 6 parameters, no loops or branches (outcome routing is
   `robotics/graph.py`'s job on the composed `RobotRuntime`).
4. **Not a skill:** the orchestrator runs it as TOP-LEVEL `execute()` calls, so every step gets its
   own trace row (`tier: program`), keyframes, watcher pause, postcondition, effects obligation,
   memory frame and envelope row. The MCP `run_program` is a host-side tool like `world_state`.
5. **Not a second control path:** the runner has no motion API; the SafetyHarness, through each
   skill's `SafeArm`, stays the sole authority that refuses motion.

## Lifecycle

1. **Consult** only when tiers 1/2 had no plan (a failed habit goes straight to tier 3).
2. **Author (one LLM turn):** the prompt carries the composable catalog, the world model as labels
   only (no positions) and at most three PROMOTED programs; the brain answers with a program plus
   bindings, `{"use": <name>, "bind": {...}}`, or `NONE`. Candidates and demoted programs are never
   offered and a `use` of one is refused.
3. **Validate, bind and ground before anything runs:** every `$target` resolves through
   `localize_object` before step 1; one unresolved query aborts with ZERO motion.
4. **Execute and stop honestly:** each step's verdict is read from the effects-ledger rows it added,
   never its self-report. The program stops at the first failed, refused, refuted or UNVERIFIED
   step with a `next_action` (what executed, what the arm holds, do not re-run) and falls through
   to tier 3; a `stuck` step ends the task with the ask relayed verbatim.
5. **Report and admit:** success only through the existing report contract (`_finish_report`).
   Admission needs completion, ≥ 1 registered effect, all confirmed, and no motion step without a
   postcondition (`wave`, `point_at`, `move_relative`, `sort_by_color` keep it out). An admitted
   program that ran steps without verifying counts a loss; an abort before step 1 counts nothing.
   Receipt: `<run>/programs/<n>_<name>.json` (spec, sha256, bindings, grounded queries, per-step
   verdicts).

## Storage and retrieval

`memory/programs.py::ProgramLibrary`: one JSON object per line in `runs/programs.jsonl`
(`memory.programs_path`, `CASCADE_PROGRAMS_PATH`), written atomically. A record holds `signature`
(sha256 of the canonical skeleton: parameters by position of first use, offsets quantized to 1 cm,
name/description excluded), the program, `occurrences`, `source_tasks`, `source_runs`, `origins`,
`losses`, `summary`. Promotion: `n_tasks >= min_tasks` and `occurrences > losses`
(`agent.program_min_tasks: 1` is the explicit relaxation, must be >= 1). A corrupt line costs only
itself.

Retrieval (authoring prompt, MCP `list_programs` with a `query`) sits behind the promotion gate:

1. **keyword overlap** (default): records sharing a content word, by overlap then evidence.
2. **text embedding** (opt-in, B42): with `memory.embedder` / `CASCADE_MEMORY_EMBEDDER` (`hash` |
   `siglip` | `clip`), a record qualifies at the embedder's `text_floor` OR on a shared content word
   (the skill library's floor-or-guard rule), so an embedder can add a paraphrase but never drop a
   keyword match. Measured with `hash` (floor 0.3) only; SigLIP/CLIP text floors are uncalibrated.
3. Several processes may share one store (the CLI and one MCP server per chat session): every write
   takes an advisory lock (`<store>.lock`), re-reads and applies its change to the current records;
   readers adopt others' writes with `refresh()`.

## MCP chat hosts

With the tier on (`agent.programs: true` / `CASCADE_PROGRAMS=1`; arm servers only, never
mobile/composed) the MCP server offers two host-side tools; the host's `run_program` call replaces
the authoring turn and everything after it is the CLI's code.

1. **`list_programs(query?, limit?)`**: PROMOTED programs only, ranked as above, with parameters,
   normalized steps, evidence (`distinct_tasks`, `verified_runs`, `failed_runs`) and a ready
   `run_with`; one naming a tool this server withholds or the operator hid carries an
   `unavailable` reason. Source instructions are not exposed.
2. **`run_program(program | spec, bindings, task)`**: exactly one of `program` (a promoted name or
   signature) or `spec` (validated like an authored one; a JSON string is accepted); `task` is
   required. **Storage and promotion: the same rule.** A submitted spec runs once and is stored as a
   candidate only if the ledger CONFIRMED it; a host's claim of success is not consulted. A host
   can never promote a program by writing it, only by running it to a confirmed effect in two
   different tasks.
3. **The served surface binds programs:** a step or `$target` query withheld by the capability
   matrix or hidden by `CASCADE_HIDE_TOOLS` is refused before anything runs. `list_programs` needs
   the `programs` cell; `run_program` needs `programs` and `verifier`.
4. **Stop:** `run_program` is a motion tool: `notifications/cancelled` for it latches the e-stop,
   as do `emergency_stop`, the dashboard STOP and SIGINT. A halt check runs before grounding and
   before every step; the result is `status: stopped`, `estop_latched: true` and nothing resumes
   until `reset_stop`. The program holds the server's execution lock; with the read-only lane
   (`mcp.readonly_lane`, B46) only `world_state`, `robot_knowledge`, `verify_last_action` and
   `camera_snapshot` answer during it.
5. **Result:** `ok` only for a completed, CONFIRMED run; otherwise `isError` with `status`
   (`completed` | `stopped` | `aborted` | `stuck` | `refused`), per-step verdicts, `stopped_at`,
   `reason`, `next_action`, `grounded`, `library` (`outcome`: admitted / loss / not admitted) and the
   receipt path; `last_path` is `program`.

## Refusal and stop cases

| case | where | effect |
| --- | --- | --- |
| unregistered / non-composable tool, bad schema, raw coordinate, pixel tool, numeric param, > 12 steps | validation | program refused, zero motion, tier 3 with the reason |
| `use` of a candidate, demoted or unknown program | retrieval gate | refused, zero motion, tier 3 |
| missing / extra / non-string binding | binding | refused, zero motion, tier 3 |
| a `$target` label not found, or found without a 3D fix (a zone) | grounding | aborted before step 1, zero motion, tier 3 |
| harness refusal (`SafetyViolation`, e-stop latched) | step | stop, `next_action`, tier 3; an admitted program counts a loss (conservative: the library cannot tell an operator stop from an unsafe target, and a loss can only make promotion harder) |
| skill failure, refuted effect, unverified registered effect | step | stop, `next_action`, tier 3; an admitted program counts a loss |
| `stuck` step | step | task ends, ask relayed verbatim, no retry |
| brain answers `NONE`, a tool call, or raises | authoring | no program; tier 3 unchanged |
| MCP: both or neither of `program` / `spec`, no `task`, non-object `bindings` | `run_program` request | refused, zero motion |
| MCP: a step or `$target` query the capability matrix withholds or the operator hid | served surface | refused, zero motion |
| MCP: a stop latched before the call, or arriving while it runs (cancel, `emergency_stop`, dashboard STOP, SIGINT) | halt check + harness | `aborted` with zero motion / `stopped`, no later step dispatched, `next_action`; an admitted program that ran steps counts a loss |

Advisory pieces never veto: rank, promotion and evidence tags only decide what is OFFERED. The
plausibility critic and milestone tracker do not run on program steps.

## Opt-in, and what "off" means

`agent.programs: false` (default) or `CASCADE_PROGRAMS=0` gives `AgentOrchestrator(programs=None)`:
the pre-change dispatch exactly, no store touched (golden-pinned in `tests/test_programs_tier.py`).
On the MCP server, off means `list_programs` and `run_program` are not catalog candidates, there is
no `programs` capability cell, and a cancel of a call named `run_program` is not a motion cancel
(golden-pinned in `tests/test_programs_mcp.py`). Mobile and composed runtimes, and the CLI's mock
brain, never get the tier.

## Not claimed, and still open

1. No physical or simulator acceptance; no claim that programs raise task success or that a real
   brain writes valid programs.
2. ~~Not exposed to MCP chat hosts~~ (landed 2026-10-09, B42); whether a local Qwen writes valid
   specs over MCP, or reuses promoted ones, is unmeasured.
3. No branching or loops; motions without a registered postcondition never make a program
   admissible.
4. ~~Retrieval is keyword overlap, not embedding~~ (landed 2026-10-09, B42: opt-in
   `memory.embedder`, floor-or-guard); SigLIP/CLIP text floors are uncalibrated on real weights.
