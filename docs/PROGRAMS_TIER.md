# Programs tier (Waddle "programs", ROADMAP follow-up #8)

Status: implemented behind an opt-in flag (`agent.programs: false` by default,
`CASCADE_PROGRAMS=1` to enable) on 2026-10-08; measured on the mock stack with
scripted brains only. Nothing here was run on a physical rig or in Isaac.
2026-10-09 (B42): the same switch exposes the tier to MCP chat hosts
(`list_programs`, `run_program`; see [MCP chat hosts](#mcp-chat-hosts)), and an
opt-in memory embedder ranks programs by text embedding; again mock stack and
the dependency-free `hash` embedder only.

## The question

Waddle Labs describes three tiers: `primitives` (fixed platform functions) →
`skills` (agent-authored, reusable, composed from primitives) → `programs`
("full task-specific policies composed from skills, written fresh per
instruction"). CASCADE had the first two (`TOOL_SPECS`, and the ASPIRE notes in
`skills_library/*.md`) and nothing above a skill: every multi-step task was
either a reflex/habit replay (tiers 1–2) or a turn-by-turn LLM tool loop
(tier 3). ROADMAP #8 asked one scoping question before anything landed: is a
program **distilled** the way ASPIRE distils a note (diagnose a successful
multi-skill run, persist it), or **authored proactively** by the agent?

## Decision: both, under one admission rule

A program can come from either path, and the path it came from never decides
whether it is trusted:

| path | when | what is produced |
| --- | --- | --- |
| authored | the program tier is consulted for a task no reflex/habit plan covers: one LLM text turn writes a program for THIS instruction (Waddle's "written fresh per instruction") | a validated program, executed now |
| distilled | a tier-3 LLM run ends in a VERIFIED success (same trigger as a tier-2 recipe) | the run's motion steps as a program: coordinates → perception queries (`memory/recipes.py`), labels → parameters |

**One admission rule for both:** authorship is never evidence. A program enters
the library (`runs/programs.jsonl`) only from an execution in which every
registered effect was CONFIRMED by an independent channel (the per-task
effects ledger, not the step's own result), and no motion step lacked a
postcondition. That makes it a stored **candidate**. It is **offered for reuse**
only once **promoted**: verified in at least `PROMOTION_MIN_TASKS` (2) distinct
tasks -- upstream ASPIRE's cross-task rule, the same constant
`skills/library.py` uses -- and while its verified executions outnumber its
failed ones. Authored and distilled programs with the same structure fold into
one record (one signature), so a program the agent wrote for one instruction
and the same structure distilled from a verified LLM run of another
instruction are two distinct tasks of evidence for one program.

Why not one path only:

- **Distillation only** re-packages what the turn-by-turn loop already did; it
  never saves the LLM turns of the first run and cannot express a program the
  loop has not stumbled into. It is still worth keeping: a verified LLM run is
  free evidence for a program the agent may later reuse.
- **Authoring only** (Waddle) throws that verified work away, and on its own
  invites the failure the ROADMAP warned about: a program trusted because its
  author wrote it. Waddle's thin single `verify(...)` is the contrast, not the
  model.
- Under one admission rule the origin is metadata (`origins: [authored,
  distilled]`), and trust comes only from independent verification across
  distinct tasks.

## What a program is, and what it is not

A program is a declarative JSON object, not code:

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

- **Registered tools only.** Every step names a `TOOL_SPECS` tool and its
  arguments are checked against that tool's schema (unknown keys, missing
  required keys, wrong types, values outside an `enum` or bounds are refused).
  Tools that are not task steps are refused with a reason: `task_done` (a
  program never declares its own success), `reset_scene` (undoes a task),
  `halt_motion` (acts on an in-flight motion), `grasp_at_pixel` /
  `probe_point` (a pixel is a scene-bound coordinate), `recall_step` and the
  live-view tools. The MCP stop channel (`emergency_stop`, `reset_stop`) is not
  in `TOOL_SPECS`, so a program can neither stop nor reset stop permission.
- **No stored raw coordinates.** `place_at`'s `x`/`y`/`z` may only come from a
  `$target` perception query (Task-Specific Memory, `memory/recipes.py`):
  `localize_object(label) + offset_m`, offset within the recipe anchor radius
  (0.40 m in xy). A literal coordinate is refused at validation; nothing is
  ever stored or executed from a remembered position.
- **Parameters are labels.** `{"$param": name}` may only fill a free-form string
  argument; bindings must be non-empty strings. A number cannot be smuggled in
  through a parameter.
- **Bounded:** at most 12 steps, 6 parameters, no loops, no branches. (A
  branching, outcome-routed plan is what `robotics/graph.py`'s bounded skill
  graphs are for, on the composed `RobotRuntime`; a program is the
  manipulation runtime's straight-line task script.)
- **Not a skill.** A program is not in `TOOL_SPECS` and never runs inside a
  `SkillRuntime.execute()` call. The orchestrator executes it as a sequence of
  TOP-LEVEL `execute()` calls, so every step gets exactly what an LLM-issued
  call gets: its own trace row (`tier: program`), BEFORE/AFTER keyframes, watcher
  pause, three-state postcondition, task-effects obligation, Vesta memory frame
  and envelope row. Nesting a program inside one tool call would hide its steps
  from all of that; that is why there is no `run_program` SKILL. (The MCP
  server's `run_program` is a host-side tool like `world_state`, not a skill:
  it hands the program to the same runner, which issues the same top-level
  `execute()` calls -- see [MCP chat hosts](#mcp-chat-hosts).)
- **Not a second control path.** The runner has no motion API. The
  SafetyHarness, reached through each skill's `SafeArm`, remains the sole
  authority that refuses motion; the program validator only decides whether a
  program is well formed, never whether a motion is safe.

## Lifecycle

1. **Consult.** `AgentOrchestrator.run_task` runs tier 1/2 first. Only when
   neither had a plan (no reflex, no habit/recipe) is the program tier
   consulted; a habit that failed goes straight to tier 3, which can see the
   failure.
2. **Author (one LLM turn).** The prompt carries the composable tool catalog,
   the world model as labels only (no positions: a program must not be written
   from coordinates), and at most three PROMOTED programs whose words overlap
   the task, each with its evidence tag. The brain answers with a fresh
   program plus bindings, `{"use": <name>, "bind": {...}}` for a promoted one,
   or `NONE` (a question, or a task that needs to look before deciding).
   Candidates and demoted programs are never offered, and a `use` of one is
   refused.
3. **Validate and bind** (above). A rejected program changes nothing in the
   world; the task falls through to tier 3 with the reason.
4. **Ground before anything runs.** Every `$target` query is resolved through
   the runtime's own `localize_object` (perception, not motion) before step 1.
   One query that does not resolve aborts the program with ZERO motion; there
   is no stored coordinate to fall back to, by construction.
5. **Execute** each step through `runtime.execute()` with `current_tier =
   "program"`. Per step, the verdict is read from the task-effects ledger rows
   that step added -- the checker's own verdict -- never from the result's
   self-report. A result that merely says `postcondition: confirmed` is not
   evidence.
6. **Stop honestly.** The program stops at the first step that is failed,
   refused by the harness, refuted, or a registered effect left UNVERIFIED (a
   program does not build later steps on an effect nobody confirmed --
   continuing would make the program's own authorship the evidence for the
   earlier step). The run carries a `next_action` (what executed, what the arm
   holds, do not re-run the program) and the task falls through to tier 3
   with that note. A `stuck` step ends the task like it does in every tier: the
   ask is relayed verbatim, no retry.
7. **Report and admit.** A completed program is a task success only through
   the existing report contract (`_finish_report` folds every unverified
   obligation in). It is admitted to the library only when its verdict is
   CONFIRMED: completed, at least one registered effect, every registered
   effect confirmed, and no motion step without a postcondition (`wave`,
   `point_at`, `move_relative`, `sort_by_color` never stop a program, but their
   self-report is the only evidence for them, so they keep it out of the
   library). A verified run is also stored as a tier-2 recipe of the same
   instruction, the same rule as a verified LLM-tier run. An execution of an
   ADMITTED program (same signature, whether reused or written again) that ran
   steps without verifying counts a loss; a run aborted before step 1 (zero
   motion, e.g. its anchor is not on this table) counts nothing. Per-run receipt:
   `<run>/programs/<n>_<name>.json` (spec, sha256, bindings, grounded queries,
   per-step verdicts). A distilled run is admitted only when the task-effects
   ledger has at least one row and every row is CONFIRMED; a run with no
   ledger verdict is skipped silently (no evidence, nothing to narrate).

## Storage and retrieval

`memory/programs.py::ProgramLibrary`, one JSON object per line in
`runs/programs.jsonl` (`memory.programs_path`, `CASCADE_PROGRAMS_PATH`),
written atomically. Each record: `signature` (sha256 of the canonical skeleton:
parameter names replaced by position of first use, offsets quantized to 1 cm,
name/description excluded), the program as first admitted, `occurrences`
(distinct verified executions, idempotent per run/task id), `source_tasks`,
`source_runs`, `origins`, `losses`, `summary`. Promotion and retrieval:
`n_tasks >= min_tasks` and `occurrences > losses`; `agent.program_min_tasks: 1`
is the explicit relaxation and must be >= 1. A corrupt line costs only itself.

Retrieval for a task (the authoring prompt's offered programs, and the MCP
`list_programs` with a `query`) has two modes, behind the promotion gate either
way -- a candidate or demoted record is never ranked, however similar:

- **keyword overlap** (default): records sharing a content word with the task,
  by overlap then evidence -- the pre-B42 ranking exactly.
- **text embedding** (opt-in, B42): with a memory embedder configured
  (`memory.embedder`, `CASCADE_MEMORY_EMBEDDER`; backends `hash` | `siglip` |
  `clip`), the task is compared in the embedder's TEXT space with the closest of
  the record's own name + description and each instruction it was verified on.
  A record qualifies at the embedder's `text_floor` OR on a shared content word
  -- the skill library's floor-or-guard rule -- so switching an embedder on can
  add a paraphrase the keywords missed ("shifting cubes frontwards" finds the
  program verified on "shift the green cube front-left") but never drops a
  keyword match. Measured with the dependency-free `hash` embedder (stemmed
  hashed bag of words, floor 0.3); the SigLIP/CLIP text floors are not
  calibrated on real weights (the same caveat as the skill-library notes).

Several processes may share one store -- the CLI and one MCP server per chat
session (OpenClaw). Every write takes an advisory lock (`<store>.lock`),
re-reads the store and applies its change to the CURRENT records, so no process
erases another's records, occurrences or losses; a reader adopts other
processes' writes with `refresh()` (a stat when nothing changed), which
`list_programs` and `run_program(program=...)` call first. On main before B42
the last writer rewrote the file from its own stale view.

## MCP chat hosts

With the tier on (`agent.programs: true` / `CASCADE_PROGRAMS=1`, the CLI's
switch; arm servers only, never mobile/composed), the MCP server offers two
host-side tools. The host is the brain here, so the CLI's mock-brain exclusion
does not apply: the host's `run_program` call takes the place of the authoring
turn and everything after it is the CLI's own code (`ProgramTier`,
`run_program`, `account`).

- **`list_programs(query?, limit?)`** -- the PROMOTED programs only, ranked for
  `query` as above (all promoted, most evidence first, without one), each with
  its parameters, normalized steps, `describe()` summary, evidence
  (`distinct_tasks`, `verified_runs`, `failed_runs`) and a ready `run_with`.
  Programs are re-validated against the current catalog; one that names a
  tool this server withholds or the operator hid carries an `unavailable`
  reason. The source instructions are not exposed (other sessions' words).
- **`run_program(program | spec, bindings, task)`** -- exactly one of
  `program` (a promoted program's name or signature) or `spec` (a new program,
  validated like an authored one; a JSON-encoded string is accepted). `task`
  is required: it is the user's instruction and the key for distinct-task
  promotion. A `program` that is a candidate, demoted or unknown is refused
  with zero motion, like a `use` of one.
- **Storage and promotion: the same rule.** A submitted spec runs once. It is
  stored -- as a candidate -- only if that execution was fully CONFIRMED by the
  task-effects ledger; the program's own `verdict` is the only evidence (a
  chat host's claim of success is not consulted). It is offered by
  `list_programs` and runnable by name only once verified in >= 2 distinct
  tasks and more often than it failed. An execution of an admitted structure
  (by name, or the same structure resubmitted) that ran steps without
  verifying counts a loss. A host can therefore never promote a program by
  writing it, only by running it to a confirmed effect in two different tasks.
- **The served surface binds programs too.** The runner calls `execute()`
  directly, so before anything runs the server refuses a program whose step
  or `$target` grounding query (`localize_object`) is withheld by the
  capability matrix (e.g. 3D tools on an RGB-only rig) or hidden by
  `CASCADE_HIDE_TOOLS`. Capability cells: `list_programs` needs `programs`
  (the library opened); `run_program` needs `programs` and `verifier` --
  without the verifier no step could ever be confirmed. The `programs` cell
  exists only when the tier is attached, so with the tier off the matrix,
  banner and `tools_withheld` are the pre-tier ones.
- **Stop.** A program run moves the arm, so `run_program` is treated like a
  motion tool: `notifications/cancelled` for an in-flight `run_program`
  latches the e-stop (the stdin reader, never queued), as do `emergency_stop`,
  the dashboard STOP and SIGINT. The runner gets a halt check
  (`run_program(..., halt=...)`) evaluated before grounding and before every
  step: once a stop is latched no further step is dispatched; a step in flight
  is refused (or its effect refuted) by the harness/ledger. The result is
  `status: stopped` with `estop_latched: true` and a `next_action`; nothing
  resumes until `reset_stop`. A latched e-stop before the call means the
  program never starts (`aborted`, zero motion). The whole program holds the
  server's execution lock, so the dashboard reflex chat cannot interleave a
  motion. With the opt-in read-only lane (`mcp.readonly_lane`, B46) a
  `run_program` in flight counts as a motion: `world_state`,
  `robot_knowledge`, `verify_last_action` and `camera_snapshot` answer during
  it (marked `served_during_motion`); every other call still waits for the
  whole program.
- **Result.** `ok` is true only for a completed, CONFIRMED run; otherwise
  `isError` with `status` (`completed` | `stopped` | `aborted` | `stuck` |
  `refused`), per-step verdicts and evidence, `stopped_at`, `reason`,
  `next_action` (what executed, what the gripper holds, "do not re-run"),
  `grounded` queries, `library` (`outcome`: admitted / loss / not admitted,
  the record's status and evidence) and the per-run receipt path
  (`<run>/programs/<n>_<name>.json`, as for the CLI). Each step is its own
  trace row with `tier: program`; `last_path` is `program`.

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

Advisory pieces never veto: library rank, promotion and the evidence tags only
decide what is OFFERED to the brain. The pre-motion plausibility critic is not
consulted for program steps (the tier is LLM-free after its authoring turn,
like tier 2), and the milestone tracker is not run on it.

## Opt-in, and what "off" means

`agent.programs: false` (default) or `CASCADE_PROGRAMS=0` gives
`AgentOrchestrator(programs=None)`, whose dispatch is the pre-change path
exactly: no authoring turn, no extra runtime write, no extra message, no store
touched (pinned against a golden in `tests/test_programs_tier.py`). Mobile and
composed runtimes never get the tier and the mock brain (a labelled script)
never gets it from the CLI. On the MCP server, off means `list_programs` and
`run_program` are not catalog candidates (a call is an unknown tool, exactly
as before), the runtime carries no tier, the capability matrix has no
`programs` cell, a cancel of a call named `run_program` is not a motion
cancel, and no store is touched (golden-pinned in
`tests/test_programs_mcp.py`).

## Not claimed, and still open

- No physical or simulator acceptance: every measurement is the mock stack with
  scripted brains and, where a test needs a confirmed effect, the stand-in
  physics channel the recipe tests use. No claim that programs raise task
  success or that a real brain writes valid programs.
- ~~Not exposed to MCP chat hosts~~ (landed 2026-10-09, B42: `list_programs` /
  `run_program`, above). Not measured with a real host brain: whether a local
  Qwen writes valid specs over MCP, or reuses promoted ones, is unmeasured.
- No branching or loops (skill graphs cover outcome routing on the composed
  runtime); motions without a registered postcondition can never make a
  program admissible.
- ~~Retrieval is keyword overlap, not embedding~~ (landed 2026-10-09, B42: opt-in
  `memory.embedder`, floor-or-guard). Measured with the `hash` embedder only;
  the SigLIP/CLIP text floors are uncalibrated on real weights.
