# Agent harnessing (Harness²)

**Next action:** run one improvement step on the MuJoCo SO-101 stack. You need a logged-in Codex
CLI as the agent's brain and Claude Code as the improver; one task takes about 20-40 min:

```bash
CASCADE_OCCUPANCY=0 MUJOCO_GL=egl .venv/bin/python scripts/harness2_improve.py \
  --tasks benchmark/harness2/so101_mujoco_tasks.json --arm so101_mujoco --camera mujoco_scene \
  --llm codex_astra --improver claude --k 1 --mode parallel
```

The command writes `runs/harness2/<stamp>/` with `SUMMARY.md`, one `REPORT.md` per task and
`promoted/harness.json`. To serve that harness, set `agent.harness_path:
runs/harness2/<stamp>/promoted/harness.json` (or `CASCADE_AGENT_HARNESS=<path>`) for
`python -m cascade.apps.demo`.

`cascade.harnessing` improves the built-in agent (`AgentOrchestrator`) without changing model
weights or any safety value. It edits only the advice text around the model, runs every edit on the
same task on a real CASCADE stack, and keeps what CASCADE's own verifier confirms.

The method follows Harness² ([arXiv:2610.14224](https://arxiv.org/abs/2610.14224),
[google-research/harness2](https://github.com/google-research/harness2) @ `1cfe254`, Apache-2.0).
The improver prompts in `src/cascade/harnessing/prompts.py` are adapted from it.

Status:

1. **Works (software):** surface validation, recursion, ledger, blind selection, gate, library and
   CLI; CPU tests in `tests/test_agent_harnessing.py`, including the real executor on the mock stack.
2. **Measured:** see [evidence](evidence/agent-harnessing-20261018/REPORT.md) for the MuJoCo run.
3. **Not shown:** the kitchen route (OpenClaw on Isaac), the served local brain as executor or
   improver, physical hardware.

## What a step does

For each task (one family, in order):

1. **Base:** `k` rollouts of the current harness.
2. **Propose:**
   - `k` INV proposals (how the agent works: tool use, refusals, completion);
   - `k` DOM proposals (manipulation knowledge: objects, approach, destinations).
3. **Probe:** each candidate runs once on a fresh runtime.
4. **Compose:** `k` compositions from all the evidence (merge, adopt one, or `KEEP_BASE`), each
   run once.
5. **Select blind:** base and final executions, shuffled under random ids, with the harness hidden.
6. **Gate:**
   - default (`--gate verified`): a selected final is adopted only if CASCADE's verifier confirmed
     that execution (`TaskReport.success`: postconditions on physics, belief or gripper channels,
     milestones, no stuck step); otherwise the base is kept;
   - `--gate none` is Harness²'s behaviour.
7. **Reflect:**
   - one row per edit: `helped`, `hurt`, `no observed effect` or `confounded`;
   - at most one lesson (`cue`, `lesson`, `evidence`, `confidence`), or `NO_ENTRY`.
8. **Library:** the adopted harness and the experience are appended under
   `~/.cascade/harness2/library/<scope>/<family>/`. The next task in the family starts from that
   harness and reads earlier entries as examples.

Budget per task (asserted by the ledger):
- `4k` rollouts (parallel: `k` base + `2k` candidates + `k` finals; sequential: `k` base + `k`
  rounds of 2 candidates + 1 composition);
- `3k` improver sessions;
- plus one selection and one reflection.

## The editable surface

| component | where it goes | caps |
|---|---|---|
| `rules` | appended to the system prompt on every task | 12 lines, 240 characters each, 1500 total |
| `tool_guidance` | appended to one tool's description | 10 tools, 300 characters each |
| `skills` | injected into a task's intro when a trigger word occurs in the task | 8 skills, 1-8 trigger words, 800 characters |

Everything else is locked:
- the safety harness and every `safety.*`, `grasp.*` and gate value;
- profiles and code;
- the tools `rotate_joint`, `move_relative`, `undo_move`, `reset_scene`, `halt_motion` and `throw`,
  which a harness may neither guide nor name.

A harness is validated before any rollout uses it, and again when `AgentOrchestrator` installs it
(a refused harness stops the start). The checks:
- caps;
- known and unlocked tools only;
- a lint that refuses text which weakens a guard, claims success against a failure, conceals a
  failure or claims these notes outrank other rules.

The lint is the second line of defence: a harness has no channel to configuration.

## Rules the improver works under

These are Harness²'s rules, kept:
- **Answer-free:** the improver never sees a measured outcome, success label or score. It sees the
  trajectories (the agent's reasoning, every tool call and the result the agent received, and its
  own `task_done` claim). Measured outcomes live in `rollout.json` and the library's `meta.json`,
  which never enter an improver's context.
- **Evidence is data:** trajectories, tool results and library entries are quoted as data, never
  followed as instructions.
- **`KEEP_BASE`** is a real answer. An invalid proposal gets one retry with its problems listed,
  then inherits the harness it started from.
- **Facts vs procedure:** scene facts are anchored ("in this scene ...") and library facts are
  re-verified against the current trajectories.

## Rollouts

`CascadeExecutor` builds the configured stack fresh for every rollout.
- Every learned store is private to the rollout directory: beliefs, grasp memory, envelope,
  episodic and action-object stores, and the skill library.
- There is no fast planner (habits/experience), no programs tier and no ASPIRE notes, so a contrast
  between rollouts is the harness's.
- Any LLM profile can be the agent (`--llm`).

Improvers:
- `claude` (`claude -p`, Read/Glob/Grep only);
- `codex` (`codex exec`, read-only sandbox, no user config or MCP servers);
- `llm` (any LLM profile, with the context inlined, e.g. the served local brain);
- `ScriptedImprover` for tests.

## Limits

- One family, one stack per run; rollouts run one at a time (each builds its own runtime).
- The measured outcome is CASCADE's verifier on the chosen stack. On MuJoCo, that is MuJoCo's
  physics, not the kitchen's or hardware's.
- A harness adopted for one brain is evidence about that brain only. The library scope includes the
  arm, camera, LLM profile and model.
