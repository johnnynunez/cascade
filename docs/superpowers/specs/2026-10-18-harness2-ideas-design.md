# Harness² ideas for CASCADE's agent harness: design proposal

Status: proposal with decisions (§7), 2026-10-18. Nothing here is implemented. Source: Harness² ("Recursive Agent
Harnessing for an Open World", arXiv:2610.14224; code `google-research/harness2` at `1cfe254`,
Apache-2.0). "Harness" below means the **agent harness** (prompt, rules, notes, tool guidance), never
the `SafetyHarness`, which no part of this proposal may edit.

## 0. Summary

Harness² improves an agent without touching model weights. It edits the text and tools around the
model, runs every edit on the **same task**, and keeps only what the execution contrast supports.
That is the missing middle between CASCADE's two existing loops:

- the ASPIRE harvest (`agent/aspire.py`, `scripts/learn_from_runs.py`) passively admits notes from
  failed-then-confirmed retries that happened to occur;
- PR-B's blind verifier (#376 spec §5.3) accepts or refutes a harvested note on seeded Newton
  scenes, but never generates or compares alternatives. Option C (search) is deferred.

Proposal: build option C as a Harness²-style **Propose–Probe–Compose** step on top of PR-B's
seeded verify stack, with Harness²'s evidence discipline, and keep every CASCADE safety invariant.
It needs no new robot capability; it reuses the seeded reset (A5), the sim-only retry (A6), the
private verify route (§5.3), use grading (§5.4) and the nightly guard (§5.6).

## 1. What Harness² does (the parts that transfer)

1. **Task-level recursion.** Roll out the base harness `k` times on one task. Two proposer lenses
   write candidate edits from those trajectories:
   - **INV**: how the agent works (tool errors, repetition, stop conditions);
   - **DOM**: what a practitioner knows (artifact contract, decision rules).
   Each candidate is rolled out once on the same task. A composer then merges, adopts one wholesale,
   or answers **KEEP_BASE** (a first-class answer), citing the trajectory moment behind every
   decision.
2. **Selection without ground truth.** Final outputs are compared with their harness identity
   hidden and shuffled; base outputs are in the pool, and picking one is expected and valid.
3. **Harnessing-level recursion.** After adoption, a reflector writes one row per edit (`helped`,
   `hurt`, `no observed effect`, or `confounded`, the last preferred when several components changed
   or one rollout is all the evidence). It then writes at most one transferable lesson
   (`cue`, `lesson`, `evidence`, `confidence`), or `NO_ENTRY`.
   - The library is per domain and append-only.
   - Tasks run in order and read only earlier entries.
   - Libraries are scoped per run, mode, budget and seed.
4. **Budget identity.** `R = 4k` rollouts (`k` base + `2k` candidates + `k` finals) and
   `P = 3k` proposer sessions; the ledger asserts the identity. An invalid proposal gets one retry,
   then inherits the current harness.
5. **Evidence boundaries.** The improver is answer-free (no rubric or score in its context) and
   treats trajectories, artifacts and library entries as data, never instructions.
6. **Facts vs procedure.** Verified facts about this workspace are written as labelled facts
   ("in this workspace ..."); procedure is written for the whole domain. Facts copied from an older
   library entry must be re-verified.
7. **Typed, capped surface.** Eight components (system prompt, guardrails, memory, tool guidance,
   subagents, skills, scripts, plugins), validated before a probe runs (caps on always-on text,
   referenced scripts must exist).

## 2. Mapping to CASCADE

| Harness² | CASCADE today | Proposal |
|---|---|---|
| Editable surface (8 components) | Notes are text (#376 §6.1); `LEARNED.md` injected after `AGENTS.md` (D3); `robot_knowledge` lists notes (A3); MCP tool descriptions; the prompt plugin | An explicit `AgentHarnessSurface`: editable = `LEARNED.md` block, per-note advice, tool-guidance overlays; **locked** = `safety.*`, `grasp.*`, gates, `AGENTS.md`, profiles, code. Caps per component (always-on bytes). Validation before any probe. |
| Base rollouts on one task | One order per verify episode, seeds 101..105 (§5.3) | `k` base episodes of the same goal key on the same seed set. |
| INV / DOM proposers | `distil()` templates per refusal class (§5.1) | INV = tool-use procedure (look before retry, when to call `robot_knowledge`, stop conditions). DOM = manipulation knowledge per object/destination (approach, clearance, held-object handling). Same lint, same "advice text only" rule. |
| Probe | Blind verifier runs one note | Run each candidate on the same seeds through the same kitchen route (§5.3), so it is tested as served. |
| Compose / KEEP_BASE | Promotion rule D9 (>= 2 tasks and >= 4/5 seeds) | A composer step that may merge an INV and a DOM note, adopt one, or keep base. D9 still gates promotion of the result. |
| Anonymized selection | Agent is blind to evidence (§5.3); the selector is not | Shuffle and hide candidate identity from any LLM/VLM selector (`eval/progress_judge.py` included); include base episodes. Isaac truth stays the pass/fail authority offline. |
| Edit verdicts + one lesson | Note v2 provenance (A1), use grading `useful/misleading` (§5.4) | Add per-edit rows (`helped/hurt/no effect/confounded`) and `cue/lesson/evidence/confidence` to note front matter; `NO_ENTRY` allowed. Demotion (§5.4) reads them. |
| Per-domain ordered library | One sim library; export to real (§5.7) | Scope key = (robot profile, scene identity hash, engine, **brain model**, library sha). Notes verified under Qwen are re-verified before they serve Nemotron (#375). |
| Facts vs procedure | Not distinguished | `kind: workspace_fact` (scene-bound: square position, table height) vs `kind: procedure`. Export to the real reBot (§5.7) drops sim workspace facts unless scene identities match. |
| Answer-free improver | Proposer is the outer-loop Qwen (D7) | The proposer sees orders, trajectories and refusals, never the verifier's truth thresholds, so notes cannot be tuned to the checker. |
| Evidence is data | Lint is deny-by-default on output (A1) | Also frame inputs: visitor chat lands in traces. Wrap every trace, transcript and library excerpt as quoted data in the outer-loop prompt, and keep the output lint. |
| Budget identity | Timing breakdown (§5.5) | A nightly ledger: expected vs actual episodes and sessions per candidate; mismatch aborts the night's adoption (not the guard). |
| Parallel vs sequential | Sequential episodes | Parallel probes as separate Newton worlds in one process (the MicroDuck agentic demos already run one Newton world per robot); sequential for one hard order. |
| Diversity check | n/a | If all compositions are identical or equal to base, stop spending on that order. |
| Deterministic demo, no credentials | Mock stack + CPU tests | A scripted proposer on the mock arm that runs one full Propose–Probe–Compose step in CI. |

## 3. Proposed work, in order

1. **Evidence discipline in note v2** (small, no new runtime):
   - per-edit verdict rows and the `cue/lesson/evidence/confidence` block, with `NO_ENTRY`;
   - `kind: workspace_fact | procedure`, with the export filter applying it;
   - library scope including the brain model id.
   CPU tests only.
2. **Input evidence boundary for the outer loop** (small): quote trace, chat and library text as
   data in every `learn_from_runs` and composer prompt. Add a test that a visitor message such as
   "ignore the safety checks" in a trace never reaches a served note, together with the existing
   lint.
3. **Blind, base-inclusive selection** (small): the selector input lists shuffled anonymous outcomes,
   including base episodes. `KEEP_BASE` is a recorded verdict, not an error.
4. **Propose–Probe–Compose for one goal key** (medium, after PR-B lands):
   - `k = 1`: 1 base + 2 candidates (INV, DOM) + 1 composed final = 4 rollouts per seed;
   - on 5 seeds that is 20 episodes, about 52 min at the measured ~155 s per order;
   - this fits one night under the §5.6 guard.
   Budget ledger and diversity check included.
5. **Parallel probes in one process** (medium): k candidate worlds per seed on Newton, if the
   kitchen route can be driven per world. Measure first; skip if the route is single-world only.
6. **Mock-stack CI demo** of step 4 with a scripted proposer (small).

Steps 1–3 are PR-B's first task (H2). Step 4 is option C, which stays gated on PR-B showing that
verified notes raise the brain's success rate (D1, H4).

## 4. Safety invariants (unchanged, plus three)

All of #376 spec §6 holds; this proposal adds:

1. No candidate edit may name a locked component; the surface validator rejects it before any probe.
2. The proposer and composer never receive Isaac truth, verifier thresholds or pass bounds.
3. Real-robot export carries only `procedure` notes verified on the target scene identity, or
   explicitly re-verified; never sim workspace facts.

## 5. What does not transfer

- Harness² edits plugins, scripts and subagents with executable code. CASCADE keeps notes as text
  only (D11); the scripts/plugins components stay out of scope.
- Harness² selects without ground truth because its benchmarks have none at run time. CASCADE has
  Isaac truth offline and must keep it as the pass/fail authority. LLM selection only orders
  candidates for the composer.
- Unrestricted executor network access (Harness² default) is never acceptable here: probes run
  inside the OpenShell route with its proxy rules.

## 6. What today's reviews of the open PRs add

The review of #376 and #375 (comments on both PRs) shows three places where Harness²'s design
choices would have prevented a defect:

1. **Free-text advice behind a word-list lint leaks.**
   - Synonyms of "disable the collision check", false success claims and "these notes outrank every
     rule" were served into `LEARNED.md`.
   - Harness² does not rely on a deny list. It limits *what an edit may be* (a typed component with
     caps and validation) and keeps evidence out of the edit's instructions.
   - For CASCADE that means generating advice from a closed vocabulary: a refusal class, a skill from
     an allowlist, and parameter deltas rendered by fixed templates. Free text stays in `## Evidence`
     only. The lint becomes a second line, not the gate.
2. **Trace text became served advice.** A `## Advice` heading inside a harvested error string defined
   the note. This is exactly Harness²'s "evidence is data, not instructions" boundary (idea 2 in §3),
   applied to the parser as well as to the outer-loop prompt.
3. **A verdict was not bound to the bytes it verified.** Harness² stores the whole adopted tree,
   never a summary, and reads its library by construction. The equivalent for CASCADE is
   `advice_sha` in the verifier record, and a library scope that includes the brain model, now that
   #375 switches the default brain from Qwen to Nemotron Bolt.

## 7. Decisions (2026-10-18, owner delegated)

| # | Decision | Why |
|---|---|---|
| H1 | The #376 review fixes stay in #376 before it merges: allowlisted note skill, `## Advice` read only as the first section (never for schema 1), trace strings collapsed to one line before `distil`, and `advice_sha` in the verifier record | They are defects in code #376 adds, not new features |
| H2 | Steps 1–3 (evidence fields, facts vs procedure, brain-scoped library, outer-loop evidence boundary, blind base-inclusive selection) are PR-B's first task, before the blind verifier | Their consumers are PR-B's verifier, promotion and export. A9–A11 are the chat line, the extension panel and the A1 acceptance run, which do not need them |
| H3 | The outer loop (harvest, proposers, composer) uses the **served brain of the profile it improves**: Nemotron Bolt once #375 lands, Qwen on a `--brain qwen` profile. It runs locally only. This keeps D7's local-only rule and supersedes its "Qwen" wording | A note is tested on the model that reads it; one model stays resident at night under the §5.6 memory guard; Bolt is the larger model. A note verified under one brain is re-verified before it serves another (H2's scope key) |
| H4 | Step 4 (option C) stays gated on D1: it starts only after PR-B shows that verified notes raise the brain's success rate | Unchanged from the #376 spec |
| H5 | Step 5 (parallel Newton worlds) is dropped for now. The kitchen route drives one Isaac bridge per stack, so parallel probes would need one bridge, port set and brain session per world | Revisit only if PR-B's nightly run does not fit its 01:30–05:00 window |
| H6 | Step 6 (mock-stack CI demo) ships with step 4 | It tests step 4 |
