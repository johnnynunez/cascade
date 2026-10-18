# ASPIRE demo port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learned notes reach the production Qwen agent with provenance, the sim agent gets one bounded retry after a refusal, and (PR-B) notes are promoted only after a blind Newton verifier passes them.

**Architecture:** One read-only library snapshot per runtime (`skills/notes.py`) feeds the kitchen prompt file `LEARNED.md` (injected by the kitchen plugin), `robot_knowledge`, the chat line, the extension panel and the trace stamp. A per-runtime retry ledger (`agent/sim_retry.py`), built only for simulation profiles, grants one retry per goal and applies the changed approach itself. PR-B adds offline admission, grading, a verifier harness driven through the OpenClaw CLI, a guarded nightly runner and a verified-only export.

**Tech Stack:** Python 3.12 (stdlib + numpy already in the rig venv), pytest, Node 24/26 (`node --test`, OpenClaw plugin ESM), Isaac Sim 6.x Newton bridge (`scripts/isaac_bridge.py`), OpenClaw CLI, systemd user timer.

**Spec:** `docs/superpowers/specs/2026-10-17-aspire-demo-design.md` (sections cited as §n).

## Global Constraints

- Base: `main` 42eaddd; branch `feat/aspire-notes`; PR-A0 docs first, then PR-A1, then PR-B (each its own PR).
- Engine: every Isaac run `--engine newton`; brain Qwen3.8-27B (served id `qwen3.8-27b`); no cloud model in any product path.
- Safety: no safety margin, gate value or exemption changes; notes are advice text only; nothing a note says reaches config, arguments or gates (§6).
- Real robot in PR-A1: for every non-sim profile, payloads and prompts are byte-identical to main (§6.3).
- `profile_class(cfg) == "sim"` only when every arm `type` is in `{"isaac", "mujoco", "mock"}` and `CASCADE_BOOTH` is unset; else `"real"`.
- Retry: `agent.sim_retry` default `false` (`configs/demo.yaml`), `true` only in `configs/arms/isaac_kitchen.yaml`; kill switch `CASCADE_SIM_RETRY=0`; one retry per goal.
- Budgets: block <= 1400 chars, `LEARNED.md` <= 2000 UTF-8 bytes, top 2 notes; `robot_knowledge` notes <= 6000 chars; advice <= 300 chars; visitor block <= 400 words.
- Library paths: kitchen `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/sim/library}`; `rebot_rs` (B7) `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/real/library}`; others unchanged. The runtime never creates or writes the library.
- Seeds: A1 layouts 1..8; verifier dev seeds 101..105; held-out 201..208.
- Verifier/nightly private ports: brain 18491, proxy 18492, GraspGen-X 5576, bridge 8621, gateway 18796; profile `cascade-aspire-verify`.
- Guarded ports (LISTEN means skip): 5556, 5566, 8080, 8081, 8090, 8091, 8092, 8611, 9333, 17670, 17710, 17810, 17910, 18481, 18482, 18741, 18789, 18790, 18799.
- Never edit `/home/spark/Projects/cascade`; never touch the default OpenShell instance; `AGENTS.md` (repo) is approval-gated: propose text in the PR body only.
- Agent-facing strings are copied verbatim from spec §4.5, §4.6, §4.8.

## Review Focus

1. **A retry renamed by the agent** ("cube" after "green cube"): must count as the same goal and get no second retry. Test: A6 `test_renamed_object_is_the_same_goal`.
2. **A provisional hold at the refusal** (`_held_provisional`, a jaw that closed on something): `held_object` set means no retry and no auto-home. Test: A7 `test_refusal_with_a_provisional_hold_gets_no_retry`.
3. **Mixed two-arm rig** (one `isaac` arm + one `rebot_rs` arm via `arms:`): class `real`. Test: A1 `test_mixed_rig_is_real`.
4. **Library edited while a session runs** (nightly harvest beside a dev session): run.json, `robot_knowledge` and the block keep the build-time snapshot. Test: A3 `test_knowledge_keeps_the_build_snapshot_after_a_file_changes`.
5. **Non-ASCII titles and the byte budget**: a block of CJK titles stays <= 2000 bytes and the plugin still injects it. Test: A4 `test_learned_file_fits_2000_bytes_with_non_ascii_titles`.

---

## Conventions (every task)

```bash
WT=/home/spark/Projects/cascade-wt/aspire-notes
PY=/home/spark/Projects/cascade/.venv/bin/python      # rig venv, read-only use
cd $WT && PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 $PY -m pytest tests/<file> -q -p no:cacheprovider
M=~/.hermes/skills/robotics/cascade-feature-worktree/scripts/mutate.sh   # bash $M <file> '<old>' '<new>' tests/<file>
```

- Tests that build runtimes use the mock stack and, for observed-finger paths, the `_GateDouble` / `_gate_on` /
  `mock_runtime` (gate off and on) pattern of `tests/test_agent_results_compose.py`.
- "Real-profile runtime without hardware": monkeypatch `cascade.apps.demo.make_arm` to a `MockArm` subclass before
  `build_runtime(cfg, tmp_path/'run', lazy_arm=True)`.
- Refused before motion = `len(arm.commands) == n0`.
- Lint: `/home/spark/.local/bin/uvx ruff check <files>`; compare counts with a detached base worktree.
- Explicit working directory on every shell command; files written with write_file, never heredocs.
- After the last task of a PR: full-suite failing-ID diff vs base (skill "Full-suite baseline diff").

## File map

| File | Responsibility | Task |
|---|---|---|
| `src/cascade/sim_profile.py` (new) | `profile_class`, `sim_retry_enabled`, `learning_surface_enabled` | A1 |
| `src/cascade/skills/notes.py` (new) | note v2 parse, lint, tag, `library_sha`, `LibrarySnapshot` | A1 |
| `src/cascade/agent/trace.py` | `stamp()`, row `library_sha` | A2 |
| `src/cascade/apps/demo.py` | snapshot at build, run.json, ledger, learning state | A2, A7, A9 |
| `src/cascade/apps/mcp_server.py` | `robot_knowledge` notes, learning.json path | A3, A9 |
| `src/cascade/skills/learned_block.py` (new) | `LEARNED.md` generator + CLI | A4 |
| `demo/kitchen/openclaw-plugin/{index.mjs,attendee-tools.json}` | inject `LEARNED.md`; 7th tool; R1 identity | A4, A8 |
| `src/cascade/sim/scene_seed.py` (new), `scripts/isaac_bridge.py`, `src/cascade/sim/isaac_reset.py` | seeded reset | A5 |
| `src/cascade/agent/sim_retry.py` (new) | goal, refusal class, ledger, texts | A6 |
| `src/cascade/skills/runtime.py` | ledger wiring, auto-home, exclusion, timing | A7, A9, B4 |
| `demo/kitchen/visitor-instructions.md` | R1, R3, R4, R5 | A8 |
| `src/cascade/agent/learning_surface.py` (new) | chat line, `learning.json` | A9 |
| `scripts/spark_cameras.py`, `scripts/spark_browser.py`, `src/cascade/apps/stream_server.py`, `extensions/chrome/{learning.mjs,panel.mjs,panel.html,package.py,manifest.json}` | panel | A10 |
| `src/cascade/agent/aspire.py`, `src/cascade/skills/library.py` | MCP-run admission, v2 distil, promotion | B1, B2 |
| `src/cascade/agent/aspire_grade.py` (new) | use grading, demotion | B3 |
| `benchmark/aspire/{verify_note.py,controls/*.md}` (new) | verifier | B5 |
| `scripts/aspire_nightly.py`, `deploy/aspire/cascade-aspire-nightly.{service,timer}` (new) | nightly + guard | B6 |
| `scripts/aspire_export.py` (new), `configs/arms/rebot_rs.yaml` | export | B7 |

---

# PR-A0 (docs, Q9)

### Task A0: Credit ASPIRE's real API and call the 2-task rule cascade's own

**Files:**
- Create: `tests/test_aspire_attribution.py`
- Modify: `docs/ARCHITECTURE.md:32,333`, `docs/PROGRAMS_TIER.md:35`, `docs/DREAM_RSI_ADAPTATION.md:66-68`,
  `src/cascade/skills/library.py:10-19,30-32,157`, `src/cascade/agent/aspire.py:15-21`, `src/cascade/memory/programs.py:19`,
  `src/cascade/apps/demo.py:2147`, `scripts/learn_from_runs.py:50-53`, `configs/demo.yaml` (`skill_min_tasks` comment),
  `tests/test_aspire_promotion.py:1-5`, `tests/test_agentic_upgrades.py:526`

**Interfaces:** none.

- [ ] **Step 1: Write the failing test**

```python
CLAIMS = re.compile(r"upstream ASPIRE'?s? (cross-task |promotion )?rule|follows upstream ASPIRE|"
                    r"Upstream ASPIRE promotes|Cross-task gate \(upstream ASPIRE\)|as upstream ASPIRE does", re.I)
SKIP = ("docs/superpowers/", "tests/test_aspire_attribution.py")

def test_no_file_claims_cascades_promotion_rule_is_upstreams():
    hits = [f"{p.relative_to(REPO)}:{i}" for p in _text_files(("src", "scripts", "configs", "docs", "tests"))
            if not str(p.relative_to(REPO)).startswith(SKIP)
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1) if CLAIMS.search(line)]
    assert hits == []

def test_architecture_credits_aspire_with_its_own_api():
    line = next(l for l in (REPO / "docs/ARCHITECTURE.md").read_text().splitlines() if l.startswith("- **ASPIRE**"))
    for name in ("get_observation", "localize_object", "generate_grasps", "select_grasp", "move_to_pose"):
        assert name in line
    assert "`preview_grasp` is cascade's own" in line
```

- [ ] **Step 2: Run it:** `pytest tests/test_aspire_attribution.py` -> FAIL (lists ~11 hits; missing names).
- [ ] **Step 3: Edit the listed lines** with the spec §4.9 wording ("cascade's own rule: >= 2 distinct tasks. Upstream
  v0.1 `get_promoted_skills(min_occurrences=2)` counts occurrences in any task; v1.0 promotes on a blind verifier's
  verdict."). Docstrings only; no code change.
- [ ] **Step 4: Run** `tests/test_aspire_attribution.py tests/test_aspire_promotion.py tests/test_agentic_upgrades.py
  tests/test_llm_and_library.py` -> PASS.
- [ ] **Mutation:** `bash $M src/cascade/skills/library.py "cascade's own rule" "upstream ASPIRE's rule" tests/test_aspire_attribution.py` -> KILLED.
- [ ] **Commit:** `git commit -m "docs(aspire): credit ASPIRE's real API; the 2-task promotion rule is cascade's own"`
  (push as its own PR `docs/aspire-attribution`, base main).

---

# PR-A1 (owner's "A")

### Task A1: Profile class, note v2, library snapshot and hash

**Files:**
- Create: `src/cascade/sim_profile.py`, `src/cascade/skills/notes.py`, `tests/test_sim_profile.py`, `tests/test_aspire_notes.py`

**Interfaces:**
- Produces (`cascade.sim_profile`): `SIM_ARM_TYPES = frozenset({"isaac", "mujoco", "mock"})`;
  `arm_types(cfg) -> list[str]` (via `apps.demo._arm_cfgs`); `profile_class(cfg) -> Literal["sim", "real"]`;
  `sim_retry_enabled(cfg) -> bool`; `learning_surface_enabled(cfg) -> bool` (`profile_class` independent; reads
  `agent.learning_surface`).
- Produces (`cascade.skills.notes`): `ADVICE_MAX_CHARS = 300`, `BLOCK_MAX_CHARS = 1400`, `LEARNED_MAX_BYTES = 2000`,
  `KNOWLEDGE_MAX_CHARS = 6000`, `LEARNED_HEADER_PREFIX = "<!-- cascade-learned-notes v1 "`;
  `@dataclass(frozen=True) Note(id, title, when, advice, kind, status, learned_in, n_tasks, occurrences, verifier, grades)`;
  `parse_note(path: Path) -> Note | None`; `library_sha(root: Path | None) -> str`; `lint_advice(text: str) -> list[str]`;
  `note_tag(note: Note, sha: str) -> str`; `@dataclass(frozen=True) LibrarySnapshot(root, sha, profile_class, notes,
  served)` with `block() -> str`, `knowledge() -> str`, `matching(goal: str) -> Note | None`, `manifest() -> dict`;
  `load_snapshot(root: Path | None, profile_class: str, *, min_tasks: int = 2) -> LibrarySnapshot`.

- [ ] **Step 1: Write the failing tests** (`tests/test_sim_profile.py`)

```python
@pytest.mark.parametrize("arm", ["rebot_rs", "rebot_rs_mb", "so101", "fr3", "piper", "ros2_generic"])
def test_hardware_profiles_are_real(arm):
    assert profile_class(load_demo_config(arm=arm)) == "real"

@pytest.mark.parametrize("arm", ["mock", "isaac", "isaac_kitchen", "isaac_kitchen_gpu", "so101_mujoco"])
def test_sim_profiles_are_sim(arm):
    assert profile_class(load_demo_config(arm=arm)) == "sim"

def test_mixed_rig_is_real():                      # Review Focus 3
    cfg = load_demo_config(arm="isaac"); cfg._data["arms"] = [dict(cfg.arm._data), {"type": "rebot_rs"}]
    assert profile_class(cfg) == "real"

@pytest.mark.parametrize("arm_cfg", [{}, {"type": None}, {"type": "isaac_next"}])
def test_missing_or_unknown_type_is_real(arm_cfg): ...      # cfg.arm replaced -> "real"

def test_booth_makes_a_sim_profile_real(monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "1"); assert profile_class(load_demo_config(arm="isaac_kitchen_gpu")) == "real"

def test_sim_retry_needs_sim_class_flag_and_no_kill_switch(monkeypatch):
    cfg = load_demo_config(arm="isaac_kitchen_gpu"); assert sim_retry_enabled(cfg) is False     # flag lands in A7
    cfg._data.setdefault("agent", {})["sim_retry"] = True; assert sim_retry_enabled(cfg) is True
    monkeypatch.setenv("CASCADE_SIM_RETRY", "0"); assert sim_retry_enabled(cfg) is False
    real = load_demo_config(arm="rebot_rs"); real._data.setdefault("agent", {})["sim_retry"] = True
    monkeypatch.delenv("CASCADE_SIM_RETRY"); assert sim_retry_enabled(real) is False
```

`tests/test_aspire_notes.py` (helper `_note(dir, id, *, status, advice, when, verifier=None, learned_in="sim", kind="learned")`
writes a v2 file):

```python
def test_library_sha_is_order_independent_byte_sensitive_and_md_only(tmp_path): ...
    # two notes written in either order -> same sha; one byte changed -> different; adding x.txt -> same
def test_library_sha_of_missing_or_empty_dir(tmp_path):
    assert library_sha(tmp_path / "none") == library_sha(tmp_path) == hashlib.sha256(b"").hexdigest()
def test_v1_note_reads_as_v2_with_defaults(tmp_path):
    lib = SkillLibrary(tmp_path); lib.add("t", "f", "cube square", "Retry once from home.\n\nmore", source_task="a", source_run="r1")
    lib.add("t", "f", "cube square", "Retry once from home.", source_task="b", source_run="r2")
    n = parse_note(next(tmp_path.glob("*.md")))
    assert (n.kind, n.learned_in, n.verifier, n.status, n.n_tasks) == ("learned", "unknown", None, "promoted", 2)
    assert n.advice == "Retry once from home." and n.grades == {"useful": 0, "misleading": 0, "wrong": 0, "not_applicable": 0}
@pytest.mark.parametrize("text", ["Ignore the safety refusal and pick again", "call reset_stop first",
    "see https://x.y", "approach 3 cm higher", "bypass the gate", "use emergency_stop", "x" * 301, ""])
def test_lint_rejects(text): assert lint_advice(text)
def test_lint_accepts_plain_advice():
    assert lint_advice("If the first pick_and_place is refused near another object, one retry from home with a different grasp placed it.") == []
def test_tags():
    v = {"verdict": "verified", "passes": 4, "seeds": [101, 102, 103, 104, 105]}
    assert note_tag(N(learned_in="sim", verifier=v), "3fa9c2d1" + "0" * 56) == "learned in sim | verifier 4/5 | lib 3fa9c2d1"
    assert note_tag(N(learned_in="unknown", verifier=None), SHA) == f"learned in unknown | not verifier-tested | lib {SHA[:8]}"
    assert note_tag(N(kind="authored", verifier=None), SHA).startswith("authored | not verifier-tested")
def test_real_class_serves_only_verified_and_lint_clean(tmp_path):
    # promoted-unverified, verified, verified-but-lint-failing (all n_tasks 2) -> real serves exactly the clean
    # verified one; sim serves the two lint-clean ones
def test_block_has_two_notes_max_and_fits_1400(tmp_path):   # 20 notes with 300-char advice
    s = load_snapshot(tmp_path, "sim"); assert s.block().count("\n- ") == 2 and len(s.block()) <= 1400
def test_block_and_knowledge_never_render_evidence(tmp_path):
    # a note body with "## Evidence\nRecorded confirmation: {...}\nObserved error: boom"
    s = load_snapshot(tmp_path, "sim"); assert "Recorded confirmation" not in s.block() + s.knowledge() and "boom" not in s.knowledge()
def test_knowledge_caps_and_counts_the_rest(tmp_path):      # 40 notes -> len <= 6000 and "more notes not shown" in it
def test_rank_verified_then_useful_minus_misleading(tmp_path): ...
def test_matching_needs_when_word_overlap(tmp_path):
    assert s.matching("pick_and_place green cube -> green square").id == "cube-note" and s.matching("orange") is None
def test_load_snapshot_creates_nothing_and_skips_unreadable(tmp_path):
    s = load_snapshot(tmp_path / "missing", "sim"); assert not (tmp_path / "missing").exists() and s.served == ()
    # a note with broken front matter is skipped; the sha still covers its bytes
```

- [ ] **Step 2: Run** both files -> FAIL (`ModuleNotFoundError: cascade.sim_profile`).
- [ ] **Step 3: Implement** `cascade/sim_profile.py` (pure config reads, any exception -> `"real"`; booth =
  `cascade.config.booth_mode_enabled()`) and `cascade/skills/notes.py`:
  - front matter via `skills.library.split_front_matter`; v1 defaults per spec §4.1; `advice` = `## Advice` section,
    else first paragraph of `## Strategy`, cut at 300.
  - lint: regexes per spec §4.1; tool names from `skills.runtime.TOOL_SPECS` + `apps.mcp_server._EXTRA_TOOLS` names,
    imported lazily inside `lint_advice` (keeps `notes` import-light).
  - served: `sim` -> `n_tasks >= min_tasks`, status not `refuted`/`retired`, lint clean (stored `status` string is
    advisory); `real` -> `verifier.verdict == "verified"` and lint clean.
  - rank key: `(verdict != "verified", -(useful - misleading), -n_tasks, id)`.
  - `block()` text per spec §4.3 item 2 (without the header line; `learned_block` adds it); `""` when nothing served.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Mutations** (each KILLED):
  `bash $M src/cascade/sim_profile.py 'all(' 'any(' tests/test_sim_profile.py`;
  `bash $M src/cascade/skills/notes.py 'BLOCK_MAX_CHARS = 1400' 'BLOCK_MAX_CHARS = 14000' tests/test_aspire_notes.py`;
  `bash $M src/cascade/skills/notes.py '== "verified"' '!= "refuted"' tests/test_aspire_notes.py`;
  mutate the lint's `reset_stop` alternative away -> KILLED.
- [ ] **Commit:** `feat(aspire): note v2 with provenance tags, library hash and a read-only snapshot; fail-closed profile class`

### Task A2: Snapshot at build, run.json and the trace stamp

**Files:**
- Modify: `src/cascade/agent/trace.py` (`TraceLogger`), `src/cascade/apps/demo.py:588-` (`build_runtime`, after
  `trace = TraceLogger(run_dir)` ~860), `configs/arms/isaac_kitchen.yaml` (`overrides.memory.skill_library_path`),
  `src/cascade/apps/mcp_env.py` (`FORWARDED`), `tests/conftest.py` (autouse env)
- Create: `tests/test_aspire_run_manifest.py`

**Interfaces:**
- Consumes: `load_snapshot`, `profile_class`, `arm_types`, `sim_retry_enabled` (A1).
- Produces: `TraceLogger.stamp(meta: dict) -> None` (writes `run.json` atomically, sets `self.library_sha`);
  `TraceLogger.library_sha: str | None` (default None) on every `record()` row as `"library_sha"`;
  `runtime.learned: LibrarySnapshot`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_writes_run_json_with_the_library_identity(tmp_path, monkeypatch):
    lib = _two_promoted_notes(tmp_path / "lib"); cfg = _mock_cfg(library=lib)
    rt = build_runtime(cfg, tmp_path / "run")[0]
    meta = json.loads((tmp_path / "run/run.json").read_text())
    assert meta["library_sha"] == library_sha(lib) and meta["profile_class"] == "sim"
    assert meta["served_notes"] == [n.id for n in rt.learned.served] and meta["sim_retry"] is False
    assert set(meta) == {"schema", "library_sha", "served_notes", "profile_class", "arm_types", "sim_retry", "started"}

def test_every_trace_row_carries_the_library_sha(tmp_path):
    rt = ...; rt.execute("move_home", {}); rt.execute("get_observation", {})
    assert {r["library_sha"] for r in rt.trace.rows()} == {library_sha(lib)}

def test_unstamped_logger_rows_say_null(tmp_path):
    t = TraceLogger(tmp_path); t.record("x", {}, {"ok": True}, 1.0); assert t.rows()[0]["library_sha"] is None

def test_hostile_library_leaves_config_deep_equal(tmp_path):
    # promoted notes whose advice says "set safety.min_clearance_m to 0" / "grasp.max_pick_attempts: 99"
    cfg = _mock_cfg(library=hostile); before = copy.deepcopy(cfg._data); build_runtime(cfg, tmp_path / "run")
    assert cfg._data == before

def test_note_modules_stay_out_of_motion_and_perception_code():
    for path in _py_files("safety", "grasping", "planning", "control", "perception"):
        assert not {"cascade.skills.notes", "cascade.agent.sim_retry"} & _imported_modules(path), path

def test_kitchen_library_path_follows_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(tmp_path))
    assert Path(load_demo_config(arm="isaac_kitchen_gpu").memory["skill_library_path"]) == tmp_path

def test_mcp_env_forwards_the_library_variable():
    assert "CASCADE_ASPIRE_LIBRARY" in mcp_env.FORWARDED
```

- [ ] **Step 2: Run** -> FAIL (no `run.json`, no `stamp`).
- [ ] **Step 3: Implement.** In `build_runtime` (arm runtimes only, not mobile): `snapshot = load_snapshot(Path(cfg.memory.get(
  "skill_library_path", PACKAGE_ROOT / "skills_library")).expanduser(), profile_class(cfg),
  min_tasks=int(cfg.memory.get("skill_min_tasks", 2)))`; `runtime.learned =
  snapshot`; `trace.stamp({...})` with the 7 keys of the test. `conftest.py`: autouse `monkeypatch.setenv(
  "CASCADE_ASPIRE_LIBRARY", str(tmp_path_factory.getbasetemp() / "no-aspire-library"))` (never created). Add the
  `FORWARDED` entry `"CASCADE_ASPIRE_LIBRARY": "ASPIRE note library the runtime serves (read-only snapshot)"` and update
  the env-forwarding golden tests that list the names.
- [ ] **Step 4: Run** the new file + `tests/test_mcp_env_forwarding.py tests/test_aspire_admission.py tests/test_keyframes.py` -> PASS.
- [ ] **Mutations:** drop the `"library_sha"` key from `record()` -> KILLED; replace `profile_class(cfg)` by `"sim"` in
  `build_runtime` -> KILLED by A3's real-profile test once it exists (record in the task report).
- [ ] **Commit:** `feat(aspire): every run stamps its library sha in run.json and on each trace row`

### Task A3: `robot_knowledge` lists the served notes

**Files:**
- Modify: `src/cascade/apps/mcp_server.py:220-231` (description), `:1460-1477` (`_robot_knowledge`)
- Create: `tests/test_aspire_knowledge.py`

**Interfaces:**
- Consumes: `runtime.learned` (A2), `LibrarySnapshot.knowledge()`.
- Produces: payload keys `learned_notes: str`, `library_sha: str`, added only when `runtime.learned.served`.

- [ ] **Step 1: Write the failing tests** (call `McpSkillServer._robot_knowledge` on a server built like
  `tests/test_mcp_readonly_lane.py` builds one, or `McpSkillServer.__new__(McpSkillServer)` with the runtime passed in)

```python
def test_knowledge_is_unchanged_without_served_notes(empty_runtime):
    out = _knowledge(empty_runtime); assert set(out) == {"ok", "primitive_envelopes", "failure_models", "stats", "grasp_priors"}
def test_knowledge_lists_served_notes_with_their_tags(sim_runtime):
    out = _knowledge(sim_runtime)
    assert out["library_sha"] == sim_runtime.learned.sha
    for n in sim_runtime.learned.served: assert n.title in out["learned_notes"] and note_tag(n, out["library_sha"]) in out["learned_notes"]
def test_real_profile_hides_unverified_notes(monkeypatch, tmp_path):     # rebot_rs, make_arm -> MockArm
    out = _knowledge(_real_runtime(library=_promoted_unverified(tmp_path))); assert "learned_notes" not in out
def test_knowledge_keeps_the_build_snapshot_after_a_file_changes(sim_runtime):   # Review Focus 4
    before = _knowledge(sim_runtime); _rewrite_a_note(sim_runtime.learned.root); assert _knowledge(sim_runtime) == before
def test_lane_and_serial_answers_match(sim_runtime): ...   # call_readonly(...) payload == call_tool payload
```

- [ ] **Step 2: Run** -> FAIL (`learned_notes` missing).
- [ ] **Step 3: Implement** in `_robot_knowledge`: `snap = getattr(runtime, "learned", None)`; if `snap and snap.served`:
  `out["learned_notes"] = snap.knowledge(); out["library_sha"] = snap.sha`. Description: append "It also lists
  learned notes with where they were learned and whether a verifier passed them."
- [ ] **Step 4: Run** the file + `tests/test_mcp_readonly_lane.py tests/test_programs_mcp.py` -> PASS.
- [ ] **Mutation:** `bash $M src/cascade/apps/mcp_server.py 'if snap and snap.served' 'if snap' tests/test_aspire_knowledge.py` -> KILLED.
- [ ] **Commit:** `feat(mcp): robot_knowledge lists the served learned notes with provenance`

### Task A4: `LEARNED.md` for the kitchen prompt; `robot_knowledge` becomes the 7th attendee tool

**Files:**
- Create: `src/cascade/skills/learned_block.py`, `tests/test_learned_block.py`
- Modify: `demo/kitchen/openclaw-plugin/index.mjs` (`before_prompt_build`), `demo/kitchen/openclaw-plugin/attendee-tools.json`,
  `scripts/launch.sh` (spark branch after the visitor sync, ~1176-1182), `deploy/brev/visitor_chat.py:21`,
  `scripts/demo_proof.py:415`, `tests/test_openclaw_natural.py:11,13`, `tests/test_attendee_english.py:63`,
  `tests/test_spark_openclaw_prompt.py` (new cases), `tests/test_spark_visitor_chat.py` (seven tools)

**Interfaces:**
- Consumes: `load_snapshot`, `LEARNED_HEADER_PREFIX`, `LEARNED_MAX_BYTES` (A1).
- Produces: `sync_learned_notes(workspace: Path, snapshot: LibrarySnapshot) -> bool` (changed?);
  `python -m cascade.skills.learned_block --workspace <abs dir>` (reads `CASCADE_ARM`, `CASCADE_CAMERAS` like
  `launch.sh:954`; exit 0, prints `wrote|removed|unchanged <path> library_sha=<hex>`).

- [ ] **Step 1: Write the failing tests** (`tests/test_learned_block.py`, plus node cases via `run_hook`)

```python
def test_writes_header_block_atomically_0600(tmp_path):
    assert sync_learned_notes(tmp_path, snap_with_two) is True
    text = (tmp_path / "LEARNED.md").read_text()
    assert text.startswith(f"<!-- cascade-learned-notes v1 library_sha={snap_with_two.sha} notes=")
    assert stat.S_IMODE((tmp_path / "LEARNED.md").stat().st_mode) == 0o600
    assert sync_learned_notes(tmp_path, snap_with_two) is False          # unchanged -> no rewrite
def test_nothing_served_removes_our_file(tmp_path): ...                  # previous generated file -> removed
def test_refuses_hand_written_or_symlinked_file(tmp_path):
    (tmp_path / "LEARNED.md").write_text("operator notes\n")
    with pytest.raises(ValueError, match="LEARNED.md"): sync_learned_notes(tmp_path, snap_with_two)
    assert (tmp_path / "LEARNED.md").read_text() == "operator notes\n"
def test_learned_file_fits_2000_bytes_with_non_ascii_titles(tmp_path):    # Review Focus 5
    sync_learned_notes(tmp_path, snap_cjk); assert len((tmp_path / "LEARNED.md").read_bytes()) <= 2000
def test_regenerates_when_the_library_changes(tmp_path): ...             # different sha -> rewritten
def test_agents_md_is_never_touched(tmp_path):
    (tmp_path / "AGENTS.md").write_bytes(b"x\n"); sync_learned_notes(tmp_path, snap_with_two)
    assert (tmp_path / "AGENTS.md").read_bytes() == b"x\n"
```

Plugin (`tests/test_spark_openclaw_prompt.py`):

```python
def test_learned_block_sits_between_agents_md_and_identity(tmp_path):
    # AGENTS.md "A\n" + LEARNED.md (valid header + block) -> systemPrompt == "A\n" + "\n\n" + block + "\n\n" + identity...
def test_invalid_or_oversize_learned_file_is_ignored(tmp_path):
    # no header / 2001 bytes -> systemPrompt identical to the no-LEARNED.md prompt; no throw
def test_no_learned_file_keeps_todays_prompt(tmp_path): ...
```

Tool lists: `NAMES` in `test_openclaw_natural.py` and the set in `test_attendee_english.py:63` gain
`"robot_knowledge"`; `test_spark_visitor_chat.py` renames `..._exact_six_tools_...` to `..._exact_seven_tools_...`.

- [ ] **Step 2: Run** the four test files -> FAIL.
- [ ] **Step 3: Implement.** `learned_block.py`: file = header line + `snapshot.block()` + `"\n"`; when the encoded file
  exceeds 2000 bytes, shorten the longest advice by whole words until it fits; temp file in the workspace +
  `os.replace`, mode 0600; refuse symlinks and files whose first line lacks the prefix. `index.mjs`: read
  `join(workspaceDir, "LEARNED.md")` in a try; inject only when `Buffer.byteLength(text) <= 2000` and the first line
  starts with the prefix; the block is the text after the first line. `attendee-tools.json`: add the
  `robot_knowledge` entry with the spec §4.3 description and `{"type":"object","properties":{},"required":[],
  "additionalProperties":false}`. `launch.sh`: after the visitor heredoc, `(cd "$REPO/models" && CASCADE_ARM="$ARM"
  CASCADE_CAMERAS="$CAMERAS" PYTHONPATH="$REPO/src" "$PY" -m cascade.skills.learned_block --workspace
  "$PROFILE_WORKSPACE") || die "could not write the kitchen learned notes in $PROFILE_WORKSPACE"`. `visitor_chat.TOOLS`
  and `demo_proof` read tools gain `robot_knowledge`.
- [ ] **Step 4: Run** the four files + `tests/test_spark_launch_profile.py tests/test_launch_openshell_route.py` -> PASS.
- [ ] **Mutations:** `bash $M demo/kitchen/openclaw-plugin/index.mjs '<= 2000' '<= 200000' tests/test_spark_openclaw_prompt.py` -> KILLED;
  remove the header check in `sync_learned_notes` -> KILLED by `test_refuses_hand_written_or_symlinked_file`.
- [ ] **Commit:** `feat(kitchen): the prompt plugin injects generated LEARNED.md; robot_knowledge joins the attendee tools`

### Task A5: Seeded `reset_scene` in Isaac (simulation only)

**Files:**
- Create: `src/cascade/sim/scene_seed.py`, `tests/test_scene_seed.py`
- Modify: `scripts/isaac_bridge.py` (op dispatch ~1390, `_reset_props_verified` ~2029, `_settle_props` ~2000-2027),
  `src/cascade/sim/isaac_reset.py` (`validate_isaac_reset`), `src/cascade/skills/runtime.py` (`skill_reset_scene`, Isaac
  branch ~7130), `tests/test_isaac_reset_delivery.py` (new cases)

**Interfaces:**
- Produces: `seeded_spawns(spawns: dict[str, tuple[float, float, float]], seed: int, *, footprints: dict[str, float],
  keepouts: list[tuple[float, float, float, float]], jitter_m=0.04, min_gap_m=0.01, reach_m=0.45, max_tries=50)
  -> tuple[dict[str, tuple[float, float, float]], list[str]]`; `kitchen_layout(scene_config: dict) -> tuple[dict, dict,
  list]` (spawns, footprint radii from `dimensions`, keepout rectangles of `target_pad` and `open_box`); bridge op
  `{"op": "set_reset_seed", "seed": int | null}` -> `{"ok": true, "seed": ...}`; `reset_props` reply adds `seed`,
  `seed_fallbacks`; `skill_reset_scene` result adds `scene_seed`.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_zero_and_none_keep_the_default_spawns(): assert seeded_spawns(S, 0, **K)[0] == S
def test_same_seed_same_layout_other_seed_other_layout():
    assert seeded_spawns(S, 7, **K) == seeded_spawns(S, 7, **K) != seeded_spawns(S, 8, **K)
@pytest.mark.parametrize("seed", range(1, 201))
def test_layouts_respect_gaps_keepouts_reach_and_jitter(seed):
    out, fallbacks = seeded_spawns(S, seed, **K)
    for a, b in itertools.combinations(out, 2): assert _gap(out, a, b) >= 0.01 - 1e-9
    for name, p in out.items():
        assert math.hypot(p[0], p[1]) <= 0.45 and not _in_keepout(p, K["footprints"][name], K["keepouts"])
        assert math.dist(p[:2], S[name][:2]) <= 0.04 + 1e-9 and p[2] == S[name][2]
def test_unplaceable_prop_keeps_its_spawn_and_is_listed():   # keepout covering a prop's whole jitter disc
    out, fallbacks = seeded_spawns(S, 3, **blocked); assert out["lemon"] == S["lemon"] and fallbacks == ["lemon"]
```

Bridge (AST-extracted env of `tests/test_isaac_reset_delivery.py`):

```python
def test_set_reset_seed_then_reset_places_and_verifies_seeded_spawns(env):
    _dispatch(env, {"op": "set_reset_seed", "seed": 5}); r = _pump(env, {"op": "reset_props"})
    want, _ = seeded_spawns(env["_PROP_SPAWNS"], 5, **kitchen_layout(env["_scene_config"])[1:])
    assert r["seed"] == 5 and {n: c["spawn_position_m"] for n, c in r["reset_verification"]["props"].items()} == {n: list(p) for n, p in want.items()}
def test_no_seed_is_todays_reset(env): ...                # reply without "seed" key change except seed None; spawns == _PROP_SPAWNS
def test_validator_accepts_an_integer_seed_and_rejects_others(): ...   # seed "5" / 5.0 / True -> BridgeError
def test_reset_scene_result_carries_the_seed(runtime_with_isaac_double): assert res["scene_seed"] == 5
```

- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement.** `scene_seed.py`: `random.Random(seed)`, props in sorted name order, uniform point in the
  jitter disc, `for attempt in range(max_tries)` per prop, rectangle-vs-circle keepout test, pairwise circle gap. Bridge:
  module global `_RESET_SEED = None`; `set_reset_seed` validates `int` (not bool) or null; `_settle_props` and
  `_reset_props_verified` take `spawns = _reset_spawns()` (seeded or `_PROP_SPAWNS`) instead of reading `_PROP_SPAWNS`
  directly; the 2 cm tolerance and settle loop are unchanged. Validator: optional `seed` must be `int` or None.
- [ ] **Step 4: Run** the files + `tests/test_newton_teleport.py tests/test_isaac_bridge.py tests/test_reset_scene.py` -> PASS.
- [ ] **Mutations:** `bash $M src/cascade/sim/scene_seed.py 'min_gap_m=0.01' 'min_gap_m=-0.05' tests/test_scene_seed.py` -> KILLED;
  replace `_reset_spawns()` by `_PROP_SPAWNS` in `_reset_props_verified` -> KILLED.
- [ ] **Commit:** `feat(sim): seeded prop layouts for the Isaac reset (bridge set_reset_seed; default unchanged)`

### Task A6: The retry ledger (pure)

**Files:**
- Create: `src/cascade/agent/sim_retry.py`, `tests/test_sim_retry_ledger.py`

**Interfaces:**
- Produces (as built, 2fe139d + A6 fix 1; supersedes the first draft, which A7 must not copy):
  `RETRY_SKILLS = frozenset({"pick_and_place", "grasp_object"})`; `@dataclass(frozen=True) Goal(name: str, words:
  frozenset, destination: frozenset, arm: frozenset)` (`str(goal)` = `name`, display only); `goal_of(name, args) ->
  Goal | None` (words = NFKC + casefold, `re.findall(r"[^\W_]+")`, determiners a/an/the/this/that/these/those
  dropped, no stemming; an object without words is no goal); `same_goal(a: Goal, b: Goal) -> bool` (object words
  nested, destination words nested, arms equal or one absent; TypeError for anything that is not a `Goal`);
  `same_object(a: str, b: str) -> bool`; `refusal_class(exc_type: str, result: dict, *, held: bool, latched: bool,
  halted: bool, episode_pending: bool) -> str | None` (R6: exact exception class name); `@dataclass(frozen=True)
  RetryCall(goal: str, refused_grasps: tuple, arm_left_in_place: bool, refusal: str)`; `class
  SimRetryLedger(*, window_s=360, reserve_s=200, clock=time.monotonic)` (R9/R10; `reserve_s` must be < `MAX_RESERVE_S`
  = 300) with `precheck(name, args) -> dict | None`, `begin(name, args, *, held, latched, episode_pending,
  motion_blocked) -> RetryCall | dict | None` (a dict = the used-slot refusal, the same dict `precheck` returns: return
  it before any motion; None = run as without the ledger), `finish(name, args, result, *, exc_type, held, latched,
  halted, episode_pending, refused_grasps, retry) -> result`, `note_motion(name: str, ok: bool) -> None`,
  `clear(reason: str) -> None`, `last_event: dict | None`, `off: bool` (read-only; A6 fix 2: a fault switches the
  ledger off for good and sets `last_event = {t, kind: "off", where}`); `ledger_for(cfg) -> SimRetryLedger | None`
  (sim class + `sim_retry_enabled` + `CASCADE_OBSERVED_FINGER_GATE == "1"`, else None); constants `DEFAULT_WINDOW_S`,
  `MIN_WINDOW_S`, `DEFAULT_RESERVE_S`, `TURN_BUDGET_S`, `MAX_RESERVE_S`, `RETRY_THINK_S` (30 s: a slot opens only if
  the refused call's time + reserve + this fits TURN_BUDGET_S), `REFUSAL_CLASSES`; texts `RETRY_ALLOWED`,
  `RETRY_USED`, `RETRY_EXHAUSTED` (spec §4.5 verbatim, `{skill}` and `{goal}` formatted), `RETRY_TOO_LATE` (appended
  to the used refusal when `begin` finds the retry too late for the turn: refused before motion, slot used). An arm
  "" / "default" / "primary" is the absent arm; a call matching two or more open slots is no retry (all of them
  used). Slot states open / used / spent (DECISIONS.md "A6 fix 1", "A6 fix 2"). The ledger only sets the `sim_retry`
  field and returns the used-slot refusal;
  composing `next_action` is the runtime's job (A7), so the ledger never edits agent text.

- [ ] **Step 1: Write the failing tests**

```python
REFUSED = {"ok": False, "error": "SafetyViolation: the finger would enter the lemon", "arm_left_in_place": True}
def test_first_refusal_opens_one_slot():
    L = SimRetryLedger(); out = L.finish("pick_and_place", A, dict(REFUSED), held=False, latched=False, refused_grasps=(g1,), retry=None)
    assert out["sim_retry"] == {"allowed": True, "goal": "green cube", "refusal": "safety_refusal"}
    assert out["next_action"] == REFUSED.get("next_action")               # text untouched here
def test_next_same_goal_call_is_the_retry_and_carries_the_exclusions():
    r = L.begin("pick_and_place", A); assert r == RetryCall("green cube", (g1,), True)
    assert L.begin("pick_and_place", A) is None                       # consumed before anything moves
def test_a_failed_retry_is_final_and_a_third_call_is_refused():
    out = L.finish(..., retry=r); assert out["sim_retry"] == {"allowed": False, "used": True}
    pre = L.precheck("pick_and_place", A); assert pre["ok"] is False and pre["error"].startswith("The one simulation retry for green cube is used")
def test_renamed_object_is_the_same_goal():                           # Review Focus 1
    assert same_goal("cube", "green cube") and not same_goal("pink cube", "green cube")
    assert L.precheck("grasp_object", {"label": "cube"}) is not None
@pytest.mark.parametrize("result,held,latched", [
    ({"ok": False, "error": "SafetyViolation: x"}, True, False), ({"ok": False, "error": "SafetyViolation: x"}, False, True),
    ({"ok": False, "error": "SafetyViolation: x", "motion_blocked": True}, False, False),
    ({"ok": False, "error": "SafetyViolation: x", "contact": {"joint": 2}}, False, False),
    ({"ok": False, "outcome": "stuck", "error": "SkillStuck: x"}, False, False),
    ({"ok": False, "action_ok": True, "error": "postcondition failed: ..."}, False, False),
    ({"ok": False, "error": "unexpected RuntimeError: x"}, False, False)])
def test_non_refusals_open_no_slot(result, held, latched): assert refusal_class(result, held=held, latched=latched) is None
@pytest.mark.parametrize("error,cls", [("SafetyViolation: a", "safety_refusal"), ("SlowPerceptionError: a", "perception_refusal"),
    ("SceneChangedError: a", "perception_refusal"), ("PlanningBudgetExceeded: a", "planning_refusal")])
def test_refusal_classes(error, cls): ...
def test_clears(): ...   # confirmed success of the goal; clear("reset_scene"); a motion call with another goal; latched -> cleared
def test_move_home_clears_arm_left_in_place(): L.note_motion("move_home", True); assert L.begin(...).arm_left_in_place is False
def test_success_and_other_skills_are_untouched(): out = L.finish("pick_and_place", A, {"ok": True}, ...); assert "sim_retry" not in out
```

- [ ] **Step 2: Run** -> FAIL (module missing).
- [ ] **Step 3: Implement.** Slots `dict[str, dict]` keyed by the first goal string; lookups via `same_goal` (word-set
  containment after casefold); `refusal_class` follows spec §4.5; `last_event` = `{t, kind: "retry", goal, outcome}`
  after a retry finishes.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Mutations:** `'"used"' -> '"open"'` in the slot transition -> KILLED (third call allowed); remove `same_goal`
  containment (`==` only) -> KILLED; remove the `held` check -> KILLED.
- [ ] **Commit:** `feat(agent): sim-only retry ledger: one retry per goal after a refusal`

### Task A7: Wire the retry into the runtime (sim only), home first, refused grasps excluded

**Files:**
- Modify: `src/cascade/skills/runtime.py` (`execute` ~1230, `_ObservedSelection` ~188-310, `skill_grasp_object` batch
  entries ~3689 and ~3973, `skill_reset_scene` ~6990), `src/cascade/apps/demo.py` (`build_runtime`: ledger),
  `configs/demo.yaml` (`agent.sim_retry: false` + comment), `configs/arms/isaac_kitchen.yaml` (`agent.sim_retry: true`)
- Create: `tests/test_sim_retry_runtime.py`

**Interfaces:**
- Consumes: `ledger_for`, `SimRetryLedger`, `RetryCall`, `RETRY_ALLOWED` / `RETRY_USED` (A6, interface as built:
  see A6 "Produces"); `sim_retry_enabled` (A1).
- Produces: `runtime.sim_retry: SimRetryLedger | None`; `_ObservedSelection.refused_grasps() -> tuple` (attempted
  candidates whose outcome is not `"grasped"`); module function `_without_refused(grasps: list, excluded: tuple) ->
  list` (drops `_same_grasp` / `_near_copy` matches); runtime attributes `_retry_exclude: tuple = ()`,
  `_last_refused_grasps: tuple = ()`; module constant `_PICK_FAILED_GUIDANCE` = the exact "If the user said 'then
  stop', ... manual coordinates." text now inlined at `runtime.py:1653-1656` (moved, unchanged); module function
  `_with_retry_text(result: dict, skill: str) -> dict` (spec §4.5 composition: FAILED line and move_home hint stay in
  front; for `pick_and_place` `_PICK_FAILED_GUIDANCE` is replaced by `RETRY_ALLOWED`, for `grasp_object`
  `RETRY_ALLOWED` is appended; a used retry appends `RETRY_USED`).

- [ ] **Step 1: Write the failing tests** (mock stack, gate off and on via the `mock_runtime` pattern; `agent.sim_retry`
  set true on the mock cfg unless the test says otherwise)

```python
def test_refused_pick_in_sim_allows_one_retry_after_the_failed_line(mock_runtime):
    res = _refused_pick(mock_runtime)
    assert res["next_action"].startswith("This attempt FAILED") and res["sim_retry"]["allowed"] is True
    assert "SIMULATION ONLY" in res["next_action"] and _PICK_FAILED_GUIDANCE not in res["next_action"]
def test_flag_off_keeps_the_then_stop_guidance_byte_identical(mock_runtime_flag_off):
    assert _PICK_FAILED_GUIDANCE in _refused_pick(mock_runtime_flag_off)["next_action"]
def test_retry_goes_home_first_with_the_halt_baseline(mock_runtime, spy_move_home):
    _refused_pick(mock_runtime, left_in_place=True); mock_runtime.execute("pick_and_place", A)
    assert spy_move_home.calls[0]["_halt_generation"] == spy_move_home.generation_at_start
def test_halt_during_the_retry_home_ends_the_call_before_any_pick(mock_runtime, spy_grasp):
    # move_home spy bumps harness._halt_generation -> result error contains "halt", spy_grasp.calls == []
def test_retry_excludes_the_refused_grasp(mock_runtime):            # gate on: _turned_candidates
    first = _refused_pick(...); second = mock_runtime.execute("pick_and_place", A)
    assert not any(_near_copy(r, a) for r in REFUSED_GRASPS for a in _attempted(second))
def test_no_candidate_left_fails_before_motion(mock_runtime):
    n0 = len(arm.commands) after the home; assert "no different grasp candidate" in res["error"] and len(arm.commands) == n0
def test_third_call_is_refused_before_motion(mock_runtime):
    n0 = len(arm.commands); res = mock_runtime.execute("pick_and_place", A)
    assert res["ok"] is False and len(arm.commands) == n0 and "simulation retry" in res["error"]
def test_refusal_with_a_provisional_hold_gets_no_retry(mock_runtime):  # Review Focus 2
def test_halt_or_estop_latched_opens_no_slot(mock_runtime): ...
def test_reset_scene_clears_the_slot(mock_runtime): ...
@pytest.mark.parametrize("arm", ["rebot_rs", "so101", "fr3", "piper", "ros2_generic"])
def test_real_profiles_build_no_ledger_and_payloads_match_flag_off(arm, monkeypatch, tmp_path):
    on, off = _refused_payload(arm, sim_retry=True), _refused_payload(arm, sim_retry=False)
    assert _rt(arm).sim_retry is None and on == off and "sim_retry" not in on
def test_gates_are_identical_with_the_retry_on_and_off():
    a, b = _cfg(sim_retry=True), _cfg(sim_retry=False)
    assert a._data["safety"] == b._data["safety"] and a._data["grasp"] == b._data["grasp"]
```

- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement.** In `build_runtime`: `runtime.sim_retry = ledger_for(cfg)` (A6; None on any real or
  unknown profile, gate off or bad config). A7 moves the R8 gate check (`CASCADE_OBSERVED_FINGER_GATE == "1"`) into
  `sim_retry_enabled` (it owns `test_sim_profile` under R22) or reports `ledger is not None` in run.json, so run.json's
  `sim_retry` (`demo.py:1126`) never says true while no ledger exists. In `execute` (only when `self.sim_retry` is set
  and this is the outermost call): `pre = ledger.precheck(name, args)` -> return `_with_failure_instruction(pre)`
  before any keyframe (its own trace row, `context.sim_retry.refused`, R28); then `step = ledger.begin(name, args,
  held=self._held_or_provisional(), latched=self._stop_latched(arm), episode_pending=<the selected arm's harness
  `_pending_contact_episode` or `_pending_release_episode`>, motion_blocked=<...>)` (R7: all four are required
  keywords): a dict is the same used-slot refusal (another caller used the slot after the precheck) -> return it
  exactly like `pre`; a `RetryCall` makes this call THE retry; None runs the call as without the ledger (no auto-home,
  no exclusions). `g0` is taken inside `_execute_skill` after the arm override, from the harness `skill_move_home`
  will use (R7c), at the retry's start. The auto-home runs
  inside `_execute_skill`'s dispatch, immediately before `result = fn(**args)` (`runtime.py:1482`): when
  `self._retry_call.arm_left_in_place`, `self.skill_move_home(_halt_generation=g0)` then
  `self.arm.harness._check_halt_generation(g0)`; a refusal or MotionHalted there flows through the existing
  `except (SkillError, SafetyViolation)` handler into the call's traced failed result (slot already used, no pick).
  Set `self._retry_exclude = retry.refused_grasps` around the dispatch and
  restore it in `finally`; also keep `self._retry_call = retry` for the dispatch (None otherwise; cleared in
  `finally`). In `_execute_skill`, just before `self.trace.record(...)` (`runtime.py:1742`, after the existing
  `_with_failure_instruction`), so the trace row carries the verdict: `result = ledger.finish(name, args, result,
  exc_type=<the refusing exception's exact class name, R6>, held=bool(self._held_or_provisional()),
  latched=self._stop_latched(arm), halted=self._halted_since_skill_start(...), episode_pending=<as at begin>,
  refused_grasps=self._last_refused_grasps, retry=self._retry_call)` (`_last_refused_grasps` reset at each call start,
  R28), then `_with_retry_text(result, name)`; `trace_context["sim_retry"] = {"retry": True,
  "auto_home": <bool>}` on the retry's row;
  `ledger.note_motion(name, ok)` for every `_MOTION_SKILLS` call; `skill_reset_scene` ok ->
  `ledger.clear("reset_scene")`. In `skill_grasp_object`: `grasps = _without_refused(grasps, self._retry_exclude)`
  before each `selection.add_batch(grasps)` and before `select_grasp(` when the gate is off; an empty list after
  filtering raises `SkillError("no different grasp candidate: every planned grasp repeats the refused one")`. The
  `_observed_selection_on_failure` wrapper stores `self._last_refused_grasps = selection.refused_grasps()`.
- [ ] **Step 4: Run** the file + `tests/test_agent_results_compose.py tests/test_agent_honest_results.py
  tests/test_observed_neighbour_selection.py tests/test_reset_scene.py tests/test_persistence_leftovers.py` with
  `CASCADE_OBSERVED_FINGER_GATE` off and on -> PASS.
- [ ] **Mutations** (each KILLED, read which test killed it): build the ledger without `ledger_for` in
  `build_runtime` (`runtime.sim_retry = SimRetryLedger()`) -> real-profile test; run a call whose `begin` returned the
  used-slot dict (`if isinstance(step, dict)` -> `if False`) -> third-call test; skip the auto-home -> home test; `_halt_generation=g0` ->
  `_halt_generation=None` -> halt test; `_without_refused(grasps, ...)` -> `grasps` -> exclusion test.
- [ ] **Commit:** `feat(runtime): one sim-only retry after a refusal: home first, fresh look, refused grasps excluded`

### Task A8: Rules and recipes (R1-R5)

**Files:**
- Modify: `demo/kitchen/visitor-instructions.md`, `demo/kitchen/openclaw-plugin/index.mjs` (identity string),
  `src/cascade/skills/runtime.py` (`open_gripper` TOOL_SPECS entry ~8414), `tests/test_attendee_english.py`,
  `tests/test_spark_openclaw_prompt.py`
- Create: `tests/test_aspire_rules.py`

**Interfaces:** none (text only; strings verbatim from spec §4.6).

- [ ] **Step 1: Write the failing tests**

```python
VISITOR = (REPO / "demo/kitchen/visitor-instructions.md").read_text()
def test_r1_replaces_the_absolute_no_retry_rule():
    assert "Do not retry, reset automatically or turn a failed result into success." not in VISITOR
    assert "Only when a tool result says SIMULATION ONLY and allows one more attempt, repeat that call once" in VISITOR
def test_r3_r4_r5_are_present():
    for s in ("name describe_scene's detected objects first", "I could not place the <object>:",
              "end with exactly that line"): assert s in VISITOR
def test_visitor_block_stays_within_400_words(): assert len(VISITOR.split()) <= 400
def test_identity_allows_only_a_result_granted_retry():
    src = (REPO / "demo/kitchen/openclaw-plugin/index.mjs").read_text()
    assert "retry only when a tool result allows it" in src and "retry automatically" not in src
def test_r2_open_gripper_says_how_to_set_objects_down():
    d = next(s for s in TOOL_SPECS if s["name"] == "open_gripper")["description"]
    assert d.endswith("open_gripper releases the object where it is, in the air if the arm is raised.")
```

- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Edit the texts** (spec §4.6 verbatim; trim neighbouring sentences only where the 400-word pin needs it).
- [ ] **Step 4: Run** the file + `tests/test_attendee_english.py tests/test_spark_openclaw_prompt.py tests/test_llm_and_library.py
  tests/test_openclaw_natural.py` -> PASS.
- [ ] **Mutation:** restore "never invent success or retry automatically" in `index.mjs` -> KILLED.
- [ ] **Commit:** `feat(kitchen): retry discipline, inventory and placement recipes; open_gripper says use place_at to set down`

### Task A9: The chat line and `learning.json`

**Files:**
- Create: `src/cascade/agent/learning_surface.py`, `tests/test_learning_surface.py`
- Modify: `src/cascade/skills/runtime.py` (`execute`, after `ledger.finish`), `src/cascade/apps/demo.py`
  (`build_runtime`), `src/cascade/apps/mcp_server.py` (pass the launch state path), `configs/demo.yaml`
  (`agent.learning_surface: false`), `configs/arms/isaac_kitchen.yaml` (`true`)

**Interfaces:**
- Consumes: `LibrarySnapshot.matching`, `note_tag` (A1); `SimRetryLedger.last_event` (A6); `learning_surface_enabled` (A1).
- Binding (A3 review fix round, ruling 3; spec §4.8): no visitor surface shows a skipped note's raw id (`run.json`
  `skipped.first`, `LibrarySnapshot.skipped`): any name shown goes through `notes._shown_name`. A note's title is
  shown as the notes routes render it (`notes._ascii`), its tag from `note_tag` (ASCII, " | " separators).
- Produces: `RETRY_LINE` (spec §4.8 text 1); `learning_line(*, retried_placed: bool, result: dict, matched: Note | None,
  sha: str) -> str | None`; `with_learning_line(result: dict, line: str | None) -> dict` (sets `learning_line` and
  appends "End your reply with this exact line: <line>"); `class LearningState(paths: list[Path], snapshot:
  LibrarySnapshot, sim_retry: bool)` with `write(event: dict | None = None) -> None` (atomic, schema 1 per spec §4.8);
  `runtime.learning_state: LearningState | None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_retry_line_wins_and_is_the_only_line():
    line = learning_line(retried_placed=True, result={"ok": True}, matched=NOTE, sha=SHA); assert line == RETRY_LINE
def test_note_line_needs_success_and_a_match():
    assert learning_line(retried_placed=False, result={"ok": True}, matched=NOTE, sha=SHA) == f"From earlier runs: {NOTE.title} ({note_tag(NOTE, SHA)})."
    assert learning_line(retried_placed=False, result={"ok": False}, matched=NOTE, sha=SHA) is None
def test_kitchen_pick_success_with_a_matching_note_ends_with_the_line(sim_runtime_surface_on):
    res = sim_runtime.execute("pick_and_place", A); assert res["next_action"].endswith(f"End your reply with this exact line: {res['learning_line']}")
def test_surface_off_adds_nothing(mock_runtime_default):   # default demo.yaml: no learning_line, no learning.json
def test_learning_json_schema_and_atomic_paths(tmp_path):
    st = LearningState([tmp_path / "a.json", tmp_path / "b.json"], snap, True); st.write({"kind": "retry", ...})
    for p in (tmp_path / "a.json", tmp_path / "b.json"):
        d = json.loads(p.read_text()); assert d["schema"] == 1 and d["library_sha"] == snap.sha and d["last_event"]["kind"] == "retry"
def test_real_profile_payload_unchanged_with_surface_on(...):   # rebot_rs + unverified library -> no learning_line
```

- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement.** `build_runtime`: when `learning_surface_enabled(cfg)`: `runtime.learning_state =
  LearningState([run_dir / "learning.json"] + ([launch_state / "learning.json"] if launch_state else []), snapshot,
  runtime.sim_retry is not None)` and `write()`; `mcp_server` passes `args.launch_state_dir` into the build (keyword
  `learning_dir=`). `execute`: for `_MOTION_SKILLS` results with the switch on, compute the line (`retried_placed` = the
  call was the sanctioned retry and `ok`), apply `with_learning_line`, and `learning_state.write(event)` when a line or
  a retry event exists.
- [ ] **Step 4: Run** the file + `tests/test_wrist_narration.py::test_golden_flag_off_is_byte_identical tests/test_agent_honest_results.py` -> PASS.
- [ ] **Mutation:** drop the `ok` check from the note line -> KILLED.
- [ ] **Commit:** `feat(aspire): one learning line in the reply and learning.json for the panel (kitchen only)`

### Task A10: The extension panel

**Files:**
- Create: `extensions/chrome/learning.mjs`, `extensions/chrome/tests/learning.test.mjs`, `tests/test_spark_learning_state.py`
- Modify: `extensions/chrome/panel.mjs` (`updateCameras`), `extensions/chrome/panel.html` (a hidden `<section id="learning">`),
  `extensions/chrome/package.py` (`FILES` += `learning.mjs`), `extensions/chrome/manifest.json` + `extensions/firefox/manifest.json`
  (version 1.6.0), `scripts/spark_cameras.py` (`--learning-file`), `scripts/spark_browser.py:128`,
  `src/cascade/apps/stream_server.py` (`state()`), `tests/test_spark_browser.py`

**Interfaces:**
- Consumes: the `learning.json` schema (A9).
- Binding (A3 review fix round, ruling 3; spec §4.8): no visitor surface shows a skipped note's raw id (`run.json`
  `skipped.first`, `LibrarySnapshot.skipped`): any name shown goes through `notes._shown_name`. A note's title is
  shown as the notes routes render it (`notes._ascii`), its tag from `note_tag` (ASCII, " | " separators).
- Produces: `learningLines(learning) -> string[]` (pure: `["Learned notes · lib <sha8>", "<title> — <tag>", ...,
  "<last_event.line>"]`, `[]` for anything not schema 1); `spark_cameras.read_learning(path: Path | None) -> dict | None`
  (<= 64 KB, schema 1, else None).

- [ ] **Step 1: Write the failing tests**

```js
test('learning lines from a schema-1 state', () => {
  assert.deepEqual(learningLines({schema: 1, library_sha: '3fa9c2d1' + '0'.repeat(56),
    served: [{id: 'a', title: 'Retry from home', tag: 'learned in sim | not verifier-tested | lib 3fa9c2d1'}],
    last_event: {kind: 'retry', line: 'Learning loop: ...', outcome: 'placed'}}),
    ['Learned notes · lib 3fa9c2d1', 'Retry from home — learned in sim | not verifier-tested | lib 3fa9c2d1', 'Learning loop: ...']);
});
test('anything else renders nothing', () => { for (const v of [undefined, null, {}, {schema: 2}]) assert.deepEqual(learningLines(v), []); });
```

```python
def test_state_includes_learning_only_from_a_valid_file(tmp_path): ...   # valid -> key; missing/oversize/bad json -> no key
def test_camera_surface_gets_the_learning_file_flag(...): ...             # spark_browser command list ends with --learning-file <state>/learning.json
def test_live_view_state_has_no_learning_key_with_the_switch_off(...): ... # mock golden unchanged
```

- [ ] **Step 2: Run** `node --test extensions/chrome/tests/learning.test.mjs` and the pytest file -> FAIL.
- [ ] **Step 3: Implement.** `panel.mjs` calls `learningLines(data.learning)` in `updateCameras` and fills
  `#learning` with one `<p>` per line via `textContent` (section hidden when empty). `spark_cameras.CameraHandler`
  adds `state["learning"]` when `read_learning(self.server.learning_file)` returns a dict. `stream_server.state()`
  adds `learning` from `runtime.learning_state` only when `learning_surface_enabled(cfg)`.
- [ ] **Step 4: Run** `node --test extensions/chrome/tests/*.test.mjs`, `python3 extensions/firefox/package.py --out
  <scratch>`, the pytest file + `tests/test_spark_browser.py tests/test_wrist_narration.py tests/test_local_extension_csp.py` -> PASS.
- [ ] **Mutation:** `bash $M scripts/spark_cameras.py '64 * 1024' '64 * 1024 * 1024' tests/test_spark_learning_state.py` -> KILLED.
- [ ] **Commit:** `feat(extension): learning panel from the camera server's /state (no new port)`

### Task A11: PR-A1 acceptance run (Newton, kitchen route)

**Files:** harness and RESULTS in `.superpowers/sdd/aspire/e2e-a1/` (git-ignored; copy out of scratch).

- [ ] **Step 1:** Write `BRIEF.md` from the skill's `templates/sim-run-subagent-brief.md`: code under test = merge base
  (before) and PR head (after) in detached worktrees; kitchen route with `CASCADE_INSTALL_PROFILE=spark`,
  `CASCADE_OPENCLAW_PROFILE=cascade-aspire-e2e`, `OPENCLAW_GATEWAY_PORT=18796`, `CASCADE_BRIDGE_PORT=8621`, brain
  18491 (+ proxy 18492), GraspGen-X 5576, `--engine newton`, `CASCADE_ASPIRE_LIBRARY=<scratch empty dir>` (plus one
  mini-run of P1 x 3 with a 2-note authored library for the cost and chat-line checks). Step 0 of the brief: a
  `launch.sh --dry-run` with that env; every port/unit/profile it would touch is listed; any fixed live port ->
  stop and report.
- [ ] **Step 2:** Orders and layouts exactly as spec §8 PR-A1 items 1-3, one `openclaw --profile cascade-aspire-e2e
  agent --session-id <new> --message ... --json` per order (`deploy/brev/visitor_chat.py:177`); truth via
  `cascade.sim.truth.TruthPoseReader(BridgeClient(port=8621))`; seeded-reset check of spec §8 item 6 on the after arm.
- [ ] **Step 3:** RESULTS.md table per order (placed, motion calls, retry, first failing call, claim vs truth, prompt
  tokens, turn s) and the pass/fail of each spec §8 PR-A1 criterion with numbers; cleanup per the brief's rule 8.
- [ ] **Step 4:** Paste the before/after table and the deploy notes (kitchen library path, `learning.json`, attendee
  tools now 7, `rebot_rs` unchanged) into the PR body.

---

# PR-B (owner's "B")

### Task B1: Admission on the production path; provenance; v2 notes

**Files:**
- Modify: `src/cascade/agent/aspire.py` (`_read_summary` callers, `diagnose`, `distil`), `src/cascade/skills/library.py`
  (`add`: `learned_in`, `kind`, v2 front matter, `## Advice` / `## Evidence` body)
- Create: `tests/test_aspire_admission_mcp.py`

**Interfaces:**
- Produces: `goal_task(rec: dict) -> str` (`"<skill> <object|label> -> <destination|->"`); `run_profile_class(run_dir:
  Path) -> str` (`run.json` `profile_class`, else `"unknown"`); `SkillLibrary.add(..., learned_in: str = "unknown",
  advice: str = "")`; `ADVICE_TEMPLATES: dict[str, str]` keyed by `refusal_class`.

- [ ] **Step 1: Write the failing tests**

```python
def test_mcp_run_without_summary_uses_the_goal_as_task(tmp_path):
    run = _mcp_run(tmp_path, [_refused_pick("green cube", "green square"), _confirmed_pick("green cube", "green square")])
    assert diagnose(run).task == "pick_and_place green cube -> green square"
def test_two_goals_make_two_distinct_tasks(tmp_path): ...   # n_tasks == 2 after harvesting runs with two goals
def test_learned_in_comes_from_run_json(tmp_path): ...       # sim, real, missing -> unknown; sim+real -> mixed
def test_distil_writes_lint_clean_advice_and_keeps_evidence_out_of_it(tmp_path):
    n = parse_note(path); assert lint_advice(n.advice) == [] and "Recorded confirmation" not in n.advice
def test_note_whose_advice_fails_lint_is_not_written(tmp_path): ...
```

- [ ] **Step 2: Run** -> FAIL. **Step 3:** implement (advice = `ADVICE_TEMPLATES[refusal_class(failed result)]`
  formatted with skill and object; `learned_in` merge rule per spec §5.1). **Step 4:** run the file +
  `tests/test_aspire_admission.py tests/test_aspire_promotion.py` -> PASS.
- [ ] **Mutation:** return `""` from `goal_task` -> KILLED.
- [ ] **Commit:** `feat(aspire): admit retry pairs from MCP runs by goal; v2 notes with provenance and lint-clean advice`

### Task B2: Promotion needs both rules

**Files:**
- Modify: `src/cascade/skills/library.py` (`LibraryEntry.promoted/status`, new `set_verdict`), `src/cascade/skills/notes.py`
  (served for `sim` = promoted by the new rule), `src/cascade/agent/aspire.py` (`retrieve` reads `load_snapshot`),
  `tests/test_aspire_promotion.py`, `tests/test_agentic_upgrades.py`, `tests/test_memory_default_path.py`,
  `docs/DREAM_RSI_ADAPTATION.md` ("Cross-task promotion gate" section), `docs/ARCHITECTURE.md:333`, `docs/ROADMAP.md` (#10)

**Interfaces:**
- Produces: `SkillLibrary.set_verdict(name: str, verifier: dict) -> Path`; `LibraryEntry.verifier: dict | None`;
  `promoted` iff `n_tasks >= min_tasks and verifier.verdict == "verified" and lint clean and not demoted`;
  `notes.load_snapshot` (sim) recomputes the same rule from the file (the stored `status` stays advisory, so v1 files
  written as `promoted` by the old rule stop being served).

- [ ] **Step 1: Failing tests:** `test_two_tasks_without_verdict_stay_candidate`, `test_v1_file_marked_promoted_is_not_served_without_a_verdict`,
  `test_verified_two_task_note_is_promoted_and_served`,
  `test_refuted_note_is_never_served_and_keeps_gaps`, `test_min_tasks_1_relaxes_only_the_task_count`,
  `test_orchestrator_retrieve_uses_the_same_snapshot` (real class + unverified note -> `retrieve()` returns "").
- [ ] **Step 2-4:** run -> FAIL; implement; run the files above + `tests/test_llm_and_library.py` -> PASS.
- [ ] **Mutation:** `and verdict == "verified"` -> `or verdict == "verified"` -> KILLED.
- [ ] **Commit:** `feat(aspire): promotion needs two distinct tasks AND a verified blind run`

### Task B3: Use grading and demotion

**Files:**
- Create: `src/cascade/agent/aspire_grade.py`, `tests/test_aspire_grade.py`
- Modify: `src/cascade/skills/library.py` (`add_grades`), `scripts/learn_from_runs.py` (`--grade` step)

**Interfaces:**
- Produces: `orders(rows: list[dict]) -> list[dict]` (segments by goal: `goal`, `confirmed: bool`, `calls`);
  `grade_run(run_dir: Path, snapshot: LibrarySnapshot) -> dict[str, list[str]]` (note id -> grades);
  `apply_grades(library: SkillLibrary, grades: dict) -> list[str]` (demoted ids); `DEMOTE_MISLEADING = 3`.
- **Cross-task note (A4 fix round 1, frozen library):** on the Spark kitchen launch `run.json`'s `library_root` is
  the launch's frozen copy (`$STATE_DIR/aspire-library`), not the canonical library. Address a note by `library_sha` +
  note id in the canonical library (`~/.cascade/aspire/sim/library`, the nightly job's own config) and never write
  into a run's `library_root` (or any `notes.library_root`): the copy is replaced whole at the next launch.

- [ ] **Step 1: Failing tests:** `test_not_applicable_when_when_words_miss_the_goal`, `test_useful_on_confirmed_order`,
  `test_misleading_on_failed_order`, `test_demotion_after_three_misleading_outweighing_useful`,
  `test_only_block_notes_from_run_json_are_graded`, `test_grading_is_idempotent_per_run` (same run twice -> counted once).
- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Mutation:** `DEMOTE_MISLEADING = 3` -> `30` -> KILLED.
- [ ] **Commit:** `feat(aspire): grade served notes from run outcomes; demote misleading ones`

### Task B4: Timing breakdown

**Files:**
- Modify: `src/cascade/skills/runtime.py` (`execute`: scratchpad `_call_timing`, around `effects.verify`;
  `skill_grasp_object`: around the planner call)
- Create: `src/cascade/agent/timing_report.py`, `tests/test_timing_breakdown.py`

**Interfaces:**
- Produces: trace row `timing: {perception_s, planning_s, motion_other_s, postcondition_s, total_s}` on `_MOTION_SKILLS`
  rows; `runtime._note_call_timing(key: str, seconds: float) -> None`; `order_timing(rows: list[dict], turn_s: float,
  brain_calls: list[dict] | None) -> dict` (adds `agent_s`, `ttft_s`, `decode_s`); `timing_table(orders: list[dict]) -> str`.

- [ ] **Step 1: Failing tests:** `test_motion_rows_carry_buckets_that_sum_to_total` (mock pick; `abs(sum - total_s) <= 0.01`),
  `test_postcondition_time_is_outside_duration_ms`, `test_non_motion_rows_have_no_timing`,
  `test_order_timing_adds_agent_time_from_the_turn` (fake rows + 155 s turn -> `agent_s == 155 - tools`).
- [ ] **Step 2-4:** FAIL -> implement -> PASS (+ `tests/test_agent_results_compose.py`).
- [ ] **Mutation:** drop `postcondition_s` from `total_s` -> KILLED.
- [ ] **Commit:** `feat(trace): per-call time buckets and a per-order timing table`

### Task B5: The blind verifier harness

**Files:**
- Create: `benchmark/aspire/verify_note.py`, `benchmark/aspire/controls/{positive-green-cube,negative-open-box}.md`,
  `tests/test_aspire_verifier.py`

**Interfaces:**
- Consumes: `load_snapshot`, `LibrarySnapshot.block`, `sync_learned_notes` (A1, A4); `SkillLibrary.set_verdict` (B2);
  `order_timing` (B4); bridge `set_reset_seed` (A5).
- Produces: `class OpenClawCliDriver(cli: Path, profile: str, timeout_s: int)` with `turn(session_id: str, message: str)
  -> dict` (the CLI's JSON envelope); `episode(seed: int, order: str, *, driver, bridge, truth, trace) -> dict`
  (`{"seed", "passed": bool | None, "motion_calls", "void_reason", "timing"}`); `verdict(episodes: list[dict]) -> str`
  (`verified` >= 4 passed; `void` > 1 void; else `refuted`); `verify(note_id: str, *, library: Path, workspace: Path,
  seeds=(101, 102, 103, 104, 105), driver, bridge, truth, out: Path) -> dict`. The session's snapshot is
  `dataclasses.replace(base, served=base.served + (candidate,))` with `base = load_snapshot(library, "sim")`, written
  with `sync_learned_notes`.

- [ ] **Step 0 (before any code): isolation preflight.** Dry-run `launch.sh` with the verifier env (Global Constraints
  ports and profile) and list every port, unit, OpenClaw profile and state path it touches. A fixed live port or the
  `cascade-demo` profile anywhere -> end the task STOPPED_FOR_DESIGN with the list. Confirm `openclaw agent --help` of
  the pinned CLI accepts `--session-id --message --json --timeout`; else STOPPED_FOR_DESIGN.
- [ ] **Step 1: Failing tests** (fake driver/bridge/truth recording calls):

```python
def test_verdicts(): assert verdict([P]*4 + [F]) == "verified" and verdict([P]*3 + [F]*2) == "refuted" and verdict([P]*3 + [V]*2) == "void"
def test_blind_workspace_shows_promoted_plus_candidate_only(tmp_path):
    verify("cand", ...); text = (ws / "LEARNED.md").read_text()
    assert "cand advice" in text and "Recorded confirmation" not in text and SOURCE_TASK not in text and "other-candidate" not in text
def test_episode_order_seed_reset_then_new_session(fake):
    assert fake.calls[:3] == [("bridge", {"op": "set_reset_seed", "seed": 101}), ("turn", HOUSEKEEPING, "Let's start over."), ("turn", NEW, ORDER)]
def test_reset_without_matching_scene_seed_is_void(): ...
def test_more_than_two_motion_calls_fail_the_seed(): ...
def test_verdict_is_written_with_library_sha_engine_and_seeds(tmp_path): ...
def test_controls_are_never_written_to_a_library(tmp_path): ...    # --control path leaves the library sha unchanged
```

- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Mutation:** `>= 4` -> `>= 3` in `verdict` -> KILLED.
- [ ] **Commit:** `feat(aspire): blind Newton verifier: 5 seeded episodes on the kitchen route, verdict into the note`

### Task B6: Nightly runner with the live-demo guard

**Files:**
- Create: `scripts/aspire_nightly.py`, `deploy/aspire/cascade-aspire-nightly.service`, `deploy/aspire/cascade-aspire-nightly.timer`,
  `tests/test_aspire_nightly.py`, `tests/data/aspire_nightly/{proc_net_tcp_live,proc_net_tcp_idle,proc_net_tcp6_live,systemctl_live.txt,systemctl_idle.txt}`
- Modify: `docs/DREAM_RSI_ADAPTATION.md` (install + guard section)

**Interfaces:**
- Consumes: `harvest` (B1), `apply_grades` (B3), `verify` (B5).
- Produces: `GUARDED_PORTS` (Global Constraints list); `listening_ports(paths=(Path("/proc/net/tcp"),
  Path("/proc/net/tcp6"))) -> set[int]` (state `0A` only); `guard(now: datetime, *, ports: set[int], units: str,
  locks: list[Path], mem_available_gb: float, gpu_pids: list[int], extra_ports=()) -> tuple[bool, list[str]]`;
  `main(argv) -> int` (`--dry-run`, `--guard-port N` repeatable, `--max-candidates`).
- **Cross-task note (A4 fix round 1, frozen library):** on the Spark kitchen launch `run.json`'s `library_root` is
  the launch's frozen copy (`$STATE_DIR/aspire-library`), not the canonical library. Address a note by `library_sha` +
  note id in the canonical library (`~/.cascade/aspire/sim/library`, the nightly job's own config) and never write
  into a run's `library_root` (or any `notes.library_root`): the copy is replaced whole at the next launch.

- [ ] **Step 1: Failing tests:** captured real outputs as fixtures (`/proc/net/tcp` lines of a host with :18741 and
  :8611 listening; `systemctl --user list-units --no-legend` lines with a `cascade-openshell-*` unit), then
  `test_each_guarded_port_listening_skips` (parametrize over `GUARDED_PORTS`), `test_established_only_is_not_listening`,
  `test_ipv6_listener_counts`, `test_each_unit_pattern_skips`, `test_locks_skip`, `test_low_memory_skips`,
  `test_foreign_gpu_process_skips`, `test_outside_window_skips` (05:01 no new candidate; 05:30 teardown),
  `test_unreadable_proc_file_skips`, `test_mid_run_guard_failure_voids_the_episode_and_tears_down` (fake clock + fake
  stack: teardown called, remaining candidates untouched), `test_timer_runs_at_0130`.
- [ ] **Step 2-4:** FAIL -> implement (passive reads only; no `socket.connect` anywhere: a test greps the module's
  AST for `connect`/`create_connection`) -> PASS.
- [ ] **Mutation:** treat state `01` (ESTABLISHED) as listening -> KILLED; drop the `extra_ports` merge -> KILLED.
- [ ] **Commit:** `feat(aspire): nightly harvest, grade and verify; fail-closed guard keeps it away from the live demo`

### Task B7: Verified-only export to the real reBot

**Files:**
- Create: `scripts/aspire_export.py`, `tests/test_aspire_export.py`
- Modify: `configs/arms/rebot_rs.yaml` (`overrides.memory.skill_library_path`, `overrides.agent.learning_surface`),
  `docs/DREAM_RSI_ADAPTATION.md`

**Interfaces:**
- Produces: `export_verified(src: Path, dst: Path) -> dict` (`{"exported": [...], "removed": [...], "library_sha": ...}`).

- [ ] **Step 1: Failing tests:** `test_exports_only_verified_lint_clean_notes`, `test_removes_previously_exported_notes_no_longer_verified`,
  `test_never_touches_other_files_in_the_target`, `test_rebot_rs_serves_only_verified_with_the_sim_tag` (real-profile
  runtime: `robot_knowledge` contains `learned in sim | verifier 4/5 | lib`), `test_unverified_note_dropped_into_the_export_is_not_served`.
- [ ] **Step 2-4:** FAIL -> implement (atomic per file, `exported` front-matter key) -> PASS.
- [ ] **Mutation:** export `status == "promoted"` instead of `verdict == "verified"` -> KILLED.
- [ ] **Commit:** `feat(aspire): verified-only export to the real reBot library`

### Task B8: PR-B acceptance run

- [ ] Brief from the template (as A11, verifier ports/profile). Run the two controls on seeds 101..105, the night dry
  run (`aspire_nightly.py --dry-run --max-candidates 1 --guard-port 18999`, then a real night with the demo down), and,
  once >= 1 real note is verified, the held-out A/B of spec §8 PR-B item 7. RESULTS with every spec §8 PR-B number;
  the A/B verdict decides C (report it to the owner in one line).

---

## Self-review: spec vs plan coverage

| Spec item | Task(s) |
|---|---|
| §4.1 note v2, tags, lint | A1 |
| §4.2 hash, snapshot, run.json, row stamp, library paths | A1, A2 (kitchen path), B7 (rebot path) |
| §4.3 LEARNED.md + plugin, budget, robot_knowledge, 7th tool, pinned lists | A3, A4 |
| §4.4 profile class, served filter | A1 (+ A2 build, B2 retrieve) |
| §4.5 retry: enable rule, goal, refusal class, ledger, texts, interplay, bound | A6, A7 |
| §4.6 R1-R5 | A8 |
| §4.7 seeded reset | A5 |
| §4.8 chat line, learning.json, panel, no new port | A9, A10 |
| §4.9 docs fix | A0 |
| §5.1 admission + provenance | B1 |
| §5.2 promotion both | B2 |
| §5.3 verifier | B5 (+ B8 run) |
| §5.4 grading, demotion | B3 |
| §5.5 timing | B4 |
| §5.6 nightly + guard | B6 |
| §5.7 export | B7 |
| §6.1 notes are text | A2 (`test_hostile_library...`, import scan) |
| §6.2 no margin lowered | A7 (`test_gates_are_identical...`) |
| §6.3 real robot unaffected | A1, A3, A7, A9 real-profile tests |
| §6.4 halt ends any retry | A6, A7 |
| §6.5 never beside the live demo, real stores | B6, B7 |
| §8 acceptance | A11, B8 |

Checks run on this plan: every spec requirement has a task; names match across tasks (`load_snapshot`,
`LibrarySnapshot.block/knowledge/matching`, `SimRetryLedger.precheck/begin/finish/note_motion/clear`,
`sync_learned_notes`, `LearningState.write`, `set_verdict`); the five Review Focus inputs each have a named test in
the owning task; no step leaves a value open that the spec fixes.
