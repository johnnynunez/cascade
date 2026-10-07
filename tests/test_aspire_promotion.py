"""Cross-task promotion gate for distilled ASPIRE notes (ROADMAP follow-up #10).

Upstream ASPIRE (``aspire/sim/cap/skills/library.py``) tracks ``occurrences``
and ``source_tasks`` per distilled skill and promotes it to the active library
only once it recurs beyond a single task; one lucky repair stays a candidate.
These tests pin the port: promotion only after a second DISTINCT task, the
same task twice does not promote, legacy single-task files are stored but not
retrieved, counters persist and stay idempotent across harvests, and the only
relaxation is an explicit ``min_tasks=1``.

Fixtures are synthetic traces; nothing here is a physical trial or evidence
that a promoted note improves robot performance.
"""
import json
import re

import pytest

from cascade.agent.aspire import diagnose, distil, harvest, retrieve
from cascade.agent.effects import CONFIRMED, POSTCONDITIONS
from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
from cascade.agent.orchestrator import AgentOrchestrator
from cascade.skills.library import ENTRY_TEMPLATE, SkillLibrary

TASK_A = "pick the cube"
TASK_B = "stack the cube on the box"
QUERY = "grasp the cube"  # shares guard words with every note distilled below


def _confirmed_result(skill="grasp_object"):
    return {
        "ok": True,
        "postcondition": {
            "skill": skill, "kind": POSTCONDITIONS[skill], "status": CONFIRMED,
            "channel": "physics", "evidence": "Synthetic fixture: object rose above the table",
            "measured": {"rise_m": 0.03},
        },
    }


def _write_run(root, name, task, *, summary=True):
    """A failed grasp followed by a matching, physics-confirmed retry."""
    context = {"arm": "left", "held_object": None}
    records = [
        {"skill": "grasp_object", "args": {"label": "cube", "material": "rigid"}, "context": context,
         "result": {"ok": False, "error": "link/joint 7 would hit the table"}},
        {"skill": "grasp_object", "args": {"label": "cube", "material": "soft"}, "context": context,
         "result": _confirmed_result()},
    ]
    run = root / name
    run.mkdir(parents=True)
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    if summary:
        (run / "summary.txt").write_text(f"task: {task}\nsuccess: true")
    return run


def _front_matter(path) -> dict:
    text = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, f"no front matter in {path.name}:\n{text[:200]}"
    meta = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = json.loads(value.strip())
    return meta


def _single(library):
    records = library.records()
    assert len(records) == 1, [r.name for r in records]
    return records[0]


def test_one_task_distils_a_stored_candidate_that_is_not_retrieved(tmp_path):
    library = SkillLibrary(tmp_path / "lib")
    path = distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), library)
    assert path is not None and path.exists()
    entry = _single(library)
    assert entry.occurrences == 1
    assert entry.source_tasks == [TASK_A]
    assert entry.source_runs == ["run-a"]
    assert entry.status == "candidate" and not entry.promoted
    meta = _front_matter(path)
    assert meta["occurrences"] == 1 and meta["source_tasks"] == [TASK_A]
    assert meta["source_runs"] == ["run-a"] and meta["status"] == "candidate"
    # Stored and inspectable, but never injected into the agent context.
    assert len(library.entries()) == 1
    assert "material" in library.relevant(QUERY)[0]
    assert retrieve(library, QUERY) == ""


def test_second_distinct_task_promotes_the_entry_and_retrieval_admits_it(tmp_path):
    library = SkillLibrary(tmp_path / "lib")
    path_a = distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), library)
    path_b = distil(diagnose(_write_run(tmp_path, "run-b", TASK_B)), library)
    assert path_a == path_b  # same (skill, signature) -> one entry
    entry = _single(library)
    assert entry.occurrences == 2
    assert entry.source_tasks == [TASK_A, TASK_B]
    assert entry.source_runs == ["run-a", "run-b"]
    assert entry.promoted and entry.status == "promoted"
    assert _front_matter(path_a)["status"] == "promoted"
    injected = retrieve(library, QUERY)
    assert "ASPIRE library" in injected and "material" in injected
    assert "2 distinct tasks" in injected


@pytest.mark.parametrize("second_task", [TASK_A, "Pick the  cube", " pick the cube\t"])
def test_same_task_twice_counts_occurrences_but_does_not_promote(tmp_path, second_task):
    library = SkillLibrary(tmp_path / "lib")
    distil(diagnose(_write_run(tmp_path, "run-1", TASK_A)), library)
    distil(diagnose(_write_run(tmp_path, "run-2", second_task)), library)
    entry = _single(library)
    assert entry.occurrences == 2
    assert entry.n_tasks == 1 and entry.source_tasks == [TASK_A]
    assert not entry.promoted
    assert retrieve(library, QUERY) == ""


def test_unknown_task_counts_an_occurrence_but_never_a_distinct_task(tmp_path):
    library = SkillLibrary(tmp_path / "lib")
    distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), library)
    distil(diagnose(_write_run(tmp_path, "run-no-summary", "", summary=False)), library)
    entry = _single(library)
    assert entry.occurrences == 2
    assert entry.source_tasks == [TASK_A] and entry.n_tasks == 1
    assert not entry.promoted
    assert retrieve(library, QUERY) == ""


def test_legacy_entry_without_front_matter_is_a_candidate(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    (root / "grasp-object-geometry-link-below-table.md").write_text(ENTRY_TEMPLATE.format(
        title="grasp_object geometry link_below_table",
        failure="grasp_object -> geometry:link_below_table",
        when="cube grasp object pick",
        origin="run-legacy (task: pick the cube)",
        date="2026-09-01",
        strategy="Recorded retry: `material` 'rigid' -> 'soft'.",
    ))
    library = SkillLibrary(root)
    entry = _single(library)
    assert entry.occurrences == 1
    assert entry.source_tasks == [] and entry.source_runs == []
    assert entry.status == "candidate"
    # Backward compatible: listed and keyword-matched by the store ...
    assert len(library.entries()) == 1
    assert library.relevant(QUERY) and "rigid" in library.relevant(QUERY)[0]
    # ... but not promoted into the agent context on one unverified task.
    assert retrieve(library, QUERY) == ""
    # A second, distinct task recorded later promotes it like any other entry.
    distil(diagnose(_write_run(tmp_path, "run-b", TASK_B)), library)
    entry = _single(library)
    assert entry.occurrences == 2 and entry.source_tasks == [TASK_B]
    assert not entry.promoted  # the legacy task is unknown, so only one task is known
    distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), library)
    assert _single(library).promoted
    assert "ASPIRE library" in retrieve(library, QUERY)


def test_counters_persist_across_harvests_and_reharvest_is_idempotent(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "run-a", TASK_A)
    library = SkillLibrary(tmp_path / "lib")

    first = harvest(runs, library)
    assert first["learned"] == 1 and len(first["entries"]) == 1
    name = first["entries"][0]
    assert first["candidates"] == [name] and first["promoted"] == []
    assert first["library"] == {"entries": 1, "promoted": 0, "candidates": 1, "min_tasks": 2}
    assert _single(library).occurrences == 1

    again = harvest(runs, library)  # same runs: nothing new to learn
    assert again["learned"] == 0 and again["entries"] == []
    assert _single(library).occurrences == 1

    _write_run(runs, "run-b", TASK_B)
    third = harvest(runs, library)
    assert third["learned"] == 1 and third["promoted"] == [name] and third["candidates"] == []
    assert third["library"]["promoted"] == 1

    reopened = SkillLibrary(tmp_path / "lib")  # a new session reads the persisted counters
    entry = _single(reopened)
    assert entry.occurrences == 2 and entry.source_tasks == [TASK_A, TASK_B]
    assert entry.source_runs == ["run-a", "run-b"] and entry.promoted
    assert "ASPIRE library" in retrieve(reopened, QUERY)


def test_harvest_folds_one_signature_across_runs_into_one_counted_entry(tmp_path):
    runs = tmp_path / "runs"
    for i in range(3):
        _write_run(runs, f"run{i}", TASK_A)
    library = SkillLibrary(tmp_path / "lib")
    out = harvest(runs, library)
    assert out["diagnosed"] == 3 and out["learned"] == 1
    entry = _single(library)
    assert entry.occurrences == 3 and entry.n_tasks == 1
    assert entry.source_runs == ["run0", "run1", "run2"]
    assert not entry.promoted


def test_explicit_min_tasks_one_restores_single_observation_retrieval(tmp_path):
    from cascade.skills.library import PROMOTION_MIN_TASKS

    assert PROMOTION_MIN_TASKS == 2
    assert SkillLibrary(tmp_path / "default").min_tasks == PROMOTION_MIN_TASKS
    with pytest.raises(ValueError):
        SkillLibrary(tmp_path / "bad", min_tasks=0)

    relaxed = SkillLibrary(tmp_path / "lib", min_tasks=1)
    distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), relaxed)
    assert _single(relaxed).promoted
    assert "ASPIRE library" in retrieve(relaxed, QUERY)
    # The same files read with the default threshold stay candidates: the
    # relaxation lives in the configured instance, never in the stored note.
    strict = SkillLibrary(tmp_path / "lib")
    assert not _single(strict).promoted
    assert retrieve(strict, QUERY) == ""


class _StubRuntime:
    def __init__(self):
        from cascade.memory import EpisodicMemory

        self.memory = EpisodicMemory()
        self.last_frame = None

        class _Trace:
            def finish(self, summary):
                self.summary = summary

        self.trace = _Trace()

    def execute(self, name, args):
        return {"ok": True, "task_complete": True} if name == "task_done" else {"ok": True}

    def frame_jpeg(self):
        return None


def _intro_after_task(library, task):
    llm = MockLLM([LLMResponse(tool_calls=[ToolCall("task_done", {"success": True, "summary": "ok"})])])
    AgentOrchestrator(llm, _StubRuntime(), decompose=False, max_steps=3,
                      skill_library=library).run_task(task)
    first_user = llm.requests[0]["messages"][0]
    assert first_user["role"] == "user"
    return str(first_user["content"])


def test_orchestrator_context_only_carries_promoted_notes(tmp_path):
    library = SkillLibrary(tmp_path / "lib")
    distil(diagnose(_write_run(tmp_path, "run-a", TASK_A)), library)
    assert "ASPIRE library" not in _intro_after_task(library, QUERY)
    distil(diagnose(_write_run(tmp_path, "run-b", TASK_B)), library)
    intro = _intro_after_task(library, QUERY)
    assert "ASPIRE library" in intro and "material" in intro
