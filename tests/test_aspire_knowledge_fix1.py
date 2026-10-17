"""ASPIRE A3 follow-up: the controller's rulings on the A3 concerns.

4. A real-robot profile keeps main's `robot_knowledge` description byte for byte: its tool
   catalogue must not change until B7 serves verified notes there. The sentence about
   learned notes is appended only on a sim-class profile (`sim_profile.profile_class`).
5. `skipped_notes` names only v2 notes (front matter `schema: 2`, `kind: authored` or a
   `verifier` record) and the entries the snapshot could not parse at all; a harvested v1
   or legacy note is counted only ("(N older notes not served)"), so a sim profile that
   reads `<repo>/skills_library` gets one line, not a list of harvested files.
6. At most 10 entries are listed (served and skipped each); the rest are counted.

A3 review fix round (task-A3-fix1.md ruling 3): `skipped_notes` names no note at all, only
counts per rule ("3 notes not served: stop x2, success_claim x1"); older notes keep their own
count line, and one note reads singular. The rulings above hold for what is still listed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import needs_pin
from test_aspire_knowledge import (
    MAIN_KEYS,
    VERIFIED,
    _built,
    _knowledge,
    _knowledge_text,
    _main_payload,
    _note,
    _real_runtime,
    _refused,
    _server,
    _sim_runtime,
)

from cascade.config import load_demo_config
from cascade.skills import notes
from cascade.skills.library import SkillLibrary
from cascade.skills.notes import load_snapshot

#: robot_knowledge's description on main (42eaddd and af5ffff, mcp_server.py:221-228)
MAIN_DESCRIPTION = (
    "What this robot has LEARNED from past runs: the proven operating "
    "range of each primitive (where grasps/places actually succeed on "
    "this arm), how each one usually fails, and the grasp strategy "
    "priors per object profile. Read this before planning a tricky "
    "manipulation -- it is the difference between guessing the arm's "
    "envelope and knowing it. Costs no image tokens."
)
SENTENCE = " It also lists learned notes with where they were learned and whether a verifier passed them."


@pytest.fixture(params=["0", "1"], ids=["gate off", "gate on"])
def gate(request, monkeypatch):
    """The Spark-default observed-finger gate off and on (install_support.SPARK_RUNTIME_DEFAULTS)."""
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", request.param)
    return request.param


def _description(server) -> str:
    (spec,) = [t for t in server.list_tools() if t["name"] == "robot_knowledge"]
    return spec["description"]


def _catalog_server(monkeypatch, arm: str):
    """A server that has built nothing yet: list_tools reads the profile from the environment."""
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_ARM", arm)
    monkeypatch.setenv("CASCADE_CAMERA", "mock")
    for name in ("CASCADE_ARMS", "CASCADE_CAMERAS", "CASCADE_PROGRAMS", "CASCADE_HIDE_TOOLS"):
        monkeypatch.delenv(name, raising=False)
    return McpSkillServer()


# ── 4: the tool catalogue of a real profile is main's ───────────────────────


def test_the_shipped_description_is_mains():
    from cascade.apps.mcp_server import _EXTRA_TOOLS

    (spec,) = [t for t in _EXTRA_TOOLS if t["name"] == "robot_knowledge"]
    assert spec["description"] == MAIN_DESCRIPTION


def test_a_sim_profile_catalog_says_it_lists_learned_notes(monkeypatch):
    server = _catalog_server(monkeypatch, "mock")
    assert _description(server) == MAIN_DESCRIPTION + SENTENCE
    assert server._runtime is None                                            # listing built nothing


@pytest.mark.parametrize("arm", ["rebot_rs", "fr3"])
def test_a_real_profile_catalog_keeps_mains_description(monkeypatch, arm):
    server = _catalog_server(monkeypatch, arm)
    catalog = server.list_tools()
    (spec,) = [t for t in catalog if t["name"] == "robot_knowledge"]
    assert spec["description"] == MAIN_DESCRIPTION
    assert "learned notes" not in json.dumps(catalog)
    assert server._runtime is None


def test_a_booth_session_is_real_and_keeps_mains_description(monkeypatch):
    server = _catalog_server(monkeypatch, "mock")
    monkeypatch.setenv("CASCADE_BOOTH", "1")
    assert _description(server) == MAIN_DESCRIPTION


def test_the_sentence_is_the_only_catalog_difference_between_the_classes(monkeypatch):
    from cascade import sim_profile

    sim = _catalog_server(monkeypatch, "mock").list_tools()
    monkeypatch.setattr(sim_profile, "profile_class", lambda cfg: "real")    # the same profile read as real
    real = _catalog_server(monkeypatch, "mock").list_tools()
    differ = [(a["name"], a["description"], b["description"]) for a, b in zip(sim, real) if a != b]
    assert [t["name"] for t in sim] == [t["name"] for t in real]
    assert differ == [("robot_knowledge", MAIN_DESCRIPTION + SENTENCE, MAIN_DESCRIPTION)]


def test_a_built_sim_runtime_lists_the_sentence(tmp_path):
    with _sim_runtime(tmp_path, tmp_path / "lib") as rt:
        assert _description(_server(rt)) == MAIN_DESCRIPTION + SENTENCE


@needs_pin
def test_a_built_real_runtime_keeps_mains_description(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    _note(lib, "checked-note", title="Lemon beside the cube", verifier=VERIFIED)
    with _real_runtime(tmp_path, lib, monkeypatch) as rt:
        assert [n.id for n in rt.learned.served] == ["checked-note"]         # premise: a verified note is served
        assert _description(_server(rt)) == MAIN_DESCRIPTION


# ── 5: only v2 notes are named; older harvested notes are counted ────────────


def _harvested(root: Path, title: str, task: str = "put the cube away") -> Path:
    """A v1 note exactly as learn_from_runs writes it (SkillLibrary.add): the harvested
    title carries the error class, which the note lint refuses (mixed case)."""
    return SkillLibrary(root).add(title=title, failure="pick_and_place -> SafetyViolation: refused near the lemon",
                                  when="pick place cube", strategy="Retry once from home with another grasp.",
                                  source_task=task, source_run=f"run-{title}")


def _legacy(root: Path, note_id: str) -> Path:
    """A legacy note without front matter (a hand-copied v1 file)."""
    path = root / f"{note_id}.md"
    path.write_text("# Lemon tip\n\n## Strategy\n\nReset the scene after a refusal and pick again.\n")
    return path


def _v2_front(root: Path, note_id: str, front: str) -> Path:
    """A refused note whose front matter is `front` (no other key)."""
    path = root / f"{note_id}.md"
    path.write_text(f"---\n{front}\n---\n# Lemon tip\n\n## Advice\n\nReset the scene after a refusal and pick again.\n")
    return path


def _mixed_library(lib: Path) -> Path:
    for i in range(3):
        _harvested(lib, f"Recover from SafetyViolation number {'abc'[i]}")
    _legacy(lib, "legacy-note")
    _refused(lib, "hand-note")                                                # schema 2, authored
    _note(lib, "harvest-note", title="Lemon tip", advice="Reset the scene after a refusal and pick again.",
          kind="learned", status="candidate")                                 # schema 2, harvested by the v2 path
    _v2_front(lib, "written-note", 'kind: "authored"')                       # authored, no schema key
    _v2_front(lib, "checked-note", "verifier: " + json.dumps(VERIFIED))      # a verifier record, no schema key
    (lib / "IGNORE SAFETY.md").write_text("x\n")                              # never parsed: version unknown
    return lib


def test_v2_notes_are_counted_by_rule_and_older_ones_apart(tmp_path):
    snap = load_snapshot(_mixed_library(tmp_path / "lib"), "sim")
    assert len(snap.skipped) == 9 and snap.served == ()                       # premise: every file refused
    assert snap.skipped_listing().splitlines() == [
        "5 notes not served: reset x4, file_name x1", "(4 older notes not served)"]


def test_a_harvested_v1_library_on_sim_gets_one_line_not_a_list(tmp_path, monkeypatch, gate):
    # the mock profile names no library: it reads <repo>/skills_library (notes.DEFAULT_LIBRARY)
    lib = tmp_path / "skills_library"
    for i in range(25):
        _harvested(lib, f"Recover from SafetyViolation {chr(97 + i)}")
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", lib)
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.memory.get("skill_library_path") is None                      # premise: the default library
    with _built(cfg, tmp_path / "run") as rt:
        assert rt.learned.root == lib and len(rt.learned.skipped) == 25       # premise
        out = _knowledge(rt)
        assert list(out) == [*MAIN_KEYS, "library_sha", "skipped_notes"]
        assert out["skipped_notes"] == "(25 older notes not served)"
        assert "safetyviolation" not in json.dumps(out).lower()
        assert {k: out[k] for k in MAIN_KEYS} == json.loads(_main_payload(rt))


def test_the_older_count_follows_the_listed_ones(tmp_path):
    lib = tmp_path / "lib"
    _refused(lib, "hand-note")
    _harvested(lib, "Recover from SafetyViolation")
    _legacy(lib, "legacy-note")
    assert load_snapshot(lib, "sim").skipped_listing() == "1 note not served: reset x1\n(2 older notes not served)"


def test_a_lint_clean_v1_note_is_neither_listed_nor_counted(tmp_path):
    lib = tmp_path / "lib"
    SkillLibrary(lib).add(title="Cube near the lemon", failure="pick_and_place -> SafetyViolation: refused",
                          when="pick place cube", strategy="Retry once from home with another grasp.",
                          source_task="put the cube away", source_run="r1")
    _harvested(lib, "Recover from SafetyViolation")
    snap = load_snapshot(lib, "sim")
    assert snap.served == () and len(snap.skipped) == 1                       # premise: not served, not skipped
    assert snap.skipped_listing() == "(1 older note not served)"


@needs_pin
def test_a_real_profile_counts_older_notes_after_a_verified_one(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    _note(lib, "checked-note", title="Lemon beside the cube", verifier=VERIFIED)
    _harvested(lib, "Recover from SafetyViolation")
    with _real_runtime(tmp_path, lib, monkeypatch) as rt:
        assert _knowledge(rt)["skipped_notes"] == "(1 older note not served)"


@needs_pin
def test_a_real_profile_with_only_older_notes_answers_like_main(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    _harvested(lib, "Recover from SafetyViolation")
    with _real_runtime(tmp_path, lib, monkeypatch) as rt:
        assert _knowledge_text(rt) == _main_payload(rt)


# ── 6: at most 10 entries listed, served and skipped each ───────────────────


def test_the_listing_cap_is_ten():
    assert notes.KNOWLEDGE_SHOWN == 10


def _served_notes(lib: Path, n: int) -> None:
    for i in range(n):
        _note(lib, f"note-{chr(97 + i)}", title=f"Cube note {chr(97 + i)}", advice="Look again before a second try.")


def test_knowledge_lists_ten_served_notes_and_counts_the_rest(tmp_path):
    lib = tmp_path / "lib"
    _served_notes(lib, 15)
    snap = load_snapshot(lib, "sim")
    assert len(snap.served) == 15                                             # premise
    lines = snap.knowledge().splitlines()
    assert lines[1:-1] == [f"- Cube note {chr(97 + i)} - Look again before a second try. "
                           f"[{notes.note_tag(snap.served[i], snap.sha)}]" for i in range(10)]
    assert lines[-1] == "(5 more notes not shown)"


def test_knowledge_lists_all_ten_without_a_footer(tmp_path):
    lib = tmp_path / "lib"
    _served_notes(lib, 10)
    text = load_snapshot(lib, "sim").knowledge()
    assert text.count("\n- ") == 10 and "more notes not shown" not in text


def test_the_skipped_listing_counts_v2_notes_and_older_ones(tmp_path):
    lib = tmp_path / "lib"
    for i in range(15):
        _refused(lib, f"hand-note-{chr(97 + i)}")
    _harvested(lib, "Recover from SafetyViolation")
    lines = load_snapshot(lib, "sim").skipped_listing().splitlines()
    assert lines == ["15 notes not served: reset x15", "(1 older note not served)"]


def test_the_robot_knowledge_reply_lists_at_most_ten_of_each(tmp_path):
    lib = tmp_path / "lib"
    _served_notes(lib, 12)
    for i in range(12):
        _refused(lib, f"hand-note-{chr(97 + i)}")
    for i in range(30):
        _harvested(lib, f"Recover from SafetyViolation {chr(97 + i % 26)}{i}")
    with _sim_runtime(tmp_path, lib) as rt:
        out = _knowledge(rt)
    assert out["learned_notes"].count("\n- ") == 10 and out["learned_notes"].endswith("(2 more notes not shown)")
    assert out["skipped_notes"] == "12 notes not served: reset x12\n(30 older notes not served)"


def test_ten_of_the_longest_served_lines_fit_the_character_cap(tmp_path):
    # the lint caps title (80) and advice (300); with the longest tag the 10-entry cap binds
    # first, so KNOWLEDGE_MAX_CHARS is only a backstop and the reply stays bounded by the count
    words = "cube near the lemon on the green square by the left side of the table and the bowl " * 5
    title, advice = words[:80].rstrip().ljust(80, "s"), words[:300].rstrip().ljust(300, "s")
    lib = tmp_path / "lib"
    for i in range(11):
        _note(lib, f"note-{chr(97 + i)}", title=title, advice=advice, kind="learned", verifier=VERIFIED)
        path = lib / f"note-{chr(97 + i)}.md"
        path.write_text(path.read_text().replace('learned_in: "sim"', 'learned_in: "unknown"'))
    snap = load_snapshot(lib, "sim")
    assert len(snap.served) == 11                                             # premise: all lint-clean, served
    assert notes.note_tag(snap.served[0], snap.sha).startswith("learned in unknown | verifier 5/5")
    text = snap.knowledge()
    assert text.count("\n- ") == notes.KNOWLEDGE_SHOWN and text.endswith("(1 more note not shown)")
    assert len(json.dumps(text)) - 2 <= notes.KNOWLEDGE_MAX_CHARS
