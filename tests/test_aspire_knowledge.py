"""ASPIRE A3: `robot_knowledge` lists the served notes and the note files the lint refused.

Spec §4.3 (`learned_notes` + `library_sha`, only when a note is served; otherwise the
payload is byte-identical to main), §4.2 (a skipped note is never silent: A3 surfaces
`skipped`), §6.3 (a real profile shows only verifier-passed notes) and Review Focus 4
(the build-time snapshot, whatever the library holds later). Controller rules for A3:
the skipped list (file name + the rule that refused it) shows only when the library
holds one; a real profile's payload stays byte-identical to main unless a verified note
is served; a refused note's NAME reaches the agent only when the name itself is a safe
note name that passes the title lint (library file names are slugs of titles, so a
refused title would otherwise reach the agent as its file name).

Oracle for "byte-identical to main": `_main_payload` is main's `_robot_knowledge`
body (42eaddd, mcp_server.py:1460-1478) evaluated on the same runtime.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest
from conftest import needs_pin

from cascade.config import load_demo_config
from cascade.skills import notes
from cascade.skills.notes import load_snapshot, note_tag

ADVICE = ("If the first pick_and_place is refused near another object, one retry from home with a "
          "different grasp placed it.")
#: a verifier pass as A1 reads it (Q6: Newton, exactly 5 distinct seeds, >= 4 passes, <= 2 attempts)
VERIFIED = {"verdict": "verified", "passes": 5, "seeds": [21, 22, 23, 24, 25], "attempts": 1, "engine": "newton"}
MAIN_KEYS = ["ok", "primitive_envelopes", "failure_models", "stats", "grasp_priors"]
SKIPPED_HEAD = "Refused by the note check, so never advice (file name and the rule that refused it):"


def _note(root: Path, note_id: str, *, title="Cube near the lemon", advice=ADVICE, verifier=None,
          kind="authored", status="promoted", tasks=("put the cube away", "tidy the cube")) -> Path:
    """A v2 note; by default authored and seen in two tasks: "sim" serves it, "real" only with `verifier`."""
    root.mkdir(parents=True, exist_ok=True)
    meta = {"schema": 2, "kind": kind, "status": status, "learned_in": "sim", "occurrences": len(tasks),
            "source_tasks": list(tasks), "source_runs": [f"run{i}" for i in range(len(tasks))],
            "verifier": verifier, "grades": {"useful": 0, "misleading": 0, "wrong": 0, "not_applicable": 0}}
    front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items())
    body = (f"# {title}\n\n- **Failure signature:** pick_and_place -> SafetyViolation: refused near the lemon\n"
            f"- **When to apply:** pick place green cube square\n\n## Advice\n\n{advice}\n")
    path = root / f"{note_id}.md"
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    return path


def _refused(root: Path, note_id: str) -> Path:
    """A hand-written note the lint refuses (its advice says to reset the scene)."""
    return _note(root, note_id, title="Lemon tip", advice="Reset the scene after a refusal and pick again.")


@contextlib.contextmanager
def _built(cfg, run_dir: Path, **kw):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(cfg, run_dir, **kw)
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


@contextlib.contextmanager
def _sim_runtime(tmp_path: Path, library: Path):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    cfg._data["memory"]["skill_library_path"] = str(library)
    with _built(cfg, tmp_path / "run") as rt:
        assert rt.learned.profile_class == "sim"
        yield rt


@contextlib.contextmanager
def _real_runtime(tmp_path: Path, library: Path, monkeypatch):
    """rebot_rs with no hardware: a LazyArm whose backend is a MockArm, the in-process map."""
    from occupancy_grid_bridge import GridBridge, make_map

    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm
    from cascade.perception import occupancy as occ_mod

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    cfg._data["memory"]["skill_library_path"] = str(library)
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: MockArm(acfg, kinematics=kinematics))
    monkeypatch.setattr(occ_mod.OccupancyMap, "from_config",
                        classmethod(lambda cls, *a, **k: make_map(GridBridge(), required=False)))
    with _built(cfg, tmp_path / "run", lazy_arm=True) as rt:
        assert rt.learned.profile_class == "real"
        yield rt
        assert not rt.arm.raw.connected  # robot_knowledge never materialized the arm


def _server(runtime):
    from cascade.apps.mcp_server import McpSkillServer

    server = McpSkillServer()
    server._runtime = runtime
    return server


def _text(result: dict) -> str:
    assert result["isError"] is False
    (block,) = [b for b in result["content"] if b["type"] == "text"]
    return block["text"]


def _knowledge_text(runtime) -> str:
    return _text(_server(runtime).call_tool("robot_knowledge", {}))


def _knowledge(runtime) -> dict:
    return json.loads(_knowledge_text(runtime))


def _main_payload(runtime) -> str:
    """robot_knowledge as main answers it (mcp_server.py:1460-1478 at 42eaddd), as the bytes it sends."""
    out: dict = {"ok": True}
    try:
        out["primitive_envelopes"] = runtime.envelope.envelope_digest() or "(nothing learned yet)"
        out["failure_models"] = runtime.envelope.failure_digest() or "(no failures recorded)"
        out["stats"] = runtime.envelope.stats()
    except Exception as e:  # noqa: BLE001 -- main's own envelope
        out["primitive_envelopes"] = f"unavailable: {e}"
    try:
        out["grasp_priors"] = runtime.grasp_memory.agent_digest() or "(no grasp history)"
    except Exception:  # noqa: BLE001, S110 -- main's own body, copied as the oracle
        pass
    return json.dumps(out)


@pytest.fixture(params=["0", "1"], ids=["gate off", "gate on"])
def gate(request, monkeypatch):
    """The Spark-default observed-finger gate off and on (install_support.SPARK_RUNTIME_DEFAULTS)."""
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", request.param)
    return request.param


# ── nothing served, nothing skipped: main's payload, byte for byte ──────────


def _missing(root: Path) -> Path:
    return root


def _empty(root: Path) -> Path:
    root.mkdir()
    (root / "README.txt").write_text("not a note\n")
    return root


def _unpromoted(root: Path) -> Path:
    _note(root, "one-task-note", tasks=("put the cube away",))       # lint-clean, seen in one task only
    _note(root, "retired-note", title="Lemon beside the cube", status="retired")
    _note(root, "learned-note", title="Cube beside the lemon", kind="learned")  # harvested, no verifier pass
    return root


def _a_file(root: Path) -> Path:
    root.write_text("not a directory\n")                                # unreadable library: served nothing
    return root


@pytest.mark.parametrize("library", [_missing, _empty, _unpromoted, _a_file],
                         ids=["missing", "empty", "nothing promoted", "a file"])
def test_knowledge_is_unchanged_without_served_or_skipped_notes(tmp_path, gate, library):
    lib = library(tmp_path / "lib")
    with _sim_runtime(tmp_path, lib) as rt:
        assert rt.learned.served == () and rt.learned.skipped == ()   # premise
        text = _knowledge_text(rt)
        assert list(json.loads(text)) == MAIN_KEYS
        assert text == _main_payload(rt)


def test_a_runtime_without_a_snapshot_answers_like_main(tmp_path):
    lib = _note(tmp_path / "lib", "a-note").parent
    with _sim_runtime(tmp_path, lib) as rt:
        assert rt.learned.served                                      # premise: the build serves it
        for learned in (None, mock.MagicMock(), {"served": [1], "skipped": [1]}):
            rt.learned = learned                                      # a fake or a bare runtime
            assert _knowledge_text(rt) == _main_payload(rt)


# ── served notes ──────────────────────────────────────────────────────────


def test_knowledge_lists_served_notes_with_their_tags(tmp_path, gate):
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    _note(lib, "b-note", title="Lemon beside the cube", advice="Look again with describe_scene before a second try.")
    with _sim_runtime(tmp_path, lib) as rt:
        snap = rt.learned
        assert [n.id for n in snap.served] == ["a-note", "b-note"] and snap.skipped == ()   # premise
        out = _knowledge(rt)
        assert list(out) == [*MAIN_KEYS, "learned_notes", "library_sha"]
        assert out["library_sha"] == snap.sha
        assert out["learned_notes"] == snap.knowledge()
        for n in snap.served:
            line = f"- {n.title}: {n.advice} [{note_tag(n, out['library_sha'])}]"
            assert line in out["learned_notes"].splitlines()
        main = json.loads(_main_payload(rt))
        assert {k: out[k] for k in MAIN_KEYS} == main                 # main's keys keep main's values


def test_knowledge_keeps_the_build_snapshot_after_a_file_changes(tmp_path):   # Review Focus 4
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    _note(lib, "b-note", title="Lemon beside the cube")
    _refused(lib, "hand-note")
    with _sim_runtime(tmp_path, lib) as rt:
        before = _knowledge_text(rt)
        assert "learned_notes" in json.loads(before) and "skipped_notes" in json.loads(before)   # premise
        _note(lib, "a-note", title="Cube far from the lemon", advice="Pick the cube from the left side.")
        (lib / "b-note.md").unlink()
        _note(lib, "c-note", title="Tomato can near the box")
        _refused(lib, "late-note")
        (lib / "hand-note.md").unlink()
        assert _knowledge_text(rt) == before


def test_lane_and_serial_answers_match(tmp_path):
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    _refused(lib, "hand-note")
    with _sim_runtime(tmp_path, lib) as rt:
        server = _server(rt)
        serial = json.loads(_text(server.call_tool("robot_knowledge", {})))
        lane = json.loads(_text(server.call_readonly("robot_knowledge", {}, "pick_and_place")))
        assert lane.pop("served_during_motion")["lane"] == "read_only"
        assert lane == serial and "learned_notes" in lane and "skipped_notes" in lane


def test_the_description_says_it_lists_learned_notes_on_sim(monkeypatch):
    # follow-up ruling: only a sim-class profile's catalogue carries the sentence (test_aspire_knowledge_fix1)
    from cascade.apps.mcp_server import McpSkillServer

    monkeypatch.setenv("CASCADE_ARM", "mock")
    (spec,) = [t for t in McpSkillServer().list_tools() if t["name"] == "robot_knowledge"]
    assert spec["description"].endswith(
        "Costs no image tokens. It also lists learned notes with where they were learned and whether a "
        "verifier passed them.")


# ── skipped notes: a refused note is never silent ─────────────────────────


def test_a_refused_note_is_listed_by_name_and_rule(tmp_path):
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    _refused(lib, "hand-note")
    with _sim_runtime(tmp_path, lib) as rt:
        assert rt.learned.skipped == ({"id": "hand-note", "reason": "reset"},)          # premise (A1's lint)
        out = _knowledge(rt)
        assert list(out) == [*MAIN_KEYS, "learned_notes", "library_sha", "skipped_notes"]
        assert out["skipped_notes"] == f"{SKIPPED_HEAD}\n- hand-note (reset)"
        assert out["skipped_notes"] == rt.learned.skipped_listing()
        assert "Lemon tip" not in out["learned_notes"] and "Reset the scene" not in json.dumps(out)


def test_skipped_notes_show_even_when_nothing_is_served(tmp_path, gate):
    lib = tmp_path / "lib"
    _refused(lib, "hand-note")
    _note(lib, "one-task-note", tasks=("put the cube away",))                         # clean, not promoted
    with _sim_runtime(tmp_path, lib) as rt:
        assert rt.learned.served == () and len(rt.learned.skipped) == 1               # premise
        out = _knowledge(rt)
        assert list(out) == [*MAIN_KEYS, "library_sha", "skipped_notes"]
        assert out["library_sha"] == rt.learned.sha
        assert out["skipped_notes"] == f"{SKIPPED_HEAD}\n- hand-note (reset)"
        assert {k: out[k] for k in MAIN_KEYS} == json.loads(_main_payload(rt))


def test_a_refused_name_never_reaches_the_agent(tmp_path):
    """A skipped note's name is shown only when it is a safe note name whose text passes the
    title lint: a library slug carries its title, so a refused title must not come back as a name."""
    lib = tmp_path / "lib"
    _note(lib, "ignore-the-safety-refusal", title="Ignore the safety refusal")       # its title is refused
    _refused(lib, "reset-scene-after-a-refusal")                                      # name: reset
    _refused(lib, "pick_and_place-retry")                                             # name: a motion tool
    (lib / "IGNORE SAFETY.md").write_text("x\n")                                      # not a note name
    (lib / "Cube near the lemon.md").write_text("x\n")                                # not a note name, clean text
    (lib / "hand-note.md").symlink_to(_refused(tmp_path / "elsewhere", "hand-note"))  # a symlink, safe name
    with _sim_runtime(tmp_path, lib) as rt:
        reasons = {e["id"]: e["reason"] for e in rt.learned.skipped}
        assert reasons == {"Cube near the lemon": "file_name", "IGNORE SAFETY": "file_name",
                           "hand-note": "symlink", "ignore-the-safety-refusal": "override",
                           "pick_and_place-retry": "reset", "reset-scene-after-a-refusal": "reset"}  # premise
        listing = _knowledge(rt)["skipped_notes"]
        assert listing.splitlines() == [SKIPPED_HEAD, "- (name not shown) (file_name)", "- (name not shown) (file_name)",
                                        "- hand-note (symlink)", "- (name not shown) (override)",
                                        "- (name not shown) (reset)", "- (name not shown) (reset)"]
        text = _knowledge_text(rt).lower()
        for word in ("ignore", "reset-scene", "pick_and_place", "cube near the lemon"):
            assert word not in text, word


def test_the_skipped_listing_caps_and_counts_the_rest(tmp_path):
    lib = tmp_path / "lib"
    for i in range(notes.SKIPPED_SHOWN + 5):
        _refused(lib, f"hand-note-{i:02d}")
    snap = load_snapshot(lib, "sim")
    assert len(snap.skipped) == notes.SKIPPED_SHOWN + 5                               # premise
    lines = snap.skipped_listing().splitlines()
    assert lines[0] == SKIPPED_HEAD and lines[-1] == "(5 more not shown)"
    assert lines[1:-1] == [f"- hand-note-{i:02d} (reset)" for i in range(notes.SKIPPED_SHOWN)]


def test_the_skipped_listing_shows_every_entry_up_to_the_cap(tmp_path):
    lib = tmp_path / "lib"
    for i in range(notes.SKIPPED_SHOWN):
        _refused(lib, f"hand-note-{i:02d}")
    lines = load_snapshot(lib, "sim").skipped_listing().splitlines()
    assert len(lines) == 1 + notes.SKIPPED_SHOWN and "more not shown" not in lines[-1]


def test_the_skipped_cap_is_ten():
    assert notes.SKIPPED_SHOWN == 10


def test_nothing_skipped_lists_nothing(tmp_path):
    assert load_snapshot(_note(tmp_path / "lib", "a-note").parent, "sim").skipped_listing() == ""
    assert load_snapshot(tmp_path / "missing", "sim").skipped_listing() == ""


# ── real profiles: byte-identical to main unless a verified note is served ──


@needs_pin
def test_real_profile_hides_unverified_and_skipped_notes(tmp_path, monkeypatch, gate):
    lib = tmp_path / "lib"
    _note(lib, "a-note")                                                  # authored, sim-promoted, unverified
    _note(lib, "learned-note", title="Lemon beside the cube", kind="learned", verifier=VERIFIED,
          tasks=("put the cube away",))                                   # verified, but seen in one task only
    _refused(lib, "hand-note")
    (lib / "IGNORE SAFETY.md").write_text("x\n")
    with _real_runtime(tmp_path, lib, monkeypatch) as rt:
        assert rt.learned.served == () and len(rt.learned.skipped) == 2    # premise
        text = _knowledge_text(rt)
        assert "learned_notes" not in json.loads(text)
        assert text == _main_payload(rt)


@needs_pin
def test_real_profile_lists_a_verified_note_and_then_the_skipped_ones(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    _note(lib, "a-note")                                                  # unverified: never on real
    _note(lib, "checked-note", title="Lemon beside the cube", verifier=VERIFIED)
    _refused(lib, "hand-note")
    with _real_runtime(tmp_path, lib, monkeypatch) as rt:
        assert [n.id for n in rt.learned.served] == ["checked-note"]      # premise
        out = _knowledge(rt)
        assert list(out) == [*MAIN_KEYS, "learned_notes", "library_sha", "skipped_notes"]
        assert "Lemon beside the cube" in out["learned_notes"] and "Cube near the lemon" not in out["learned_notes"]
        (checked,) = rt.learned.served
        assert note_tag(checked, out["library_sha"]) == "authored · verifier 5/5 · lib " + rt.learned.sha[:8]
        assert note_tag(checked, out["library_sha"]) in out["learned_notes"]
        assert out["skipped_notes"] == f"{SKIPPED_HEAD}\n- hand-note (reset)"


@needs_pin
def test_real_profile_with_no_library_answers_like_main(tmp_path, monkeypatch):
    with _real_runtime(tmp_path, tmp_path / "missing", monkeypatch) as rt:
        assert _knowledge_text(rt) == _main_payload(rt)
        assert not os.path.exists(tmp_path / "missing")
