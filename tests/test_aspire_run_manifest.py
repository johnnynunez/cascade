"""ASPIRE A2: one library snapshot per runtime, `run.json` and the trace stamp.

Spec §4.2 and preflight R18 (`purpose`), R23 (`block_notes`; a real-profile run),
R24 (the suite never serves the machine's own library), R29 (class-level
`learned`) and R3 (every new variable reaches the robot server). The snapshot is
read ONCE at build: run.json, every trace row and `runtime.learned` carry that one
view, a library edited later changes none of them, and a library the runtime cannot
read serves nothing and never fails the build (fail closed, reason logged).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from conftest import REPO, needs_pin

from cascade.agent import trace as trace_mod
from cascade.agent.trace import TraceLogger
from cascade.config import PACKAGE_ROOT, load_demo_config
from cascade.skills import notes
from cascade.skills.notes import library_sha, load_snapshot

EMPTY_SHA = hashlib.sha256(b"").hexdigest()
#: run.json keys: spec §4.2's seven, R23's `block_notes`, R18's `purpose` and fix round 1's
#: `library_root` / `library_error` (an unreadable library never looks like an empty one)
RUN_KEYS = {"schema", "library_sha", "served_notes", "block_notes", "profile_class", "arm_types", "sim_retry",
            "started", "purpose", "library_root", "library_error"}
#: every profile that loads configs/arms/isaac_kitchen.yaml (directly or by `extends:`)
KITCHEN = ("isaac_kitchen", "isaac_kitchen_gpu", "isaac_kitchen_cumotion", "isaac_kitchen_hug")
ADVICE = ("If the first pick_and_place is refused near another object, one retry from home with a "
          "different grasp placed it.")


def _note(root: Path, note_id: str, *, title="Cube near the lemon", advice=ADVICE, verifier=None,
          tasks=("put the cube away", "tidy the cube")) -> Path:
    """An authored note seen in two distinct tasks: "sim" serves it, "real" (no verifier pass) never does."""
    root.mkdir(parents=True, exist_ok=True)
    meta = {"schema": 2, "kind": "authored", "status": "promoted", "learned_in": "sim",
            "occurrences": len(tasks), "source_tasks": list(tasks),
            "source_runs": [f"run{i}" for i in range(len(tasks))], "verifier": verifier,
            "grades": {"useful": 0, "misleading": 0, "wrong": 0, "not_applicable": 0}}
    front = "\n".join(f"{k}: {json.dumps(v)}" for k, v in meta.items())
    body = (f"# {title}\n\n- **Failure signature:** pick_and_place -> SafetyViolation: refused near the lemon\n"
            f"- **When to apply:** pick place green cube square\n\n## Advice\n\n{advice}\n")
    path = root / f"{note_id}.md"
    path.write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")
    return path


def _two_served_notes(root: Path) -> Path:
    _note(root, "a-note")
    _note(root, "b-note", title="Lemon beside the cube")
    assert [n.id for n in load_snapshot(root, "sim").served] == ["a-note", "b-note"]   # premise: sim serves both
    assert load_snapshot(root, "real").served == ()                                   # and real serves neither
    return root


def _mock_cfg(library: Path | None = None):
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    if library is not None:
        cfg._data["memory"]["skill_library_path"] = str(library)
    return cfg


@contextlib.contextmanager
def _built(cfg, run_dir: Path, **kw):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(cfg, run_dir, **kw)
    try:
        yield runtime
    finally:
        shutdown_runtime(runtime, arm)


def _meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


# ── run.json at build ─────────────────────────────────────────────────────


def test_build_writes_run_json_with_the_library_identity(tmp_path):
    lib = _two_served_notes(tmp_path / "lib")
    before = datetime.now(timezone.utc).replace(microsecond=0)
    with _built(_mock_cfg(lib), tmp_path / "run") as rt:
        after = datetime.now(timezone.utc)
        meta = _meta(tmp_path / "run")
        assert set(meta) == RUN_KEYS
        assert meta["schema"] == 1
        assert meta["library_sha"] == library_sha(lib) != EMPTY_SHA
        assert (meta["profile_class"], meta["arm_types"], meta["purpose"]) == ("sim", ["mock"], "live")
        assert meta["served_notes"] == [n.id for n in rt.learned.served] == ["a-note", "b-note"]
        assert meta["block_notes"] == rt.learned.manifest()["block_notes"] == ["a-note", "b-note"]
        assert meta["sim_retry"] is False
        started = datetime.fromisoformat(meta["started"])
        assert started.tzinfo is not None and before <= started <= after
        assert rt.learned.root == lib and rt.learned.sha == meta["library_sha"]


def test_block_notes_are_the_two_the_block_renders(tmp_path):
    # R23: B3 grades the notes the prompt block showed and cannot recover them later
    lib = _two_served_notes(tmp_path / "lib")
    _note(lib, "c-note", title="Green cube on the square")
    with _built(_mock_cfg(lib), tmp_path / "run"):
        meta = _meta(tmp_path / "run")
    assert meta["served_notes"] == ["a-note", "b-note", "c-note"] and meta["block_notes"] == ["a-note", "b-note"]


def test_the_sim_retry_flag_is_the_one_the_runtime_computes(tmp_path, monkeypatch):
    from cascade.sim_profile import sim_retry_enabled

    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")      # A7 (preflight R8) also needs the Spark gate
    cfg = _mock_cfg()
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    assert sim_retry_enabled(cfg) is True                         # premise
    with _built(cfg, tmp_path / "run"):
        assert _meta(tmp_path / "run")["sim_retry"] is True


@pytest.mark.parametrize("value, purpose", [(None, "live"), ("", "live"), ("live", "live"), ("verify", "verify"),
                                            (" Acceptance\n", "acceptance"), ("nightly", "unknown"),
                                            ("verify-ish", "unknown")])
def test_the_run_purpose_comes_from_the_environment_and_never_defaults_an_unknown_value_to_live(
        monkeypatch, value, purpose):
    # R18: harvest and grading count only `live` runs, so a verifier or acceptance
    # stack's runs never teach the library; a value nobody defined is not `live`
    from cascade.apps import demo

    if value is None:
        monkeypatch.delenv("CASCADE_RUN_PURPOSE", raising=False)
    else:
        monkeypatch.setenv("CASCADE_RUN_PURPOSE", value)
    assert demo._run_purpose() == purpose


def test_a_verify_stack_run_records_its_purpose(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_RUN_PURPOSE", "verify")
    with _built(_mock_cfg(), tmp_path / "run"):
        assert _meta(tmp_path / "run")["purpose"] == "verify"


# ── the trace stamp ───────────────────────────────────────────────────────


@pytest.mark.parametrize("gate", ["0", "1"], ids=["gate off", "gate on"])
def test_every_trace_row_carries_the_library_sha(tmp_path, monkeypatch, gate):
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", gate)
    lib = _two_served_notes(tmp_path / "lib")
    with _built(_mock_cfg(lib), tmp_path / "run") as rt:
        for skill in ("move_home", "get_observation", "list_objects"):
            rt.execute(skill, {})
        rows = rt.trace.rows()
    assert len(rows) == 3 and {r["library_sha"] for r in rows} == {library_sha(lib)}


def test_rows_and_run_json_keep_the_build_snapshot_after_the_library_changes(tmp_path):
    # Review Focus 4: a nightly harvest beside a running session changes nothing it serves or records
    lib = _two_served_notes(tmp_path / "lib")
    sha0 = library_sha(lib)
    with _built(_mock_cfg(lib), tmp_path / "run") as rt:
        text0 = (tmp_path / "run" / "run.json").read_text()
        _note(lib, "c-note", title="Green cube on the square")
        (lib / "a-note.md").write_text((lib / "a-note.md").read_text().replace("Cube near", "Cubes near"))
        assert library_sha(lib) != sha0                                   # premise: the library did change
        rt.execute("move_home", {})
        assert rt.trace.rows()[-1]["library_sha"] == sha0
        assert (tmp_path / "run" / "run.json").read_text() == text0
        assert rt.learned.sha == sha0 and [n.id for n in rt.learned.served] == ["a-note", "b-note"]


def test_unstamped_logger_rows_say_null(tmp_path):
    t = TraceLogger(tmp_path)
    t.record("x", {}, {"ok": True}, 1.0)
    assert t.rows()[0]["library_sha"] is None and not (tmp_path / "run.json").exists()


def test_stamp_writes_run_json_atomically_then_stamps_later_rows(tmp_path, monkeypatch):
    t = TraceLogger(tmp_path)
    t.record("before", {}, {"ok": True}, 1.0)
    replaced = []
    real_replace = os.replace

    def spy(src, dst):
        replaced.append((Path(src), Path(dst)))
        assert json.loads(Path(src).read_text()) == meta       # complete before it becomes run.json
        return real_replace(src, dst)

    monkeypatch.setattr(trace_mod.os, "replace", spy)
    meta = {"schema": 1, "library_sha": "ab" * 32, "served_notes": ["n"]}
    t.stamp(meta)
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == tmp_path / "run.json" and src.parent == tmp_path and src.name != "run.json"
    assert json.loads((tmp_path / "run.json").read_text()) == meta
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keyframes", "run.json", "trace.jsonl"]
    t.record("after", {}, {"ok": True}, 1.0)
    assert [r["library_sha"] for r in t.rows()] == [None, "ab" * 32]


def test_a_stamp_that_cannot_be_written_keeps_the_old_file_and_stamps_nothing(tmp_path, monkeypatch):
    t = TraceLogger(tmp_path)
    (tmp_path / "run.json").write_text("old\n")

    def refuse(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(trace_mod.os, "replace", refuse)
    with pytest.raises(OSError):
        t.stamp({"library_sha": "cd" * 32})
    assert (tmp_path / "run.json").read_text() == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keyframes", "run.json"]   # no temp file left
    t.record("x", {}, {"ok": True}, 1.0)
    assert t.rows()[0]["library_sha"] is None


def test_a_run_json_the_build_cannot_write_serves_nothing_and_never_fails_the_build(tmp_path, caplog):
    lib = _two_served_notes(tmp_path / "lib")
    (tmp_path / "run" / "run.json").mkdir(parents=True)          # os.replace onto a directory fails
    with caplog.at_level(logging.WARNING), _built(_mock_cfg(lib), tmp_path / "run") as rt:
        assert rt.learned.served == ()
        rt.execute("move_home", {})
        assert rt.trace.rows()[-1]["library_sha"] is None        # no run record, no stamp
    assert "run.json" in caplog.text and "serving no note" in caplog.text


def test_a_mobile_runtime_writes_no_run_json_and_its_rows_say_null(tmp_path):
    from cascade.apps import demo

    rt, _ = demo.build_runtime(load_demo_config(base="microduck_mock"), tmp_path)
    try:
        rt.execute("walk_velocity", {"vx": .05, "vy": 0., "wz": 0., "duration_s": .04})
        assert [r["library_sha"] for r in rt.trace.rows()] == [None]
    finally:
        rt.close()
    assert not (tmp_path / "run.json").exists()


# ── the real profile (R23) ────────────────────────────────────────────────


@needs_pin
def test_a_real_profile_run_records_class_real_and_serves_no_unverified_note(tmp_path, monkeypatch):
    from occupancy_grid_bridge import GridBridge, make_map

    from cascade.apps import demo
    from cascade.control.mock_arm import MockArm
    from cascade.perception import occupancy as occ_mod

    lib = _two_served_notes(tmp_path / "lib")
    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    cfg._data["memory"]["skill_library_path"] = str(lib)
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: MockArm(acfg, kinematics=kinematics))
    monkeypatch.setattr(occ_mod.OccupancyMap, "from_config",
                        classmethod(lambda cls, *a, **k: make_map(GridBridge(), required=False)))
    with _built(cfg, tmp_path / "run", lazy_arm=True) as rt:
        meta = _meta(tmp_path / "run")
        assert (meta["profile_class"], meta["arm_types"], meta["sim_retry"]) == ("real", ["rebot_rs"], False)
        assert meta["served_notes"] == [] and meta["block_notes"] == [] and rt.learned.served == ()
        assert meta["library_sha"] == library_sha(lib) and rt.learned.profile_class == "real"
        rt.execute("list_objects", {})
        assert [r["library_sha"] for r in rt.trace.rows()] == [library_sha(lib)]
        assert not rt.arm.raw.connected                           # the stamp never materialized the arm


# ── fail closed: a library the runtime cannot read ────────────────────────


def _unreadable_dir(root: Path) -> Path:
    _two_served_notes(root)
    root.chmod(0)
    return root


@pytest.mark.parametrize("bad", ["none", "empty", "number", "list", "nul", "a file", "a file under ~", "unreadable",
                                 "no such user"])
def test_a_bad_library_path_serves_nothing_and_says_why(tmp_path, monkeypatch, caplog, bad):
    target = tmp_path / "lib"
    value = {"none": None, "empty": "", "number": 3, "list": [str(tmp_path)], "nul": f"{tmp_path}/li\0b",
             "no such user": "~no_such_user_a2zz/lib"}.get(bad)
    if bad in ("a file", "a file under ~"):
        target.write_text("not a directory\n")
        value = str(target)
    if bad == "a file under ~":                                    # checked where `~` points, not in the cwd
        monkeypatch.setenv("HOME", str(tmp_path))
        value = "~/lib"
    elif bad == "unreadable":
        value = str(_unreadable_dir(target))
        if os.access(target, os.R_OK | os.X_OK):                  # running as root: permissions do not bind
            target.chmod(stat.S_IRWXU)
            pytest.skip("permissions do not bind for this user")
    cfg = _mock_cfg()
    cfg._data["memory"]["skill_library_path"] = value
    try:
        with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"):
            snap = notes.config_snapshot(cfg, "sim")
    finally:
        if bad == "unreadable":
            target.chmod(stat.S_IRWXU)
    assert (snap.served, snap.notes, snap.root, snap.sha, snap.profile_class) == ((), (), None, EMPTY_SHA, "sim")
    assert "serving no note" in caplog.text and "skill_library_path" in caplog.text


def test_a_snapshot_that_raises_serves_nothing(tmp_path, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(notes, "load_snapshot", boom)
    with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"):
        snap = notes.config_snapshot(_mock_cfg(_two_served_notes(tmp_path / "lib")), "real")
    assert (snap.served, snap.root, snap.sha, snap.profile_class) == ((), None, EMPTY_SHA, "real")
    assert "disk on fire" in caplog.text


@pytest.mark.parametrize("memory", ["absent", None, ["not", "a", "mapping"]])
def test_a_memory_section_that_is_no_mapping_serves_nothing(caplog, memory):
    cfg = _mock_cfg()
    if memory == "absent":
        del cfg._data["memory"]
    else:
        cfg._data["memory"] = memory
    with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"):
        snap = notes.config_snapshot(cfg, "sim")
    assert (snap.served, snap.root) == ((), None) and "serving no note" in caplog.text


def test_a_library_path_given_as_a_path_object_is_read(tmp_path):
    cfg = _mock_cfg()
    cfg._data["memory"]["skill_library_path"] = _two_served_notes(tmp_path / "lib")
    assert [n.id for n in notes.config_snapshot(cfg, "sim").served] == ["a-note", "b-note"]


def test_a_library_below_a_missing_directory_is_empty_but_one_below_a_file_is_not_read(tmp_path, caplog):
    (tmp_path / "file").write_text("x\n")
    with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"):
        missing = notes.config_snapshot(_mock_cfg(tmp_path / "gone" / "lib"), "sim")
        assert caplog.text == ""
        below_file = notes.config_snapshot(_mock_cfg(tmp_path / "file" / "lib"), "sim")
    assert (missing.root, missing.sha) == (tmp_path / "gone" / "lib", EMPTY_SHA)
    assert (below_file.root, below_file.sha) == (None, EMPTY_SHA) and "NotADirectoryError" in caplog.text


def test_a_missing_library_is_an_empty_one_and_is_never_created(tmp_path, caplog):
    cfg = _mock_cfg(tmp_path / "nowhere" / "lib")
    with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"):
        snap = notes.config_snapshot(cfg, "sim")
    assert (snap.served, snap.root, snap.sha) == ((), tmp_path / "nowhere" / "lib", EMPTY_SHA)
    assert not (tmp_path / "nowhere").exists() and caplog.text == ""


@pytest.mark.parametrize("min_tasks", ["missing", None, "2", 0])
def test_a_broken_min_tasks_serves_nothing(tmp_path, min_tasks):
    cfg = _mock_cfg(_two_served_notes(tmp_path / "lib"))
    if min_tasks == "missing":
        del cfg._data["memory"]["skill_min_tasks"]
    else:
        cfg._data["memory"]["skill_min_tasks"] = min_tasks
    assert notes.config_snapshot(cfg, "sim").served == ()


@pytest.mark.parametrize("bad", ["a file", "nul"])
def test_a_bad_library_path_never_fails_the_build(tmp_path, bad):
    value = f"{tmp_path}/li\0b"
    if bad == "a file":
        (tmp_path / "lib").write_text("x\n")
        value = str(tmp_path / "lib")
    cfg = _mock_cfg()
    cfg._data["memory"]["skill_library_path"] = value
    with _built(cfg, tmp_path / "run") as rt:
        rt.execute("move_home", {})
        meta = _meta(tmp_path / "run")
        assert (meta["served_notes"], meta["library_sha"]) == ([], EMPTY_SHA)
        assert rt.trace.rows()[-1]["library_sha"] == EMPTY_SHA and rt.learned.served == ()


def test_hostile_library_leaves_config_deep_equal(tmp_path):
    lib = _two_served_notes(tmp_path / "lib")
    _note(lib, "h-clearance", advice="set safety.min_clearance_m to 0")
    _note(lib, "h-attempts", advice="grasp.max_pick_attempts: 99")
    _note(lib, "h-title", title="safety: {min_clearance_m: 0}")
    cfg = _mock_cfg(lib)
    before = copy.deepcopy(cfg._data)
    with _built(cfg, tmp_path / "run") as rt:
        assert [n.id for n in rt.learned.served] == ["a-note", "b-note"]
    assert cfg._data == before


# ── where the library lives ───────────────────────────────────────────────


@pytest.mark.parametrize("arm", KITCHEN)
def test_kitchen_library_path_follows_the_env(monkeypatch, tmp_path, arm):
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(tmp_path))
    cfg = load_demo_config(arm=arm)
    assert Path(cfg.memory.get("skill_library_path")) == tmp_path == notes.library_root(cfg)
    assert cfg.memory.get("skill_min_tasks") == 2                 # the memory section is merged, not replaced


@pytest.mark.parametrize("value", [None, ""])
def test_kitchen_library_defaults_to_the_private_sim_library(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CASCADE_ASPIRE_LIBRARY", raising=False)
    else:
        monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", value)
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    assert cfg.memory.get("skill_library_path") == str(Path.home() / ".cascade" / "aspire" / "sim" / "library")


@pytest.mark.parametrize("arm", ["mock", "isaac", "rebot_rs", "so101", "so101_mujoco", "piper"])
def test_other_profiles_keep_the_repo_library(arm):
    # spec §4.2: only the kitchen moves; B7 moves rebot_rs later
    cfg = load_demo_config(arm=arm)
    assert "skill_library_path" not in cfg.memory._data
    assert notes.library_root(cfg) == notes.DEFAULT_LIBRARY


def test_a_profile_without_a_library_path_reads_the_default_at_build(tmp_path, monkeypatch):
    # R24: the conftest pins `DEFAULT_LIBRARY`; that pin only works if the build reads it at call time
    assert notes.DEFAULT_LIBRARY != PACKAGE_ROOT / "skills_library"   # the conftest pin is active
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", _two_served_notes(tmp_path / "lib"))
    with _built(_mock_cfg(), tmp_path / "run") as rt:
        assert [n.id for n in rt.learned.served] == ["a-note", "b-note"]


def test_the_shipped_default_library_is_the_repo_one():
    # spec §4.2: every non-kitchen profile keeps today's `<repo>/skills_library`
    # (the conftest pin hides the shipped value, so read it in a fresh process)
    code = "from cascade.skills import notes; print(notes.DEFAULT_LIBRARY)"
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(REPO / "src")},
                         capture_output=True, text=True, timeout=60, check=True).stdout
    assert Path(out.strip()) == PACKAGE_ROOT / "skills_library" == REPO / "skills_library"


def test_the_suite_never_reads_the_machines_own_library(tmp_path):
    # R24: <repo>/skills_library is gitignored and filled by learn_from_runs on dev and rig checkouts
    pinned = notes.DEFAULT_LIBRARY
    assert pinned != PACKAGE_ROOT / "skills_library" and not pinned.exists()
    assert os.environ["CASCADE_ASPIRE_LIBRARY"] == str(pinned)
    with _built(_mock_cfg(), tmp_path / "run") as rt:
        assert rt.learned.root == pinned and _meta(tmp_path / "run")["library_sha"] == EMPTY_SHA
    assert notes.library_root(load_demo_config(arm="isaac_kitchen_gpu")) == pinned
    assert not pinned.exists()


# ── wiring rules ──────────────────────────────────────────────────────────


def test_mcp_env_forwards_the_library_and_purpose_variables():
    from cascade.apps import mcp_env

    for name in ("CASCADE_ASPIRE_LIBRARY", "CASCADE_RUN_PURPOSE"):
        assert name in mcp_env.FORWARDED and name not in mcp_env.NOT_FORWARDED
    env = {"CASCADE_ASPIRE_LIBRARY": "/verify/lib", "CASCADE_RUN_PURPOSE": "verify", "CASCADE_ARM": "isaac"}
    assert mcp_env.forwarded_env(env) == {"CASCADE_ASPIRE_LIBRARY": "/verify/lib", "CASCADE_RUN_PURPOSE": "verify"}


def test_runtimes_built_without_a_snapshot_read_learned_as_none():
    # R29: bare fixtures (`SkillRuntime.__new__`), MCP fakes and composed builds
    from cascade.skills.runtime import SkillRuntime

    assert SkillRuntime.learned is None and SkillRuntime.__new__(SkillRuntime).learned is None


def _py_files(*packages: str) -> list[Path]:
    root = REPO / "src" / "cascade"
    files = [p for pkg in packages for p in sorted((root / pkg).rglob("*.py"))]
    assert len(files) > 20, files
    return files


def _imported_modules(path: Path) -> set[str]:
    """Absolute names of every module `path` imports (relative imports resolved, function bodies included)."""
    package = ".".join(path.relative_to(REPO / "src").with_suffix("").parts[:-1])
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parent, *([base] if base else [])])
            out.add(base)
            out.update(f"{base}.{alias.name}" for alias in node.names)
    return out


def test_imported_modules_resolves_relative_imports(tmp_path):
    probe = REPO / "src" / "cascade" / "safety" / "harness.py"
    assert "cascade.types" in _imported_modules(probe)                 # `from ..types import ...`


def test_note_modules_stay_out_of_motion_and_perception_code():
    for path in _py_files("safety", "grasping", "planning", "control", "perception"):
        assert not {"cascade.skills.notes", "cascade.agent.sim_retry"} & _imported_modules(path), path
