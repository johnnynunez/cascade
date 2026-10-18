"""ASPIRE A4: the generated `LEARNED.md` the kitchen prompt plugin injects after AGENTS.md.

Spec §4.3 (kitchen route), §7 (failure behaviour) and preflight R25 (dry run, pins),
R26 (a broken library or a hand-written file never stops the launch: warn, remove only
our own generated file, run without notes), R36 (no AGENTS.md is ever edited). The file
is ONE header line plus exactly A1's rendered block (`LibrarySnapshot.block()`), the text
every other route reads. Its sha is compared with the robot server's run.json: a
mismatch is a visible warning and the file is removed, so the prompt never shows notes
from another library than the one `robot_knowledge` and the trace stamp come from.
The plugin side is pinned in tests/test_spark_openclaw_prompt.py.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from test_aspire_notes import _note

from cascade.config import load_demo_config
from cascade.skills import learned_block, notes
from cascade.skills.learned_block import check_learned_notes, sync_learned_notes
from cascade.skills.notes import LEARNED_HEADER_PREFIX, LEARNED_MAX_BYTES, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/launch.sh"
KITCHEN = ("isaac_kitchen_gpu", "isaac,isaac_side,isaac_proof")


def _library(root: Path, n: int = 2) -> Path:
    titles = ("Cube near the lemon", "Lemon beside the cube", "Green cube on the square")
    for i in range(n):
        _note(root, f"{'abc'[i]}-note", title=titles[i])
    return root


@pytest.fixture
def snap_with_two(tmp_path):
    snap = load_snapshot(_library(tmp_path / "lib"), "sim")
    assert [n.id for n in snap.served] == ["a-note", "b-note"] and snap.block()   # premise
    return snap


@pytest.fixture
def workspace(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()
    return folder


def _header(snap) -> str:
    manifest = snap.manifest()
    return (f"{LEARNED_HEADER_PREFIX}library_sha={snap.sha} block_sha={manifest['block_sha']} "
            f"notes={','.join(manifest['block_notes'])} -->")


# ── the generator ─────────────────────────────────────────────────────────


def test_writes_header_block_atomically_0600(workspace, snap_with_two):
    assert sync_learned_notes(workspace, snap_with_two) is True
    path = workspace / "LEARNED.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith(f"<!-- cascade-learned-notes v1 library_sha={snap_with_two.sha} block_sha=")
    assert text == _header(snap_with_two) + "\n" + snap_with_two.block() + "\n"   # A1's block, verbatim
    assert _header(snap_with_two).endswith(" notes=a-note,b-note -->")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in workspace.iterdir()) == ["LEARNED.md"]           # no temp file left
    before = path.stat()
    assert sync_learned_notes(workspace, snap_with_two) is False                   # unchanged -> no rewrite
    after = path.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_a_rewrite_goes_through_a_temp_file_and_replace(workspace, snap_with_two, monkeypatch):
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((Path(src), Path(dst), Path(src).read_bytes()))
        return real(src, dst)

    monkeypatch.setattr(learned_block.os, "replace", spy)
    sync_learned_notes(workspace, snap_with_two)
    [(src, dst, data)] = seen
    assert dst == workspace / "LEARNED.md" and src.parent == workspace and src.name != "LEARNED.md"
    assert data == (workspace / "LEARNED.md").read_bytes()                        # complete before it is visible


def test_a_failed_replace_leaves_no_temp_file_and_the_old_file(workspace, snap_with_two, tmp_path, monkeypatch):
    sync_learned_notes(workspace, snap_with_two)
    before = (workspace / "LEARNED.md").read_bytes()

    def refuse(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(learned_block.os, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        sync_learned_notes(workspace, load_snapshot(_library(tmp_path / "other", 3), "sim"))
    assert sorted(p.name for p in workspace.iterdir()) == ["LEARNED.md"]
    assert (workspace / "LEARNED.md").read_bytes() == before


def test_nothing_served_removes_our_file(workspace, snap_with_two, tmp_path):
    empty = load_snapshot(tmp_path / "empty-lib", "sim")
    assert empty.served == () and empty.block() == ""                               # premise
    sync_learned_notes(workspace, snap_with_two)
    assert sync_learned_notes(workspace, empty) is True
    assert not (workspace / "LEARNED.md").exists()
    assert sync_learned_notes(workspace, empty) is False
    assert list(workspace.iterdir()) == []


def test_a_block_that_does_not_fit_writes_no_file(workspace, snap_with_two, monkeypatch):
    sync_learned_notes(workspace, snap_with_two)
    monkeypatch.setattr(notes, "BLOCK_MAX_CHARS", 100)                              # A1 renders no block
    assert snap_with_two.served and snap_with_two.block() == ""                     # premise
    assert sync_learned_notes(workspace, snap_with_two) is True
    assert not (workspace / "LEARNED.md").exists()


def _bounded(fn, *args):
    """fn(*args) on a daemon thread joined for at most 10 s: a call that blocks fails the test."""
    outcome = {}

    def run():
        try:
            outcome["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 -- re-raised below
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), f"{fn.__name__} blocked"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.mark.parametrize("kind", ["hand_written", "symlink_to_ours", "dangling_symlink", "directory", "fifo"])
@pytest.mark.parametrize("served", [True, False], ids=["notes", "no_notes"])
def test_refuses_hand_written_or_symlinked_file(workspace, snap_with_two, tmp_path, kind, served):
    path = workspace / "LEARNED.md"
    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text(_header(snap_with_two) + "\n" + snap_with_two.block() + "\n")
    if kind == "hand_written":
        path.write_text("operator notes\n")
    elif kind == "symlink_to_ours":
        path.symlink_to(elsewhere)
    elif kind == "dangling_symlink":
        path.symlink_to(tmp_path / "missing.md")
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    snap = snap_with_two if served else load_snapshot(tmp_path / "empty-lib", "sim")
    with pytest.raises(ValueError, match="LEARNED.md"):
        _bounded(sync_learned_notes, workspace, snap)       # a FIFO without a writer never blocks the launch
    if kind == "hand_written":
        assert path.read_text() == "operator notes\n"
    elif kind in ("symlink_to_ours", "dangling_symlink"):
        assert path.is_symlink() and elsewhere.read_text().startswith(LEARNED_HEADER_PREFIX)
    else:
        assert path.exists() and not path.is_file()
    assert sorted(p.name for p in workspace.iterdir()) == ["LEARNED.md"]


def test_a_fifo_holding_a_file_in_our_format_is_judged_on_its_descriptor(workspace, snap_with_two):
    # only the descriptor's regular-file check decides: the FIFO holds a forged header followed by
    # EOF (its writer closed; a reader keeps the pipe), so a plain read would take it for our file
    path = workspace / "LEARNED.md"
    os.mkfifo(path)
    keeper = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    writer = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    os.write(writer, (LEARNED_HEADER_PREFIX + "forged -->\n").encode())
    os.close(writer)
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            _bounded(sync_learned_notes, workspace, snap_with_two)
    finally:
        os.close(keeper)
    assert stat.S_ISFIFO(os.lstat(path).st_mode)


def test_without_o_nofollow_the_lstat_still_refuses_a_symlink(workspace, snap_with_two, tmp_path, monkeypatch):
    # a platform without O_NOFOLLOW (getattr fallback 0): only the lstat keeps the link unfollowed
    target = tmp_path / "ours.md"
    target.write_text(_header(snap_with_two) + "\n" + snap_with_two.block() + "\n")
    (workspace / "LEARNED.md").symlink_to(target)
    monkeypatch.delattr(learned_block.os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match="symlink"):
        sync_learned_notes(workspace, load_snapshot(tmp_path / "empty-lib", "sim"))
    assert (workspace / "LEARNED.md").is_symlink() and target.exists()


def test_a_file_swapped_for_a_symlink_after_the_listing_is_never_followed(workspace, snap_with_two, tmp_path,
                                                                         monkeypatch):
    # lstat said "regular file", then the path became a link to a file in our format: the open
    # refuses to follow it (O_NOFOLLOW), so the link is neither read through nor replaced
    path, target = workspace / "LEARNED.md", tmp_path / "ours.md"
    target.write_text(_header(snap_with_two) + "\n" + snap_with_two.block() + "\n")
    path.symlink_to(target)
    real = os.lstat

    def lstat(where, *args, **kwargs):
        found = real(where, *args, **kwargs)
        if Path(where) == path:
            return os.stat_result((stat.S_IFREG | 0o600, *tuple(found)[1:]))
        return found

    monkeypatch.setattr(learned_block.os, "lstat", lstat)
    with pytest.raises(ValueError, match="symlink"):
        sync_learned_notes(workspace, load_snapshot(tmp_path / "empty-lib", "sim"))
    assert stat.S_ISLNK(real(path).st_mode) and target.read_text().startswith(LEARNED_HEADER_PREFIX)


def test_a_file_that_only_starts_like_ours_is_replaced(workspace, snap_with_two):
    # our prefix on the first line = generated by this module (an older format or a torn header)
    (workspace / "LEARNED.md").write_text(LEARNED_HEADER_PREFIX + "library_sha=old -->\nold block\n")
    assert sync_learned_notes(workspace, snap_with_two) is True
    assert (workspace / "LEARNED.md").read_text().startswith(_header(snap_with_two) + "\n")


def test_the_header_names_only_the_notes_the_block_renders(workspace, tmp_path):
    snap = load_snapshot(_library(tmp_path / "lib", 3), "sim")
    assert [n.id for n in snap.served] == ["a-note", "b-note", "c-note"]          # premise: three served
    assert snap.manifest()["block_notes"] == ["a-note", "b-note"]                  # the block shows two
    sync_learned_notes(workspace, snap)
    assert (workspace / "LEARNED.md").read_text().split("\n", 1)[0].endswith(" notes=a-note,b-note -->")


def test_learned_file_fits_2000_bytes_with_non_ascii_titles(workspace, tmp_path):
    # Review Focus 5: the lint serves ASCII titles only, so a CJK title never reaches the file,
    # and the longest lint-clean notes under the longest ids still fit LEARNED_MAX_BYTES.
    lib = tmp_path / "lib"
    _note(lib, "cjk-note", title="立方体とレモン")
    _note(lib, "a" * 64, title="T" * 80, advice=("word " * 70)[:300].strip(), learned_in="unknown")
    _note(lib, "b" * 64, title="U" * 80, advice=("more " * 70)[:300].strip(), learned_in="unknown")
    snap = load_snapshot(lib, "sim")
    assert "cjk-note" not in [n.id for n in snap.served]                            # the lint refused it
    assert snap.manifest()["block_notes"] == ["a" * 64, "b" * 64]
    sync_learned_notes(workspace, snap)
    data = (workspace / "LEARNED.md").read_bytes()
    assert len(data) <= LEARNED_MAX_BYTES == 2000
    assert data.isascii() and data.decode().count(" | ") == 4                       # two ASCII tags, two bars each


def test_an_oversize_file_is_never_written_and_never_cut(workspace, snap_with_two, monkeypatch):
    sync_learned_notes(workspace, snap_with_two)                                    # an earlier library's file
    size = len((workspace / "LEARNED.md").read_bytes())
    monkeypatch.setattr(notes, "LEARNED_MAX_BYTES", size - 1)
    with pytest.raises(ValueError, match="LEARNED.md"):
        sync_learned_notes(workspace, snap_with_two)
    assert not (workspace / "LEARNED.md").exists()                                  # never the older notes
    monkeypatch.setattr(notes, "LEARNED_MAX_BYTES", size)
    assert sync_learned_notes(workspace, snap_with_two) is True                     # exactly the cap fits


@pytest.mark.parametrize("bad_id", ["a,b", "x-->y", "Upper", "new\nline", "", "-lead", "é"])
def test_a_note_id_that_is_no_safe_file_name_never_reaches_the_header(workspace, snap_with_two, bad_id):
    sync_learned_notes(workspace, snap_with_two)
    forged = dataclasses.replace(snap_with_two.served[0], id=bad_id)
    snap = dataclasses.replace(snap_with_two, served=(forged, *snap_with_two.served[1:]))
    with pytest.raises(ValueError, match="LEARNED.md"):
        sync_learned_notes(workspace, snap)
    assert not (workspace / "LEARNED.md").exists()                                  # the old file is gone too


@pytest.mark.parametrize("sha", ["abc", "A" * 64, "g" * 64, "0" * 63 + "\n"])
def test_a_library_sha_that_is_no_hex_digest_never_reaches_the_header(workspace, snap_with_two, sha):
    with pytest.raises(ValueError, match="LEARNED.md"):
        sync_learned_notes(workspace, dataclasses.replace(snap_with_two, sha=sha))
    assert not (workspace / "LEARNED.md").exists()


def test_regenerates_when_the_library_changes(workspace, snap_with_two, tmp_path):
    sync_learned_notes(workspace, snap_with_two)
    lib = snap_with_two.root
    path = lib / "a-note.md"
    path.write_text(path.read_text().replace("different grasp", "other grasp"))
    changed = load_snapshot(lib, "sim")
    assert changed.sha != snap_with_two.sha                                         # premise
    assert sync_learned_notes(workspace, changed) is True
    assert (workspace / "LEARNED.md").read_text() == _header(changed) + "\n" + changed.block() + "\n"


def test_agents_md_is_never_touched(workspace, snap_with_two, tmp_path):
    agents = workspace / "AGENTS.md"
    agents.write_bytes(b"x\n")
    os.utime(agents, ns=(1, 1))
    sync_learned_notes(workspace, snap_with_two)
    sync_learned_notes(workspace, load_snapshot(tmp_path / "empty-lib", "sim"))
    assert agents.read_bytes() == b"x\n" and agents.stat().st_mtime_ns == 1


@pytest.mark.parametrize("folder", ["relative/workspace", "", None])
def test_the_workspace_must_be_an_absolute_path(folder, snap_with_two):
    with pytest.raises(ValueError):
        sync_learned_notes(folder, snap_with_two)


# ── the command line (launch.sh, spark branch) ────────────────────────────


def _cli(monkeypatch, capsys, argv, *, arm=KITCHEN[0], cameras=KITCHEN[1], library=None):
    monkeypatch.setenv("CASCADE_ARM", arm)
    monkeypatch.setenv("CASCADE_CAMERAS", cameras)
    monkeypatch.delenv("CASCADE_ARMS", raising=False)
    if library is not None:
        monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(library))
    code = learned_block.main(argv)
    out = capsys.readouterr()
    return code, out.out.strip(), out.err


def _server_snapshot(arm, cameras):
    """The snapshot the robot server takes for the same environment (mcp_server + build_runtime)."""
    from cascade import sim_profile

    cfg = load_demo_config(cameras=cameras.split(","), arms=[arm], llm="mock")
    return notes.config_snapshot(cfg, sim_profile.profile_class(cfg))


def test_cli_writes_reports_and_removes_with_the_library_sha(tmp_path, monkeypatch, capsys):
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=lib)
    snap = _server_snapshot(*KITCHEN)
    assert snap.root == lib and snap.sha == notes.library_sha(lib)                  # premise: same library
    assert (code, out) == (0, f"wrote {ws / 'LEARNED.md'} library_sha={snap.sha}")
    assert (ws / "LEARNED.md").read_text() == _header(snap) + "\n" + snap.block() + "\n"
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=lib)
    assert (code, out) == (0, f"unchanged {ws / 'LEARNED.md'} library_sha={snap.sha}")
    empty = tmp_path / "empty"
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=empty)
    assert (code, out) == (0, f"removed {ws / 'LEARNED.md'} library_sha={notes.library_sha(empty)}")
    assert not (ws / "LEARNED.md").exists()


def test_cli_reads_the_robot_servers_arm_selection(tmp_path, monkeypatch, capsys):
    # mcp_server: CASCADE_ARMS wins over CASCADE_ARM; the mock profile reads the repo default library
    kitchen, default = _library(tmp_path / "kitchen"), _library(tmp_path / "default", 1)
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", default)
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("CASCADE_ARMS", "mock")
    monkeypatch.setenv("CASCADE_ARM", KITCHEN[0])
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(kitchen))
    monkeypatch.setenv("CASCADE_CAMERAS", "mock")
    assert learned_block.main(["--workspace", str(ws)]) == 0
    assert f"library_sha={notes.library_sha(default)}" in capsys.readouterr().out
    monkeypatch.delenv("CASCADE_ARMS")
    assert learned_block.main(["--workspace", str(ws)]) == 0
    assert f"library_sha={notes.library_sha(kitchen)}" in capsys.readouterr().out


def test_cli_on_a_real_profile_writes_no_unverified_note(tmp_path, monkeypatch, capsys):
    # the real route never runs the generator; even so, a real profile serves only verified notes
    lib = tmp_path / "lib"
    _note(lib, "authored-note", kind="authored", verifier=None)
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", lib)
    ws = tmp_path / "ws"
    ws.mkdir()
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws)], arm="rebot_rs", cameras="mock")
    assert load_snapshot(lib, "sim").served and not load_snapshot(lib, "real").served   # premise
    assert (code, out) == (0, f"unchanged {ws / 'LEARNED.md'} library_sha={notes.library_sha(lib)}")
    assert list(ws.iterdir()) == []


def test_cli_refusal_exits_nonzero_names_the_file_and_keeps_the_operator_file(tmp_path, monkeypatch, capsys):
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    (ws / "LEARNED.md").write_text("operator notes\n")
    code, out, err = _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=lib)
    assert code == 1 and str(ws / "LEARNED.md") in out + err
    assert (ws / "LEARNED.md").read_text() == "operator notes\n"


def test_cli_any_failure_removes_only_our_file(tmp_path, monkeypatch, capsys):
    # R26: a generator that cannot finish leaves no stale block behind and never a traceback for the launch
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    assert _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=lib)[0] == 0

    def boom(*_a, **_k):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(learned_block, "_server_config", boom)
    code, out, err = _cli(monkeypatch, capsys, ["--workspace", str(ws)], library=lib)
    assert code == 1 and "config exploded" in out + err and "Traceback" not in out + err
    assert not (ws / "LEARNED.md").exists()


def test_cli_relative_workspace_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, out, err = _cli(monkeypatch, capsys, ["--workspace", "ws"], library=_library(tmp_path / "lib"))
    assert code == 1 and "absolute" in out + err
    assert not (tmp_path / "ws").exists()


def test_python_dash_m_entry_point(tmp_path):
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    env = {**os.environ, "CASCADE_ARM": KITCHEN[0], "CASCADE_CAMERAS": KITCHEN[1],
           "CASCADE_ASPIRE_LIBRARY": str(lib), "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("CASCADE_ARMS", None)
    result = subprocess.run([sys.executable, "-m", "cascade.skills.learned_block", "--workspace", str(ws)],
                            env=env, cwd=ROOT / "models", capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"wrote {ws / 'LEARNED.md'} library_sha={notes.library_sha(lib)}\n"


# ── the cross-check with the robot server's run.json ──────────────────────


def _run_json(run_dir: Path, snap, **override) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"schema": 1, **snap.manifest(), **override}
    (run_dir / "run.json").write_text(json.dumps(meta))
    return run_dir


def test_check_matches_when_the_server_serves_the_same_library(workspace, snap_with_two, tmp_path):
    sync_learned_notes(workspace, snap_with_two)
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two))
    assert ok is True and f"library_sha={snap_with_two.sha}" in message
    assert (workspace / "LEARNED.md").exists()


@pytest.mark.parametrize("override", [
    {"library_sha": "f" * 64},                                                     # another library
    {"block_notes": ["a-note"]},                                                   # same bytes, other block
    {"block_notes": ["b-note", "a-note"]},                                         # other order
    {"block_notes": []},                                                           # server shows no block
], ids=["sha", "notes", "order", "no_block"])
def test_check_mismatch_warns_and_removes_our_file(workspace, snap_with_two, tmp_path, override):
    sync_learned_notes(workspace, snap_with_two)
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two, **override))
    assert ok is False and "LEARNED.md" in message and "run.json" in message
    assert snap_with_two.sha in message and "removed" in message
    assert not (workspace / "LEARNED.md").exists()


@pytest.mark.parametrize("damage", ["missing", "not_json", "no_sha", "list", "sha_not_str", "notes_not_list"])
def test_check_without_a_readable_run_json_removes_our_file(workspace, snap_with_two, tmp_path, damage):
    sync_learned_notes(workspace, snap_with_two)
    run = _run_json(tmp_path / "run", snap_with_two)
    meta = json.loads((run / "run.json").read_text())
    if damage == "missing":
        (run / "run.json").unlink()
    elif damage == "not_json":
        (run / "run.json").write_text("{")
    elif damage == "no_sha":
        del meta["library_sha"]
        (run / "run.json").write_text(json.dumps(meta))
    elif damage == "list":
        (run / "run.json").write_text("[]")
    elif damage == "sha_not_str":
        (run / "run.json").write_text(json.dumps({**meta, "library_sha": 7}))
    else:
        (run / "run.json").write_text(json.dumps({**meta, "block_notes": "a-note,b-note"}))
    ok, message = check_learned_notes(workspace, run)
    assert ok is False and "run.json" in message
    assert not (workspace / "LEARNED.md").exists()


def test_check_without_our_file_warns_when_the_server_has_a_block(workspace, snap_with_two, tmp_path):
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two))
    assert ok is False and "a-note" in message and not (workspace / "LEARNED.md").exists()
    empty = load_snapshot(tmp_path / "empty-lib", "sim")
    ok, _ = check_learned_notes(workspace, _run_json(tmp_path / "run2", empty))
    assert ok is True and list(workspace.iterdir()) == []


def test_an_oversize_run_json_is_no_run_record(workspace, snap_with_two, tmp_path, monkeypatch):
    sync_learned_notes(workspace, snap_with_two)
    run = _run_json(tmp_path / "run", snap_with_two)
    monkeypatch.setattr(learned_block, "RUN_JSON_MAX_BYTES", len((run / "run.json").read_bytes()) - 1)
    ok, message = check_learned_notes(workspace, run)
    assert ok is False and "too large" in message and not (workspace / "LEARNED.md").exists()


@pytest.mark.parametrize("override", [{"block_notes": {}}, {"block_notes": [None]}, {"library_sha": 7},
                                      {"library_sha": "f" * 63}], ids=["dict", "not_str", "sha_int", "sha_short"])
def test_a_malformed_run_json_is_never_read_as_no_block(workspace, tmp_path, override):
    # nothing in the prompt and a record whose block reads as empty would otherwise "match"
    empty = load_snapshot(tmp_path / "empty-lib", "sim")
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", empty, **override))
    assert ok is False and "cannot be read" in message


@pytest.mark.parametrize("kind", ["hand_written", "symlink_to_ours"])
def test_check_never_removes_a_file_it_did_not_write(workspace, snap_with_two, tmp_path, kind):
    path = workspace / "LEARNED.md"
    if kind == "hand_written":
        path.write_text("operator notes\n")
    else:
        target = tmp_path / "ours.md"
        target.write_text(_header(snap_with_two) + "\n" + snap_with_two.block() + "\n")
        path.symlink_to(target)
    ok, _ = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two, library_sha="f" * 64))
    assert ok is False
    assert path.is_symlink() if kind != "hand_written" else path.read_text() == "operator notes\n"


def test_check_against_a_real_build_and_a_library_changed_after_the_launch(tmp_path, monkeypatch, capsys):
    """The launch writes LEARNED.md, then the robot server builds; a note added in between
    makes the server serve another library: the check sees it and removes the file."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", lib)               # the mock profile's library
    assert _cli(monkeypatch, capsys, ["--workspace", str(ws)], arm="mock", cameras="mock")[0] == 0
    cfg = load_demo_config(cameras=["mock"], arms=["mock"], llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run1", lazy_arm=True, serve=False, view=False)
    shutdown_runtime(runtime, arm)
    capsys.readouterr()                                              # the build's banner
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws), "--check-run", str(tmp_path / "run1")],
                        arm="mock", cameras="mock")
    assert (code, out) == (0, f"matches {ws / 'LEARNED.md'} library_sha={notes.library_sha(lib)}")
    _note(lib, "c-note", title="Green cube on the square")                        # library changes after launch
    runtime, arm = build_runtime(cfg, tmp_path / "run2", lazy_arm=True, serve=False, view=False)
    shutdown_runtime(runtime, arm)
    capsys.readouterr()
    code, out, _ = _cli(monkeypatch, capsys, ["--workspace", str(ws), "--check-run", str(tmp_path / "run2")],
                        arm="mock", cameras="mock")
    assert code == 1 and "removed" in out and notes.library_sha(lib) in out
    assert not (ws / "LEARNED.md").exists()


# ── launch.sh (spark branch) ──────────────────────────────────────────────


def _function(name: str) -> str:
    source = LAUNCH.read_text()
    start = source.index(f"\n{name}() {{\n") + 1
    return source[start:source.index("\n}\n", start) + 3]


def _run_launch_function(tmp_path, call: str, **env):
    prelude = "set -euo pipefail\nlog() { printf '[launch] %s\\n' \"$*\"; }\n" \
              "warn() { printf '[launch] WARNING: %s\\n' \"$*\" >&2; }\n" \
              "die() { printf '[launch] ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp_path)), "REPO": str(ROOT),
            "PY": sys.executable, "ARM": KITCHEN[0], "CAMERAS": KITCHEN[1], "DRY": "0",
            "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(["bash", "-c", prelude + _function("learned_notes") + call + "\necho CONTINUED\n"],
                          env={**base, **env}, capture_output=True, text=True, timeout=120, check=False)


def test_launch_writes_the_file_with_the_launch_arm(tmp_path):
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    result = _run_launch_function(tmp_path, f'learned_notes "{ws}"', CASCADE_ASPIRE_LIBRARY=str(lib))
    assert result.returncode == 0, result.stderr
    assert f"[launch] learned notes: wrote {ws / 'LEARNED.md'} library_sha={notes.library_sha(lib)}" in result.stdout
    assert (ws / "LEARNED.md").read_text().startswith(LEARNED_HEADER_PREFIX)


def test_launch_dry_run_prints_and_writes_nothing(tmp_path):
    # preflight R25: the generator is a command like any other under --dry-run
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    result = _run_launch_function(tmp_path, f'learned_notes "{ws}"', DRY="1",
                                  CASCADE_ASPIRE_LIBRARY=str(lib))
    assert result.returncode == 0 and "CONTINUED" in result.stdout
    assert "cascade.skills.learned_block --workspace" in result.stdout
    assert list(ws.iterdir()) == []


@pytest.mark.parametrize("failure", ["generator_refuses", "python_broken"])
def test_launch_failure_warns_removes_only_our_file_and_continues(tmp_path, failure):
    # preflight R26: never `die`; notes off; an operator's file stays
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    env = {"CASCADE_ASPIRE_LIBRARY": str(lib)}
    assert _run_launch_function(tmp_path, f'learned_notes "{ws}"', **env).returncode == 0
    assert (ws / "LEARNED.md").exists()                                              # ours, from a good run
    if failure == "python_broken":
        broken = tmp_path / "python"
        broken.write_text("#!/bin/sh\necho 'interpreter exploded' >&2\nexit 3\n")
        broken.chmod(0o755)
        env["PY"] = str(broken)
    else:
        env["CASCADE_ASPIRE_LIBRARY"] = "relative/library"                         # the CLI fails closed: no notes
    result = _run_launch_function(tmp_path, f'learned_notes "{ws}"', **env)
    assert result.returncode == 0 and "CONTINUED" in result.stdout, result.stderr
    if failure == "python_broken":
        assert "[launch] WARNING: learned notes:" in result.stderr and "interpreter exploded" in result.stderr
        assert not (ws / "LEARNED.md").exists()
    else:
        assert not (ws / "LEARNED.md").exists() and "removed" in result.stdout
    (ws / "LEARNED.md").write_text("operator notes\n")
    result = _run_launch_function(tmp_path, f'learned_notes "{ws}"', **env)
    assert result.returncode == 0 and "CONTINUED" in result.stdout
    assert "[launch] WARNING: learned notes:" in result.stderr
    assert (ws / "LEARNED.md").read_text() == "operator notes\n"


def test_launch_check_mismatch_is_a_visible_warning_and_the_launch_continues(tmp_path, snap_with_two):
    ws = tmp_path / "ws"
    ws.mkdir()
    sync_learned_notes(ws, snap_with_two)
    run = _run_json(tmp_path / "run", snap_with_two, library_sha="f" * 64)
    result = _run_launch_function(tmp_path, f'learned_notes "{ws}" --check-run "{run}"')
    assert result.returncode == 0 and "CONTINUED" in result.stdout
    assert "[launch] WARNING: learned notes:" in result.stderr and "f" * 64 in result.stderr
    assert not (ws / "LEARNED.md").exists()


def test_launch_calls_the_generator_after_the_visitor_sync_and_checks_after_the_proof():
    source = LAUNCH.read_text()
    sync = source.index("sync_visitor_instructions(Path(sys.argv[2]))\nPYEOF\n")
    gateway = source.index('if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then\n'
                           "    log \"starting the checkout's foreground gateway")
    generate = source.index('learned_notes "$PROFILE_WORKSPACE"\n')
    assert sync < generate < gateway
    proof = source.index('PROOF_TRACE="$(')
    judge = source.index("    judge_pass\nfi\n")
    check = source.index('learned_notes "$LEARNED_WORKSPACE" --check-run "$(dirname "$PROOF_TRACE")"')
    assert proof < check < judge
    function = _function("learned_notes")
    assert "die" not in function and "|| die" not in source[generate:generate + 120]
    # the OpenShell route (real reBot) never reaches either call: it ends the launch earlier
    assert source.index('. "$REPO/scripts/launch_openshell.sh"\n    exit 0') < generate
    assert re.search(r"cascade\.skills\.learned_block", (ROOT / "scripts/launch_openshell.sh").read_text()) is None
