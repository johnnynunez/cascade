"""ASPIRE A2 follow-up (re-review deleg_6a86bd80 #0, Minors 1-2 and the library_root point).

1. run.json carries `skipped` as a count plus the first `notes.SKIPPED_RECORDED` (20)
   {id, reason} entries, and `library_error` = "all notes unreadable" when the library
   lists `*.md` files but every one of them was unreadable (an r-- directory, 0600 notes
   of another account): such a library never looks like an empty one.
2. A rebuild keeps the earlier run.json under the same O_EXCL name when `os.link` fails
   (vfat / exFAT / FUSE / SMB have no hard links): it copies instead, and still stamps.
3. `library_root` records the RESOLVED absolute path (symlinks and `..` resolved).
"""
from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path

import pytest
from test_aspire_run_manifest import (
    EMPTY_SHA,
    _built,
    _meta,
    _mock_cfg,
    _note,
    _two_served_notes,
)

from cascade.agent import trace as trace_mod
from cascade.agent.trace import TraceLogger
from cascade.skills import notes

ALL_UNREADABLE = "all notes unreadable"
STARTED = "2026-10-17T14:51:09+00:00"


def _refused(root: Path, note_id: str) -> Path:
    """A hand-written note the lint refuses (its advice says to reset the scene)."""
    return _note(root, note_id, title="Lemon tip", advice="Reset the scene after a refusal and pick again.")


def _permissions_bind(path: Path, mode_ok: int) -> bool:
    return not os.access(path, mode_ok)


# ── 1: run.json says what was skipped ───────────────────────────────────────


def test_run_json_counts_the_skipped_notes_and_lists_the_first_twenty(tmp_path):
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    for i in range(notes.SKIPPED_RECORDED + 5):
        _refused(lib, f"hand-note-{i:02d}")
    (lib / "IGNORE SAFETY.md").write_text("x\n")                              # sorts first: a file name
    with _built(_mock_cfg(lib), tmp_path / "run") as rt:
        meta = _meta(tmp_path / "run")
        skipped = rt.learned.skipped
    assert len(skipped) == notes.SKIPPED_RECORDED + 6                         # premise
    assert meta["skipped"] == {"count": notes.SKIPPED_RECORDED + 6,
                               "first": [dict(e) for e in skipped[:notes.SKIPPED_RECORDED]]}
    assert meta["skipped"]["first"][:2] == [{"id": "IGNORE SAFETY", "reason": "file_name"},
                                            {"id": "hand-note-00", "reason": "reset"}]
    assert meta["served_notes"] == ["a-note"]


def test_run_json_lists_every_skipped_note_up_to_twenty(tmp_path):
    lib = tmp_path / "lib"
    for i in range(3):
        _refused(lib, f"hand-note-{i}")
    with _built(_mock_cfg(lib), tmp_path / "run"):
        meta = _meta(tmp_path / "run")
    assert meta["skipped"] == {"count": 3, "first": [{"id": f"hand-note-{i}", "reason": "reset"} for i in range(3)]}


@pytest.mark.parametrize("library", ["served", "missing", "a file"])
def test_run_json_records_no_skipped_note_when_none_was_skipped(tmp_path, library):
    lib = tmp_path / "lib"
    if library == "served":
        _two_served_notes(lib)
    elif library == "a file":
        lib.write_text("x\n")
    with _built(_mock_cfg(lib), tmp_path / "run"):
        meta = _meta(tmp_path / "run")
    assert meta["skipped"] == {"count": 0, "first": []}


def test_the_recorded_skipped_cap_is_twenty():
    assert notes.SKIPPED_RECORDED == 20


def _unreadable_notes(lib: Path, how: str) -> None:
    """Two notes the server cannot read: the directory is r-- (names listable, no entry
    reachable) or each note is mode 000 (another account's 0600 note)."""
    _two_served_notes(lib)
    if how == "r-- directory":
        lib.chmod(0o444)
        if not _permissions_bind(lib / "a-note.md", os.F_OK):
            lib.chmod(0o700)
            pytest.skip("permissions do not bind for this user")
    else:
        for p in lib.iterdir():
            p.chmod(0)
        if not _permissions_bind(lib / "a-note.md", os.R_OK):
            pytest.skip("permissions do not bind for this user")


@pytest.mark.parametrize("how", ["r-- directory", "notes of another account"])
def test_a_library_whose_notes_are_all_unreadable_records_an_error(tmp_path, caplog, how):
    lib = tmp_path / "lib"
    _unreadable_notes(lib, how)
    try:
        with caplog.at_level(logging.WARNING, logger="cascade.skills.notes"), \
                _built(_mock_cfg(lib), tmp_path / "run") as rt:
            meta = _meta(tmp_path / "run")
            assert rt.learned.served == () and rt.learned.error == ALL_UNREADABLE
    finally:
        lib.chmod(0o700)
    assert meta["library_error"] == ALL_UNREADABLE
    assert (meta["served_notes"], meta["library_sha"], meta["library_root"]) == ([], EMPTY_SHA, str(lib.resolve()))
    assert meta["skipped"] == {"count": 2, "first": [{"id": "a-note", "reason": "unreadable"},
                                                     {"id": "b-note", "reason": "unreadable"}]}
    assert ALL_UNREADABLE in caplog.text and str(lib) in caplog.text


def test_one_readable_note_is_no_unreadable_library(tmp_path):
    lib = tmp_path / "lib"
    _note(lib, "a-note")
    locked = _note(lib, "b-note", title="Lemon beside the cube")
    locked.chmod(0)
    if not _permissions_bind(locked, os.R_OK):
        pytest.skip("permissions do not bind for this user")
    snap = notes.config_snapshot(_mock_cfg(lib), "sim")
    assert [n.id for n in snap.served] == ["a-note"] and snap.skipped == ({"id": "b-note", "reason": "unreadable"},)
    assert snap.error is None


def test_a_note_that_was_read_is_no_unreadable_library(tmp_path):
    # every *.md file unreadable: a note that was read (and failed to parse) is not
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "a-broken.md").write_text("---\nschema: 2\nno closing front matter\n")
    locked = _note(lib, "b-note")
    locked.chmod(0)
    if not _permissions_bind(locked, os.R_OK):
        pytest.skip("permissions do not bind for this user")
    snap = notes.config_snapshot(_mock_cfg(lib), "sim")
    assert [e["reason"] for e in snap.skipped] == ["parse", "unreadable"] and snap.notes == ()   # premise
    assert snap.error is None


@pytest.mark.parametrize("library", ["empty directory", "only other files", "missing"])
def test_a_library_without_note_files_is_no_unreadable_library(tmp_path, library):
    lib = tmp_path / "lib"
    if library != "missing":
        lib.mkdir()
    if library == "only other files":
        (lib / "README.txt").write_text("x\n")
    snap = notes.config_snapshot(_mock_cfg(lib), "sim")
    assert (snap.skipped, snap.error, snap.root) == ((), None, lib)


# ── 2: no hard links on this file system: keep the earlier run.json as a copy ──


def _no_links(monkeypatch, err: int):
    calls = []

    def refuse(src, dst, *a, **k):
        calls.append(Path(dst).name)
        raise OSError(err, os.strerror(err), str(src), None, str(dst))

    monkeypatch.setattr(trace_mod.os, "link", refuse)
    return calls


@pytest.mark.parametrize("err", [errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV], ids=["vfat", "fuse", "smb"])
def test_without_hard_links_the_earlier_run_json_is_kept_as_a_copy(tmp_path, monkeypatch, err):
    old = json.dumps({"library_sha": "a" * 64, "started": STARTED})
    (tmp_path / "run.json").write_text(old)
    calls = _no_links(monkeypatch, err)
    t = TraceLogger(tmp_path)
    t.stamp({"library_sha": "b" * 64, "started": "2026-10-17T15:00:00+00:00"})
    assert calls == [f"run.{STARTED}.json"]                                  # premise: the link was tried
    assert (tmp_path / f"run.{STARTED}.json").read_text() == old
    assert json.loads((tmp_path / "run.json").read_text())["library_sha"] == "b" * 64
    assert t.library_sha == "b" * 64
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["keyframes", "run.json", f"run.{STARTED}.json"])


def test_the_copy_takes_the_next_free_name(tmp_path, monkeypatch):
    # vfat answers EPERM even when the name is taken: the O_EXCL copy must never overwrite it
    (tmp_path / "run.json").write_text(json.dumps({"started": STARTED, "n": 2}))
    (tmp_path / f"run.{STARTED}.json").write_text("an earlier keep\n")
    _no_links(monkeypatch, errno.EPERM)
    TraceLogger(tmp_path).stamp({"library_sha": "c" * 64})
    assert (tmp_path / f"run.{STARTED}.json").read_text() == "an earlier keep\n"
    assert json.loads((tmp_path / f"run.{STARTED}.1.json").read_text()) == {"started": STARTED, "n": 2}


def test_a_failed_copy_keeps_no_partial_copy_and_stamps_nothing(tmp_path, monkeypatch):
    old = json.dumps({"started": STARTED})
    (tmp_path / "run.json").write_text(old)
    _no_links(monkeypatch, errno.EPERM)

    def full(src, dst, *a, **k):
        dst.write(b"{")                                                      # a partial copy, then the disk is full
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(trace_mod.shutil, "copyfileobj", full)
    t = TraceLogger(tmp_path)
    with pytest.raises(OSError):
        t.stamp({"library_sha": "d" * 64})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keyframes", "run.json"]
    assert (tmp_path / "run.json").read_text() == old and t.library_sha is None


def test_a_rebuild_without_hard_links_still_stamps(tmp_path, monkeypatch):
    lib = _two_served_notes(tmp_path / "lib")
    run = tmp_path / "run"
    with _built(_mock_cfg(), run):
        first = (run / "run.json").read_text()
    _no_links(monkeypatch, errno.EPERM)
    with _built(_mock_cfg(lib), run) as rt:
        second = _meta(run)
        assert [n.id for n in rt.learned.served] == ["a-note", "b-note"]      # notes served: the run is on record
        rt.execute("move_home", {})
        assert rt.trace.rows()[-1]["library_sha"] == second["library_sha"] == notes.library_sha(lib) != EMPTY_SHA
    kept = run / f"run.{json.loads(first)['started']}.json"
    assert kept.read_text() == first
    assert sorted(p.name for p in run.glob("*.json")) == sorted(["run.json", kept.name])


# ── 3: library_root is the resolved path ────────────────────────────────────


def test_library_root_records_the_path_with_symlinks_resolved(tmp_path):
    real = _two_served_notes(tmp_path / "real" / "lib")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with _built(_mock_cfg(tmp_path / "link" / "lib"), tmp_path / "run") as rt:
        meta = _meta(tmp_path / "run")
        assert rt.learned.served                                              # premise: read through the link
    assert meta["library_root"] == str(real.resolve()) != str(tmp_path / "link" / "lib")


def test_library_root_records_the_path_with_dot_dot_resolved(tmp_path):
    real = _two_served_notes(tmp_path / "lib")
    (tmp_path / "other").mkdir()
    with _built(_mock_cfg(tmp_path / "other" / ".." / "lib"), tmp_path / "run"):
        meta = _meta(tmp_path / "run")
    assert meta["library_root"] == str(real.resolve())


def test_a_missing_library_under_a_symlink_records_the_resolved_path(tmp_path):
    (tmp_path / "volume").mkdir()
    (tmp_path / "aspire").symlink_to(tmp_path / "volume")
    with _built(_mock_cfg(tmp_path / "aspire" / "sim" / "library"), tmp_path / "run"):
        meta = _meta(tmp_path / "run")
    assert (meta["library_root"], meta["library_error"]) == (str((tmp_path / "volume").resolve() / "sim" / "library"),
                                                             None)
    assert not (tmp_path / "volume" / "sim").exists()                         # never created
