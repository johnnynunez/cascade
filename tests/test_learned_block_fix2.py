"""ASPIRE polish round (A3/A4 re-review deleg_1e3157c0 #0, Minor 6): the frozen library's housekeeping.

A failed freeze points every robot server of the launch at ONE notes-off directory,
`$STATE_DIR/aspire-library-off`: emptied and reused by every launch, removed by a launch whose
freeze succeeds (and with it any `aspire-library-off.*` an earlier launcher made). The copies a
freeze moves aside (`.aspire-library.old.*`) and could not remove go at the next freeze, whether
that one succeeds or fails. Everything runs in isolation with a private STATE_DIR and HOME: the
launch function alone, never launch.sh, no live port.
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest
from test_learned_block import _library
from test_learned_block_fix1 import _freeze, _run_launch

from cascade.skills import learned_block

#: entries beside the frozen copy that are not this module's: never removed
FOREIGN = (".aspire-library.older", "aspire-library.old.1", ".other.old.1")


def _old_copy(state: Path, tag: str, lib: Path) -> Path:
    """A copy an earlier freeze moved aside and could not remove: read-only notes in a directory."""
    old = state / f".aspire-library.old.{tag}"
    old.mkdir()
    for note in lib.iterdir():
        (old / note.name).write_bytes(note.read_bytes())
        (old / note.name).chmod(0o444)
    return old


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


# ── the copies earlier freezes left aside ─────────────────────────────────────


@pytest.mark.parametrize("outcome", ["frozen", "failed"])
def test_a_freeze_removes_the_copies_earlier_freezes_left_aside(tmp_path, monkeypatch, capsys, outcome):
    lib, state = _library(tmp_path / "lib"), tmp_path / "state"
    state.mkdir()
    for tag in ("4242.0a1b2c3d", "4243.deadbeef"):
        _old_copy(state, tag, lib)
    for name in FOREIGN:
        (state / name).mkdir()
    code = _freeze(monkeypatch, capsys, state / "aspire-library",
                   library=lib if outcome == "frozen" else "relative/library")[0]
    assert code == (0 if outcome == "frozen" else 1)
    assert _names(state) == sorted((["aspire-library"] if outcome == "frozen" else []) + list(FOREIGN))


def test_a_copy_left_aside_when_its_removal_fails_goes_at_the_next_freeze(tmp_path, monkeypatch, capsys):
    lib, state = _library(tmp_path / "lib"), tmp_path / "state"
    state.mkdir()
    assert _freeze(monkeypatch, capsys, state / "aspire-library", library=lib)[0] == 0
    with monkeypatch.context() as patched:                       # rmtree(ignore_errors=True) removed nothing
        patched.setattr(learned_block.shutil, "rmtree", lambda path, ignore_errors=False: None)
        assert _freeze(monkeypatch, capsys, state / "aspire-library", library=lib)[0] == 0
    left = [name for name in _names(state) if name.startswith(".aspire-library.old.")]
    assert len(left) == 1                                        # premise: the failed removal left one
    assert _freeze(monkeypatch, capsys, state / "aspire-library", library=lib)[0] == 0
    assert _names(state) == ["aspire-library"]


# ── launch.sh: one notes-off directory ───────────────────────────────────────


def _lib_of(result) -> Path:
    assert result.returncode == 0 and "CONTINUED" in result.stdout, result.stderr
    return Path(re.search(r"^LIB=(.*)$", result.stdout, re.MULTILINE).group(1))


FREEZE = 'freeze_note_library\necho "LIB=$CASCADE_ASPIRE_LIBRARY"'
REFUSED = {"CASCADE_ASPIRE_LIBRARY": "relative/library"}


def test_failed_freezes_reuse_one_empty_notes_off_directory(tmp_path):
    state = tmp_path / "state"
    off = _lib_of(_run_launch(tmp_path, FREEZE, **REFUSED))
    assert off == state / "aspire-library-off" and off.is_dir() and not off.is_symlink() and _names(off) == []
    (off / "a-note.md").write_text("written after the launch")              # never served by the next one
    legacy = state / "aspire-library-off.Ab12Cd"                             # the earlier launcher's mktemp dirs
    legacy.mkdir()
    (legacy / "b-note.md").write_text("a note")
    again = _run_launch(tmp_path, FREEZE, **REFUSED)
    assert _lib_of(again) == off and _names(off) == [] and stat.S_IMODE(off.stat().st_mode) == 0o700
    assert _names(state) == ["aspire-library-off"]
    assert again.stderr.count("[launch] WARNING: learned notes:") == 1


def test_a_freeze_that_succeeds_removes_the_notes_off_directories(tmp_path):
    lib, state = _library(tmp_path / "lib"), tmp_path / "state"
    _lib_of(_run_launch(tmp_path, FREEZE, **REFUSED))
    (state / "aspire-library-off.Ab12Cd").mkdir()
    assert _lib_of(_run_launch(tmp_path, FREEZE, CASCADE_ASPIRE_LIBRARY=str(lib))) == state / "aspire-library"
    assert _names(state) == ["aspire-library"]


def test_a_symlinked_notes_off_path_is_replaced_and_its_target_kept(tmp_path):
    state, target = tmp_path / "state", tmp_path / "elsewhere"
    target.mkdir()
    (target / "a-note.md").write_text("a note")
    state.mkdir()
    (state / "aspire-library-off").symlink_to(target)
    off = _lib_of(_run_launch(tmp_path, FREEZE, **REFUSED))
    assert off == state / "aspire-library-off" and not off.is_symlink() and _names(off) == []
    assert _names(target) == ["a-note.md"]


def test_a_notes_off_directory_that_cannot_be_emptied_is_never_exported(tmp_path):
    # fail closed: an off directory whose notes cannot be removed is left alone, and the
    # launch exports a new empty directory instead
    state = tmp_path / "state"
    stuck = state / "aspire-library-off" / "stuck"
    stuck.mkdir(parents=True)
    (stuck / "a-note.md").write_text("a note")
    stuck.chmod(0o500)
    try:
        if os.access(stuck, os.W_OK):
            pytest.skip("the test runs with permissions that ignore the mode bits (root)")
        off = _lib_of(_run_launch(tmp_path, FREEZE, **REFUSED))
        assert off.parent == state and off.name.startswith("aspire-library-off.") and _names(off) == []
        assert (stuck / "a-note.md").exists()
    finally:
        stuck.chmod(0o700)


def test_a_dry_run_removes_nothing(tmp_path):
    state = tmp_path / "state"
    for name in ("aspire-library-off", "aspire-library-off.Ab12Cd", ".aspire-library.old.1.ab"):
        (state / name).mkdir(parents=True)
    result = _run_launch(tmp_path, "freeze_note_library", DRY="1", **REFUSED)
    assert result.returncode == 0 and "CONTINUED" in result.stdout
    assert _names(state) == [".aspire-library.old.1.ab", "aspire-library-off", "aspire-library-off.Ab12Cd"]
