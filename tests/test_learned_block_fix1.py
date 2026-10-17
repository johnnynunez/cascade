"""ASPIRE A4 fix round 1 (task-A4-fix1.md rulings 3-7; A4 review deleg_8cabb30b #1).

3. One frozen note library per launch: in the spark branch, before the robot server's
   environment is built, launch.sh copies the note files the library reader accepts
   (safe names, regular files, no symlink, at most `NOTE_MAX_BYTES`) into
   `$STATE_DIR/aspire-library` (a temp directory renamed into place) and exports
   CASCADE_ASPIRE_LIBRARY to it, so every robot server of the launch and LEARNED.md read
   the same bytes. A failed freeze points it at a new empty directory (notes off
   everywhere) with a warning and the launch goes on (R26).
4. `block_sha` (sha256 of the rendered block) is in run.json and in the LEARNED.md header;
   `--check-run` compares it and checks the file's block against it.
6. The plugin's note-id pattern is A1's note file-name rule without ".md".
7. One safe reader (`notes.read_regular`: O_NOFOLLOW, O_NONBLOCK, fstat, a byte limit)
   serves notes.py and learned_block.py.
The plugin's BOM refusal (ruling 5) is in tests/test_spark_openclaw_prompt.py.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from test_aspire_notes import _note
from test_learned_block import (
    KITCHEN,
    LAUNCH,
    ROOT,
    _bounded,
    _function,
    _library,
    _run_json,
    _server_snapshot,
)

from cascade.config import load_demo_config
from cascade.skills import learned_block, notes
from cascade.skills.learned_block import check_learned_notes, sync_learned_notes
from cascade.skills.notes import LEARNED_HEADER_PREFIX, library_sha, load_snapshot

EMPTY_SHA = hashlib.sha256(b"").hexdigest()


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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── 4: block_sha ─────────────────────────────────────────────────────────────


def test_the_manifest_carries_the_sha256_of_the_rendered_block(snap_with_two, tmp_path):
    assert snap_with_two.manifest()["block_sha"] == _sha(snap_with_two.block())
    empty = load_snapshot(tmp_path / "none", "sim")
    assert empty.manifest()["block_sha"] == EMPTY_SHA


def test_run_json_records_the_block_sha(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    lib = _library(tmp_path / "lib")
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", lib)                         # the mock profile's library
    cfg = load_demo_config(cameras=["mock"], arms=["mock"], llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", lazy_arm=True, serve=False, view=False)
    try:
        block = runtime.learned.block()
    finally:
        shutdown_runtime(runtime, arm)
    meta = json.loads((tmp_path / "run" / "run.json").read_text())
    assert block and meta["block_sha"] == _sha(block) and meta["block_notes"] == ["a-note", "b-note"]


def test_the_header_carries_the_block_sha(workspace, snap_with_two):
    sync_learned_notes(workspace, snap_with_two)
    first, body = (workspace / "LEARNED.md").read_text().split("\n", 1)
    assert first == (f"{LEARNED_HEADER_PREFIX}library_sha={snap_with_two.sha} "
                     f"block_sha={_sha(snap_with_two.block())} notes=a-note,b-note -->")
    assert body == snap_with_two.block() + "\n"


def test_check_matches_on_all_three(workspace, snap_with_two, tmp_path):
    sync_learned_notes(workspace, snap_with_two)
    ok, _ = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two))
    assert ok is True and (workspace / "LEARNED.md").exists()


def test_check_compares_the_block_sha(workspace, snap_with_two, tmp_path):
    # same library, same note ids, another rendering (a tag or cut the server renders otherwise)
    sync_learned_notes(workspace, snap_with_two)
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two, block_sha="f" * 64))
    assert ok is False and "removed" in message and "f" * 64 in message and _sha(snap_with_two.block()) in message
    assert not (workspace / "LEARNED.md").exists()


def test_check_refuses_a_block_edited_after_it_was_written(workspace, snap_with_two, tmp_path):
    sync_learned_notes(workspace, snap_with_two)
    path = workspace / "LEARNED.md"
    text = path.read_text()
    assert "different grasp" in text.split("\n", 1)[1]                            # premise
    path.write_text(text.replace("different grasp", "other grasp", 1))            # the header kept
    ok, message = check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two))
    assert ok is False and "removed" in message and "block_sha" in message
    assert not path.exists()


@pytest.mark.parametrize("value", [None, 7, "f" * 63, "F" * 64], ids=["missing", "int", "short", "upper"])
def test_a_run_json_without_a_block_sha_is_no_run_record(workspace, snap_with_two, tmp_path, value):
    sync_learned_notes(workspace, snap_with_two)
    run = _run_json(tmp_path / "run", snap_with_two)
    meta = json.loads((run / "run.json").read_text())
    if value is None:
        del meta["block_sha"]
    else:
        meta["block_sha"] = value
    (run / "run.json").write_text(json.dumps(meta))
    ok, message = check_learned_notes(workspace, run)
    assert ok is False and "cannot be read" in message and "block_sha" in message
    assert not (workspace / "LEARNED.md").exists()


# ── 6: the plugin's id pattern is A1's file-name rule ─────────────────────────


def test_the_plugin_note_pattern_is_the_note_file_name_rule():
    source = (ROOT / "demo/kitchen/openclaw-plugin/index.mjs").read_text()
    [js] = re.findall(r'^const LEARNED_NOTE = "([^"]*)";$', source, re.MULTILINE)
    assert notes._NOTE_NAME_RE.pattern.endswith(r"\.md")
    assert js == notes._NOTE_NAME_RE.pattern.removesuffix(r"\.md")


# ── 7: one safe reader ───────────────────────────────────────────────────────


def _calls(module, name: str) -> list[str]:
    """The functions of `module` that call `os.<name>`."""
    tree = ast.parse(Path(module.__file__).read_text())
    found = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "os"):
                    found.append(fn.name)
    return found


def test_one_reader_opens_reads_and_fstats_for_both_modules():
    for name in ("open", "read", "fstat"):
        assert _calls(notes, name) == ["read_regular"], name
        assert _calls(learned_block, name) == [], name


def test_both_modules_read_through_the_shared_reader(workspace, snap_with_two, tmp_path, monkeypatch):
    seen = []
    real = notes.read_regular

    def spy(path, limit, *, whole):
        seen.append((Path(path).name, whole))
        return real(path, limit, whole=whole)

    monkeypatch.setattr(notes, "read_regular", spy)
    load_snapshot(tmp_path / "lib", "sim")
    sync_learned_notes(workspace, snap_with_two)
    check_learned_notes(workspace, _run_json(tmp_path / "run", snap_with_two))
    assert {("a-note.md", True), ("b-note.md", True)} <= set(seen)
    assert ("LEARNED.md", False) in seen and ("run.json", False) in seen


def test_the_reader_refuses_a_symlink_on_the_lstat_and_on_the_open(tmp_path, monkeypatch):
    target = tmp_path / "target.md"
    target.write_text("x")
    (tmp_path / "link.md").symlink_to(target)
    with pytest.raises(notes.UnsafeFile) as caught:
        notes.read_regular(tmp_path / "link.md", 100, whole=True)
    assert caught.value.reason == "symlink"
    monkeypatch.setattr(notes.os, "lstat", lambda path, real=os.lstat: real(target))  # swapped after the lstat
    with pytest.raises(notes.UnsafeFile) as caught:
        notes.read_regular(tmp_path / "link.md", 100, whole=True)
    assert caught.value.reason == "symlink"


@pytest.mark.parametrize("kind", ["fifo", "directory"])
def test_the_reader_judges_the_descriptor_and_never_blocks(tmp_path, kind):
    path = tmp_path / "x.md"
    os.mkfifo(path) if kind == "fifo" else path.mkdir()
    for whole in (True, False):
        with pytest.raises(notes.UnsafeFile) as caught:
            _bounded(lambda: notes.read_regular(path, 100, whole=whole))     # noqa: B023 -- called at once
        assert caught.value.reason == "not_a_file"


def test_the_reader_reads_a_whole_file_or_refuses_it_on_its_size(tmp_path, monkeypatch):
    path = tmp_path / "x.md"
    path.write_bytes(b"a" * 10)
    assert notes.read_regular(path, 10, whole=True) == b"a" * 10
    assert notes.read_regular(path, 4, whole=False) == b"aaaa"                  # a prefix: never refused
    monkeypatch.setattr(notes.os, "read", lambda *a: pytest.fail("read past an oversize fstat"))
    with pytest.raises(notes.UnsafeFile) as caught:
        notes.read_regular(path, 9, whole=True)
    assert caught.value.reason == "too_large"


def test_the_reader_stops_one_byte_past_the_cap_when_a_file_grows(tmp_path, monkeypatch):
    path = tmp_path / "x.md"
    path.write_bytes(b"a" * 4)
    real_fstat = os.fstat

    def small(fd):                                                             # it was 1 byte at the fstat
        fields = list(real_fstat(fd))[:10]
        fields[stat.ST_SIZE] = 1
        return os.stat_result(fields)

    monkeypatch.setattr(notes.os, "fstat", small)
    reads = []
    real_read = os.read
    monkeypatch.setattr(notes.os, "read", lambda fd, n: reads.append(n) or real_read(fd, n))
    with pytest.raises(notes.UnsafeFile) as caught:
        notes.read_regular(path, 2, whole=True)
    assert caught.value.reason == "too_large" and sum(reads) <= 3


def test_the_reader_lets_a_missing_file_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        notes.read_regular(tmp_path / "none.md", 10, whole=True)


# ── 3: the frozen library ────────────────────────────────────────────────────


def _messy_library(root: Path) -> Path:
    """Two notes plus every entry the reader refuses: a bad name, a symlink, an oversize file,
    a directory named like a note, and a file that is no note at all."""
    _library(root)
    (root / "Upper.md").write_bytes((root / "a-note.md").read_bytes())
    (root / "link.md").symlink_to(root / "a-note.md")
    (root / "big.md").write_bytes(b"x" * (notes.NOTE_MAX_BYTES + 1))
    (root / "sub.md").mkdir()
    (root / "README.txt").write_text("not a note")
    return root


def _freeze(monkeypatch, capsys, dest: Path, *, library=None, arm=KITCHEN[0]):
    monkeypatch.setenv("CASCADE_ARM", arm)
    monkeypatch.setenv("CASCADE_CAMERAS", KITCHEN[1])
    monkeypatch.delenv("CASCADE_ARMS", raising=False)
    if library is not None:
        monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(library))
    code = learned_block.main(["--freeze", str(dest)])
    out = capsys.readouterr()
    return code, out.out.strip(), out.err.strip()


def test_freeze_copies_exactly_the_note_files_the_reader_accepts(tmp_path, monkeypatch, capsys):
    lib, dest = _messy_library(tmp_path / "lib"), tmp_path / "state" / "aspire-library"
    dest.parent.mkdir()
    code, out, err = _freeze(monkeypatch, capsys, dest, library=lib)
    assert code == 0, err
    assert sorted(p.name for p in dest.iterdir()) == ["a-note.md", "b-note.md"]
    for name in ("a-note.md", "b-note.md"):
        assert (dest / name).read_bytes() == (lib / name).read_bytes()
        assert not (dest / name).is_symlink() and stat.S_IMODE((dest / name).stat().st_mode) == 0o444  # read-only
    assert library_sha(dest) == library_sha(lib)
    assert out == (f"froze {lib} into {dest}: 2 note files, library_sha={library_sha(lib)}; "
                   "left out 4 entries: file_name x1, not_a_file x1, symlink x1, too_large x1")
    assert sorted(p.name for p in dest.parent.iterdir()) == ["aspire-library"]       # no temp directory left


def test_freeze_replaces_an_earlier_copy_whole(tmp_path, monkeypatch, capsys):
    lib, dest = _library(tmp_path / "lib", 3), tmp_path / "state" / "aspire-library"
    dest.parent.mkdir()
    assert _freeze(monkeypatch, capsys, dest, library=lib)[0] == 0
    (lib / "c-note.md").unlink()
    _note(lib, "a-note", title="Cube near the lemon", advice="Look again first.")
    assert _freeze(monkeypatch, capsys, dest, library=lib)[0] == 0
    assert sorted(p.name for p in dest.iterdir()) == ["a-note.md", "b-note.md"]
    assert library_sha(dest) == library_sha(lib) and "Look again first." in (dest / "a-note.md").read_text()
    assert sorted(p.name for p in dest.parent.iterdir()) == ["aspire-library"]


def test_a_missing_library_freezes_as_an_empty_one(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "state" / "aspire-library"
    dest.parent.mkdir()
    code, out, _ = _freeze(monkeypatch, capsys, dest, library=tmp_path / "not-yet")
    assert code == 0 and list(dest.iterdir()) == [] and library_sha(dest) == EMPTY_SHA
    assert out == f"froze {tmp_path / 'not-yet'} into {dest}: 0 note files, library_sha={EMPTY_SHA}"
    assert not (tmp_path / "not-yet").exists()                                    # never created


@pytest.mark.parametrize("failure", ["relative", "unreadable", "dangling", "unbound_profile", "copy_fails",
                                     "all_unreadable"])
def test_a_failed_freeze_exits_nonzero_and_leaves_no_partial_copy(tmp_path, monkeypatch, capsys, failure):
    lib, dest = _library(tmp_path / "lib"), tmp_path / "state" / "aspire-library"
    dest.parent.mkdir()
    arm, library = KITCHEN[0], lib
    if failure == "relative":
        library = "relative/library"
    elif failure == "unreadable":
        library = tmp_path / "file"
        library.write_text("a file, not a directory")
    elif failure == "dangling":
        library = tmp_path / "volume"
        library.symlink_to(tmp_path / "unmounted")
    elif failure == "unbound_profile":
        arm = "mock"                                       # reads its own library, never CASCADE_ASPIRE_LIBRARY
    elif failure == "all_unreadable":
        for path in lib.iterdir():                         # notes of another account: no note can be read
            path.chmod(0)
    else:
        monkeypatch.setattr(learned_block.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    code, out, err = _freeze(monkeypatch, capsys, dest, library=library, arm=arm)
    assert code == 1 and out == "" and err.count("\n") == 0 and "Traceback" not in err
    assert {"relative": "relative path", "unreadable": "NotADirectoryError", "dangling": "symlink to a missing",
            "unbound_profile": "CASCADE_ASPIRE_LIBRARY", "copy_fails": "disk full",
            "all_unreadable": notes.ALL_UNREADABLE}[failure] in err
    assert list(dest.parent.iterdir()) == []                                      # no copy, no temp directory


def test_a_failed_freeze_keeps_an_earlier_copy_untouched(tmp_path, monkeypatch, capsys):
    lib, dest = _library(tmp_path / "lib"), tmp_path / "state" / "aspire-library"
    dest.parent.mkdir()
    assert _freeze(monkeypatch, capsys, dest, library=lib)[0] == 0
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    assert _freeze(monkeypatch, capsys, dest, library="relative/library")[0] == 1
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == before


def test_the_frozen_copy_is_what_the_server_and_learned_md_read(tmp_path, monkeypatch, capsys):
    """Every reader of the launch reads the frozen bytes: a note added to the library after
    the freeze reaches neither LEARNED.md nor the robot server's snapshot."""
    lib, dest, ws = _library(tmp_path / "lib"), tmp_path / "state" / "aspire-library", tmp_path / "ws"
    dest.parent.mkdir()
    ws.mkdir()
    assert _freeze(monkeypatch, capsys, dest, library=lib)[0] == 0
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(dest))                       # what launch.sh exports
    _note(lib, "c-note", title="Green cube on the square", tasks=("t1", "t2", "t3"))
    assert learned_block.main(["--workspace", str(ws)]) == 0
    server = _server_snapshot(*KITCHEN)
    assert server.root == dest and server.sha == library_sha(dest) != library_sha(lib)
    assert [n.id for n in server.served] == ["a-note", "b-note"]
    run = _run_json(tmp_path / "run", server)
    capsys.readouterr()
    assert learned_block.main(["--workspace", str(ws), "--check-run", str(run)]) == 0


@pytest.mark.parametrize("arm", ["isaac_kitchen_gpu", "isaac_kitchen_cumotion"])
def test_every_spark_arm_reads_its_library_from_the_launch_variable(tmp_path, monkeypatch, arm):
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(tmp_path / "frozen"))
    cfg = load_demo_config(cameras=KITCHEN[1].split(","), arms=[arm], llm="mock")
    assert notes.library_root(cfg) == tmp_path / "frozen"


def test_the_command_line_takes_one_mode(tmp_path):
    for argv in ([], ["--workspace", str(tmp_path), "--freeze", str(tmp_path / "f")],
                 ["--freeze", str(tmp_path / "f"), "--check-run", str(tmp_path)]):
        with pytest.raises(SystemExit) as caught:
            learned_block.main(argv)
        assert caught.value.code == 2


# ── 3: launch.sh ─────────────────────────────────────────────────────────────


def _run_launch(tmp_path, call: str, **env):
    prelude = "set -euo pipefail\nlog() { printf '[launch] %s\\n' \"$*\"; }\n" \
              "warn() { printf '[launch] WARNING: %s\\n' \"$*\" >&2; }\n" \
              "die() { printf '[launch] ERROR: %s\\n' \"$*\" >&2; exit 1; }\n"
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    base = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO": str(ROOT), "STATE_DIR": str(state),
            "PY": sys.executable, "ARM": KITCHEN[0], "CAMERAS": KITCHEN[1], "DRY": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "CASCADE_OCCUPANCY": "0"}
    script = prelude + _function("freeze_note_library") + _function("learned_notes") + call
    return subprocess.run(["bash", "-c", script + "\necho CONTINUED\n"], env={**base, **env},
                          capture_output=True, text=True, timeout=180, check=False)


def test_launch_freezes_exports_and_learned_md_reads_the_copy(tmp_path):
    lib, ws = _messy_library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    frozen = tmp_path / "state" / "aspire-library"
    result = _run_launch(tmp_path, f'freeze_note_library\necho "LIB=$CASCADE_ASPIRE_LIBRARY"\n'
                                   f'learned_notes "{ws}"', CASCADE_ASPIRE_LIBRARY=str(lib))
    assert result.returncode == 0 and "CONTINUED" in result.stdout, result.stderr
    assert f"LIB={frozen}\n" in result.stdout
    assert f"[launch] learned notes: froze {lib} into {frozen}: 2 note files" in result.stdout
    assert sorted(p.name for p in frozen.iterdir()) == ["a-note.md", "b-note.md"]
    assert f"library_sha={library_sha(frozen)}" in (ws / "LEARNED.md").read_text().split("\n", 1)[0]


def test_launch_exports_the_copy_to_every_later_reader(tmp_path):
    # no CASCADE_ASPIRE_LIBRARY in the launch environment: the kitchen default (~/.cascade/aspire/sim/library)
    # is frozen, and only the exported variable can point the later readers at the copy
    canonical, ws = _library(tmp_path / ".cascade/aspire/sim/library"), tmp_path / "ws"
    ws.mkdir()
    frozen = tmp_path / "state" / "aspire-library"
    call = (f'freeze_note_library\nlearned_notes "{ws}"\n'
            '"$PY" -c "import os; print(\'CHILD=\' + os.environ[\'CASCADE_ASPIRE_LIBRARY\'])"\n')
    result = _run_launch(tmp_path, call)
    assert result.returncode == 0 and "CONTINUED" in result.stdout, result.stderr
    assert f"CHILD={frozen}\n" in result.stdout and f"froze {canonical} into {frozen}" in result.stdout
    header = (ws / "LEARNED.md").read_text().split("\n", 1)[0]
    assert f"library_sha={library_sha(frozen)}" in header and library_sha(frozen) == library_sha(canonical)


@pytest.mark.parametrize("failure", ["helper_refuses", "python_broken"])
def test_a_failed_freeze_turns_notes_off_with_a_warning_and_the_launch_goes_on(tmp_path, failure):
    lib, ws = _library(tmp_path / "lib"), tmp_path / "ws"
    ws.mkdir()
    env = {"CASCADE_ASPIRE_LIBRARY": str(lib)}
    first = _run_launch(tmp_path, "freeze_note_library", **env)
    assert first.returncode == 0 and (tmp_path / "state" / "aspire-library" / "a-note.md").exists()
    if failure == "python_broken":
        broken = tmp_path / "python"
        broken.write_text("#!/bin/sh\necho 'interpreter exploded' >&2\nexit 3\n")
        broken.chmod(0o755)
        env["PY"] = str(broken)
    else:
        env["CASCADE_ASPIRE_LIBRARY"] = "relative/library"
    result = _run_launch(tmp_path, 'freeze_note_library\necho "LIB=$CASCADE_ASPIRE_LIBRARY"', **env)
    assert result.returncode == 0 and "CONTINUED" in result.stdout, result.stderr
    assert result.stderr.count("[launch] WARNING: learned notes:") == 1 and "notes off" in result.stderr
    assert ("interpreter exploded" if failure == "python_broken" else "relative path") in result.stderr
    off = Path(re.search(r"^LIB=(.*)$", result.stdout, re.MULTILINE).group(1))
    assert off.parent == tmp_path / "state" and off.is_dir() and not off.is_symlink() and list(off.iterdir()) == []
    assert off != tmp_path / "state" / "aspire-library"                            # never the earlier copy


def test_launch_freeze_dry_run_prints_and_creates_nothing(tmp_path):
    result = _run_launch(tmp_path, "freeze_note_library", DRY="1", CASCADE_ASPIRE_LIBRARY=str(tmp_path / "lib"))
    assert result.returncode == 0 and "CONTINUED" in result.stdout
    assert "cascade.skills.learned_block --freeze" in result.stdout
    assert list((tmp_path / "state").iterdir()) == []


def test_launch_freezes_in_the_spark_branch_before_any_reader():
    source = LAUNCH.read_text()
    call = source.index('if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then\n    freeze_note_library\nfi\n')
    assert source.index('STATE_DIR="$(ownerctl state-dir)"') < call < source.index('MCP_JSON="$(')
    assert call < source.index("\nruntime_check() {") and call < source.index('learned_notes "$PROFILE_WORKSPACE"')
    assert call < source.index('. "$REPO/scripts/launch_openshell.sh"')
    function = _function("freeze_note_library")
    assert "die" not in function and "export CASCADE_ASPIRE_LIBRARY=" in function
    assert re.search(r"--freeze", (ROOT / "scripts/launch_openshell.sh").read_text()) is None


def test_without_a_robot_proof_the_frozen_copy_is_named_instead_of_a_warning():
    source = LAUNCH.read_text()
    check = source.index('learned_notes "$LEARNED_WORKSPACE" --check-run')
    tail = source[check:source.index("    judge_pass\nfi\n")]
    assert "WARNING" not in tail and "warn " not in tail and "CASCADE_ASPIRE_LIBRARY" in tail
