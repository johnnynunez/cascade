"""The kitchen prompt's learned notes: ``<workspace>/LEARNED.md`` (ASPIRE spec §4.3).

The kitchen prompt plugin (``demo/kitchen/openclaw-plugin/index.mjs``) injects this file
right after the workspace ``AGENTS.md`` on every turn; no ``AGENTS.md`` is ever edited
here. The file is generated whole: ONE header line naming the library's sha, the sha256
of the block it renders and the ids of the block's notes, then exactly A1's prompt block
(`LibrarySnapshot.block()`, the linted text every other route reads), never cut or
re-rendered here. Nothing to render -> no file: one this module wrote earlier is removed,
so the prompt is byte-identical to main's. Only a regular file whose first line starts
with `LEARNED_HEADER_PREFIX` counts as ours: an operator's file, a symlink or anything
else is never overwritten or removed (`ValueError` naming the path; the plugin ignores
such a file anyway). Every file here is read with A1's one safe reader
(`notes.read_regular`).

At launch (``scripts/launch.sh``, spark branch) `freeze_library` first copies the note
files of the library the robot server would read into the launch's state directory, and
the launcher points CASCADE_ASPIRE_LIBRARY at that copy: every robot server of the launch
and this generator read the same bytes, whatever the library does meanwhile. The
generator takes the snapshot the robot server takes (same configuration, same library)
and prints the sha it rendered; after the launch proof bound that server,
`check_learned_notes` still compares the header with the server's ``run.json`` (an
assertion: any difference removes the file and reports it). A failure never stops the
launch (preflight R26): the command exits non-zero with a one-line message, having
removed only its own file (or written no copy), and the kitchen runs without notes.

    python -m cascade.skills.learned_block --workspace <abs dir> [--check-run <run dir>]
    python -m cascade.skills.learned_block --freeze <abs dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

from . import notes
from .notes import LEARNED_HEADER_PREFIX

LEARNED_FILE = "LEARNED.md"
#: a run.json larger than this is no run record (the robot server writes ~1-3 KB)
RUN_JSON_MAX_BYTES = 1 << 20
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_HEADER_RE = re.compile(re.escape(LEARNED_HEADER_PREFIX)
                        + r"library_sha=([0-9a-f]{64}) block_sha=([0-9a-f]{64}) notes=(\S+) -->")
#: the variable every kitchen profile reads its note library from (configs/arms/isaac_kitchen.yaml)
LIBRARY_VARIABLE = "CASCADE_ASPIRE_LIBRARY"


def _workspace(workspace) -> Path:
    if not isinstance(workspace, (str, os.PathLike)) or not str(workspace).strip():
        raise ValueError("an explicit absolute kitchen workspace is required")
    folder = Path(workspace).expanduser()
    if not folder.is_absolute():
        raise ValueError(f"the kitchen workspace must be absolute, not {str(workspace)!r}")
    return folder


def _read_regular(path: Path, limit: int) -> bytes | None:
    """At most `limit` bytes of the regular file at `path` (`notes.read_regular`);
    None when nothing is there. ValueError for a symlink (never followed) or anything
    but a regular file, judged on the opened descriptor (a FIFO is never read)."""
    try:
        return notes.read_regular(path, limit, whole=False)
    except FileNotFoundError:
        return None
    except notes.UnsafeFile as exc:
        if exc.reason == "symlink":
            raise ValueError(f"{path} is a symlink: never followed, overwritten or removed") from None
        raise ValueError(f"{path} is not a regular file: never overwritten or removed") from None


def _ours(data: bytes | None) -> bool:
    return data is not None and data.startswith(LEARNED_HEADER_PREFIX.encode("ascii"))


def _header(data: bytes) -> tuple[str, str, list[str]] | None:
    """(library_sha, block_sha, note ids) of a well-formed header line, else None."""
    first = data.split(b"\n", 1)[0]
    try:
        match = _HEADER_RE.fullmatch(first.decode("ascii"))
    except UnicodeDecodeError:
        return None
    return None if match is None else (match.group(1), match.group(2), match.group(3).split(","))


def _body_sha(data: bytes) -> str:
    """sha256 of the block a LEARNED.md carries: everything after its header line, without
    the one final newline the generator adds."""
    body = data.split(b"\n", 1)[1] if b"\n" in data else b""
    return hashlib.sha256(body.removesuffix(b"\n")).hexdigest()


def render_learned(snapshot) -> str:
    """The whole LEARNED.md for `snapshot`: the header line, A1's block verbatim and
    a final newline; "" when the snapshot renders no block. ValueError when the
    library sha is no sha256 hex digest, a rendered note's id is no safe note file
    name (A1's rule) or the file would exceed `notes.LEARNED_MAX_BYTES` (never cut:
    the block must stay the text every other route reads)."""
    block = snapshot.block()
    if not block:
        return ""
    manifest = snapshot.manifest()
    sha, ids, block_sha = snapshot.sha, manifest["block_notes"], manifest["block_sha"]
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"library_sha {sha!r} is no sha256 hex digest")
    unsafe = [note_id for note_id in ids if not notes.is_note_id(note_id)]
    if not ids or unsafe:
        raise ValueError(f"note ids {unsafe!r} are no safe note file names; never written into the header")
    text = f"{LEARNED_HEADER_PREFIX}library_sha={sha} block_sha={block_sha} notes={','.join(ids)} -->\n{block}\n"
    size = len(text.encode("utf-8"))
    if size > notes.LEARNED_MAX_BYTES:
        raise ValueError(f"{size} bytes, over the {notes.LEARNED_MAX_BYTES}-byte cap the plugin reads")
    return text


def _write(folder: Path, path: Path, data: bytes) -> None:
    """Temp file in the workspace, 0600, fsync, then `os.replace` onto `path`."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=f".{LEARNED_FILE}.", delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sync_learned_notes(workspace, snapshot) -> bool:
    """Make `<workspace>/LEARNED.md` the header + block of `snapshot` (no file when it
    renders no block); return whether the file changed. Raises ValueError (naming the
    path) for a relative workspace, for a LEARNED.md that is a symlink, no regular
    file or not ours (never touched), and when `render_learned` refuses the snapshot
    (then our earlier file is removed first: an older library's notes never stay)."""
    folder = _workspace(workspace)
    path = folder / LEARNED_FILE
    existing = _read_regular(path, notes.LEARNED_MAX_BYTES + 1)
    if existing is not None and not _ours(existing):
        raise ValueError(f"{path} was not written by cascade (no learned-notes header): never overwritten")
    try:
        text = render_learned(snapshot)
    except ValueError as exc:
        if existing is not None:
            path.unlink()
        raise ValueError(f"{path} not written: {exc}") from None
    if not text:
        if existing is None:
            return False
        path.unlink()
        return True
    data = text.encode("utf-8")
    if existing == data:
        return False
    _write(folder, path, data)
    return True


def _run_record(path: Path) -> tuple[str, str, list[str]]:
    """(library_sha, block_sha, block_notes) of the robot server's run.json; ValueError
    when it is missing, unreadable or malformed."""
    data = _read_regular(path, RUN_JSON_MAX_BYTES + 1)
    if data is None:
        raise ValueError("missing")
    if len(data) > RUN_JSON_MAX_BYTES:
        raise ValueError("too large")
    try:
        meta = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not JSON ({exc})") from None
    sha = meta.get("library_sha") if isinstance(meta, dict) else None
    block_sha = meta.get("block_sha") if isinstance(meta, dict) else None
    block = meta.get("block_notes") if isinstance(meta, dict) else None
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"no library_sha ({sha!r})")
    if not isinstance(block_sha, str) or _SHA_RE.fullmatch(block_sha) is None:
        raise ValueError(f"no block_sha ({block_sha!r})")
    if not isinstance(block, list) or not all(isinstance(note_id, str) for note_id in block):
        raise ValueError(f"no block_notes list ({block!r})")
    return sha, block_sha, block


def check_learned_notes(workspace, run_dir) -> tuple[bool, str]:
    """Compare `<workspace>/LEARNED.md` with the robot server's `<run_dir>/run.json`:
    (True, message) when the prompt shows exactly the block the server renders (same
    library_sha, block_sha and block notes, the file's block hashing to its header's
    block_sha; or no block on either side). Otherwise (False, warning): our file is
    removed whenever it does not match a readable run.json exactly; a file that is not
    ours is never touched (the plugin injects nothing from it)."""
    folder = _workspace(workspace)
    path = folder / LEARNED_FILE
    record = Path(run_dir) / "run.json"
    try:
        existing = _read_regular(path, notes.LEARNED_MAX_BYTES + 1)
    except ValueError:
        existing = None                      # a symlink or no regular file: never injected, never ours
    ours = _ours(existing)
    try:
        server_sha, server_block_sha, server_block = _run_record(record)
    except (OSError, ValueError) as exc:
        if ours:
            path.unlink()
            return False, (f"removed {path}: the robot server's {record} cannot be read ({exc}), so the notes "
                           "it serves cannot be confirmed; the kitchen prompt runs without learned notes")
        return False, f"the robot server's {record} cannot be read ({exc})"
    if ours:
        header = _header(existing)
        if header is not None and _body_sha(existing) != header[1]:
            path.unlink()
            return False, (f"removed {path}: its block does not hash to its header's block_sha={header[1]} (edited "
                           "after it was written); the kitchen prompt runs without learned notes")
        if header == (server_sha, server_block_sha, server_block):
            return True, f"matches {path} library_sha={server_sha}"
        path.unlink()
        shown = ("an unreadable header" if header is None
                 else f"library_sha={header[0]} block_sha={header[1]} notes={','.join(header[2])}")
        return False, (f"removed {path}: it rendered {shown}, but the robot server serves library_sha={server_sha} "
                       f"block_sha={server_block_sha} block_notes={','.join(server_block) or '(none)'} ({record}); "
                       "the kitchen prompt runs without learned notes")
    if server_block:
        return False, (f"the robot server's prompt block notes {','.join(server_block)} ({record}) are not in the "
                       f"kitchen prompt: {path} is missing or not cascade's")
    return True, f"no learned notes in {path} or in the robot server's block ({record})"


def _server_config():
    """The configuration the robot server builds (`apps.mcp_server._ensure_runtime`):
    CASCADE_CAMERAS (or CASCADE_CAMERA) and CASCADE_ARMS (or CASCADE_ARM), default
    mock, with the mock LLM."""
    from ..config import load_demo_config

    def names(primary: str, fallback: str) -> list[str]:
        return [n.strip() for n in os.environ.get(primary, os.environ.get(fallback, "mock")).split(",") if n.strip()]

    return load_demo_config(cameras=names("CASCADE_CAMERAS", "CASCADE_CAMERA"),
                            arms=names("CASCADE_ARMS", "CASCADE_ARM"), llm="mock")


def _library_with(value: str) -> Path:
    """The library the robot server's configuration names while `LIBRARY_VARIABLE` is
    `value` (`notes.library_root`)."""
    before = os.environ.get(LIBRARY_VARIABLE)
    os.environ[LIBRARY_VARIABLE] = value
    try:
        return notes.library_root(_server_config())
    finally:
        if before is None:
            os.environ.pop(LIBRARY_VARIABLE, None)
        else:
            os.environ[LIBRARY_VARIABLE] = before


def _discard(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _old_copies(dest: Path) -> list[Path]:
    """The copies `_swap_in` moved aside beside `dest` (`.<name>.old.*`)."""
    prefix = f".{dest.name}.old."
    try:
        with os.scandir(dest.parent) as entries:
            return [Path(entry.path) for entry in entries if entry.name.startswith(prefix)]
    except FileNotFoundError:
        return []


def _swap_in(temporary: Path, dest: Path) -> None:
    """Rename `temporary` onto `dest`; an earlier `dest` is moved aside first and removed
    after (put back when the rename fails)."""
    old = None
    if os.path.lexists(dest):
        old = dest.with_name(f".{dest.name}.old.{os.getpid()}.{os.urandom(4).hex()}")
        os.rename(dest, old)
    try:
        os.rename(temporary, dest)
    except BaseException:
        if old is not None:
            os.rename(old, dest)
        raise
    if old is not None:
        _discard(old)


def freeze_library(dest) -> str:
    """Copy the note library the robot server reads (the configuration of `_server_config`,
    `notes.library_root`) into the new directory `dest` and return the one-line report. Only
    the files A1's reader accepts and hashes are copied (`notes._scan`: a safe note name, a
    regular file, no symlink, at most `NOTE_MAX_BYTES`), byte for byte and read-only, into a
    temp directory beside `dest` that is renamed onto it (an earlier copy there is replaced
    whole); the copy hashes to the library's sha. Raises -- leaving `dest` as it was and no
    temp directory -- for a relative `dest`, a library `notes.config_snapshot` would not read
    (no path or a relative one, no directory, a symlink to a missing target, every note file
    unreadable), a configuration that does not read `LIBRARY_VARIABLE` (the copy would bind
    no robot server) and any read or write error. A library that does not exist yet freezes
    as an empty one (never created). Copies an earlier freeze moved aside and could not
    remove go first, whatever this freeze does: they never pile up across launches."""
    dest = Path(dest)
    if not dest.is_absolute():
        raise ValueError(f"the frozen library must be an absolute path, not {str(dest)!r}")
    for old in _old_copies(dest):
        _discard(old)
    root = notes.library_root(_server_config())
    if _library_with(str(dest)) != dest:
        raise ValueError(f"the robot server's profile does not read its note library from {LIBRARY_VARIABLE}, "
                         "so a frozen copy would bind none of its servers")
    try:
        with os.scandir(root):
            pass
        entries = notes._scan(root)
    except FileNotFoundError:
        link = notes._dangling_link(root)
        if link is not None:
            raise FileNotFoundError(f"{link} is a symlink to a missing target ({os.readlink(link)})") from None
        entries = []
    named = [why for name, _, why in entries if notes._is_note_name(name)]
    if named and all(why == "unreadable" for why in named):
        raise PermissionError(f"{root}: {notes.ALL_UNREADABLE} ({len(named)} files)")
    sha = notes._sha_of(entries)
    temporary = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
    try:
        for name, data, _ in entries:
            if data is None:
                continue
            with open(temporary / name, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o444)
        if notes.library_sha(temporary) != sha:
            raise OSError(f"the copy in {temporary} does not hash to library_sha={sha}")
        _swap_in(temporary, dest)
        temporary = None
    finally:
        if temporary is not None:
            _discard(temporary)
    copied = sum(data is not None for _, data, _ in entries)
    report = f"froze {root} into {dest}: {notes._counted(copied, 'note file')}, library_sha={sha}"
    left = Counter(why for _, data, why in entries if data is None)
    if left:
        total = sum(left.values())
        report += (f"; left out {total} {'entry' if total == 1 else 'entries'}: "
                   + ", ".join(f"{why} x{n}" for why, n in sorted(left.items())))
    return report


def _remove_ours(path: Path | None) -> bool:
    try:
        if path is not None and _ours(_read_regular(path, len(LEARNED_HEADER_PREFIX))):
            path.unlink()
            return True
    except (OSError, ValueError):
        pass
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cascade.skills.learned_block",
                                     description="Write (or check) the kitchen workspace's LEARNED.md, or freeze "
                                                 "the note library for one launch.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--workspace", help="absolute kitchen agent workspace (the one holding AGENTS.md)")
    mode.add_argument("--freeze", metavar="DIR",
                      help=f"copy the note library the robot server reads into DIR (absolute; replaced whole) "
                           f"for {LIBRARY_VARIABLE}")
    parser.add_argument("--check-run", metavar="RUN_DIR",
                        help="compare LEARNED.md with this robot server run's run.json; remove it on any difference")
    args = parser.parse_args(argv)
    if args.check_run is not None and args.workspace is None:
        parser.error("--check-run needs --workspace")
    if args.freeze is not None:
        try:
            print(freeze_library(args.freeze))
            return 0
        except Exception as exc:  # noqa: BLE001 -- preflight R26: one line, no traceback; launch.sh turns notes off
            print(f"{type(exc).__name__}: {exc}; the note library was not frozen", file=sys.stderr)
            return 1
    path = None
    try:
        folder = _workspace(args.workspace)
        path = folder / LEARNED_FILE
        if args.check_run is not None:
            ok, message = check_learned_notes(folder, Path(args.check_run))
            print(message)
            return 0 if ok else 1
        from .. import sim_profile

        cfg = _server_config()
        snapshot = notes.config_snapshot(cfg, sim_profile.profile_class(cfg))
        changed = sync_learned_notes(folder, snapshot)
        verb = ("wrote" if path.exists() else "removed") if changed else "unchanged"
        print(f"{verb} {path} library_sha={snapshot.sha}")
        return 0
    except Exception as exc:  # noqa: BLE001 -- preflight R26: never a traceback; notes off, the launch goes on
        removed = _remove_ours(path)
        print(f"{f'removed {path}: ' if removed else ''}{type(exc).__name__}: {exc}; "
              "the kitchen prompt runs without learned notes", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
