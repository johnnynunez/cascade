"""The kitchen prompt's learned notes: ``<workspace>/LEARNED.md`` (ASPIRE spec §4.3).

The kitchen prompt plugin (``demo/kitchen/openclaw-plugin/index.mjs``) injects this file
right after the workspace ``AGENTS.md`` on every turn; no ``AGENTS.md`` is ever edited
here. The file is generated whole: ONE header line naming the library's sha and the ids
of the notes it renders, then exactly A1's prompt block (`LibrarySnapshot.block()`, the
linted text every other route reads), never cut or re-rendered here. Nothing to render ->
no file: one this module wrote earlier is removed, so the prompt is byte-identical to
main's. Only a regular file whose first line starts with `LEARNED_HEADER_PREFIX` counts
as ours: an operator's file, a symlink or anything else is never overwritten or removed
(`ValueError` naming the path; the plugin ignores such a file anyway).

At launch (``scripts/launch.sh``, spark branch) the generator takes the snapshot the
robot server will take (same configuration, same library) and prints the sha it
rendered. The robot server reads its library again when it builds, so after the launch
proof bound that server, `check_learned_notes` compares the header with the server's
``run.json``: any difference removes the file (the prompt never shows notes from another
library than the one ``robot_knowledge`` and the trace stamp come from) and reports it.
A failure never stops the launch (preflight R26): the command exits non-zero with a
message, having removed only its own file, and the kitchen runs without notes.

    python -m cascade.skills.learned_block --workspace <abs dir> [--check-run <run dir>]
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

from . import notes
from .notes import LEARNED_HEADER_PREFIX

LEARNED_FILE = "LEARNED.md"
#: a run.json larger than this is no run record (the robot server writes ~1-3 KB)
RUN_JSON_MAX_BYTES = 1 << 20
_SHA_RE = re.compile(r"[0-9a-f]{64}")
_HEADER_RE = re.compile(re.escape(LEARNED_HEADER_PREFIX) + r"library_sha=([0-9a-f]{64}) notes=(\S+) -->")


def _workspace(workspace) -> Path:
    if not isinstance(workspace, (str, os.PathLike)) or not str(workspace).strip():
        raise ValueError("an explicit absolute kitchen workspace is required")
    folder = Path(workspace).expanduser()
    if not folder.is_absolute():
        raise ValueError(f"the kitchen workspace must be absolute, not {str(workspace)!r}")
    return folder


def _read_regular(path: Path, limit: int) -> bytes | None:
    """At most `limit` bytes of the regular file at `path`; None when nothing is
    there. ValueError for a symlink (never followed) or anything but a regular file,
    judged on the opened descriptor (a FIFO is opened non-blocking, never read)."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(mode):
        raise ValueError(f"{path} is a symlink: never followed, overwritten or removed")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        if exc.errno == errno.ELOOP:          # swapped for a symlink after the lstat
            raise ValueError(f"{path} is a symlink: never followed, overwritten or removed") from None
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path} is not a regular file: never overwritten or removed")
        chunks, size = [], 0
        while size < limit and (chunk := os.read(fd, limit - size)):
            chunks.append(chunk)
            size += len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _ours(data: bytes | None) -> bool:
    return data is not None and data.startswith(LEARNED_HEADER_PREFIX.encode("ascii"))


def _header(data: bytes) -> tuple[str, list[str]] | None:
    """(library_sha, note ids) of a well-formed header line, else None."""
    first = data.split(b"\n", 1)[0]
    try:
        match = _HEADER_RE.fullmatch(first.decode("ascii"))
    except UnicodeDecodeError:
        return None
    return None if match is None else (match.group(1), match.group(2).split(","))


def render_learned(snapshot) -> str:
    """The whole LEARNED.md for `snapshot`: the header line, A1's block verbatim and
    a final newline; "" when the snapshot renders no block. ValueError when the
    library sha is no sha256 hex digest, a rendered note's id is no safe note file
    name (A1's rule) or the file would exceed `notes.LEARNED_MAX_BYTES` (never cut:
    the block must stay the text every other route reads)."""
    block = snapshot.block()
    if not block:
        return ""
    sha, ids = snapshot.sha, snapshot.manifest()["block_notes"]
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"library_sha {sha!r} is no sha256 hex digest")
    unsafe = [note_id for note_id in ids if not notes.is_note_id(note_id)]
    if not ids or unsafe:
        raise ValueError(f"note ids {unsafe!r} are no safe note file names; never written into the header")
    text = f"{LEARNED_HEADER_PREFIX}library_sha={sha} notes={','.join(ids)} -->\n{block}\n"
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


def _run_record(path: Path) -> tuple[str, list[str]]:
    """(library_sha, block_notes) of the robot server's run.json; ValueError when it
    is missing, unreadable or malformed."""
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
    block = meta.get("block_notes") if isinstance(meta, dict) else None
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"no library_sha ({sha!r})")
    if not isinstance(block, list) or not all(isinstance(note_id, str) for note_id in block):
        raise ValueError(f"no block_notes list ({block!r})")
    return sha, block


def check_learned_notes(workspace, run_dir) -> tuple[bool, str]:
    """Compare `<workspace>/LEARNED.md` with the robot server's `<run_dir>/run.json`:
    (True, message) when the prompt shows exactly the block the server renders
    (same library_sha and block notes, or no block on either side). Otherwise
    (False, warning): our file is removed whenever it does not match a readable
    run.json exactly; a file that is not ours is never touched (the plugin injects
    nothing from it)."""
    folder = _workspace(workspace)
    path = folder / LEARNED_FILE
    record = Path(run_dir) / "run.json"
    try:
        existing = _read_regular(path, notes.LEARNED_MAX_BYTES + 1)
    except ValueError:
        existing = None                      # a symlink or no regular file: never injected, never ours
    ours = _ours(existing)
    try:
        server_sha, server_block = _run_record(record)
    except (OSError, ValueError) as exc:
        if ours:
            path.unlink()
            return False, (f"removed {path}: the robot server's {record} cannot be read ({exc}), so the notes "
                           "it serves cannot be confirmed; the kitchen prompt runs without learned notes")
        return False, f"the robot server's {record} cannot be read ({exc})"
    if ours:
        header = _header(existing)
        if header == (server_sha, server_block):
            return True, f"matches {path} library_sha={server_sha}"
        path.unlink()
        shown = "an unreadable header" if header is None else f"library_sha={header[0]} notes={','.join(header[1])}"
        return False, (f"removed {path}: it rendered {shown}, but the robot server serves library_sha={server_sha} "
                       f"block_notes={','.join(server_block) or '(none)'} ({record}); "
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
                                     description="Write (or check) the kitchen workspace's LEARNED.md.")
    parser.add_argument("--workspace", required=True, help="absolute kitchen agent workspace (the one holding AGENTS.md)")
    parser.add_argument("--check-run", metavar="RUN_DIR",
                        help="compare LEARNED.md with this robot server run's run.json; remove it on any difference")
    args = parser.parse_args(argv)
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
