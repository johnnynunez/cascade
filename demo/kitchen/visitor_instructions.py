"""Maintain only the demo's instruction block in the selected agent workspace."""
from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


SOURCE_PATH = Path(__file__).with_name('visitor-instructions.md')
BEGIN_MARKER = '<!-- BEGIN CASCADE KITCHEN VISITOR INSTRUCTIONS -->'
END_MARKER = '<!-- END CASCADE KITCHEN VISITOR INSTRUCTIONS -->'


def sync_visitor_instructions(workspace: str | Path, *, source: Path = SOURCE_PATH) -> bool:
    """Return whether AGENTS.md changed; never inspect OpenClaw credentials.

    The caller supplies the effective main-agent workspace from configuration.
    Preserve all bytes outside our unique managed block and avoid rewriting an
    unchanged file. Malformed markers need review rather than a guessed repair.
    """
    if not isinstance(workspace, (str, Path)) or not str(workspace).strip():
        raise ValueError('An explicit main-agent workspace is required')
    folder = Path(workspace).expanduser()
    if not folder.is_absolute():
        raise ValueError('The main-agent workspace must be absolute')
    target = folder / 'AGENTS.md'
    if target.is_symlink():
        raise ValueError('The main-agent AGENTS.md must not be a symlink')
    instructions = Path(source).read_text(encoding='utf-8').strip()
    if not instructions or BEGIN_MARKER in instructions or END_MARKER in instructions:
        raise ValueError('Visitor instructions must be nonempty and contain no managed markers')
    block = (BEGIN_MARKER + '\n' + instructions + '\n' + END_MARKER).encode('utf-8')
    before = target.read_bytes() if target.exists() else b''
    begin, end = BEGIN_MARKER.encode(), END_MARKER.encode()
    if begin not in before and end not in before:
        after = block + b'\n\n' + before
    else:
        if before.count(begin) != 1 or before.count(end) != 1:
            raise ValueError('AGENTS.md has incomplete or duplicate visitor instruction markers')
        start, finish = before.index(begin), before.index(end)
        if finish < start:
            raise ValueError('AGENTS.md visitor instruction markers are out of order')
        after = before[:start] + block + before[finish + len(end):]
    if after == before:
        return False

    folder.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o600
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.AGENTS.md.', delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), mode)
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True
