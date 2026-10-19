"""The learning surface (ASPIRE spec §4.8): one chat line and `learning.json` for the panel.

SIMULATION ONLY. A runtime builds a `LearningState` only for a sim-class profile
with `agent.learning_surface: true` (the kitchen profile, `learning_state_for`);
a real profile never builds one, whatever its switch says (B7 decides the
reBot), so every hardware payload stays main's.

The chat line. A motion result may carry ONE line the runtime grants
(`learning_line`), with the instruction "End your reply with this exact line:
<line>" (`with_learning_line`): the kitchen rules' R5 ends the reply with
exactly that line, and only when a tool result says so. The runtime is the only
writer: it appends the instruction to the granted result AFTER the rule-trigger
scrub (`runtime._with_retry_text`), which withholds R5's words from every other
string the agent reads -- an operator's halt reason, an object name, a nested
step, world_state, the memory events -- so no text the agent supplied can ever
read as the grant. Priority:

1. THE retry of this order (the sim retry ledger's own) placed the object:
   RETRY_LINE when the refused attempt was a gate refusal whose refused grasps
   the retry excluded, else RETRY_LINE_FRESH_LOOK (preflight R30: no grasp was
   refused after a perception or planning refusal);
2. the call succeeded and a served note's `when` words name every word of the
   order's object (`matched_note`, R30: one shared word such as "green" is no
   match): NOTE_LINE with the note's title as the notes routes render it and its
   provenance tag.

"Succeeded" means `ok` is True, the outcome is not stuck and the result is not
an unverified placement (`verified: false`: the agent must start with
"Placement unverified."). A call an operator stop met gets no line (the
runtime's rule: latched at its end, a halt since the skill started, or the rig's
stop token changed during it). No line on failures, on non-motion tools, or
with the switch off.

`learning.json` (schema 1, at most MAX_BYTES): written atomically (a sibling
temporary file, then `os.replace`) at build and after each learning event, to
the run directory and, when the MCP server has one, its launch state, which the
Spark camera surface serves to the extension panel. Writing never fails a call:
a path that cannot be written is logged and skipped.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from ..sim_profile import learning_surface_enabled, profile_class
from ..skills import notes
from .sim_retry import _word_set, goal_of

logger = logging.getLogger(__name__)

#: spec §4.8 text 1: a retry placed the object after a gate refusal whose refused grasps it excluded
RETRY_LINE = "Learning loop: my first grasp was refused, so I retried once in simulation with a different grasp."
#: preflight R30: the retry after a perception or planning refusal (no grasp was refused)
RETRY_LINE_FRESH_LOOK = ("Learning loop: my first attempt was refused, so I retried once in simulation after a "
                         "fresh look.")
#: spec §4.8 text 2
NOTE_LINE = "From earlier runs: {title} ({tag})."
#: the one instruction that grants the reply line (R5 quotes it)
REPLY_LINE_INSTRUCTION = "End your reply with this exact line: {line}"
#: learning.json
LEARNING_FILE = "learning.json"
SCHEMA = 1
MAX_BYTES = 64 * 1024
#: the most served notes learning.json lists (robot_knowledge's own count)
SERVED_SHOWN = notes.KNOWLEDGE_SHOWN


def succeeded(result) -> bool:
    """A success the agent may report as one: ok, its outcome ok (never stuck or failed), not an
    unverified placement."""
    return (isinstance(result, Mapping) and result.get("ok") is True and result.get("outcome") in (None, "ok")
            and result.get("verified") is not False)


def matched_note(snapshot, name, args) -> notes.Note | None:
    """The best-ranked served note whose `when` words contain every word of the order's object
    (`goal_of`: pick_and_place's object, grasp_object's label), else None (R30)."""
    if snapshot is None:
        return None
    goal = goal_of(name, args)
    if goal is None:
        return None
    for note in snapshot.served:
        if goal.words <= (_word_set(note.when) or frozenset()):
            return note
    return None


def learning_line(*, retried_placed: bool, result, matched: notes.Note | None, sha: str,
                  grasps_refused: bool = False) -> str | None:
    """The one line this result grants, or None (priority: the retry, then a matched note)."""
    if not succeeded(result):
        return None
    if retried_placed:
        return RETRY_LINE if grasps_refused else RETRY_LINE_FRESH_LOOK
    if matched is None:
        return None
    return NOTE_LINE.format(title=notes._ascii(matched.title), tag=notes.note_tag(matched, sha))


def with_learning_line(result, line: str | None):
    """`result` with `learning_line` set and the granting instruction appended to `next_action`
    (once); unchanged when `line` is None."""
    if line is None or not isinstance(result, dict):
        return result
    instruction = REPLY_LINE_INSTRUCTION.format(line=line)
    own = str(result.get("next_action") or "").strip()
    if instruction not in own:
        result["next_action"] = f"{own} {instruction}" if own else instruction
    result["learning_line"] = line
    return result


def _counts(snapshot) -> dict:
    parsed = tuple(getattr(snapshot, "notes", ()) or ())
    return {"promoted": sum(n.status == "promoted" for n in parsed),
            "verified": sum(notes._verified(n.verifier) is not None for n in parsed),
            "candidates": sum(n.status == "candidate" for n in parsed),
            "refuted": sum(n.status == "refuted" or notes._refuted(n.verifier) for n in parsed)}


class LearningState:
    """What the panel shows (spec §4.8): the library identity, the served notes and the last
    learning event, written to every path in `paths` (`write`). Thread-safe."""

    def __init__(self, paths, snapshot, sim_retry: bool):
        self._paths = [Path(p) for p in paths]
        self._snapshot = snapshot
        self._sim_retry = bool(sim_retry)
        self._lock = threading.Lock()
        self._last_event: dict | None = None

    @property
    def snapshot(self):
        return self._snapshot

    def add_path(self, path) -> None:
        """Also write to `path` from now on, starting now."""
        with self._lock:
            if Path(path) not in self._paths:
                self._paths.append(Path(path))
        self.write()

    def payload(self) -> dict:
        """learning.json's content (schema 1)."""
        snapshot = self._snapshot
        sha = str(getattr(snapshot, "sha", ""))
        served = [{"id": notes._shown_name(note.id), "title": notes._ascii(note.title),
                   "tag": notes.note_tag(note, sha)}
                  for note in tuple(getattr(snapshot, "served", ()) or ())[:SERVED_SHOWN]]
        with self._lock:
            event = None if self._last_event is None else dict(self._last_event)
        return {"schema": SCHEMA, "library_sha": sha,
                "profile_class": getattr(snapshot, "profile_class", "real"), "sim_retry": self._sim_retry,
                "served": served, "counts": _counts(snapshot), "last_event": event}

    def write(self, event: dict | None = None) -> None:
        """Record `event` (when given) as the last event, then write learning.json to every path,
        atomically. Never raises: a path that cannot be written is logged and skipped."""
        try:
            with self._lock:
                if event is not None:
                    self._last_event = dict(event)
                paths = list(self._paths)
            data = json.dumps(self.payload(), ensure_ascii=True, sort_keys=True).encode("ascii")
            if len(data) > MAX_BYTES:
                logger.warning("learning.json would be %d bytes (max %d); not written", len(data), MAX_BYTES)
                return
        except Exception:  # noqa: BLE001 -- the panel never fails a call
            logger.exception("learning.json not rendered")
            return
        for path in paths:
            _write_atomic(path, data)


def _write_atomic(path: Path, data: bytes) -> None:
    temp = None
    try:
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.chmod(temp, 0o644)
        os.replace(temp, path)
        temp = None
    except OSError as exc:
        logger.warning("learning.json not written to %s: %s", path, exc)
    finally:
        if temp is not None:
            try:
                os.unlink(temp)
            except OSError:
                pass


def learning_event(kind: str, line: str | None, outcome: str) -> dict:
    """A `last_event` entry: `{t, kind: retry|note, line, outcome}`."""
    return {"t": time.time(), "kind": kind, "line": line, "outcome": outcome}


def learning_state_for(cfg, snapshot, sim_retry, run_dir) -> LearningState | None:
    """The runtime's learning state: only for a sim-class profile with `agent.learning_surface:
    true` (a real profile never gets one, whatever the switch says), written at once to
    `<run_dir>/learning.json`; None otherwise, and None when anything about it fails."""
    try:
        if profile_class(cfg) != "sim" or not learning_surface_enabled(cfg):
            return None
        state = LearningState([Path(run_dir) / LEARNING_FILE], snapshot, sim_retry is not None)
        state.write()
        return state
    except Exception as exc:  # noqa: BLE001 -- fail closed: no surface
        logger.warning("learning surface off: %s: %s", type(exc).__name__, exc)
        return None
