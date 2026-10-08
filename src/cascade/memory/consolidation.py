"""Action<->object consolidation over tier-2 experience (ROADMAP #7).

``ExperienceMemory`` keys every habit and recipe on the INSTRUCTION TEXT, so
what the robot learned about one action on one object is scattered over every
wording that ever reached it: "put the red cube in the bin", "bin the red cube,
please" and a sub-goal "pick up the red cube" are three records that know
nothing of each other. This store consolidates the same outcome stream by what
was DONE to WHAT: ``(skill, object)`` -> wins / losses / the wordings that led
there.

Rules, each one deliberate:

* One credit per EXECUTED call. The orchestrator reports each plan once
  (fast tier: after it ran; LLM tier: a verified recipe), never again per
  curriculum sub-goal -- ``FastPlanner.note_subgoal_outcome`` re-credits the
  same execution under the clause text, and counting that too would double
  every compound task. In a failed plan the calls before the failing one are
  credited as wins, the failing call as a loss, and calls that never ran get
  nothing.
* Motion skills only, and only a plain string object argument (``object``,
  ``label`` or ``query``; for ``place_on_object`` that is the destination). A
  recipe's symbolic ``$target`` query is not an object name and is skipped.
* Object identity is the normalized LABEL (lower-case content words; plural
  and inflections share a key, so "red cubes" is an alias of "red cube").
  It is deliberately NOT merged by embedding similarity: in a text or image
  embedding "red cube" and "blue cube" are near neighbours, and merging them
  would credit one object's outcomes to the other.

Advisory only: the digest tells the planner what happened on this rig before;
it never vetoes, reorders or confirms anything -- the harness still refuses
motion and the verifiers still decide outcomes.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from .embedder import content_tokens

#: Argument names that carry the object a skill acts on, in priority order.
OBJECT_ARGS = ("object", "label", "query")

#: Distinct instruction wordings kept per pair (the count keeps growing).
MAX_INSTRUCTIONS = 16

_DISPLAY_STOPWORDS = frozenset("the a an this that those these my your el la los las un una".split())


def _motion_skills() -> frozenset:
    from ..skills.runtime import _MOTION_SKILLS  # lazy: runtime imports memory

    return frozenset(_MOTION_SKILLS)


def display_name(text: str) -> str:
    """'the Red Cube' -> 'red cube' (surface form, articles dropped)."""
    words = [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _DISPLAY_STOPWORDS]
    return " ".join(words)


def object_key(text: str) -> str:
    """Identity of an object name: its stemmed content words, in order."""
    return " ".join(content_tokens(text))


def call_object(args) -> str | None:
    """The plain object name a call acts on, or None (symbolic/absent)."""
    if not isinstance(args, dict):
        return None
    for key in OBJECT_ARGS:
        val = args.get(key)
        if isinstance(val, str) and object_key(val):
            return val
    return None


class ActionObjectMemory:
    """Outcome counts per (skill, object), persisted as JSON (atomic writes)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else None
        self._rows: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            try:
                for row in json.loads(self.path.read_text()):
                    key = (str(row["action"]), object_key(row["object"]))
                    self._rows[key] = {
                        "action": str(row["action"]),
                        "object": str(row["object"]),
                        "aliases": [str(a) for a in row.get("aliases", [])],
                        "wins": int(row.get("wins", 0)),
                        "losses": int(row.get("losses", 0)),
                        "instructions": int(row.get("instructions", 0)),
                        "wordings": [str(w) for w in row.get("wordings", [])][:MAX_INSTRUCTIONS],
                        "updated": float(row.get("updated", 0.0)),
                    }
            except Exception:  # noqa: BLE001 -- a corrupt file is an empty memory, never a crash
                self._rows = {}

    # ── recording ────────────────────────────────────────────────────────

    def record_plan(self, calls, success: bool, *, instruction: str = "",
                    failed_at: int | None = None) -> int:
        """Credit one EXECUTED plan; returns how many (skill, object) credits
        were recorded. ``failed_at`` is the index of the call that failed (the
        ones before it ran and succeeded, the ones after never ran); a failure
        without it credits a loss to every pair, the conservative reading."""
        motion = _motion_skills()
        outcomes: list[tuple[str, str, bool]] = []
        for i, call in enumerate(list(calls)):
            try:
                skill, args = call[0], call[1]
            except (TypeError, IndexError, KeyError):
                continue
            if success:
                won = True
            elif failed_at is None:
                won = False
            elif i < failed_at:
                won = True
            elif i == failed_at:
                won = False
            else:
                break  # never executed
            obj = call_object(args)
            if str(skill) in motion and obj is not None:
                outcomes.append((str(skill), obj, won))
        if not outcomes:
            return 0
        wording = " ".join(str(instruction).split()).lower()
        with self._lock:
            for skill, obj, won in outcomes:
                key = (skill, object_key(obj))
                name = display_name(obj)
                row = self._rows.get(key)
                if row is None:
                    row = self._rows[key] = {
                        "action": skill, "object": name, "aliases": [], "wins": 0, "losses": 0,
                        "instructions": 0, "wordings": [], "updated": 0.0,
                    }
                elif name != row["object"] and name not in row["aliases"]:
                    row["aliases"].append(name)
                row["wins" if won else "losses"] += 1
                if wording and wording not in row["wordings"]:
                    row["instructions"] += 1
                    if len(row["wordings"]) < MAX_INSTRUCTIONS:
                        row["wordings"].append(wording)
                row["updated"] = time.time()
            snapshot = json.dumps(self._sorted_rows(), indent=1)
        self._save(snapshot)
        return len(outcomes)

    def _save(self, snapshot: str) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                fh.write(snapshot)
            os.replace(tmp, self.path)
        except OSError:
            pass  # memory bookkeeping never fails a finished task

    # ── reading ──────────────────────────────────────────────────────────

    def _sorted_rows(self) -> list[dict]:
        rows = [dict(r, aliases=list(r["aliases"]), wordings=list(r["wordings"]))
                for r in self._rows.values()]
        rows.sort(key=lambda r: (-(r["wins"] + r["losses"]), r["action"], r["object"]))
        return rows

    def stats(self) -> list[dict]:
        """Every pair, most evidence first: ``{"action", "object", "aliases",
        "wins", "losses", "instructions", "wordings", "updated"}``."""
        with self._lock:
            return self._sorted_rows()

    def lookup(self, action: str, obj: str) -> dict | None:
        with self._lock:
            row = self._rows.get((str(action), object_key(obj)))
            return None if row is None else dict(row)

    def relevant(self, task: str, max_rows: int = 6) -> list[dict]:
        """Pairs whose object (or an alias) is named in `task`: every stemmed
        content word of the object name appears among the task's."""
        words = set(content_tokens(task))
        out = []
        for row in self.stats():
            names = [row["object"], *row["aliases"]]
            if any(set(content_tokens(n)) and set(content_tokens(n)) <= words for n in names):
                out.append(row)
        return out[:max_rows]

    def agent_digest(self, task: str, max_rows: int = 6) -> str:
        """Advisory text for the LLM tier; '' when nothing named in `task`."""
        rows = self.relevant(task, max_rows=max_rows)
        if not rows:
            return ""
        lines = [
            "Action-object outcomes recorded on this rig for objects named in this task "
            "(consolidated across instruction wordings; advisory history, not a prediction -- "
            "the safety harness and the verifiers still decide):"
        ]
        for r in rows:
            alias = f" (also called: {', '.join(r['aliases'])})" if r["aliases"] else ""
            lines.append(
                f"- {r['action']} x {r['object']}: {r['wins']} succeeded, {r['losses']} failed "
                f"over {r['instructions']} instruction wording(s){alias}"
            )
        return "\n".join(lines)
