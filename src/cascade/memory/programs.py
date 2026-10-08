"""Program library: stored programs and their cross-task evidence (ROADMAP #8).

A program (``agent/programs.py``, design note ``docs/PROGRAMS_TIER.md``) is a
bounded, declarative list of registered tool calls. This module is where
programs are REMEMBERED, and the only place that decides which programs are
offered for reuse. It executes nothing and cannot verify anything: callers
hand it the verdict the independent checker produced for an execution (the
task-effects ledger, read by the program runner), and it refuses to admit
anything that is not CONFIRMED -- a program is never trusted because of who
wrote it.

Admission and promotion mirror the ASPIRE library (``skills/library.py``):

* ``admit()`` folds one VERIFIED execution into the record of its structural
  signature. ``occurrences`` counts distinct executions (idempotent per run
  id), ``source_tasks`` distinct task texts (case/whitespace-insensitive),
  ``origins`` how the evidence arrived (authored / distilled / reused).
* A record is PROMOTED once it recurred in >= ``min_tasks`` distinct tasks
  (default ``PROMOTION_MIN_TASKS`` = 2, upstream ASPIRE's rule); before that it
  is a stored CANDIDATE that is never offered. ``min_tasks=1`` is the explicit
  relaxation and must be configured; nothing lowers it silently.
* ``record_failure()`` counts an execution of an admitted program that did not
  verify (one per run). A record whose failures caught up with its verified
  executions is DEMOTED and no longer offered.
* ``retrievable(task)`` = promoted, not demoted, sharing a content word with
  the task, ranked by overlap then evidence (keyword match, like the ASPIRE
  notes). With a memory ``embedder`` (opt-in, ``memory.embedder``; B42) the
  promoted records are ranked by text-embedding cosine instead, with the
  skill library's floor-or-guard rule: a record qualifies at the embedder's
  ``text_floor`` OR on a shared content word, so an embedder can add a
  paraphrase the keywords missed but never drops a keyword match. The
  promotion gate comes first either way.

Persistence: one JSON object per line (``runs/programs.jsonl``), rewritten
atomically (temp file + ``os.replace``); a corrupt line costs only itself.
Several processes may share one store (the CLI, one MCP server per chat
session): every write re-reads the store under an advisory lock
(``<store>.lock``) before applying its change, so no process erases another's
records, occurrences or losses; ``refresh()`` adopts other processes' writes
for a reader. Programs never contain raw coordinates (``agent/programs.py``
refuses them at validation), so neither does this file.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIRMED = "confirmed"  # agent/effects.py CONFIRMED; kept literal so this store imports nothing heavy

#: Origins an execution can credit a record with.
ORIGINS = ("authored", "distilled", "reused")

_STOPWORDS = frozenset(
    "the and then with from into onto for that this its please you can could would "
    "what where how its are was".split()
)
_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")


def _task_key(task: str) -> str:
    """Same identity rule as skills/library.py: 'Pick the  cube' == 'pick the cube'."""
    return " ".join(str(task).split()).casefold()


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if len(w) >= 3 and w not in _STOPWORDS}


@dataclass
class ProgramRecord:
    """One stored program and the evidence behind it."""

    signature: str
    name: str
    program: dict
    occurrences: int = 0
    source_tasks: list[str] = field(default_factory=list)
    source_runs: list[str] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)
    losses: int = 0
    loss_runs: list[str] = field(default_factory=list)
    summary: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def n_tasks(self) -> int:
        """Distinct recorded tasks; an empty task never counts."""
        return len({_task_key(t) for t in self.source_tasks if str(t).strip()})

    def promoted(self, min_tasks: int) -> bool:
        return self.n_tasks >= int(min_tasks)

    def retrievable(self, min_tasks: int) -> bool:
        """Offered for reuse: promoted AND verified more often than it failed."""
        return self.promoted(min_tasks) and self.occurrences > self.losses

    def status(self, min_tasks: int) -> str:
        if not self.promoted(min_tasks):
            return "candidate"
        return "promoted" if self.occurrences > self.losses else "demoted"

    def vocabulary(self) -> set[str]:
        params = self.program.get("params") or {}
        text = " ".join([
            self.name.replace("-", " "),
            str(self.program.get("description") or ""),
            *[str(v) for v in params.values()],
            *self.source_tasks,
        ])
        return _words(text)

    def retrieval_texts(self) -> list[str]:
        """What a task is compared with under embedding retrieval: the
        program's own name + description, and each instruction it was
        verified on (the closest of them decides -- a long concatenation
        would dilute every match)."""
        own = f"{self.name.replace('-', ' ')} {self.program.get('description') or ''}".strip()
        return [t for t in (own, *self.source_tasks) if str(t).strip()]


class ProgramLibrary:
    """Persisted programs with ASPIRE-style cross-task promotion."""

    def __init__(self, path: str | Path | None = None, min_tasks: int | None = None):
        if min_tasks is None:
            from ..skills.library import PROMOTION_MIN_TASKS  # one constant for notes and programs

            min_tasks = PROMOTION_MIN_TASKS
        min_tasks = int(min_tasks)
        if min_tasks < 1:
            raise ValueError(f"min_tasks must be >= 1, got {min_tasks}")
        #: distinct tasks a program must be verified in before it is offered;
        #: 1 is the explicit single-task relaxation.
        self.min_tasks = min_tasks
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._records: dict[str, ProgramRecord] = {}
        #: (inode, mtime_ns, size) of the store as this instance last read or
        #: wrote it; another process's write changes it (see `refresh`)
        self._disk_sig: tuple | None = None
        if self.path is not None and self.path.exists():
            self._disk_sig = self._stat()
            self._records = self._read_disk()

    def _stat(self) -> tuple | None:
        if self.path is None:
            return None
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _read_disk(self) -> dict[str, ProgramRecord]:
        records: dict[str, ProgramRecord] = {}
        if self.path is None:
            return records
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = ProgramRecord(**json.loads(line))
            except Exception:  # noqa: BLE001 -- one corrupt line must not cost the others
                continue
            if not _SIGNATURE.match(str(rec.signature)) or not isinstance(rec.program, dict):
                continue
            records[rec.signature] = rec
        return records

    def refresh(self) -> bool:
        """Adopt the store's current content if another process wrote it
        since this instance last read or wrote it (a stat, nothing more, when
        it did not). True when the in-memory records were replaced. An
        unreadable store keeps the in-memory state."""
        if self.path is None:
            return False
        with self._io_lock:
            return self._adopt_disk()

    def _adopt_disk(self) -> bool:
        """`refresh` with the I/O lock already held."""
        try:
            sig = self._stat()
            if sig == self._disk_sig:
                return False
            records = self._read_disk() if sig is not None else {}
        except OSError:
            return False
        with self._lock:
            self._records = records
            self._disk_sig = sig
        return True

    @contextlib.contextmanager
    def _exclusive(self):
        """One read-modify-write at a time: across threads (the I/O lock)
        and, best effort, across processes (an advisory `flock` on
        `<store>.lock`; a host without fcntl or a read-only directory still
        serializes its own threads)."""
        with self._io_lock:
            handle = None
            if self.path is not None:
                try:
                    import fcntl

                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = open(self.path.with_name(self.path.name + ".lock"), "a")
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
            try:
                yield
            finally:
                if handle is not None:
                    handle.close()  # releases the flock

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> list[ProgramRecord]:
        with self._lock:
            return list(self._records.values())

    def get(self, key: str) -> ProgramRecord | None:
        """By signature or by (unique) name."""
        if not isinstance(key, str):
            return None
        with self._lock:
            rec = self._records.get(key)
            if rec is not None:
                return rec
            return next((r for r in self._records.values() if r.name == key), None)

    def admit(self, program: dict, signature: str, *, task: str, run: str, origin: str,
              verdict: str, summary: str = "") -> ProgramRecord:
        """Fold one VERIFIED execution into the record for ``signature``.

        ``verdict`` is the independent verdict of the execution (the program
        runner's, read from the task-effects ledger). Anything but CONFIRMED is
        refused: an unverified, refuted or failed execution is not evidence,
        whoever authored the program. The same ``run`` is counted once.
        """
        if verdict != CONFIRMED:
            raise ValueError(f"only a CONFIRMED execution is admitted, got {verdict!r}")
        if not _SIGNATURE.match(str(signature)):
            raise ValueError("signature must be a sha256 hex digest")
        if not isinstance(program, dict) or not program.get("steps"):
            raise ValueError("program must be a validated program spec")
        task = " ".join(str(task).split())
        run = str(run).strip()
        with self._exclusive():
            self._adopt_disk()  # another process may have written since this one read
            with self._lock:
                rec = self._records.get(signature)
                if rec is None:
                    name = self._unique_name(str(program.get("name") or "program"))
                    stored = json.loads(json.dumps(program))
                    stored["name"] = name
                    rec = ProgramRecord(signature=signature, name=name, program=stored,
                                        summary=str(summary or "")[:200])
                    self._records[signature] = rec
                elif run and run in rec.source_runs:
                    return rec  # same execution seen before: nothing new to count
                rec.occurrences += 1
                if task and _task_key(task) not in {_task_key(t) for t in rec.source_tasks}:
                    rec.source_tasks.append(task)
                if run:
                    rec.source_runs.append(run)
                if origin and origin not in rec.origins:
                    rec.origins.append(str(origin))
                if summary and not rec.summary:
                    rec.summary = str(summary)[:200]
                rec.updated = time.time()
                text = self._snapshot()
            self._write(text)
        return rec

    def record_failure(self, signature: str, *, run: str, reason: str = "") -> ProgramRecord | None:
        """Count one execution of an ADMITTED program that did not verify.

        Unknown signatures gain no record (only programs that worked at least
        once are remembered); the same run counts once.
        """
        run = str(run).strip()
        with self._exclusive():
            self._adopt_disk()  # the record may have been admitted by another process
            with self._lock:
                rec = self._records.get(signature)
                if rec is None:
                    return None
                if run and run in rec.loss_runs:
                    return rec
                rec.losses += 1
                if run:
                    rec.loss_runs.append(run)
                rec.updated = time.time()
                text = self._snapshot()
            self._write(text)
        return rec

    def retrievable(self, task: str, max_entries: int = 3, *, embedder=None,
                    min_sim: float | None = None) -> list[ProgramRecord]:
        """Promoted, non-demoted records relevant to ``task`` (see `ranked`).

        Candidates and demoted records stay on disk and out of every prompt,
        whatever their wording -- the same gate ``aspire.retrieve`` applies to
        skill notes.
        """
        return [rec for rec, _ in self.ranked(task, max_entries, embedder=embedder, min_sim=min_sim)]

    def ranked(self, task: str, max_entries: int = 3, *, embedder=None,
               min_sim: float | None = None) -> list[tuple[ProgramRecord, float | None]]:
        """(record, similarity) for the promoted, non-demoted records relevant
        to ``task``, best first.

        Without ``embedder``: records sharing a content word with the task,
        by overlap then evidence; similarity is None (the pre-B42 ranking).
        With one (opt-in ``memory.embedder``): the cosine between the task and
        the closest of the record's `retrieval_texts` in the embedder's TEXT
        space; a record qualifies at ``min_sim`` (default: the embedder's
        ``text_floor``) OR on a shared content word -- the skill library's
        floor-or-guard rule, so an embedder adds paraphrases and never drops
        a keyword match. The promotion gate is applied first, unchanged.
        """
        if embedder is not None:
            return self._ranked_by_embedding(task, max_entries, embedder, min_sim)
        words = _words(task)
        scored = []
        for rec in self.records():
            if not rec.retrievable(self.min_tasks):
                continue
            overlap = len(words & rec.vocabulary())
            if overlap:
                scored.append((-overlap, -rec.occurrences, rec.name, rec))
        scored.sort(key=lambda t: t[:3])
        return [(rec, None) for *_, rec in scored[: max(int(max_entries), 0)]]

    def _ranked_by_embedding(self, task: str, max_entries: int, embedder,
                             min_sim: float | None) -> list[tuple[ProgramRecord, float | None]]:
        import numpy as np

        floor = float(min_sim if min_sim is not None else getattr(embedder, "text_floor", 0.5))
        query = np.asarray(embedder.embed_text(task), dtype=np.float64)
        words = _words(task)
        scored = []
        for rec in self.records():
            if not rec.retrievable(self.min_tasks):
                continue
            sim = max((float(np.dot(query, np.asarray(embedder.embed_text(text), dtype=np.float64)))
                       for text in rec.retrieval_texts()), default=0.0)
            overlap = len(words & rec.vocabulary())
            if sim >= floor or overlap:
                scored.append((-sim, -overlap, -rec.occurrences, rec.name, rec))
        scored.sort(key=lambda t: t[:4])
        return [(rec, round(-neg_sim, 3)) for neg_sim, *_, rec in scored[: max(int(max_entries), 0)]]

    def listing(self, max_entries: int = 20) -> list[ProgramRecord]:
        """Every promoted, non-demoted record, most evidence first: what a
        chat host sees when it asks without an instruction to match."""
        recs = [r for r in self.records() if r.retrievable(self.min_tasks)]
        recs.sort(key=lambda r: (-r.n_tasks, -r.occurrences, r.name))
        return recs[: max(int(max_entries), 0)]

    def summary(self) -> dict:
        records = self.records()
        status = [r.status(self.min_tasks) for r in records]
        return {"programs": len(records), "promoted": status.count("promoted"),
                "candidates": status.count("candidate"), "demoted": status.count("demoted"),
                "min_tasks": self.min_tasks}

    # ── internals ───────────────────────────────────────────────────────────

    def _unique_name(self, name: str) -> str:
        taken = {r.name for r in self._records.values()}
        if name not in taken:
            return name
        n = 2
        while f"{name}-{n}" in taken:
            n += 1
        return f"{name}-{n}"

    def _snapshot(self) -> str:
        """JSONL of every record -- call with the lock held."""
        return "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in self._records.values())

    def _write(self, text: str) -> None:
        """Atomic rewrite of ``text`` -- call inside `_exclusive()`.

        The snapshot was taken after re-reading the store under the same
        lock, so the last writer writes the newest state of EVERY writer;
        retrieval only ever takes the record lock and never waits on the
        disk. An unwritable store keeps the in-memory state.
        """
        if self.path is None:
            return
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".programs-", suffix=".jsonl", dir=self.path.parent)
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            os.replace(tmp, self.path)
            sig = self._stat()
            with self._lock:
                self._disk_sig = sig
        except OSError:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
