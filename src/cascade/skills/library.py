"""Markdown guidance storage and keyword retrieval for the ASPIRE library.

Entries carry a failure signature, when-to-apply keywords, strategy and
origin, plus a front-matter block of cross-task evidence counters
(``occurrences``, ``source_tasks``, ``source_runs``, ``status``). Files live
in <repo>/skills_library/<slug>.md. ``relevant(task)`` selects notes for
``agent.aspire.retrieve``, which the built-in orchestrator calls at task start
and which admits only PROMOTED entries.

Promotion follows upstream ASPIRE (``aspire/sim/cap/skills/library.py``): a
distilled entry is promoted once it recurs in at least ``PROMOTION_MIN_TASKS``
distinct tasks; an entry seen in a single task -- however many runs -- stays a
stored CANDIDATE, visible to ``entries()``/``relevant()`` but never injected
into the agent context. Entries without front matter (legacy files, manual
notes) count as one occurrence from an unknown task and are candidates.
``SkillLibrary(root, min_tasks=1)`` is the only relaxation and must be
configured explicitly. Admission of newly harvested retry evidence belongs to
``agent.aspire``; this store also accepts manual notes and does not validate
their claims or enforce robot/scene identity during retrieval.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Upstream ASPIRE promotes a distilled skill when it recurs beyond a single
#: task. One confirmed retry is an observed association in one task, not a
#: pattern; the second DISTINCT task is the first cross-task evidence.
PROMOTION_MIN_TASKS = 2

_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def _task_key(task: str) -> str:
    """Whitespace/case-insensitive identity: 'Pick the  cube' is 'pick the cube'."""
    return " ".join(str(task).split()).casefold()


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


@dataclass
class LibraryEntry:
    """One stored note with its cross-task evidence counters."""

    name: str
    text: str  # markdown body without the front matter
    occurrences: int = 1
    source_tasks: list[str] = field(default_factory=list)
    source_runs: list[str] = field(default_factory=list)
    min_tasks: int = PROMOTION_MIN_TASKS

    @property
    def n_tasks(self) -> int:
        """Distinct recorded tasks; unknown (empty) tasks never count."""
        return len({_task_key(t) for t in self.source_tasks if t.strip()})

    @property
    def promoted(self) -> bool:
        if self.min_tasks <= 1:
            # Explicitly configured legacy mode: one observation suffices,
            # including legacy notes whose task was never recorded.
            return self.occurrences >= 1
        return self.n_tasks >= self.min_tasks

    @property
    def status(self) -> str:
        return "promoted" if self.promoted else "candidate"

    def front_matter(self) -> str:
        lines = [
            f"occurrences: {json.dumps(int(self.occurrences))}",
            f"source_tasks: {json.dumps(self.source_tasks)}",
            f"source_runs: {json.dumps(self.source_runs)}",
            f"status: {json.dumps(self.status)}",
        ]
        return "---\n" + "\n".join(lines) + "\n---\n"


def split_front_matter(text: str) -> tuple[dict, str]:
    """-> (metadata, body). Missing or unreadable front matter -> ({}, text)."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        if not sep or not key.strip():
            continue
        try:
            meta[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            meta[key.strip()] = value.strip()
    return meta, text[m.end():]


ENTRY_TEMPLATE = """# {title}

- **Failure signature:** {failure}
- **When to apply:** {when}
- **Origin task:** {origin}
- **Recorded:** {date}

## Strategy

{strategy}
"""


class SkillLibrary:
    def __init__(self, root: str | Path, min_tasks: int = PROMOTION_MIN_TASKS):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        min_tasks = int(min_tasks)
        if min_tasks < 1:
            raise ValueError(f"min_tasks must be >= 1, got {min_tasks}")
        #: distinct tasks an entry must recur in before retrieval admits it.
        #: 1 is the explicit legacy mode (retrieve after one confirmed retry).
        self.min_tasks = min_tasks

    def _path(self, title: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "entry"
        return self.root / f"{slug}.md"

    def _parse(self, path: Path) -> LibraryEntry:
        meta, body = split_front_matter(path.read_text())
        occurrences = meta.get("occurrences", 1)
        if isinstance(occurrences, bool) or not isinstance(occurrences, int) or occurrences < 1:
            occurrences = 1  # legacy or unreadable counter: one observation
        return LibraryEntry(
            name=path.stem,
            text=body,
            occurrences=occurrences,
            source_tasks=_str_list(meta.get("source_tasks")),
            source_runs=_str_list(meta.get("source_runs")),
            min_tasks=self.min_tasks,
        )

    def add(self, title: str, failure: str, when: str, strategy: str, origin: str = "",
            *, source_task: str = "", source_run: str = "") -> Path:
        """Write a note, or fold new evidence into the note with the same slug.

        ``source_task`` / ``source_run`` identify the observation. A run that
        already contributed (same ``source_run``) is NOT counted again, so
        re-harvesting the same runs directory is idempotent; a new run adds one
        occurrence and, when its task is new, one distinct source task. The
        body is refreshed to the latest observation (as upstream ASPIRE does),
        the front matter keeps the aggregate. Status is written for human
        readers; retrieval recomputes it with the configured threshold.
        """
        path = self._path(title)
        source_task = " ".join(str(source_task).split())
        source_run = str(source_run).strip()
        if path.exists():
            entry = self._parse(path)
            if source_run and source_run in entry.source_runs:
                return path  # same evidence seen before: nothing new to count
            entry.occurrences += 1
            if source_task and _task_key(source_task) not in {_task_key(t) for t in entry.source_tasks}:
                entry.source_tasks.append(source_task)
            if source_run:
                entry.source_runs.append(source_run)
        else:
            entry = LibraryEntry(
                name=path.stem, text="", occurrences=1,
                source_tasks=[source_task] if source_task else [],
                source_runs=[source_run] if source_run else [],
                min_tasks=self.min_tasks,
            )
        entry.text = ENTRY_TEMPLATE.format(
            title=title,
            failure=failure,
            when=when,
            origin=origin or "(manual)",
            date=time.strftime("%Y-%m-%d"),
            strategy=strategy,
        )
        path.write_text(entry.front_matter() + entry.text)
        return path

    def records(self) -> list[LibraryEntry]:
        """Every stored note with its counters, promoted or candidate."""
        return [self._parse(p) for p in sorted(self.root.glob("*.md"))]

    def entries(self) -> list[tuple[str, str]]:
        """-> [(name, markdown), ...] -- the raw files, front matter included."""
        return [(p.stem, p.read_text()) for p in sorted(self.root.glob("*.md"))]

    def summary(self) -> dict:
        records = self.records()
        promoted = sum(1 for r in records if r.promoted)
        return {"entries": len(records), "promoted": promoted,
                "candidates": len(records) - promoted, "min_tasks": self.min_tasks}

    def relevant_entries(self, task: str, max_entries: int = 3,
                         promoted_only: bool = False) -> list[LibraryEntry]:
        """Naive keyword guard match against the 'When to apply' line.

        ``promoted_only`` applies the cross-task gate; it is what the agent
        context path uses. The default lists candidates too, for inspection
        and for manual notes consulted outside the agent loop.
        """
        task_words = set(re.findall(r"[a-z]+", task.lower()))
        scored = []
        for entry in self.records():
            if promoted_only and not entry.promoted:
                continue
            m = re.search(r"\*\*When to apply:\*\*(.+)", entry.text)
            guard_words = set(re.findall(r"[a-z]+", (m.group(1) if m else "").lower()))
            overlap = len(task_words & guard_words)
            if overlap:
                scored.append((overlap, entry))
        scored.sort(key=lambda t: -t[0])
        return [entry for _, entry in scored[:max_entries]]

    def relevant(self, task: str, max_entries: int = 3, promoted_only: bool = False) -> list[str]:
        """Matched note bodies (front matter stripped); see ``relevant_entries``."""
        return [e.text for e in self.relevant_entries(task, max_entries=max_entries,
                                                      promoted_only=promoted_only)]
