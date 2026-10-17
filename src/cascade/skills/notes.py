"""ASPIRE learned notes, format v2: parse, lint, provenance tag, library hash, snapshot.

A note stays one markdown file ``<library>/<slug>.md`` with the front matter
`skills.library` writes (``occurrences``, ``source_tasks``, ``source_runs``,
``status``). v2 adds ``schema``, ``kind``, ``learned_in``, ``verifier`` and
``grades``; a missing key takes its v1 default, so every v1 file is a v2 note
(spec §4.1). The body has ``# <title>``, ``- **When to apply:** <words>``,
``## Advice`` (what the agent reads) and ``## Evidence`` (for humans only, never
rendered). A v1 body has no ``## Advice``: its advice is the first paragraph of
``## Strategy``, cut at `ADVICE_MAX_CHARS`.

Notes are advice TEXT. This module renders them and nothing else: no value in a
note reaches a config, a tool argument or a gate. A note is served only when its
lint is clean (`lint_note`): every rendered field (title and advice) is short,
carries no URL, no geometric value, no "ignore/bypass ... safety/gate" phrasing,
and names only read-only tools (`READ_ONLY_TOOLS`) and the note's own skill --
never the stop or reset tools.

`load_snapshot` reads the library ONCE (it never creates or writes it) and keeps
that view for the session: the prompt block, `robot_knowledge` and the run stamp
all come from the same bytes, so a nightly rewrite never changes a running
session. What it serves depends on the profile class (`cascade.sim_profile`):
"sim" serves notes on their distinct-task count; "real" (and anything that is
not exactly "sim") serves only notes a verifier passed.
"""
from __future__ import annotations

import functools
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .library import _FRONT_MATTER_RE, _str_list, _task_key, split_front_matter

ADVICE_MAX_CHARS = 300
TITLE_MAX_CHARS = 80
BLOCK_MAX_CHARS = 1400
LEARNED_MAX_BYTES = 2000
KNOWLEDGE_MAX_CHARS = 6000
LEARNED_HEADER_PREFIX = "<!-- cascade-learned-notes v1 "
#: notes in the prompt block (the rest are only in robot_knowledge)
BLOCK_NOTES = 2

#: tools a note may never name, even as its own skill: they clear or raise a stop
#: latch, or reset the scene (which also clears the sim retry ledger)
FORBIDDEN_TOOLS = frozenset({"reset_scene", "reset_stop", "emergency_stop", "halt_motion"})
#: the tools that only look or read memory: besides its own skill, the only tools a
#: note may name. A tool missing here (a new one, a motion, a program run, a tool
#: that opens a port or ends the task) keeps the note out until it is listed.
READ_ONLY_TOOLS = frozenset({
    "analyze_scene", "annotated_view", "camera_snapshot", "count_objects", "describe_scene",
    "get_observation", "list_arms", "list_objects", "live_view_status", "localize_object",
    "locate_pixel", "preview_grasp", "probe_point", "recall_memory", "recall_step",
    "robot_knowledge", "verify_last_action", "world_state",
})

KINDS = frozenset({"learned", "authored"})
STATUSES = frozenset({"candidate", "promoted", "refuted", "retired"})
LEARNED_IN = frozenset({"sim", "real", "mixed", "unknown"})
GRADE_KEYS = ("useful", "misleading", "wrong", "not_applicable")
_NEVER_SERVED = frozenset({"refuted", "retired"})

_BLOCK_HEAD = "## Learned from earlier runs"
_ADVICE_ONLY = ("Advice only: tool results and safety refusals always win. "
                "Use a note only when it matches the order.")
_BLOCK_FOOT = ("If a note shaped what you did, the tool result tells you the one line to add "
               "(do not add your own).")

_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_WHEN_RE = re.compile(r"\*\*When to apply:\*\*(.*)")
_SKILL_RE = re.compile(r"\*\*Failure signature:\*\*\s*([a-z][a-z0-9_]*)")
_SECTION_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://|\bwww\.", re.IGNORECASE)
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_WORD_RE = re.compile(r"[a-z0-9]+")
_OVERRIDE_RE = re.compile(r"(?:ignor|bypass|disabl|overrid|skip)")
_GUARDED_RE = re.compile(r"(?:safety|refus|gate|check|limit|margin|stop)")
#: words between an override verb and what it would override
_OVERRIDE_REACH = 6
_GEOMETRY_RE = re.compile(
    r"\d(?:[\d.,]*\d)?\s*(?:mm|cm|m|deg|degrees?|°|rad|radians?|(?:milli|centi)?met(?:er|re)s?)(?![a-z])",
    re.IGNORECASE)


@dataclass(frozen=True)
class Note:
    """One parsed note. `skill` is the tool its failure signature names (the
    note's own skill, the only motion tool its text may name)."""

    id: str
    title: str
    when: str
    advice: str
    kind: str
    status: str
    learned_in: str
    n_tasks: int
    occurrences: int
    verifier: dict | None
    grades: dict
    skill: str = ""


# ── parsing ───────────────────────────────────────────────────────────────


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _cut(text: str, limit: int) -> str:
    """`text` when it fits `limit` chars, else its leading whole words + "…"."""
    if len(text) <= limit:
        return text
    if limit < 2:
        return ""
    head = text[: limit - 1]
    space = head.rfind(" ")
    if space > 0:
        head = head[:space]
    return head.rstrip(" ,;:") + "…"


def _sections(body: str) -> dict[str, str]:
    """`## Name` -> its text (first section of each name)."""
    found: dict[str, str] = {}
    heads = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        found.setdefault(m.group(1).strip().casefold(), body[m.end():end].strip())
    return found


def _advice(body: str) -> str:
    sections = _sections(body)
    if "advice" in sections:
        return _one_line(sections["advice"])
    strategy = sections.get("strategy", "")
    first = re.split(r"\n\s*\n", strategy, maxsplit=1)[0] if strategy else ""
    return _cut(_one_line(first), ADVICE_MAX_CHARS)


def _one_of(value, allowed: frozenset[str]) -> bool:
    return isinstance(value, str) and value in allowed


def _grades(value) -> dict | None:
    if value is None:
        return dict.fromkeys(GRADE_KEYS, 0)
    if not isinstance(value, dict):
        return None
    grades = {}
    for key in GRADE_KEYS:
        n = value.get(key, 0)
        if type(n) is not int or n < 0:
            return None
        grades[key] = n
    return grades


def _parse_text(note_id: str, text: str) -> Note | None:
    if text.startswith("---\n") and not _FRONT_MATTER_RE.match(text):
        return None  # an opened front matter that never closes is not a note
    meta, body = split_front_matter(text)
    schema = meta.get("schema", 1)
    kind = meta.get("kind", "learned")
    status = meta.get("status", "candidate")
    learned_in = meta.get("learned_in", "unknown")
    verifier = meta.get("verifier")
    grades = _grades(meta.get("grades"))
    if (type(schema) is not int or schema not in (1, 2) or not _one_of(kind, KINDS)
            or not _one_of(status, STATUSES) or not _one_of(learned_in, LEARNED_IN)
            or not (verifier is None or isinstance(verifier, dict)) or grades is None):
        return None
    occurrences = meta.get("occurrences", 1)
    if type(occurrences) is not int or occurrences < 1:
        occurrences = 1  # the library's rule for a legacy or unreadable counter
    tasks = _str_list(meta.get("source_tasks"))
    title = _TITLE_RE.search(body)
    when = _WHEN_RE.search(body)
    skill = _SKILL_RE.search(body)
    return Note(
        id=note_id,
        title=_one_line(title.group(1)) if title else "",
        when=_one_line(when.group(1)) if when else "",
        advice=_advice(body),
        kind=kind,
        status=status,
        learned_in=learned_in,
        n_tasks=len({_task_key(t) for t in tasks}),
        occurrences=occurrences,
        verifier=dict(verifier) if verifier is not None else None,
        grades=grades,
        skill=skill.group(1) if skill else "",
    )


def _parse_bytes(note_id: str, data: bytes) -> Note | None:
    try:
        return _parse_text(note_id, data.decode("utf-8"))
    except UnicodeDecodeError:
        return None


def parse_note(path: Path) -> Note | None:
    """The note in `path`, or None when it cannot be read or its front matter is bad."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _parse_bytes(path.stem, data)


# ── library hash ──────────────────────────────────────────────────────────


def _library_files(root: Path | None) -> list[tuple[str, bytes]]:
    """(file name, bytes) of every readable `*.md` file in `root`, sorted by name."""
    if root is None:
        return []
    try:
        paths = sorted((p for p in Path(root).expanduser().glob("*.md") if p.is_file()), key=lambda p: p.name)
    except OSError:
        return []
    files = []
    for path in paths:
        try:
            files.append((path.name, path.read_bytes()))
        except OSError:
            continue  # unreadable: skipped, the hash covers the bytes present
    return files


def _sha_of(files: list[tuple[str, bytes]]) -> str:
    lines = "".join(f"{name}\t{hashlib.sha256(data).hexdigest()}\n" for name, data in files)
    return hashlib.sha256(lines.encode("utf-8", "surrogateescape")).hexdigest()


def library_sha(root: Path | None) -> str:
    """sha256 over `"<file name>\\t<sha256 of the file>\\n"` for every `*.md` in
    `root`, sorted by name. Missing or empty -> sha256 of the empty string."""
    return _sha_of(_library_files(root))


# ── lint ──────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _tool_names() -> frozenset[str]:
    """Every agent tool: `TOOL_SPECS` plus the MCP server's own tools. Imported
    here, not at module load, so the notes module stays light."""
    from ..apps.mcp_server import _EXTRA_TOOLS
    from .runtime import TOOL_SPECS

    return frozenset(spec["name"] for spec in TOOL_SPECS) | frozenset(t["name"] for t in _EXTRA_TOOLS)


def _lint_text(field: str, text: str, max_chars: int, skill: str) -> list[str]:
    text = text if isinstance(text, str) else ""
    if not text.strip():
        return [f"{field}: empty"]
    problems = []
    if len(text) > max_chars:
        problems.append(f"{field}: {len(text)} chars (max {max_chars})")
    if _URL_RE.search(text):
        problems.append(f"{field}: names a URL")
    lowered = text.lower()
    for name in dict.fromkeys(_SNAKE_RE.findall(lowered)):
        if name in FORBIDDEN_TOOLS:
            problems.append(f"{field}: names {name}, which a note may never name")
        elif name not in _tool_names():
            problems.append(f"{field}: {name} is not a tool name")
        elif name != skill and name not in READ_ONLY_TOOLS:
            problems.append(f"{field}: names {name}, which is neither read-only nor this note's skill")
    words = _WORD_RE.findall(lowered)
    verbs = [i for i, w in enumerate(words) if _OVERRIDE_RE.match(w)]
    guarded = [i for i, w in enumerate(words) if _GUARDED_RE.match(w)]
    pairs = [(words[i], words[j]) for i in verbs for j in guarded if abs(i - j) <= _OVERRIDE_REACH]
    if pairs:
        problems.append(f"{field}: '{pairs[0][0]}' near '{pairs[0][1]}'")
    geometry = _GEOMETRY_RE.search(text)
    if geometry:
        problems.append(f"{field}: carries the geometric value '{geometry.group(0)}'")
    return problems


def lint_advice(text: str, *, skill: str = "") -> list[str]:
    """Why `text` may not be served as advice ([] = clean). `skill` is the note's
    own skill, the only tool outside `READ_ONLY_TOOLS` the text may name."""
    return _lint_text("advice", text, ADVICE_MAX_CHARS, skill)


def lint_note(note: Note) -> list[str]:
    """`lint_advice` over every field a note renders: its title and its advice."""
    return (_lint_text("title", note.title, TITLE_MAX_CHARS, note.skill)
            + _lint_text("advice", note.advice, ADVICE_MAX_CHARS, note.skill))


# ── provenance tag ────────────────────────────────────────────────────────


def _verified(verifier) -> tuple[int, int] | None:
    """(passes, seeds) of a well-formed `verified` verdict, else None."""
    if not (isinstance(verifier, dict) and verifier.get("verdict") == "verified"):
        return None
    passes, seeds = verifier.get("passes"), verifier.get("seeds")
    if type(passes) is not int or not isinstance(seeds, list) or not seeds or not 0 <= passes <= len(seeds):
        return None
    return passes, len(seeds)


def _refuted(verifier) -> bool:
    return isinstance(verifier, dict) and verifier.get("verdict") == "refuted"


def note_tag(note: Note, sha: str) -> str:
    """`learned in <where> · <verifier part> · lib <sha8>` (`authored · ...` for
    hand-written notes); the verifier part is `verifier <passes>/<seeds>` only for
    a well-formed verified verdict."""
    verdict = _verified(note.verifier)
    tested = f"verifier {verdict[0]}/{verdict[1]}" if verdict else "not verifier-tested"
    origin = "authored" if note.kind == "authored" else f"learned in {note.learned_in}"
    return f"{origin} · {tested} · lib {str(sha)[:8]}"


# ── snapshot ──────────────────────────────────────────────────────────────


def _serves(note: Note, profile_class: str, min_tasks: int) -> bool:
    if note.status in _NEVER_SERVED or _refuted(note.verifier) or lint_note(note):
        return False
    if profile_class == "sim":
        return note.n_tasks >= min_tasks
    return _verified(note.verifier) is not None


def _rank(note: Note) -> tuple:
    useful = note.grades["useful"] - note.grades["misleading"]
    return (_verified(note.verifier) is None, -useful, -note.n_tasks, note.id)


def _line(title: str, advice: str, tag: str) -> str:
    return f"- {title}: {advice} [{tag}]"


def _render_block(items: list[tuple[str, str, str]]) -> str:
    return "\n".join([_BLOCK_HEAD, _ADVICE_ONLY, *(_line(*item) for item in items), _BLOCK_FOOT])


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


@dataclass(frozen=True)
class LibrarySnapshot:
    """The library as read once at build: every parsed note and, in rank order,
    the ones this profile class serves."""

    root: Path | None
    sha: str
    profile_class: str
    notes: tuple[Note, ...]
    served: tuple[Note, ...]

    def block(self) -> str:
        """The prompt block (spec §4.3, without the LEARNED.md header line): the top
        `BLOCK_NOTES` served notes in at most `BLOCK_MAX_CHARS` characters, the
        longest advice cut by whole words when needed; "" when nothing is served
        or the block cannot fit."""
        items = [(n.title, n.advice, note_tag(n, self.sha)) for n in self.served[:BLOCK_NOTES]]
        if not items:
            return ""
        text = _render_block(items)
        for _ in range(2 * len(items)):
            excess = len(text) - BLOCK_MAX_CHARS
            if excess <= 0:
                break
            i = max(range(len(items)), key=lambda k: len(items[k][1]))
            title, advice, tag = items[i]
            if not advice:
                break
            items[i] = (title, _cut(advice, max(0, len(advice) - excess)), tag)
            text = _render_block(items)
        return text if len(text) <= BLOCK_MAX_CHARS else ""

    def knowledge(self) -> str:
        """Every served note, one line each with its tag, in at most
        `KNOWLEDGE_MAX_CHARS` characters, then "(N more notes not shown)"; ""
        when nothing is served."""
        if not self.served:
            return ""
        reserve = len(f"\n({len(self.served)} more notes not shown)")
        text, shown = _ADVICE_ONLY, 0
        for i, note in enumerate(self.served):
            line = "\n" + _line(note.title, note.advice, note_tag(note, self.sha))
            more = reserve if i < len(self.served) - 1 else 0
            if len(text) + len(line) + more > KNOWLEDGE_MAX_CHARS:
                break
            text, shown = text + line, shown + 1
        if shown < len(self.served):
            text += f"\n({len(self.served) - shown} more notes not shown)"
        return text

    def matching(self, goal: str) -> Note | None:
        """The served note whose `when` words share the most words with `goal`
        (ties: the better-ranked note); None without any shared word."""
        goal_words = _words(goal)
        best, best_overlap = None, 0
        for note in self.served:
            overlap = len(goal_words & _words(note.when))
            if overlap > best_overlap:
                best, best_overlap = note, overlap
        return best

    def manifest(self) -> dict:
        """The library identity a run records (JSON-ready)."""
        return {"library_sha": self.sha, "profile_class": self.profile_class,
                "served_notes": [n.id for n in self.served],
                "block_notes": [n.id for n in self.served[:BLOCK_NOTES]] if self.block() else []}


def load_snapshot(root: Path | None, profile_class: str, *, min_tasks: int = 2) -> LibrarySnapshot:
    """Read the library at `root` once, read-only (a missing directory is an empty
    library, never created). Every class serves only lint-clean notes and never a
    refuted or retired one. `profile_class` "sim" serves notes seen in at least
    `min_tasks` (>= 1) distinct tasks; any other class serves only notes with a
    well-formed verified verdict, whatever their task count."""
    root = None if root is None else Path(root).expanduser()
    files = _library_files(root)
    notes = tuple(note for name, data in files if (note := _parse_bytes(name[: -len(".md")], data)) is not None)
    klass = "sim" if profile_class == "sim" else "real"
    floor = max(1, int(min_tasks))
    served = tuple(sorted((n for n in notes if _serves(n, klass, floor)), key=_rank))
    return LibrarySnapshot(root=root, sha=_sha_of(files), profile_class=klass, notes=notes, served=served)
