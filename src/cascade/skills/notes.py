"""ASPIRE learned notes, format v2: parse, lint, provenance tag, library hash, snapshot.

A note stays one markdown file ``<library>/<slug>.md`` with the front matter
`skills.library` writes (``occurrences``, ``source_tasks``, ``source_runs``,
``status``). v2 adds ``schema``, ``kind``, ``learned_in``, ``verifier`` and
``grades``; a missing key takes its v1 default, so every v1 file is a v2 note
(spec §4.1). The header (before the first ``## `` section) has ``# <title>``,
``- **When to apply:** <words>`` and the failure signature naming the note's own
skill; ``## Advice`` is what the agent reads and ``## Evidence`` is for humans only,
never rendered. A v1 body has no ``## Advice``: its advice is the first paragraph of
``## Strategy`` (a longer one than `ADVICE_MAX_CHARS` fails the lint: the note is
skipped, never cut into acceptance). Only regular files (never a symlink) whose name
is a safe id (`_NOTE_NAME_RE`) are notes, hashed or served; every other `*.md` entry
and every note that fails the lint is listed in the snapshot's `skipped` with its
reason and logged, never dropped silently.

Notes are advice TEXT. This module renders them and nothing else: no value in a
note reaches a config, a tool argument or a gate. A note is served only when its
lint is clean (`lint_note`), a deny-by-default STRUCTURAL gate on every rendered
field (title and advice): short plain English in plain case, printable ASCII, at
most one number (digits or words) and no measurement, no code, markup, role or
provenance look-alike, naming only read-only tools and the note's own skill as
identifiers and without arguments, and no reset, stop, loop, bypass, weakening or
success-claim advice.

`load_snapshot` reads the library ONCE (it never creates or writes it) and keeps
that view for the session: the prompt block, `robot_knowledge` and the run stamp
all come from the same bytes, so a nightly rewrite never changes a running
session. Harvested advice (``kind: learned``) is served only once PROMOTED -- seen
in ``min_tasks`` distinct tasks AND passed by the verifier (`_verified`) -- on
every profile class (`cascade.sim_profile`). An authored note follows the task
counters on "sim"; "real" (and anything that is not exactly "sim") serves only
verifier-passed notes.
"""
from __future__ import annotations

import errno
import functools
import hashlib
import itertools
import logging
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .library import read_note_text

log = logging.getLogger(__name__)

ADVICE_MAX_CHARS = 300
TITLE_MAX_CHARS = 80
BLOCK_MAX_CHARS = 1400
LEARNED_MAX_BYTES = 2000
KNOWLEDGE_MAX_CHARS = 6000
LEARNED_HEADER_PREFIX = "<!-- cascade-learned-notes v1 "
#: notes in the prompt block (the rest are only in robot_knowledge)
BLOCK_NOTES = 2
#: a verifier pass (DECISIONS Q6): at least this many of exactly `VERIFIER_SEEDS` seeds
#: passed, each within `VERIFIER_MAX_ATTEMPTS` motion attempts ...
VERIFIER_MIN_PASSES = 4
VERIFIER_SEEDS = 5
VERIFIER_MAX_ATTEMPTS = 2
#: ... on the physics engine every sim run uses (owner rule)
VERIFIER_ENGINE = "newton"

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

#: a note's file name: its id is safe in a JSON manifest, a LEARNED.md header line
#: and the hash's "<name>\t<sha>\n" lines (no tab, newline, comma or "-->")
_NOTE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\.md")
# Every pattern below is linear: one quantifier per run, no nested or adjacent
# overlapping quantifiers (the lint also never sees more than its field's cap).
_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_WHEN_RE = re.compile(r"\*\*When to apply:\*\*(.*)")
_SKILL_RE = re.compile(r"\*\*Failure signature:\*\*\s*([a-z][a-z0-9_]*)")
_SECTION_RE = re.compile(r"^## +(.*)$", re.MULTILINE)


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
    """`text` when it fits `limit` chars, else its leading whole words + "…" in at
    most `limit` chars; "" when not even its first word fits (never half a word)."""
    if len(text) <= limit:
        return text
    if limit < 2:
        return ""
    head = text[:limit]  # the words of the cut plus one char to see where a word ends
    space = head.rfind(" ")
    head = head[:space].rstrip(" ,;:") if space > 0 else ""
    return head + "…" if head else ""


def _sections(body: str) -> dict[str, str]:
    """`## Name` -> its text (first section of each name)."""
    found: dict[str, str] = {}
    heads = list(_SECTION_RE.finditer(body))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        found.setdefault(m.group(1).strip().casefold(), body[m.end():end].strip())
    return found


def _header(body: str) -> str:
    """The body before its first `## ` section: the only place title, when and skill
    are read from (an `## Evidence` line can never supply them)."""
    first = _SECTION_RE.search(body)
    return body[: first.start()] if first else body


def _first_paragraph(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def _advice(body: str) -> str:
    sections = _sections(body)
    if "advice" in sections:
        return _one_line(sections["advice"])
    return _one_line(_first_paragraph(sections.get("strategy", "")))


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
    stored = read_note_text(text)
    if stored is None:
        return None  # an opened front matter that never closes is not a note
    meta, body, n_tasks = stored
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
    header = _header(body)
    title = _TITLE_RE.search(header)
    when = _WHEN_RE.search(header)
    skill = _SKILL_RE.search(header)
    return Note(
        id=note_id,
        title=_one_line(title.group(1)) if title else "",
        when=_one_line(when.group(1)) if when else "",
        advice=_advice(body),
        kind=kind,
        status=status,
        learned_in=learned_in,
        n_tasks=n_tasks,
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


def _is_note_name(name: str) -> bool:
    return _NOTE_NAME_RE.fullmatch(name) is not None


def _read_regular(path: Path) -> tuple[bytes | None, str]:
    """(bytes, "") of the regular file at `path`, else (None, why). The open never
    follows a symlink (O_NOFOLLOW, so a link swapped in after the listing is refused
    too) and never blocks on a FIFO (O_NONBLOCK, then fstat)."""
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return None, "symlink" if exc.errno == errno.ELOOP else "unreadable"
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None, "not_a_file"
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks), ""
    except OSError:
        return None, "unreadable"
    finally:
        os.close(fd)


def parse_note(path: Path) -> Note | None:
    """The note in `path`, or None when its name is not a safe note id, it is a
    symlink or not a regular file, it cannot be read or its front matter is bad."""
    path = Path(path)
    if not _is_note_name(path.name):
        return None
    data, _ = _read_regular(path)
    return None if data is None else _parse_bytes(path.stem, data)


# ── library hash ──────────────────────────────────────────────────────────


def _scan(root: Path | None) -> list[tuple[str, bytes | None, str]]:
    """(file name, bytes, "") for every note file in `root` and (file name, None,
    why) for every other `*.md` entry, sorted by name. A note file has a safe id as
    its name (`_NOTE_NAME_RE`: no control character, tab, newline, upper case or
    non-UTF-8 byte), is a regular file and no symlink (lstat, then an O_NOFOLLOW
    read) and is readable; only note files are hashed, so the hash's line format
    stays unambiguous."""
    if root is None:
        return []
    try:
        paths = sorted(Path(root).expanduser().glob("*.md"), key=lambda p: p.name)
    except OSError:
        return []
    entries: list[tuple[str, bytes | None, str]] = []
    for path in paths:
        if not _is_note_name(path.name):
            entries.append((path.name, None, "file_name"))
            continue
        try:
            mode = os.lstat(path).st_mode
        except OSError:
            entries.append((path.name, None, "unreadable"))
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            entries.append((path.name, None, "symlink" if stat.S_ISLNK(mode) else "not_a_file"))
            continue
        data, why = _read_regular(path)
        entries.append((path.name, data, why))
    return entries


def _sha_of(entries: list[tuple[str, bytes | None, str]]) -> str:
    lines = "".join(f"{name}\t{hashlib.sha256(data).hexdigest()}\n" for name, data, _ in entries if data is not None)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def library_sha(root: Path | None) -> str:
    """sha256 over `"<file name>\\t<sha256 of the file>\\n"` for every note file
    in `root`, sorted by name. Missing or empty -> sha256 of the empty string."""
    return _sha_of(_scan(root))


# ── lint ──────────────────────────────────────────────────────────────────
#
# A deny-by-default STRUCTURAL gate, not a blocklist: a field that cannot be shown
# to be short plain English is refused, and its note is skipped (never cut into
# acceptance). In order:
#   1. length: a field over its cap is refused before any pattern runs;
#   2. characters: no control or format character (Cc/Cf), NFKC-stable, and only
#      printable ASCII letters, digits, space and `_PUNCTUATION`, plus `_TYPOGRAPHY`
#      (so no homoglyph, CJK, fullwidth, zero-width, markup or code character);
#   3. structure: every word lower, Title or UPPER case; then over the field read as
#      words (`_View`) -- (a) the words split at camelCase humps, digits and any
#      separator, (b) the unsplit runs of letters and digits, and (c) at each word the
#      2-3 words that join into one whole phrase word (`re set`, `ig nore`, `DIS-able`)
#      -- no URL, no unknown identifier, no letter-digit mix, no dotted name, no
#      spelled-out word, no role or chat-template marker, nothing that imitates the
#      provenance tag, at most one number (digits or number words) and no measurement
#      or coordinate, only read-only tools and the note's own skill as identifiers and
#      never with arguments, and no reset, stop, loop, override, weakening or
#      success-claim advice. A rule refuses a field under its name (`skipped` reasons).

#: the non-ASCII characters a field may carry: typographic quotes and dashes
_TYPOGRAPHY = frozenset("\u2018\u2019\u201c\u201d\u2013\u2014")
#: the ASCII punctuation a field may carry besides letters and digits; every other
#: symbol ([ ] { } ( ) = < > | ` # * @ $ ^ ~ and the backslash) is code or markup,
#: so "<|", "|>" and "```" never pass either
_PUNCTUATION = frozenset(" .,;:!?'\"-/_&+%")
_EDGE_PUNCTUATION = ".,;:!?'\"-/&+%" + "".join(sorted(_TYPOGRAPHY))

_RUN_RE = re.compile(r"[A-Za-z0-9]+")
_IDENT_RE = re.compile(r"[A-Za-z0-9_]+")
_NUMBER_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_LETTERS_RE = re.compile(r"[A-Za-z]+")
_ORDINAL_RE = re.compile(r"[0-9]+(?:st|nd|rd|th)")
_HEX_RE = re.compile(r"[0-9a-f]+")
#: a hex-like run this long reads as a hash or a library sha
_HEX_MIN_CHARS = 6

# Phrases over the words of a field: "word" matches exactly, "stem*" any word
# starting with the stem (compiled once by `_Phrases`).
#: followed by ":" or a dash ("Assistant - Done"), or by one word and ":" ("user said:")
_ROLE_WORDS = frozenset({"assistant", "user", "system", "developer", "human", "tool", "observation", "function",
                         "output", "operator", "owner"})
#: chat-template and prompt-injection markers, refused wherever they stand
_TEMPLATE = (("im", "start"), ("im", "end"), ("system", "prompt*"), ("developer", "mode*"), ("new", "instruction*"))
#: reset_scene / reset_stop in any spelling, "reset the scene" and phrasings that only
#: mean a reset: a reset clears the retry ledger (R4)
_RESET = (("reset*",), ("respawn*",), ("begin*", "again"), ("start*", "from", "scratch"))
#: a reset only with a `_SCENE` noun as their object ("restart the scene", "redo everything")
_RESET_VERBS = (("restart*",), ("reload*",), ("redo*",), ("redid",), ("reinitialis*",), ("reinitializ*",),
                ("restor*",))
_SCENE = (("scene*",), ("layout*",), ("table*",), ("everything",), ("kitchen*",), ("world*",), ("simulation*",))
_PROVENANCE = (("verifier*",), ("verified",), ("promoted",), ("provenance",), ("authored",), ("passing", "seed*"),
               ("tested", "in"), ("tested", "on"), ("tested", "by"), ("newton*",), ("physx",))
_LEARNED = (("learn*", "in"), ("learn*", "on"))
_STOPS = (("emergenc*", "stop*"), ("e", "stop*"), ("estop*",), ("emergencystop*",), ("halt*", "motion*"),
          ("halt*", "the", "motion*"), ("haltmotion*",), ("stop", "latch*"), ("emergenc*", "button*"))
_NEGATORS = (("never",), ("not",), ("no",), ("don",), ("dont",), ("without",), ("cannot",))
_HALTS = (("stop*",), ("halt*",), ("paus*",))
_LOOPS = (("start", "over"), ("try", "again", "until"), ("repeat*", "until"), ("until", "it", "succe*"),
          ("until", "it", "work*"), ("until", "succe*"), ("keep", "call*"), ("keep", "try*"), ("keep", "retr*"),
          ("keep", "going"), ("retr*", "until"), ("retr*", "till"), ("forever",), ("over", "and", "over"),
          ("never", "give", "up"), ("no", "limit*"), ("as", "many", "times"))
#: refused wherever they stand: no plain advice needs them
_OVERRIDES = (("ignor*",), ("disregard*",), ("bypass*",), ("by", "pass*"), ("circumvent*",), ("overrid*",),
              ("over", "rid*"))
#: refused when one of `_GUARDS` is their direct object (`_direct_object`)
_WEAKEN = (("disabl*",), ("turn*", "off"), ("switch*", "off"), ("shut*", "off"), ("skip*",), ("lower*",),
           ("relax*",), ("reduc*",), ("shrink*",), ("rais*",), ("smaller",), ("decreas*",), ("loosen*",),
           ("suppress*",), ("silenc*",), ("mute*",), ("weaken*",), ("work*", "around"), ("get*", "around"),
           ("cut*",), ("drop*",), ("zero*",), ("waiv*",), ("tighten*",), ("remov*",), ("neglect*",), ("dismiss*",),
           ("forget*",), ("forgot*",), ("overrul*",), ("defeat*",), ("sidestep*",), ("never", "mind"),
           ("set*", "aside"), ("look*", "past"), ("get*", "past"), ("pay*", "no", "attention", "to"))
_GUARDS = (("gate*",), ("check*",), ("refus*",), ("warn*",), ("veto*",), ("clearanc*",), ("margin*",),
           ("limit*",), ("safet*",), ("collis*",), ("collid*",), ("stop*",), ("halt*",), ("guard*",),
           ("harness*",), ("interlock*",), ("watchdog*",), ("threshold*",), ("toleranc*",))
_CLAIM_AS = (("treat*",), ("consider*",), ("regard*",), ("declar*",), ("mark*",))
_CLAIM_SAY = (("tell*",), ("say*",), ("said",), ("report*",), ("announc*",), ("reply",), ("replie*",),
              ("answer*",))
_SUCCESS = (("succe*",), ("done",), ("ok",), ("okay",), ("complete",), ("completed",), ("finished",),
            ("worked",))
#: units of length or angle (and percent), refused wherever they stand
_UNITS = (("mm",), ("cm",), ("um",), ("ft",), ("deg",), ("degs",), ("rad",), ("rads",), ("millimet*",),
          ("centimet*",), ("micromet*",), ("micron*",), ("metre*",), ("meter*",), ("inch*",), ("foot",),
          ("feet",), ("degree*",), ("radian*",), ("percent*",), ("dm",), ("decimet*",), ("km",), ("kilomet*",),
          ("yard*",), ("yd",), ("yds",), ("rev",), ("revs",), ("revolution*",), ("mrad",), ("milliradian*",),
          ("thou",), ("mil",), ("mils",), ("pct",))  # "per cent" joins into "percent" (`_View`)
#: a unit only right after a number ("3 m", "2 in"; and "50%")
_NUMBER_UNITS = frozenset({"m", "in"})
#: a unit only right after a number, digits or words ("2 turns", "a quarter turn")
_TURNS = frozenset({"turn", "turns"})
#: every family a word or a join of words is read against (`_View`)
_FAMILIES = (_TEMPLATE, _RESET, _RESET_VERBS, _SCENE, _PROVENANCE, _LEARNED, _STOPS, _NEGATORS, _HALTS, _LOOPS,
             _OVERRIDES, _WEAKEN, _GUARDS, _CLAIM_AS, _CLAIM_SAY, _SUCCESS, _UNITS)
#: spelled-out numbers: two in one field ("one" aside: "pick the one on top"), two side
#: by side ("minus one", "twenty-one") or one after an axis letter is refused
_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "point", "half", "quarter", "minus", "negative"})
#: tool parameters a note may not set (`name:`), besides every single-word parameter of a tool
_PARAMETERS = frozenset({"object", "destination", "target", "x", "y", "z", "arm", "force", "speed", "frame", "pose",
                         "position", "grasp", "offset"})
#: the plain endings a joined word may have after a stem ("ig nore" = ignor + e); a
#: join that only STARTS like a stem ("in check" ~ inch...) is no word
_ENDINGS = ("", "s", "e", "es", "ed", "d", "t", "ing", "er", "ers", "y", "ies", "ied", "ion", "ions", "ance",
            "ence", "al", "ly", "ty", "ting", "ted", "ping", "ped", "den", "ment", "ments")
#: separators across which two words may join (never an apostrophe: "cup's top" is no "stop")
_JOINABLE = frozenset(" -_\u2013\u2014")
_APOSTROPHES = "'\u2019"
#: a weakening verb's object: the guard noun among its first words, in its clause
_OBJECT_REACH = 4  # "skip the pre-close check"
_MORE_OBJECTS = 2  # "disable the light, the gate and the check": listed objects count too
_DETERMINERS = frozenset({"the", "a", "an", "its", "their", "this", "that", "these", "those", "your", "my", "our",
                          "his", "her", "every", "each", "all", "any", "some", "both"})
_CLAUSE_ENDS = frozenset({"and", "to", "for", "then", "so", "but"})
_PREPOSITIONS = frozenset({"near", "over", "under", "onto", "into", "on", "in", "at", "by", "with", "from", "above",
                           "below", "beside", "behind", "past", "around", "through", "across", "toward", "towards",
                           "along", "against", "beyond", "inside", "outside", "upon", "via", "within", "between",
                           "beneath", "underneath"})
_SAY_REACH = 5  # "tell the user it is done"
_TREAT_REACH = 6  # "treat every refused result as a success"
_NEGATION_REACH = 2  # "don't stop", "do not ever stop"


@functools.lru_cache(maxsize=1)
def _tool_names() -> frozenset[str]:
    """Every agent tool: `TOOL_SPECS` plus the MCP server's own tools. Imported
    here, not at module load, so the notes module stays light."""
    from ..apps.mcp_server import _EXTRA_TOOLS
    from .runtime import TOOL_SPECS

    return frozenset(spec["name"] for spec in TOOL_SPECS) | frozenset(t["name"] for t in _EXTRA_TOOLS)


@functools.lru_cache(maxsize=1)
def _parameter_names() -> frozenset[str]:
    """`_PARAMETERS` plus every single-word parameter of a tool ("label", "query")."""
    from ..apps.mcp_server import _EXTRA_TOOLS
    from .runtime import TOOL_SPECS

    names = set(_PARAMETERS)
    for spec in (*TOOL_SPECS, *_EXTRA_TOOLS):
        names.update(p for p in (spec.get("parameters") or {}).get("properties", {}) if "_" not in p)
    return frozenset(names)


class _Words:
    """Exact words and stems, with the stems bucketed by their first two letters so a
    lookup tests only the stems that can match (a one-letter stem is always tested)."""

    def __init__(self, parts):
        self.exact = frozenset(p for p in parts if not p.endswith("*"))
        self.stems = frozenset(p[:-1] for p in parts if p.endswith("*"))
        self.short = tuple(stem for stem in sorted(self.stems) if len(stem) < 2)
        buckets: dict[str, list[str]] = {}
        for stem in sorted(self.stems):
            if len(stem) >= 2:
                buckets.setdefault(stem[:2], []).append(stem)
        self.buckets = {key: (*value, *self.short) for key, value in buckets.items()}

    def starts(self, word: str) -> bool:
        """`word` is an exact word or starts with a stem."""
        return word in self.exact or word.startswith(self.buckets.get(word[:2], self.short))


@functools.lru_cache(maxsize=1)
def _vocabulary() -> _Words:
    """Every part of every phrase family (what a join of words may read as)."""
    return _Words([part for family in _FAMILIES for phrase in family for part in phrase])


@functools.lru_cache(maxsize=1)
def _heads() -> _Words:
    """The parts that can START a phrase of any family: a word that is none of them
    starts no match, so the scans skip it (`_View.live`)."""
    return _Words([phrase[0] for family in _FAMILIES for phrase in family])


_ENDING_SET = frozenset(_ENDINGS)
_ENDING_SIZES = tuple(sorted({len(end) for end in _ENDINGS}))


def _whole_word(joined: str) -> bool:
    """`joined` (2-3 words run together) reads as one phrase word: an exact part, or
    a stem plus a plain ending (`_ENDINGS`)."""
    vocabulary = _vocabulary()
    if joined in vocabulary.exact:
        return True
    if not vocabulary.starts(joined):
        return False
    n = len(joined)
    return any(joined[n - k:] in _ENDING_SET and joined[:n - k] in vocabulary.stems for k in _ENDING_SIZES if k < n)


def _character_problem(text: str) -> str | None:
    for ch in text:
        if unicodedata.category(ch) in ("Cc", "Cf"):
            return f"control or format character U+{ord(ch):04X}"
    if unicodedata.normalize("NFKC", text) != text:
        return "changes under NFKC normalisation (fullwidth, compatibility or decomposed characters)"
    for ch in text:
        if ch.isascii():
            if not (ch.isalnum() or ch in _PUNCTUATION):
                return f"carries {ch!r}, which plain advice does not use"
        elif ch not in _TYPOGRAPHY:
            kind = "letter" if unicodedata.category(ch).startswith("L") else "character"
            return f"non-ASCII {kind} U+{ord(ch):04X} (notes are plain English)"
    return None


def _mixed_case(text: str) -> str | None:
    """The first word that is neither lower, Title nor UPPER case ("reSet", "iGnore")."""
    for m in _LETTERS_RE.finditer(text):
        word = m.group(0)
        if not (word.islower() or word.isupper() or (word[0].isupper() and word[1:].islower())):
            return word
    return None


def _tokens(text: str) -> list[tuple[str, int, int]]:
    """(lower-case word, start, end) for every word of `text`: runs of letters and
    digits split at camelCase humps and letter/digit borders, so `reset_scene`,
    `reset-scene`, `resetScene`, `reset.scene` and `reset scene` read alike."""
    out = []
    for m in _RUN_RE.finditer(text):
        run, base, start = m.group(0), m.start(), 0
        for k in range(1, len(run)):
            a, b = run[k - 1], run[k]
            if (a.islower() and b.isupper()) or a.isdigit() != b.isdigit():
                out.append((run[start:k].lower(), base + start, base + k))
                start = k
        out.append((run[start:].lower(), base + start, base + len(run)))
    return out


def _break(gap: str) -> str:
    """What separates two words: "." (a clause break: . ; : ! ? or a spaced dash),
    "," (a comma) or "" (spaces, a hyphen, quotes)."""
    if any(ch in ".;:!?\u2013\u2014" for ch in gap) or ("-" in gap and " " in gap):
        return "."
    return "," if "," in gap else ""


class _View:
    """One reading of a field as lower-case `words`. `cands[i]` are the readings that
    start at word i: the word itself, then the 2-3 words from i joined when only
    spaces or dashes separate them, word i is no contraction tail and the join reads
    as one phrase word (`_whole_word`), each with the number of words it covers.
    `breaks[i]` is what separates word i from the word before it (`_break`) and
    `live` the positions where some phrase of some family can start (`_heads`)."""

    __slots__ = ("breaks", "cands", "live", "words")

    def __init__(self, text: str, spans: list[tuple[str, int, int]]):
        self.words = [word for word, _, _ in spans]
        gaps = [""] + [text[spans[k - 1][2]:spans[k][1]] for k in range(1, len(spans))]
        self.breaks = [_break(gap) for gap in gaps]
        self.cands = []
        for i, (word, start, _) in enumerate(spans):
            here = [(word, 1)]
            if not (start and text[start - 1] in _APOSTROPHES):
                joined = word
                for k in (1, 2):
                    if i + k >= len(spans) or not _JOINABLE.issuperset(gaps[i + k]):
                        break
                    joined += self.words[i + k]
                    if _whole_word(joined):
                        here.append((joined, k + 1))
            self.cands.append(here)
        heads = _heads()
        self.live = [i for i, here in enumerate(self.cands) if any(heads.starts(word) for word, _ in here)]


class _Phrases:
    """A family of phrases, compiled: each part is (text, is_stem); a word can start
    a match only if it equals an exact first part or starts with a stem one."""

    def __init__(self, phrases):
        self.phrases = tuple(tuple((p[:-1], True) if p.endswith("*") else (p, False) for p in phrase)
                             for phrase in phrases)
        self.exact = frozenset(ph[0][0] for ph in self.phrases if not ph[0][1])
        self.stems = tuple(ph[0][0] for ph in self.phrases if ph[0][1])

    def at(self, view: _View, i: int) -> int:
        """The number of words the first phrase matching at word `i` covers, else 0."""
        here = view.cands[i]
        for word, _ in here:
            if word in self.exact or word.startswith(self.stems):
                break
        else:
            return 0
        for phrase in self.phrases:
            j = i
            for part, stem in phrase:
                if j >= len(view.words):
                    break
                for word, used in view.cands[j]:
                    if word.startswith(part) if stem else word == part:
                        j += used
                        break
                else:
                    break
            else:
                return j - i
        return 0


@functools.cache
def _compiled(phrases: tuple) -> _Phrases:
    return _Phrases(phrases)


def _first(view: _View, phrases) -> str | None:
    family = _compiled(phrases)
    for i in view.live:
        n = family.at(view, i)
        if n:
            return " ".join(view.words[i:i + n])
    return None


def _followed(view: _View, heads, tails, reach: int) -> tuple[str, str] | None:
    """The first (head, tail) where a `tails` phrase starts among the `reach` words
    right after a `heads` phrase; None when there is none."""
    head_family, tail_family = _compiled(heads), _compiled(tails)
    for i in view.live:
        n = head_family.at(view, i)
        if not n:
            continue
        for j in range(i + n, min(i + n + reach, len(view.words))):
            m = tail_family.at(view, j)
            if m:
                return " ".join(view.words[i:i + n]), " ".join(view.words[j:j + m])
    return None


def _direct_object(view: _View, start: int, nouns: _Phrases) -> tuple[int, int] | None:
    """(position, size) of a `nouns` phrase in the object that starts at word
    `start` -- among its first `_OBJECT_REACH` words, before a preposition and the end
    of the clause (punctuation, `_CLAUSE_ENDS`). "and" or a comma before a
    determiner lists another object ("the light and the gate")."""
    words, breaks, n = view.words, view.breaks, len(view.words)
    j, seen, more = start, 0, _MORE_OBJECTS
    while j < n and seen < _OBJECT_REACH:
        word = words[j]
        if breaks[j] == ".":
            return None
        if breaks[j] == ",":
            if not (more and word in _DETERMINERS):
                return None
            seen, more = 0, more - 1
        if word in _CLAUSE_ENDS:
            if word == "and" and more and j + 1 < n and not breaks[j + 1] and words[j + 1] in _DETERMINERS:
                j, seen, more = j + 1, 0, more - 1
                continue
            return None
        if word in _PREPOSITIONS:
            return None
        size = nouns.at(view, j)
        if size:
            return j, size
        j, seen = j + 1, seen + 1
    return None


def _verb_object(view: _View, verbs, nouns) -> tuple[str, str] | None:
    """The first (verb, noun) where a `nouns` phrase is the direct object of a `verbs` phrase."""
    verb_family, noun_family = _compiled(verbs), _compiled(nouns)
    for i in view.live:
        size = verb_family.at(view, i)
        if size:
            hit = _direct_object(view, i + size, noun_family)
            if hit:
                return " ".join(view.words[i:i + size]), " ".join(view.words[hit[0]:hit[0] + hit[1]])
    return None


def _next_char(text: str, i: int) -> str:
    while i < len(text) and text[i] == " ":
        i += 1
    return text[i] if i < len(text) else ""


def _dotted_name(text: str) -> str | None:
    """A config-style dotted name ("grasp.approach", "www.example.com"); "e.g." and
    "0.42" are not."""
    for chunk in text.split():
        core = chunk.strip(_EDGE_PUNCTUATION)
        if "." not in core:
            continue
        parts = core.split(".")
        if (all(part and _IDENT_RE.fullmatch(part) for part in parts) and any(len(part) > 1 for part in parts)
                and any(c.isalpha() for c in core)):
            return core
    return None


def _spelled_out(words: list[str]) -> bool:
    """Three single letters in a row ("i g n o r e")."""
    run = 0
    for word in words:
        run = run + 1 if len(word) == 1 and word.isalpha() else 0
        if run >= 3:
            return True
    return False


def _is_number(word: str) -> bool:
    return word.isdigit() or word in _NUMBER_WORDS


def _adjacent(text: str, spans, first, second, gap: str = " -\u2013") -> str | None:
    """'<a> <b>' for the first two neighbouring words with first(a) and second(b),
    separated only by `gap` characters."""
    for (a, _, end), (b, start, _) in itertools.pairwise(spans):
        if first(a) and second(b) and set(text[end:start]) <= set(gap):
            return f"{a} {b}"
    return None


def _score(words: list[str]) -> str | None:
    """'out of <n>' or '<n> of <n>' ("five out of five", "4 of 5"): a verifier score."""
    for a, b, c in zip(words, words[1:], words[2:]):
        if b == "of" and _is_number(c) and (a == "out" or _is_number(a)):
            return f"{a} of {c}"
    return None


def _role_marker(text: str, spans) -> str | None:
    """'<role>:' -- a `_ROLE_WORDS` word followed by ':' or a dash, or by one word and ':'."""
    for k, (word, _, end) in enumerate(spans):
        if word not in _ROLE_WORDS:
            continue
        after = _next_char(text, end)
        if after == ":" or (after and after in "\u2013\u2014") or (after == "-" and text[end] == " "):
            return f"{word}:"
        if k + 1 < len(spans):
            nxt, start, nxt_end = spans[k + 1]
            if not text[end:start].strip(" ") and _next_char(text, nxt_end) == ":":
                return f"{word} {nxt}:"
    return None


def _measurement(text: str, numbers, spans) -> str | None:
    """'<number> <unit>': a number followed by "%", a `_NUMBER_UNITS` word or (digits
    or a number word) a `_TURNS` word."""
    for m in numbers:
        i = m.end()
        while i < len(text) and text[i] == " ":
            i += 1
        if text.startswith("%", i):
            return f"{m.group(0)}%"
        unit = _LETTERS_RE.match(text, i)
        if unit and unit.group(0).lower() in _NUMBER_UNITS:
            return f"{m.group(0)} {unit.group(0)}"
    return _adjacent(text, spans, _is_number, _TURNS.__contains__)


def _identifier_problems(text: str, skill: str) -> list[tuple[str, str]]:
    """Identifiers (unsplit `_IDENT_RE` runs, any case): every snake_case one is a
    tool, read-only or the note's own skill and never a forbidden one; no identifier
    or tool parameter is followed by ':' (arguments)."""
    tools, parameters = _tool_names(), _parameter_names()
    found: dict[str, str] = {}
    for m in _IDENT_RE.finditer(text):
        raw = m.group(0)
        name = raw.lower()
        if "_" in name:
            if name in FORBIDDEN_TOOLS:
                found.setdefault("tool", f"names {name}, which a note may never name")
            elif name not in tools:
                found.setdefault("identifier", f"{raw} is not a tool name")
            elif name != skill and name not in READ_ONLY_TOOLS:
                found.setdefault("tool", f"names {name}, which is neither read-only nor this note's skill")
        if ("_" in name or name in parameters) and _next_char(text, m.end()) == ":":
            found.setdefault("arguments", f"gives arguments ('{raw}:'; a note names a tool, never how to call it)")
    return list(found.items())


def _structure_problems(text: str, skill: str) -> list[tuple[str, str]]:
    """(rule, reason) for every structural rule the field breaks, in rule order."""
    spans = _tokens(text)
    words = [w for w, _, _ in spans]
    run_spans = [(m.group(0).lower(), m.start(), m.end()) for m in _RUN_RE.finditer(text)]
    views = [_View(text, spans)]
    if [w for w, _, _ in run_spans] != words:  # (b): only a camelCase or digit split differs
        views.append(_View(text, run_spans))
    numbers = list(_NUMBER_RE.finditer(text))
    found: list[tuple[str, str | None]] = []

    def phrase(family) -> str | None:
        return next((hit for view in views if (hit := _first(view, family))), None)

    def verb_object(verbs, nouns) -> tuple[str, str] | None:
        return next((hit for view in views if (hit := _verb_object(view, verbs, nouns))), None)

    def followed(heads, tails, reach) -> tuple[str, str] | None:
        return next((hit for view in views if (hit := _followed(view, heads, tails, reach))), None)

    # code-like spellings
    found.append(("url", "names a URL" if "://" in text else None))
    mixed = _mixed_case(text)
    found.append(("mixed_case", f"mixes upper and lower case in '{mixed}'" if mixed else None))
    identifiers = dict(_identifier_problems(text, skill))
    found.append(("identifier", identifiers.get("identifier")))
    mixed_run = next((r for r, _, _ in run_spans
                      if not r.isalpha() and not r.isdigit() and not _ORDINAL_RE.fullmatch(r)), None)
    found.append(("letter_digit", f"mixes letters and digits in '{mixed_run}'" if mixed_run else None))
    dotted = _dotted_name(text)
    found.append(("dotted_name", f"carries the dotted name '{dotted}'" if dotted else None))
    found.append(("spelled_out", "spells a word out letter by letter" if _spelled_out(words) else None))
    role = _role_marker(text, spans)
    found.append(("role_marker", f"carries the role marker '{role}'" if role else None))
    template = phrase(_TEMPLATE)
    found.append(("role_marker", f"carries the chat-template marker '{template}'" if template else None))
    # look-alikes of the provenance tag
    tag = phrase(_PROVENANCE)
    found.append(("provenance", f"imitates the provenance tag ('{tag}')" if tag else None))
    learned = phrase(_LEARNED)
    found.append(("provenance", f"claims where it was learned ('{learned}')" if learned else None))
    lib = next((run_spans[k + 1][0] for k in range(len(run_spans) - 1)
                if run_spans[k][0] == "lib" and _HEX_RE.fullmatch(run_spans[k + 1][0])), None)
    found.append(("provenance", f"imitates a library hash ('lib {lib}')" if lib else None))
    hexed = next((r for r, _, _ in run_spans if len(r) >= _HEX_MIN_CHARS and _HEX_RE.fullmatch(r)), None)
    found.append(("provenance", f"carries the hex token '{hexed}' (a hash)" if hexed else None))
    score = _score(words)
    found.append(("provenance", f"claims a score ('{score}')" if score else None))
    # numbers and measurements
    counted = len(numbers) + sum(1 for w in words if w in _NUMBER_WORDS and w != "one")
    spelled = _adjacent(text, spans, _NUMBER_WORDS.__contains__, _NUMBER_WORDS.__contains__)
    found.append(("numbers", f"carries {counted} numbers (at most one)" if counted > 1
                  else f"spells out the number '{spelled}'" if spelled else None))
    measured = _measurement(text, numbers, spans)
    found.append(("measurement", f"carries the measurement '{measured}'" if measured else None))
    unit = phrase(_UNITS)
    found.append(("unit", f"names the unit '{unit}'" if unit else None))
    coordinate = _adjacent(text, spans, ("x", "y", "z").__contains__, _is_number, " :=+-")
    found.append(("coordinate", f"carries the coordinate '{coordinate}'" if coordinate else None))
    # tools: read-only ones and the note's own skill, as identifiers, without arguments
    found.append(("tool", identifiers.get("tool")))
    found.append(("arguments", identifiers.get("arguments")))
    # advice a note may never give
    reset = phrase(_RESET)
    if not reset and (scene := verb_object(_RESET_VERBS, _SCENE)):
        reset = f"{scene[0]} ... {scene[1]}"
    found.append(("reset", f"advises a reset ('{reset}')" if reset else None))
    stop = phrase(_STOPS)
    found.append(("stop", f"advises a stop ('{stop}')" if stop else None))
    negated = followed(_NEGATORS, _HALTS, _NEGATION_REACH)
    found.append(("negated_stop", f"tells the agent not to stop ('{negated[0]} {negated[1]}')" if negated else None))
    loop = phrase(_LOOPS)
    found.append(("loop", f"advises an unbounded retry ('{loop}')" if loop else None))
    override = phrase(_OVERRIDES)
    found.append(("override", f"advises an override ('{override}')" if override else None))
    weakened = verb_object(_WEAKEN, _GUARDS)
    found.append(("weaken", f"'{weakened[0]}' near '{weakened[1]}'" if weakened else None))
    claim = followed(_CLAIM_AS, _SUCCESS, _TREAT_REACH) or followed(_CLAIM_SAY, _SUCCESS, _SAY_REACH)
    found.append(("success_claim", f"claims success ('{claim[0]}' ... '{claim[1]}')" if claim else None))
    return [(rule, reason) for rule, reason in found if reason]


def _lint_field(field: str, text: str, max_chars: int, skill: str) -> list[tuple[str, str]]:
    """(rule, "<field>: <reason>") for every rule `text` breaks ([] = clean)."""
    if not isinstance(text, str):
        return [("empty", f"{field}: empty")]
    if len(text) > max_chars:  # before any pattern: an oversized field costs nothing
        return [("length", f"{field}: {len(text)} chars (max {max_chars})")]
    if not text.strip():
        return [("empty", f"{field}: empty")]
    character = _character_problem(text)
    if character:
        return [("character", f"{field}: {character}")]
    return [(rule, f"{field}: {reason}") for rule, reason in _structure_problems(text, skill)]


def _note_problems(note: Note) -> list[tuple[str, str]]:
    return (_lint_field("title", note.title, TITLE_MAX_CHARS, note.skill)
            + _lint_field("advice", note.advice, ADVICE_MAX_CHARS, note.skill))


def lint_advice(text: str, *, skill: str = "") -> list[str]:
    """Why `text` may not be served as advice ([] = clean). `skill` is the note's
    own skill, the only tool outside `READ_ONLY_TOOLS` the text may name."""
    return [reason for _, reason in _lint_field("advice", text, ADVICE_MAX_CHARS, skill)]


def lint_note(note: Note) -> list[str]:
    """`lint_advice` over every field a note renders: its title and its advice."""
    return [reason for _, reason in _note_problems(note)]


# ── provenance tag ────────────────────────────────────────────────────────


def _verified(verifier) -> tuple[int, int] | None:
    """(passes, seeds) of a verifier PASS, else None. A pass is the verdict
    "verified" on the Newton engine (`VERIFIER_ENGINE`, the owner's rule for every
    sim run) over exactly `VERIFIER_SEEDS` distinct integer seeds with
    `VERIFIER_MIN_PASSES` <= passes <= seeds, an integer `attempts` from 1 to
    `VERIFIER_MAX_ATTEMPTS` and no `gaps` (absent or []) -- DECISIONS Q6. Anything
    else -- a missing or other engine, bool/float/None values, another seed count,
    too few or too many passes, more attempts, a gap -- fails closed."""
    if not isinstance(verifier, dict):
        return None
    if verifier.get("verdict") != "verified" or verifier.get("engine") != VERIFIER_ENGINE:
        return None
    passes, seeds, attempts = verifier.get("passes"), verifier.get("seeds"), verifier.get("attempts")
    if type(passes) is not int or not isinstance(seeds, list) or len(seeds) != VERIFIER_SEEDS:
        return None
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        return None
    if not VERIFIER_MIN_PASSES <= passes <= len(seeds):
        return None
    if type(attempts) is not int or not 1 <= attempts <= VERIFIER_MAX_ATTEMPTS:
        return None
    if verifier.get("gaps", []) != []:
        return None
    return passes, len(seeds)


def _refuted(verifier) -> bool:
    return isinstance(verifier, dict) and verifier.get("verdict") == "refuted"


def note_tag(note: Note, sha: str) -> str:
    """`learned in <where> · <verifier part> · lib <sha8>` (`authored · ...` for
    hand-written notes); the verifier part is `verifier <passes>/<seeds>` only for
    a verifier pass (`_verified`), else `not verifier-tested`."""
    verdict = _verified(note.verifier)
    tested = f"verifier {verdict[0]}/{verdict[1]}" if verdict else "not verifier-tested"
    origin = "authored" if note.kind == "authored" else f"learned in {note.learned_in}"
    return f"{origin} · {tested} · lib {str(sha)[:8]}"


# ── snapshot ──────────────────────────────────────────────────────────────


def _serves(note: Note, profile_class: str, min_tasks: int) -> bool:
    """For a lint-clean note, on every class: never refuted or retired. Harvested
    advice (any kind but "authored") only once promoted: `min_tasks` distinct tasks
    AND a verifier pass, on every class. An authored note follows the task counters
    on "sim"; "real" needs the verifier pass."""
    if note.status in _NEVER_SERVED or _refuted(note.verifier):
        return False
    promoted = note.n_tasks >= min_tasks
    passed = _verified(note.verifier) is not None
    if note.kind == "authored":
        return promoted if profile_class == "sim" else passed
    return promoted and passed


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
    """The library as read once at build: every parsed note, in rank order the ones
    this profile class serves, and `skipped`: {"id", "reason"} for every `*.md`
    entry that is no lint-clean note (the lint rule's name, or file_name, symlink,
    not_a_file, unreadable, parse), by file name -- a skipped note is never silent."""

    root: Path | None
    sha: str
    profile_class: str
    notes: tuple[Note, ...]
    served: tuple[Note, ...]
    skipped: tuple[dict, ...] = ()

    def _block(self) -> tuple[str, tuple[str, ...]]:
        """(block text, ids of the notes it renders)."""
        items = [[n.id, n.title, n.advice, note_tag(n, self.sha)] for n in self.served[:BLOCK_NOTES]]
        for _ in range(2 * len(items) + 1):  # each pass fits, cuts one advice to fit, or drops one note
            if not items:
                break
            text = _render_block([(title, advice, tag) for _, title, advice, tag in items])
            excess = len(text) - BLOCK_MAX_CHARS
            if excess <= 0:
                return text, tuple(item[0] for item in items)
            k = max(range(len(items)), key=lambda j: len(items[j][2]))
            cut = _cut(items[k][2], len(items[k][2]) - excess)
            if cut:
                items[k][2] = cut
            else:
                del items[k]  # not one whole word of its advice fits: drop the note, never render it empty
        return "", ()

    def block(self) -> str:
        """The prompt block (spec §4.3, without the LEARNED.md header line): the top
        `BLOCK_NOTES` served notes in at most `BLOCK_MAX_CHARS` characters, the
        longest advice cut by whole words when needed and a note dropped when none
        of its advice fits; "" when nothing is served or the block cannot fit."""
        return self._block()[0]

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
                "block_notes": list(self._block()[1])}


def _printable_id(name: str) -> str:
    """A file name's stem as printable ASCII (escapes for anything else)."""
    return ascii(name.removesuffix(".md"))[1:-1]


def load_snapshot(root: Path | None, profile_class: str, *, min_tasks: int = 2) -> LibrarySnapshot:
    """Read the library at `root` once, read-only (a missing directory is an empty
    library, never created). Every class serves only lint-clean notes and never a
    refuted or retired one. Harvested notes need `min_tasks` distinct tasks AND a
    verifier pass on every class; authored notes need the task count on "sim" and
    the verifier pass on any other class ("real"). A `min_tasks` that is not an
    integer >= 1 (a missing or broken `memory.skill_min_tasks`) serves nothing.
    Every `*.md` entry that is no lint-clean note is listed in `skipped` and logged
    (one warning line each) with its reason."""
    root = None if root is None else Path(root).expanduser()
    entries = _scan(root)
    notes, clean, skipped = [], [], []
    for name, data, why in entries:
        detail = ""
        if data is not None:
            note = _parse_bytes(name[: -len(".md")], data)
            if note is None:
                why = "parse"
            else:
                notes.append(note)
                problems = _note_problems(note)
                if not problems:
                    clean.append(note)
                    continue
                why, detail = problems[0]
        entry = {"id": _printable_id(name), "reason": why}
        skipped.append(entry)
        log.warning("ASPIRE notes: skipped %s (%s)%s", entry["id"], why, f": {detail}" if detail else "")
    klass = "sim" if profile_class == "sim" else "real"
    if type(min_tasks) is int and min_tasks >= 1:
        served = tuple(sorted((n for n in clean if _serves(n, klass, min_tasks)), key=_rank))
    else:
        log.warning("ASPIRE notes: min_tasks %r is not an integer >= 1 (memory.skill_min_tasks); "
                    "serving no note", min_tasks)
        served = ()
    return LibrarySnapshot(root=root, sha=_sha_of(entries), profile_class=klass, notes=tuple(notes),
                           served=served, skipped=tuple(skipped))
