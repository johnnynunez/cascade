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
skipped, never cut into acceptance). Only regular files (never a symlink) of at most
`NOTE_MAX_BYTES` whose name is a safe id (`_NOTE_NAME_RE`) are notes, hashed or
served; every other `*.md` entry and every note that fails the lint is listed in the
snapshot's `skipped` with its reason and logged, never dropped silently.

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
import logging
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..config import PACKAGE_ROOT
from .library import SkillLibrary, read_note_text

log = logging.getLogger(__name__)

ADVICE_MAX_CHARS = 300
TITLE_MAX_CHARS = 80
BLOCK_MAX_CHARS = 1400
LEARNED_MAX_BYTES = 2000
KNOWLEDGE_MAX_CHARS = 6000
#: a note file larger than this is skipped (`too_large`) on its fstat size, before any read
NOTE_MAX_BYTES = 64 * 1024
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
    too) and never blocks on a FIFO (O_NONBLOCK, then fstat). A file over
    `NOTE_MAX_BYTES` is `too_large` on its fstat size, before any read; the read stops
    one byte past the cap too, so a file that grows after the fstat is never read whole."""
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return None, "symlink" if exc.errno == errno.ELOOP else "unreadable"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "not_a_file"
        if st.st_size > NOTE_MAX_BYTES:
            return None, "too_large"
        chunks, size = [], 0
        while size <= NOTE_MAX_BYTES and (chunk := os.read(fd, NOTE_MAX_BYTES + 1 - size)):
            chunks.append(chunk)
            size += len(chunk)
        if size > NOTE_MAX_BYTES:
            return None, "too_large"
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
    non-UTF-8 byte), is a regular file of at most `NOTE_MAX_BYTES` and no symlink (lstat,
    then an O_NOFOLLOW read) and is readable; only note files are hashed, so the hash's
    line format stays unambiguous."""
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
#   3. structure: every word lower, Title or UPPER case; no hard stem (`_HARD_STEMS`:
#      reset, emergency stop, halt motion, the override verbs, start over, system prompt)
#      however it is split -- a one-word stem inside one word's letters or joined from
#      whole neighbouring words, a phrase stem anywhere in the field's letters-only
#      skeleton (`_hard_stems`); then over the
#      field read as words (`_View`) -- (a) the words split at camelCase humps, digits
#      and any separator, (b) the unsplit runs of letters and digits, and (c) at each
#      word the 2-3 words that join into one whole phrase word (`dis able`, `DIS-able`)
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
#: a role marker: one of these followed by ":" or a dash ("Assistant - Done"), or by a
#: `_SPEECH` verb and ":" ("user said:")
_ROLE_WORDS = frozenset({"assistant", "user", "system", "developer", "human", "tool", "observation", "function",
                         "output", "operator", "owner"})
_SPEECH = frozenset({"said", "says"})
#: the two-word role markers: followed by ":" at the field's start or after a sentence break
_ROLE_PAIRS = frozenset({("function", "result"), ("tool", "result")})
#: chat-template and prompt-injection markers, refused wherever they stand (and "system
#: prompt" through `_HARD_STEMS`)
_TEMPLATE = (("im", "start"), ("im", "end"), ("developer", "mode*"), ("new", "instruction*"))
#: letters-only stems, each with the rule it refuses under: reset_scene / reset_stop,
#: emergency_stop / halt_motion in any spelling, the override verbs, "start over" and
#: "system prompt" (`_hard_stems`). A one-word stem is read inside each whitespace-delimited
#: word, every non-letter deleted ("Re/set", "Ig+nore", "DIS;regard"), and across whole
#: neighbouring words that join into it plus a plain ending ("re set", "ig no re", "by pass":
#: `_joined_stems`), never across an arbitrary word boundary ("before setting", "sign or",
#: "over ridges"). A word that merely contains one is refused too ("Resettle", "Ignore the
#: crumbs": accepted over-blocks)
_HARD_STEMS = (("reset", "reset"), ("emergencystop", "stop"), ("haltmotion", "stop"), ("ignor", "override"),
               ("disregard", "override"), ("bypass", "override"), ("circumvent", "override"),
               ("overrid", "override"), ("startover", "loop"), ("systemprompt", "role_marker"))
#: the phrase stems: read across words, anywhere in the field's letters-only skeleton
#: (`_skeleton`: "Emer gency stop", "SYSTEMPROMPT", "restart over")
_PHRASE_STEMS = frozenset({"emergencystop", "haltmotion", "startover", "systemprompt"})
#: whole words left out of every stem reading (their neighbours still join): common words
#: that carry a one-word stem ("preset", "pre-set") or its first letters ("you're set")
_SKELETON_ALLOW = frozenset({"preset", "presets", "pre-set", "pre-sets", "you're", "we're", "they're"})
#: phrasings that only mean a reset (and "reset" through `_HARD_STEMS`): a reset clears
#: the retry ledger (R4)
_RESET = (("respawn*",), ("begin*", "again"), ("start*", "from", "scratch"))
#: a reset only with a `_SCENE` noun as their object ("restart the scene", "redo everything")
_RESET_VERBS = (("restart*",), ("reload*",), ("redo*",), ("redid",), ("reinitialis*",), ("reinitializ*",),
                ("restor*",))
_SCENE = (("scene*",), ("layout*",), ("table*",), ("everything",), ("kitchen*",), ("world*",), ("simulation*",))
_PROVENANCE = (("verifier*",), ("verified",), ("promoted",), ("provenance",), ("authored",), ("passing", "seed*"),
               ("tested", "in"), ("tested", "on"), ("tested", "by"), ("newton*",), ("physx",))
_LEARNED = (("learn*", "in"), ("learn*", "on"))
_STOPS = (("emergenc*", "stop*"), ("e", "stop*"), ("estop*",), ("halt*", "motion*"), ("halt*", "the", "motion*"),
          ("stop", "latch*"), ("emergenc*", "button*"))
_NEGATORS = (("never",), ("not",), ("no",), ("don",), ("dont",), ("without",), ("cannot",))
_HALTS = (("stop*",), ("halt*",), ("paus*",))
_LOOPS = (("try", "again", "until"), ("repeat*", "until"), ("until", "it", "succe*"),
          ("until", "it", "work*"), ("until", "succe*"), ("keep", "call*"), ("keep", "try*"), ("keep", "retr*"),
          ("keep", "going"), ("retr*", "until"), ("retr*", "till"), ("forever",), ("over", "and", "over"),
          ("never", "give", "up"), ("no", "limit*"), ("as", "many", "times"))
#: refused when one of `_GUARDS` is in their direct object (`_direct_object`)
_WEAKEN = (("disabl*",), ("turn*", "off"), ("switch*", "off"), ("shut*", "off"), ("shut*", "down"), ("skip*",),
           ("lower*",), ("relax*",), ("reduc*",), ("shrink*",), ("rais*",), ("smaller",), ("decreas*",),
           ("loosen*",), ("suppress*",), ("silenc*",), ("mute*",), ("weaken*",), ("work*", "around"),
           ("get*", "around"))
#: verbs with everyday kitchen objects ("cut the safety seal"): refused only when a
#: `_GUARDS` noun is the HEAD of their direct object (`_ends_object`)
_WEAKEN_HEAD = (("zero*",), ("waiv*",), ("tighten*",), ("neglect*",), ("dismiss*",), ("forget*",), ("forgot*",),
                ("overrul*",), ("defeat*",), ("sidestep*",), ("never", "mind"), ("set*", "aside"), ("look*", "past"),
                ("get*", "past"), ("pay*", "no", "attention", "to"))
#: removal verbs: refused like `_WEAKEN_HEAD`, and when a `_GUARDS` word MODIFIES a
#: `_GUARDED_HEADS` head noun (`_guarded_head`): "drop the safety system", not "drop the
#: warning label", "cut the safety seal" or "remove the guard ring"
_REMOVE = (("cut*",), ("drop*",), ("remov*",), ("tak*", "off"))
#: the safety head nouns a guard word may not modify after a `_REMOVE` verb (plurals too)
_GUARDED_HEADS = frozenset({"system", "feature", "rule", "check", "gate", "interlock", "limit", "margin", "stop",
                            "sensor", "monitor"})
#: phrasal verbs split around their object ("turn the gate off", "shut the interlock
#: down"): refused when a `_GUARDS` noun is the object's head and the particle follows
_PHRASAL_OFF = (("turn*",), ("switch*",), ("shut*",))
_PHRASAL_DOWN = (("shut*",),)
#: guard words; the stop guard is the word "stop" / "stops" only ("the stopper", "the
#: stopwatch" are kitchen nouns), while "guard*" stays (accepted over-block: "the blade guard")
_GUARDS = (("gate*",), ("check*",), ("refus*",), ("warn*",), ("veto*",), ("clearanc*",), ("margin*",),
           ("limit*",), ("safet*",), ("collis*",), ("collid*",), ("stop",), ("stops",), ("halt*",), ("guard*",),
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
          ("thou",), ("mil",), ("mils",), ("pct",),  # "per cent" joins into "percent" (`_View`)
          ("mms",), ("cms",), ("kms",), ("mrads",), ("dms",), ("ums",), ("nm",), ("nms",), ("nanomet*",),
          ("mile",), ("miles",), ("arcmin*",), ("grad",), ("grads",))
#: a unit only right after a number ("3 m", "2 in"; and "50%")
_NUMBER_UNITS = frozenset({"m", "in"})
#: a unit when a number (digits or words: "half", "quarter" too) or a `_TURN_QUALIFIERS`
#: word stands among the `_TURN_REACH` words before it, in its clause ("half a turn", "one
#: full turn", "three quarters of a turn", "2 more turns"; not "wait your turn")
_TURNS = frozenset({"turn", "turns"})
_TURN_QUALIFIERS = frozenset({"full", "quarters"})
_TURN_REACH = 3
#: every family a word or a join of words is read against (`_View`)
_FAMILIES = (_TEMPLATE, _RESET, _RESET_VERBS, _SCENE, _PROVENANCE, _LEARNED, _STOPS, _NEGATORS, _HALTS, _LOOPS,
             _WEAKEN, _WEAKEN_HEAD, _REMOVE, _PHRASAL_OFF, _GUARDS, _CLAIM_AS, _CLAIM_SAY, _SUCCESS, _UNITS)
#: spelled-out numbers: two in one field ("one" aside: "pick the one on top"), two side
#: by side ("minus one", "twenty-one") or one after an axis letter is refused; "point" is
#: a number only next to a number ("point five", "2 point 5"; not "point at the bowl")
_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "point", "half", "quarter", "minus", "negative"})
_POINT = "point"
#: tool parameters a note may not set: refused before ":" (`name:`) wherever they stand
_PARAMETERS = frozenset({"object", "destination", "target", "x", "y", "z", "arm", "force", "speed", "frame", "pose",
                         "position", "grasp", "offset"})
#: every other single-word parameter of a tool (TOOL_SPECS and the MCP server's own tools,
#: frozen here: a new tool parameter never changes which notes are served): an argument
#: only in a field with at least `_NAME_PAIRS` `name:` pairs ("Side: grasp across" is a title)
_TOOL_PARAMETERS = frozenset({
    "amount", "bindings", "camera", "cycles", "degrees", "direction", "joint", "k", "label", "limit", "material",
    "n", "name", "normalized", "program", "query", "reason", "relation", "side", "spec", "steps", "success",
    "summary", "task", "turns", "u", "v", "vector"})
_NAME_PAIRS = 2
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
#: closed-class words that end a noun phrase (as do punctuation, a clause end, a
#: preposition, a determiner, a number and any "-ly" adverb): the word before is the
#: phrase's HEAD noun (`_ends_object`; "the safety check", not "the safety seal")
_OBJECT_TAIL = frozenset({
    "off", "on", "out", "up", "down", "away", "back", "aside", "first", "again", "now", "next", "too", "also",
    "here", "there", "anyway", "altogether", "instead", "once", "twice", "today", "always", "never", "ever",
    "still", "already", "just", "only", "even", "more", "less", "please",
    "if", "when", "whenever", "while", "because", "unless", "until", "till", "since", "as", "though",
    "although", "before", "after", "where", "or", "nor", "yet", "than", "which", "who",
    "it", "them", "they", "you", "we", "i", "he", "she", "him", "us", "me",
    "is", "are", "was", "were", "be", "been", "can", "could", "will", "would", "should", "may", "might",
    "must", "do", "does", "did", "has", "have", "had",
    # prepositions that end a noun phrase but do not stop the object scan (`_PREPOSITIONS`
    # do): "remove the safety check of the arm", yet "skip part of the check"
    "of", "about", "during", "without", "like", "among", "despite", "except", "per", "throughout", "amid",
    "regarding", "besides"})
#: a split phrasal verb's particle: among the words right after its object's head
_PARTICLE_REACH = 4  # "turn the clearance gate fully off"
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
#: the one-word stems, and their first letters short of the whole stem (a join of whole
#: words goes on while it spells only these)
_WORD_STEMS = tuple(stem for stem, _ in _HARD_STEMS if stem not in _PHRASE_STEMS)
_STEM_STARTS = frozenset(stem[:k] for stem in _WORD_STEMS for k in range(1, len(stem)))


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


_SKELETON_LETTERS_RE = re.compile(r"[a-z]+")


def _word_runs(text: str) -> list[list[str]]:
    """The runs of letters of each whitespace-delimited word of the field (NFKC, lower
    case), without the `_SKELETON_ALLOW` words (left out whole, so the words around one
    still join)."""
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'")
    return [_SKELETON_LETTERS_RE.findall(word) for word in text.split()
            if word.strip(_EDGE_PUNCTUATION) not in _SKELETON_ALLOW]


def _skeleton(text: str) -> str:
    """The field's letters only: NFKC, lower case, every space, punctuation mark and
    digit deleted, the `_SKELETON_ALLOW` words left out (the phrase stems' reading)."""
    return "".join(run for runs in _word_runs(text) for run in runs)


def _joined_stems(runs: list[str]) -> set[str]:
    """The one-word stems that whole neighbouring runs of letters join into, read from a
    run's first letter and ending in a plain ending (`_ENDINGS`): "re set", "Ig. nore",
    "ci rc um ve nt", "by passing" -- never from inside a word, so "are set", "sign or"
    and "over ridges" join into nothing. Each join stops once it can no longer spell one."""
    found: set[str] = set()
    for i, joined in enumerate(runs):
        for k in range(i + 1, len(runs)):
            joined += runs[k]
            if joined in _STEM_STARTS:
                continue
            stem = next((s for s in _WORD_STEMS if joined.startswith(s)), None)
            if stem is None or len(joined) > len(stem) + _ENDING_SIZES[-1]:
                break
            if joined[len(stem):] in _ENDING_SET:
                found.add(stem)
                break
    return found


def _hard_stems(text: str) -> dict[str, str]:
    """rule -> the first `_HARD_STEMS` stem the field spells: a phrase stem anywhere in its
    letters-only skeleton, a one-word stem inside the letters of one word or joined from
    whole neighbouring words (`_joined_stems`)."""
    by_word = _word_runs(text)
    words = ["".join(runs) for runs in by_word]
    skeleton, apart = "".join(words), " ".join(words)
    joined = _joined_stems([run for runs in by_word for run in runs])
    found: dict[str, str] = {}
    for stem, rule in _HARD_STEMS:
        if stem in (skeleton if stem in _PHRASE_STEMS else apart) or stem in joined:
            found.setdefault(rule, stem)
    return found


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
    `breaks[i]` is what separates word i from the word before it (`_break`),
    `live` the positions where some phrase of some family can start (`_heads`) and
    `by_key` those positions by the first letter of their readings."""

    __slots__ = ("breaks", "by_key", "cands", "live", "words")

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
        self.by_key: dict[str, list[int]] = {}
        for i in self.live:
            for key in {word[0] for word, _ in self.cands[i]}:
                self.by_key.setdefault(key, []).append(i)


class _Phrases:
    """A family of phrases, compiled: each part is (text, is_stem); a word can start
    a match only if it equals an exact first part or starts with a stem one, so only at
    a live position whose reading starts with a first part's first letter (`starts`)."""

    def __init__(self, phrases):
        self.phrases = tuple(tuple((p[:-1], True) if p.endswith("*") else (p, False) for p in phrase)
                             for phrase in phrases)
        self.exact = frozenset(ph[0][0] for ph in self.phrases if not ph[0][1])
        self.stems = tuple(ph[0][0] for ph in self.phrases if ph[0][1])
        self.keys = frozenset(ph[0][0][0] for ph in self.phrases)

    def starts(self, view: _View) -> list[int]:
        """The live positions of `view` where a phrase of this family can start, in order."""
        found = [view.by_key[key] for key in self.keys if key in view.by_key]
        return found[0] if len(found) == 1 else sorted(set().union(*found))

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
    for i in family.starts(view):
        n = family.at(view, i)
        if n:
            return " ".join(view.words[i:i + n])
    return None


def _followed(view: _View, heads, tails, reach: int) -> tuple[str, str] | None:
    """The first (head, tail) where a `tails` phrase starts among the `reach` words
    right after a `heads` phrase; None when there is none."""
    head_family, tail_family = _compiled(heads), _compiled(tails)
    for i in head_family.starts(view):
        n = head_family.at(view, i)
        if not n:
            continue
        for j in range(i + n, min(i + n + reach, len(view.words))):
            m = tail_family.at(view, j)
            if m:
                return " ".join(view.words[i:i + n]), " ".join(view.words[j:j + m])
    return None


def _ends_object(view: _View, k: int) -> bool:
    """Word `k` ends the noun phrase before it, so the word before is that phrase's head:
    the end of the field, punctuation, a clause end, a preposition, a determiner, an
    `_OBJECT_TAIL` word, an "-ly" adverb or a number ("the safety check first"; in "the
    safety seal" the guard word is no head)."""
    if k >= len(view.words) or view.breaks[k]:
        return True
    word = view.words[k]
    return (word in _CLAUSE_ENDS or word in _PREPOSITIONS or word in _DETERMINERS or word in _OBJECT_TAIL
            or word.endswith("ly") or _is_number(word))


def _guarded_head(view: _View, k: int) -> int:
    """The end of the noun phrase that goes on at word `k` (after a guard word used as a
    modifier) when its head noun is a `_GUARDED_HEADS` word or its plural ("the safety
    system"), else 0 ("the safety seal")."""
    for m in range(k + 1, k + 1 + _OBJECT_REACH):
        if _ends_object(view, m):
            noun = view.words[m - 1]
            return m if noun in _GUARDED_HEADS or (noun.endswith("s") and noun[:-1] in _GUARDED_HEADS) else 0
    return 0


def _direct_object(view: _View, start: int, nouns: _Phrases, head: bool = False,
                   modifier: bool = False) -> tuple[int, int] | None:
    """(position, size) of a `nouns` phrase in the object that starts at word
    `start` -- among its first `_OBJECT_REACH` words, before a preposition and the end
    of the clause (punctuation, `_CLAUSE_ENDS`); with `head`, only as the HEAD noun of
    its phrase (`_ends_object`), and with `modifier` also as a modifier of a safety head
    noun (`_guarded_head`; then the size runs to that head). "and" or a comma before a
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
        if size and (not head or _ends_object(view, j + size)):
            return j, size
        end = _guarded_head(view, j + size) if size and modifier else 0
        if end:
            return j, end - j
        j, seen = j + 1, seen + 1
    return None


def _verb_object(view: _View, verbs, nouns, head: bool = False, modifier: bool = False) -> tuple[str, str] | None:
    """The first (verb, noun) where a `nouns` phrase is the direct object (with `head`:
    the object's head noun; with `modifier`: or a modifier of a safety head noun) of a
    `verbs` phrase."""
    verb_family, noun_family = _compiled(verbs), _compiled(nouns)
    for i in verb_family.starts(view):
        size = verb_family.at(view, i)
        if size:
            hit = _direct_object(view, i + size, noun_family, head, modifier)
            if hit:
                return " ".join(view.words[i:i + size]), " ".join(view.words[hit[0]:hit[0] + hit[1]])
    return None


def _split_phrasal(view: _View) -> tuple[str, str] | None:
    """The first ('<verb> ... <particle>', noun) where a split phrasal verb's object has a
    `_GUARDS` head and its particle follows among the next `_PARTICLE_REACH` words, before
    any punctuation: "turn the clearance gate off", "shut the safety interlock down"."""
    nouns = _compiled(_GUARDS)
    for verbs, particle in ((_PHRASAL_OFF, "off"), (_PHRASAL_DOWN, "down")):
        family = _compiled(verbs)
        for i in family.starts(view):
            size = family.at(view, i)
            hit = _direct_object(view, i + size, nouns, head=True) if size else None
            if not hit:
                continue
            end = hit[0] + hit[1]
            for k in range(end, min(end + _PARTICLE_REACH, len(view.words))):
                if view.breaks[k]:
                    break
                if view.words[k] == particle:
                    return f"{' '.join(view.words[i:i + size])} ... {particle}", " ".join(view.words[hit[0]:end])
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


def _number_flags(words: list[str]) -> list[bool]:
    """Which words are numbers: digits and `_NUMBER_WORDS`, "point" only next to another
    number ("point five", "2 point 5"; "point at the bowl" is a verb)."""
    plain = [word.isdigit() or (word in _NUMBER_WORDS and word != _POINT) for word in words]
    return [plain[k] or (word == _POINT and ((k > 0 and plain[k - 1]) or (k + 1 < len(words) and plain[k + 1])))
            for k, word in enumerate(words)]


def _adjacent(text: str, spans, first, second, gap: str = " -\u2013") -> str | None:
    """'<a> <b>' for the first two neighbouring words k, k + 1 with first(k) and
    second(k + 1), separated only by `gap` characters."""
    for k in range(len(spans) - 1):
        (a, _, end), (b, start, _) = spans[k], spans[k + 1]
        if first(k) and second(k + 1) and set(text[end:start]) <= set(gap):
            return f"{a} {b}"
    return None


def _score(words: list[str]) -> str | None:
    """'out of <n>' or '<n> of <n>' ("five out of five", "4 of 5"): a verifier score."""
    for a, b, c in zip(words, words[1:], words[2:]):
        if b == "of" and _is_number(c) and (a == "out" or _is_number(a)):
            return f"{a} of {c}"
    return None


def _role_marker(text: str, spans) -> str | None:
    """'<role>:' -- a `_ROLE_WORDS` word followed by ':' or a dash, or by a `_SPEECH`
    verb and ':' ("user said:"); or an exact two-word marker (`_ROLE_PAIRS`, "tool
    result:") at the field's start or after a sentence break. Any other role word + word
    + ':' is plain English ("Use the tool rack: ...", "Ask the user why: ...")."""
    for k, (word, start, end) in enumerate(spans):
        if word not in _ROLE_WORDS:
            continue
        after = _next_char(text, end)
        if after == ":" or (after and after in "\u2013\u2014") or (after == "-" and text[end] == " "):
            return f"{word}:"
        if k + 1 == len(spans):
            continue
        nxt, nxt_start, nxt_end = spans[k + 1]
        if text[end:nxt_start].strip(" ") or _next_char(text, nxt_end) != ":":
            continue
        opens = k == 0 or _break(text[spans[k - 1][2]:start]) == "."
        if nxt in _SPEECH or ((word, nxt) in _ROLE_PAIRS and opens):
            return f"{word} {nxt}:"
    return None


def _measurement(text: str, numbers) -> str | None:
    """'<number> <unit>': a number followed by "%" or a `_NUMBER_UNITS` word."""
    for m in numbers:
        i = m.end()
        while i < len(text) and text[i] == " ":
            i += 1
        if text.startswith("%", i):
            return f"{m.group(0)}%"
        unit = _LETTERS_RE.match(text, i)
        if unit and unit.group(0).lower() in _NUMBER_UNITS:
            return f"{m.group(0)} {unit.group(0)}"
    return None


def _qualified_turn(view: _View, number: list[bool]) -> str | None:
    """'<qualifier> ... turn': a `_TURNS` word with a number (digits or words) or a
    `_TURN_QUALIFIERS` word among the `_TURN_REACH` words before it, in its clause."""
    words, breaks = view.words, view.breaks
    for k, word in enumerate(words):
        if word not in _TURNS:
            continue
        for j in range(k - 1, max(k - 1 - _TURN_REACH, -1), -1):
            if breaks[j + 1] or words[j] in _CLAUSE_ENDS:
                break
            if number[j] or words[j] in _TURN_QUALIFIERS:
                return " ".join(words[j:k + 1])
    return None


def _identifier_problems(text: str, skill: str) -> list[tuple[str, str]]:
    """Identifiers (unsplit `_IDENT_RE` runs, any case): every snake_case one is a
    tool, read-only or the note's own skill and never a forbidden one; no identifier
    or `_PARAMETERS` name is followed by ':' (arguments), nor a `_TOOL_PARAMETERS` name
    in a field with `_NAME_PAIRS` or more `name:` pairs ("=" is a refused character)."""
    tools = _tool_names()
    found: dict[str, str] = {}
    idents = [(m, _next_char(text, m.end()) == ":") for m in _IDENT_RE.finditer(text)]
    pairs = sum(colon for _, colon in idents)
    for m, colon in idents:
        raw = m.group(0)
        name = raw.lower()
        if "_" in name:
            if name in FORBIDDEN_TOOLS:
                found.setdefault("tool", f"names {name}, which a note may never name")
            elif name not in tools:
                found.setdefault("identifier", f"{raw} is not a tool name")
            elif name != skill and name not in READ_ONLY_TOOLS:
                found.setdefault("tool", f"names {name}, which is neither read-only nor this note's skill")
        if colon and ("_" in name or name in _PARAMETERS or (name in _TOOL_PARAMETERS and pairs >= _NAME_PAIRS)):
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
    number = _number_flags(words)
    hidden = _hard_stems(text)
    found: list[tuple[str, str | None]] = []

    def phrase(family) -> str | None:
        return next((hit for view in views if (hit := _first(view, family))), None)

    def verb_object(verbs, nouns, head: bool = False, modifier: bool = False) -> tuple[str, str] | None:
        return next((hit for view in views if (hit := _verb_object(view, verbs, nouns, head, modifier))), None)

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
    template = hidden.get("role_marker") or phrase(_TEMPLATE)
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
    def number_word(k: int) -> bool:
        return number[k] and not words[k].isdigit()

    counted = len(numbers) + sum(1 for k, w in enumerate(words) if number_word(k) and w != "one")
    spelled = _adjacent(text, spans, number_word, number_word)
    found.append(("numbers", f"carries {counted} numbers (at most one)" if counted > 1
                  else f"spells out the number '{spelled}'" if spelled else None))
    measured = _measurement(text, numbers) or _qualified_turn(views[0], number)
    found.append(("measurement", f"carries the measurement '{measured}'" if measured else None))
    unit = phrase(_UNITS)
    found.append(("unit", f"names the unit '{unit}'" if unit else None))
    coordinate = _adjacent(text, spans, lambda k: words[k] in ("x", "y", "z"), number.__getitem__,
                           " :=+-\u2013\u2014/")
    found.append(("coordinate", f"carries the coordinate '{coordinate}'" if coordinate else None))
    # tools: read-only ones and the note's own skill, as identifiers, without arguments
    found.append(("tool", identifiers.get("tool")))
    found.append(("arguments", identifiers.get("arguments")))
    # advice a note may never give
    reset = hidden.get("reset") or phrase(_RESET)
    if not reset and (scene := verb_object(_RESET_VERBS, _SCENE)):
        reset = f"{scene[0]} ... {scene[1]}"
    found.append(("reset", f"advises a reset ('{reset}')" if reset else None))
    stop = hidden.get("stop") or phrase(_STOPS)
    found.append(("stop", f"advises a stop ('{stop}')" if stop else None))
    negated = followed(_NEGATORS, _HALTS, _NEGATION_REACH)
    found.append(("negated_stop", f"tells the agent not to stop ('{negated[0]} {negated[1]}')" if negated else None))
    loop = hidden.get("loop") or phrase(_LOOPS)
    found.append(("loop", f"advises an unbounded retry ('{loop}')" if loop else None))
    override = hidden.get("override")
    found.append(("override", f"advises an override ('{override}')" if override else None))
    weakened = (verb_object(_WEAKEN, _GUARDS) or verb_object(_WEAKEN_HEAD, _GUARDS, head=True)
                or verb_object(_REMOVE, _GUARDS, head=True, modifier=True)
                or next((hit for view in views if (hit := _split_phrasal(view))), None))
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
    not_a_file, too_large, unreadable, parse), by file name -- a skipped note is never
    silent. `error` says why the library could not be read ("Exc: msg"; then `root` is
    None and nothing is served), None when it was read or does not exist yet."""

    root: Path | None
    sha: str
    profile_class: str
    notes: tuple[Note, ...]
    served: tuple[Note, ...]
    skipped: tuple[dict, ...] = ()
    error: str | None = None

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


# ── the library a configuration serves ─────────────────────────────────────

#: The library of a profile whose `memory` names no `skill_library_path`: the
#: repo's `skills_library/` that scripts/learn_from_runs.py fills (spec §4.2). Read
#: each time a snapshot is taken, so the test suite pins it to a path that never
#: exists (preflight R24) and no machine's own notes reach a test runtime.
DEFAULT_LIBRARY = PACKAGE_ROOT / "skills_library"

_ABSENT = object()


def library_root(cfg) -> Path:
    """The library directory `cfg` names: `memory.skill_library_path` (`~`
    expanded), or `DEFAULT_LIBRARY` when the key is absent. Raises ValueError for
    a value that is no path (null, empty, not a string), a RELATIVE path (it would
    resolve against whatever directory the process started in, e.g. a relative
    CASCADE_ASPIRE_LIBRARY) or a missing or non-mapping `memory` section;
    `config_snapshot` fails closed on it."""
    memory = cfg.get("memory")
    try:
        value = memory.get("skill_library_path", _ABSENT)
    except AttributeError:
        raise ValueError(f"memory is not a mapping: {type(memory).__name__}") from None
    if value is _ABSENT:
        return DEFAULT_LIBRARY
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"not a path: {value!r}")
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError(f"relative path {value!r}: a note library path must be absolute "
                         "(CASCADE_ASPIRE_LIBRARY / memory.skill_library_path), never read from the working directory")
    return root


def _dangling_link(root: Path) -> Path | None:
    """`root` or the first directory above it that is a symlink to a missing target
    (an unmounted volume), else None."""
    for path in (root, *root.parents):
        if os.path.islink(path) and not os.path.exists(path):
            return path
    return None


def _unread(klass: str, root: Path | None, exc: BaseException) -> LibrarySnapshot:
    """The snapshot of a library that could not be read: no root, the empty
    library's sha, nothing served, the reason in `error` and one warning line."""
    error = f"{type(exc).__name__}: {exc}"
    log.warning("ASPIRE notes: library %s (memory.skill_library_path) not read (%s); serving no note",
                "?" if root is None else root, error)
    return LibrarySnapshot(root=None, sha=library_sha(None), profile_class=klass, notes=(), served=(), error=error)


def config_snapshot(cfg, profile_class: str) -> LibrarySnapshot:
    """The snapshot a runtime built from `cfg` serves for its whole session: the
    library at `library_root(cfg)`, read once with `memory.skill_min_tasks`.
    Never raises. A directory that does not exist yet is an empty library (never
    created, `error` None). A library that cannot be read -- a configured value that
    is no path or a relative one, a file, an unreadable directory, a symlink (on the
    path) to a missing target, any error while reading, a snapshot whose manifest
    cannot be rendered -- fails closed: no root, the empty library's sha, nothing
    served, `error` = "Exc: msg" and one warning line with the reason."""
    klass = "sim" if profile_class == "sim" else "real"
    root = None
    try:
        root = library_root(cfg)
        # load_snapshot would read a path it cannot list (a file, a directory
        # without read permission) as an empty library; only a path that does not
        # exist yet is one, so the library is listed here first
        try:
            with os.scandir(root):
                pass
        except FileNotFoundError:
            link = _dangling_link(root)
            if link is not None:
                raise FileNotFoundError(f"{link} is a symlink to a missing target ({os.readlink(link)})") from None
        snapshot = load_snapshot(root, klass, min_tasks=cfg.get("memory").get("skill_min_tasks"))
        snapshot.manifest()  # what run.json records: a snapshot that cannot render it serves nothing
        return snapshot
    except Exception as e:  # noqa: BLE001 -- fail closed: a broken library never stops a robot
        return _unread(klass, root, e)


class _ServedNotesLibrary(SkillLibrary):
    """The demo CLI's legacy library view of one snapshot (`agent.aspire.retrieve`
    reads it): only the notes the snapshot serves, parsed from the very bytes it
    hashed. A library whose bytes changed since the snapshot yields no note (a note
    rewritten or added mid-session was never served). Read-only; never creates the
    directory."""

    def __init__(self, snapshot: LibrarySnapshot, min_tasks: int):
        super().__init__(snapshot.root, min_tasks, create=False)
        self._sha = snapshot.sha
        self._served = frozenset(note.id for note in snapshot.served)

    def entries(self) -> list[tuple[str, str]]:
        found = _scan(self.root)
        if _sha_of(found) != self._sha:
            return []
        out = []
        for name, data, _ in found:
            stem = name[: -len(".md")]
            if data is not None and stem in self._served:
                try:
                    out.append((stem, data.decode("utf-8")))
                except UnicodeDecodeError:  # the snapshot never serves one; kept out all the same
                    continue
        return out

    def records(self):
        return [self._parse_text(name, text) for name, text in self.entries()]

    def add(self, *args, **kwargs):
        raise TypeError("a snapshot's library view is read-only")


def served_library(snapshot: LibrarySnapshot | None, min_tasks) -> SkillLibrary | None:
    """The `SkillLibrary` the demo CLI's orchestrator retrieves from: the notes
    `snapshot` serves and nothing else (`_ServedNotesLibrary`), or None when it
    serves none or its directory is gone. Never creates the directory."""
    if snapshot is None or snapshot.root is None or not snapshot.served or not snapshot.root.is_dir():
        return None
    return _ServedNotesLibrary(snapshot, min_tasks)
