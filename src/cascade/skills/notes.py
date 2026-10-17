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
skipped, never cut into acceptance). Only files whose name is a safe id
(`_NOTE_NAME_RE`) are notes, hashed or served.

Notes are advice TEXT. This module renders them and nothing else: no value in a
note reaches a config, a tool argument or a gate. A note is served only when its
lint is clean (`lint_note`), a deny-by-default STRUCTURAL gate on every rendered
field (title and advice): short plain English, printable ASCII, at most one number
and no measurement, no code or markup, naming only read-only tools and the note's
own skill, and no reset, stop, loop, bypass or success-claim advice.

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

import functools
import hashlib
import itertools
import logging
import re
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
#: a verifier pass (DECISIONS Q6): at least this many of the seeds passed ...
VERIFIER_MIN_PASSES = 4
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


def parse_note(path: Path) -> Note | None:
    """The note in `path`, or None when its name is not a safe note id, it cannot
    be read or its front matter is bad."""
    path = Path(path)
    if not _is_note_name(path.name):
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return _parse_bytes(path.stem, data)


# ── library hash ──────────────────────────────────────────────────────────


def _library_files(root: Path | None) -> list[tuple[str, bytes]]:
    """(file name, bytes) of every readable note file in `root`, sorted by name. A
    file whose name is not a safe note id (`_NOTE_NAME_RE`: no control character,
    tab, newline, upper case or non-UTF-8 byte) is neither a note nor hashed, so
    the hash's line format stays unambiguous."""
    if root is None:
        return []
    try:
        paths = sorted((p for p in Path(root).expanduser().glob("*.md") if _is_note_name(p.name) and p.is_file()),
                       key=lambda p: p.name)
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
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def library_sha(root: Path | None) -> str:
    """sha256 over `"<file name>\\t<sha256 of the file>\\n"` for every note file
    in `root`, sorted by name. Missing or empty -> sha256 of the empty string."""
    return _sha_of(_library_files(root))


# ── lint ──────────────────────────────────────────────────────────────────
#
# A deny-by-default STRUCTURAL gate, not a blocklist: a field that cannot be shown
# to be short plain English is refused, and its note is skipped (never cut into
# acceptance). In order:
#   1. length: a field over its cap is refused before any pattern runs;
#   2. characters: no control or format character (Cc/Cf), NFKC-stable, and only
#      printable ASCII letters, digits, space and `_PUNCTUATION`, plus `_TYPOGRAPHY`
#      (so no homoglyph, CJK, fullwidth, zero-width, markup or code character);
#   3. structure, over the normalised word sequence (`_tokens`: snake_case, kebab,
#      camelCase, dotted and spaced spellings all give the same words): no URL, no
#      unknown identifier, no letter-digit mix, no dotted name, no spelled-out word,
#      no role marker, nothing that imitates the provenance tag, at most one number
#      and no measurement or coordinate, only read-only tools and the note's own
#      skill (by name, without arguments), and no reset, stop, loop, override,
#      weakening or success-claim advice.

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

# Phrases over the normalised word sequence: "word" matches exactly, "stem*" any
# word starting with the stem (compiled once by `_Phrases`).
_ROLE_WORDS = frozenset({"assistant", "user", "system", "developer", "human"})
#: reset_scene / reset_stop in any spelling, and "reset the scene": a reset clears the retry ledger
_RESET = (("reset*",),)
_LEARNED = (("learn*", "in"),)
_STOPS = (("emergenc*", "stop*"), ("e", "stop*"), ("estop*",), ("halt*", "motion*"), ("halt*", "the", "motion*"))
_NEGATORS = (("never",), ("not",), ("no",), ("don",), ("dont",), ("without",), ("cannot",))
_HALTS = (("stop*",), ("halt*",), ("paus*",))
_LOOPS = (("start", "over"), ("try", "again", "until"), ("repeat*", "until"), ("until", "it", "succe*"),
          ("until", "it", "work*"), ("until", "succe*"), ("keep", "call*"), ("keep", "try*"), ("keep", "retr*"),
          ("keep", "going"))
#: refused wherever they stand: no plain advice needs them
_OVERRIDES = (("ignor*",), ("disregard*",), ("bypass*",), ("by", "pass*"), ("circumvent*",), ("overrid*",),
              ("over", "rid*"))
#: refused when one of `_GUARDS` follows within `_WEAKEN_REACH` words
_WEAKEN = (("disabl*",), ("turn*", "off"), ("switch*", "off"), ("shut*", "off"), ("skip*",), ("lower*",),
           ("relax*",), ("reduc*",), ("shrink*",), ("rais*",), ("smaller",), ("decreas*",), ("loosen*",),
           ("suppress*",), ("silenc*",), ("mute*",), ("weaken*",), ("work*", "around"), ("get*", "around"))
_GUARDS = (("gate*",), ("check*",), ("refus*",), ("warn*",), ("veto*",), ("clearanc*",), ("margin*",),
           ("limit*",), ("safet*",), ("collis*",), ("collid*",), ("stop*",), ("halt*",), ("guard*",),
           ("harness*",), ("interlock*",), ("watchdog*",), ("threshold*",), ("toleranc*",))
_CLAIM_AS = (("treat*",), ("consider*",), ("regard*",), ("declar*",), ("mark*",))
_CLAIM_SAY = (("tell*",), ("say*",), ("said",), ("report*",), ("announc*",), ("reply",), ("replie*",),
              ("answer*",))
_SUCCESS = (("succe*",), ("done",), ("ok",), ("okay",), ("complete",), ("completed",), ("finished",),
            ("worked",))
#: units of length or angle, refused wherever they stand
_UNITS = (("mm",), ("cm",), ("um",), ("ft",), ("deg",), ("degs",), ("rad",), ("rads",), ("millimet*",),
          ("centimet*",), ("micromet*",), ("micron*",), ("metre*",), ("meter*",), ("inch*",), ("foot",),
          ("feet",), ("degree*",), ("radian*",), ("percent*",))
#: a unit only right after a number ("3 m", "2 in"; and "50%")
_NUMBER_UNITS = frozenset({"m", "in"})
_WEAKEN_REACH = 4  # "lower the safety margin": a guard among the 4 words after the verb
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
def _tool_index() -> dict[str, tuple[tuple[tuple[str, ...], str], ...]]:
    """first word -> ((words of the tool name, tool name), ...)."""
    index: dict[str, list] = {}
    for name in sorted(_tool_names()):
        words = tuple(name.split("_"))
        index.setdefault(words[0], []).append((words, name))
    return {first: tuple(entries) for first, entries in index.items()}


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


class _Phrases:
    """A family of phrases, compiled: each part is (text, is_stem); a word can start
    a match only if it equals an exact first part or starts with a stem one."""

    def __init__(self, phrases):
        self.phrases = tuple(tuple((p[:-1], True) if p.endswith("*") else (p, False) for p in phrase)
                             for phrase in phrases)
        self.exact = frozenset(ph[0][0] for ph in self.phrases if not ph[0][1])
        self.stems = tuple(ph[0][0] for ph in self.phrases if ph[0][1])

    def at(self, words: list[str], i: int) -> int:
        """The length of the first phrase matching at word `i`, else 0."""
        word = words[i]
        if word not in self.exact and not word.startswith(self.stems):
            return 0
        for phrase in self.phrases:
            if i + len(phrase) <= len(words) and all(
                    words[i + k].startswith(part) if stem else words[i + k] == part
                    for k, (part, stem) in enumerate(phrase)):
                return len(phrase)
        return 0


@functools.cache
def _compiled(phrases: tuple) -> _Phrases:
    return _Phrases(phrases)


def _first(words: list[str], phrases) -> str | None:
    family = _compiled(phrases)
    for i in range(len(words)):
        n = family.at(words, i)
        if n:
            return " ".join(words[i:i + n])
    return None


def _followed(words: list[str], heads, tails, reach: int) -> tuple[str, str] | None:
    """The first (head, tail) where a `tails` phrase starts among the `reach` words
    right after a `heads` phrase; None when there is none."""
    head_family, tail_family = _compiled(heads), _compiled(tails)
    for i in range(len(words)):
        n = head_family.at(words, i)
        if not n:
            continue
        for j in range(i + n, min(i + n + reach, len(words))):
            m = tail_family.at(words, j)
            if m:
                return " ".join(words[i:i + n]), " ".join(words[j:j + m])
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


def _measurement(text: str, numbers) -> str | None:
    """'<number> <unit>' for a number followed by "%" or a `_NUMBER_UNITS` word."""
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


def _coordinate(text: str, tokens) -> str | None:
    """'x 0.42': an axis letter followed by a number."""
    for (word, _, end), (nxt, start, _) in itertools.pairwise(tokens):
        if word in ("x", "y", "z") and nxt.isdigit() and set(text[end:start]) <= set(" :=+-"):
            return f"{word} {nxt}"
    return None


def _tool_problem(text: str, tokens, skill: str) -> str | None:
    words = [w for w, _, _ in tokens]
    index = _tool_index()
    for i, word in enumerate(words):
        for seq, name in index.get(word, ()):
            if tuple(words[i:i + len(seq)]) != seq:
                continue
            if name in FORBIDDEN_TOOLS:
                return f"names {name}, which a note may never name"
            if name != skill and name not in READ_ONLY_TOOLS:
                return f"names {name}, which is neither read-only nor this note's skill"
            if _next_char(text, tokens[i + len(seq) - 1][2]) == ":":
                return f"gives {name} arguments (a note names a tool, never how to call it)"
    return None


def _structure_problems(text: str, skill: str) -> list[str]:
    tokens = _tokens(text)
    words = [w for w, _, _ in tokens]
    runs = [m.group(0) for m in _RUN_RE.finditer(text)]
    numbers = list(_NUMBER_RE.finditer(text))
    found: list[str | None] = []
    # code-like spellings
    found.append("names a URL" if "://" in text else None)
    ident = next((m.group(0) for m in _IDENT_RE.finditer(text)
                  if "_" in m.group(0) and m.group(0).lower() not in _tool_names()), None)
    found.append(f"{ident} is not a tool name" if ident is not None else None)
    mixed = next((r for r in runs if not r.isalpha() and not r.isdigit() and not _ORDINAL_RE.fullmatch(r.lower())),
                 None)
    found.append(f"mixes letters and digits in '{mixed}'" if mixed else None)
    dotted = _dotted_name(text)
    found.append(f"carries the dotted name '{dotted}'" if dotted else None)
    found.append("spells a word out letter by letter" if _spelled_out(words) else None)
    role = next((w for w, _, end in tokens if w in _ROLE_WORDS and _next_char(text, end) == ":"), None)
    found.append(f"carries the role marker '{role}:'" if role else None)
    # look-alikes of the provenance tag
    found.append("names the verifier" if any(w.startswith("verifier") for w in words) else None)
    learned = _first(words, _LEARNED)
    found.append(f"claims where it was learned ('{learned}')" if learned else None)
    lib = next((runs[k + 1] for k in range(len(runs) - 1)
                if runs[k].lower() == "lib" and _HEX_RE.fullmatch(runs[k + 1].lower())), None)
    found.append(f"imitates a library hash ('lib {lib}')" if lib else None)
    # numbers and measurements
    found.append(f"carries {len(numbers)} numbers (at most one)" if len(numbers) > 1 else None)
    measured = _measurement(text, numbers)
    found.append(f"carries the measurement '{measured}'" if measured else None)
    unit = _first(words, _UNITS)
    found.append(f"names the unit '{unit}'" if unit else None)
    coordinate = _coordinate(text, tokens)
    found.append(f"carries the coordinate '{coordinate}'" if coordinate else None)
    # tools: read-only ones and the note's own skill, by name, without arguments
    found.append(_tool_problem(text, tokens, skill))
    # advice a note may never give
    reset = _first(words, _RESET)
    found.append(f"advises a reset ('{reset}')" if reset else None)
    stop = _first(words, _STOPS)
    found.append(f"advises a stop ('{stop}')" if stop else None)
    negated = _followed(words, _NEGATORS, _HALTS, _NEGATION_REACH)
    found.append(f"tells the agent not to stop ('{negated[0]} {negated[1]}')" if negated else None)
    loop = _first(words, _LOOPS)
    found.append(f"advises an unbounded retry ('{loop}')" if loop else None)
    override = _first(words, _OVERRIDES)
    found.append(f"advises an override ('{override}')" if override else None)
    weakened = _followed(words, _WEAKEN, _GUARDS, _WEAKEN_REACH)
    found.append(f"'{weakened[0]}' near '{weakened[1]}'" if weakened else None)
    claim = (_followed(words, _CLAIM_AS, _SUCCESS, _TREAT_REACH)
             or _followed(words, _CLAIM_SAY, _SUCCESS, _SAY_REACH))
    found.append(f"claims success ('{claim[0]}' ... '{claim[1]}')" if claim else None)
    return [problem for problem in found if problem]


def _lint_text(field: str, text: str, max_chars: int, skill: str) -> list[str]:
    if not isinstance(text, str):
        return [f"{field}: empty"]
    if len(text) > max_chars:  # before any pattern: an oversized field costs nothing
        return [f"{field}: {len(text)} chars (max {max_chars})"]
    if not text.strip():
        return [f"{field}: empty"]
    character = _character_problem(text)
    if character:
        return [f"{field}: {character}"]
    return [f"{field}: {problem}" for problem in _structure_problems(text, skill)]


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
    """(passes, seeds) of a verifier PASS, else None. A pass is the verdict
    "verified" on the Newton engine (`VERIFIER_ENGINE`, the owner's rule for every
    sim run) with a non-empty list of distinct integer seeds and
    `VERIFIER_MIN_PASSES` <= passes <= seeds (DECISIONS Q6). Anything else -- a
    missing or other engine, bool/float/None seeds, too few or too many passes --
    fails closed."""
    if not isinstance(verifier, dict):
        return None
    if verifier.get("verdict") != "verified" or verifier.get("engine") != VERIFIER_ENGINE:
        return None
    passes, seeds = verifier.get("passes"), verifier.get("seeds")
    if type(passes) is not int or not isinstance(seeds, list) or not seeds:
        return None
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        return None
    if not VERIFIER_MIN_PASSES <= passes <= len(seeds):
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
    """Every class: never refuted or retired, lint clean. Harvested advice (any
    kind but "authored") only once promoted: `min_tasks` distinct tasks AND a
    verifier pass, on every class. An authored note follows the task counters on
    "sim"; "real" needs the verifier pass."""
    if note.status in _NEVER_SERVED or _refuted(note.verifier) or lint_note(note):
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
    """The library as read once at build: every parsed note and, in rank order,
    the ones this profile class serves."""

    root: Path | None
    sha: str
    profile_class: str
    notes: tuple[Note, ...]
    served: tuple[Note, ...]

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


def load_snapshot(root: Path | None, profile_class: str, *, min_tasks: int = 2) -> LibrarySnapshot:
    """Read the library at `root` once, read-only (a missing directory is an empty
    library, never created). Every class serves only lint-clean notes and never a
    refuted or retired one. Harvested notes need `min_tasks` distinct tasks AND a
    verifier pass on every class; authored notes need the task count on "sim" and
    the verifier pass on any other class ("real"). A `min_tasks` that is not an
    integer >= 1 (a missing or broken `memory.skill_min_tasks`) serves nothing."""
    root = None if root is None else Path(root).expanduser()
    files = _library_files(root)
    notes = tuple(note for name, data in files if (note := _parse_bytes(name[: -len(".md")], data)) is not None)
    klass = "sim" if profile_class == "sim" else "real"
    if type(min_tasks) is int and min_tasks >= 1:
        served = tuple(sorted((n for n in notes if _serves(n, klass, min_tasks)), key=_rank))
    else:
        log.warning("ASPIRE notes: min_tasks %r is not an integer >= 1 (memory.skill_min_tasks); "
                    "serving no note", min_tasks)
        served = ()
    return LibrarySnapshot(root=root, sha=_sha_of(files), profile_class=klass, notes=notes, served=served)
