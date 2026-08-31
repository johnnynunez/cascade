"""The fast command path: reflex grammar + experience memory.

Human behavior is tiered: reflexes fire in milliseconds, habits replay
learned routines, and deliberation is reserved for the genuinely novel. The
Anthropic robotics study measured why this matters for LLM robots -- one
LLM turn costs 2-15 s, so anything routine must NOT wait on the model.

Tier 1 (reflex):   a template grammar compiles routine commands ("pick and
                   place pink object", "put the cube in the bowl") straight
                   into deterministic skill calls, in microseconds.
Tier 2 (habit):    Agentic-VLA-style experience memory -- successful plans
                   indexed by a hashed bag-of-words embedding of the
                   instruction, retrieved by cosine similarity and replayed.
Tier 3 (deliberation): the full LLM tool-calling loop (orchestrator), which
                   remains the fallback for novel or failed commands.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from zlib import crc32

import numpy as np

from ..memory.vector_index import QuantizedIndex

SkillCall = tuple[str, dict]

_POLITENESS = re.compile(
    r"^(?:please|hey|ok|now|hermes|robot|can you|could you|would you|"
    r"por favor|puedes|podrias|oye)[\s,]+",
    re.IGNORECASE,
)

_PICK = r"(?:pick|grab|take|fetch|coge|agarra|toma|recoge)"
_PLACE = r"(?:place|put|drop|set|save|store|pon|coloca|deja|guarda|mete)"
_PREP = r"(?:in|into|inside|on|onto|to|at|en|dentro de|sobre|a)"
_ART = r"(?:the\s+|a\s+|an\s+|el\s+|la\s+|un\s+|una\s+)?"

_RULES: list[tuple[re.Pattern, str]] = [
    # "place it in the bowl" (destination for the already-held object);
    # must outrank the generic place rule or the pronoun becomes the object
    (
        re.compile(rf"^{_PLACE}\s+(?:it|this|that|lo|la|eso)\s+{_PREP}\s+{_ART}(?P<dest>.+)$"),
        "place_held",
    ),
    # "pick up the X and put it in the Y" / "coge la X y ponla en la Y"
    (
        re.compile(
            rf"^{_PICK}(?:\s+up)?\s+{_ART}(?P<obj>.+?)\s+(?:and|y)\s+{_PLACE}"
            rf"(?:\s*(?:it|lo|la))?\s+{_PREP}\s+{_ART}(?P<dest>.+)$"
        ),
        "pick_and_place",
    ),
    # "put/place the pink cube in the bowl" / "pon el cubo rosa en el bol"
    (
        re.compile(
            rf"^{_PLACE}\s+{_ART}(?P<obj>.+?)\s+{_PREP}\s+{_ART}(?P<dest>.+)$"
        ),
        "pick_and_place",
    ),
    # "grab the banana and throw it" / "coge la banana y lánzala" /
    # "throw the cube (to the left)" / "lanza el cubo (a la izquierda)"
    # MUST precede the generic "pick" rule, whose lazy object group would
    # otherwise swallow "banana and throw it" as the thing to grasp.
    (
        re.compile(
            rf"^(?:{_PICK}(?:\s+up)?\s+{_ART}(?P<obj>.+?)\s+(?:and|y)\s+)?"
            r"(?:throw|toss|launch|l[aá]nza|tira|arroja)"
            r"(?:la|lo|las|los)?"      # enclitic pronoun: lánzala / tíralo
            r"(?:\s+(?:it|lo|la))?"
            rf"(?:\s+{_ART}(?P<obj2>.+?))?"
            r"(?:\s+(?:to\s+the\s+|a\s+la\s+|hacia\s+)?"
            r"(?P<dir>left|right|forward|back|izquierda|derecha|delante|detras|detrás))?$"
        ),
        "throw",
    ),
    # "pick (up) (and place/put) X (in/on Y)" / "coge y coloca el objeto rosa"
    (
        re.compile(
            rf"^{_PICK}(?:\s+up)?\s+(?:(?P<andplace>(?:and|y)\s+{_PLACE}(?:\s+it)?)\s+)?"
            rf"{_ART}(?P<obj>.+?)"
            rf"(?:\s+{_PREP}\s+{_ART}(?P<dest>.+?))?$"
        ),
        "pick",
    ),
    # "push the box left"
    (
        re.compile(
            rf"^(?:push|empuja)\s+{_ART}(?P<obj>.+?)\s+(?:to\s+the\s+)?"
            r"(?P<dir>left|right|forward|back)(?:wards?)?$"
        ),
        "push",
    ),
    # "stack the red cube on the blue box" == pick A, place on B
    (
        re.compile(rf"^(?:stack|apila)\s+{_ART}(?P<obj>.+?)\s+(?:on(?:\s+top\s+of)?|sobre|en)\s+{_ART}(?P<dest>.+)$"),
        "pick_and_place",
    ),
    # "hand me / give me the pink object" -> handover
    (
        re.compile(rf"^(?:hand|give|pass|bring)\s+(?:me\s+|us\s+)?{_ART}(?P<obj>.+?)$|^(?:dame|pasame|traeme)\s+{_ART}(?P<obj2>.+?)$"),
        "handover",
    ),
    # "point at/to the pink object" / "señala el objeto rosa"
    (
        re.compile(rf"^(?:point\s+(?:at|to)|show\s+me|senala|señala)\s+{_ART}(?P<obj>.+)$"),
        "point_at",
    ),
    (re.compile(r"^(?:wave|say\s+hi|say\s+hello|saluda)(?:\s+(?:at|to)\s+\S.*)?$"), "wave"),
    (
        re.compile(r"^(?:sort|organize|group|ordena|organiza)\s+.*(?:color|colour)e?s?.*$"),
        "sort_by_color",
    ),
    (
        re.compile(rf"^(?:how\s+many|count|cuantos|cuantas)\s+{_ART}(?P<obj>.+?)"
                   r"(?:\s+(?:do\s+you\s+see|are\s+there|hay))?\??$"),
        "count",
    ),
    (
        re.compile(rf"^move\s+(?:a\s+(?:bit|little)\s+|slightly\s+)?"
                   r"(?P<dir>forward|back|left|right|up|down)(?:\s+a\s+(?:bit|little))?$"),
        "move_relative",
    ),
    (re.compile(r"^(?:go\s+home|move\s+home|home|park(?:\s+the\s+arm)?)$"), "home"),
    (
        re.compile(
            r"^(?:look(?:\s+around)?|observe|scan|"
            r"mira|observa)$"
        ),
        "look",
    ),
    # scene questions answer instantly from the world model
    (
        re.compile(
            r"^(?:what\s+do\s+you\s+see\??|what(?:'s|\s+is)\s+on\s+the\s+table\??|"
            r"describe\s+the\s+(?:scene|table)|que\s+ves\??|describe\s+la\s+mesa)$"
        ),
        "describe",
    ),
    (re.compile(r"^(?:open\s+(?:the\s+)?gripper|let\s+go|release|abre\s+la\s+pinza|suelta(?:lo|la)?)$"), "open_gripper"),
    (re.compile(r"^(?:stop|para|alto)$"), None),  # never a reflex: stop is for e-stop tools
]


@dataclass
class ReflexPlan:
    intent: str
    calls: list[SkillCall]
    object_query: str | None = None
    destination: str | None = None


def parse_command(text: str) -> ReflexPlan | None:
    """Compile a routine command into skill calls; None -> not routine."""
    cmd = text.strip().lower()
    cmd = re.sub(r"[.!?¡¿]+", "", cmd)
    for _ in range(3):  # peel stacked politeness ("hey, can you please ...")
        stripped = _POLITENESS.sub("", cmd).strip()
        if stripped == cmd:
            break
        cmd = stripped
    cmd = re.sub(r"\s+", " ", cmd)
    if not cmd:
        return None
    # Compound/sequenced commands are NOT reflexes: executing only the first
    # clause and reporting success silently drops the rest. Send them to the
    # LLM tier, which can plan multi-step ("wave and then pick up the cube").
    if re.search(r"\b(?:and then|then|after that|luego|despues|después)\b", cmd):
        return None

    for rule, intent in _RULES:
        m = rule.match(cmd)
        if not m:
            continue
        if intent is None:
            return None
        g = m.groupdict()
        obj = (g.get("obj") or "").strip() or None
        dest = (g.get("dest") or "").strip() or None
        if obj:
            # "pick and place the banana and save it in the box": the lazy
            # object group of the pick..and..place rule swallows a leading
            # place-verb clause -- strip it, or the robot spends two minutes
            # trying to localize 'and place the banana'.
            obj = re.sub(
                rf"^(?:and|y)\s+{_PLACE}(?:\s+(?:it|lo|la))?\s+{_ART}", "", obj
            ).strip() or None
            if obj is None:
                continue
        if intent == "pick":
            # "pick X" alone is a grasp; a place-verb or destination makes it
            # a full pick-and-place.
            if g.get("andplace") or dest:
                intent = "pick_and_place"
        if intent == "pick_and_place":
            if not obj:
                return None
            args = {"object": obj}
            if dest:
                args["destination"] = dest
            return ReflexPlan(intent, [("pick_and_place", args)], obj, dest)
        if intent == "pick":
            return ReflexPlan(intent, [("grasp_object", {"label": obj})], obj)
        if intent == "place_held":
            if not dest:
                return None
            return ReflexPlan(intent, [("place_on_object", {"label": dest})], None, dest)
        if intent == "push":
            return ReflexPlan(
                intent,
                [("push_object", {"label": obj, "direction": g["dir"]})],
                obj,
            )
        if intent == "throw":
            tobj = obj or (g.get("obj2") or "").strip() or None
            _dirmap = {
                "izquierda": "left", "derecha": "right",
                "delante": "forward", "detras": "back", "detrás": "back",
            }
            raw_dir = (g.get("dir") or "").strip()
            tdir = _dirmap.get(raw_dir, raw_dir) or "forward"
            args = {"direction": tdir}
            if tobj:
                args["label"] = tobj
            return ReflexPlan(intent, [("throw", args)], tobj)
        if intent == "handover":
            obj = obj or (g.get("obj2") or "").strip() or None
            if not obj:
                return None
            return ReflexPlan(intent, [("handover", {"label": obj})], obj)
        if intent == "point_at":
            return ReflexPlan(intent, [("point_at", {"label": obj})], obj)
        if intent == "wave":
            return ReflexPlan(intent, [("wave", {})])
        if intent == "sort_by_color":
            return ReflexPlan(intent, [("sort_by_color", {})])
        if intent == "count":
            return ReflexPlan(intent, [("count_objects", {"query": obj} if obj else {})], obj)
        if intent == "describe":
            return ReflexPlan(intent, [("describe_scene", {})])
        if intent == "move_relative":
            return ReflexPlan(intent, [("move_relative", {"direction": g["dir"]})])
        if intent == "home":
            return ReflexPlan(intent, [("move_home", {})])
        if intent == "look":
            return ReflexPlan(intent, [("get_observation", {})])
        if intent == "open_gripper":
            return ReflexPlan(intent, [("open_gripper", {})])
    return None


# ── tier 2: experience memory (Agentic-VLA EM, inference-time) ───────────


_STOPWORDS = frozenset(
    "the a an it this that those these to of and or please can you could my me "
    "el la los las un una y o por favor lo esa ese esto".split()
)


def _embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashed bag-of-words embedding over content words (no
    model, no network); stopwords out so "hand me the X" == "hand me that X"."""
    v = np.zeros(dim, dtype=np.float64)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOPWORDS:
            continue
        h = crc32(tok.encode())
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class ExperienceMemory:
    """Instruction -> proven skill plan, retrieved by embedding similarity.

    Successful plans are stored with win/loss counts and persisted as JSON so
    habits survive process restarts (the demo learns across sessions).
    """

    def __init__(self, path: Path | None = None, dim: int = 256):
        self._dim = dim
        self._path = path
        self._index = QuantizedIndex(dim=dim, bits=4)
        self._records: list[dict] = []
        self._lock = threading.Lock()
        if path is not None and path.exists():
            try:
                self._records = json.loads(path.read_text())
                for i, rec in enumerate(self._records):
                    self._index.add(_embed(rec["task"], dim), meta=i)
            except Exception:
                self._records = []

    def __len__(self) -> int:
        return len(self._records)

    def recall(self, task: str, min_sim: float = 0.9) -> dict | None:
        """Best stored plan for a similar instruction, if it mostly worked."""
        with self._lock:
            for sim, i in self._index.search(_embed(task, self._dim), k=3):
                if sim < min_sim:
                    break
                rec = self._records[i]
                if rec["wins"] > rec["losses"]:
                    return {"sim": round(sim, 3), **rec}
        return None

    def record(self, task: str, calls: list[SkillCall], success: bool, duration_s: float) -> None:
        snapshot = None
        with self._lock:
            for rec in self._records:
                if rec["task"] == task.strip().lower():
                    rec["wins" if success else "losses"] += 1
                    n = rec["wins"] + rec["losses"]
                    rec["avg_s"] = round(rec["avg_s"] + (duration_s - rec["avg_s"]) / n, 2)
                    if success:
                        rec["calls"] = [[c, dict(a)] for c, a in calls]
                    snapshot = json.dumps(self._records, indent=1)
                    break
            else:
                if not success:
                    return  # only remember plans that have worked at least once
                self._records.append(
                    {
                        "task": task.strip().lower(),
                        "calls": [[c, dict(a)] for c, a in calls],
                        "wins": 1,
                        "losses": 0,
                        "avg_s": round(duration_s, 2),
                    }
                )
                self._index.add(_embed(task, self._dim), meta=len(self._records) - 1)
                snapshot = json.dumps(self._records, indent=1)
        # File I/O happens OUTSIDE the lock: recall() on the command hot
        # path must never wait on a disk write.
        if snapshot is not None:
            self._save(snapshot)

    def _save(self, snapshot: str) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(snapshot)
        except OSError:
            pass

    def stats(self) -> dict:
        return {"plans": len(self._records)}


@dataclass
class FastPlan:
    source: str  # "reflex" | "experience" | "curriculum"
    calls: list[SkillCall]
    detail: str = ""
    #: Sub-goals this plan is made of, in order (Agentic-VLA curriculum
    #: decomposition). Empty for a plain single-step reflex.
    subgoals: list[str] = field(default_factory=list)
    #: How many entries of `calls` each sub-goal contributed, same order and
    #: length as `subgoals`. Recorded at build time rather than recomputed by
    #: the caller: a clause can compile to more than one call, so re-deriving
    #: the split later would mis-credit outcomes to the wrong sub-goal.
    subgoal_call_counts: list[int] = field(default_factory=list)
    #: Where each call came from: "recalled" (a proven plan for this sub-goal,
    #: i.e. the warm start) or "reflex". Same length as `calls`.
    provenance: list[str] = field(default_factory=list)

    @property
    def warm_started(self) -> bool:
        return any(p == "recalled" for p in self.provenance)

    def subgoal_spans(self):
        """Yield (subgoal, calls_for_that_subgoal) in order."""
        idx = 0
        for name, n in zip(self.subgoals, self.subgoal_call_counts):
            yield name, self.calls[idx: idx + n]
            idx += n


# ── curriculum decomposition (Agentic-VLA, inference-time) ───────────────
#
# Agentic-VLA (arXiv:2605.22896) decomposes a complex task into learnable
# sub-goals and warm-starts each from retrieved experience. Its version trains
# a policy online with synthesized rewards; cascade runs a FROZEN brain and
# analytic skills, so the trainable parts do not transplant. What does
# transplant is the STRUCTURE, and it fits the fast path exactly:
#
#   decompose -> for each sub-goal, retrieve a proven plan (warm start),
#   else fall back to the reflex grammar; only give up to the LLM if a
#   sub-goal has neither.
#
# Why this belongs in the FAST tier rather than the orchestrator: the
# orchestrator already decomposes, but it spends an LLM turn to do it
# (`_decompose`), and then another turn per step. A conjunction of two routine
# commands ("pick up the cube and then wave") is not novel -- it is two
# reflexes -- yet `parse_command` deliberately refuses compound commands
# because executing only the first clause and reporting success would silently
# drop the rest. That refusal is correct but it sends genuinely routine work to
# the slow path. Splitting on the connective and planning each clause turns
# that whole class into microseconds, WITHOUT the "silently dropped clause"
# failure the refusal was protecting against: a clause that cannot be planned
# aborts the whole fast plan.

#: Connectives that separate sequential clauses. `parse_command` rejects any
#: command containing these; here they are the split points. Ordered longest
#: first so "and then" wins over "and".
_SEQUENCE_SPLIT = re.compile(
    r"\s*(?:,\s*)?\b(?:and\s+then|after\s+that|then|luego|despu[eé]s|"
    r"y\s+luego|y\s+despu[eé]s)\b\s*",
    re.IGNORECASE,
)


def split_subgoals(task: str) -> list[str]:
    """Split a sequential instruction into ordered clauses.

    Only splits on explicit SEQUENCE connectives, never on a bare "and":
    "pick up the cube and put it in the box" is ONE pick_and_place that the
    grammar already handles, and splitting it would turn a single motion into
    two half-motions. This is why the reflex rules keep their own "and"
    handling and this function is deliberately conservative.
    """
    parts = [p.strip(" ,.") for p in _SEQUENCE_SPLIT.split(task.strip())]
    return [p for p in parts if p]


class FastPlanner:
    """Tier 1 + tier 2 resolver; returns None when only the LLM will do.

    Resolution order, cheapest first:

    1. **reflex** -- the whole command matches the grammar (microseconds);
    2. **experience** -- a proven plan for a similar instruction (warm start);
    3. **curriculum** -- the command is a SEQUENCE of clauses, each of which
       resolves by (2) or (1). This is the Agentic-VLA structure at inference
       time: decompose into sub-goals, warm-start each from retrieved
       experience, and only escalate what genuinely has no proven plan.

    (3) is all-or-nothing on purpose: if any clause cannot be planned the whole
    fast plan is abandoned to the LLM. A partially-executed sequence that
    reports success is the exact failure `parse_command` refuses compound
    commands to avoid, and it must not reappear here.
    """

    def __init__(self, experience: ExperienceMemory | None = None,
                 curriculum: bool = True):
        self.experience = experience
        self.curriculum = curriculum

    # ── one clause ──────────────────────────────────────────────────────

    def _plan_one(self, text: str) -> tuple[list[SkillCall], str, str] | None:
        """(calls, provenance, detail) for a single clause, or None.

        Experience is consulted BEFORE the grammar: a plan that has actually
        worked on this rig beats a freshly compiled one, and that preference IS
        the warm start. `recall` already refuses records whose losses outnumber
        wins, so a habit that stopped working stops being retrieved.
        """
        if self.experience is not None:
            rec = self.experience.recall(text)
            if rec is not None:
                calls = [(c, dict(a)) for c, a in rec["calls"]]
                return (
                    calls,
                    "recalled",
                    f"~{rec['task']!r} (sim {rec['sim']}, "
                    f"{rec['wins']}w/{rec['losses']}l)",
                )
        reflex = parse_command(text)
        if reflex is not None:
            return reflex.calls, "reflex", reflex.intent
        return None

    def plan(self, task: str) -> FastPlan | None:
        reflex = parse_command(task)
        if reflex is not None:
            return FastPlan("reflex", reflex.calls, reflex.intent,
                            provenance=["reflex"] * len(reflex.calls))
        if self.experience is not None:
            rec = self.experience.recall(task)
            if rec is not None:
                calls = [(c, dict(a)) for c, a in rec["calls"]]
                return FastPlan(
                    "experience",
                    calls,
                    f"~{rec['task']!r} (sim {rec['sim']}, {rec['wins']}w/{rec['losses']}l)",
                    provenance=["recalled"] * len(calls),
                )
        if self.curriculum:
            return self._plan_curriculum(task)
        return None

    def _plan_curriculum(self, task: str) -> FastPlan | None:
        subgoals = split_subgoals(task)
        if len(subgoals) < 2:
            return None  # nothing to decompose; the LLM tier owns this
        calls: list[SkillCall] = []
        provenance: list[str] = []
        details: list[str] = []
        counts: list[int] = []
        for clause in subgoals:
            planned = self._plan_one(clause)
            if planned is None:
                # One unplannable sub-goal sinks the whole fast plan: running
                # the prefix and stopping would report partial work as done.
                return None
            clause_calls, prov, detail = planned
            calls.extend(clause_calls)
            counts.append(len(clause_calls))
            provenance.extend([prov] * len(clause_calls))
            details.append(f"{clause!r}->{detail}")
        warm = sum(1 for p in provenance if p == "recalled")
        return FastPlan(
            "curriculum",
            calls,
            f"{len(subgoals)} sub-goals ({warm} warm-started): " + "; ".join(details),
            subgoals=subgoals,
            subgoal_call_counts=counts,
            provenance=provenance,
        )

    def note_outcome(self, task: str, calls: list[SkillCall], success: bool, duration_s: float) -> None:
        if self.experience is not None:
            self.experience.record(task, calls, success, duration_s)

    def note_subgoal_outcome(self, subgoal: str, calls: list[SkillCall],
                             success: bool, duration_s: float) -> None:
        """Record a CLAUSE's outcome under its own key.

        This is what makes the curriculum compound: a sub-goal proven inside
        one sequence is retrievable as a warm start for any later task that
        contains it, including a different sequence. Without this the memory
        would only ever learn whole instructions verbatim.
        """
        if self.experience is not None and success:
            self.experience.record(subgoal, calls, True, duration_s)
