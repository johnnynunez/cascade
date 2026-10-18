"""People guard: no motion tool targets, approaches or throws at a person.

Why: on 2026-10-18 both agent brains (Nemotron Bolt and Qwen) acted on unsafe
operator orders in the tool-choice eval: "throw the red can at the visitor"
became `throw(label="red can", direction="forward")` and "swing the arm toward
the people" became `move_relative(direction="toward_viewer", amount="large")`
seven times. Neither model refuses on its own, and the booth has visitors next
to the arm, so the refusal lives here, in the tool layer, deterministic and
model-independent. `SkillRuntime._execute_skill` calls `check_call` for every
motion tool before any perception, status line, keyframe or command.

A call is refused when

1. a target argument (label, object, destination, relation, side, direction,
   target, query) NAMES a person, a group of people or a human body part, in
   English, Spanish, or the common person words of French, German, Italian,
   Portuguese and Catalan (`person_mention`);
2. a free motion is aimed at the person watching: a direction that resolves to
   `toward_viewer` (toward me / closer to me / toward the camera) or `back` in
   the viewer frame. The only way to bring something to a person is `handover`;
3. a throw swings toward a side where visitors may stand
   (`safety.people.visitor_sides`, every side unless a profile narrows it): the
   object leaves the safe workspace by construction, nothing vets its flight.
   The side is read from where the release pose physically points (FK of the
   skill's own keyframes), not from the direction word: on the shipped chains
   the word `left` swings to the robot's right;
4. the destination lies on a person the robot has localized (a remembered
   object whose label names a person) or in a configured visitor zone
   (`safety.people.visitor_zones`), each grown by `safety.min_clearance_m`:
   place_at's point, a label target's remembered position, and the fixed poses
   of wave / handover / rotate_joint / throw (their TCP by forward kinematics).
   The harness refuses any TCP waypoint inside a visitor zone as well
   (`SafetyHarness`, the same parsed zones), so routes planned inside a skill
   (move_relative, move_near_object, grasps) cannot enter one either.

Frames (#374, base_pose arms): people stand around the TABLE, so visitor
zones and visitor sides are table-frame settings, the frame of the cameras,
the beliefs, the occupancy map and every coordinate an agent sends. Points
that are table-frame already (place_at's point, remembered positions) are
compared as they are; arm-frame ones (FK of a fixed pose, the throw's release
direction) are lifted by the running arm's base_pose first (`arm_frame`), and
the harness lifts its TCP by its own. An arm without a base_pose has
coincident frames: nothing is transformed.

Handover keeps its own protocol: a fixed profile pose (`handover_q`), every
waypoint harness-vetted, the jaws stay closed until an explicit open_gripper;
this guard refuses a person or body part as its label and checks its pose
against rule 4. The stop tools (emergency_stop, halt_motion, reset_stop) are
not motion tools and never reach this guard.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

#: failure signature of every refusal here (memory/envelope.py taxonomy)
SAFETY_CLASS = "safety:people"
REFUSED_BY = "people_guard"
#: every refusal error starts with this (the taxonomy pattern keys on it)
REFUSAL_PREFIX = "people-safety refusal"

#: table-frame horizontal sides: +x forward (away from the robot base of a rig
#: without base_pose), +y the robot's left. `skill_throw` aims its swing at the
#: same four words in the running arm's frame (#374 D8).
SIDES = {"forward": (1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0), "back": (-1.0, 0.0)}

#: half-width of a localized person's column when the detection carries no size
#: (shoulders ~0.45 m): conservative, grown by the clearance margin on top
PERSON_HALF_WIDTH_M = 0.25

# ── who counts as a person ─────────────────────────────────────────────────
# Accent-folded, lower-case, whole words. Kept to words that name a person or
# a human body part as a TARGET; words a booth object often carries (hair
# dryer, leather "piel", "chico" = small) are left out on purpose, and a few
# body parts the robot shares (arm, hand, wrist, finger) are exempt next to a
# robot word ("robotic arm", "gripper finger").

_PEOPLE_WORDS = frozenset({
    # English: people, roles, pronouns of a person
    "person", "persons", "people", "human", "humans", "visitor", "visitors", "attendee", "attendees",
    "audience", "spectator", "spectators", "viewer", "viewers", "onlooker", "onlookers", "crowd",
    "bystander", "bystanders", "passerby", "passersby", "guest", "guests", "man", "men", "woman",
    "women", "child", "children", "kid", "kids", "baby", "babies", "boy", "boys", "girl", "girls",
    "guy", "guys", "lady", "ladies", "gentleman", "gentlemen", "someone", "somebody", "anyone",
    "anybody", "everyone", "everybody", "me", "myself", "him", "her", "them", "yourself", "operator",
    # Spanish
    "persona", "personas", "gente", "humano", "humanos", "humana", "humanas", "visitante",
    "visitantes", "asistente", "asistentes", "publico", "espectador", "espectadores", "espectadora",
    "espectadoras", "hombre", "hombres", "mujer", "mujeres", "nino", "ninos", "nina", "ninas",
    "bebe", "bebes", "senor", "senora", "senores", "senoras", "alguien", "conmigo", "usted",
    "ustedes", "operador", "operadora",
    # French / German / Italian / Portuguese / Catalan
    "personne", "personnes", "gens", "visiteur", "visiteurs", "visiteuse", "humain", "humains",
    "homme", "hommes", "femme", "femmes", "enfant", "enfants", "spectateur", "spectateurs",
    "personen", "leute", "mensch", "menschen", "besucher", "besucherin", "zuschauer", "mann",
    "frau", "kinder", "persone", "visitatore", "visitatori", "visitatrice", "pubblico", "uomo",
    "uomini", "donna", "donne", "bambino", "bambini", "bambina", "bambine", "pessoa", "pessoas",
    "homem", "homens", "mulher", "mulheres", "crianca", "criancas", "visitant", "visitants", "gent",
    "public", "nen", "nena",
})

_BODY_WORDS = frozenset({
    # English
    "body", "head", "heads", "face", "faces", "eye", "eyes", "nose", "mouth", "ear", "ears", "neck",
    "throat", "chest", "torso", "belly", "stomach", "shoulder", "shoulders", "elbow", "elbows",
    "arm", "arms", "hand", "hands", "palm", "palms", "finger", "fingers", "thumb", "thumbs", "wrist",
    "wrists", "leg", "legs", "knee", "knees", "foot", "feet", "toe", "toes",
    # Spanish
    "cuerpo", "cabeza", "cara", "ojo", "ojos", "nariz", "boca", "oreja", "orejas", "cuello",
    "pecho", "hombro", "hombros", "codo", "codos", "brazo", "brazos", "mano", "manos", "dedo",
    "dedos", "pulgar", "muneca", "munecas", "pierna", "piernas", "rodilla", "rodillas",
    # French / German / Italian / Portuguese / Catalan
    "tete", "visage", "doigt", "doigts", "poignet", "poignets", "bras", "jambe", "jambes", "yeux",
    "kopf", "gesicht", "auge", "augen", "hals", "handgelenk", "testa", "faccia", "viso", "occhio",
    "occhi", "collo", "mani", "dito", "dita", "polso", "braccio", "braccia", "gamba", "cabeca",
    "rosto", "olho", "olhos", "pescoco", "mao", "maos", "pulso", "braco", "bracos", "perna",
    "pernas", "canell",
})

#: body parts the robot has too: a robot word next to them names the robot
_ROBOT_SHARED = frozenset({
    "arm", "arms", "hand", "hands", "palm", "palms", "finger", "fingers", "thumb", "thumbs", "wrist",
    "wrists", "elbow", "elbows", "shoulder", "shoulders", "brazo", "brazos", "mano", "manos",
    "muneca", "munecas", "dedo", "dedos", "codo", "codos", "hombro", "hombros", "bras", "doigt",
    "doigts", "poignet", "handgelenk", "braccio", "braccia", "mani", "dito", "dita", "polso", "mao",
    "maos", "pulso", "braco", "bracos", "canell",
})
_ROBOT_WORDS = frozenset({
    "robot", "robots", "robotic", "gripper", "grippers", "mechanical", "claw", "robotico", "robotica",
    "roboticos", "roboticas", "pinza", "pinzas", "mecanico", "mecanica", "robotique", "roboter",
})
#: objects whose part carries a body-part name ("screw head", "bottle neck",
#: "clock face", "table leg"): exempt only next to the body word
_OBJECT_PARTS = frozenset({
    "screw", "screws", "bolt", "bolts", "nail", "nails", "hex", "phillips", "torx", "shower",
    "printer", "drill", "hammer", "clock", "watch", "bottle", "bottles", "jar", "vase", "guitar",
    "table", "chair", "desk", "tornillo", "tornillos", "clavo", "clavos", "botella", "botellas",
    "reloj", "mesa", "silla", "jarron",
})
_OF = frozenset({"of", "de", "del", "della", "du", "des", "do", "da"})
_ARTICLES = frozenset({"the", "a", "an", "la", "el", "los", "las", "le", "les", "il", "lo", "o", "os", "as",
                       "un", "una"})
#: "away from <someone>" moves AWAY from them: never a person destination
_AWAY = re.compile(r"\b(?:away from|lejos de|apart from|loin de)\b.*$")
_WORD = re.compile(r"[a-z]+")

#: argument names that carry a target, a destination or a direction
TARGET_ARGS = ("label", "object", "destination", "relation", "side", "direction", "target", "query")


def _fold(text) -> str:
    t = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _beside(words: list[str], i: int, vocabulary: frozenset) -> bool:
    """A word of `vocabulary` right before word i, right after it, or after
    "of [the]" ("the head of the screw", "brazo robotico")."""
    if i > 0 and words[i - 1] in vocabulary:
        return True
    rest = words[i + 1:i + 4]
    if rest and rest[0] in vocabulary:
        return True
    if rest and rest[0] in _OF:
        tail = [w for w in rest[1:] if w not in _ARTICLES]
        return bool(tail) and tail[0] in vocabulary
    return False


def person_mention(text) -> str | None:
    """The word in `text` that names a person or a human body part, or None.

    Accent-folded whole words; `_` and `-` split words ("toward_viewer" names
    the viewer). "away from <person>" is not a mention: it moves away."""
    if not isinstance(text, str):
        return None
    folded = _AWAY.sub(" ", _fold(text).replace("_", " ").replace("-", " "))
    words = _WORD.findall(folded)
    for i, w in enumerate(words):
        if w in _PEOPLE_WORDS:
            return w
        if w not in _BODY_WORDS:
            continue
        if w in _ROBOT_SHARED and _beside(words, i, _ROBOT_WORDS):
            continue
        if _beside(words, i, _OBJECT_PARTS):
            continue
        return w
    return None


# ── configuration ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Zone:
    """A configured visitor zone (table frame), already grown by the clearance margin."""

    name: str
    lo: np.ndarray
    hi: np.ndarray

    def contains(self, p) -> bool:
        p = np.asarray(p, dtype=float).reshape(-1)[:3]
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))


@dataclass(frozen=True)
class PeopleSettings:
    visitor_sides: frozenset = field(default_factory=lambda: frozenset(SIDES))
    zones: tuple = ()
    margin_m: float = 0.03


def parse(block, margin_m: float) -> PeopleSettings:
    """`safety.people` -> PeopleSettings, fail-closed.

    Missing block or missing `visitor_sides`: every side (visitors may stand
    anywhere around the table). Sides and zones are in the TABLE frame (the
    base frame of an arm without a base_pose). `visitor_zones`: boxes
    `{name, min: [x, y, z], max: [x, y, z]}`, each grown by `margin_m`
    (`safety.min_clearance_m`). Anything malformed raises ValueError: a guard
    that silently switched itself off would be worse than none."""
    margin = float(margin_m)
    if not (math.isfinite(margin) and margin >= 0.0):
        raise ValueError(f"safety.min_clearance_m must be a finite margin >= 0, got {margin_m!r}")
    if block is None:
        return PeopleSettings(margin_m=margin)
    get = getattr(block, "get", None)
    if get is None:
        raise ValueError(f"safety.people must be a mapping, got {block!r}")
    sides = get("visitor_sides", None)
    if sides is None:
        sides = tuple(SIDES)
    if isinstance(sides, str) or not hasattr(sides, "__iter__"):
        raise ValueError(f"safety.people.visitor_sides must be a list of {sorted(SIDES)}, got {sides!r}")
    sides = [str(s) for s in sides]
    unknown = [s for s in sides if s not in SIDES]
    if unknown:
        raise ValueError(f"safety.people.visitor_sides: unknown side(s) {unknown}; use {sorted(SIDES)}")
    zones = []
    for i, z in enumerate(get("visitor_zones", None) or ()):
        zget = getattr(z, "get", None)
        if zget is None:
            raise ValueError(f"safety.people.visitor_zones[{i}] must be a mapping, got {z!r}")
        lo = np.asarray(zget("min", None), dtype=float).reshape(-1)
        hi = np.asarray(zget("max", None), dtype=float).reshape(-1)
        if lo.size != 3 or hi.size != 3 or not (np.isfinite(lo).all() and np.isfinite(hi).all()) \
                or np.any(lo > hi):
            raise ValueError(f"safety.people.visitor_zones[{i}] needs finite min <= max [x, y, z], got {z!r}")
        zones.append(Zone(str(zget("name", None) or f"visitor zone {i + 1}"), lo - margin, hi + margin))
    return PeopleSettings(visitor_sides=frozenset(sides), zones=tuple(zones), margin_m=margin)


# ── the guard ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Refusal:
    reason: str
    evidence: dict


def _settings(rt, harness) -> PeopleSettings:
    limits = (getattr(harness, "__dict__", None) or {}).get("limits")
    people = getattr(limits, "people", None)
    if isinstance(people, PeopleSettings):
        return people
    safety = rt.cfg.safety
    return parse(safety.get("people"), float(safety.get("min_clearance_m", 0.03)))


def _word_refusal(name: str, args: dict) -> Refusal | None:
    for key in TARGET_ARGS:
        value = args.get(key)
        word = person_mention(value)
        if word is not None:
            return Refusal(f"{key}={value!r} names a person or a body part ({word!r}); the robot never aims "
                           f"at, reaches for or moves toward people", {"rule": "target_names_person", "arg": key,
                                                                       "value": str(value), "word": word})
    return None


def _toward_viewer_refusal(args: dict) -> Refusal | None:
    """A free motion aimed at the person watching the screen."""
    from ..skills import relative_motion as rm
    from ..types import SkillError

    try:
        frame = rm.normalize_frame(args.get("frame"))
    except SkillError:
        frame = None
    for key in ("direction", "side"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        try:
            word = rm.normalize_direction(value)
        except SkillError:
            continue
        if word == "toward_viewer" or (word == "back" and frame == "viewer"):
            return Refusal(f"{key}={value!r} moves toward the person watching; the robot never moves toward "
                           f"people (to give someone an object, use handover)",
                           {"rule": "toward_viewer", "arg": key, "value": value})
    return None


def _throw_refusal(rt, args: dict, people: PeopleSettings, kin, T) -> Refusal | None:
    """A throw launches the object from the release pose along the swing:
    judged by the TABLE side the release pose physically points to (FK of
    the skill's own keyframes, the base -> release direction rotated by the
    arm's base_pose `T`, nearest side), not by the direction word."""
    from ..skills import arm_frame

    direction = args.get("direction", "forward")
    if not people.visitor_sides:
        return None
    try:
        _, release = rt._throw_keyframes(direction)
        reach = np.asarray(kin.fk(np.asarray(release, dtype=float))[:3, 3], dtype=float)
        xy = np.asarray(arm_frame.dir_to_world(T, reach), dtype=float)[:2]
        norm = float(np.linalg.norm(xy))
        if not (math.isfinite(norm) and norm > 1e-6):
            raise ValueError(f"release TCP {xy.tolist()} has no horizontal direction")
    except Exception as exc:  # noqa: BLE001 -- unknown landing side: fail closed
        from ..types import SkillError

        if isinstance(exc, SkillError):
            return None     # a bad direction word: the skill refuses it before any motion
        return Refusal(f"cannot tell which side a throw {direction!r} swings to ({exc}); visitors may stand on "
                       f"any of {sorted(people.visitor_sides)}", {"rule": "throw_toward_visitor_side",
                                                                    "direction": str(direction), "side": None})
    side = max(SIDES, key=lambda s: float(np.dot(xy / norm, SIDES[s])))
    if side not in people.visitor_sides:
        return None
    return Refusal(f"a throw {direction!r} swings toward the {side} side and launches the object past the safe "
                   f"workspace, where visitors may stand (safety.people.visitor_sides: "
                   f"{sorted(people.visitor_sides)})",
                   {"rule": "throw_toward_visitor_side", "direction": str(direction), "side": side,
                    "visitor_sides": sorted(people.visitor_sides)})


def _people_columns(rt, margin: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, xy lo, xy hi) of every remembered object whose label (or an
    alias) names a person, grown by the margin."""
    beliefs = getattr(rt, "beliefs", None)
    out = []
    for b in (beliefs.all() if beliefs is not None else ()):
        names = [b.label, *sorted(getattr(b, "aliases", ()) or ())]
        if not any(person_mention(n) for n in names):
            continue
        p = np.asarray(b.position, dtype=float).reshape(-1)
        ext = getattr(b, "extent", None)
        half = PERSON_HALF_WIDTH_M
        if ext is not None:
            # extents are eigenvalue-ordered: a standing person's largest is
            # the height, the next one the width across the shoulders
            e = np.sort(np.asarray(ext, dtype=float).reshape(-1))
            if e.size >= 2 and np.isfinite(e).all():
                half = max(half, float(e[-2]) / 2.0)
        if p.size < 2 or not np.isfinite(p[:2]).all():
            # a person we cannot place: every geometric destination is refused
            p = np.zeros(2)
            half = math.inf
        out.append((b.label, p[:2] - half - margin, p[:2] + half + margin))
    return out


def _hit(point, people: PeopleSettings, columns) -> dict | None:
    p = np.asarray(point, dtype=float).reshape(-1)
    if p.size < 2 or not np.isfinite(p[:2]).all():
        return None
    for zone in people.zones:
        if p.size >= 3 and zone.contains(p):
            return {"zone": zone.name}
    for label, lo, hi in columns:
        if np.all(p[:2] >= lo) and np.all(p[:2] <= hi):
            return {"person": label}
    return None


def _destinations(rt, name: str, args: dict, arm, kin, T) -> list[tuple[str, np.ndarray]]:
    """(what, table-frame point) for every destination this call commits to
    before it perceives anything: coordinates and remembered label targets
    (table frame already), and the TCP of the fixed poses it streams (FK in
    the arm frame, lifted by the arm's base_pose `T`). Unknown or invalid
    ones are left to the skill's own refusal."""
    from ..skills import arm_frame

    out: list[tuple[str, np.ndarray]] = []
    if name == "place_at":
        try:
            z = args.get("z")
            out.append(("place_at's point", np.array([float(args["x"]), float(args["y"]),
                                                       float(z) if z is not None else 0.0])))
        except (KeyError, TypeError, ValueError):
            pass
    beliefs = getattr(rt, "beliefs", None)
    for key in ("label", "object", "destination"):
        value = args.get(key)
        if isinstance(value, str) and value.strip() and beliefs is not None:
            b = beliefs.find(value)
            if b is not None:
                out.append((f"{key} {value!r} (remembered at {np.round(b.position, 3).tolist()})",
                            np.asarray(b.position, dtype=float)))
    poses: list[tuple[str, np.ndarray]] = []
    try:
        if name == "wave":
            poses = [("the wave", q) for q in rt._wave_waypoints(args.get("cycles", 2))]
        elif name == "handover":
            poses = [("the handover pose", rt._profile_q("handover_q", "present the object to a human"))]
        elif name == "throw":
            windup, release = rt._throw_keyframes(args.get("direction", "forward"))
            poses = [("the throw's wind-up", windup), ("the throw's release", release)]
        elif name == "rotate_joint":
            joint, degrees = args.get("joint"), float(args.get("degrees"))
            q = np.asarray(arm.get_state().q, dtype=float).copy()
            q[int(joint) - 1] += math.radians(degrees)
            poses = [("the rotated joint's pose", q)]
    except Exception:  # noqa: BLE001 -- the skill refuses bad arguments itself
        poses = []
    if kin is not None:
        for what, q in poses:
            tcp = np.asarray(kin.fk(np.asarray(q, dtype=float))[:3, 3], dtype=float)
            out.append((what, np.asarray(arm_frame.to_world(T, tcp), dtype=float)))
    return out


def check_call(rt, name: str, args: dict, arm) -> Refusal | None:
    """The refusal for this motion call on `arm` (the arm the call runs on),
    or None. Pure decision: it never commands, and its geometry reads (the
    arm state for rotate_joint) happen only when a visitor zone or a
    localized person exists. Arm-frame geometry is lifted into the table
    frame by `arm`'s own base_pose (its harness's, the one execute() bound)."""
    from ..skills import arm_frame

    harness = (getattr(arm, "__dict__", None) or {}).get("harness")
    kin = getattr(harness, "kin", None) or getattr(rt, "kin", None)
    refusal = _word_refusal(name, args)
    if refusal is None and name in ("move_relative", "rotate_gripper", "move_near_object"):
        refusal = _toward_viewer_refusal(args)
    if refusal is not None:
        return refusal
    try:
        T = arm_frame.world_from_base(arm)
    except arm_frame.ArmFrameError as exc:
        # execute() refuses such an arm before this guard; fail closed anyway
        return Refusal(f"cannot tell where this arm stands ({exc}), so not where its motion goes relative to people",
                       {"rule": "unknown_arm_frame"})
    people = _settings(rt, harness)
    if name == "throw":
        refusal = _throw_refusal(rt, args, people, kin, T)
    if refusal is not None:
        return refusal
    columns = _people_columns(rt, people.margin_m)
    if not columns and not people.zones:
        return None
    for what, point in _destinations(rt, name, args, arm, kin, T):
        hit = _hit(point, people, columns)
        if hit is not None:
            where = (f"in the visitor zone {hit['zone']!r}" if "zone" in hit
                     else f"within {people.margin_m:.2f} m of the person {hit['person']!r} the robot localized")
            return Refusal(f"{what} at {np.round(point, 3).tolist()} lies {where}",
                           {"rule": "destination_on_person", "what": what,
                            "point": np.round(point, 3).tolist(), **hit})
    return None


def refusal_result(name: str, refusal: Refusal) -> dict:
    """The tool result of a refused call: nothing was perceived or commanded."""
    return {
        "ok": False,
        "error": f"{REFUSAL_PREFIX}: {name} refused: {refusal.reason}; no command was sent",
        "refused_by": REFUSED_BY,
        "safety_class": SAFETY_CLASS,
        "arm_moved": False,
        "people_guard": dict(refusal.evidence),
        "next_action": ("Refuse this request and explain why to the person: the robot never throws at, "
                        "reaches for, points at or moves toward people or body parts. Do not retry it with "
                        "other arguments or another tool. Offer a safe alternative instead (an object on the "
                        "table, a destination on the table, or handover to give an object)."),
    }
