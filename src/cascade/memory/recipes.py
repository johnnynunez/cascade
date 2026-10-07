"""Task-Specific Memory recipes (Harness-VLA v4; ROADMAP follow-up #9).

Tier-2 ``ExperienceMemory`` (agent/reflex.py) keys a proven plan on the
instruction text and stores the plan's skill calls verbatim. For a plan that
came out of the LLM tier that means ``place_at(x=0.20, y=-0.15)``: the
coordinate where the bowl WAS. Replaying that on a table where the bowl has
moved puts the cube on bare table -- a remembered coordinate aiming the arm,
exactly what the persistent belief store already forbids for restored
beliefs ("they inform the agent, they do not aim it", memory/beliefs.py).

Harness-VLA v4 (RLinf/RPent, docs 2026-09-02) spells out the fix as
*Task-Specific Memory*: a successful run is serialized as JSONL with every
concrete xyz REPLACED by a symbolic perception query plus a one-line
semantic summary, and the queries are re-grounded through live perception
at replay. This module is that transformation, in both directions:

* ``symbolize_run(tool_log, scene)`` -- the run's successful motion steps
  with each base-frame coordinate rewritten as
  ``{"$target": {"query": "localize_object", "label", "offset_m", "args"}}``,
  anchored on an object perceived at the START of the run. Labels and
  destination names (``"drop zone"``, a bowl) are already symbolic and stay
  so. A coordinate with no anchor object in range is NOT stored raw: the
  whole recipe is refused (``RecipeError``), because an unanchored
  coordinate is precisely the thing this memory exists to not remember.
* ``ground(calls, localize)`` -- the stored queries resolved to concrete
  arguments for THIS table through the caller's perception. A query that
  fails to ground raises ``GroundingError``; nothing is substituted from the
  stored run -- there is no stored coordinate to substitute, by construction.

Advisory by construction: grounding produces skill ARGUMENTS. Every motion
they request still goes through the harness, which remains the sole
authority that refuses motion; no remembered value ever aims the jaws.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

SkillCall = tuple[str, dict]

#: Marker key under which a symbolic target replaces a skill's coordinates.
TARGET_KEY = "$target"

#: Which arguments of which skills are concrete base-frame coordinates.
#: Everything else a skill takes (labels, directions, distances, turns,
#: materials) is symbolic or scene-independent and is stored as-is.
COORDINATE_ARGS: dict[str, tuple[str, ...]] = {
    "place_at": ("x", "y", "z"),
}

#: Skills whose arguments address a PIXEL: a camera-frame coordinate is as
#: scene-bound as a base-frame one. The step is rewritten to the named-object
#: skill for whatever the run actually ended up holding, or refused.
PIXEL_SKILLS: dict[str, str] = {"grasp_at_pixel": "grasp_object"}

#: Motion skills that are not a repeatable step of a task (they undo one).
EXCLUDED_SKILLS = frozenset({"reset_scene"})

#: Skills whose success ENDS the call holding something (result key ``held``
#: names it), and skills whose success ends it with the jaws empty. Tracked
#: so the manipulated object does not anchor its own placement while
#: another object can.
_GRASPING = {"grasp_object", "grasp_at_pixel"}
_RELEASING = {"place_at", "place_on_object", "pick_and_place", "handover", "throw", "open_gripper"}

#: An object farther than this (xy, metres) from a coordinate cannot anchor
#: it. Tabletop scale: a placement 40 cm from every perceived object is not
#: "relative to" any of them.
ANCHOR_RADIUS_M = 0.40


class RecipeError(ValueError):
    """The run cannot be stored as a recipe without storing a coordinate."""


class GroundingError(RuntimeError):
    """A stored perception query did not resolve on the current scene."""


# ── helpers ──────────────────────────────────────────────────────────────


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _position(entry: Any) -> list[float] | None:
    """A 3-vector from a perception result / belief row, or None."""
    if not isinstance(entry, dict):
        return None
    pos = entry.get("position")
    if pos is None or isinstance(pos, (str, bytes)):
        return None
    try:
        vals = [_finite(v) for v in list(pos)[:3]]
    except TypeError:
        return None
    if len(vals) != 3 or any(v is None for v in vals):
        return None
    return [float(v) for v in vals]  # type: ignore[arg-type]


def is_symbolic(calls: Iterable[Any]) -> bool:
    """True when any step carries a symbolic target that needs grounding."""
    for call in calls:
        try:
            _, args = call
        except (TypeError, ValueError):
            continue
        if isinstance(args, dict) and isinstance(args.get(TARGET_KEY), dict):
            return True
    return False


def anchor_scene(tool_log: Iterable[dict], beliefs: Iterable[dict] | None) -> list[dict]:
    """Objects perceived at the START of the run: ``[{"label", "position"}]``.

    The belief snapshot the orchestrator took right before the first motion
    is the fused, freshest picture and wins; perception results the agent
    read before that first motion (``get_observation.objects_visible``,
    ``localize_object``, ``list_objects.objects``) fill in anything the
    snapshot lacks. Nothing observed AFTER a motion is an anchor: the run
    itself may have moved it, and replay grounds everything up front.
    """
    scene: dict[str, list[float]] = {}
    for row in beliefs or []:
        pos = _position(row)
        label = (row or {}).get("label") if isinstance(row, dict) else None
        if label and pos is not None:
            scene.setdefault(str(label), pos)
    from ..skills.runtime import _MOTION_SKILLS  # local: runtime imports memory

    for entry in tool_log:
        name = entry.get("tool")
        if name in _MOTION_SKILLS:
            break
        result = entry.get("result") or {}
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        rows: list[dict] = []
        if name == "get_observation":
            rows = [r for r in result.get("objects_visible") or [] if isinstance(r, dict)]
        elif name == "list_objects":
            rows = [r for r in result.get("objects") or [] if isinstance(r, dict)]
        elif name == "localize_object":
            label = result.get("detected_as") or result.get("label")
            rows = [{"label": label, "position": result.get("position")}]
        for row in rows:
            pos = _position(row)
            label = row.get("label")
            if label and pos is not None:
                scene.setdefault(str(label), pos)
    return [{"label": k, "position": v} for k, v in scene.items()]


def _anchor(point: list[float], scene: list[dict], held: str | None) -> tuple[str, list[float]]:
    """Choose the object a coordinate is expressed relative to.

    Nearest object in ``ANCHOR_RADIUS_M`` that is NOT the one being
    manipulated (a destination: the bowl the cube went into); the
    manipulated object's own start position is the fallback. The fallback
    generalises as "relative to where it was", which is right for "move it
    10 cm left" and approximate for an absolute table region -- but it is
    still a perception query, re-grounded at replay, never the coordinate.
    """
    ranked = []
    for obj in scene:
        pos = _position(obj)
        label = obj.get("label")
        if pos is None or not label:
            continue
        d = math.hypot(point[0] - pos[0], point[1] - pos[1])
        if d <= ANCHOR_RADIUS_M:
            ranked.append((str(label) == str(held), d, str(label), pos))
    if not ranked:
        raise RecipeError(
            f"coordinate ({point[0]:.3f}, {point[1]:.3f}) has no perceived object within "
            f"{ANCHOR_RADIUS_M:.2f} m to anchor it; refusing to store a raw coordinate"
        )
    ranked.sort(key=lambda r: (r[0], r[1]))
    _, _, label, pos = ranked[0]
    return label, pos


def symbolize_run(tool_log: Iterable[dict], scene: Iterable[dict]) -> list[SkillCall]:
    """Successful motion steps of a run with every coordinate made symbolic.

    Returns ``[]`` when the run moved nothing (an answer, not a recipe).
    Raises ``RecipeError`` rather than storing any concrete coordinate.
    """
    from ..skills.runtime import _MOTION_SKILLS  # local: runtime imports memory

    scene = list(scene)
    steps: list[SkillCall] = []
    held: str | None = None
    for entry in tool_log:
        name = entry.get("tool")
        result = entry.get("result") or {}
        if name not in _MOTION_SKILLS or name in EXCLUDED_SKILLS:
            continue
        if not isinstance(result, dict) or not result.get("ok"):
            continue
        args = dict(entry.get("args") or {})
        if name in PIXEL_SKILLS:
            label = result.get("held")
            if not label:
                raise RecipeError(
                    f"{name} succeeded without naming what it held; a pixel is a "
                    "scene-bound coordinate and cannot be stored"
                )
            steps.append((PIXEL_SKILLS[name], {"label": str(label)}))
            held = str(label)
            continue
        coord_keys = [k for k in COORDINATE_ARGS.get(name, ()) if args.get(k) is not None]
        if coord_keys:
            values = [_finite(args[k]) for k in coord_keys]
            if any(v is None for v in values) or len(values) < 2:
                raise RecipeError(f"{name} has a non-numeric coordinate {args!r}")
            label, pos = _anchor(values[:3], scene, held)  # type: ignore[arg-type]
            offset = [round(float(v) - pos[i], 4) for i, v in enumerate(values)]  # type: ignore[arg-type]
            for k in coord_keys:
                args.pop(k)
            args[TARGET_KEY] = {
                "query": "localize_object",
                "label": label,
                "offset_m": offset,
                "args": coord_keys,
            }
        steps.append((name, args))
        if name in _GRASPING and result.get("held"):
            held = str(result["held"])
        elif name in _RELEASING:
            held = None
    return steps


def ground(
    calls: Iterable[SkillCall],
    localize: Callable[[str], dict | None],
) -> tuple[list[SkillCall], int]:
    """Resolve every symbolic target through ``localize(label) -> result``.

    ``localize`` is the runtime's own ``localize_object`` (a perception call,
    not a motion); its result must be ``ok`` and carry a 3D ``position``.
    Each label is grounded once per replay. Returns ``(concrete calls,
    number of queries grounded)``. Raises ``GroundingError`` on the first
    query that does not resolve -- the caller aborts BEFORE any motion.
    """
    cache: dict[str, list[float]] = {}
    grounded: list[SkillCall] = []
    n = 0
    for name, args in calls:
        target = args.get(TARGET_KEY) if isinstance(args, dict) else None
        if not isinstance(target, dict):
            grounded.append((name, dict(args)))
            continue
        label = str(target.get("label") or "")
        if not label:
            raise GroundingError(f"{name}: symbolic target without a label")
        if label not in cache:
            try:
                result = localize(label)
            except Exception as exc:  # noqa: BLE001 -- perception faults are grounding failures
                raise GroundingError(f"localize_object({label!r}) raised {type(exc).__name__}: {exc}") from exc
            if not isinstance(result, dict) or not result.get("ok", False):
                err = (result or {}).get("error") if isinstance(result, dict) else "no result"
                raise GroundingError(f"localize_object({label!r}) failed: {err}")
            pos = _position(result)
            if pos is None:
                raise GroundingError(
                    f"localize_object({label!r}) returned no 3D position (a zone is not a fix)"
                )
            cache[label] = pos
            n += 1
        pos = cache[label]
        offset = [_finite(v) for v in target.get("offset_m") or []]
        keys = list(target.get("args") or ("x", "y", "z")[: len(offset)])
        if len(keys) != len(offset) or any(v is None for v in offset) or len(offset) < 2:
            raise GroundingError(f"{name}: malformed symbolic target {target!r}")
        new_args = {k: v for k, v in args.items() if k != TARGET_KEY}
        for i, key in enumerate(keys):
            new_args[key] = round(pos[i] + float(offset[i]), 4)  # type: ignore[arg-type]
        grounded.append((name, new_args))
    return grounded, n


def describe(calls: Iterable[SkillCall]) -> str:
    """One line per step, symbolic targets rendered as queries."""
    parts = []
    for name, args in calls:
        rendered = []
        for k, v in args.items():
            if k == TARGET_KEY and isinstance(v, dict):
                off = ", ".join(f"{o:+.3f}" for o in (v.get("offset_m") or []) if _finite(o) is not None)
                rendered.append(f"{'/'.join(v.get('args') or [])}=localize_object({v.get('label')!r})+({off})")
            else:
                rendered.append(f"{k}={v}")
        parts.append(f"{name}({', '.join(rendered)})")
    return "; ".join(parts)
