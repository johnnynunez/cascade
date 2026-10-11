"""Frames, magnitudes and history for natural-language arm motion.

A chat model driving this robot hears instructions like "lift it up", "a bit
more", "move it to the right", "bring it toward me", "tilt it down a little"
or "go back". Every one of them is a small relative motion whose meaning
depends on two things the words do not pin down: HOW FAR ("a bit") and IN
WHICH FRAME ("right" for the robot is not "right" on the screen). This module
is the single place that turns those words into one base-frame displacement
or rotation, and into a sentence that says how they were read. It never
moves anything: `SkillRuntime` plans, vets through the SafetyHarness and
executes; this module only interprets, validates and remembers.

Why the interpretation is always returned
-----------------------------------------
On the reference rig the scene camera sits on the robot's LEFT looking back
across the table, so "right on the screen" is the robot's BACK (-x) and
"toward the camera" is the robot's LEFT (+y). The same word therefore names
directions up to 90 degrees apart depending on whose point of view the
speaker took. Guessing silently would make a mid-size LLM confidently move
the arm the wrong way; instead every result carries the frame used, the
base-frame displacement and the same displacement described in the other
frame ("5 cm to the robot's right = away from the camera on screen"), so the
agent can tell the user what happened and the user can correct it.

The default frame for bare left/right/forward/back is the ROBOT BASE
---------------------------------------------------------------------
`DEFAULT_FRAME = "base"`. Reasons, in order of weight:

1. It is always resolvable. The viewer frame depends on a hand-eye
   calibration that can be missing, rejected, or invalidated at runtime by
   the extrinsic drift monitor; a default that can vanish mid-session would
   make the same sentence work at 10:00 and fail at 10:05.
2. It is the frame of every number the agent already sees (`world_state`,
   `localize_object`, `place_at` coordinates) and of the pre-existing
   `move_relative` and the reflex grammar ("move a bit left" -> +y). A
   second meaning for the same word in a different tier would be a silent
   inconsistency.
3. "forward/back" have a canonical meaning only for the robot (away from /
   toward its base). For a viewer they are ambiguous (into the screen? toward
   the camera?).

The viewer frame is selected explicitly (`frame="viewer"`) or IMPLIED by a
word that names the viewer (`toward_viewer`, `away_from_viewer`). The tool
descriptions tell the agent to use it whenever the person refers to
themselves or the screen ("my left", "right on the screen", "toward me").

Frames
------
- base:   +x forward (away from the robot base), +y the robot's left, +z up.
          Free vectors are [x, y, z].
- viewer: built from the fixed scene camera the demo screen shows. "right"
          is the image's +x axis projected onto the table plane; "away" is
          up x right (= "up on the screen"; for an oblique camera this is
          also away from the camera); "up" is world vertical. Free vectors
          are [right, away, up] -- a right-handed frame like the base.
- tool:   along the gripper. "in" moves along the approach axis (the way the
          jaws point), "out" backs off along it. Free vectors are
          [approach, jaw-opening axis, third axis] (`tool_axis_order` decides
          which TCP columns those are on each arm).

"up"/"down" are world vertical in every frame: gravity is the one direction
nobody disagrees about, so refusing them for the tool frame would only cost
the agent a retry.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..types import SkillError

#: per-call translation cap (m). A nudge is fine control: anything longer is
#: a carry (place_at / pick_and_place own those, with their pre-carry lift),
#: and a mis-heard frame can only ever cost this much.
MAX_STEP_M = 0.15
#: below this a move is noise against the arm's repeatability (and a typo for
#: centimetres passed as metres, 0.002, would otherwise look legitimate)
MIN_STEP_M = 0.005
#: per-call rotation cap (degrees): "a little" never swings a held object
#: through a large arc in one go
MAX_TURN_DEG = 30.0
MIN_TURN_DEG = 1.0

#: what the vague words mean (the tool descriptions publish this table so the
#: agent maps "a bit" -> small without guessing a number)
AMOUNTS_M = {"tiny": 0.01, "small": 0.02, "medium": 0.05, "large": 0.10}
AMOUNTS_DEG = {"tiny": 5.0, "small": 10.0, "medium": 15.0, "large": 30.0}

FRAMES = ("base", "viewer", "tool")
DEFAULT_FRAME = "base"


def _finite_number(value, name: str) -> float:
    """`value` as a float, refusing bools, strings and non-finite numbers.

    LLM tool calls carry JSON: a bool is a number in Python and "5 cm" is a
    string. Neither must reach a motion as a magnitude.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SkillError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise SkillError(f"{name} must be finite, got {value!r}")
    return out


def resolve_distance(distance_m=None, amount=None, *, default: str = "medium") -> tuple[float, str]:
    """(metres, how it was chosen) for one translation.

    An explicit `distance_m` wins over an `amount` word (the person said a
    number); the source string says the word was ignored so the agent can
    notice it sent both. Out-of-cap values are REFUSED, not clamped: a
    silent 20 -> 15 cm change is a different motion than the one asked for,
    and the refusal message tells the agent how to split it.
    """
    if distance_m is not None:
        d = _finite_number(distance_m, "distance_m")
        if d < MIN_STEP_M or d > MAX_STEP_M:
            raise SkillError(
                f"distance_m={d:g} is outside {MIN_STEP_M:g}..{MAX_STEP_M:g} m per call; "
                "for more, move in several calls (or use place_at for a carry)")
        source = "distance_m"
        if amount is not None:
            source += f" (amount {amount!r} ignored: distance_m given)"
        return d, source
    if amount is None:
        return AMOUNTS_M[default], f"default ({default}, {AMOUNTS_M[default] * 100:g} cm)"
    word = str(amount).strip().lower()
    if word not in AMOUNTS_M:
        raise SkillError(f"amount must be one of {sorted(AMOUNTS_M)}, got {amount!r}")
    return AMOUNTS_M[word], f"amount {word} ({AMOUNTS_M[word] * 100:g} cm)"


def resolve_angle(angle_deg=None, amount=None, *, default: str = "medium") -> tuple[float, str]:
    """(degrees, how it was chosen) for one orientation nudge; same rules
    as `resolve_distance` with the rotation caps."""
    if angle_deg is not None:
        a = _finite_number(angle_deg, "angle_deg")
        if a < MIN_TURN_DEG or a > MAX_TURN_DEG:
            raise SkillError(
                f"angle_deg={a:g} is outside {MIN_TURN_DEG:g}..{MAX_TURN_DEG:g} degrees per call")
        source = "angle_deg"
        if amount is not None:
            source += f" (amount {amount!r} ignored: angle_deg given)"
        return a, source
    if amount is None:
        return AMOUNTS_DEG[default], f"default ({default}, {AMOUNTS_DEG[default]:g} deg)"
    word = str(amount).strip().lower()
    if word not in AMOUNTS_DEG:
        raise SkillError(f"amount must be one of {sorted(AMOUNTS_DEG)}, got {amount!r}")
    return AMOUNTS_DEG[word], f"amount {word} ({AMOUNTS_DEG[word]:g} deg)"


# ── frames ──────────────────────────────────────────────────────────────────

_UP = np.array([0.0, 0.0, 1.0])

#: |table-plane part of the image's x axis| below this (cos 60 deg) means the
#: camera is rolled so far that "right on the screen" has no direction on the
#: table: refuse rather than amplify a near-vertical axis into a full move.
_MIN_RIGHT_HORIZONTAL = 0.5


@dataclass(frozen=True)
class ViewerAxes:
    """The on-screen directions as base-frame unit vectors.

    right: image +x projected onto the table plane; away: up x right ("up on
    the screen", which for an oblique camera is also away from it); up: world
    vertical. (right, away, up) is right-handed, like the base frame.
    """

    right: np.ndarray
    away: np.ndarray
    up: np.ndarray
    camera: str = "camera"


def viewer_axes(T_cam2base, camera: str = "camera") -> ViewerAxes:
    """Viewer axes from the scene camera's T_cam2base.

    Built from the image's x axis rather than the optical axis on purpose: a
    top-down camera has no horizontal optical axis ("toward the camera" is
    undefined physically) but its image still has a right and an up, and for
    an oblique camera both constructions agree. Only a camera rolled so its
    image x is near-vertical is refused.
    """
    T = np.asarray(T_cam2base, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise SkillError(f"viewer frame unavailable: camera {camera!r} has no finite calibrated "
                         "transform; use frame='base' (the robot's own directions)")
    x_cam = T[:3, 0]
    norm = float(np.linalg.norm(x_cam))
    horizontal = x_cam - float(x_cam @ _UP) * _UP
    h = float(np.linalg.norm(horizontal))
    if norm <= 0 or h < _MIN_RIGHT_HORIZONTAL * norm:
        tilt = math.degrees(math.acos(min(1.0, h / norm))) if norm > 0 else 90.0
        raise SkillError(f"viewer frame unavailable: camera {camera!r} image right axis is "
                         f"{tilt:.0f} deg out of the table plane; use frame='base'")
    right = horizontal / h
    away = np.cross(_UP, right)
    return ViewerAxes(right=right, away=away, up=_UP.copy(), camera=str(camera))


#: (approach, jaw-opening, third) TCP column indices per `tool_axis_order`
#: (see grasping.obb_grasp._yaw_rotation for the measured conventions)
_TOOL_COLUMNS = {"down_open": (0, 1, 2), "open_down": (2, 0, 1), "third_open_down": (2, 1, 0)}


@dataclass(frozen=True)
class ToolAxes:
    """The gripper's own axes in the base frame."""

    approach: np.ndarray
    jaw: np.ndarray
    third: np.ndarray


def tool_axes(R, axis_order: str) -> ToolAxes:
    if axis_order not in _TOOL_COLUMNS:
        raise SkillError(f"unknown tool_axis_order {axis_order!r}")
    R = np.asarray(R, dtype=float)
    a, j, t = _TOOL_COLUMNS[axis_order]
    return ToolAxes(approach=R[:, a].copy(), jaw=R[:, j].copy(), third=R[:, t].copy())


# ── direction words ─────────────────────────────────────────────────────────

_BASE_DIRS = {"forward": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
              "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}
_VERTICAL = {"up": 1.0, "down": -1.0}
_VIEWER_ONLY = ("toward_viewer", "away_from_viewer")
_TOOL_ONLY = ("in", "out")
DIRECTIONS = ("forward", "back", "left", "right", "up", "down") + _VIEWER_ONLY + _TOOL_ONLY
AGAIN = "again"

#: unambiguous spellings a chat model produces; the result always reports the
#: canonical word, so normalizing here costs no transparency
_ALIASES = {
    "forwards": "forward", "ahead": "forward", "backward": "back", "backwards": "back",
    "upward": "up", "upwards": "up", "raise": "up", "lift": "up", "higher": "up",
    "downward": "down", "downwards": "down", "lower": "down",
    "toward_me": "toward_viewer", "towards_me": "toward_viewer", "to_me": "toward_viewer",
    "toward_the_camera": "toward_viewer", "towards_the_camera": "toward_viewer",
    "toward_camera": "toward_viewer", "towards_viewer": "toward_viewer",
    "closer_to_me": "toward_viewer",
    "away_from_me": "away_from_viewer", "away_from_the_camera": "away_from_viewer",
    "away_from_camera": "away_from_viewer", "away": "away_from_viewer",
    "inward": "in", "inwards": "in", "deeper": "in", "outward": "out", "outwards": "out",
    "retract": "out",
    "more": AGAIN, "a_bit_more": AGAIN, "keep_going": AGAIN, "repeat": AGAIN,
    "same_again": AGAIN, "further": AGAIN,
}

#: words that name two different motions: refused with what to call instead
_AMBIGUOUS = {
    "go_back": ("'go back' is ambiguous: to return to where the gripper was before the last "
                "move call undo_move; to move toward the robot base use direction='back'"),
    "return": ("'return' is ambiguous: call undo_move to retrace the last move, move_home "
               "for the raised initial pose, or move_to_rest for home / pose 0 / rest"),
}


def normalize_direction(word) -> str:
    """Canonical direction word (or "again"), refusing ambiguous/unknown ones."""
    if not isinstance(word, str):
        raise SkillError(f"direction must be a word, got {word!r}")
    w = "_".join(word.strip().lower().replace("-", " ").split())
    if w in _AMBIGUOUS:
        raise SkillError(_AMBIGUOUS[w])
    w = _ALIASES.get(w, w)
    if w not in DIRECTIONS and w != AGAIN:
        raise SkillError(f"direction must be one of {list(DIRECTIONS) + [AGAIN]}, got {word!r}")
    return w


def normalize_frame(frame) -> str | None:
    if frame is None:
        return None
    if not isinstance(frame, str) or frame.strip().lower() not in FRAMES:
        raise SkillError(f"frame must be one of {list(FRAMES)}, got {frame!r}")
    return frame.strip().lower()


# ── descriptions ────────────────────────────────────────────────────────────


def _cm(metres: float) -> str:
    return f"{metres * 100:.1f} cm"


def fmt_vec(v) -> str:
    return "[" + ", ".join(f"{float(c):+.3f}" for c in np.asarray(v, dtype=float)) + "]"


def _dominant(names_and_values) -> str:
    """'mostly A, slightly B' over signed components, largest first."""
    parts = sorted(((abs(v), n) for n, v in names_and_values if abs(v) > 1e-9), reverse=True)
    if not parts:
        return "no displacement"
    top = parts[0][0]
    named = [n for mag, n in parts if mag >= 0.25 * top]
    if len(named) == 1:
        return named[0]
    return f"mostly {named[0]}, " + ", ".join(f"partly {n}" for n in named[1:])


def describe_base(v) -> str:
    """A base-frame vector in the robot's words (+y is the robot's LEFT)."""
    x, y, z = (float(c) for c in np.asarray(v, dtype=float))
    return _dominant([
        ("forward (away from the robot base, +x)" if x > 0 else "back (toward the robot base, -x)", x),
        ("to the robot's left (+y)" if y > 0 else "to the robot's right (-y)", y),
        ("up (+z)" if z > 0 else "down (-z)", z),
    ])


def describe_on_screen(v, axes: ViewerAxes) -> str:
    """The same vector as the person watching camera `axes.camera` sees it."""
    v = np.asarray(v, dtype=float)
    r, a, u = float(v @ axes.right), float(v @ axes.away), float(v @ axes.up)
    return _dominant([
        ("right on the screen" if r > 0 else "left on the screen", r),
        ("away from the camera (up on the screen)" if a > 0
         else "toward the camera (down on the screen)", a),
        ("up" if u > 0 else "down", u),
    ])


# ── translation resolution ──────────────────────────────────────────────────


@dataclass
class Translation:
    """One resolved translation: what was asked, in which frame, and the
    base-frame displacement it became."""

    frame: str
    direction: str | None
    vector: tuple | None
    distance: float
    delta: np.ndarray
    words: str
    camera: str | None = None
    notes: list = field(default_factory=list)

    def request(self) -> dict:
        """What 'again' re-resolves: the words, not the vector, so a tool-frame
        repeat follows the gripper as it is now."""
        return {"direction": self.direction, "vector": self.vector, "frame": self.frame}


def _parse_vector(vector) -> np.ndarray:
    if isinstance(vector, (str, bytes)) or not hasattr(vector, "__len__") or len(vector) != 3:
        raise SkillError(f"vector must be 3 numbers in metres, got {vector!r}")
    v = np.array([_finite_number(c, "vector component") for c in vector], dtype=float)
    n = float(np.linalg.norm(v))
    if n < MIN_STEP_M or n > MAX_STEP_M:
        raise SkillError(f"vector length {n:.3f} m is outside {MIN_STEP_M:g}..{MAX_STEP_M:g} m per call")
    return v


def resolve_translation(direction=None, *, frame=None, vector=None, distance=None,
                        viewer=None, tool=None) -> Translation:
    """Resolve a named direction OR a free vector into a base-frame delta.

    `viewer` / `tool` are zero-argument callables returning ViewerAxes /
    ToolAxes, called only when that frame is needed: a base move must never
    fail because a camera calibration is missing. A contradictory explicit
    frame ("toward_viewer" in the tool frame) is refused, never re-read.
    """
    frame = normalize_frame(frame)
    if direction is not None and vector is not None:
        raise SkillError("give either a direction or a vector, not both")
    if vector is not None:
        if distance is not None:
            raise SkillError("a vector carries its own length: omit distance_m/amount with vector")
        v = _parse_vector(vector)
        frame = frame or DEFAULT_FRAME
        if frame == "base":
            delta, cam = v.copy(), None
            basis = "base [x forward, y left, z up]"
        elif frame == "viewer":
            axes = _viewer_or_refuse(viewer)
            delta, cam = v[0] * axes.right + v[1] * axes.away + v[2] * axes.up, axes.camera
            basis = f"viewer [right, away, up] of camera {axes.camera!r}"
        else:
            t = tool()
            delta, cam = v[0] * t.approach + v[1] * t.jaw + v[2] * t.third, None
            basis = "tool [approach, jaw opening, third]"
        n = float(np.linalg.norm(delta))
        words = (f"{_cm(n)} along {fmt_vec(v)} in the {basis} frame = base {fmt_vec(delta)} "
                 f"({describe_base(delta)})")
        return Translation(frame, None, tuple(float(c) for c in v), n, delta, words, cam)
    if direction is None:
        raise SkillError("give a direction (e.g. 'left', 'up', 'toward_viewer') or a vector")
    word = normalize_direction(direction)
    if word == AGAIN:
        raise SkillError("'again' repeats the previous move; resolve it against the history first")
    if distance is None:
        raise SkillError("distance missing")
    d = float(distance)
    if word in _VERTICAL:
        delta = _UP * _VERTICAL[word] * d
        return Translation(frame or DEFAULT_FRAME, word, None, d, delta,
                           f"{_cm(d)} {word} (world vertical, the same in every frame)")
    if word in _VIEWER_ONLY:
        if frame not in (None, "viewer"):
            raise SkillError(f"direction {word!r} names the viewer, so frame must be 'viewer' "
                             f"(or omitted), not {frame!r}")
        frame = "viewer"
    elif word in _TOOL_ONLY:
        if frame not in (None, "tool"):
            raise SkillError(f"direction {word!r} is along the gripper, so frame must be 'tool' "
                             f"(or omitted), not {frame!r}")
        frame = "tool"
    else:
        frame = frame or DEFAULT_FRAME
        if frame == "tool":
            raise SkillError(f"the tool frame only knows 'in'/'out' (along the gripper); for "
                             f"{word!r} use frame='base' (the robot's own) or frame='viewer' "
                             "(as seen on the screen)")
    if frame == "base":
        delta = np.asarray(_BASE_DIRS[word], dtype=float) * d
        return Translation("base", word, None, d, delta,
                           f"{_cm(d)} {describe_base(delta)} -- the robot's own directions")
    if frame == "viewer":
        axes = _viewer_or_refuse(viewer)
        unit = {"right": axes.right, "left": -axes.right,
                "forward": axes.away, "away_from_viewer": axes.away,
                "back": -axes.away, "toward_viewer": -axes.away}[word]
        delta = unit * d
        said = {"forward": "away from the viewer (forward into the screen)",
                "back": "toward the viewer (back out of the screen)",
                "toward_viewer": "toward the viewer/camera", "away_from_viewer": "away from the viewer/camera",
                "left": "to the viewer's left", "right": "to the viewer's right"}[word]
        return Translation("viewer", word, None, d, delta,
                           f"{_cm(d)} {said} as seen on camera {axes.camera!r} = base "
                           f"{fmt_vec(delta)} ({describe_base(delta)})", axes.camera)
    t = tool()
    unit = t.approach if word == "in" else -t.approach
    delta = unit * d
    said = "along the gripper's approach (in, where the jaws point)" if word == "in" else \
        "back along the gripper's approach (out, away from where the jaws point)"
    return Translation("tool", word, None, d, delta,
                       f"{_cm(d)} {said} = base {fmt_vec(delta)} ({describe_base(delta)})")


def _viewer_or_refuse(viewer) -> ViewerAxes:
    if viewer is None:
        raise SkillError("viewer frame unavailable: no scene camera wired; use frame='base'")
    try:
        return viewer()
    except SkillError as exc:
        msg = str(exc)
        if "frame='base'" not in msg:
            msg += "; use frame='base' (the robot's own directions)"
        raise SkillError(msg) from exc


# ── orientation nudges ──────────────────────────────────────────────────────


def rotation_matrix(axis, angle_rad: float) -> np.ndarray:
    """Rodrigues: rotation by `angle_rad` about unit `axis` (right-hand rule)."""
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


_ROLLS = {"clockwise": 1.0, "counterclockwise": -1.0}
_ROLL_ALIASES = {"cw": "clockwise", "ccw": "counterclockwise", "anticlockwise": "counterclockwise",
                 "anti_clockwise": "counterclockwise", "counter_clockwise": "counterclockwise"}
_TILTS = ("up", "down", "left", "right", "forward", "back") + _VIEWER_ONLY
ROTATION_DIRECTIONS = tuple(_ROLLS) + _TILTS

#: The wrist camera is mounted ON TOP of the gripper (user safety
#: requirement, 2026-10-11): rolled under, it faces the table and hits it.
#: Words that ask for exactly that are refused by name, before any angle.
CAMERA_ON_TOP = "the wrist camera must stay on top"
_UPSIDE_DOWN_WORDS = ("upside_down", "flip", "flip_over", "flip_it", "invert", "turn_over",
                      "half_turn", "roll_over")


def normalize_rotation_direction(word) -> str:
    if not isinstance(word, str):
        raise SkillError(f"direction must be a word, got {word!r}")
    w = "_".join(word.strip().lower().replace("-", " ").split())
    for prefix in ("tilt_", "point_", "lean_", "roll_", "rotate_", "turn_", "twist_"):
        if w.startswith(prefix) and w[len(prefix):]:
            w = w[len(prefix):]
            break
    if w in _UPSIDE_DOWN_WORDS or "upside" in w:
        raise SkillError(f"refused: {CAMERA_ON_TOP} (it is mounted on the top of the gripper and "
                         "would hit the table); the gripper is never rolled upside down")
    w = _ROLL_ALIASES.get(w, w)
    w = _ALIASES.get(w, w) if w not in _ROLLS else w
    if w == AGAIN:
        return w
    if w not in ROTATION_DIRECTIONS:
        raise SkillError(f"direction must be one of {list(ROTATION_DIRECTIONS) + [AGAIN]}, got {word!r}")
    return w


@dataclass
class Rotation:
    """One resolved orientation nudge about the TCP point."""

    kind: str            # "roll" | "tilt"
    direction: str
    frame: str | None
    angle_deg: float
    axis: np.ndarray     # base-frame unit axis
    words: str

    def request(self) -> dict:
        return {"direction": self.direction, "frame": self.frame}


#: below this |a x d| the gripper points (anti)parallel to the tilt target
_MIN_TILT_LEVER = math.sin(math.radians(5.0))


def resolve_rotation(direction, *, frame=None, angle_deg: float, tool, viewer=None) -> Rotation:
    """Resolve a roll ("clockwise"/"counterclockwise", about the gripper's own
    approach axis, as seen looking along it -- for a downward gripper that is
    as seen from above) or a tilt toward a direction (the jaws swing toward
    it) into a base-frame rotation axis through the TCP.

    A tilt that would overshoot its target direction is refused with the
    room that is left, so "tilt it down a lot" from nearly-vertical never
    swings past vertical into the other side.
    """
    word = normalize_rotation_direction(direction)
    if word == AGAIN:
        raise SkillError("'again' repeats the previous rotation; resolve it against the history first")
    frame = normalize_frame(frame)
    angle = float(angle_deg)
    t = tool()
    if word in _ROLLS:
        if frame is not None:
            raise SkillError("a roll turns about the gripper's own axis: omit frame for "
                             "clockwise/counterclockwise")
        axis = t.approach / np.linalg.norm(t.approach) * _ROLLS[word]
        return Rotation("roll", word, None, angle, axis,
                        f"roll {angle:g} deg {word} as seen looking along the gripper "
                        "(from behind the wrist; for a gripper pointing down, as seen from above)")
    if word in _VIEWER_ONLY:
        if frame not in (None, "viewer"):
            raise SkillError(f"direction {word!r} names the viewer, so frame must be 'viewer' "
                             f"(or omitted), not {frame!r}")
        frame = "viewer"
    elif word in _VERTICAL:
        frame = frame or DEFAULT_FRAME
    else:
        frame = frame or DEFAULT_FRAME
    if frame == "tool":
        raise SkillError("a tilt needs a direction in the world: use frame='base' or 'viewer'")
    if word in _VERTICAL:
        target, said = _UP * _VERTICAL[word], word
    elif frame == "viewer":
        axes = _viewer_or_refuse(viewer)
        target = {"right": axes.right, "left": -axes.right, "forward": axes.away,
                  "away_from_viewer": axes.away, "back": -axes.away,
                  "toward_viewer": -axes.away}[word]
        said = f"{word.replace('_', ' ')} as seen on camera {axes.camera!r}"
    else:
        target, said = np.asarray(_BASE_DIRS[word], dtype=float), describe_base(_BASE_DIRS[word])
    a = t.approach / np.linalg.norm(t.approach)
    cross = np.cross(a, target)
    lever = float(np.linalg.norm(cross))
    room = math.degrees(math.atan2(lever, float(a @ target)))
    if lever < _MIN_TILT_LEVER:
        if float(a @ target) > 0:
            raise SkillError(f"the gripper already points {said}; there is nothing to tilt toward")
        raise SkillError(f"the gripper points the opposite way from {said}; a tilt toward it has "
                         "no single direction -- tilt sideways first")
    if angle > room + 1e-6:
        raise SkillError(f"only {room:.0f} deg left before the gripper points straight {said}; "
                         f"asked {angle:g} deg")
    return Rotation("tilt", word, frame, angle, cross / lever,
                    f"tilt {angle:g} deg so the jaws point more {said} (about the TCP)")


def camera_up_violation(rotations, up_tcp, min_z: float = 0.0, tol: float = 1e-9) -> str | None:
    """Why a path of TCP orientations would turn the wrist camera under, or None.

    `up_tcp` is the camera's up axis in the TCP frame (reBot: TCP +z, which
    equals base +z at home); its base-frame z component must stay >= `min_z`
    at EVERY sample, because a roll whose ends are fine can still pass
    through upside-down. Escape rule, like the harness's: from a pose that is
    already below the limit, samples may not go LOWER than the start (so the
    camera can always be brought back up, never further down). `tol` absorbs
    IK residuals on a pose that sits exactly on the limit (a top-down reBot
    grasp holds the camera axis exactly horizontal).
    """
    up = np.asarray(up_tcp, dtype=float)
    zs = [float((np.asarray(R, dtype=float) @ up)[2]) for R in rotations]
    if not zs:
        return None
    floor = min(float(min_z), zs[0])
    worst = min(zs)
    if worst < floor - float(tol):
        deg = math.degrees(math.asin(max(-1.0, min(1.0, worst))))
        return (f"{CAMERA_ON_TOP}: this motion would turn the gripper's camera-up axis to "
                f"{deg:+.0f} deg from horizontal (z component {worst:.2f} < {float(min_z):.2f}); "
                "it is never rolled upside down")
    return None


# ── wrist camera volume ─────────────────────────────────────────────────────

#: largest camera dimension the config accepts (m): beyond this it is a typo
_MAX_CAMERA_DIM_M = 0.30


@dataclass(frozen=True)
class CameraBox:
    """The wrist camera as an oriented box in the base frame.

    The reBot camera sits on TOP of the gripper and sticks out 6-8 cm above
    it (user, 2026-10-11). The orientation rule (`camera_up_violation`) keeps
    it from facing the table; this box is what lets a check see that a
    rolled or tilted gripper can still drop one side of the camera -- or the
    whole camera -- into the table or a keep-out zone while the TCP itself is
    fine. Local axes: approach (the camera reaches `behind` back from the
    fingertip plane), jaw axis (+-`half_width`), camera up (0..`protrusion`).
    """

    center: np.ndarray
    axes: np.ndarray        # columns: approach, jaw, up (base frame, orthonormal)
    half: np.ndarray        # half extents along those columns

    def corners(self) -> np.ndarray:
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return self.center + (signs * self.half) @ self.axes.T

    def min_z(self) -> float:
        # the lowest point of a box is a corner: closed form of corners()[:, 2].min()
        return float(self.center[2] - np.sum(self.half * np.abs(self.axes[2, :])))

    def points(self, spacing_m: float = 0.02) -> np.ndarray:
        """A grid over the box (faces included), at most `spacing_m` apart."""
        grids = []
        for h in self.half:
            n = max(1, math.ceil(2.0 * float(h) / spacing_m))
            grids.append(np.linspace(-float(h), float(h), n + 1) if h > 0 else np.zeros(1))
        local = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
        return self.center + local @ self.axes.T


def camera_box(T_tcp, approach_tcp, up_tcp, *, protrusion_m: float, behind_m: float,
               half_width_m: float) -> CameraBox:
    """The camera's box for TCP pose `T_tcp`. `approach_tcp` / `up_tcp` are the
    gripper's approach and the camera's up axis IN THE TCP FRAME."""
    dims = {"protrusion_m": protrusion_m, "behind_m": behind_m, "half_width_m": half_width_m}
    for name, v in dims.items():
        if (isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))
                or not math.isfinite(float(v)) or not 0.0 <= float(v) <= _MAX_CAMERA_DIM_M):
            raise SkillError(f"relative_motion.wrist_camera.{name} must be within "
                             f"0..{_MAX_CAMERA_DIM_M:g} m, got {v!r}")
    T = np.asarray(T_tcp, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    a = R @ (np.asarray(approach_tcp, dtype=float) / np.linalg.norm(approach_tcp))
    u = R @ np.asarray(up_tcp, dtype=float)
    u = u - float(u @ a) * a
    if float(np.linalg.norm(u)) < 1e-6:
        raise SkillError("relative_motion.tool_up_axis must not be parallel to the approach")
    u = u / np.linalg.norm(u)
    j = np.cross(u, a)
    axes = np.column_stack([a, j, u])
    center = p - a * (float(behind_m) / 2.0) + u * (float(protrusion_m) / 2.0)
    half = np.array([float(behind_m) / 2.0, float(half_width_m), float(protrusion_m) / 2.0])
    return CameraBox(center=center, axes=axes, half=half)


def box_overlaps_aabb(center, axes, half, aabb_min, aabb_max) -> bool:
    """Exact oriented-box vs axis-aligned-box test (separating axes: the 3
    world axes, the 3 box axes and their 9 edge cross products). Touching
    counts as overlapping."""
    c = np.asarray(center, dtype=float)
    A = np.asarray(axes, dtype=float)
    e = np.asarray(half, dtype=float)
    lo, hi = np.asarray(aabb_min, dtype=float), np.asarray(aabb_max, dtype=float)
    bc, f = (lo + hi) / 2.0, (hi - lo) / 2.0
    d = c - bc
    candidates = [np.eye(3)[i] for i in range(3)] + [A[:, i] for i in range(3)]
    candidates += [np.cross(np.eye(3)[i], A[:, k]) for i in range(3) for k in range(3)]
    for L in candidates:
        n = float(np.linalg.norm(L))
        if n < 1e-9:  # parallel edges: covered by the face axes
            continue
        L = L / n
        r_box = float(np.sum(e * np.abs(L @ A)))
        r_aabb = float(np.sum(f * np.abs(L)))
        if abs(float(L @ d)) > r_box + r_aabb + 1e-12:
            return False
    return True


# ── history ─────────────────────────────────────────────────────────────────

#: an undo is only safe from where the last recorded move ENDED; further than
#: this (another skill moved the arm) the history is stale and is cleared
UNDO_POS_TOL_M = 0.015
UNDO_ROT_TOL_DEG = 5.0


def pose_matches(T_a, T_b, pos_tol: float = UNDO_POS_TOL_M, rot_tol_deg: float = UNDO_ROT_TOL_DEG) -> bool:
    T_a, T_b = np.asarray(T_a, dtype=float), np.asarray(T_b, dtype=float)
    if float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])) > pos_tol:
        return False
    R = T_a[:3, :3].T @ T_b[:3, :3]
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(cos)) <= rot_tol_deg


@dataclass
class MoveRecord:
    """One completed move of the spatial tools: the MEASURED joint vector at
    its start and after every segment, so an undo retraces the same route
    backwards (never a shortcut through space nobody vetted for it)."""

    tool: str
    words: str
    waypoints: list
    tcp_end: np.ndarray


class MoveHistory:
    """Bounded LIFO of this tool family's own moves (one per arm)."""

    def __init__(self, maxlen: int = 10):
        self._items: deque = deque(maxlen=max(1, int(maxlen)))

    def __len__(self) -> int:
        return len(self._items)

    def push(self, record: MoveRecord) -> None:
        self._items.append(record)

    def last(self) -> MoveRecord | None:
        return self._items[-1] if self._items else None

    def pop(self) -> MoveRecord:
        return self._items.pop()

    def clear(self) -> None:
        self._items.clear()
