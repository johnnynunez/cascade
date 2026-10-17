"""The cursor: a queryable point probe over the camera image.

Anthropic's *Claude Plays Robotics* (Jul 2026) ran an ablation over visual aids
for a Panda arm and found something that cuts against the obvious intuition:

* depth maps overlaid on the frame -- **roughly neutral**
* labeled segmentation overlays  -- **roughly neutral**
* a **cursor tool** ("a small red X on the gripper cam that the model can move
  and query for the object and distance at that point") -- **large uplift for
  every model tested**; on a 10-task manipulation subset their strongest model
  went from **6% to 32%** success.

Their reading: "models mainly need better orientation, not a different view of
the scene". The static overlays carry the right information but the signal is
*too diffuse* to act on. The cursor works because it answers a specific
question with a specific number, on demand.

That distinction is the whole design of this module. A probe is not a picture:

    probe_point(u, v)  ->  depth at that pixel, the 3D point in the robot's
                           base frame, which tracked object is there, whether
                           it is inside the configured workspace, and how far it is
                           from the current gripper position.

Everything returned is a scalar the agent can compare, not a texture it has to
interpret. The inverse direction (`locate_pixel`) closes the loop: given an
object the agent knows about, where is it *in the image it is looking at*.

Note on this repo's existing surfaces: the dashboard's depth view and the VIA
annotated view are **human** diagnostics -- they are rendered for a person
watching a browser, not fed into the model's context. Anthropic's neutral
result is about overlays *given to the model*, so it does not condemn them;
but it does say plainly that adding more such overlays to the agent's context
is not where the win is. The win is here.

IMPORTANT SEMANTICS -- a probe returns the VISIBLE SURFACE, not the centroid.
Deprojecting a pixel gives the first surface the ray hits, so probing the
middle of a cube returns a point on its TOP FACE, offset toward the camera by
parallax. Measured on the live rig against Isaac physics truth: probing a
cube whose centre is at z=0.040 m returns z=0.062 m (+22 mm -- the top face of
a ~4.5 cm cube) and ~+24 mm in x (the ray meets the top face nearer the
camera). Round-trip 3D -> pixel -> 3D closes to **1.6 mm**, so the geometry is
sound; the offset is physics, not error.

Practical consequence: do NOT feed a probe straight into a grasp centre. Use
it for *relative* judgements -- where is this point, which object is here, how
far is the gripper -- and let the grasp planner keep using segmented point
clouds for the object frame. `probe_top_z` is reported separately for the case
where the top face is exactly what you want (grasp depth).
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

from .visual_interface import camera_extrinsics

#: Radius (px) of the median window used to read depth, so one dead pixel in
#: a sensor depth map does not produce a wild 3D point.
PATCH_R = 3

#: A probe within this distance (m) of a tracked object's centre is reported
#: as being "on" that object.
HIT_RADIUS_M = 0.06

#: Words that end the object part of a query ("mug on the shelf" -> "mug").
#: "of" does not end it: "bag of chips" names the bag AND the chips.
_RELATION_WORDS = frozenset({"on", "in", "inside", "near", "next", "beside", "by", "at", "under",
                             "behind", "above", "below", "with", "from", "to"})
#: Two words for the same object, matched both ways ("green block" = "green cube").
_SYNONYMS = (frozenset({"cube", "block"}), frozenset({"cup", "mug"}))
#: Plurals the -s / -es rule misses.
_IRREGULAR_PLURALS = {"cherries": "cherry", "berries": "berry", "strawberries": "strawberry",
                      "knives": "knife", "leaves": "leaf", "loaves": "loaf",
                      "tomatoes": "tomato", "potatoes": "potato"}
#: Category nouns: in "orange fruit" the colour word IS the object's name, so
#: a label "orange" matches. "orange cube" keeps "orange" a colour only.
_CATEGORY_NOUNS = frozenset({"fruit", "fruits", "vegetable", "vegetables", "food", "snack", "produce"})
#: Shade words: "dark green" is a shade of green, "light blue" one of blue (and
#: it also reads as cyan, `_words`). A shade is compatible with its base colour;
#: two stated shades must be the same one (pale = light, deep = dark).
_SHADES = {"light": "light", "pale": "light", "dark": "dark", "deep": "dark", "bright": "bright"}
#: "dark-green" reads as "dark green".
_HYPHENATED_SHADE = re.compile(r"\b(" + "|".join(_SHADES) + r")-(?=[a-z])")
_PUNCTUATION = ".,;:!?\"'()[]{}"


def _words(text: str) -> list[str]:
    """The words of a query or a label: lower case, "light blue" -> "cyan",
    punctuation stripped from each word, counts ("2", "#3") dropped."""
    text = _HYPHENATED_SHADE.sub(r"\1 ", str(text).lower()).replace("light blue", "cyan")
    out = []
    for w in text.replace(",", " ").split():
        w = w.strip(_PUNCTUATION)
        if w and not w.lstrip("#").replace(".", "", 1).isdigit():
            out.append(w)
    return out


def _object_part(words: list[str]) -> list[str]:
    """The words before a relation ("red mug on the shelf" -> red mug)."""
    out = []
    for w in words:
        if w in _RELATION_WORDS:
            break
        out.append(w)
    return out


def _forms(word: str) -> set[str]:
    """A word, its singular guesses ("cubes" -> cube, "knives" -> knife) and
    its synonyms ("block" -> cube)."""
    out = {word, _IRREGULAR_PLURALS.get(word, word)}
    if len(word) > 3 and word.endswith("s"):
        out.add(word[:-1])
    if len(word) > 4 and word.endswith("es"):
        out.add(word[:-2])
    for group in _SYNONYMS:
        if out & group:
            out |= group
    return out


def _shares(word: str, words) -> bool:
    forms = _forms(word)
    return any(forms & _forms(w) for w in words)


def _content(words: list[str]) -> list[str]:
    """Words that can name an object (no articles, relations or "of")."""
    from .colors import _ARTICLES

    return [w for w in words if w not in _ARTICLES and w not in _RELATION_WORDS and w != "of"]


def _head_noun(words: list[str]) -> str | None:
    """The last word naming the object, before any "of" ("bag of chips" ->
    bag); a label's colour word counts ("orange" -> orange)."""
    from .colors import GENERIC_NOUNS

    part = _object_part(words)
    if "of" in part:
        part = part[:part.index("of")]
    names = [w for w in _content(part) if w not in GENERIC_NOUNS]
    return names[-1] if names else None


def _shades(text: str) -> set[tuple[str, str]]:
    """(shade, base colour) pairs the object part of a phrase states:
    "light blue mug" -> {(light, blue)}, "dark-green cube" -> {(dark, green)},
    "pale blue" -> {(light, blue)}."""
    from .colors import canonical_color

    raw = [w.strip(_PUNCTUATION) for w in str(text).lower().replace("-", " ").replace(",", " ").split()]
    part = _object_part(raw)
    return {(_SHADES[shade], c) for shade, word in zip(part, part[1:])
            if shade in _SHADES and (c := canonical_color(word)) is not None}


def _shade_bases(text: str) -> set[str]:
    """Base colours the object part of a phrase names through a shade word:
    "light blue mug" -> {blue}, "dark green cube" -> {green}."""
    return {base for _shade, base in _shades(text)}


def _stated_colours(text: str) -> set[str]:
    """The colours the object part of a phrase states, each shade with its
    base colour ("light blue mug" -> {cyan, blue})."""
    from .colors import canonical_color

    stated = {c for c in (canonical_color(w) for w in _object_part(_words(text))) if c is not None}
    return stated | _shade_bases(text)


class _Kind(NamedTuple):
    """What a label must agree with to name the queried object."""
    head: str | None          # the query's head noun; None = any object
    color: str | None         # its first colour (named in a refusal)
    colours: frozenset        # colours a label may state; empty = no constraint
    of_nouns: frozenset       # nouns after 'of', exact form ("bag of chips" -> chips)
    category_name: str | None  # a colour word under a category head ("orange fruit" -> orange)
    shades: frozenset = frozenset()  # (shade, base) pairs it states ("light blue" -> (light, blue))


def _queried_kind(query: str) -> _Kind:
    """What a label must agree with: "green cube" -> head cube, colour green;
    "pink object" -> no head, colour pink; "bag of chips" -> head bag,
    contents chips; "light blue mug" -> head mug, colours cyan or blue."""
    from .colors import _ARTICLES, GENERIC_NOUNS, canonical_color

    color = color_word = None
    nouns: list[str] = []
    before_of: list[str] = []
    after_of: list[str] = []
    seen_of = False
    for w in _object_part(_words(query)):
        if w == "of":
            seen_of = True
            continue
        if w in _ARTICLES:
            continue
        c = canonical_color(w)
        if c is not None and color is None:
            color, color_word = c, w
        elif w not in GENERIC_NOUNS:
            nouns.append(w)
            (after_of if seen_of else before_of).append(w)
    head = (before_of or nouns or [None])[-1]
    colours = frozenset({color} | _shade_bases(query)) if color is not None else frozenset()
    return _Kind(head, color, colours, frozenset(after_of),
                 color_word if head in _CATEGORY_NOUNS and color_word is not None else None,
                 frozenset(_shades(query)))


def _names_kind(name: str, kind: _Kind) -> bool:
    """Does a label name the queried kind? The query's head noun is one of its
    words; or its own head noun is a noun the query names after 'of' in the
    same form ("bag of chips" -> chips, never "can of tomatoes" -> tomato), or
    the colour word of a category query ("orange fruit" -> orange). A modifier
    word never names the object ("tomato can" is a can, not a tomato). And a
    colour it states is a queried one (a shade counts as its base colour).
    When both state a shade, it is the same one: "light blue mug" is never a
    lone "dark blue mug"."""
    words = _words(name)
    if kind.head is not None and not _shares(kind.head, words):
        own = _head_noun(words)
        if own is None or not (own in kind.of_nouns
                               or (kind.category_name is not None and _shares(own, [kind.category_name]))):
            return False
    own_shades = _shades(name) if kind.shades else set()
    if own_shades and not own_shades & kind.shades:
        return False
    stated = _stated_colours(name)
    return not kind.colours or not stated or bool(stated & kind.colours)


def _median_depth(depth: np.ndarray, u: int, v: int, r: int = PATCH_R) -> float | None:
    """Median of the valid depth samples in a small window around (u, v).

    Single-pixel reads are the classic way to get a 3D point that is metres
    off: sensor depth maps have holes and flying pixels, especially on object
    silhouettes (documented on the L515 in perception/grounding.py).
    """
    import os
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import median_depth
        return median_depth(depth, u, v, r)
    h, w = depth.shape[:2]
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def deproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Pixel + depth -> 3D point in the camera frame."""
    K = np.asarray(K, dtype=float)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


class PointProbe:
    """Queryable cursor over a camera frame.

    ``runtime`` is a ``SkillRuntime``; everything else (intrinsics, extrinsics,
    beliefs, workspace limits, TCP) is read from it at call time so the probe
    never holds stale calibration.
    """

    def __init__(self, runtime):
        self.rt = runtime

    # ── geometry ─────────────────────────────────────────────────────────

    def _frame_and_extrinsics(self, camera: str | None):
        rig = getattr(self.rt, "rig", None)
        if camera and rig is not None:
            stream = rig.get(camera)          # KeyError -> caller reports it
            frame = stream.latest()
        else:
            frame = self.rt.last_frame or self.rt.observe()
        if frame is None:
            raise ValueError("no frame available yet")
        T = frame.T_base_cam
        if T is None:
            extrinsics = camera_extrinsics(self.rt, camera)
            if extrinsics is None:
                raise ValueError(f"no extrinsics available for camera {camera!r}")
            T = extrinsics.cam_to_base()
        return frame, np.asarray(T, dtype=float)

    def _workspace(self) -> dict:
        cfg = self.rt.cfg
        ws_raw = cfg.safety.get("workspace", {}) or {}
        return dict(getattr(ws_raw, "_data", ws_raw))

    # ── the cursor ───────────────────────────────────────────────────────

    def probe(self, u, v, camera: str | None = None, normalized: bool = False) -> dict:
        """What is at this pixel? Returns metric facts, not an image.

        ``normalized=True`` accepts u, v in 0..1, which is what a model that
        reasoned over a resized frame will naturally produce.
        """
        frame, T = self._frame_and_extrinsics(camera)
        h, w = frame.rgb.shape[:2]
        u, v = float(u), float(v)
        # A model that reasoned over a resized frame naturally emits 0..1.
        # Auto-detect that rather than silently probing the top-left corner:
        # on a real frame, pixel (0.4, 0.7) is never a meaningful target.
        if normalized or (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and max(w, h) > 4):
            u, v = u * (w - 1), v * (h - 1)
        ui, vi = int(round(u)), int(round(v))
        out: dict = {
            "pixel": [ui, vi],
            "camera": camera or getattr(self.rt.camera, "name", "primary"),
            "image_size": [w, h],
        }
        if not (0 <= ui < w and 0 <= vi < h):
            out["ok"] = False
            out["error"] = f"pixel ({ui}, {vi}) is outside the {w}x{h} image"
            return out

        if not frame.has_depth or frame.depth_m is None:
            out["ok"] = False
            out["error"] = "this camera has no depth; cannot lift a pixel to 3D"
            out["depth_source"] = frame.depth_source
            return out

        z = _median_depth(np.asarray(frame.depth_m, dtype=np.float32), ui, vi)
        out["depth_source"] = frame.depth_source
        if z is None:
            out["ok"] = False
            out["error"] = (
                f"no valid depth within {PATCH_R} px of ({ui}, {vi}) -- "
                "a depth hole; probe a nearby pixel on the object body"
            )
            return out

        p_cam = deproject(ui, vi, z, frame.K)
        p_base = (T @ np.append(p_cam, 1.0))[:3]
        out["ok"] = True
        out["distance_m"] = round(float(z), 4)
        out["position"] = [round(float(x), 4) for x in p_base]
        # Say this out loud in the payload: a ray hits the FIRST surface, so
        # probing the middle of a cube returns its top face (verified: +22 mm
        # in z, +24 mm in x toward the camera, on a 4.5 cm cube). An agent
        # that feeds this straight into a grasp centre will grasp high.
        out["measures"] = "visible surface at this pixel, not the object centre"

        # which tracked object is here (the "query for the object" half)
        hit, hit_d = None, None
        try:
            for b in self.rt.beliefs.all():
                d = float(np.linalg.norm(np.asarray(b.position, dtype=float)[:3] - p_base))
                if hit_d is None or d < hit_d:
                    hit, hit_d = b, d
        except Exception:
            pass
        if hit is not None and hit_d is not None and hit_d <= HIT_RADIUS_M:
            out["object"] = {
                "label": str(getattr(hit, "label", "?")),
                "offset_m": round(hit_d, 4),
                "center": [round(float(x), 4) for x in np.asarray(hit.position)[:3]],
            }
        else:
            out["object"] = None
            if hit is not None and hit_d is not None:
                out["nearest_object"] = {
                    "label": str(getattr(hit, "label", "?")),
                    "offset_m": round(hit_d, 4),
                }

        # A workspace bound is a configured limit, not an IK solution. The
        # old default X band incorrectly rejected proven kitchen grasp poses.
        ws = self._workspace()
        try:
            lo, hi = ws["min"], ws["max"]
            inside = all(float(lo[i]) <= p_base[i] <= float(hi[i]) for i in range(3))
        except (KeyError, TypeError, IndexError):
            inside = None
        out["reachable"] = {
            "in_workspace": inside,
            "workspace_source": "configuration",
            "ik_checked": False,
            "grasp_checked": False,
            "note": (
                "Workspace membership does not establish reachability. "
                "No inverse-kinematics solution or grasp plan was checked."
            ),
        }
        if inside is False:
            out["reachable"]["note"] = (
                "This surface point is outside the configured TCP workspace; "
                "a direct TCP motion to it would be refused. "
                "No inverse-kinematics solution or grasp plan was checked."
            )

        # how far is the gripper from this point right now
        try:
            tcp = self.rt._tcp_if_awake()  # None in standby: a probe never powers the arm
            if tcp is None:
                return out
            tcp = np.asarray(tcp, dtype=float)[:3]
            delta = p_base - tcp
            out["from_gripper"] = {
                "distance_m": round(float(np.linalg.norm(delta)), 4),
                "delta_xyz_m": [round(float(x), 4) for x in delta],
            }
        except Exception:
            pass
        return out

    # ── the inverse: where is this object in the image? ───────────────────

    def locate_pixel(self, label: str, camera: str | None = None) -> dict:
        """Project a tracked object into image coordinates.

        Lets the agent move the cursor to something it knows by name, and
        reason about the scene in the same pixel space it is looking at.
        """
        frame, T = self._frame_and_extrinsics(camera)
        h, w = frame.rgb.shape[:2]
        want = _content(_words(label))
        kind = _queried_kind(str(label))
        best, score = None, 0
        near: list[tuple[int, str]] = []  # labels sharing a word but naming another kind
        try:
            for b in self.rt.beliefs.all():
                names = [str(getattr(b, "label", ""))]
                names += sorted(str(a) for a in (getattr(b, "aliases", None) or ()))
                overlap = 0
                for name in names:
                    have = _content(_words(name))
                    s = sum(1 for w in set(want) if _shares(w, have))
                    overlap = max(overlap, s)
                    # r3 (kitchen run): "green cube" answered ok as the green
                    # translucent square on the colour word alone, and the
                    # agent grasped there. A match must name the queried kind.
                    if s > score and _names_kind(name, kind):
                        best, score = b, s
                if overlap:
                    near.append((overlap, names[0]))
        except Exception:
            pass
        if best is None:
            out = {"ok": False, "error": f"{label!r} is not in the world model",
                   "hint": "call get_observation or localize_object first"}
            if near:
                candidates = list(dict.fromkeys(n for _, n in sorted(near, key=lambda x: (-x[0], x[1]))))
                asked = " ".join(x for x in (kind.color, kind.head) if x)
                out.update(
                    error=f"{label!r} is not in the world model: no tracked object is a {asked!r}",
                    candidates=candidates,
                    hint=("these tracked labels share a word but name another object; use one only if "
                          "it IS the requested object, else call localize_object or get_observation"))
            return out

        p_base = np.asarray(best.position, dtype=float)[:3]
        p_cam = (np.linalg.inv(T) @ np.append(p_base, 1.0))[:3]
        if p_cam[2] <= 1e-6:
            return {"ok": False, "error": f"{best.label!r} is behind the camera"}
        K = np.asarray(frame.K, dtype=float)
        uv = K @ (p_cam / p_cam[2])
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        return {
            "ok": True,
            "label": str(best.label),
            "pixel": [u, v],
            "image_size": [w, h],
            "normalized": [round(u / max(w - 1, 1), 4), round(v / max(h - 1, 1), 4)],
            "position": [round(float(x), 4) for x in p_base],
            "distance_m": round(float(p_cam[2]), 4),
            "in_view": bool(0 <= u < w and 0 <= v < h),
        }


def draw_cursor(img: np.ndarray, u: int, v: int, text: str = "") -> np.ndarray:
    """Draw the red X the ablation used, for the human-facing views.

    The mark is for the person watching the dashboard; the *agent* consumes
    the numbers from ``PointProbe.probe``. Keeping those two channels separate
    is the point -- an overlay is neutral, a queryable number is not.
    """
    import cv2

    out = img if img.flags.writeable else img.copy()
    cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.circle(out, (int(u), int(v)), 14, (0, 0, 255), 1, cv2.LINE_AA)
    if text:
        cv2.putText(out, text, (int(u) + 18, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (int(u) + 18, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 60, 255), 1, cv2.LINE_AA)
    return out
