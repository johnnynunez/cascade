"""Parse referring expressions into perception constraints.

VoLo's RoboVoLo benchmark makes "complex references" one of its four capability
suites: spatial, ordinal, size and negation cues. ASPIRE ships the same idea as
a learned skill, quoted from its skill library:

    Detect all instances, sort by the axis implied by the qualifier (X for
    front/back, Y for left/right in robot frame), then select by keyword index.

wrc_demo already had the axis map (`grounding._SPATIAL_AXES`) and honoured a
`spatial_hint`, but nothing extracted that hint from what the user actually
typed: the LLM had to pass it as a separate argument, and a booth visitor
saying "grab the second cup from the left" got no disambiguation at all.

This module closes that gap on the way in. It is deliberately small and
rule-based rather than a model call: it runs on every localization, so it has
to be free, and a regex that fails simply yields no constraint and the old
behaviour resumes.

Everything here is vocabulary-independent. It never names an object class, only
the ways a person can point at one, so it stays valid for objects nobody has
enumerated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Ordinal words -> zero-based index into the sorted candidate list.
_ORDINALS = {
    "first": 0, "1st": 0, "leftmost": 0, "nearest": 0, "closest": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "last": -1, "furthest": -1, "farthest": -1, "rightmost": -1,
}

#: Size words -> (sort by max extent, descending?)
_SIZES = {
    "biggest": True, "largest": True, "big": True, "large": True,
    "tallest": True,
    "smallest": False, "tiniest": False, "small": False, "tiny": False,
}

#: Spatial qualifiers, kept in sync with grounding._SPATIAL_AXES.
_SPATIAL_WORDS = ("left", "right", "front", "near", "back", "far")

_NEGATION = re.compile(
    r"\b(?:not|other than|except|besides|instead of)\s+(?:the\s+)?([a-z][a-z ]{0,20}?)\b",
    re.I,
)


@dataclass(frozen=True)
class Reference:
    """Constraints extracted from a referring expression.

    Attributes:
        noun: the query with reference words stripped, i.e. what to detect.
        spatial: a key of `grounding._SPATIAL_AXES`, or None.
        ordinal: zero-based index (negative counts from the end), or None.
        size: True for biggest-first, False for smallest-first, None if unset.
        exclude: a label fragment the chosen candidate must NOT match.
    """

    noun: str
    spatial: str | None = None
    ordinal: int | None = None
    size: bool | None = None
    exclude: str | None = None

    @property
    def is_plain(self) -> bool:
        """True when no constraint was found, so callers can skip the work."""
        return (
            self.spatial is None
            and self.ordinal is None
            and self.size is None
            and self.exclude is None
        )


def parse_reference(query: str) -> Reference:
    """Extract reference constraints from a natural phrase.

    Examples:
        "the second cup from the left" -> ordinal=1, spatial='left', noun='cup'
        "the biggest block"            -> size=True, noun='block'
        "the cup, not the red one"     -> exclude='red one', noun='cup'
        "cup"                          -> plain
    """
    text = (query or "").strip().lower()
    if not text:
        return Reference(noun="")

    exclude = None
    m = _NEGATION.search(text)
    if m:
        exclude = m.group(1).strip() or None
        text = text[: m.start()] + " " + text[m.end():]

    words = re.findall(r"[a-z0-9]+", text)
    kept: list[str] = []
    spatial = ordinal = size = None

    for w in words:
        if w in _ORDINALS and ordinal is None:
            ordinal = _ORDINALS[w]
            # Superlatives name an axis as well as an index. They pair a
            # sort direction with index 0 or -1 (set above), so "rightmost"
            # is "sort left-first, take last", not "sort right-first".
            if w == "leftmost":
                spatial = spatial or "left"
            elif w == "rightmost":
                spatial = spatial or "left"    # ordinal is -1: last = rightmost
            elif w in ("nearest", "closest"):
                spatial = spatial or "near"
            elif w in ("furthest", "farthest"):
                spatial = spatial or "near"    # ordinal is -1: last = furthest
            continue
        if w in _SIZES and size is None:
            size = _SIZES[w]
            continue
        if w in _SPATIAL_WORDS and spatial is None:
            spatial = w
            continue
        if w in ("the", "a", "an", "from", "of", "one", "please", "that", "is"):
            continue
        kept.append(w)

    return Reference(
        noun=" ".join(kept).strip(),
        spatial=spatial,
        ordinal=ordinal,
        size=size,
        exclude=exclude,
    )


def apply_reference(candidates: list, ref: Reference, axes: dict) -> list:
    """Order and filter candidates by the parsed constraints.

    `candidates` are objects with `.position` (3,) and optionally `.extent`
    and `.detection.label`. `axes` is `grounding._SPATIAL_AXES`.

    Ordering is applied before selection so an ordinal counts along the axis
    the phrase named, which is what "second from the left" means. Returns the
    reordered list; the caller takes element 0 as before.
    """
    if not candidates or ref.is_plain:
        return candidates

    out = list(candidates)

    if ref.exclude:
        kept = [
            c for c in out
            if ref.exclude not in str(getattr(getattr(c, "detection", None),
                                              "label", "")).lower()
        ]
        # Never let a negation empty the list: an unsatisfiable exclusion
        # should degrade to "no preference", not to "object not found".
        if kept:
            out = kept

    if ref.size is not None:
        def _bulk(c):
            e = getattr(c, "extent", None)
            return float(max(e)) if e is not None else 0.0
        out.sort(key=_bulk, reverse=ref.size)
    elif ref.spatial and ref.spatial in axes:
        axis, descending = axes[ref.spatial]
        out.sort(key=lambda c: float(c.position[axis]), reverse=descending)

    if ref.ordinal is not None and out:
        idx = ref.ordinal if ref.ordinal >= 0 else len(out) + ref.ordinal
        idx = max(0, min(idx, len(out) - 1))
        out = [out[idx]] + out[:idx] + out[idx + 1:]

    return out
