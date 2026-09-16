"""Visual differencing: an independent verification channel for real hardware.

CaP-X (arXiv:2603.22435) benchmarks code-as-policy agents and finds that
scaling *agentic test-time computation* -- multi-turn interaction, structured
execution feedback, **visual differencing**, skill synthesis, ensembled
reasoning -- recovers most of the reliability lost when human-crafted
abstractions are removed. Visual differencing is the cheapest of those and the
one this repo most needs.

Why it is needed here, concretely. `PostconditionChecker` has two channels:

    physics   TruthPoseReader -> PhysX. Independent, exact -- and SIM ONLY.
    belief    the world model. On the real rig this is the ONLY channel, and
              it is written by the same perception pass the skill just ran,
              so agreeing with it proves nothing. That tautology was measured
              on 2026-07-31: a placement reported "0.0 cm from the requested
              drop point (belief)" while the cube sat 30 cm away.

So on the B601-RS with an L515 there is currently *no* independent evidence
that a motion did anything. This module supplies one: compare the pixels
before and after the motion. The camera is not the actuator, so what it sees
is not a replay of the robot's own intention.

The measurement is deliberately coarse. It answers "did this region of the
world change?", not "where is the object now" -- a binary that survives
lighting noise, mild camera shake and depth holes. Anything finer would be
re-implementing perception, and perception is what we are trying to check.

THE PITFALL THAT MAKES THIS HARD: the arm is in frame. A gripper parked over
the workspace changes those pixels regardless of whether the object moved, so
a naive diff confirms everything. Two defences:

  1. The caller should sample the "after" frame with the arm retreated
     (`retreat_first=True` on the runtime hook).
  2. If the region cannot be trusted -- too much of the frame changed, or the
     ROI is out of view -- the channel returns UNKNOWN rather than guessing.
     An honest abstain is the whole point; a verifier that always answers is
     the bug we are fixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Per-pixel intensity delta (0-255) above which a pixel counts as changed.
#: 25 is well above sensor noise on both the L515 and Isaac's RTX cameras and
#: well below a real object appearing or leaving.
PIXEL_DELTA = 25

#: Fraction of an ROI that must change before we call it "changed".
ROI_CHANGE_FRAC = 0.12

#: If more than this fraction of the WHOLE frame changed, something global
#: happened (arm swept through, lighting shift, camera moved) and no ROI
#: verdict is trustworthy.
GLOBAL_CHANGE_ABORT = 0.45

#: Half-width (px) of the square ROI sampled around a projected point.
ROI_HALF_PX = 45

UNKNOWN, CHANGED, UNCHANGED = "unknown", "changed", "unchanged"


@dataclass
class DiffVerdict:
    """What the pixels say happened, plus the numbers behind it."""

    status: str = UNKNOWN
    source_roi: str = UNKNOWN       # where the object was
    target_roi: str = UNKNOWN       # where it was supposed to go
    global_change: float = 0.0
    detail: str = ""
    measured: dict | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "source_roi": self.source_roi,
            "target_roi": self.target_roi,
            "global_change": round(self.global_change, 4),
            "detail": self.detail,
            "measured": self.measured or {},
        }


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.int16)
    return img[..., :3].mean(axis=2).astype(np.int16)


def _changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels whose intensity moved more than PIXEL_DELTA."""
    import os
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import changed_fraction
        return changed_fraction(a, b, threshold=PIXEL_DELTA)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float((np.abs(_gray(a) - _gray(b)) > PIXEL_DELTA).mean())


def _roi(img: np.ndarray, uv, half: int = ROI_HALF_PX):
    """Square window around a pixel, clipped to the frame. None if outside."""
    if uv is None:
        return None
    h, w = img.shape[:2]
    u, v = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= u < w and 0 <= v < h):
        return None
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    if u1 - u0 < 8 or v1 - v0 < 8:
        return None
    return (v0, v1, u0, u1)


class VisualDiffChannel:
    """Before/after pixel comparison as a third verification channel.

    ``project`` maps a base-frame 3D point to a pixel in the frames handed in
    (the repo already has this: ``PointProbe.locate_pixel`` / the extrinsics
    on the camera profile).
    """

    def __init__(self, project=None):
        self._project = project

    # ── the whole-frame guard ────────────────────────────────────────────

    def _global(self, before: np.ndarray, after: np.ndarray) -> float:
        return _changed_fraction(before, after)

    # ── the verdict ──────────────────────────────────────────────────────

    def compare(
        self,
        before: np.ndarray | None,
        after: np.ndarray | None,
        source_xyz=None,
        target_xyz=None,
    ) -> DiffVerdict:
        """Did the object leave `source_xyz`, and did something arrive at `target_xyz`?

        Both 3D points are optional: with neither, this degrades to a global
        "did anything change at all" check, which is weak but still not a
        tautology.
        """
        v = DiffVerdict()
        if before is None or after is None:
            v.detail = "no before/after frame pair captured"
            return v
        if before.shape != after.shape:
            v.detail = f"frame size changed {before.shape} -> {after.shape}"
            return v

        v.global_change = self._global(before, after)
        if v.global_change > GLOBAL_CHANGE_ABORT:
            v.detail = (
                f"{v.global_change:.0%} of the frame changed -- the arm is probably "
                "still in view, so no region verdict is trustworthy"
            )
            return v

        measured: dict = {"global_change": round(v.global_change, 4)}

        def _region_verdict(xyz, name) -> str:
            if xyz is None or self._project is None:
                return UNKNOWN
            uv = self._project(xyz)
            box = _roi(after, uv)
            if box is None:
                measured[f"{name}_roi"] = "out of view"
                return UNKNOWN
            v0, v1, u0, u1 = box
            frac = _changed_fraction(before[v0:v1, u0:u1], after[v0:v1, u0:u1])
            measured[f"{name}_change"] = round(frac, 4)
            measured[f"{name}_px"] = [int(uv[0]), int(uv[1])]
            return CHANGED if frac >= ROI_CHANGE_FRAC else UNCHANGED

        v.source_roi = _region_verdict(source_xyz, "source")
        v.target_roi = _region_verdict(target_xyz, "target")
        v.measured = measured

        # Interpret. The strong signal is "left the source AND arrived at the
        # target"; either alone is weaker but still real evidence.
        if v.source_roi == CHANGED and v.target_roi == CHANGED:
            v.status = CHANGED
            v.detail = "pixels changed both where the object was and where it was sent"
        elif v.source_roi == UNCHANGED and v.target_roi == UNCHANGED:
            v.status = UNCHANGED
            v.detail = "nothing changed at either end -- the scene looks untouched"
        elif v.source_roi == CHANGED:
            v.status = CHANGED
            v.detail = "the object's original position changed (destination not checked)"
        elif v.target_roi == CHANGED:
            v.status = CHANGED
            v.detail = "the destination changed (origin not checked)"
        elif v.source_roi == UNCHANGED and v.target_roi == UNKNOWN:
            v.status = UNCHANGED
            v.detail = "the object appears to still be where it started"
        else:
            v.status = UNKNOWN
            v.detail = "not enough of the scene was observable to judge"
        return v


def frames_differ_globally(before, after) -> float:
    """Convenience: fraction of the frame that changed. Used by tests/tools."""
    if before is None or after is None:
        return 0.0
    return _changed_fraction(before, after)
