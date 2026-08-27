"""Language -> 3D grounding (ASPIRE-style localize_object).

Pipeline: text prompt(s) -> open-vocab detection -> mask depth sampling ->
backprojection to camera-frame points -> base frame via extrinsics -> OBB
center (more robust than the mean for elongated/partially occluded objects).

Extrinsics supports both mounting styles:
- eye_to_hand: static T_cam2base (camera on a tripod/frame looking at the arm)
- eye_in_hand: T_cam2gripper composed with FK at capture time
It can read the baseline repo's hand_eye.npz files (key T_result) unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..types import Detection, Frame, ObjectFix, SkillError, transform_points
from .reference import Reference, apply_reference, parse_reference


class Extrinsics:
    def __init__(
        self,
        mode: str = "eye_to_hand",
        T: np.ndarray | None = None,
        fk_tcp2base: Callable[[], np.ndarray] | None = None,
    ):
        """`T` is T_cam2base (eye_to_hand) or T_cam2gripper (eye_in_hand)."""
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"bad extrinsics mode {mode!r}")
        self.mode = mode
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float)
        self._fk = fk_tcp2base

    @classmethod
    def from_config(cls, cfg, fk_tcp2base=None) -> "Extrinsics":
        mode = cfg.get("mode", "eye_to_hand")
        npz_path = cfg.get("hand_eye_npz")
        if npz_path and Path(npz_path).exists():
            data = np.load(npz_path, allow_pickle=True)
            T = np.asarray(data["T_result"], dtype=float)
            if "mode" in data:
                # The baseline saves mode as a 1-element string array.
                saved_mode = str(np.asarray(data["mode"]).ravel()[0])
            else:
                saved_mode = mode
            return cls(mode=saved_mode, T=T, fk_tcp2base=fk_tcp2base)
        mat = cfg.get("T")
        T = np.asarray(mat, dtype=float).reshape(4, 4) if mat is not None else None
        return cls(mode=mode, T=T, fk_tcp2base=fk_tcp2base)

    def cam_to_base(self) -> np.ndarray:
        """T_cam2base at this instant (uses live FK for eye-in-hand)."""
        if self.mode == "eye_to_hand":
            return self.T
        if self._fk is None:
            raise SkillError("eye_in_hand extrinsics need a FK callback")
        return self._fk() @ self.T


def mask_to_points_cam(
    frame: Frame,
    mask: np.ndarray,
    max_points: int = 4000,
    depth_band: tuple[float, float] = (0.05, 0.95),
) -> np.ndarray:
    """Backproject mask pixels with valid depth into camera-frame points.

    The inter-quantile depth band drops mixed/flying pixels at mask edges
    (a real problem on the L515 around object silhouettes).
    """
    if not frame.has_depth:
        raise SkillError("no depth available; cannot lift mask to 3D")
    ys, xs = np.nonzero(mask)
    zs = frame.depth_m[ys, xs]
    good = zs > 0
    ys, xs, zs = ys[good], xs[good], zs[good]
    if zs.size == 0:
        return np.empty((0, 3))
    lo, hi = np.quantile(zs, depth_band)
    keep = (zs >= lo) & (zs <= hi)
    ys, xs, zs = ys[keep], xs[keep], zs[keep]
    if zs.size > max_points:
        idx = np.random.default_rng(0).choice(zs.size, max_points, replace=False)
        ys, xs, zs = ys[idx], xs[idx], zs[idx]
    K = frame.K
    pts = np.stack(
        [(xs - K[0, 2]) / K[0, 0] * zs, (ys - K[1, 2]) / K[1, 1] * zs, zs], axis=-1
    )
    return pts


def oriented_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA oriented bounding box -> (center, extents desc-sorted, axes cols)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    proj = centered @ axes
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    center = centroid + axes @ ((mins + maxs) / 2)
    extents = maxs - mins
    return center, extents, axes


def _recentre_by_size(center, pts_base, extents, cam_pos):
    """Re-centre a fitted box using the object's own size along the view ray.

    The centre of an oriented box fitted to a depth cloud is biased TOWARD the
    camera, measured here at 18.3 mm for a 5 cm cube. It is not a one-face
    shell -- the cloud genuinely wraps the object (59.5 mm of depth span) --
    it is density: near faces subtend more pixels per unit area than far ones,
    so the fit is dragged forward. Two smaller effects push the same way: the
    detection mask is inflated downward by the object's shadow (+8.1 px, i.e.
    0.69 cm at 0.93 m) and the resulting skirt of table points sits nearer the
    camera still.

    Left uncorrected this is a 1.6-1.9 cm lateral error on the grasp target.
    Against a 2.5 cm half-width the finger catches the edge, shoves the object
    away and the jaws close on nothing -- the "air grasp" failure.

    Rather than trusting the fitted centre, anchor on the NEAR surface (which
    depth measures well) and step half the object's own measured size along
    the view ray. Using the fitted extent rather than a hard-coded size keeps
    this honest for objects that are not 5 cm cubes; the extent is inflated
    laterally by the mask, so take the smallest axis, which is the one least
    corrupted by shadow and table bleed.

    Measured on five HELD-OUT positions (not used to design it):

        baseline   mean 1.85 cm   max 2.61 cm
        this fix   mean 0.56 cm   max 1.09 cm

    A constant offset fitted to the earlier positions scored a better mean
    (0.38 cm) but a WORSE maximum (1.32 cm) and degraded at the far corner,
    which is what a hard-coded constant does off its fitting set. Worst case
    is what decides whether a grasp lands, so the principled correction wins.
    """
    pts = np.asarray(pts_base, dtype=float)
    if pts.shape[0] < 20:
        return center
    cam = np.asarray(cam_pos, dtype=float)[:3]
    rng = np.linalg.norm(pts - cam, axis=1)
    near = pts[rng <= np.percentile(rng, 10)].mean(axis=0)
    d = near - cam
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return center
    d /= n
    size = float(np.min(np.asarray(extents, dtype=float)))
    size = float(np.clip(size, 0.01, 0.30))
    return near + d * (size / 2.0)


def _bbox_mask(frame: Frame, det: Detection) -> np.ndarray:
    h, w = frame.rgb.shape[:2]
    m = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = det.bbox.astype(int)
    m[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)] = True
    return m


# Base frame: +x points away from the robot, +y points left (matches
# skill_push_object's direction map). Each hint -> (axis, sort_descending):
# "left" = max y, "right" = min y, "front"/"near" = min x, "back"/"far" = max x.
_SPATIAL_AXES = {
    "left": (1, True),
    "right": (1, False),
    "front": (0, False),
    "near": (0, False),
    "back": (0, True),
    "far": (0, True),
}


def localize_object(
    frame: Frame,
    label: str,
    detector,
    extrinsics: Extrinsics,
    prompts: list[str] | None = None,
    min_points: int = 10,
    spatial_hint: str | None = None,
    color: str | None = None,
    near_xyz: np.ndarray | None = None,
    vocab: list[str] | None = None,
    prefer_label: str | None = None,
) -> ObjectFix:
    """Find `label` in the frame and return its base-frame 3D fix.

    Tries an ordered prompt fallback list -- or, with `vocab`, ONE detector
    pass over a whole vocabulary where every detection is a candidate (how
    color queries like "pink object" work on closed-set detectors).
    Candidates are then narrowed by mask color (`color`), a spatial hint word
    ("left", "front", ...), and/or proximity to a remembered position
    (`near_xyz`). Raises SkillError with an agent-readable reason on failure.
    """
    from .colors import color_matches, detection_color

    T_cam2base = extrinsics.cam_to_base()
    if vocab:
        rounds = [(vocab, "vocabulary")]
    else:
        rounds = [([p], p) for p in (prompts or [label])]

    last_reason = f"no detections for {vocab or prompts or [label]}"
    for classes, what in rounds:
        dets = [d for d in detector.detect(frame, classes=classes) if d.conf > 0]
        if not dets:
            continue
        exact: list[ObjectFix] = []   # color matches the palette band
        loose: list[ObjectFix] = []   # neighbor band / unknown color
        for det in sorted(dets, key=lambda d: -d.conf):
            det_color = None
            if color is not None:
                # detection_color uses the mask when present, else the bbox
                # CENTER (whole-box medians let the background veto objects).
                # STRICT here: grabbing the red box when asked for the pink
                # one is a visible error; neighbor tolerance is for verbal
                # recall (beliefs.find), not for choosing a grasp target.
                det_color = detection_color(frame.rgb, det)
                if not color_matches(color, det_color, strict=True):
                    last_reason = f"saw {det.label!r} but it is {det_color}, not {color}"
                    continue
            mask = det.mask if det.mask is not None else _bbox_mask(frame, det)
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < min_points:
                last_reason = (
                    f"detected {what!r} but only {pts_cam.shape[0]} valid depth "
                    f"points (<{min_points}); object may be out of depth range"
                )
                continue
            pts_base = transform_points(T_cam2base, pts_cam)
            center, extents, axes = oriented_bbox(pts_base)
            center = _recentre_by_size(center, pts_base, extents,
                                       T_cam2base[:3, 3])
            fix = ObjectFix(
                label=label,
                position=center,
                points=pts_base,
                detection=det,
                extent=extents,
                axes=axes,
            )
            if color is None or det_color == color:
                exact.append(fix)
            else:
                loose.append(fix)
        candidates = exact or loose
        if not candidates:
            continue
        # Referring expressions ("the second cup from the left", "the biggest
        # block", "not the red one"). VoLo makes complex references one of its
        # four capability suites and ASPIRE ships the same ordering rule as a
        # learned skill. An explicit `spatial_hint` from the caller overrides
        # any axis word in the phrase, because the caller knows more.
        ref = parse_reference(label)
        if spatial_hint:
            ref = Reference(noun=ref.noun, spatial=spatial_hint,
                            ordinal=ref.ordinal, size=ref.size,
                            exclude=ref.exclude)
        if not ref.is_plain and len(candidates) > 1:
            candidates = apply_reference(candidates, ref, _SPATIAL_AXES)
        elif near_xyz is not None and len(candidates) > 1:
            anchor = np.asarray(near_xyz, dtype=float).reshape(3)
            candidates.sort(key=lambda f: float(np.linalg.norm(f.position - anchor)))
        if prefer_label and len(candidates) > 1:
            # Stable partition: same-class detections first, prior ordering
            # (hint/proximity/confidence) preserved within each group.
            candidates.sort(key=lambda f: f.detection.label != prefer_label)
        return candidates[0]
    raise SkillError(f"localize {label!r} failed: {last_reason}")
