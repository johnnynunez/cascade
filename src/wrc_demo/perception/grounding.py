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
) -> ObjectFix:
    """Find `label` in the frame and return its base-frame 3D fix.

    Tries an ordered prompt fallback list; disambiguates duplicate instances
    with a spatial hint word if given ("left", "front", ...). Raises
    SkillError with an agent-readable reason on failure.
    """
    prompt_list = prompts or [label]
    T_cam2base = extrinsics.cam_to_base()

    last_reason = f"no detections for any of {prompt_list}"
    for prompt in prompt_list:
        dets = detector.detect(frame, classes=[prompt])
        dets = [d for d in dets if d.conf > 0]
        if not dets:
            continue
        candidates: list[ObjectFix] = []
        for det in sorted(dets, key=lambda d: -d.conf):
            mask = det.mask if det.mask is not None else _bbox_mask(frame, det)
            pts_cam = mask_to_points_cam(frame, mask)
            if pts_cam.shape[0] < min_points:
                last_reason = (
                    f"detected {prompt!r} but only {pts_cam.shape[0]} valid depth "
                    f"points (<{min_points}); object may be out of depth range"
                )
                continue
            pts_base = transform_points(T_cam2base, pts_cam)
            center, extents, axes = oriented_bbox(pts_base)
            candidates.append(
                ObjectFix(
                    label=label,
                    position=center,
                    points=pts_base,
                    detection=det,
                    extent=extents,
                    axes=axes,
                )
            )
        if not candidates:
            continue
        if spatial_hint and spatial_hint in _SPATIAL_AXES and len(candidates) > 1:
            axis, descending = _SPATIAL_AXES[spatial_hint]
            candidates.sort(key=lambda f: f.position[axis], reverse=descending)
        return candidates[0]
    raise SkillError(f"localize {label!r} failed: {last_reason}")
