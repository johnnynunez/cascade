"""Language -> 3D grounding (ASPIRE-style localize_object).

Pipeline: text prompt(s) -> open-vocab detection -> mask depth sampling ->
backprojection to camera-frame points -> base frame via extrinsics -> OBB
center (more robust than the mean for elongated/partially occluded objects).

Extrinsics supports both mounting styles:
- eye_to_hand: static T_cam2base (camera on a tripod/frame looking at the arm)
- eye_in_hand: T_cam2gripper composed with FK at capture time
It can read the baseline repo's hand_eye.npz files (key T_result) unchanged,
and the gated schema-v1 records scripts/calibrate_handeye.py writes
(`extrinsics.hand_eye_json`, see docs/HANDEYE_CALIBRATION.md).
"""

from __future__ import annotations

from pathlib import Path
import os
import threading
from typing import Callable

import numpy as np

from ..types import Detection, Frame, ObjectFix, SkillError, transform_points
from .reference import Reference, ReferenceResolutionError, apply_reference, parse_reference


class Extrinsics:
    def __init__(
        self,
        mode: str = "eye_to_hand",
        T: np.ndarray | None = None,
        fk_tcp2base: Callable[[], np.ndarray] | None = None,
        *,
        compensation_m=None,
        source: str = "inline",
        calibration_error: str | None = None,
    ):
        """`T` is T_cam2base (eye_to_hand) or T_cam2gripper (eye_in_hand).

        `compensation_m` is the profile's per-camera `hand_eye_compensation_m`
        (ADR-0009, ported from WRC): a base-frame translation applied on the
        LEFT -- eye_to_hand: T_comp @ T; eye_in_hand: T_comp @ FK @ T. For
        eye_to_hand it is baked into `.T`, so every reader of the static
        matrix (e.g. grasp-evidence audits) sees the transform actually used.

        `calibration_error` marks a camera whose configured calibration
        cannot be trusted (missing / malformed / rejected record, wrong
        serial): it must not fuse, and `cam_to_base()` refuses with the
        reason instead of returning a confident wrong transform.
        """
        if mode not in ("eye_to_hand", "eye_in_hand"):
            raise ValueError(f"bad extrinsics mode {mode!r}")
        # Guards the (calibration_error, T) pair against a runtime
        # invalidate()/adopt() from the extrinsic drift monitor's thread.
        self._lock = threading.Lock()
        self.mode = mode
        self.T_compensation = _compensation_transform(compensation_m)
        self.T_hand_eye = np.eye(4) if T is None else np.asarray(T, dtype=float)
        self.calibration_error = calibration_error
        self.source = source
        if calibration_error is not None:
            self.T = np.full((4, 4), np.nan)
        elif mode == "eye_to_hand":
            self.T = self.T_compensation @ self.T_hand_eye
        else:
            self.T = self.T_hand_eye
        self._fk = fk_tcp2base

    @property
    def calibrated(self) -> bool:
        return self.calibration_error is None

    def invalidate(self, reason: str) -> None:
        """Mark this camera UNCALIBRATED at runtime (the extrinsic drift
        monitor saw it move): same semantics as a rejected record --
        `cam_to_base()` refuses with `reason`, `.T` reads NaN. Every holder
        of this object (WatchedCamera, SkillRuntime) sees it at once."""
        with self._lock:
            self.calibration_error = str(reason)
            self.T = np.full((4, 4), np.nan)

    def adopt(self, T_cam2base, source: str) -> None:
        """Trust a new eye-to-hand transform (a gated re-calibration). It is
        the full T_cam2base as measured, so the profile's compensation is
        folded into T_hand_eye rather than applied a second time."""
        if self.mode != "eye_to_hand":
            raise ValueError("adopt() is eye_to_hand only")
        T = np.asarray(T_cam2base, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(T)):
            raise ValueError("refusing to adopt a non-finite transform")
        with self._lock:
            self.T_hand_eye = np.linalg.inv(self.T_compensation) @ T
            self.T = T.copy()
            self.source = source
            self.calibration_error = None

    @classmethod
    def from_config(cls, cfg, fk_tcp2base=None, *, camera_serial=None) -> "Extrinsics":
        """Build from a camera profile's `extrinsics:` block.

        Sources, first match wins: `hand_eye_json` (schema-v1 record from
        scripts/calibrate_handeye.py, gated), `hand_eye_npz` (the baseline
        repo's file, key T_result), inline `T`. `camera_serial` is the
        profile's pinned serial; a record made for another unit is refused.
        """
        mode = cfg.get("mode", "eye_to_hand")
        comp = cfg.get("hand_eye_compensation_m")
        json_path = cfg.get("hand_eye_json")
        if json_path:
            return cls._from_hand_eye_json(Path(str(json_path)).expanduser(), cfg.get("mode"),
                                           comp, fk_tcp2base, camera_serial)
        npz_path = cfg.get("hand_eye_npz")
        if npz_path and Path(npz_path).exists():
            data = np.load(npz_path, allow_pickle=True)
            T = np.asarray(data["T_result"], dtype=float)
            if "mode" in data:
                # The baseline saves mode as a 1-element string array.
                saved_mode = str(np.asarray(data["mode"]).ravel()[0])
            else:
                saved_mode = mode
            return cls(mode=saved_mode, T=T, fk_tcp2base=fk_tcp2base, compensation_m=comp,
                       source=f"npz:{npz_path}")
        mat = cfg.get("T")
        T = np.asarray(mat, dtype=float).reshape(4, 4) if mat is not None else None
        return cls(mode=mode, T=T, fk_tcp2base=fk_tcp2base, compensation_m=comp)

    @classmethod
    def _from_hand_eye_json(cls, path, profile_mode, comp, fk, camera_serial) -> "Extrinsics":
        from ..calibration.dataset import read_hand_eye

        def refused(reason, mode=None):
            msg = (f"hand-eye calibration {path} not usable: {reason}; recalibrate "
                   "(docs/HANDEYE_CALIBRATION.md)")
            return cls(mode=mode or profile_mode or "eye_to_hand", fk_tcp2base=fk,
                       compensation_m=comp, source=f"hand_eye_json:{path}",
                       calibration_error=msg)

        try:
            record = read_hand_eye(path)
        except FileNotFoundError:
            return refused("file not found")
        except ValueError as e:
            return refused(f"malformed hand-eye record ({e})")
        if profile_mode and profile_mode != record.mode:
            # An authoring error, not a calibration result: the profile and
            # the record disagree about where the camera is mounted.
            raise ValueError(f"extrinsics mode {profile_mode!r} contradicts the {record.mode!r} "
                             f"hand-eye record {path}")
        if not record.acceptable:
            return refused("; ".join(record.rejection_reasons), record.mode)
        if camera_serial and record.camera_serial and str(camera_serial) != record.camera_serial:
            return refused(f"recorded for camera serial {record.camera_serial}, but the "
                           f"profile pins {camera_serial}", record.mode)
        return cls(mode=record.mode, T=record.T_hand_eye, fk_tcp2base=fk, compensation_m=comp,
                   source=f"hand_eye_json:{path}")

    def cam_to_base(self) -> np.ndarray:
        """T_cam2base at this instant (uses live FK for eye-in-hand)."""
        with self._lock:
            error, T = self.calibration_error, self.T
        if error is not None:
            raise SkillError(error)
        if self.mode == "eye_to_hand":
            return T
        if self._fk is None:
            raise SkillError("eye_in_hand extrinsics need a FK callback")
        return self.T_compensation @ self._fk() @ self.T


def _compensation_transform(comp) -> np.ndarray:
    """ADR-0009 `{x, y, z}` (metres, base frame) -> 4x4; None = identity."""
    T = np.eye(4)
    if comp is None:
        return T
    getter = comp.get if hasattr(comp, "get") else None
    keys = set(comp.as_dict()) if hasattr(comp, "as_dict") else (
        set(comp) if isinstance(comp, dict) else None)
    if getter is None or keys is None or not keys <= {"x", "y", "z"}:
        raise ValueError(f"hand_eye_compensation_m must be a mapping of x/y/z metres, got {comp!r}")
    try:
        vals = [float(getter(k, 0.0)) for k in ("x", "y", "z")]
    except (TypeError, ValueError) as e:
        raise ValueError(f"hand_eye_compensation_m values must be numbers: {comp!r}") from e
    if not all(np.isfinite(vals)):
        raise ValueError(f"hand_eye_compensation_m must be finite: {comp!r}")
    T[:3, 3] = vals
    return T


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
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import backproject
        return backproject(frame, mask, max_points=max_points, depth_band=depth_band)
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
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import oriented_bbox as cuda_bbox
        return cuda_bbox(points)
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
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import recentre
        return recentre(center, pts_base, extents, cam_pos)
    pts = np.asarray(pts_base, dtype=float)
    if pts.shape[0] < 20:
        return center
    cam = np.asarray(cam_pos, dtype=float)[:3]
    # Decompose along the view ray (camera -> cloud centroid). The camera
    # bias this function exists to remove lives ALONG the ray (near surfaces
    # oversampled -> the fit dragged toward the camera), so only the along-ray
    # coordinate is re-anchored on the near surface; the LATERAL coordinates
    # come from the cloud centroid, which a flat face samples uniformly.
    #
    # The original version anchored all three axes on the nearest-RANGE
    # decile. Range varies with lateral offset too, and on a flat top face
    # seen near-vertically that is all it varies by -- so the anchor picked
    # the face's near EDGE, not the near face: measured on a 35 mm box 87 px
    # off the principal point (mock_small + so101_mujoco), the fix landed
    # 14 mm lateral of the object, the finger caught the edge and shoved it,
    # and every grasp failed as "air grasp". The reBot/Isaac rigs never saw
    # this because their objects sat near the image centre, where range and
    # along-ray agree. tests/test_perception_truth_mujoco.py pins both rigs'
    # numbers; the off-centre case is pinned by
    # test_localize_off_center_object_stays_lateral in test_perception.py.
    d = pts.mean(axis=0) - cam
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return center
    d /= n
    rel = pts - cam
    s = rel @ d
    s_near = float(s[s <= np.percentile(s, 10)].mean())
    lateral = rel.mean(axis=0) - float(rel.mean(axis=0) @ d) * d
    near = cam + lateral + d * s_near
    size = float(np.min(np.asarray(extents, dtype=float)))
    size = float(np.clip(size, 0.01, 0.30))
    return near + d * (size / 2.0)


# Sphere refinement gates (see refine_sphere_center).
SPHERE_MIN_POINTS = 200
SPHERE_RADIUS_M = (0.015, 0.06)
SPHERE_SKIRT_M = 0.004
SPHERE_MIN_ARC = np.pi / 2


def refine_sphere_center(center, pts_base):
    """Horizontal centre of a visibly spherical object from its own surface.

    `_recentre_by_size` steps half the smallest box extent along the view
    ray. For a ball seen from above-and-to-the-side the visible cap is thin
    along that ray, so the step is short and the centre stays on the near
    side: measured on the kitchen orange (52 mm), 13.4 mm off under PhysX and
    13.5 mm under Newton -- enough that the jaw pinched the fruit's near
    shoulder and lost it. A least-squares sphere through the observed surface
    recovers the centre regardless of how much of it is visible: 0.7 / 0.5 mm
    on the same clouds.

    Only the x/y centre changes, and only when the cloud is unmistakably a
    sphere; otherwise the prior centre is returned unchanged:
      - at least 200 surface points above the table skirt (the lowest 4 mm);
      - radius 15-60 mm;
      - fit residual RMS <= min(1.5 mm, 4% r) and p95 <= 2.5x that (the
        orange measured 0.07 / 0.14 mm; the kitchen cubes, lemon and can all
        exceed 1.9 mm RMS and are rejected);
      - the surface seen around the centre spans at least 90 degrees;
      - the correction moves the centre by at most one radius.
    Returns (center, report).
    """
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import refine_sphere
        return refine_sphere(center, pts_base)
    report = {"accepted": False}
    pts = np.asarray(pts_base, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < SPHERE_MIN_POINTS:
        return center, {**report, "reason": "insufficient_points"}
    pts = pts[pts[:, 2] > pts[:, 2].min() + SPHERE_SKIRT_M]
    if len(pts) < SPHERE_MIN_POINTS or not np.isfinite(pts).all():
        return center, {**report, "reason": "insufficient_points"}
    A = np.c_[2.0 * pts, np.ones(len(pts))]
    sol, *_ = np.linalg.lstsq(A, (pts ** 2).sum(axis=1), rcond=None)
    c = sol[:3]
    r2 = float(sol[3] + c @ c)
    if not np.isfinite(r2) or r2 <= 0.0:
        return center, {**report, "reason": "invalid_sphere"}
    r = float(np.sqrt(r2))
    res = np.linalg.norm(pts - c, axis=1) - r
    rms = float(np.sqrt(np.mean(res ** 2)))
    p95 = float(np.percentile(np.abs(res), 95))
    report.update(radius_m=r, rms_m=rms, p95_m=p95)
    if not SPHERE_RADIUS_M[0] <= r <= SPHERE_RADIUS_M[1]:
        return center, {**report, "reason": "unsupported_radius"}
    tol = min(0.0015, 0.04 * r)
    if rms > tol or p95 > 2.5 * tol:
        return center, {**report, "reason": "not_spherical"}
    ang = np.sort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    arc = 2.0 * np.pi - float(np.max(np.diff(np.r_[ang, ang[0] + 2.0 * np.pi])))
    report["visible_arc_deg"] = float(np.degrees(arc))
    if arc < SPHERE_MIN_ARC:
        return center, {**report, "reason": "insufficient_visible_arc"}
    out = np.asarray(center, dtype=float).copy()
    if float(np.linalg.norm(c[:2] - out[:2])) > r:
        return center, {**report, "reason": "unsupported_center_correction"}
    out[:2] = c[:2]
    report["accepted"] = True
    return out, report


def _bbox_mask(frame: Frame, det: Detection) -> np.ndarray:
    h, w = frame.rgb.shape[:2]
    if os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1":
        from .cuda_math import bbox_mask
        return bbox_mask((h, w), det.bbox)
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
    workspace_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    require_unique: bool = False,
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
    if workspace_bounds is not None:
        workspace_min, workspace_max = [
            np.asarray(bound, dtype=float).reshape(3) for bound in workspace_bounds
        ]
        if (not np.all(np.isfinite([workspace_min, workspace_max]))
                or np.any(workspace_min > workspace_max)):
            raise SkillError("invalid localization workspace bounds")
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
            if (os.environ.get("CASCADE_REQUIRE_CUDA", "0") == "1"
                    and "can" in str(det.label).lower().replace("_", " ").split()):
                # A partial cylindrical shell is not a box of the same
                # thickness. Use only observed shape evidence; uncertain
                # cylinders and all other objects retain the prior centre.
                from .cuda_math import refine_upright_cylinder
                center, _ = refine_upright_cylinder(center, pts_base)
            else:
                # Shape evidence only (no label): a ball's visible cap defeats
                # the view-ray step above. Non-spherical clouds are untouched.
                center, _ = refine_sphere_center(center, pts_base)
            # Rank only actionable measured candidates. A high-confidence
            # background fruit must not hide a lower-confidence countertop
            # prop, nor prevent the caller trying another camera/prompt.
            if workspace_bounds is not None and (
                    not np.all(np.isfinite(center))
                    or np.any(center < workspace_min) or np.any(center > workspace_max)):
                last_reason = (
                    f"detected {det.label!r} at {np.round(center, 3).tolist()} "
                    "outside the active arm workspace"
                )
                continue
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
        if require_unique and len(candidates) != 1:
            raise ReferenceResolutionError(
                f"Configured visual description for {label!r} matches "
                f"{len(candidates)} reachable objects; refusing an ambiguous grasp"
            )
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
        # Even a single candidate can violate an exclusion or an ordinal.
        # Let the typed reference refusal propagate; it is not a prompt miss.
        if not ref.is_plain:
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
