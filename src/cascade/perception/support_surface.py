"""Measure the local supporting surface from the same RGB-D observation."""
from __future__ import annotations

import numpy as np

from ..types import SkillError, transform_points


def estimate_support_z(frame, fix) -> float:
    """Fit nearby table/container-floor depth; refuse an unobserved support.

    The returned height is evaluated at the object XY in the robot base
    frame. No configured table height or remembered object position supplies
    missing geometry. The object mask is excluded before fitting.
    """
    if frame is None or frame.depth_m is None or frame.T_base_cam is None:
        raise SkillError("grasp needs depth and its captured camera pose to measure support height")
    depth = np.asarray(frame.depth_m)
    h, w = frame.rgb.shape[:2]
    if depth.shape != (h, w):
        raise SkillError("RGB/depth resolution mismatch while measuring grasp support")
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0)
    mask = fix.detection.mask
    if mask is not None:
        import cv2

        if mask.shape != depth.shape:
            raise SkillError("object mask is not aligned to depth")
        # Reject mixed depth at the segmentation boundary as well as the object.
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
        valid &= mask[ys, xs] == 0
    else:
        x0, y0, x1, y1 = fix.detection.bbox
        valid &= ~((xs >= x0 - 3) & (xs <= x1 + 3) &
                   (ys >= y0 - 3) & (ys <= y1 + 3))
    z, xs, ys = z[valid], xs[valid], ys[valid]
    K = frame.K
    cam = np.column_stack(((xs - K[0, 2]) * z / K[0, 0],
                           (ys - K[1, 2]) * z / K[1, 1], z))
    pts = transform_points(frame.T_base_cam, cam)
    xy = np.asarray(fix.position[:2])
    pts = pts[np.linalg.norm(pts[:, :2] - xy, axis=1) < .15]
    if len(pts) < 80:
        raise SkillError("not enough nearby depth to measure the grasp support surface")
    rng = np.random.default_rng(0)
    if len(pts) > 2000:
        pts = pts[rng.choice(len(pts), 2000, replace=False)]
    A = np.column_stack((pts[:, :2], np.ones(len(pts))))
    top = float(np.percentile(fix.points[:, 2], 95))
    at_object = np.r_[xy, 1.]
    best = None
    for _ in range(160):
        sample = rng.choice(len(pts), 3, replace=False)
        if abs(np.linalg.det(A[sample])) < 1e-7:
            continue
        plane = np.linalg.solve(A[sample], pts[sample, 2])
        # A supporting surface is approximately horizontal in the base frame;
        # a box wall or the robot itself must not supply its height.
        if np.linalg.norm(plane[:2]) > .35 or plane @ at_object >= top - .002:
            continue
        inliers = np.abs(A @ plane - pts[:, 2]) < .004
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < max(80, .35 * len(pts)):
        raise SkillError("no reliable supporting surface in nearby depth; reobserve from overview")
    plane = np.linalg.lstsq(A[best], pts[best, 2], rcond=None)[0]
    height = float(plane @ at_object)
    if np.linalg.norm(plane[:2]) > .35 or height >= top - .002:
        raise SkillError("measured support surface is inconsistent with object depth")
    return height
