"""Analytic synthetic raster only; never reads native images or camera metadata."""

import hashlib
import json
from pathlib import Path

import numpy as np


def read_bound_json(path, expected_sha256):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("frozen synthetic input digest differs")
    return json.loads(data)


def project(h, xy):
    a = np.c_[xy, np.ones(len(xy))] @ np.asarray(h, float).T
    if not np.isfinite(a).all() or (a[:, 2] <= 0).any():
        raise ValueError("synthetic homography at or behind infinity")
    return a[:, :2] / a[:, 2, None]


def _local(inv, u, v):
    d = inv[2, 0] * u + inv[2, 1] * v + inv[2, 2]
    if (d <= 0).any():
        raise ValueError("synthetic pixel footprint crosses infinity")
    return (
        (inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]) / d,
        (inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]) / d,
    )


def raster(h, rectangles, *, width=640, height=480, samples=16, exhaustive=False):
    """Exact regular N×N integration, accelerated only on uniform footprints.

    Input rectangles are disjoint dark cells in local metric coordinates.
    All pixel centers use integer OpenCV coordinates. Full-inside and full-
    outside classification uses the four footprint vertices: projective maps
    with positive denominator preserve the convex rectangle's half-planes.
    Boundary pixels still evaluate every declared sample. exhaustive=True is
    an independent equivalence control for small CPU fixtures.
    """
    h = np.asarray(h, float)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or type(samples) is not int
        or samples not in (2, 4, 16, 32)
        or type(width) is not int
        or type(height) is not int
        or not 0 < width * height <= 640 * 480
    ):
        raise ValueError("bounded synthetic raster required")
    inv = np.linalg.inv(h)
    total = np.zeros((height, width), np.uint32)
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    for rect in rectangles:
        a, b, c, d = map(float, rect)
        if not np.isfinite(rect).all() or a >= c or b >= d:
            raise ValueError("finite nonempty local rectangle required")
        quad = project(h, [[a, b], [c, b], [c, d], [a, d]])
        lo = np.maximum(np.floor(quad.min(axis=0) - 0.5).astype(int), [0, 0])
        hi = np.minimum(np.ceil(quad.max(axis=0) + 0.5).astype(int), [width, height])
        if (lo >= hi).any():
            continue
        uu, vv = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]))
        # An exact area-interior pixel needs no256 repeated membership tests.
        xs, ys = _local(
            inv,
            uu[..., None] + [-0.5, 0.5, 0.5, -0.5],
            vv[..., None] + [-0.5, -0.5, 0.5, 0.5],
        )
        inside = (
            (xs.min(axis=-1) >= a)
            & (xs.max(axis=-1) < c)
            & (ys.min(axis=-1) >= b)
            & (ys.max(axis=-1) < d)
        )
        outside = (
            (xs.max(axis=-1) < a)
            | (xs.min(axis=-1) >= c)
            | (ys.max(axis=-1) < b)
            | (ys.min(axis=-1) >= d)
        )
        if exhaustive:
            inside[:] = False
            outside[:] = False
        counts = inside.astype(np.uint32) * samples * samples
        indices = np.where(~inside & ~outside)
        u, v = uu[indices], vv[indices]
        for oy in offsets:
            for ox in offsets:
                x, y = _local(inv, u + ox, v + oy)
                counts[indices] += (x >= a) & (x < c) & (y >= b) & (y < d)
        total[lo[1] : hi[1], lo[0] : hi[0]] += counts
    if (total > samples * samples).any():
        raise ValueError("synthetic dark rectangles overlap")
    gray = np.rint(255 * (1 - total.astype(float) / (samples * samples))).astype(
        np.uint8
    )
    return np.repeat(gray[:, :, None], 3, axis=2)


def filtered(rgb, treatment):
    """The fixed corpus filter; does not run any detector or fit."""
    sigma = treatment["gaussian_sigma_px"]
    if sigma == 0:
        return np.array(rgb, copy=True)
    import cv2

    kernel = treatment["kernel"]
    pad = kernel // 2
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    blurred = cv2.GaussianBlur(
        padded,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    return np.ascontiguousarray(blurred[pad:-pad, pad:-pad])
