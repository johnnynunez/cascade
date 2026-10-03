"""Conservative full-tag resolution bound from observed image corners only."""

from fractions import Fraction
from math import isqrt

import numpy as np

ALGORITHM = "rational_det_over_frobenius_8x8_v1"
MIN_SINGULAR_PX_PER_CELL = 4.0


def _sqrt_upper(value):
    scale = 10**24
    root = isqrt(value.numerator * scale * scale // value.denominator)
    if root * root * value.denominator < value.numerator * scale * scale:
        root += 1
    return Fraction(root, scale)


def whole_tag_lower_bound(matrix):
    """Bound sigma_min everywhere in the six-by-six-cell square.

    Binary64 coefficients are treated as exact rationals. Each derivative's
    numerator and the projective denominator are affine; their vertex extrema
    enclose each of 8x8 tiles. |det J|/||J||F <= sigma_min, with sqrt rounded up
    by integer arithmetic. This certifies the represented image homography,
    not the true camera, renderer filtering or successful bit decoding.
    """
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.abs(h).max() > 1e6:
        raise ValueError("bounded finite tag homography required")
    m = [[Fraction(float(v)) for v in row] for row in h]
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if det <= 0:
        raise ValueError("nondegenerate, nonreflected tag homography required")
    minimum = None
    for x in range(8):
        for y in range(8):
            corners = [
                (Fraction(6 * (x + dx), 8), Fraction(6 * (y + dy), 8), 1)
                for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))
            ]
            values = [
                [sum(a * b for a, b in zip(row, p, strict=True)) for row in m]
                for p in corners
            ]
            lo = min(p[2] for p in values)
            hi = max(p[2] for p in values)
            if lo <= 0:
                raise ValueError("tag homography crosses or lies behind infinity")
            maxima = [
                max(abs(m[r][c] * p[2] - m[2][c] * p[r]) for p in values)
                for r in (0, 1)
                for c in (0, 1)
            ]
            upper = _sqrt_upper(sum(v * v for v in maxima)) / lo**2
            value = det / hi**3 / upper
            minimum = value if minimum is None else min(minimum, value)
    return minimum


def require_tag_support(corners):
    """Accept one cyclic OpenCV tag quad; no K/T/depth or design projection."""
    import cv2

    points = np.asarray(corners, dtype=float)
    if (
        points.shape != (4, 2)
        or not np.isfinite(points).all()
        or np.abs(points).max() > 8192
    ):
        raise ValueError("bounded finite tag corners required")
    edges = np.roll(points, -1, axis=0) - points
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(
        edges[:, 0], -1
    )
    if (turns <= 0).any():
        raise ValueError("cyclic convex nonreflected tag corners required")
    h = cv2.getPerspectiveTransform(
        np.array([[0, 0], [6, 0], [6, 6], [0, 6]], np.float32),
        points.astype(np.float32),
    )
    bound = whole_tag_lower_bound(h)
    # Compare rationals, never a rounded-up float at the admission boundary.
    if bound < Fraction(4):
        raise ValueError("whole-tag singular resolution below four pixels per cell")
    return {
        "algorithm": ALGORITHM,
        "minimum_required_px_per_cell": MIN_SINGULAR_PX_PER_CELL,
        "lower_bound_px_per_cell": float(bound),
        "lower_bound_numerator": str(bound.numerator),
        "lower_bound_denominator": str(bound.denominator),
        "image_homography_hex": [[float(v).hex() for v in row] for row in h],
    }
