"""Build an ObjectFix from a pixel, with no detector and no class name.

VIA (arXiv:2607.11119) withholds perception APIs entirely and has the agent
click in an RGB-D point cloud, because a frontier model can see the object
perfectly well; what it lacks is a way to *address* it metrically. Line 169:

    The agent observes only what the interface renders and has no access to
    privileged simulator state. For simplicity, we also do not provide any
    perception-assisting APIs such as segmentation functions, so visual
    understanding is solely the job of the agent and its underlying model.

cascade's skills all route through `_localize(label)`, so the agent can only
act on things the detector names. That is a hard ceiling, measured: on LIBERO
frames the open-vocabulary detector emits 38-44 detections and never once says
`bowl`, so a label-addressed interface cannot express the task at all, while
the object is plainly visible in the image.

This module removes the class name from the addressing path. Given a pixel it
segments the object by depth connectivity (a flood fill in 3D, not in colour),
then produces the same `ObjectFix` the detector path produces, so every
downstream consumer -- grasp planning, the safety harness, the belief store --
works unchanged.

Why depth connectivity rather than a segmentation model: it needs no weights,
no vocabulary and no network call, it runs in single-digit milliseconds, and it
is exactly as class-agnostic as a booth requires. A learned segmenter (SAM3, as
ASPIRE and VoLo use) is a strictly better fill for cluttered scenes and should
sit behind the same interface later; this is the zero-dependency floor.
"""

from __future__ import annotations

import numpy as np

from ..types import Detection, Frame, ObjectFix, SkillError, transform_points
from .grounding import oriented_bbox
from .probe import deproject

#: Depth discontinuity that separates two objects, meters. A 2 cm step at
#: tabletop range is far larger than sensor noise on the same surface and far
#: smaller than the gap between distinct objects.
DEPTH_STEP_M = 0.02

#: Cap the region so a flood fill that escapes onto the table cannot swallow
#: the scene. At 256x256 this is ~15% of the frame.
#:
#: MEASURED LIMITATION: on a LIBERO agentview frame this cap is REACHED
#: (10000 px, extent 102 cm) when probing a bowl resting on a table. Depth
#: connectivity cannot separate an object from the surface it touches: the
#: depth step across the contact line is smaller than DEPTH_STEP_M, so the
#: fill escapes onto the table and the OBB centre lands 34 cm from truth.
#: Hitting the cap therefore means "this fill escaped", and callers must treat
#: a capped region as a failure rather than as an object. See
#: `fix_from_pixel`, which rejects it.
#:
#: The fix is a learned segmenter (SAM3, as ASPIRE/VoLo/RPent use) behind the
#: same interface: it separates touching objects by appearance, which is
#: exactly the information depth alone lacks. Depth connectivity remains a
#: valid zero-dependency path for objects with a visible depth gap all round.
MAX_REGION_PX = 10000

#: Below this a region is a depth speckle, not an object.
MIN_REGION_PX = 24


def segment_at_pixel(
    depth_m: np.ndarray,
    u: int,
    v: int,
    step_m: float = DEPTH_STEP_M,
    max_px: int = MAX_REGION_PX,
) -> np.ndarray:
    """Flood fill the connected surface containing (u, v).

    Grows only across neighbours whose depth differs by less than `step_m`, so
    the region stops at the object's silhouette instead of bleeding onto the
    table behind it. Returns a boolean mask.

    Iterative rather than recursive: a 256x256 region would blow the Python
    recursion limit.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    h, w = d.shape
    if not (0 <= v < h and 0 <= u < w):
        raise SkillError(f"pixel ({u}, {v}) outside the {w}x{h} depth image")

    seed = d[v, u]
    if not np.isfinite(seed) or seed <= 0:
        raise SkillError(
            f"no depth at pixel ({u}, {v}); probe a pixel on the object body"
        )

    mask = np.zeros((h, w), dtype=bool)
    valid = np.isfinite(d) & (d > 0)
    stack = [(v, u)]
    mask[v, u] = True
    count = 1

    while stack and count < max_px:
        y, x = stack.pop()
        z = d[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or mask[ny, nx]:
                continue
            if not valid[ny, nx]:
                continue
            if abs(float(d[ny, nx]) - float(z)) > step_m:
                continue
            mask[ny, nx] = True
            count += 1
            stack.append((ny, nx))
            if count >= max_px:
                break
    return mask


def fix_from_mask(
    frame: Frame,
    T_cam2base: np.ndarray,
    mask: np.ndarray,
    label: str = "object at pixel",
) -> ObjectFix:
    """Lift an arbitrary boolean mask to a full ObjectFix.

    Shared by the depth-connectivity path and the learned-segmenter path so
    the 3D geometry is computed identically either way: only the segmentation
    differs, which keeps a comparison between them honest.
    """
    if not frame.has_depth or frame.depth_m is None:
        raise SkillError("this camera has no depth; cannot lift a mask to 3D")

    d = np.asarray(frame.depth_m, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(d) & (d > 0)
    n = int(valid.sum())
    if n < MIN_REGION_PX:
        raise SkillError(
            f"only {n} pixels with valid depth in the mask; too few to "
            "localize an object"
        )

    ys, xs = np.nonzero(valid)
    zs = d[ys, xs]
    pts_cam = np.stack(
        [deproject(float(x), float(y), float(z), frame.K)
         for x, y, z in zip(xs, ys, zs)]
    )
    pts_base = transform_points(np.asarray(T_cam2base, dtype=float), pts_cam)
    center, extents, axes = oriented_bbox(pts_base)

    det = Detection(
        label=label,
        conf=1.0,          # geometric, not probabilistic: the pixel was given
        bbox=np.array([int(xs.min()), int(ys.min()),
                       int(xs.max()), int(ys.max())], dtype=np.float32),
        mask=valid,
    )
    return ObjectFix(
        label=label,
        position=center,
        points=pts_base,
        detection=det,
        extent=extents,
        axes=axes,
    )


def fix_from_pixel(
    frame: Frame,
    T_cam2base: np.ndarray,
    u: int,
    v: int,
    label: str = "object at pixel",
    step_m: float = DEPTH_STEP_M,
) -> ObjectFix:
    """Lift a pixel to a full ObjectFix: centre, extents, axes, point cloud.

    The centre is the OBB centre of the segmented region, NOT the probed
    surface point. `PointProbe.probe` documents that a ray hits the first
    surface, so probing the middle of a cube returns its top face (+22 mm in z
    on a 4.5 cm cube). Feeding that straight into a grasp centre grasps high
    and skims the object; segmenting first and taking the region's OBB is what
    makes the pixel usable as a grasp target.
    """
    if not frame.has_depth or frame.depth_m is None:
        raise SkillError("this camera has no depth; cannot lift a pixel to 3D")

    mask = segment_at_pixel(frame.depth_m, int(u), int(v), step_m=step_m)
    n = int(mask.sum())
    if n < MIN_REGION_PX:
        raise SkillError(
            f"only {n} connected pixels at ({u}, {v}); that is a depth "
            "speckle, not an object surface"
        )
    if n >= MAX_REGION_PX:
        # The fill hit its cap, which means it escaped onto a connected
        # surface (typically the table the object rests on) instead of
        # stopping at the object silhouette. Measured on LIBERO: the returned
        # centre would be 34 cm from truth with a 102 cm extent. Returning
        # that as a grasp target would drive the arm into the table, so this
        # fails loudly instead. A learned segmenter is the fix.
        raise SkillError(
            f"the surface at ({u}, {v}) is not separable by depth: the region "
            f"grew to {n} px and merged with what it rests on. This object "
            "needs appearance-based segmentation, not depth connectivity."
        )

    return fix_from_mask(frame, T_cam2base, mask, label=label)
