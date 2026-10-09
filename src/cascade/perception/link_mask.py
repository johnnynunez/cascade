"""Robot self-mask from the arm's LINK GEOMETRY (backlog B39).

B32a taught both fusion paths (WorldWatcher._tick and the runtime's
get_observation / _reobserve) to consult `Frame.robot_mask`: a detection more
than half robot pixels IS the robot, and robot pixels never reach 3D. Live on
Isaac that took the bare scene's phantom rate from 17.9-19.6 % to 0 %
(docs/evidence/b32-fusion-self-mask-20261008/). Only Isaac frames carry that
render mask, though. On the real rig the arm's upper link, standing outside
the workspace filter's base cylinder, is still fused as an object whenever an
open-vocabulary detector names it.

This module builds the same mask from what the stack already has on any rig:
the arm's URDF collision geometry, its joint state, and each camera's
intrinsics and extrinsics. `LinkSelfMask.attach(frame, T)` hands fusion a
copy of the frame whose `robot_mask` is the arm's silhouette, and the B32a
gate consumes it unchanged. Opt-in: `workspace_filter.link_self_mask`.

GEOMETRY (`LinkGeometry`, once at load time). Every link's collision geometry
becomes a few small convex PIECES in the link frame:

- meshes (binary/ASCII STL, OBJ): triangles are split until no edge exceeds
  `cell_m`, bucketed by centroid into link-frame cells of `cell_m`, and each
  bucket becomes the 3D hull of its triangles' vertices. A piece CONTAINS its
  triangles, so the union of the projected pieces contains the projected mesh:
  coverage is 1 by construction, up to rasterisation at the edge, which the
  shipped 2 px of dilation absorbs. Bloat is what a hull adds inside one cell.
  Measured on the reBot RS against a per-triangle rasterisation (Isaac cam0 +
  side, 1280x720, four poses, no dilation; docs/evidence/
  b39-link-self-mask-20261009/): one hull per link adds 4.5-33 % of the
  silhouette, 5 cm cells 0.8-5.2 % at 3-5 ms per frame, 2 cm cells
  0.06-1.7 % at 16-23 ms.
- box / cylinder / sphere / capsule primitives: their corners, or the corners
  of a circumscribed polygon / polyhedron, which contains the primitive.
- a mesh that cannot be read (missing, a git-lfs pointer that was never
  fetched, an unsupported format): a capsule from the link origin to each of
  its child joints, `capsule_radius_m` thick, and `sources` says so. A link
  with no collision geometry at all is a frame, not a body: nothing is made up.

The 3D hulls use scipy when it is importable. Without it every vertex is kept:
the same polygons (the 2D hull of the projected points is the projection of
the 3D hull; only sub-pixel rounding at the edge differs, a few pixels in
10^4), measured at 15-31 ms per frame instead of 3-5 ms on the RS arm.

RASTERISATION (`rasterize_pieces`). Each piece is moved into the camera frame,
clipped at `Z_NEAR_M` in front of the camera (a convex piece's clipped vertices
are its vertices in front plus where its edges cross the plane, both kept), and
the 2D convex hull of its projection is filled. Pieces behind the camera add
nothing; pieces outside the image are clipped by the fill. A configurable
dilation (`dilate_px`) is applied on a padded canvas, so a piece just outside
the image still dilates into it.

TIME. The mask is posed with the joint sample NEAREST the frame's time
(`Frame.t` and `RobotState.t` are both `time.monotonic()`). The watcher samples
the arm before inference, the runtime when it takes a frame for
get_observation / _reobserve, and `attach` samples on demand when no sample is
near enough. If the nearest sample is more than `max_skew_s` away, NO mask is
built and `last` says why: missing evidence stays missing, it is never
interpolated or invented. The sample ring and that classification are the
shared capture-time alignment component (`cascade.sensing.alignment`, B50).

FINGERS. Joints past the `n_controlled` commanded ones (the RS's two
prismatic fingers) are posed from the measured gripper opening: `gripper_pos`
mapped through the profile's `closed_pos`/`open_pos` to a fraction, the lower
joint limit being "together" (the Isaac bridge's documented fraction contract;
on the RS URDF the fingers are 0.1 mm apart at the lower limit and 100 mm at
the upper). When the opening is unknown (a failed gripper read, no mapping),
the fingers are swept over their whole range: more mask, never less.
"""

from __future__ import annotations

import math
import re
import struct
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from ..sensing.alignment import ALIGNED, MISSING, SampleBuffer, classify

#: Points nearer the camera plane than this (metres) are clipped: perspective
#: projection is undefined at z = 0 and mirrors what lies behind the camera.
Z_NEAR_M = 0.01

#: A piece straddling the near plane is clipped pairwise (every vertex in front
#: with every vertex behind); above this many pairs its bounding box is clipped
#: instead, a superset (more mask, never less).
_MAX_CLIP_PAIRS = 20_000

#: Polygons reaching past this many pixels are clipped to the canvas (plus a
#: margin) before the fixed-point fill, far from any int32 limit.
_CLIP_LIMIT_PX = 1.0e6
_CLIP_MARGIN_PX = 64.0

#: Subdividing one mesh past this many triangles is refused (MeshUnavailable ->
#: the capsule fallback, reason recorded). The largest reBot RS mesh needs
#: 1.7e5 at the shipped 5 cm (7.4e5 for the whole arm); a mesh authored in
#: millimetres without a URDF `scale` would ask for billions and exhaust
#: memory instead.
_MAX_TRIANGLES = 2_000_000

#: Samples per swept passive REVOLUTE joint. A prismatic sweep is exact with its
#: two ends (the hull of a translated convex piece); a revolute one is a chord
#: approximation, kept for joints without a measured position.
_REVOLUTE_SAMPLES = 5

#: Shipped defaults (configs/demo.yaml `workspace_filter.link_self_mask`).
DEFAULT_DILATE_PX = 2
DEFAULT_MAX_SKEW_S = 0.15
DEFAULT_CELL_M = 0.05
DEFAULT_CAPSULE_RADIUS_M = 0.06

_LFS_POINTER = b"version https://git-lfs"
_STL_RECORD = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])


class MeshUnavailable(ValueError):
    """A collision mesh cannot be read: missing, a git-lfs pointer, unsupported."""


# ── meshes ────────────────────────────────────────────────────────────────


def load_mesh_triangles(path) -> np.ndarray:
    """(N, 3, 3) float64 triangles of a binary/ASCII STL or a Wavefront OBJ.

    Raises MeshUnavailable for a missing file, a git-lfs pointer (the mesh was
    never fetched: CI checkouts do not pull LFS), or any other format.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in (".stl", ".obj"):
        raise MeshUnavailable(f"unsupported mesh format {suffix!r}: {p.name}")
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        raise MeshUnavailable(f"no such file: {p}") from None
    except OSError as exc:
        raise MeshUnavailable(f"unreadable mesh {p}: {exc}") from exc
    if data.startswith(_LFS_POINTER):
        raise MeshUnavailable(f"git-lfs pointer, mesh not fetched: {p.name}")
    try:
        tris = _parse_obj(data) if suffix == ".obj" else _parse_stl(data)
    except (ValueError, IndexError) as exc:
        raise MeshUnavailable(f"unreadable mesh {p.name}: {exc}") from exc
    if tris.size == 0 or not np.isfinite(tris).all():
        raise MeshUnavailable(f"no finite triangles in {p.name}")
    return tris


def _parse_stl(data: bytes) -> np.ndarray:
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):  # binary, whatever its header says ("solid ..." happens)
            rec = np.frombuffer(data, dtype=_STL_RECORD, count=n, offset=84)
            return rec["v"].reshape(n, 3, 3).astype(np.float64)
    text = data.decode("ascii", errors="ignore")
    vals = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not vals or len(vals) % 3:
        raise ValueError("neither a binary STL nor ASCII STL facets")
    return np.asarray(vals, dtype=np.float64).reshape(-1, 3, 3)


def _parse_obj(data: bytes) -> np.ndarray:
    verts, faces = [], []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif len(parts) >= 4 and parts[0] == "f":
            faces.append([int(tok.split("/")[0]) for tok in parts[1:]])
    n = len(verts)
    tris = []
    for face in faces:
        idx = [i - 1 if i > 0 else n + i for i in face]  # 1-based; negative = relative
        tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    if not tris:
        return np.empty((0, 3, 3))
    tri = np.asarray(tris)
    if tri.min() < 0 or tri.max() >= n:
        raise ValueError("OBJ face index out of range")
    return np.asarray(verts, dtype=np.float64)[tri]


def _subdivide(tris: np.ndarray, max_edge: float) -> np.ndarray:
    """Split triangles 1 -> 4 at their edge midpoints until no edge exceeds
    `max_edge`: the same surface, in pieces small enough to bucket by cell."""
    done, todo = [], np.asarray(tris, dtype=np.float64)
    while len(todo):
        edges = np.linalg.norm(todo - np.roll(todo, 1, axis=1), axis=2).max(axis=1)
        big = edges > max_edge
        done.append(todo[~big])
        t = todo[big]
        if not len(t):
            break
        if sum(len(x) for x in done) + 4 * len(t) > _MAX_TRIANGLES:
            raise MeshUnavailable(f"mesh too large for {max_edge} m cells (a missing URDF scale?): "
                                  f"more than {_MAX_TRIANGLES} triangles")
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        todo = np.concatenate([np.stack(s, axis=1) for s in
                               ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return np.concatenate(done) if done else np.empty((0, 3, 3))


def _hull_vertices(points) -> np.ndarray:
    """The 3D convex-hull vertices of `points` (all unique points without
    scipy: the silhouette of their hull is the same, only costlier)."""
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 3), axis=0)
    if len(pts) <= 4:
        return pts
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return pts
    try:
        # QJ joggles degenerate (flat) cells instead of refusing them
        return pts[ConvexHull(pts, qhull_options="QJ").vertices]
    except Exception:  # noqa: BLE001 - qhull refusing a set: keep every point (same silhouette)
        return pts


def _mesh_pieces(tris: np.ndarray, cell_m: float) -> list[np.ndarray]:
    tris = _subdivide(tris, cell_m)
    if not len(tris):
        return []
    key = np.floor(tris.mean(axis=1) / cell_m).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    cuts = np.flatnonzero(np.diff(inv[order])) + 1
    return [_hull_vertices(tris[group].reshape(-1, 3)) for group in np.split(order, cuts)]


# ── primitives (corners of a shape that CONTAINS the primitive) ─────────────


def _box_points(size) -> np.ndarray:
    h = np.abs(np.asarray(size, dtype=np.float64).reshape(3)) / 2
    return np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])


def _cylinder_points(radius: float, length: float, n: int = 16) -> np.ndarray:
    r = abs(radius) / math.cos(math.pi / n)  # a circumscribed n-gon contains the circle
    ang = np.arange(n) * 2 * math.pi / n
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    half = abs(length) / 2
    return np.concatenate([np.c_[ring, np.full(n, -half)], np.c_[ring, np.full(n, half)]])


def _unit_sphere_cover() -> np.ndarray:
    """Vertices of a once-subdivided icosahedron, scaled so that every face lies
    at least 1 from the centre: their hull contains the unit sphere."""
    p = (1 + 5 ** 0.5) / 2
    v = [(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p),
         (0, -1, -p), (0, 1, -p), (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.asarray(x, dtype=np.float64) / np.linalg.norm(x) for x in v]
    mids: dict[tuple[int, int], int] = {}

    def mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in mids:
            m = verts[i] + verts[j]
            verts.append(m / np.linalg.norm(m))
            mids[key] = len(verts) - 1
        return mids[key]

    fine = []
    for a, b, c in faces:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        fine += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    V = np.asarray(verts)
    tri = V[np.asarray(fine)]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    d_min = float(np.abs(np.einsum("ij,ij->i", normal, tri[:, 0])).min())
    return V / d_min


_SPHERE_COVER = _unit_sphere_cover()


def _capsule_points(a, b, radius: float) -> np.ndarray:
    cover = _SPHERE_COVER * abs(float(radius))
    return np.concatenate([cover + np.asarray(a, dtype=np.float64), cover + np.asarray(b, dtype=np.float64)])


# ── URDF ──────────────────────────────────────────────────────────────────


def _origin(el) -> np.ndarray:
    from ..types import pose_to_transform

    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return pose_to_transform(xyz + rpy)


def _resolve_mesh(filename: str, base_dir: Path) -> Path:
    """package://<pkg>/<rel> -> <pkg> in the nearest ancestor of the model
    holding a directory of that name (the model's own package directory is
    found from its parent); file:// and relative paths -> next to the model."""
    if filename.startswith("package://"):
        pkg, _, rel = filename[len("package://"):].partition("/")
        for d in (base_dir, *base_dir.parents):
            if (d / pkg).is_dir():
                return d / pkg / rel
        return base_dir / rel  # reported as "no such file" by the loader
    if filename.startswith("file://"):
        filename = filename[len("file://"):]
    p = Path(filename)
    return p if p.is_absolute() else base_dir / p


def _transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]


@dataclass(frozen=True)
class LinkGeometry:
    """Each link's collision geometry as convex pieces in the link frame.

    `pieces[link]` is a tuple of (k, 3) point sets whose convex hulls together
    contain the link's collision geometry; `sources[link]` says how it was
    modelled ("mesh", "box", ..., or "capsule (<why the geometry was not read>)").
    """

    pieces: dict[str, tuple[np.ndarray, ...]]
    sources: dict[str, str]

    @property
    def links(self) -> tuple[str, ...]:
        return tuple(self.pieces)

    def n_points(self) -> int:
        return sum(len(p) for pieces in self.pieces.values() for p in pieces)

    def n_pieces(self) -> int:
        return sum(len(pieces) for pieces in self.pieces.values())

    @classmethod
    def from_urdf(cls, path, *, cell_m: float = DEFAULT_CELL_M,
                  capsule_radius_m: float = DEFAULT_CAPSULE_RADIUS_M) -> "LinkGeometry":
        """From the arm profile's `model`. A USD model carries no collision
        geometry here (usd_model drops it), so every link is a capsule."""
        path = Path(path)
        if path.suffix.lower() in (".usd", ".usda"):
            from ..control.usd_model import urdf_xml_from_usd

            return cls.from_urdf_xml(urdf_xml_from_usd(path), path.parent, cell_m=cell_m,
                                     capsule_radius_m=capsule_radius_m, geometry_dropped=True)
        return cls.from_urdf_xml(path.read_text(), path.parent, cell_m=cell_m,
                                 capsule_radius_m=capsule_radius_m)

    @classmethod
    def from_urdf_xml(cls, xml: str, base_dir, *, cell_m: float = DEFAULT_CELL_M,
                      capsule_radius_m: float = DEFAULT_CAPSULE_RADIUS_M,
                      geometry_dropped: bool = False) -> "LinkGeometry":
        if not (isinstance(cell_m, (int, float)) and math.isfinite(cell_m) and cell_m > 0):
            raise ValueError(f"cell_m must be a positive number, got {cell_m!r}")
        if not (isinstance(capsule_radius_m, (int, float)) and math.isfinite(capsule_radius_m)
                and capsule_radius_m > 0):
            raise ValueError(f"capsule_radius_m must be a positive number, got {capsule_radius_m!r}")
        base_dir = Path(base_dir)
        root = ET.fromstring(xml)
        children: dict[str, list[np.ndarray]] = {}
        for joint in root.iter("joint"):
            parent = joint.find("parent")
            if parent is not None:
                children.setdefault(parent.get("link"), []).append(_origin(joint.find("origin"))[:3, 3])

        def capsules(name: str) -> list[np.ndarray]:
            ends = children.get(name) or [np.zeros(3)]
            return [_capsule_points(np.zeros(3), end, capsule_radius_m) for end in ends]

        pieces: dict[str, tuple[np.ndarray, ...]] = {}
        sources: dict[str, str] = {}
        for link in root.iter("link"):
            name = link.get("name")
            if geometry_dropped:
                pieces[name] = tuple(capsules(name))
                sources[name] = "capsule (the model file carries no collision geometry)"
                continue
            got: list[np.ndarray] = []
            kinds: list[str] = []
            missing: list[str] = []
            for col in link.findall("collision"):
                T = _origin(col.find("origin"))
                geo = col.find("geometry")
                el = geo[0] if geo is not None and len(geo) else None
                kind = el.tag if el is not None else "nothing"
                try:
                    if kind == "mesh":
                        tris = load_mesh_triangles(_resolve_mesh(el.get("filename", ""), base_dir))
                        scale = [float(v) for v in el.get("scale", "1 1 1").split()]
                        tris = _transform(tris * np.asarray(scale, dtype=np.float64), T)
                        got += _mesh_pieces(tris, float(cell_m))
                    elif kind == "box":
                        got.append(_transform(_box_points([float(v) for v in el.get("size").split()]), T))
                    elif kind == "cylinder":
                        got.append(_transform(_cylinder_points(float(el.get("radius")),
                                                               float(el.get("length"))), T))
                    elif kind == "sphere":
                        got.append(_transform(_SPHERE_COVER * abs(float(el.get("radius"))), T))
                    elif kind == "capsule":  # non-standard, but common: along z, like a cylinder
                        half = float(el.get("length")) / 2
                        got.append(_transform(_capsule_points([0, 0, -half], [0, 0, half],
                                                              float(el.get("radius"))), T))
                    else:
                        raise MeshUnavailable(f"unsupported collision geometry <{kind}>")
                    kinds.append(kind)
                except MeshUnavailable as exc:
                    missing.append(str(exc))
                except (TypeError, ValueError, AttributeError) as exc:
                    missing.append(f"malformed <{kind}>: {exc}")
            if missing:
                got += capsules(name)
                sources[name] = "capsule (" + "; ".join(missing) + ")"
            elif kinds:
                sources[name] = "+".join(sorted(set(kinds)))
            else:
                continue  # no collision geometry: a frame, not a body
            pieces[name] = tuple(got)
        return cls(pieces=pieces, sources=sources)


# ── rasterisation ─────────────────────────────────────────────────────────


def _clip_near(pc: np.ndarray, front: np.ndarray) -> np.ndarray:
    """A convex piece cut at z = Z_NEAR_M: its vertices in front, plus every
    front-behind segment's crossing (a superset of its edges' crossings, all
    inside the piece, so the hull is exactly the clipped piece)."""
    F, B = pc[front], pc[~front]
    if len(F) * len(B) > _MAX_CLIP_PAIRS:
        lo, hi = pc.min(axis=0), pc.max(axis=0)
        box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        f = box[:, 2] > Z_NEAR_M
        F, B = box[f], box[~f]
    s = (Z_NEAR_M - F[:, None, 2]) / (B[None, :, 2] - F[:, None, 2])
    cross = F[:, None, :] + s[..., None] * (B[None, :, :] - F[:, None, :])
    return np.concatenate([F, cross.reshape(-1, 3)])


def _clip_polygon(poly: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Sutherland-Hodgman: a convex polygon cut to a rectangle (exact)."""
    for axis, bound, above in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if len(poly) == 0:
            break
        out = []
        prev = poly[-1]
        prev_in = prev[axis] >= bound if above else prev[axis] <= bound
        for cur in poly:
            cur_in = cur[axis] >= bound if above else cur[axis] <= bound
            if cur_in != prev_in:
                out.append(prev + (bound - prev[axis]) / (cur[axis] - prev[axis]) * (cur - prev))
            if cur_in:
                out.append(cur)
            prev, prev_in = cur, cur_in
        poly = np.asarray(out, dtype=np.float64).reshape(-1, 2)
    return poly


def rasterize_pieces(pieces: Iterable[np.ndarray], K, T_cam2base, shape, dilate_px: int = 0) -> np.ndarray:
    """(H, W) bool: the union of the convex pieces (base-frame points) as seen
    by a pinhole camera with intrinsics `K` and cam->base extrinsics, dilated
    by `dilate_px` pixels."""
    h, w = int(shape[0]), int(shape[1])
    d = max(int(dilate_px), 0)
    canvas = np.zeros((h + 2 * d, w + 2 * d), dtype=np.uint8)
    parts = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in pieces]
    parts = [p for p in parts if len(p)]
    if parts:
        T = np.asarray(T_cam2base, dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        Kf = np.asarray(K, dtype=np.float64)
        cam = (np.concatenate(parts) - t) @ R          # base -> camera frame
        bounds = np.cumsum([0] + [len(p) for p in parts])
        cw, ch = w + 2 * d, h + 2 * d
        for i in range(len(parts)):
            pc = cam[bounds[i]:bounds[i + 1]]
            front = pc[:, 2] > Z_NEAR_M
            if not front.any():
                continue                                # wholly behind the camera
            if not front.all():
                pc = _clip_near(pc, front)
            x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
            uv = np.stack([Kf[0, 0] * x + Kf[0, 1] * y + Kf[0, 2] + d, Kf[1, 1] * y + Kf[1, 2] + d], axis=1)
            if (uv[:, 0].max() < -0.5 or uv[:, 0].min() > cw - 0.5
                    or uv[:, 1].max() < -0.5 or uv[:, 1].min() > ch - 0.5):
                continue                                # wholly outside the canvas
            hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.float64)
            if np.abs(hull).max() > _CLIP_LIMIT_PX:
                hull = _clip_polygon(hull, -_CLIP_MARGIN_PX, -_CLIP_MARGIN_PX,
                                     cw + _CLIP_MARGIN_PX, ch + _CLIP_MARGIN_PX)
                if not len(hull):
                    continue
            cv2.fillConvexPoly(canvas, np.round(hull * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    if d:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        canvas = cv2.dilate(canvas, kernel)[d:d + h, d:d + w]
    return canvas.astype(bool)


# ── time alignment ────────────────────────────────────────────────────────


class JointSamples(SampleBuffer):
    """A bounded, thread-safe ring of RobotState samples by their monotonic `t`.

    The ring and its nearest-sample lookup are the shared capture-time
    alignment component (`cascade.sensing.alignment`, B50).
    """

    def record(self, state) -> bool:
        """Keep `state` if its time and joint vector are finite; False otherwise."""
        try:
            t = float(state.t)
            q = np.asarray(state.q, dtype=np.float64).reshape(-1)
        except (AttributeError, TypeError, ValueError):
            return False
        if not math.isfinite(t) or q.size == 0 or not np.isfinite(q).all():
            return False
        return self.add(t, state)


# ── link poses from the arm model ─────────────────────────────────────────


class KinematicsLinkPoses:
    """Link placements (base frame) from cascade's Kinematics model.

    `joint_signs` are baked into that model at load time, so `state.q` is in
    the LOCAL convention like every q in the repo (AGENTS.md "Joint mirror").
    Returns one {link: 4x4} per configuration: one when every passive joint's
    position is known, several when the fingers are swept (see module docs).
    Pinocchio is the `kinematics` extra; it is imported here, lazily, and the
    arm's own Kinematics cannot exist without it.
    """

    def __init__(self, kin, links, *, gripper_open=None, gripper_closed=None):
        import pinocchio as pin

        self._pin = pin
        self._model = kin.model
        self._n = int(kin.n)
        self._local = threading.local()
        self._fids: dict[str, int] = {}
        for name in links:
            fid = self._model.getFrameId(name)
            if fid >= len(self._model.frames):
                raise KeyError(f"link {name!r} is not a frame of the arm model")
            self._fids[name] = fid
        self._open = None if gripper_open is None else float(gripper_open)
        self._closed = None if gripper_closed is None else float(gripper_closed)
        self._neutral = np.asarray(pin.neutral(self._model), dtype=np.float64)
        #: (q index, lower, upper, prismatic) of every 1-DoF joint past the commanded ones
        self._passive: list[tuple[int, float, float, bool]] = []
        for j in range(1, self._model.njoints):
            joint = self._model.joints[j]
            if joint.nq != 1 or joint.idx_q < self._n:
                continue
            kind = joint.shortname()
            prismatic = kind.startswith("JointModelP") and "Planar" not in kind
            lo = float(self._model.lowerPositionLimit[joint.idx_q])
            hi = float(self._model.upperPositionLimit[joint.idx_q])
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                lo, hi = (0.0, 0.0) if prismatic else (-math.pi, math.pi)
            self._passive.append((int(joint.idx_q), lo, hi, prismatic))

    def _data(self):
        data = getattr(self._local, "data", None)
        if data is None:
            data = self._local.data = self._model.createData()
        return data

    def _fraction(self, state) -> float | None:
        """Measured opening as a fraction of the passive joints' travel, or None."""
        if self._open is None or self._closed is None or self._open == self._closed:
            return None
        if not getattr(state, "gripper_valid", False):
            return None
        try:
            pos = float(state.gripper_pos)
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(pos):
            return None
        return min(max((pos - self._closed) / (self._open - self._closed), 0.0), 1.0)

    def __call__(self, state) -> list[dict[str, np.ndarray]]:
        q = np.asarray(state.q, dtype=np.float64).reshape(-1)
        if q.size < self._n or not np.isfinite(q[: self._n]).all():
            raise ValueError(f"joint state has {q.size} finite values, the arm commands {self._n}")
        base = self._neutral.copy()
        base[: self._n] = q[: self._n]
        frac = self._fraction(state)
        swept = []
        for idx, lo, hi, prismatic in self._passive:
            if prismatic and frac is not None:
                base[idx] = lo + frac * (hi - lo)   # the lower limit is "together"
            else:
                swept.append((idx, lo, hi, prismatic))
        configs = [base]
        if swept:
            n = 2 if all(p for *_, p in swept) else _REVOLUTE_SAMPLES
            configs = []
            for s in np.linspace(0.0, 1.0, n):
                c = base.copy()
                for idx, lo, hi, _ in swept:
                    c[idx] = lo + s * (hi - lo)
                configs.append(c)
        pin, data, out = self._pin, self._data(), []
        for c in configs:
            pin.forwardKinematics(self._model, data, c)
            pin.updateFramePlacements(self._model, data)
            out.append({name: np.array(data.oMf[fid].homogeneous) for name, fid in self._fids.items()})
        return out


# ── the provider fusion consults ──────────────────────────────────────────


@dataclass
class _Arm:
    name: str
    geometry: LinkGeometry
    poses: Callable
    state_fn: Callable
    base_T: np.ndarray | None
    samples: JointSamples
    error: str | None = None


def _distinct(transforms: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for T in transforms:
        if not any(np.allclose(T, kept, atol=1e-12) for kept in out):
            out.append(T)
    return out


class LinkSelfMask:
    """The robot's own pixels from link geometry, for frames without a render mask.

    `last` describes the latest decision: {"reason", "t", "arms": {name:
    {"reason", "skew_s", "error"}}, "pixels", "ms"}; `counts` tallies reasons:
    "link_mask" (built), "render_mask" (the frame had one: untouched),
    "no_joint_state", "stale_joint_state" (no mask: missing evidence stays
    missing), "error" (no mask; fusion falls back to the base cylinder alone).
    """

    def __init__(self, *, dilate_px: int = DEFAULT_DILATE_PX, max_skew_s: float = DEFAULT_MAX_SKEW_S,
                 history: int = 64):
        if isinstance(dilate_px, bool) or not isinstance(dilate_px, int) or dilate_px < 0:
            raise ValueError(f"dilate_px must be a non-negative integer, got {dilate_px!r}")
        if not (isinstance(max_skew_s, (int, float)) and math.isfinite(max_skew_s) and max_skew_s > 0):
            raise ValueError(f"max_skew_s must be a positive number, got {max_skew_s!r}")
        self.dilate_px = int(dilate_px)
        self.max_skew_s = float(max_skew_s)
        self._history = int(history)
        self.arms: list[_Arm] = []
        self.counts: Counter = Counter()
        self.last: dict | None = None
        self._lock = threading.Lock()
        self._logged_error: str | None = None

    def add_arm(self, name: str, geometry: LinkGeometry, poses: Callable, state_fn: Callable,
                *, base_T=None) -> None:
        """`poses(state) -> [{link: 4x4}]` places the links in the arm's base
        frame; `base_T` (the profile's `base_pose`) moves them to the table
        frame the camera extrinsics use; `state_fn() -> RobotState | None`."""
        self.arms.append(_Arm(str(name), geometry, poses, state_fn,
                              None if base_T is None else np.asarray(base_T, dtype=np.float64).reshape(4, 4),
                              JointSamples(self._history)))

    # -- joint samples ------------------------------------------------------

    def sample(self) -> None:
        """Read every arm once (call it as a frame is taken, before inference)."""
        for arm in self.arms:
            self._sample(arm)

    @staticmethod
    def _sample(arm: _Arm) -> None:
        try:
            state = arm.state_fn()
        except Exception as exc:  # noqa: BLE001 - a failed read is missing evidence, never a fusion error
            arm.error = f"{type(exc).__name__}: {exc}"
            return
        if state is None:
            arm.error = "no joint state (the arm is in standby or not readable)"
        elif not arm.samples.record(state):
            arm.error = "unusable joint state (non-finite q or t)"
        else:
            arm.error = None

    def _arm_pieces(self, arm: _Arm, t: float):
        pairing = classify(t, arm.samples.nearest(t), max_skew_s=self.max_skew_s)
        if pairing.status != ALIGNED:
            self._sample(arm)  # on demand: nothing was sampled near this frame
            pairing = classify(t, arm.samples.nearest(t), max_skew_s=self.max_skew_s)
        if pairing.status == MISSING:
            return None, {"reason": "no_joint_state", "skew_s": None, "error": arm.error}
        if pairing.status != ALIGNED:   # stale: the value is withheld, never interpolated
            return None, {"reason": "stale_joint_state", "skew_s": pairing.skew_s, "error": arm.error}
        state, skew = pairing.sample, pairing.skew_s
        pose_sets = arm.poses(state)
        out = []
        for link, link_pieces in arm.geometry.pieces.items():
            placed = _distinct([poses[link] for poses in pose_sets if link in poses])
            if not placed:
                continue  # a link the pose source does not place cannot be drawn
            if arm.base_T is not None:
                placed = [arm.base_T @ T for T in placed]
            for piece in link_pieces:
                out.append(np.concatenate([_transform(piece, T) for T in placed]))
        return out, {"reason": "link_mask", "skew_s": skew, "error": None}

    # -- masks ----------------------------------------------------------------

    def mask_for(self, frame, T_cam2base) -> np.ndarray | None:
        """The arms' silhouette at `frame.t` in `frame`'s image, or None (see `last`)."""
        t0 = time.perf_counter()
        arms: dict[str, dict] = {}
        try:
            t = float(frame.t)
            pieces = []
            for arm in self.arms:
                got, info = self._arm_pieces(arm, t)
                arms[arm.name] = info
                if got is not None:
                    pieces += got
            if not any(info["reason"] == "link_mask" for info in arms.values()):
                stale = any(info["reason"] == "stale_joint_state" for info in arms.values())
                return self._done(frame, None, "stale_joint_state" if stale else "no_joint_state", arms, t0)
            h, w = frame.rgb.shape[:2]
            mask = rasterize_pieces(pieces, frame.K, T_cam2base, (h, w), self.dilate_px)
            return self._done(frame, mask, "link_mask", arms, t0)
        except Exception as exc:  # noqa: BLE001 - never break fusion: no mask, the cylinder alone
            return self._done(frame, None, "error", arms, t0, error=f"{type(exc).__name__}: {exc}")

    def attach(self, frame, T_cam2base):
        """`frame` itself when it carries a render self-mask (never overwritten)
        or when no link mask can be built; otherwise a shallow COPY whose
        `robot_mask` is the link mask. The camera's frame is never written: the
        occupancy map reads `robot_mask` too and keeps its own body masking."""
        if getattr(frame, "robot_mask", None) is not None:
            self._done(frame, None, "render_mask", {}, time.perf_counter())
            return frame
        mask = self.mask_for(frame, T_cam2base)
        return frame if mask is None else replace(frame, robot_mask=mask)

    def _done(self, frame, mask, reason: str, arms: dict, t0: float, error: str | None = None):
        last = {"reason": reason, "t": getattr(frame, "t", None), "arms": arms,
                "pixels": 0 if mask is None else int(np.count_nonzero(mask)),
                "ms": (time.perf_counter() - t0) * 1e3}
        if error is not None:
            last["error"] = error
            if error != self._logged_error:  # state changes, not 3 Hz spam
                print(f"[link-self-mask] no mask: {error}", file=sys.stderr)
            self._logged_error = error
        with self._lock:
            self.counts[reason] += 1
            self.last = last
        return mask

    def stats(self) -> dict:
        with self._lock:
            counts, last = dict(self.counts), None if self.last is None else dict(self.last)
        return {
            "dilate_px": self.dilate_px, "max_skew_s": self.max_skew_s, "counts": counts, "last": last,
            "arms": [{"name": a.name, "links": len(a.geometry.pieces), "pieces": a.geometry.n_pieces(),
                      "points": a.geometry.n_points(), "sources": dict(a.geometry.sources),
                      "samples": len(a.samples), "error": a.error} for a in self.arms],
        }


# ── configuration ─────────────────────────────────────────────────────────


def _number(block, key: str, default, *, allow_none: bool = False) -> float | None:
    value = block.get(key, default)
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"workspace_filter.link_self_mask.{key} must be a positive number, got {value!r}")
    return float(value)


def link_self_mask_settings(cfg) -> dict | None:
    """Parsed `workspace_filter.link_self_mask`, or None when it is off.

    Off (the shipped default) also when `workspace_filter.self_mask` is false:
    the link mask only feeds that gate. Bad values raise ValueError.
    """
    wf = cfg.get("workspace_filter") if hasattr(cfg, "get") else None
    block = wf.get("link_self_mask") if hasattr(wf, "get") else None
    if block is None:
        return None
    if not hasattr(block, "get"):
        raise ValueError(f"workspace_filter.link_self_mask must be a mapping, got {block!r}")
    enabled = block.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"workspace_filter.link_self_mask.enabled must be true or false, got {enabled!r}")
    if not enabled or wf.get("self_mask", True) is False:
        return None
    dilate = block.get("dilate_px", DEFAULT_DILATE_PX)
    if isinstance(dilate, bool) or not isinstance(dilate, int) or dilate < 0:
        raise ValueError(f"workspace_filter.link_self_mask.dilate_px must be a non-negative integer, "
                         f"got {dilate!r}")
    return {"dilate_px": int(dilate),
            "max_skew_s": _number(block, "max_skew_s", DEFAULT_MAX_SKEW_S),
            "cell_m": _number(block, "cell_m", DEFAULT_CELL_M),
            "capsule_radius_m": _number(block, "capsule_radius_m", None, allow_none=True)}


def _state_reader(arm) -> Callable:
    def read():
        # Never materialize a standby LazyArm: powering the motors is not a
        # side effect of looking (no state: no mask).
        if not getattr(arm, "connected", True):
            return None
        return arm.get_state()

    return read


def build_link_self_mask(cfg, arms) -> LinkSelfMask | None:
    """The opt-in link self-mask for a runtime, or None (nothing loaded, no arm
    read) when `workspace_filter.link_self_mask.enabled` is false.

    `arms`: (name, arm profile, Kinematics, arm) per arm, in rig order. Each
    arm's geometry comes from its profile's `model`, its finger mapping from
    `gripper.open_pos/closed_pos`, its table placement from `base_pose`, and
    the capsule fallback radius from `body_mask_radius_m` (the occupancy body
    mask's) unless the block sets `capsule_radius_m`.
    """
    settings = link_self_mask_settings(cfg)
    if settings is None:
        return None
    try:
        import pinocchio  # noqa: F401 - the `kinematics` extra; FK needs it
    except ImportError as exc:
        print(f"[cascade] link self-mask: OFF -- it needs pinocchio (the `kinematics` extra): {exc}",
              file=sys.stderr)
        return None
    from ..types import pose_to_transform

    out = LinkSelfMask(dilate_px=settings["dilate_px"], max_skew_s=settings["max_skew_s"])
    for name, acfg, kin, arm in arms:
        model = acfg.get("model")
        if not model:
            raise ValueError(f"workspace_filter.link_self_mask: arm {name!r} has no `model` to take links from")
        radius = settings["capsule_radius_m"]
        if radius is None:
            radius = float(acfg.get("body_mask_radius_m", DEFAULT_CAPSULE_RADIUS_M))
        geometry = LinkGeometry.from_urdf(model, cell_m=settings["cell_m"], capsule_radius_m=radius)
        grip = acfg.get("gripper") or {}
        poses = KinematicsLinkPoses(kin, geometry.links, gripper_open=grip.get("open_pos"),
                                    gripper_closed=grip.get("closed_pos"))
        pose = acfg.get("base_pose")
        out.add_arm(name, geometry, poses, _state_reader(arm),
                    base_T=None if pose is None else pose_to_transform(pose))
        capsules = sorted(k for k, v in geometry.sources.items() if v.startswith("capsule"))
        print(f"[cascade] link self-mask: {name}: {len(geometry.links)} links, {geometry.n_pieces()} pieces "
              f"({len(capsules)} capsule fallback{'' if not capsules else ': ' + ', '.join(capsules)}); "
              f"dilate {out.dilate_px} px, max skew {out.max_skew_s} s", file=sys.stderr)
    return out
