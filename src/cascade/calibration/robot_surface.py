"""The arm's own surface, sampled from its URDF meshes and posed by FK.

Markerless hand-eye calibration (``calibration/markerless.py``) and the
runtime extrinsic drift monitor fit depth against THIS model, so two things
are deliberate here:

* The geometry is posed by the caller's ``Kinematics`` object -- its own
  Pinocchio model, built from the URDF with the profile's ``joint_signs``
  baked in -- never by a second, separately loaded model. A second copy
  without the sign bake would pose every link mirrored and the fit would
  converge, confidently, onto the wrong arm.
* Only links whose pose FK actually knows are included: those driven by the
  first ``kin.n`` (measured) joints. ``Kinematics`` zero-pads the rest -- on
  the reBot that is the two passive finger slides -- so their geometry would
  sit wherever zero puts it, not where the jaws are.

The meshes are CAD exports: they carry internal parts (bores, screw heads,
ribs) that no camera can see. Sampled as-is, those points pass a coarse
z-buffer whenever they sit within its tolerance of the outer skin and pull
the fit behind the surface. ``exterior_mask`` keeps only points visible from
outside the link from at least one of 26 directions (orthographic, per
link, once), so the model is the skin a depth camera can measure.

Pure numpy: binary and ASCII STL are read directly (no trimesh / open3d /
coal mesh loaders). Units are whatever the URDF says, i.e. metres; the
tests pin that against the kinematic link lengths.
"""

from __future__ import annotations

import functools
import math
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Default model density: ~7 mm between samples on the visible skin --
#: finer than a D455's footprint at working range is wasted on a 6-DoF fit.
DEFAULT_POINTS_PER_M2 = 20_000.0
#: Reference density for the exterior test (and the synthetic renderer):
#: ~1.6 mm spacing, dense enough that a 3x3-cell z-buffer has no holes.
DENSE_POINTS_PER_M2 = 400_000.0
_MAX_DENSE_PER_LINK = 400_000

_STL_RECORD = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


# ── STL ──────────────────────────────────────────────────────────────────


def read_stl(path) -> np.ndarray:
    """(N, 3, 3) float64 triangles from a binary or ASCII STL."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) >= 84:
        n = int(np.frombuffer(data[80:84], "<u4")[0])
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data[84:], _STL_RECORD, count=n)
            return rec["v"].astype(np.float64)
    head = data[:512].lstrip().lower()
    if head.startswith(b"solid") and b"facet" in data[:4096].lower() + data[-4096:].lower():
        return _read_ascii_stl(data.decode("ascii", errors="replace"), path)
    raise ValueError(f"{path}: not a valid binary STL (size does not match its triangle "
                     "count) and not an ASCII STL")


def _read_ascii_stl(text: str, path) -> np.ndarray:
    verts = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            verts.append([float(v) for v in parts[1:]])
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: ASCII STL with {len(verts)} vertices (not a multiple of 3)")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)


def triangle_normals_areas(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    area = 0.5 * norm
    nrm = np.zeros_like(cross)
    ok = norm > 0
    nrm[ok] = cross[ok] / norm[ok, None]
    return nrm, area


def orient_outward(tris: np.ndarray) -> np.ndarray:
    """Flip the winding if the mesh's signed volume is negative (inward
    normals would make every point back-facing from outside)."""
    vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()) / 6.0
    return tris[:, [0, 2, 1]] if vol < 0 else tris


def sample_triangles(tris: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """``n`` area-weighted uniform surface samples and their face normals."""
    nrm, area = triangle_normals_areas(tris)
    keep = area > 0
    tris, nrm, area = tris[keep], nrm[keep], area[keep]
    if n <= 0 or len(tris) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    cdf = np.cumsum(area)
    idx = np.searchsorted(cdf, rng.random(n) * cdf[-1], side="right")
    idx = np.minimum(idx, len(tris) - 1)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = (1.0 - s), s * (1.0 - r2), s * r2
    t = tris[idx]
    pts = a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2]
    return pts, nrm[idx].copy()


# ── exterior (skin) test ─────────────────────────────────────────────────


def _view_directions() -> np.ndarray:
    d = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                  if (x, y, z) != (0, 0, 0)], dtype=float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def exterior_mask(points, normals, *, ref_points=None, cell=None, tol=None) -> np.ndarray:
    """True for points visible from outside from at least one of 26
    directions (orthographic z-buffer per direction, 3x3-cell max filter so
    sampling gaps do not leak the interior). ``ref_points`` (default: the
    points themselves) build the z-buffers; ``cell`` defaults to 1.5x the
    mean sample spacing of the reference, ``tol`` to 2 cells."""
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    ref = pts if ref_points is None else np.asarray(ref_points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    if cell is None:
        # Mean spacing from the reference's bounding-box surface estimate is
        # unreliable for thin parts; use the nearest-neighbour proxy of a
        # uniform sample: spacing ~ sqrt(area / n), area ~ hull-ish proxy.
        ext = np.ptp(ref, axis=0)
        area_proxy = 2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2])
        cell = 1.5 * math.sqrt(max(area_proxy, 1e-12) / max(len(ref), 1))
    tol = 2.0 * cell if tol is None else float(tol)
    keep = np.zeros(len(pts), dtype=bool)
    for d in _view_directions():
        e1, e2 = _basis(d)
        ru, rv, rz = ref @ e1, ref @ e2, ref @ d
        u0, v0 = ru.min() - 2 * cell, rv.min() - 2 * cell
        nu = int((ru.max() - u0) / cell) + 4
        nv = int((rv.max() - v0) / cell) + 4
        grid = np.full((nu, nv), -np.inf)
        iu = ((ru - u0) / cell).astype(int)
        iv = ((rv - v0) / cell).astype(int)
        np.maximum.at(grid, (iu, iv), rz)
        g = grid.copy()   # 3x3 max filter
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if du or dv:
                    g = np.maximum(g, np.roll(np.roll(grid, du, 0), dv, 1))
        pu = np.clip(((pts @ e1 - u0) / cell).astype(int), 0, nu - 1)
        pv = np.clip(((pts @ e2 - v0) / cell).astype(int), 0, nv - 1)
        facing = nrm @ d > 0.0
        keep |= facing & (pts @ d >= g[pu, pv] - tol)
    return keep


# ── URDF geometry ────────────────────────────────────────────────────────


def _origin_T(elem) -> np.ndarray:
    from ..types import pose_to_transform

    if elem is None:
        return np.eye(4)
    xyz = [float(v) for v in (elem.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (elem.get("rpy") or "0 0 0").split()]
    return np.asarray(pose_to_transform(xyz + rpy), dtype=float)


def _resolve_mesh(uri: str, urdf_path: Path) -> Path:
    if uri.startswith("package://"):
        pkg, _, rel = uri[len("package://"):].partition("/")
        for parent in urdf_path.resolve().parents:
            if parent.name == pkg or _package_name(parent) == pkg:
                return parent / rel
        raise FileNotFoundError(f"{uri}: no ancestor of {urdf_path} is package {pkg!r}")
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    p = Path(uri)
    return p if p.is_absolute() else urdf_path.parent / p


def _package_name(d: Path) -> str | None:
    xml = d / "package.xml"
    if not xml.is_file():
        return None
    try:
        name = ET.parse(xml).getroot().find("name")
    except ET.ParseError:
        return None
    return None if name is None or name.text is None else name.text.strip()


def parse_collision_meshes(urdf_path, geometry: str = "collision") -> dict[str, list]:
    """{link: [(T_mesh2link, mesh_path, scale(3,)), ...]} for mesh geometry.

    Primitive shapes (box/cylinder/sphere) are not meshes and are skipped;
    the reBot URDF has none."""
    if geometry not in ("collision", "visual"):
        raise ValueError(f"geometry must be 'collision' or 'visual', got {geometry!r}")
    urdf_path = Path(urdf_path)
    root = ET.parse(urdf_path).getroot()
    out: dict[str, list] = {}
    for link in root.iter("link"):
        parts = []
        for g in link.findall(geometry):
            mesh = g.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()], float)
            if scale.size == 1:
                scale = np.repeat(scale, 3)
            parts.append((_origin_T(g.find("origin")),
                          _resolve_mesh(str(mesh.get("filename")), urdf_path), scale))
        if parts:
            out[str(link.get("name"))] = parts
    return out


def _link_triangles(parts) -> np.ndarray:
    chunks = []
    for T, path, scale in parts:
        if not path.is_file():
            raise FileNotFoundError(
                f"robot mesh {path} not found (meshes are needed for the surface model; for "
                "fetched assets run scripts/fetch_robot_assets.py)")
        tris = orient_outward(read_stl(path) * scale)
        chunks.append(tris @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(chunks) if chunks else np.empty((0, 3, 3))


@functools.lru_cache(maxsize=16)
def _dense_link_skin(path_str: str, mtime_ns: int, geometry: str, link: str,
                     density: float, seed: int, exterior_only: bool):
    """Dense exterior samples of one link, link frame (cached, read-only)."""
    parts = parse_collision_meshes(Path(path_str), geometry)[link]
    tris = _link_triangles(parts)
    _, area = triangle_normals_areas(tris)
    n = int(min(_MAX_DENSE_PER_LINK, max(1000, round(float(area.sum()) * density))))
    rng = np.random.default_rng([seed, zlib.crc32(link.encode())])
    pts, nrm = sample_triangles(tris, n, rng)
    if exterior_only and len(pts):
        keep = exterior_mask(pts, nrm)
        pts, nrm = pts[keep], nrm[keep]
    # Exterior area estimate: the kept fraction of a uniform sample.
    skin_area = float(area.sum()) * (len(pts) / max(n, 1))
    perm = np.random.default_rng([seed, 7, zlib.crc32(link.encode())]).permutation(len(pts))
    pts, nrm = pts[perm], nrm[perm]
    for a in (pts, nrm):
        a.flags.writeable = False
    return pts, nrm, skin_area


@functools.lru_cache(maxsize=32)
def _local_samples(path_str, mtime_ns, geometry, links, points_per_m2, seed, exterior_only,
                   dense_per_m2):
    pts, nrm, areas = [], [], []
    for link in links:
        dp, dn, skin = _dense_link_skin(path_str, mtime_ns, geometry, link, dense_per_m2, seed,
                                        exterior_only)
        k = int(min(len(dp), max(50, round(skin * points_per_m2))))
        pts.append(dp[:k])
        nrm.append(dn[:k])
        areas.append(skin)
    return tuple(pts), tuple(nrm), tuple(areas)


# ── the posed model ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurfaceCloud:
    points: np.ndarray    # (N, 3) base frame
    normals: np.ndarray   # (N, 3) unit, outward
    link_ids: np.ndarray  # (N,) index into RobotSurface.links


class RobotSurface:
    """Per-link surface samples (link frame) + the kinematics that pose them."""

    def __init__(self, kin, links, frame_ids, local_points, local_normals, *,
                 points_per_m2: float, skin_areas=()):
        self.kin = kin
        self.links = tuple(links)
        self.frame_ids = tuple(int(f) for f in frame_ids)
        self.local_points = local_points
        self.local_normals = local_normals
        self.points_per_m2 = float(points_per_m2)
        self.skin_areas = tuple(skin_areas)
        self._P = np.concatenate(local_points) if local_points else np.empty((0, 3))
        self._N = np.concatenate(local_normals) if local_normals else np.empty((0, 3))
        self._ids = np.concatenate([np.full(len(p), i, dtype=np.int32)
                                    for i, p in enumerate(local_points)]) \
            if local_points else np.empty(0, dtype=np.int32)

    @classmethod
    def from_kinematics(cls, kin, model_path, *, points_per_m2: float = DEFAULT_POINTS_PER_M2,
                        seed: int = 0, geometry: str = "collision", exterior_only: bool = True,
                        links=None, dense_per_m2: float = DENSE_POINTS_PER_M2) -> "RobotSurface":
        """``links=None``: every link with a mesh whose pose the measured
        joints determine; ``links="all"`` also includes passively driven
        ones (a renderer, never a calibration model). ``model_path`` must be
        the URDF ``kin`` was built from (USD models carry no mesh paths
        cascade can read)."""
        path = Path(model_path)
        if path.suffix.lower() != ".urdf":
            raise ValueError(f"surface model needs the arm's URDF (meshes), got {path.name}")
        meshes = parse_collision_meshes(path, geometry)
        everything = links == "all"
        if everything:
            links = None
        model = kin.model
        names, fids = [], []
        for fid, fr in enumerate(model.frames):
            if str(fr.type) != "BODY" or fr.name not in meshes:
                continue
            if links is not None and fr.name not in links:
                continue
            if links is None and not everything and int(fr.parentJoint) > int(kin.n):
                continue          # passive / unmeasured joint (e.g. finger slides)
            names.append(fr.name)
            fids.append(fid)
        if links is not None:
            missing = set(links) - set(names)
            if missing:
                raise ValueError(f"links {sorted(missing)} have no {geometry} mesh in {path}")
        if not names:
            raise ValueError(f"no {geometry} meshes found for the measured links of {path}")
        pts, nrm, areas = _local_samples(str(path.resolve()), path.stat().st_mtime_ns, geometry,
                                         tuple(names), float(points_per_m2), int(seed),
                                         bool(exterior_only), float(dense_per_m2))
        return cls(kin, names, fids, pts, nrm, points_per_m2=points_per_m2, skin_areas=areas)

    def __len__(self) -> int:
        return len(self._P)

    def link_poses(self, q, passive=None) -> np.ndarray:
        """(L, 4, 4) T_link2base at the (measured) joints ``q``. ``passive``
        sets the joints kin zero-pads (a renderer posing the finger slides);
        a calibration model never passes it."""
        kin = self.kin
        pin = kin._pin
        data = kin.data
        qf = kin._pad(np.asarray(q, dtype=float))
        if passive is not None:
            extra = np.asarray(passive, dtype=float).ravel()
            m = min(extra.size, kin.nq - kin.n)
            qf[kin.n: kin.n + m] = extra[:m]
        pin.forwardKinematics(kin.model, data, qf)
        pin.updateFramePlacements(kin.model, data)
        return np.stack([np.asarray(data.oMf[f].homogeneous) for f in self.frame_ids])

    def points_at(self, q, passive=None) -> SurfaceCloud:
        T = self.link_poses(q, passive)
        R = T[self._ids, :3, :3]
        t = T[self._ids, :3, 3]
        P = np.einsum("nij,nj->ni", R, self._P) + t
        N = np.einsum("nij,nj->ni", R, self._N)
        return SurfaceCloud(points=P, normals=N, link_ids=self._ids)
