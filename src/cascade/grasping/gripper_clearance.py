"""Forearm/wrist/gripper clearance vet for angled grasp candidates (B73, opt-in).

B45 (the `*_reach` arm profiles) lets the analytic planner lean its approach
30 / 45 / 90 degrees away from the base. The harness then vets each candidate
with link PROXIES -- joint origins -- and during the descent it exempts every
proxy inside the grasp cylinder (radius `grasp.exempt_radius_m`, floor
table_z - 0.06), because a top-down descent legitimately puts the wrist over
the object. A tilted approach brings the housing, the wrist motors and the far
finger near the support surface and the target instead, which the proxies
cannot see. MEASURED (B45 grid, `rebot_rs_reach`): a side grasp at 5 cm that
IK and every harness check admit drives `link5`'s collision mesh 3.1 mm below
the table.

This module places the arm's collision geometry -- one convex hull per
connected component of each collision mesh from `link3` (forearm) onward,
`assets/grasp_geometry/rebot_rs_clearance_hulls.json`, built offline from the
URDF by `scripts/build_gripper_clearance_hulls.py` -- at every pose of the
approach the runtime will execute (the joint-space line q_pre -> q_grasp,
sampled so no hull vertex moves more than `APPROACH_STEP_M` between two vetted
poses), with the fingers open at the commanded pre-grasp gap, and measures:

* support plane: lowest hull vertex minus the support height. EXACT for the
  mesh (the lowest point of a hull is one of its vertices, and every hull
  vertex is a mesh vertex);
* observed box: a separating-axis LOWER BOUND of the distance from each hull
  to the target's observed box (`target_box`), over the box's three axes and
  the hull's face normals. Lower bound = never optimistic. The jaw gap needs
  no exemption: no collision hull is there, and the planner sizes the
  opening so the open fingers clear the box.

A candidate is refused when either clearance is below the margin,
`margin_m` = `grasp.width_pad_m` / 2: the clearance the planner already gives
each open jaw against the observed object (7.5 mm on the reBot profiles).
Between two vetted poses no vertex moves more than `APPROACH_STEP_M`, so the
continuous approach keeps at least margin - APPROACH_STEP_M / 2.

Opt-in (`grasp.angled_clearance_vet`, the `*_reach` profiles) and only for
the analytic planner's tilted candidates (`obb_grasp.AngledGrasp`). It
refuses a CANDIDATE at selection time, like the existing IK/harness pre-vet;
the next candidate is tried, and the harness stays the motion authority.
Geometry that cannot be loaded or bound fails closed: every angled candidate
is refused with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = ROOT / "assets" / "grasp_geometry" / "rebot_rs_clearance_hulls.json"
SCHEMA = "cascade.gripper_clearance_hulls/1"
#: no hull vertex moves more than this between two vetted approach poses
APPROACH_STEP_M = 0.001
#: the runtime's descent samples (`skill_grasp_object._vet`), plus the pregrasp
_SEED_SAMPLES = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
#: an approach that needs more poses than this is refused (fail closed)
_MAX_SAMPLES = 4096
#: hull-normal separating axes are evaluated for components whose box-axis
#: gap is below max(margin, this): the reported box clearance is the tight
#: lower bound up to here, a looser (still valid) one beyond it
_REPORT_RANGE_M = 0.02


class ClearanceGeometryError(ValueError):
    """The clearance hulls cannot be loaded or bound to this arm model."""


@dataclass(frozen=True)
class Box:
    """An oriented box: centre (3,), axes (3, 3) as columns, half extents (3,)."""

    centre: np.ndarray
    axes: np.ndarray
    half: np.ndarray

    def as_evidence(self) -> dict:
        return {"centre": np.asarray(self.centre, dtype=float).tolist(),
                "axes": np.asarray(self.axes, dtype=float).tolist(),
                "half": np.asarray(self.half, dtype=float).tolist()}


@dataclass
class ClearanceResult:
    ok: bool
    reason: str | None
    plane_clearance_m: float
    plane_part: str
    plane_s: float
    box_clearance_m: float | None
    box_part: str | None
    box_s: float | None
    samples: tuple

    def as_evidence(self) -> dict:
        return {"decision": "admitted" if self.ok else "refused", "reason": self.reason,
                "plane_clearance_m": self.plane_clearance_m, "plane_part": self.plane_part,
                "plane_s": self.plane_s, "box_clearance_m": self.box_clearance_m,
                "box_part": self.box_part, "box_s": self.box_s, "samples": len(self.samples)}


def margin_m(gcfg) -> float:
    """The vet's margin: `grasp.width_pad_m` / 2 (the planner's default pad
    0.015 when unset), the clearance the planner gives each open jaw."""
    return float(gcfg.get("width_pad_m", 0.015)) / 2.0


def jaw_gap_m(width_m, max_width_m, open_margin_m=None) -> float:
    """The jaw opening the runtime commands before the approach: fully open
    (`max_width_m`, the framework's linear width map) unless the profile sets
    `gripper.pregrasp_open_margin_m` (then width + margin, never wider)."""
    if open_margin_m is None:
        return float(max_width_m)
    return min(float(width_m) + float(open_margin_m), float(max_width_m))


def object_box(points, axes, extent, position, table_z) -> Box:
    from .obb_grasp import object_vertical_span

    axes = np.asarray(axes, dtype=float)
    extent = np.asarray(extent, dtype=float)
    centre = np.asarray(position, dtype=float)
    # Jaw axis: the narrowest horizontal footprint axis (the planner's rule).
    horiz = []
    for i in range(3):
        a = axes[:, i].copy()
        a[2] = 0.0
        n = float(np.linalg.norm(a))
        if n < 0.3:
            continue
        horiz.append((float(extent[i]) * n, a / n))
    horiz.sort(key=lambda t: t[0])
    u = horiz[0][1] if horiz else np.array([1.0, 0.0, 0.0])
    v = np.cross([0.0, 0.0, 1.0], u)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    corners = (signs * (extent / 2.0)) @ axes.T
    bottom, top = object_vertical_span(np.asarray(points, dtype=float), float(table_z))
    return Box(centre=np.array([centre[0], centre[1], (bottom + top) / 2.0]),
               axes=np.column_stack([u, v, [0.0, 0.0, 1.0]]),
               half=np.array([float(np.abs(corners @ u).max()), float(np.abs(corners @ v).max()),
                              (top - bottom) / 2.0]))


def target_box(fix, table_z) -> Box:
    """The target's observed box: upright, yawed to the planner's jaw axis,
    enclosing the fix's OBB footprint, from the support plane (when the cloud
    does not resolve the bottom -- the planner's rule) to the observed top."""
    return object_box(fix.points, fix.axes, fix.extent, fix.position, table_z)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _load(path: str, model: str, stamp: tuple) -> ClearanceGeometry:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ClearanceGeometryError(f"clearance hulls unreadable: {exc}") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ClearanceGeometryError(f"clearance hulls: unexpected schema {data.get('schema')!r}"
                                     if isinstance(data, dict) else "clearance hulls: not an object")
    sources = data.get("sources") or {}
    model_path = Path(model).resolve()
    urdf = [k for k in sources if k.endswith(".urdf")]
    if len(urdf) != 1 or (ROOT / urdf[0]).resolve() != model_path:
        raise ClearanceGeometryError(
            f"no clearance hulls for arm model {model_path.name} (hulls describe {urdf})")
    for rel, digest in sources.items():
        try:
            actual = _sha256(ROOT / rel)
        except OSError:
            actual = None
        if actual != digest:
            raise ClearanceGeometryError(f"clearance hulls are stale: {rel} changed; "
                                         "re-run scripts/build_gripper_clearance_hulls.py")
    return ClearanceGeometry(data, _sha256(Path(path)))


def load_geometry(model_path, path: Path = GEOMETRY) -> ClearanceGeometry:
    """The clearance hulls for `model_path` (the arm's URDF). Raises
    ClearanceGeometryError for another model, a changed mesh or a bad file."""
    p = Path(path)
    try:
        st = p.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError as exc:
        raise ClearanceGeometryError(f"clearance hulls missing: {exc}") from None
    return _load(str(p.resolve()), str(model_path), stamp)


class ClearanceGeometry:
    """Hull vertices per link (link frame) and per-component SAT axes."""

    def __init__(self, data: dict, sha256: str):
        self.sha256 = sha256
        self.links = tuple(data["links"])
        self.vertices, self.starts, self.axes = {}, {}, {}
        for name, components in data["links"].items():
            verts = [np.asarray(c["vertices"], dtype=float).reshape(-1, 3) for c in components]
            if not verts or any(len(v) < 4 for v in verts):
                raise ClearanceGeometryError(f"clearance hulls: link {name!r} has a degenerate component")
            self.vertices[name] = np.concatenate(verts)
            self.starts[name] = np.cumsum([0, *[len(v) for v in verts[:-1]]])
            self.axes[name] = [np.asarray(c["axes"], dtype=float).reshape(-1, 3) for c in components]
        self.finger_joints = tuple((j["name"], float(j["lower_m"]), float(j["upper_m"]))
                                   for j in data["finger_joints"])


class ApproachClearance:
    """The clearance vet bound to one arm's kinematics (same model as IK)."""

    def __init__(self, geometry: ClearanceGeometry, kin):
        self.geometry = geometry
        self.kin = kin
        model = kin.model
        for name in geometry.links:
            if model.getFrameId(name) >= len(model.frames):
                raise ClearanceGeometryError(f"link {name!r} not in the arm model")
        for name, _, _ in geometry.finger_joints:
            if model.getJointId(name) >= model.njoints:
                raise ClearanceGeometryError(f"finger joint {name!r} not in the arm model")

    def _finger_values(self, jaw_gap: float) -> dict:
        # The URDF fingers' inner faces are 2 q (+0.1 mm) apart.
        return {name: float(np.clip(jaw_gap / 2.0, lo, hi))
                for name, lo, hi in self.geometry.finger_joints}

    def poses(self, q, jaw_gap: float) -> dict:
        """{link: (R, t)} base-frame placements at q, fingers at the gap."""
        frames = self.kin.frame_poses(q, self.geometry.links, self._finger_values(jaw_gap))
        return {name: (T[:3, :3], T[:3, 3]) for name, T in zip(self.geometry.links, frames)}

    def placed(self, q, jaw_gap: float) -> dict:
        """{link: (n, 3) base-frame hull vertices} at q."""
        return {name: self.geometry.vertices[name] @ R.T + t
                for name, (R, t) in self.poses(q, jaw_gap).items()}

    def _approach(self, q_pre, q_grasp, jaw_gap):
        """{s: (poses, placed)} over [0, 1], bisected until no hull vertex
        moves more than APPROACH_STEP_M between consecutive samples."""
        q_pre = np.asarray(q_pre, dtype=float)
        dq = np.asarray(q_grasp, dtype=float) - q_pre
        cache = {}

        def at(s):
            if s not in cache:
                poses = self.poses(q_pre + s * dq, jaw_gap)
                cache[s] = (poses, {n: self.geometry.vertices[n] @ R.T + t for n, (R, t) in poses.items()})
            return cache[s]

        samples = list(_SEED_SAMPLES)
        while True:
            refined = [samples[0]]
            for a, b in zip(samples, samples[1:]):
                pa, pb = at(a)[1], at(b)[1]
                chord = max(float(np.linalg.norm(pb[n] - pa[n], axis=1).max()) for n in pa)
                if chord > APPROACH_STEP_M:
                    refined.append((a + b) / 2.0)
                refined.append(b)
            if len(refined) == len(samples):
                return [(s, *at(s)) for s in samples]
            if len(refined) > _MAX_SAMPLES:
                raise ClearanceGeometryError(
                    f"approach needs more than {_MAX_SAMPLES} poses at {APPROACH_STEP_M * 1000:.0f} mm")
            samples = refined

    def check(self, q_pre, q_grasp, *, support_z: float, box: Box | None, margin_m: float,
              jaw_gap_m: float) -> ClearanceResult:
        """Clearance of the hulls to the support plane and the observed box
        over the approach q_pre -> q_grasp (pregrasp s = 0, grasp s = 1)."""
        margin = float(margin_m)
        plane = (np.inf, "", 0.0)
        boxed = (np.inf, None, None)
        samples = self._approach(q_pre, q_grasp, float(jaw_gap_m))
        for s, poses, placed in samples:
            for name, verts in placed.items():
                low = float(verts[:, 2].min()) - float(support_z)
                if low < plane[0]:
                    plane = (low, name, s)
                if box is not None:
                    gap = _box_clearance(verts, self.geometry.starts[name], self.geometry.axes[name],
                                         poses[name][0], box, max(margin, _REPORT_RANGE_M))
                    if gap < boxed[0]:
                        boxed = (gap, name, s)
        reason = None
        if plane[0] < margin:
            reason = (f"{plane[1]} clears the support plane by {plane[0] * 1000:+.1f} mm at approach "
                      f"s={plane[2]:.2f} (< {margin * 1000:.1f} mm margin)")
        elif box is not None and boxed[0] < margin:
            reason = (f"{boxed[1]} clears the target's observed box by {boxed[0] * 1000:+.1f} mm at "
                      f"approach s={boxed[2]:.2f} (< {margin * 1000:.1f} mm margin)")
        return ClearanceResult(
            ok=reason is None, reason=reason, plane_clearance_m=float(plane[0]),
            plane_part=plane[1], plane_s=float(plane[2]),
            box_clearance_m=None if box is None else float(boxed[0]),
            box_part=boxed[1], box_s=None if box is None else float(boxed[2]),
            samples=tuple(s for s, _, _ in samples))


def _box_clearance(verts, starts, axes, R, box: Box, refine_below: float) -> float:
    """Smallest separating-axis lower bound over a link's hull components."""
    Bc, Ba, h = (np.asarray(box.centre, dtype=float), np.asarray(box.axes, dtype=float),
                 np.asarray(box.half, dtype=float))
    local = (verts - Bc) @ Ba                       # vertices in the box frame
    lo = np.minimum.reduceat(local, starts, axis=0)
    hi = np.maximum.reduceat(local, starts, axis=0)
    gaps = np.max(np.maximum(lo - h, -h - hi), axis=1)   # box face normals
    ends = [*starts[1:], len(verts)]
    for i in np.flatnonzero(gaps < refine_below):
        n = axes[i] @ R.T @ Ba                      # hull face normals, box frame
        proj = local[starts[i]:ends[i]] @ n.T
        r = np.abs(n) @ h                           # box half-width along each axis
        gaps[i] = max(float(gaps[i]),
                      float(np.max(np.maximum(proj.min(axis=0) - r, -r - proj.max(axis=0)))))
    return float(gaps.min())
