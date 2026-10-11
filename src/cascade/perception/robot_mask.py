"""Robot-body masking for the occupancy map.

A depth camera sees the ARM. Integrated as-is, the robot's own links become
"obstacles" sitting 0 mm from the robot's own collision proxies, and the
clearance gate rejects every motion (measured: at the SO-101's home pose,
three of five link points read clearance 0.000-0.007 m against a 0.03 m
minimum -- the first grasp after the map went live was refused). Every
production ESDF stack masks the robot out before integration (nvblox /
Isaac ROS: URDF-driven body masking; cuRobo: robot-sphere carving). This is
cascade's version, built from what the stack already has: the arm's
kinematic link positions and a per-link radius.

`robot_mask(depth, K, T_base_cam, link_points, radii)` returns a boolean
(H,W) mask that is True on pixels whose 3-D point lies within `radius` of
any link SEGMENT (consecutive joint frames, plus the TCP). Those pixels are
zeroed before the frame ships to the bridge, so the ray-casting backends
treat them as "no measurement" -- not free, not occupied. Under-masking
leaves phantom obstacles; over-masking hides real ones next to the arm, so
radii are per-arm profile data (`arm.body_mask_radius_m`, default 0.06 m)
and the held object is NOT masked (it is a real obstacle for other links).

Capsules fit a thin arm, not a wide gripper: on the real reBot (2026-10-11)
they left the open jaws and the wrist D455F in the map, 2.8 cm from the
chain. `MeshBodyMask` (below, arm profile `body_mask: mesh`) masks the
arm's URDF silhouette instead -- every link, the fingers at the measured
opening, profile attachments, a metric margin -- and refuses to build rather
than fall back to capsules when a mesh cannot be read. docs/NVBLOX.md
"Robot body mask on real arms" has the measured before/after.
"""

from __future__ import annotations

import numpy as np


def _segment_distances(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (N,3) to the segment a-b."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


def robot_mask(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
               link_points: np.ndarray, radius_m: float | np.ndarray = 0.06,
               stride: int = 1) -> np.ndarray:
    """Boolean (H,W) mask of depth pixels that belong to the robot body.

    link_points: (L,3) base-frame points along the arm (joint frames in
    order, TCP last). Segments are consecutive pairs. radius_m: scalar or
    (L-1,) per-segment radius.
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    mask = np.zeros((H, W), dtype=bool)
    valid = depth > 0
    if not valid.any() or link_points is None or len(link_points) < 1:
        return mask
    ys, xs = np.nonzero(valid)
    zs = depth[ys, xs]
    K = np.asarray(K, dtype=np.float64)
    pts_cam = np.stack([(xs - K[0, 2]) / K[0, 0] * zs, (ys - K[1, 2]) / K[1, 1] * zs, zs], -1)
    T = np.asarray(T_base_cam, dtype=np.float64)
    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    lp = np.asarray(link_points, dtype=np.float64).reshape(-1, 3)
    radii = np.broadcast_to(np.asarray(radius_m, dtype=np.float64), (max(len(lp) - 1, 1),))
    hit = np.zeros(len(pts_base), dtype=bool)
    if len(lp) == 1:
        hit |= np.linalg.norm(pts_base - lp[0], axis=1) <= radii[0]
    for i in range(len(lp) - 1):
        # cheap AABB reject before the exact segment distance
        lo = np.minimum(lp[i], lp[i + 1]) - radii[i]
        hi = np.maximum(lp[i], lp[i + 1]) + radii[i]
        cand = np.all((pts_base >= lo) & (pts_base <= hi), axis=1)
        if not cand.any():
            continue
        d = _segment_distances(pts_base[cand], lp[i], lp[i + 1])
        idx = np.nonzero(cand)[0][d <= radii[i]]
        hit[idx] = True
    mask[ys[hit], xs[hit]] = True
    return mask


def arm_link_points(kin, q: np.ndarray) -> np.ndarray:
    """Joint-frame positions + TCP for an arm at q -- the polyline the mask
    thickens. Base frame."""
    links = np.asarray(kin.link_positions(np.asarray(q, dtype=float)))
    tcp = kin.fk(np.asarray(q, dtype=float))[:3, 3]
    return np.vstack([links, tcp[None, :]])


# ── the mesh body mask (real arms, `body_mask: mesh`) ──────────────────────


class BodyMaskUnavailable(RuntimeError):
    """No usable joint state for the mesh body mask: the frame must not be
    integrated (an unknown pose is not an empty scene)."""


#: Shipped metric inflation of the mesh body mask (arm profile
#: `body_mask.inflate_m`), metres, applied to every convex piece at load.
#: Chosen on the real reBot capture (2026-10-11, fixed D455F, markerless
#: hand-eye, 24 frames at 640x360, docs/NVBLOX.md): with 2 px of dilation,
#: 1 / 1.5 / 2 cm left 108 / 37 / 0 unmasked depth points per frame within
#: 3 cm of the URDF surface (the 6 cm capsules: 3859). 2 cm is also below
#: `min_clearance_m` (3 cm): whatever the margin hides lies inside the band
#: the clearance gate already keeps the arm away from.
DEFAULT_BODY_INFLATE_M = 0.02

#: Shipped pixel dilation of the mesh body mask (arm profile
#: `body_mask.dilate_px`). It absorbs what a metric margin does not: the
#: rasterisation edge and the D455F's flying pixels, which sit on the
#: silhouette's border in the IMAGE whatever their depth. On the real
#: capture, 2 px roughly halved the residual left by the margin alone.
DEFAULT_BODY_DILATE_PX = 2

_BODY_MASK_KEYS = {"mode", "inflate_m", "dilate_px", "cell_m", "attachments"}
_ATTACHMENT_KEYS = {"name", "link", "box", "xyz", "rpy"}


class MeshBodyMask:
    """The arm's URDF geometry rasterized into a depth frame's pixels.

    WHY (measured on the real reBot rig, 2026-10-11): the capsule mask above
    tests each pixel's 3D point against `radius_m` around the joint chain. A
    gripper is wide where the chain is thin (the open jaws ~14 cm across,
    the wrist D455F on top of them), so 108 occupied voxels stayed on the
    gripper and wrist, the nearest 2.8 cm from the chain, under the 3 cm
    clearance. This mask draws the robot's SILHOUETTE instead: every convex
    piece of every link (LinkGeometry: URDF collision meshes, the profile's
    box attachments, inflated once at load by a metric margin), posed by FK
    at the current joint state, projected and filled. Two consequences
    beyond the jaws:

    - it does not read the depth values, so a wrong depth on a robot pixel
      (stereo speckle on the jaws, a mismatch at the wrist camera's close
      range) is masked too; the capsule test would integrate it;
    - it masks the robot's pixels, not the space around the chain, so a
      real object 7 cm from the wrist is no longer hidden by a 6 cm capsule.

    `poses(state) -> [{link: 4x4}]` is `KinematicsLinkPoses` in production:
    passive finger joints at the measured opening, or swept over their whole
    travel when the opening is unknown (more mask, never less). `state_fn()
    -> RobotState | None` is read on every call; None, a non-finite q or
    fewer than `n_joints` values raise BodyMaskUnavailable and a failing read propagates, because the
    occupancy map must then integrate NOTHING (its fail-closed contract).

    `__call__(depth, K, T_base_cam)` works for any camera: eye-to-hand
    (a fixed T) and eye-in-hand (T = FK(q) @ X, the wrist camera, whose
    jaws straddle the near plane and are clipped there). `last` records the
    latest call: pixels, milliseconds, how many configurations were posed
    (2 = fingers swept).
    """

    def __init__(self, geometry, poses, state_fn, *, base_T=None, n_joints: int | None = None,
                 dilate_px: int = DEFAULT_BODY_DILATE_PX, name: str = "arm"):
        if isinstance(dilate_px, bool) or not isinstance(dilate_px, int) or dilate_px < 0:
            raise ValueError(f"dilate_px must be a non-negative integer, got {dilate_px!r}")
        self.geometry = geometry
        self._poses = poses
        self._state_fn = state_fn
        self.base_T = None if base_T is None else np.asarray(base_T, dtype=np.float64).reshape(4, 4)
        self.dilate_px = int(dilate_px)
        self.n_joints = None if n_joints is None else int(n_joints)
        self.name = str(name)
        self.last: dict | None = None

    def __call__(self, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
        import time

        from .link_mask import posed_pieces, rasterize_pieces

        t0 = time.perf_counter()
        state = self._state_fn()
        if state is None:
            raise BodyMaskUnavailable(f"{self.name}: no joint state (the arm is in standby or not readable)")
        q = np.asarray(getattr(state, "q", None), dtype=np.float64).reshape(-1)
        if q.size == 0 or q.size < (self.n_joints or 0) or not np.isfinite(q).all():
            raise BodyMaskUnavailable(f"{self.name}: unusable joint state q={q.tolist()}")
        pose_sets = self._poses(state)
        mask = rasterize_pieces(posed_pieces(self.geometry, pose_sets, self.base_T), K, T_base_cam,
                                np.shape(depth)[:2], self.dilate_px)
        self.last = {"pixels": int(np.count_nonzero(mask)), "configs": len(pose_sets),
                     "ms": (time.perf_counter() - t0) * 1e3}
        return mask


# ── configuration: the arm profile's `body_mask` ────────────────────────────


def _bad(msg: str) -> ValueError:
    return ValueError(f"arm profile body_mask: {msg}")


def _as_dict(block) -> dict:
    if hasattr(block, "as_dict"):
        return block.as_dict()
    if isinstance(block, dict):
        return dict(block)
    raise _bad(f"expected `capsule`, `mesh` or a mapping, got {block!r}")


def _vector(value, n: int, what: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _bad(f"{what} must be {n} numbers, got {value!r}") from None
    if out.shape != (n,) or not np.isfinite(out).all():
        raise _bad(f"{what} must be {n} finite numbers, got {value!r}")
    return out


def body_mask_settings(acfg) -> dict | None:
    """The arm profile's `body_mask`, parsed: None for the capsule mask (the
    default, `body_mask` absent or `capsule`), else {inflate_m, dilate_px,
    cell_m, attachments: [(name, link, size, T_link_box)]}.

    Accepted forms: `body_mask: mesh` (every default) or a mapping with
    `mode: mesh` plus any of `inflate_m` (metres >= 0), `dilate_px`
    (integer >= 0), `cell_m` (LinkGeometry's mesh cell, metres > 0) and
    `attachments` (a list of {name, link, box: [sx, sy, sz], xyz, rpy}:
    rigid boxes posed in a link frame, for parts the URDF lacks). Any other
    key or value raises ValueError: a typo in a safety mask must not become
    a silent default.
    """
    block = acfg.get("body_mask") if hasattr(acfg, "get") else None
    if block is None or block == "capsule":
        return None
    if block == "mesh":
        block = {"mode": "mesh"}
    elif isinstance(block, str) or not (hasattr(block, "as_dict") or isinstance(block, dict)):
        raise _bad(f"expected `capsule`, `mesh` or a mapping, got {block!r}")
    block = _as_dict(block)
    unknown = set(block) - _BODY_MASK_KEYS
    if unknown:
        raise _bad(f"unknown keys {sorted(unknown)} (allowed: {sorted(_BODY_MASK_KEYS)})")
    mode = block.get("mode", "mesh")
    if mode == "capsule":
        return None
    if mode != "mesh":
        raise _bad(f"mode must be `capsule` or `mesh`, got {mode!r}")
    inflate = block.get("inflate_m", DEFAULT_BODY_INFLATE_M)
    if isinstance(inflate, bool) or not isinstance(inflate, (int, float)) or not np.isfinite(inflate) \
            or inflate < 0:
        raise _bad(f"inflate_m must be a number of metres >= 0, got {inflate!r}")
    dilate = block.get("dilate_px", DEFAULT_BODY_DILATE_PX)
    if isinstance(dilate, bool) or not isinstance(dilate, int) or dilate < 0:
        raise _bad(f"dilate_px must be an integer >= 0, got {dilate!r}")
    from .link_mask import DEFAULT_CELL_M

    cell = block.get("cell_m", DEFAULT_CELL_M)
    if isinstance(cell, bool) or not isinstance(cell, (int, float)) or not np.isfinite(cell) or cell <= 0:
        raise _bad(f"cell_m must be a positive number of metres, got {cell!r}")
    raw = block.get("attachments") or []
    if not isinstance(raw, list):
        raise _bad(f"attachments must be a list, got {raw!r}")
    from ..types import pose_to_transform

    attachments = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise _bad(f"attachments[{i}] must be a mapping, got {item!r}")
        unknown = set(item) - _ATTACHMENT_KEYS
        if unknown:
            raise _bad(f"attachments[{i}]: unknown keys {sorted(unknown)} (allowed: {sorted(_ATTACHMENT_KEYS)})")
        name = str(item.get("name") or f"attachment{i}")
        link = item.get("link")
        if not isinstance(link, str) or not link:
            raise _bad(f"attachment {name!r} needs the `link` it is bolted to")
        if "box" not in item:
            raise _bad(f"attachment {name!r} needs a `box: [sx, sy, sz]` (metres)")
        size = _vector(item["box"], 3, f"attachment {name!r} box")
        if (size <= 0).any():
            raise _bad(f"attachment {name!r} box sizes must be positive, got {item['box']!r}")
        xyz = _vector(item.get("xyz", [0.0, 0.0, 0.0]), 3, f"attachment {name!r} xyz")
        rpy = _vector(item.get("rpy", [0.0, 0.0, 0.0]), 3, f"attachment {name!r} rpy")
        attachments.append((name, link, tuple(float(v) for v in size),
                            pose_to_transform(list(xyz) + list(rpy))))
    return {"inflate_m": float(inflate), "dilate_px": int(dilate), "cell_m": float(cell),
            "attachments": attachments}


def build_mesh_body_mask(acfg, kin, state_fn, *, name: str) -> MeshBodyMask | None:
    """The occupancy mesh body mask for one arm profile, or None when the
    profile keeps the capsule mask (nothing is loaded, nothing is read).

    Geometry: the profile `model`'s URDF collision meshes (LinkGeometry,
    `cell_m` pieces), plus `attachments`, inflated by `inflate_m`. Fingers:
    `KinematicsLinkPoses` with the profile's `gripper.open_pos/closed_pos`
    (measured opening -> finger joints; unknown -> swept). Table frame:
    `base_pose`.

    CAPSULE FALLBACK: refused. LinkGeometry replaces a mesh it cannot read
    (missing, a git-lfs pointer never fetched, an unsupported format) by a
    capsule around the link, which is right for fusion's detection gate but
    is precisely the occupancy mask that left the reBot's jaws in the map.
    A profile that asked for `body_mask: mesh` therefore fails to build,
    naming the links and the reason, and the operator either fetches the
    meshes (`git lfs pull`) or sets `body_mask: capsule` knowingly. Never a
    silent downgrade. The same holds for an attachment on a link the model
    does not have.
    """
    settings = body_mask_settings(acfg)
    if settings is None:
        return None
    import sys

    from .link_mask import KinematicsLinkPoses, LinkGeometry

    model = acfg.get("model")
    if not model:
        raise _bad(f"mesh: arm {name!r} has no `model` to take its geometry from")
    geometry = LinkGeometry.from_urdf(model, cell_m=settings["cell_m"])
    fallbacks = {link: why for link, why in geometry.sources.items() if why.startswith("capsule")}
    if fallbacks:
        detail = "; ".join(f"{link!r}: {why}" for link, why in sorted(fallbacks.items()))
        raise _bad(f"mesh for arm {name!r} cannot load the geometry of {len(fallbacks)} link(s) "
                   f"({detail}). Refusing a capsule fallback for an occupancy mask: fetch the meshes "
                   f"(git lfs pull) or set `body_mask: capsule` explicitly.")
    geometry = geometry.with_boxes(settings["attachments"]).inflated(settings["inflate_m"])
    grip = acfg.get("gripper") or {}
    try:
        poses = KinematicsLinkPoses(kin, geometry.links, gripper_open=grip.get("open_pos"),
                                    gripper_closed=grip.get("closed_pos"))
    except KeyError as exc:
        raise _bad(f"mesh for arm {name!r}: {exc.args[0] if exc.args else exc} "
                   f"(an attachment's `link` must be a link of {model})") from None
    pose = acfg.get("base_pose")
    base_T = None
    if pose is not None:
        from ..types import pose_to_transform

        base_T = pose_to_transform(pose)
    body = MeshBodyMask(geometry, poses, state_fn, base_T=base_T, n_joints=int(kin.n),
                        dilate_px=settings["dilate_px"], name=name)
    boxes = ", ".join(a[0] for a in settings["attachments"]) or "none"
    print(f"[cascade] occupancy body mask: {name}: URDF mesh, {len(geometry.links)} links, "
          f"{geometry.n_pieces()} pieces, inflate {settings['inflate_m']} m, dilate {settings['dilate_px']} px, "
          f"attachments: {boxes}", file=sys.stderr)
    return body
