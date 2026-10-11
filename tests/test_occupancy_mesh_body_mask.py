"""Occupancy body mask from the arm's URDF MESHES (real, non-Isaac arms).

Measured on the real reBot rig (2026-10-11, calibrated fixed D455F, arm at
its rest pose q = 0): the production occupancy body mask -- capsules of
`body_mask_radius_m` (6 cm) around the joint polyline -- left 108 occupied
voxels on the gripper and wrist (the open jaws are ~14 cm wide, the wrist
D455F sits on top of them), the nearest 2.8 cm from the arm chain, under
`min_clearance_m` 0.03. The robot's own body polluted the map and motion from
rest could be refused.

`body_mask: mesh` in an arm profile rasterizes the arm's URDF geometry (every
link, the fingers at the measured opening, plus profile-declared rigid
attachments such as the wrist camera) into each depth frame's mask before
integration, with a metric inflation and a pixel dilation. These tests pin:

- the geometry operations (metric inflation, attachments) against independent
  geometric truth;
- the mask against depth RENDERED from the test's own triangles (a z-buffer
  written here, not the code under test), including the jaw case the capsule
  misses and an eye-in-hand camera whose jaws straddle the near plane;
- the fail-closed contract: no joint state -> nothing is integrated;
- the profile parsing and the refusal to fall back silently to capsules.
"""
from __future__ import annotations

import typing

import numpy as np
import pytest

from cascade.perception import link_mask as lm


def _hull_contains(points: np.ndarray, query: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Independent containment truth: scipy's hull half-spaces."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(points).equations
    return (query @ eq[:, :3].T + eq[:, 3] <= tol).all(axis=1)


def _unit_dirs(n: int, seed: int = 0) -> np.ndarray:
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cube_geometry(size=0.04) -> lm.LinkGeometry:
    corners = lm._box_points((size, size, size))
    sliver = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.002, 0], [0, 0, 0.002]], dtype=float)
    return lm.LinkGeometry(pieces={"cube": (corners,), "sliver": (sliver,)},
                           sources={"cube": "box", "sliver": "mesh"})


# ── slice 1: metric inflation ─────────────────────────────────────────────


@pytest.mark.parametrize("margin", [0.005, 0.02])
def test_inflation_contains_every_point_within_the_margin_of_each_piece(margin):
    geo = _cube_geometry()
    fat = geo.inflated(margin)
    assert fat.links == geo.links and fat.sources == geo.sources
    for link in geo.links:
        for piece, fat_piece in zip(geo.pieces[link], fat.pieces[link], strict=True):
            # every point at exactly `margin` from any vertex (the Minkowski
            # sum's extreme points) and the piece itself must lie inside
            probe = (piece[:, None, :] + margin * _unit_dirs(400)[None]).reshape(-1, 3)
            assert _hull_contains(fat_piece, np.concatenate([piece, probe])).all()


def test_inflation_is_bounded_by_the_circumscribed_cover():
    """A metric inflation, not unbounded bloat: no inflated vertex lies more
    than the cover's circumradius (1.26 x margin for an icosahedron whose
    inradius is the margin) from the original piece."""
    geo, margin = _cube_geometry(), 0.01
    fat = geo.inflated(margin)
    for link in geo.links:
        for piece, fat_piece in zip(geo.pieces[link], fat.pieces[link], strict=True):
            gap = np.linalg.norm(fat_piece[:, None, :] - piece[None], axis=2).min(axis=1)
            assert gap.max() <= 1.26 * margin + 1e-12


def test_zero_inflation_is_the_identity_and_bad_margins_are_refused():
    geo = _cube_geometry()
    assert geo.inflated(0.0) is geo
    for bad in (-0.01, float("nan"), float("inf"), "1cm", True):
        with pytest.raises(ValueError, match="margin"):
            geo.inflated(bad)


# ── slice 2: rigid attachments the URDF does not model ──────────────────


def _rpy_T(xyz, rpy) -> np.ndarray:
    from cascade.types import pose_to_transform

    return pose_to_transform(list(xyz) + list(rpy))


def test_a_box_attachment_becomes_a_piece_of_its_link_in_the_link_frame():
    """The reBot URDF has no wrist D455F: its body must be declarable on the
    link it is bolted to, posed in that link's frame."""
    geo = _cube_geometry()
    T = _rpy_T((-0.09, 0.0, 0.06), (0.0, 0.0, np.pi / 2))
    out = geo.with_boxes([("wrist_cam", "cube", (0.06, 0.14, 0.05), T)])
    assert len(out.pieces["cube"]) == len(geo.pieces["cube"]) + 1
    added = out.pieces["cube"][-1]
    # independent truth: the box's 8 corners, rotated and moved by hand
    half = np.array([0.03, 0.07, 0.025])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    want = corners @ T[:3, :3].T + T[:3, 3]
    assert sorted(map(tuple, np.round(added, 9))) == sorted(map(tuple, np.round(want, 9)))
    assert "wrist_cam" in out.sources["cube"]
    assert geo.pieces["cube"] is not out.pieces["cube"], "the original geometry must not change"


def test_a_box_on_a_link_without_geometry_creates_that_link():
    geo = _cube_geometry()
    out = geo.with_boxes([("mount", "flange", (0.02, 0.02, 0.02), np.eye(4))])
    assert "flange" in out.links and len(out.pieces["flange"]) == 1
    assert "mount" in out.sources["flange"]


@pytest.mark.parametrize("size", [(0.0, 0.1, 0.1), (-0.01, 0.1, 0.1), (np.nan, 0.1, 0.1), (0.1, 0.1)])
def test_a_degenerate_box_is_refused(size):
    with pytest.raises(ValueError, match="box"):
        _cube_geometry().with_boxes([("bad", "cube", size, np.eye(4))])


# ── slice 3: the per-frame mesh mask against ray-cast depth ──────────────
#
# A synthetic arm of boxes whose gripper is WIDER than the capsule: a 16 cm
# palm and two jaws 12 cm apart (the reBot's open jaws are ~14 cm wide),
# posed by a hand-written FK. The truth is a ray caster written here (slab
# test per pixel against every posed box), independent of the projection +
# hull fill under test.

W, H = 640, 360
FX = 320.0
K = np.array([[FX, 0.0, W / 2], [0.0, FX, H / 2], [0.0, 0.0, 1.0]])

ARM_BOXES = {            # link: (size, centre in the link frame)
    "base": ((0.12, 0.12, 0.08), (0.0, 0.0, 0.04)),
    "upper": ((0.06, 0.06, 0.24), (0.0, 0.0, 0.12)),
    "fore": ((0.30, 0.05, 0.05), (0.15, 0.0, 0.0)),
    "palm": ((0.04, 0.16, 0.04), (0.02, 0.0, 0.0)),
    "finger_l": ((0.06, 0.015, 0.03), (0.03, 0.0075, 0.0)),
    "finger_r": ((0.06, 0.015, 0.03), (0.03, -0.0075, 0.0)),
}
JAWS = ("palm", "finger_l", "finger_r")
SCENE_BOXES = {          # base frame: (size, centre)
    "table": ((3.0, 3.0, 0.02), (0.2, 0.0, -0.01)),
    "wall": ((3.0, 0.02, 3.0), (0.2, 0.7, 0.5)),
    "object": ((0.06, 0.06, 0.06), (0.22, 0.12, 0.03)),
}


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _T(R=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = t
    return T


def synth_poses(state) -> list[dict[str, np.ndarray]]:
    """FK of the synthetic arm. q[0] yaws the post, q[1] pitches the fore
    link; `gripper_pos` is the jaw gap in metres (unknown -> both ends, the
    swept convention of KinematicsLinkPoses)."""
    q = np.asarray(state.q, dtype=float)
    upper = _T(_rz(q[0]), (0.0, 0.0, 0.08))
    fore = upper @ _T(t=(0.0, 0.0, 0.24)) @ _T(_ry(q[1]))
    palm = fore @ _T(t=(0.30, 0.0, 0.0))
    gaps = [float(state.gripper_pos)] if getattr(state, "gripper_valid", True) else [0.02, 0.12]
    return [{"base": np.eye(4), "upper": upper, "fore": fore, "palm": palm,
             "finger_l": palm @ _T(t=(0.04, gap / 2, 0.0)), "finger_r": palm @ _T(t=(0.04, -gap / 2, 0.0))}
            for gap in gaps]


def synth_chain(state) -> np.ndarray:
    """The capsule mask's polyline: joint frames + TCP (front of the palm)."""
    P = synth_poses(state)[0]
    return np.array([[0, 0, 0], P["upper"][:3, 3], P["fore"][:3, 3], P["palm"][:3, 3],
                     (P["palm"] @ np.array([0.04, 0, 0, 1.0]))[:3]])


def synth_geometry() -> lm.LinkGeometry:
    return lm.LinkGeometry(
        pieces={k: (lm._box_points(s) + np.asarray(c),) for k, (s, c) in ARM_BOXES.items()},
        sources={k: "box" for k in ARM_BOXES})


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """cam->base for an optical frame (x right, y down, z forward)."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return _T(np.stack([x, np.cross(z, x), z], axis=1), eye)


def render(state, T_base_cam):
    """(depth, label) by ray casting every posed box: label is the box name
    of the first hit per pixel ('' = nothing)."""
    boxes = [(name, s, P[name] @ _T(t=c)) for name, (s, c) in ARM_BOXES.items()
             for P in synth_poses(state)[:1]]
    boxes += [(name, s, _T(t=c)) for name, (s, c) in SCENE_BOXES.items()]
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d_cam = np.stack([(u - K[0, 2]) / FX, (v - K[1, 2]) / FX, np.ones_like(u)], -1).reshape(-1, 3)
    R, o = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = d_cam @ R.T
    best = np.full(len(d), np.inf)
    label = np.full(len(d), "", dtype=object)
    for name, size, Tb in boxes:
        Rb, cb, h = Tb[:3, :3], Tb[:3, 3], np.asarray(size) / 2
        ol, dl = (o - cb) @ Rb, d @ Rb
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-h - ol) / dl, (h - ol) / dl
        tn = np.nanmax(np.minimum(t1, t2), axis=1)
        tf = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tn <= tf) & (tn > 1e-6) & (tn < best)
        best[hit], label[hit] = tn[hit], name
    depth = np.where(np.isfinite(best), best, 0.0).reshape(H, W).astype(np.float32)  # d_cam has z = 1
    return depth, label.reshape(H, W)


def _state(q=(0.0, 0.0), gap=0.12, valid=True):
    from cascade.types import RobotState

    return RobotState(q=np.asarray(q, dtype=float), gripper_pos=gap, gripper_valid=valid)


SIDE_CAM = look_at((0.20, -0.90, 0.45), (0.20, 0.0, 0.22))


def _mesh_mask(state, **kw):
    from cascade.perception.robot_mask import MeshBodyMask

    return MeshBodyMask(synth_geometry(), synth_poses, lambda: state, **kw)


def test_premise_the_capsule_leaves_the_wide_jaws_in_the_map():
    """The failure measured on the rig, reproduced: 6 cm capsules around the
    joint chain do not reach the wide palm and jaws."""
    from cascade.perception.robot_mask import robot_mask

    state = _state()
    depth, label = render(state, SIDE_CAM)
    jaws = np.isin(label, JAWS)
    assert jaws.sum() > 500
    capsule = robot_mask(depth, K, SIDE_CAM, synth_chain(state), radius_m=0.06)
    assert (jaws & ~capsule).sum() > 100, "premise: the capsule must miss part of the jaws"


def test_mesh_mask_covers_every_robot_pixel_and_spares_the_object_beside_it():
    state = _state()
    depth, label = render(state, SIDE_CAM)
    robot = np.isin(label, list(ARM_BOXES))
    mask = _mesh_mask(state, dilate_px=2)(depth, K, SIDE_CAM)
    assert mask.shape == depth.shape and mask.dtype == np.bool_
    assert robot.sum() > 5000 and (robot & ~mask).sum() == 0, "a robot pixel reached the map"
    obj = label == "object"
    assert obj.sum() > 300 and not (mask & obj).any(), "the object next to the arm was masked"


def test_mesh_mask_ignores_the_depth_values():
    """A silhouette, not a 3D distance test: a wrong depth on a robot pixel
    (D455F speckle on the jaws, a stereo mismatch) is still masked. The
    capsule test reads the depth and would let it through."""
    state = _state()
    depth, label = render(state, SIDE_CAM)
    robot = np.isin(label, list(ARM_BOXES))
    garbage = np.where(robot, 2.5, depth).astype(np.float32)   # every arm pixel reads 2.5 m
    mask = _mesh_mask(state, dilate_px=2)(garbage, K, SIDE_CAM)
    assert (robot & ~mask).sum() == 0


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.4, -0.5), (-0.6, 0.3)])
def test_eye_in_hand_jaws_straddling_the_near_plane_are_masked_and_fixed_in_the_image(q):
    """The wrist camera: T_base_cam = FK(q) @ X. Its jaws sit centimetres
    away and the palm crosses the camera plane, so pieces are clipped at
    the near plane. The hand-mounted geometry must be covered, and since the
    camera moves with the hand, its mask must not depend on q."""
    X = look_at((0.02, 0.0, 0.035), (0.12, 0.0, -0.01))            # palm frame, 1.5 cm above it
    palm_size, palm_centre = ARM_BOXES["palm"]
    palm_in_cam = (lm._box_points(palm_size) + np.asarray(palm_centre) - X[:3, 3]) @ X[:3, :3]
    assert palm_in_cam[:, 2].min() < 0 < palm_in_cam[:, 2].max(), "premise: the palm crosses the camera plane"
    state = _state(q)
    T = synth_poses(state)[0]["palm"] @ X
    depth, label = render(state, T)
    hand = np.isin(label, JAWS)
    assert (label == "palm").sum() > 10_000 and hand.sum() > 50_000
    mask = _mesh_mask(state, dilate_px=0)(depth, K, T)
    assert (hand & ~mask).sum() == 0
    ref_state = _state((0.0, 0.0))
    ref = _mesh_mask(ref_state, dilate_px=0)(depth, K, synth_poses(ref_state)[0]["palm"] @ X)
    assert (mask != ref).sum() <= 10, "the hand's mask moved in the wrist camera's image"


@pytest.mark.parametrize("state, why", [(None, "no joint state"), (_state((np.nan, 0.0)), "unusable"),
                                         (_state((0.0,)), "unusable")])
def test_unknown_or_unusable_joint_state_refuses_instead_of_masking_nothing(state, why):
    """The occupancy contract: an unknown pose is not an empty scene."""
    from cascade.perception.robot_mask import BodyMaskUnavailable, MeshBodyMask

    depth, _ = render(_state(), SIDE_CAM)
    body = MeshBodyMask(synth_geometry(), synth_poses, lambda: state, n_joints=2)
    with pytest.raises(BodyMaskUnavailable, match=why):
        body(depth, K, SIDE_CAM)


def test_the_pixel_dilation_grows_the_silhouette_by_that_many_pixels():
    """`dilate_px` absorbs flying pixels on the silhouette's border: a ring
    of exactly that width (Euclidean distance transform as the truth)."""
    from scipy.ndimage import distance_transform_edt

    state = _state()
    depth, _ = render(state, SIDE_CAM)
    m0 = _mesh_mask(state, dilate_px=0)(depth, K, SIDE_CAM)
    m3 = _mesh_mask(state, dilate_px=3)(depth, K, SIDE_CAM)
    dist = distance_transform_edt(~m0)
    assert (m0 & ~m3).sum() == 0
    assert m3[(dist > 0) & (dist <= 2.5)].all() and not m3[dist > 3.5].any()
    assert (m3 & ~m0).sum() > 500


def test_a_failing_state_read_propagates_and_is_never_swallowed():
    from cascade.perception.robot_mask import MeshBodyMask

    def boom():
        raise RuntimeError("bus read failed")

    body = MeshBodyMask(synth_geometry(), synth_poses, boom)
    with pytest.raises(RuntimeError, match="bus read failed"):
        body(np.zeros((H, W), np.float32), K, SIDE_CAM)


def test_unknown_opening_sweeps_the_jaws_and_a_measured_one_does_not():
    """Fingers at the measured gap; unknown -> the whole travel (more mask,
    never less). A thin object between the measured jaws stays visible."""
    depth, _ = render(_state(gap=0.12), SIDE_CAM)
    measured = _mesh_mask(_state(gap=0.12), dilate_px=0)(depth, K, SIDE_CAM)
    swept = _mesh_mask(_state(gap=0.0, valid=False), dilate_px=0)(depth, K, SIDE_CAM)
    assert (measured & ~swept).sum() == 0 and (swept & ~measured).sum() > 0


def test_base_pose_moves_the_arm_into_the_table_frame():
    """A profile's `base_pose`: link poses are in the arm's base frame, the
    camera extrinsics in the table frame."""
    from cascade.perception.robot_mask import MeshBodyMask

    state = _state()
    shift = _T(t=(0.10, 0.0, 0.0))
    depth, label = render(state, np.linalg.inv(shift) @ SIDE_CAM)   # the arm seen from a camera moved back
    body = MeshBodyMask(synth_geometry(), synth_poses, lambda: state, base_T=shift, dilate_px=2)
    mask = body(depth, K, SIDE_CAM)
    robot = np.isin(label, list(ARM_BOXES))
    assert (robot & ~mask).sum() == 0


# ── slice 4: the occupancy map integrates only what the mesh mask leaves ──


class _Bridge:
    """Records what would go over the wire; `native` = an nvblox bridge
    that negotiated integrate_masked_depth; `points_only` = an old bridge
    without the depth protocol (the legacy cloud path)."""

    def __init__(self, *, native=False, points_only=False):
        self.native, self.points_only = native, points_only
        self.requests: list[dict] = []

    def request(self, payload, timeout_ms=None):
        from cascade.perception.occupancy import OccupancyError

        self.requests.append(payload)
        action = payload["action"]
        if action == "probe":
            return {"ok": True, "backend": "nvblox" if self.native else "fake", "device": "cpu",
                    "voxel": 0.01, "esdf": True, "masked_depth": self.native}
        if action in ("integrate_depth", "integrate_masked_depth") and self.points_only:
            raise OccupancyError(f"unknown action {action!r}")
        if action in ("integrate_depth", "integrate_masked_depth", "integrate"):
            return {"ms": 0.1}
        if action == "query":
            return {"points": np.zeros((0, 3), np.float32), "grid": np.full((2, 2, 2), np.inf, np.float32),
                    "origin": np.zeros(3, np.float32), "voxel": 0.01}
        raise OccupancyError(f"unknown action {action!r}")

    def probe(self, timeout_ms=300):
        return self.request({"action": "probe"})

    def sent(self, action):
        return [p for p in self.requests if p["action"] == action]


def _occupancy(bridge, stride=1):
    from cascade.perception.occupancy import OccupancyMap

    return OccupancyMap(client=bridge, region_min=np.array([-0.3, -0.5, -0.05]),
                        region_max=np.array([0.7, 0.8, 0.6]), depth_stride=stride)


def _frame(depth):
    from cascade.types import Frame

    return Frame(rgb=np.zeros(depth.shape + (3,), np.uint8), depth_m=depth, K=K.copy())


@pytest.mark.parametrize("stride", [1, 2])
def test_the_map_never_receives_a_robot_pixel_and_keeps_the_object(stride):
    state = _state()
    depth, label = render(state, SIDE_CAM)
    bridge = _Bridge()
    occ = _occupancy(bridge, stride)
    occ.add_robot_body_mask(_mesh_mask(state))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert occ.last_error is None, occ.last_error
    shipped = bridge.sent("integrate_depth")[-1]["depth"]
    lab = label[::stride, ::stride]
    assert shipped.shape == lab.shape
    assert (shipped[np.isin(lab, list(ARM_BOXES))] == 0).all(), "a robot pixel reached the map"
    obj = lab == "object"
    np.testing.assert_allclose(shipped[obj], depth[::stride, ::stride][obj])
    assert occ.last_masked_px >= int(np.isin(lab, list(ARM_BOXES)).sum())


def test_native_nvblox_masking_gets_the_mesh_mask_as_inactive_rays():
    state = _state()
    depth, label = render(state, SIDE_CAM)
    bridge = _Bridge(native=True)
    occ = _occupancy(bridge)
    occ.add_robot_body_mask(_mesh_mask(state))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert occ.last_error is None, occ.last_error
    packet = bridge.sent("integrate_masked_depth")[-1]
    np.testing.assert_array_equal(packet["depth"], depth)            # raw depth: nvblox masks natively
    robot = np.isin(label, list(ARM_BOXES))
    assert not packet["active_mask"][robot].any() and packet["active_mask"][label == "object"].all()


def test_the_legacy_cloud_path_drops_the_robot_points_too():
    state = _state()
    depth, label = render(state, SIDE_CAM)
    bridge = _Bridge(points_only=True)
    occ = _occupancy(bridge)
    occ.stride = 1
    occ.add_robot_body_mask(_mesh_mask(state))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert occ.last_error is None, occ.last_error
    pts = np.asarray(bridge.sent("integrate")[-1]["points"], dtype=float)
    # independent truth: back-project the robot's own pixels; none may be sent
    ys, xs = np.nonzero(np.isin(label, list(ARM_BOXES)) & (depth > 0))
    z = depth[ys, xs]
    robot_pts = (np.stack([(xs - K[0, 2]) / FX * z, (ys - K[1, 2]) / FX * z, z], -1)
                 @ SIDE_CAM[:3, :3].T + SIDE_CAM[:3, 3])
    from scipy.spatial import cKDTree

    assert cKDTree(robot_pts).query(pts, distance_upper_bound=1e-4)[0].min() > 1e-4
    assert len(pts) > 1000


@pytest.mark.parametrize("points_only", [False, True])
@pytest.mark.parametrize("state_fn", [lambda: None, lambda: (_ for _ in ()).throw(RuntimeError("read failed"))])
def test_no_joint_state_integrates_nothing_and_blocks_clearance(state_fn, points_only):
    """Standby (LazyArm) or a failed read: NOTHING reaches the map, and the
    fault cannot age into a clearance bypass."""
    from cascade.perception.robot_mask import MeshBodyMask
    from cascade.types import SafetyViolation

    depth, _ = render(_state(), SIDE_CAM)
    bridge = _Bridge(points_only=points_only)
    occ = _occupancy(bridge)
    if points_only:
        occ._depth_supported = False                                  # straight to the legacy cloud path
    occ.add_robot_body_mask(MeshBodyMask(synth_geometry(), synth_poses, state_fn))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert not [p for p in bridge.requests if p["action"].startswith("integrate")]
    assert "robot body pose" in occ.last_error and occ.last_masked_px == 0
    occ._last_refresh = 0.0                                           # an aged cache
    with pytest.raises(SafetyViolation, match="robot body pose"):
        occ.clearance([[0.3, 0.0, 0.3]])


def test_losing_the_joint_state_after_a_good_refresh_relatches_the_fault():
    """The arm masked fine, then its read fails (or it drops to standby):
    the next frame is refused and clearance is refused with it, even though
    the previous grid is still fresh."""
    from cascade.perception.robot_mask import MeshBodyMask
    from cascade.types import SafetyViolation

    current = {"state": _state()}
    depth, _ = render(current["state"], SIDE_CAM)
    bridge = _Bridge()
    occ = _occupancy(bridge)
    occ.add_robot_body_mask(MeshBodyMask(synth_geometry(), synth_poses, lambda: current["state"]))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert occ.last_error is None
    occ.clearance([[0.3, 0.0, 0.3]])                                  # validated: no fault raised
    sent = len(bridge.sent("integrate_depth"))
    current["state"] = None
    occ.refresh(_frame(depth), SIDE_CAM)
    assert len(bridge.sent("integrate_depth")) == sent, "an unmasked frame reached the map"
    with pytest.raises(SafetyViolation, match="mesh mask"):
        occ.clearance([[0.3, 0.0, 0.3]])


@pytest.mark.parametrize("bad", [lambda d: np.zeros(d.shape, np.uint8), lambda d: np.zeros((3, 3), bool),
                                 lambda d: None])
def test_a_malformed_body_mask_is_refused(bad):
    depth, _ = render(_state(), SIDE_CAM)
    bridge = _Bridge()
    occ = _occupancy(bridge)
    occ.add_robot_body_mask(lambda d, K_, T_: bad(d))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert not bridge.sent("integrate_depth") and "robot body pose" in occ.last_error


@pytest.mark.parametrize("points_only", [False, True])
def test_a_capsule_arm_and_a_mesh_arm_are_both_masked(points_only):
    """Two arms on one map: each keeps its own masking, on both wire paths."""
    from cascade.perception.robot_mask import _segment_distances, robot_mask

    state = _state()
    depth, label = render(state, SIDE_CAM)
    other = np.array([[0.55, 0.0, 0.0], [0.55, 0.0, 0.25]])          # a second arm's chain (a post)
    bridge = _Bridge(points_only=points_only)
    occ = _occupancy(bridge)
    occ.stride = 1
    occ.add_robot_body(lambda: other, radius_m=0.05)
    occ.add_robot_body_mask(_mesh_mask(state))
    occ.refresh(_frame(depth), SIDE_CAM)
    assert occ.last_error is None, occ.last_error
    capsule = robot_mask(depth, K, SIDE_CAM, other, radius_m=0.05)
    assert capsule.sum() > 0
    if not points_only:
        shipped = bridge.sent("integrate_depth")[-1]["depth"]
        assert (shipped[capsule] == 0).all()
        assert (shipped[np.isin(label, list(ARM_BOXES))] == 0).all()
        return
    pts = np.asarray(bridge.sent("integrate")[-1]["points"], dtype=float)
    assert len(pts) > 1000 and _segment_distances(pts, other[0], other[1]).min() > 0.05


def test_a_render_mask_is_never_written_by_the_mesh_mask():
    """The frame's own `robot_mask` (Isaac) is shared with fusion: OR-ing the
    mesh mask in must build a new array, never write the frame's."""
    state = _state()
    depth, _ = render(state, SIDE_CAM)
    frame = _frame(depth)
    frame.robot_mask = np.zeros(depth.shape, bool)
    occ = _occupancy(_Bridge())
    occ.add_robot_body_mask(_mesh_mask(state))
    occ.refresh(frame, SIDE_CAM)
    assert occ.last_error is None and not frame.robot_mask.any()


def test_payload_tracking_and_the_mesh_mask_are_refused_together():
    """Payload replay re-sends history with the frames' render masks only; a
    mesh-masked arm's history would replay unmasked."""
    from cascade.perception.occupancy import OccupancyError

    occ = _occupancy(_Bridge())
    occ.add_robot_body_mask(_mesh_mask(_state()))
    with pytest.raises(OccupancyError, match="mesh"):
        occ.track_payload(lambda frame: np.eye(4))
    occ = _occupancy(_Bridge())
    occ.track_payload(lambda frame: np.eye(4))
    with pytest.raises(OccupancyError, match="mesh"):
        occ.add_robot_body_mask(_mesh_mask(_state()))


# ── slice 5: the arm profile's `body_mask` and the builder ───────────────


def _settings(block):
    from cascade.config import Cfg
    from cascade.perception.robot_mask import body_mask_settings

    return body_mask_settings(Cfg({"body_mask": block} if block is not ... else {}))


def test_capsule_stays_the_default_and_mesh_opts_in_with_documented_defaults():
    from cascade.perception.robot_mask import (
        DEFAULT_BODY_DILATE_PX,
        DEFAULT_BODY_INFLATE_M,
    )

    assert _settings(...) is None and _settings("capsule") is None and _settings({"mode": "capsule"}) is None
    got = _settings("mesh")
    assert got == {"inflate_m": DEFAULT_BODY_INFLATE_M, "dilate_px": DEFAULT_BODY_DILATE_PX,
                   "cell_m": lm.DEFAULT_CELL_M, "attachments": []}
    assert DEFAULT_BODY_INFLATE_M == 0.02 and DEFAULT_BODY_DILATE_PX == 2


def test_a_full_mesh_block_with_an_attachment():
    got = _settings({"mode": "mesh", "inflate_m": 0.015, "dilate_px": 3, "cell_m": 0.03,
                     "attachments": [{"name": "wrist_d455f", "link": "gripper_end", "box": [0.065, 0.14, 0.055],
                                      "xyz": [-0.0925, 0.0, 0.0625], "rpy": [0.0, 0.0, 0.1]}]})
    assert got["inflate_m"] == 0.015 and got["dilate_px"] == 3 and got["cell_m"] == 0.03
    (name, link, size, T), = got["attachments"]
    assert (name, link) == ("wrist_d455f", "gripper_end") and tuple(size) == (0.065, 0.14, 0.055)
    np.testing.assert_allclose(T, _rpy_T((-0.0925, 0.0, 0.0625), (0.0, 0.0, 0.1)))


@pytest.mark.parametrize("block", [
    "meshes", 3, {"mode": "voxels"}, {"mode": "mesh", "inflate_m": -0.01}, {"mode": "mesh", "inflate_m": "2cm"},
    {"mode": "mesh", "dilate_px": 1.5}, {"mode": "mesh", "dilate_px": -1}, {"mode": "mesh", "cell_m": 0},
    {"mode": "mesh", "inflate": 0.02},                                           # a typo is not a default
    {"mode": "mesh", "attachments": {"link": "gripper_end"}},
    {"mode": "mesh", "attachments": [{"name": "cam", "box": [0.1, 0.1, 0.1]}]},            # no link
    {"mode": "mesh", "attachments": [{"name": "cam", "link": "gripper_end", "box": [0.1, 0.1]}]},
    {"mode": "mesh", "attachments": [{"name": "cam", "link": "gripper_end", "box": [0.1, 0.1, 0.1],
                                      "xyz": [0.0, 0.0]}]},
    {"mode": "mesh", "attachments": [{"name": "cam", "link": "gripper_end", "box": [0.1, 0.1, 0.1],
                                      "colour": "silver"}]},
])
def test_bad_body_mask_settings_are_refused_loudly(block):
    with pytest.raises(ValueError, match="body_mask"):
        _settings(block)


def _urdf_with_meshes(tmp_path, *, lfs: bool) -> str:
    """A two-link URDF whose second mesh is either a real binary STL or an
    unfetched git-lfs pointer (what a CI checkout without LFS has)."""
    import struct

    (tmp_path / "meshes").mkdir()
    tris = np.array([[[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], [[0, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]], np.float32)
    data = b"stl".ljust(80, b" ") + struct.pack("<I", len(tris))
    data += b"".join(struct.pack("<3f", 0, 0, 0) + t.tobytes() + b"\0\0" for t in tris)
    (tmp_path / "meshes" / "a.stl").write_bytes(data)
    (tmp_path / "meshes" / "b.stl").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 9\n" if lfs else data)
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("""<robot name="t">
  <link name="a"><collision><geometry><mesh filename="meshes/a.stl"/></geometry></collision></link>
  <link name="b"><collision><geometry><mesh filename="meshes/b.stl"/></geometry></collision></link>
  <joint name="j" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.2"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>""")
    return str(urdf)


def test_a_mesh_that_cannot_load_refuses_the_mesh_mask_instead_of_a_silent_capsule(tmp_path):
    """`body_mask: mesh` promised the URDF geometry. A capsule in its place
    (LinkGeometry's fallback, fine for fusion's gate) would be exactly the
    mask that left the jaws in the map, so the build refuses and says why."""
    from cascade.config import Cfg
    from cascade.perception.robot_mask import build_mesh_body_mask

    acfg = Cfg({"model": _urdf_with_meshes(tmp_path, lfs=True), "body_mask": "mesh"})
    with pytest.raises(ValueError, match=r"(?s)body_mask: mesh.*'b'.*git-lfs.*body_mask: capsule"):
        build_mesh_body_mask(acfg, kin=None, state_fn=lambda: None, name="arm0")


def test_the_capsule_profile_builds_no_mesh_mask_and_reads_nothing(tmp_path):
    from cascade.config import Cfg
    from cascade.perception.robot_mask import build_mesh_body_mask

    acfg = Cfg({"model": _urdf_with_meshes(tmp_path, lfs=True)})                  # no body_mask
    assert build_mesh_body_mask(acfg, kin=None, state_fn=lambda: pytest.fail("read"), name="arm0") is None


def _rs_meshes_available() -> bool:
    from conftest import URDF

    probe = URDF.parents[1] / "meshes" / "link2.STL"
    try:
        return probe.exists() and not probe.read_bytes()[:64].startswith(b"version https://git-lfs")
    except OSError:
        return False


def _has_pin() -> bool:
    from conftest import has_pinocchio

    return has_pinocchio()


needs_rs = pytest.mark.skipif(not (_has_pin() and _rs_meshes_available()),
                              reason="needs pinocchio and the RS meshes (git-lfs)")


def _rs_acfg(**block):
    from conftest import JOINT_SIGNS, URDF

    from cascade.config import Cfg

    return Cfg({"model": str(URDF), "ee_frame": "gripper_end", "n_joints": 6, "joint_signs": JOINT_SIGNS,
                "gripper": {"open_pos": 6.2, "closed_pos": 0.0},
                "body_mask": {"mode": "mesh", **block}})


def _rs_kin():
    from conftest import JOINT_SIGNS, URDF

    from cascade.control.kinematics import Kinematics

    return Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)


@needs_rs
@pytest.mark.parametrize("q", [[0.0] * 6, [0.0, 1.2, 1.2, 0.0, 0.0, 0.0], [0.3, 0.6, 1.6, 0.2, -0.4, 0.5]])
def test_the_rs_mesh_mask_contains_every_rs_triangle_at_the_measured_opening(q):
    """The real reBot RS model, end to end through the builder: the mask at
    640x360 contains a per-triangle fill of the RS collision meshes (truth
    drawn by test_link_self_mask's helpers, one triangle per fill call),
    fingers at the measured opening, no metric inflation and 1 px of
    dilation (exact geometry; the pixel absorbs the two fills' different
    edge rounding, as in test_link_self_mask), within the 20 ms budget."""
    from test_link_self_mask import _fill, _pin_link_poses, _posed, _rs_truth_triangles

    from cascade.perception.robot_mask import build_mesh_body_mask
    from cascade.types import RobotState

    kin = _rs_kin()
    state = RobotState(q=np.asarray(q, float), gripper_pos=3.1, gripper_valid=True)    # half open: 50 mm
    body = build_mesh_body_mask(_rs_acfg(inflate_m=0.0, dilate_px=1), kin, lambda: state, name="rebot")
    T = look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.20))
    truth = _fill(_posed(_rs_truth_triangles(), _pin_link_poses(kin, np.asarray(q, float), 0.025)), T, (H, W), K)
    assert truth.sum() > 3000
    mask = body(np.zeros((H, W), np.float32), K, T)
    times = []
    for _ in range(5):
        body(np.zeros((H, W), np.float32), K, T)
        times.append(body.last["ms"])
    ms = min(times)
    print(f"\nRS mesh body mask q={q}: {int(mask.sum())} px, truth {int(truth.sum())} px, {ms:.1f} ms")
    assert (truth & ~mask).sum() == 0
    assert ms < 20.0


@needs_rs
def test_the_inflation_reaches_past_the_surface_by_the_margin():
    """inflate_m in metres, checked in 3D: every surface point of the RS
    collision meshes pushed 1.9 cm outward (any direction) still projects
    into the 2 cm-inflated mask."""
    from test_link_self_mask import _pin_link_poses, _posed, _rs_truth_triangles

    from cascade.perception.robot_mask import build_mesh_body_mask
    from cascade.types import RobotState

    kin = _rs_kin()
    q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    state = RobotState(q=q, gripper_pos=3.1, gripper_valid=True)
    body = build_mesh_body_mask(_rs_acfg(inflate_m=0.02, dilate_px=0), kin, lambda: state, name="rebot")
    T = look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.20))
    pts = _posed(_rs_truth_triangles(), _pin_link_poses(kin, q, 0.025)).reshape(-1, 3)
    pts = pts[np.random.default_rng(1).choice(len(pts), 20000, replace=False)]
    pushed = pts + 0.019 * _unit_dirs(len(pts), seed=2)
    cam = (pushed - T[:3, 3]) @ T[:3, :3]
    uv = np.round(cam[:, :2] / cam[:, 2:] * FX + K[:2, 2]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mask = body(np.zeros((H, W), np.float32), K, T)
    assert inside.sum() > 15000 and mask[uv[inside, 1], uv[inside, 0]].all()


@needs_rs
def test_an_attachment_on_a_link_the_model_lacks_is_refused():
    from cascade.perception.robot_mask import build_mesh_body_mask

    acfg = _rs_acfg(attachments=[{"name": "cam", "link": "wrist_cam_mount", "box": [0.1, 0.1, 0.1]}])
    with pytest.raises(ValueError, match="wrist_cam_mount"):
        build_mesh_body_mask(acfg, _rs_kin(), lambda: None, name="rebot")


@needs_rs
def test_the_rs_fingers_follow_the_profile_gripper_mapping():
    """From above the gripper at home the jaw gap is visible: the measured
    opening (gripper.open_pos/closed_pos maps the motor angle onto the
    URDF's finger travel) leaves it open, an unknown one sweeps it shut."""
    from cascade.perception.robot_mask import build_mesh_body_mask
    from cascade.types import RobotState

    kin = _rs_kin()
    q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    tcp = kin.fk(q)[:3, 3]
    T = look_at(tcp + np.array([0.0, 0.0, 0.45]), tcp, up=(1.0, 0.0, 0.0))
    current = {}
    body = build_mesh_body_mask(_rs_acfg(inflate_m=0.0, dilate_px=0), kin, lambda: current["s"], name="rebot")
    masks = {}
    for tag, pos, valid in (("closed", 0.0, True), ("open", 6.2, True), ("unknown", 0.0, False)):
        current["s"] = RobotState(q=q, gripper_pos=pos, gripper_valid=valid)
        masks[tag] = body(np.zeros((H, W), np.float32), K, T)
    assert (masks["open"] & ~masks["unknown"]).sum() == 0 and (masks["closed"] & ~masks["unknown"]).sum() == 0
    assert (masks["unknown"] & ~masks["open"]).sum() > 200, "the measured opening must leave the jaw gap visible"
    assert (masks["open"] != masks["closed"]).sum() > 200


@needs_rs
def test_the_profile_base_pose_places_the_arm_in_the_table_frame():
    from cascade.perception.robot_mask import build_mesh_body_mask
    from cascade.types import RobotState

    kin = _rs_kin()
    state = RobotState(q=np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0]), gripper_pos=3.1, gripper_valid=True)
    T = look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.20))
    plain = build_mesh_body_mask(_rs_acfg(), kin, lambda: state, name="rebot")
    acfg = _rs_acfg()
    shifted_cfg = type(acfg)({**acfg.as_dict(), "base_pose": [0.12, 0.0, 0.0, 0.0, 0.0, 0.0]})
    shifted = build_mesh_body_mask(shifted_cfg, kin, lambda: state, name="rebot")
    shift = _T(t=(0.12, 0.0, 0.0))
    a = shifted(np.zeros((H, W), np.float32), K, T)
    b = plain(np.zeros((H, W), np.float32), K, np.linalg.inv(shift) @ T)
    assert a.sum() > 3000 and (a != b).sum() <= 0.005 * a.sum()
    assert (a != plain(np.zeros((H, W), np.float32), K, T)).sum() > 1000


# ── slice 6: the demo wiring (`_build_arm`) ─────────────────────────────


def _rebot_profile(**override):
    from cascade.apps import demo
    from cascade.config import Cfg, load_demo_config

    cfg = load_demo_config(arm="rebot_rs", llm="mock")
    acfg = demo._arm_cfgs(cfg)[0]
    return cfg, Cfg({**acfg.as_dict(), **override})


@needs_rs
def test_a_standby_arm_is_never_woken_and_nothing_is_integrated(monkeypatch):
    """LazyArm in standby: the mesh mask has no joint state, so the frame is
    refused (fail closed) and the motors are never materialized by looking."""
    from cascade.apps import demo

    built = []
    monkeypatch.setattr(demo, "make_arm", lambda *a, **k: built.append(1) or pytest.fail("materialized"))
    cfg, acfg = _rebot_profile(body_mask="mesh")
    bridge = _Bridge()
    occ = _occupancy(bridge)
    arm, _, _ = demo._build_arm(acfg, True, occ, cfg)
    assert len(occ._pixel_body_fns) == 1 and occ._body_fns == []
    depth = np.full((H, W), 1.0, np.float32)
    occ.refresh(_frame(depth), look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.2)))
    assert not [p for p in bridge.requests if p["action"].startswith("integrate")]
    assert "robot body pose" in occ.last_error and "standby" in occ.last_error
    assert built == [] and arm.connected is False


@needs_rs
def test_a_connected_arm_is_masked_by_its_meshes_at_its_measured_state():
    """A connected arm (mock backend, the rebot_rs profile's model): the
    shipped frame has every pixel of the arm's RS triangles zeroed (truth:
    per-triangle fill at the arm's own state, fingers at its gripper)."""
    from test_link_self_mask import _fill, _pin_link_poses, _posed, _rs_truth_triangles

    from cascade.apps import demo

    cfg, acfg = _rebot_profile(body_mask="mesh", type="mock")
    bridge = _Bridge()
    occ = _occupancy(bridge)
    arm, _, kin = demo._build_arm(acfg, False, occ, cfg)
    state = arm.get_state()
    np.testing.assert_allclose(state.q, acfg.home_q)
    T = look_at((0.25, -0.75, 0.45), (0.25, 0.0, 0.2))
    depth = np.full((H, W), 1.0, np.float32)
    occ.refresh(_frame(depth), T)
    assert occ.last_error is None, occ.last_error
    finger = 0.05 * (state.gripper_pos - 0.0) / 6.2                    # profile closed 0, open 6.2 rad
    truth = _fill(_posed(_rs_truth_triangles(), _pin_link_poses(kin, np.asarray(state.q), finger)), T, (H, W), K)
    shipped = bridge.sent("integrate_depth")[-1]["depth"]
    assert truth.sum() > 3000 and (shipped[truth] == 0).all()
    assert occ.last_masked_px >= truth.sum()


@needs_rs
def test_the_capsule_profile_keeps_the_capsule_body():
    from cascade.apps import demo

    cfg, acfg = _rebot_profile(body_mask="capsule")
    occ = _occupancy(_Bridge())
    demo._build_arm(acfg, True, occ, cfg)
    assert len(occ._body_fns) == 1 and occ._pixel_body_fns == []
    assert occ._body_radii == [pytest.approx(float(acfg.get("body_mask_radius_m", 0.06)))]


@needs_rs
def test_the_mesh_mask_is_refused_for_an_isaac_arm():
    """Isaac frames carry the render mask and their CAPTURE-time joint state;
    a mesh mask posed at the live state would mask yesterday's image."""
    from cascade.apps import demo

    cfg, acfg = _rebot_profile(body_mask="mesh", type="isaac")
    with pytest.raises(ValueError, match="isaac"):
        demo._build_arm(acfg, True, _occupancy(_Bridge()), cfg)


@needs_rs
def test_without_an_occupancy_map_nothing_is_built():
    from cascade.apps import demo

    cfg, acfg = _rebot_profile(body_mask="mesh")
    arm, _, _ = demo._build_arm(acfg, True, None, cfg)
    assert arm.connected is False


# ── slice 7: the shipped profiles ───────────────────────────────────────


def test_every_shipped_arm_profile_has_a_valid_body_mask():
    from cascade.config import CONFIG_DIR, load_profile
    from cascade.perception.robot_mask import body_mask_settings

    names = sorted(p.stem for p in (CONFIG_DIR / "arms").glob("*.yaml"))
    assert "rebot_rs" in names
    for name in names:
        body_mask_settings(load_profile("arms", name))          # raises on a malformed block


def test_the_rebot_rs_profile_opts_in_with_the_wrist_camera_attached():
    """The real reBot: URDF meshes, the shipped margins, and the wrist D455F
    (not in the URDF) as a box on gripper_end that encloses the depth points
    the camera's body left on the real capture (gripper_end frame, metres:
    x -0.11..-0.07, y -0.06..0.05, z 0.06..0.08, measured 2026-10-11)."""
    from cascade.config import load_profile
    from cascade.perception.robot_mask import (
        DEFAULT_BODY_DILATE_PX,
        DEFAULT_BODY_INFLATE_M,
        body_mask_settings,
    )

    got = body_mask_settings(load_profile("arms", "rebot_rs"))
    assert got is not None and got["inflate_m"] == DEFAULT_BODY_INFLATE_M
    assert got["dilate_px"] == DEFAULT_BODY_DILATE_PX
    (name, link, size, T), = got["attachments"]
    assert (name, link) == ("wrist_d455f", "gripper_end")
    lo = T[:3, 3] - np.abs(T[:3, :3]) @ (np.asarray(size) / 2)
    hi = T[:3, 3] + np.abs(T[:3, :3]) @ (np.asarray(size) / 2)
    observed_lo, observed_hi = np.array([-0.11, -0.062, 0.041]), np.array([-0.07, 0.062, 0.08])
    assert (lo <= observed_lo).all() and (hi >= observed_hi).all(), "the box must enclose the camera's points"
    assert (np.asarray(size) <= 0.16).all(), "an envelope, not a blanket over the gripper"


# ── scripts/eval_body_mask.py: capsule vs mesh on a recorded capture ─────


def _eval_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_body_mask.py"
    spec = importlib.util.spec_from_file_location("eval_body_mask", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _painter_depth(tris_base, T, shape=(H, W)):
    """A synthetic depth image of triangles (the test's own z-buffer):
    farthest first, each filled with its mean camera depth."""
    import cv2

    pc = (np.asarray(tris_base, float) - T[:3, 3]) @ T[:3, :3]
    z = pc[..., 2].mean(axis=1)
    uv = np.stack([pc[..., 0] / pc[..., 2] * FX + K[0, 2], pc[..., 1] / pc[..., 2] * FX + K[1, 2]], -1)
    depth = np.zeros(shape, np.float32)
    for i in np.argsort(-z):
        cv2.fillConvexPoly(depth, np.round(uv[i] * 16).astype(np.int32), float(z[i]), cv2.LINE_8, 4)
    return depth


@needs_rs
def test_eval_script_scores_capsule_against_mesh_on_a_rendered_capture(tmp_path):
    """End to end on a synthetic capture: the RS arm at rest (fingers half
    open) in front of a wall, rendered from its own triangles. The capsule
    leaves arm points within 3 cm of the URDF surface; the mesh mask none.
    Same metrics the real-data report uses (docs/NVBLOX.md)."""
    from test_link_self_mask import _pin_link_poses, _posed, _rs_truth_triangles

    q = np.zeros(6)
    kin = _rs_kin()
    tris = lm._subdivide(_posed(_rs_truth_triangles(), _pin_link_poses(kin, q, 0.025)), 0.01)
    T = look_at((0.10, 0.55, 0.45), (0.10, 0.0, 0.15))
    arm = _painter_depth(tris, T)
    depth = np.where(arm > 0, arm, 1.6).astype(np.float32)          # a wall behind the arm
    cap = tmp_path / "cap.npz"
    np.savez(cap, depth=np.stack([depth, depth]), K=np.stack([K, K]), T_cam2base=T)
    ev = _eval_script()
    report = ev.evaluate(ev.load_capture(cap), arm="rebot_rs", q=q, gripper_pos=3.1, stride=1)
    cap_r, mesh_r = report["capsule"], report["mesh"]
    assert cap_r["unmasked_within_3cm"] > 100
    assert mesh_r["unmasked_within_3cm"] == 0 and mesh_r["unmasked_within_5cm"] == 0
    assert mesh_r["occupied_voxels_within_3cm"] == 0 < cap_r["occupied_voxels_within_3cm"]
    assert mesh_r["mask_ms_median"] < 20.0 and report["frames"] == 2


class _FakeTsdf:
    """Stands in for the warp TSDF backend: records the depth each method
    integrates, and reports one occupied voxel 9 mm above the gripper's top
    face and 5 cm above the joint chain, for the capsule run only."""

    runs: typing.ClassVar[list] = []

    def __init__(self, voxel, region_min, region_max):
        self.voxel, self.depths = voxel, []
        _FakeTsdf.runs.append(self)

    def integrate_depth(self, depth, K, T):
        self.depths.append(np.array(depth))

    def query(self, region_min, region_max):
        first = len(_FakeTsdf.runs) == 1
        pts = np.array([[0.202, 0.0, 0.218 + 0.05]]) if first else np.zeros((0, 3))
        return {"points": pts.astype(np.float32), "grid": np.full((4, 4, 4), 0.5, np.float32),
                "origin": np.asarray(region_min, np.float32), "voxel": 0.25}


@needs_rs
def test_eval_script_runs_each_mask_through_a_tsdf_backend(tmp_path):
    """The optional TSDF pass (the production non-native path: masked pixels
    zeroed, then integrated) gets each method's own zeroed depth and scores
    the backend's occupied voxels with the same surface metrics."""
    from test_link_self_mask import _pin_link_poses, _posed, _rs_truth_triangles

    q = np.zeros(6)
    tris = lm._subdivide(_posed(_rs_truth_triangles(), _pin_link_poses(_rs_kin(), q, 0.025)), 0.01)
    T = look_at((0.10, 0.55, 0.45), (0.10, 0.0, 0.15))
    arm = _painter_depth(tris, T)
    depth = np.where(arm > 0, arm, 1.6).astype(np.float32)
    cap = tmp_path / "cap.npz"
    np.savez(cap, depth=depth, K=K, T_cam2base=T)
    _FakeTsdf.runs = []
    ev = _eval_script()
    report = ev.evaluate(ev.load_capture(cap), arm="rebot_rs", q=q, gripper_pos=3.1, stride=1,
                         tsdf_factory=_FakeTsdf)
    capsule_run, mesh_run = _FakeTsdf.runs
    assert len(capsule_run.depths) == len(mesh_run.depths) == 1
    assert (mesh_run.depths[0][arm > 0] == 0).all(), "the mesh run integrated arm depth"
    assert (capsule_run.depths[0][arm > 0] > 0).any(), "premise: the capsule leaves some arm depth"
    assert report["capsule"]["tsdf_occupied_within_3cm"] == 1 and report["mesh"]["tsdf_occupied_within_5cm"] == 0
    assert report["capsule"]["tsdf_nearest_to_chain_m"] == pytest.approx(0.05, abs=1e-3)
    assert report["mesh"]["tsdf_nearest_to_chain_m"] == float("inf")
    assert report["capsule"]["tsdf_harness_clearance_m"] == pytest.approx(0.5)
