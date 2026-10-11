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
