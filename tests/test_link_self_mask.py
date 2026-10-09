"""B39: fusion's robot self-mask from the arm's LINK GEOMETRY (real-rig frames).

B32a (#257) taught both fusion paths to drop a detection that is more than
half robot pixels and to keep robot pixels out of 3D, but only on frames that
carry a render self-mask (`Frame.robot_mask`: Isaac with
CASCADE_ISAAC_PIXEL_MASK=1). The real rig's frames carry none, so there the
arm's upper link still becomes a belief whenever it stands outside the base
cylinder (on Isaac that one phantom WAS the bare scene's 17.9-19.6 % phantom
rate, docs/evidence/b32-fusion-self-mask-20261008/).

`perception/link_mask.py` rasterizes the arm's link geometry, posed by FK at
the joint sample nearest the frame's capture time, into the camera, and the
existing gate consumes it like a render mask. Opt-in:
`workspace_filter.link_self_mask.enabled` (default false = byte-identical).

The truth is independent of the code under test: the test's own triangles
(written to STL/OBJ for the module to read back), posed by the same link
transforms and filled ONE TRIANGLE PER CALL. `cv2.fillPoly` with a list of
polygons fills even-odd, so a closed mesh's front and back faces cancel
(pinned below).
"""
from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from conftest import JOINT_SIGNS, URDF, needs_pin
from cascade.memory.beliefs import BeliefStore
from cascade.perception.workspace import WorkspaceFilter
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import Detection, Frame, RobotState

W, H = 640, 360
FX = 18.0 / 20.955 * W  # the Isaac bridge optics (18 mm / 20.955 mm aperture) at 640 x 360
K = np.array([[FX, 0.0, W / 2], [0.0, FX, H / 2], [0.0, 0.0, 1.0]])

_BOX_FACES = np.array([(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                       (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)])


def _lm():
    """The module under test, imported lazily so the premise tests run on main."""
    from cascade.perception import link_mask
    return link_mask


def _cam(name: str) -> np.ndarray:
    from cascade.config import load_profile
    return np.asarray(load_profile("cameras", name).extrinsics.T, dtype=float).reshape(4, 4)


def _box_tris(size, center=(0.0, 0.0, 0.0)) -> np.ndarray:
    h = np.asarray(size, dtype=float) / 2
    v = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
    return v[_BOX_FACES] + np.asarray(center, dtype=float)


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# A synthetic arm whose links are NOT convex: an L-shaped upper link (the bar
# a per-link hull would fill in), a fore link authored in millimetres behind a
# package:// URI, and a U-shaped hand (palm + two fingers) read from OBJ.
UPPER_TRIS = np.concatenate([_box_tris((0.30, 0.05, 0.05), (0.15, 0.0, 0.0)),
                             _box_tris((0.05, 0.05, 0.14), (0.275, 0.0, -0.07))])
FORE_TRIS_MM = _box_tris((40.0, 40.0, 160.0), (0.0, 0.0, -80.0))
HAND_TRIS = np.concatenate([_box_tris((0.02, 0.08, 0.04)),
                            _box_tris((0.06, 0.015, 0.04), (0.04, 0.0325, 0.0)),
                            _box_tris((0.06, 0.015, 0.04), (0.04, -0.0325, 0.0))])
TRUTH_TRIS = {"base_link": _box_tris((0.14, 0.20, 0.075), (0.0, 0.0, 0.0375)),
              "upper": UPPER_TRIS, "fore": FORE_TRIS_MM * 0.001, "hand": HAND_TRIS}

SYNTH_URDF = """<?xml version="1.0"?>
<robot name="synth">
  <link name="base_link">
    <collision><origin xyz="0 0 0.0375" rpy="0 0 0"/><geometry><box size="0.14 0.2 0.075"/></geometry></collision>
  </link>
  <link name="upper"><collision><geometry><mesh filename="meshes/upper.stl"/></geometry></collision></link>
  <link name="fore">
    <collision><geometry><mesh filename="package://synth/meshes/fore.stl" scale="0.001 0.001 0.001"/></geometry></collision>
  </link>
  <link name="hand"><collision><geometry><mesh filename="meshes/hand.obj"/></geometry></collision></link>
  <joint name="j1" type="revolute"><parent link="base_link"/><child link="upper"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="upper"/><child link="fore"/>
    <origin xyz="0.3 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="1" velocity="1"/></joint>
  <joint name="j3" type="fixed"><parent link="fore"/><child link="hand"/><origin xyz="0 0 -0.18"/></joint>
</robot>
"""

Q_A = (0.2, -0.4)    # the upper link high and well in the side camera's view
Q_B = (0.6, -1.2)
Q_C = (-0.3, 0.5)
PROP_CENTER = (0.45, 0.15, 0.025)  # a 5 cm cube, 73 px from the arm in the side view at Q_A


def _synth_poses(state) -> list[dict[str, np.ndarray]]:
    """FK of SYNTH_URDF, written out (no Pinocchio): j1 at z 0.4 and j2 at
    x 0.3 about +y, hand fixed 0.18 below the fore link."""
    a, b = float(state.q[0]), float(state.q[1])
    up = _T(_ry(a), (0.0, 0.0, 0.40))
    fore = up @ _T(_ry(b), (0.30, 0.0, 0.0))
    hand = fore @ _T(np.eye(3), (0.0, 0.0, -0.18))
    return [{"base_link": np.eye(4), "upper": up, "fore": fore, "hand": hand}]


def _write_binary_stl(path: Path, tris: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"binary stl written by test_link_self_mask".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in np.asarray(tris, dtype=np.float32):
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0) + t.tobytes() + b"\0\0")


def _write_ascii_stl(path: Path, tris: np.ndarray) -> None:
    lines = ["solid synth"]
    for t in tris:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in t]
        lines += ["    endloop", "  endfacet"]
    lines.append("endsolid synth")
    path.write_text("\n".join(lines) + "\n")


def _write_obj(path: Path, tris: np.ndarray) -> None:
    flat = np.asarray(tris, dtype=float).reshape(-1, 3)
    lines = ["# obj written by test_link_self_mask"] + [f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in flat]
    lines += [f"f {3 * i + 1}/1/1 {3 * i + 2}/1/1 {3 * i + 3}/1/1" for i in range(len(tris))]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def synth_urdf(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("b39") / "synth"
    (root / "meshes").mkdir(parents=True)
    _write_binary_stl(root / "meshes" / "upper.stl", UPPER_TRIS)
    _write_ascii_stl(root / "meshes" / "fore.stl", FORE_TRIS_MM)
    _write_obj(root / "meshes" / "hand.obj", HAND_TRIS)
    path = root / "synth.urdf"
    path.write_text(SYNTH_URDF)
    return path


@pytest.fixture(scope="module")
def synth_geometry(synth_urdf):
    return _lm().LinkGeometry.from_urdf(synth_urdf)


def _posed(tris_by_link: dict, poses: dict, links=None) -> np.ndarray:
    names = list(tris_by_link) if links is None else list(links)
    return np.concatenate([tris_by_link[n] @ poses[n][:3, :3].T + poses[n][:3, 3] for n in names])


def _fill(tris_base: np.ndarray, T_cam2base: np.ndarray, shape=(H, W), K_=K) -> np.ndarray:
    """Per-triangle truth, independent of the module: one fillConvexPoly per triangle."""
    R, t = T_cam2base[:3, :3], T_cam2base[:3, 3]
    pc = (np.asarray(tris_base, dtype=float) - t) @ R          # base -> camera
    assert (pc[..., 2] > 0.05).all(), "truth fixture expects every vertex in front of the camera"
    uv = np.stack([K_[0, 0] * pc[..., 0] / pc[..., 2] + K_[0, 2],
                   K_[1, 1] * pc[..., 1] / pc[..., 2] + K_[1, 2]], axis=-1)
    m = np.zeros(shape, np.uint8)
    for tri in np.round(uv * 16).astype(np.int32):
        cv2.fillConvexPoly(m, tri, 1, cv2.LINE_8, 4)
    return m.astype(bool)


def _metrics(mask: np.ndarray, truth: np.ndarray) -> dict:
    n = float(truth.sum())
    assert n > 0
    return {"coverage": float((mask & truth).sum()) / n,
            "bloat": float((mask & ~truth).sum()) / n,
            "iou": float((mask & truth).sum()) / float((mask | truth).sum())}


def _state(q, t, **kw) -> RobotState:
    return RobotState(q=np.asarray(q, dtype=float), t=float(t), **kw)


def _link_mask(geometry, state_fn, *, dilate_px=None, max_skew_s=None, poses=_synth_poses, **arm_kw):
    kw = {}
    if dilate_px is not None:
        kw["dilate_px"] = dilate_px
    if max_skew_s is not None:
        kw["max_skew_s"] = max_skew_s
    lm = _lm().LinkSelfMask(**kw)
    lm.add_arm("arm0", geometry, poses, state_fn, **arm_kw)
    return lm


# ── the fusion scene: a frame of the side camera WITHOUT a render mask ──────


def _det(label, mask, conf=0.6):
    ys, xs = np.nonzero(mask)
    bbox = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
    return Detection(label, conf, bbox, mask=mask)


def _scene(q=Q_A, *, t=None, T_on_frame=True, render_mask=False):
    """The side camera looking at the synthetic arm and one red 5 cm cube.

    The detector names the upper link "biplane" (what YOLOE called the reBot's
    upper link on Isaac, B32). Depth: the table plane, the cube at its centre's
    camera depth, every robot pixel at the upper link's centroid depth.
    """
    cam = _cam("isaac_side")
    poses = _synth_poses(_state(q, 0.0))[0]
    robot = _fill(_posed(TRUTH_TRIS, poses), cam)
    upper = _fill(_posed(TRUTH_TRIS, poses, ["upper"]), cam)
    prop = _fill(_box_tris((0.05, 0.05, 0.05), PROP_CENTER), cam)
    R, c = cam[:3, :3], cam[:3, 3]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us, float)], -1) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        table = np.where(rays[..., 2] < -1e-6, -c[2] / rays[..., 2], 0.0)
    depth = table.astype(np.float32)
    depth[prop] = float((np.asarray(PROP_CENTER) - c) @ R[:, 2])
    centroid = _posed(TRUTH_TRIS, poses, ["upper"]).reshape(-1, 3).mean(axis=0)
    depth[robot] = float((centroid - c) @ R[:, 2])
    rgb = np.full((H, W, 3), 90, dtype=np.uint8)
    rgb[prop] = (30, 30, 200)      # BGR red
    rgb[robot] = (128, 128, 128)
    frame = Frame(rgb=rgb, depth_m=depth, K=K.copy(), t=time.monotonic() if t is None else t,
                  frame_id=1, depth_source="sensor", T_base_cam=cam.copy() if T_on_frame else None)
    if render_mask:
        frame.robot_mask = robot.copy()
    dets = [_det("biplane", upper), _det("cube", prop)]
    return frame, dets, robot


def _by_label(beliefs) -> dict:
    out = {}
    for b in beliefs.all():
        for name in {b.label, *b.aliases}:
            out[name] = b
    return out


def _watch(dets, *, link_mask=None, occupancy=None):
    stream = SimpleNamespace(name="side", latest=Mock(), set_overlay=Mock())
    depth = SimpleNamespace(ensure_depth=Mock(side_effect=lambda frame: frame))
    detector = SimpleNamespace(detect=Mock(return_value=dets))
    cam = WatchedCamera(stream, depth, None, fuse=True)
    beliefs = BeliefStore()
    kw = {} if link_mask is None else {"link_mask": link_mask}
    watch = WorldWatcher([cam], detector, beliefs, workspace=WorkspaceFilter(),
                         harness=SimpleNamespace(heartbeat=Mock()), occupancy=occupancy, **kw)
    return watch, cam, beliefs


def _fuse_watcher(frame, dets, link_mask=None):
    watch, cam, beliefs = _watch(dets, link_mask=link_mask)
    cam.stream.latest.return_value = frame
    watch._tick(cam)
    return _by_label(beliefs)


def _fuse_update(frame, dets, link_mask=None, *, primary=False):
    """`SkillRuntime._update_beliefs_from_frame`: the get_observation path
    (primary camera: T from the runtime's extrinsics) or `_reobserve`'s other
    cameras (T passed in)."""
    from cascade.skills.runtime import SkillRuntime
    cam = _cam("isaac_side")
    rt = SimpleNamespace(_workspace=WorkspaceFilter(), beliefs=BeliefStore(),
                         extrinsics=SimpleNamespace(cam_to_base=lambda: cam.copy()))
    if link_mask is not None:
        rt._link_self_mask = link_mask
    if primary:
        SkillRuntime._update_beliefs_from_frame(rt, frame, dets)
    else:
        SkillRuntime._update_beliefs_from_frame(rt, frame, dets, T=cam.copy())
    return _by_label(rt.beliefs)


def _partial_runtime(frame, dets, link_mask=None, log=None):
    """A SkillRuntime with only what get_observation / _reobserve touch."""
    from cascade.skills.runtime import SkillRuntime
    rt = SkillRuntime.__new__(SkillRuntime)

    def detect(f, classes=None):
        if log is not None:
            log.append("detect")
        return dets

    rt.camera = SimpleNamespace(get_frame=Mock(return_value=frame), set_overlay=Mock())
    rt.depth = SimpleNamespace(ensure_depth=lambda f: f)
    rt.detector = SimpleNamespace(detect=Mock(side_effect=detect))
    rt._default_classes = None
    rt.held_object = None
    rt._held_det_label = None
    rt.watcher = None
    rt._workspace = WorkspaceFilter()
    rt.beliefs = BeliefStore()
    cam = _cam("isaac_side")
    rt.extrinsics = SimpleNamespace(cam_to_base=lambda: cam.copy())
    rt._arm = SimpleNamespace(harness=SimpleNamespace(heartbeat=Mock()), raw=SimpleNamespace(connected=False))
    rt.last_frame = None
    rt.memory = SimpleNamespace(add=Mock())
    rt.cfg = {}
    if link_mask is not None:
        rt._link_self_mask = link_mask
    return rt


def _beliefs_snapshot(seen: dict) -> dict:
    return {k: (b.label, b.color, [float(v) for v in b.position],
                None if b.extent is None else [float(v) for v in b.extent], b.observations)
            for k, b in sorted(seen.items())}


# ── premises: pass on main by design ────────────────────────────────────────


def test_premise_fillpoly_with_a_list_is_even_odd_so_the_truth_draws_one_triangle_per_call():
    a = np.array([[5, 5], [40, 5], [5, 40]], np.int32)
    b = np.array([[10, 10], [45, 10], [10, 45]], np.int32)
    listed = np.zeros((50, 50), np.uint8)
    cv2.fillPoly(listed, [a, b], 1)
    one_by_one = np.zeros((50, 50), np.uint8)
    for poly in (a, b):
        cv2.fillConvexPoly(one_by_one, poly, 1)
    assert listed[20, 20] == 0 and one_by_one[20, 20] == 1   # the overlap cancels in a list


@pytest.mark.parametrize("path", ["watcher", "get_observation", "reobserve_other_camera"])
def test_premise_without_a_self_mask_the_arm_link_becomes_a_belief(path):
    frame, dets, _ = _scene(T_on_frame=path != "get_observation")
    if path == "watcher":
        seen = _fuse_watcher(frame, dets)
    else:
        seen = _fuse_update(frame, dets, primary=path == "get_observation")
    # The upper link stands outside the base cylinder: on the real rig (no
    # render mask) it is fused as an object, exactly the B32 Isaac phantom.
    assert "biplane" in seen and "cube" in seen
    assert np.hypot(*seen["biplane"].position[:2]) > WorkspaceFilter().base_radius_m
    assert seen["biplane"].position[2] > 0.25


@pytest.mark.parametrize("path", ["watcher", "get_observation"])
def test_premise_a_render_mask_drops_the_link(path):
    frame, dets, _ = _scene(render_mask=True, T_on_frame=path == "watcher")
    seen = _fuse_watcher(frame, dets) if path == "watcher" else _fuse_update(frame, dets, primary=True)
    assert "biplane" not in seen and "cube" in seen


# Captured on main 4e896c3 (an export, before this change): the flag-off
# fusion of `_scene(t=1000.0)` through both paths (identical by construction:
# the watcher reads T from the frame, get_observation from the extrinsics).
# Flag off must reproduce it.
_GOLD = {
    "biplane": ("biplane", "gray", [0.1855311000012653, 0.004965818029319958, 0.3430471937627234],
                [0.3146427493378676, 0.17534878936765724, 2.0574872405166647e-16], 1),
    "cube": ("cube", "red", [0.4505903089633304, 0.1498870593248624, 0.024112744574391165],
             [0.07709733190262572, 0.07836441211235301, 1.4483840264418883e-16], 1),
}
GOLDEN_FLAG_OFF = {"watcher": _GOLD, "get_observation": _GOLD}


def _assert_golden(seen: dict, path: str) -> None:
    snap = _beliefs_snapshot(seen)
    gold = GOLDEN_FLAG_OFF[path]
    assert sorted(snap) == sorted(gold)
    for key, (label, color, pos, ext, obs) in gold.items():
        got = snap[key]
        assert got[0] == label and got[1] == color and got[4] == obs
        assert got[2] == pytest.approx(pos, abs=1e-9)
        assert (got[3] is None) == (ext is None)
        if ext is not None:
            assert got[3] == pytest.approx(ext, abs=1e-9)


@pytest.mark.parametrize("path", ["watcher", "get_observation"])
def test_golden_flag_off_is_main_behaviour(path):
    frame, dets, _ = _scene(t=1000.0, T_on_frame=path == "watcher")
    seen = _fuse_watcher(frame, dets) if path == "watcher" else _fuse_update(frame, dets, primary=True)
    _assert_golden(seen, path)


# ── geometry: meshes, primitives, fallbacks ─────────────────────────────────


def _blank(t: float, shape=(H, W)) -> Frame:
    return Frame(rgb=np.zeros(shape + (3,), np.uint8), depth_m=None, K=K.copy(), t=t)


def test_meshes_and_primitives_load_from_the_urdf(synth_geometry):
    g = synth_geometry
    assert set(g.links) == {"base_link", "upper", "fore", "hand"}
    # binary STL, ASCII STL in millimetres behind package://, OBJ, a box primitive
    assert g.sources == {"base_link": "box", "upper": "mesh", "fore": "mesh", "hand": "mesh"}
    for link, tris in TRUTH_TRIS.items():
        pts = np.concatenate(g.pieces[link])
        flat = tris.reshape(-1, 3)
        # the pieces span exactly the link's triangles (their extreme points are hull vertices)
        assert np.allclose(pts.min(axis=0), flat.min(axis=0), atol=1e-6)
        assert np.allclose(pts.max(axis=0), flat.max(axis=0), atol=1e-6)
    assert len(g.pieces["upper"]) > 1   # the L is split into cells, not one hull
    # Every mesh piece is LOCAL: its triangles are split until no edge exceeds
    # cell_m and bucketed by centroid, so each vertex lies within 2/3 cell_m of
    # a centroid inside its cell: a piece spans at most 7/3 cell_m per axis and
    # cannot bridge a concavity (the 30 cm faces of the L are 2 triangles each).
    for link in ("upper", "fore", "hand"):
        for piece in g.pieces[link]:
            assert np.ptp(piece, axis=0).max() <= 7 / 3 * 0.05 + 1e-9

def test_unavailable_meshes_fall_back_to_a_capsule_and_say_why(tmp_path):
    lm = _lm()
    root = tmp_path / "pkg"
    (root / "meshes").mkdir(parents=True)
    (root / "meshes" / "lfs.stl").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:0123\nsize 4096\n")
    (root / "meshes" / "part.dae").write_text("<COLLADA/>")
    urdf = root / "r.urdf"
    urdf.write_text("""<robot name="r">
      <link name="a"><collision><geometry><mesh filename="meshes/lfs.stl"/></geometry></collision></link>
      <link name="b"><collision><geometry><mesh filename="meshes/missing.stl"/></geometry></collision></link>
      <link name="c"><collision><geometry><mesh filename="meshes/part.dae"/></geometry></collision></link>
      <link name="tcp"/>
      <joint name="ab" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0 0 0.3"/>
        <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
      <joint name="bc" type="fixed"><parent link="b"/><child link="c"/><origin xyz="0.2 0 0"/></joint>
      <joint name="ct" type="fixed"><parent link="c"/><child link="tcp"/><origin xyz="0.1 0 0"/></joint>
    </robot>""")
    with pytest.raises(lm.MeshUnavailable, match="git-lfs"):
        lm.load_mesh_triangles(root / "meshes" / "lfs.stl")
    with pytest.raises(lm.MeshUnavailable, match="no such file"):
        lm.load_mesh_triangles(root / "meshes" / "missing.stl")
    with pytest.raises(lm.MeshUnavailable, match=r"\.dae"):
        lm.load_mesh_triangles(root / "meshes" / "part.dae")
    g = lm.LinkGeometry.from_urdf(urdf, capsule_radius_m=0.05)
    assert g.sources["a"].startswith("capsule") and "git-lfs" in g.sources["a"]
    assert g.sources["b"].startswith("capsule") and "no such file" in g.sources["b"]
    assert g.sources["c"].startswith("capsule") and ".dae" in g.sources["c"]
    # a link with no collision geometry is a frame, not a body: nothing is invented for it
    assert "tcp" not in g.pieces and "tcp" not in g.sources
    # each capsule runs from the link origin to its children's joints, 5 cm thick
    a = np.concatenate(g.pieces["a"])
    assert a.min(axis=0) == pytest.approx([-0.05, -0.05, -0.05], abs=0.01)
    assert a.max(axis=0) == pytest.approx([0.05, 0.05, 0.35], abs=0.01)
    b = np.concatenate(g.pieces["b"])
    assert b.max(axis=0)[0] == pytest.approx(0.25, abs=0.01)


def test_a_millimetre_mesh_without_its_scale_is_refused_not_subdivided_forever(tmp_path):
    """FORE_TRIS_MM read without `scale` is a 160 m link: 5 cm cells would
    need billions of triangles. It is refused (capsule, reason recorded) in
    bounded memory instead."""
    lm = _lm()
    (tmp_path / "meshes").mkdir()
    _write_binary_stl(tmp_path / "meshes" / "fore.stl", FORE_TRIS_MM)
    urdf = tmp_path / "r.urdf"
    urdf.write_text('<robot name="r"><link name="fore"><collision><geometry>'
                    '<mesh filename="meshes/fore.stl"/></geometry></collision></link></robot>')
    g = lm.LinkGeometry.from_urdf(urdf)
    assert g.sources["fore"].startswith("capsule") and "too large" in g.sources["fore"]
    assert lm.LinkGeometry.from_urdf(urdf, cell_m=500.0).sources["fore"] == "mesh"


def _primitive_surface(kind: str, rng, n: int) -> np.ndarray:
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    th = rng.uniform(0, 2 * np.pi, n)
    if kind == "box":
        p = rng.uniform(-0.5, 0.5, (n, 3))
        p[np.arange(n), rng.integers(0, 3, n)] = rng.choice([-0.5, 0.5], n)
        corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
        return np.concatenate([p, corners]) * [0.1, 0.2, 0.3]   # the corners are its extreme points
    if kind == "sphere":
        return 0.05 * u
    if kind == "cylinder":
        side = np.stack([0.04 * np.cos(th), 0.04 * np.sin(th), rng.uniform(-0.1, 0.1, n)], -1)
        cap = np.stack([0.04 * np.cos(th), 0.04 * np.sin(th), rng.choice([-0.1, 0.1], n)], -1)
        return np.concatenate([side, cap])
    side = np.stack([0.03 * np.cos(th), 0.03 * np.sin(th), rng.uniform(-0.1, 0.1, n)], -1)
    ends = 0.03 * u + np.c_[np.zeros((n, 2)), rng.choice([-0.1, 0.1], n)]
    return np.concatenate([side, ends])


@pytest.mark.parametrize("kind, xml, reach", [
    ("box", '<box size="0.1 0.2 0.3"/>', 1.0),
    ("cylinder", '<cylinder radius="0.04" length="0.2"/>', 1.05),
    ("sphere", '<sphere radius="0.05"/>', 1.15),
    ("capsule", '<capsule radius="0.03" length="0.2"/>', 1.15),
])
def test_primitive_pieces_contain_the_primitive(tmp_path, kind, xml, reach):
    from cascade.types import pose_to_transform
    urdf = tmp_path / "p.urdf"
    urdf.write_text(f'<robot name="p"><link name="l"><collision><origin xyz="0.1 0 0" rpy="0.2 0 0.3"/>'
                    f'<geometry>{xml}</geometry></collision></link></robot>')
    g = _lm().LinkGeometry.from_urdf(urdf)
    assert g.sources["l"] == kind
    pts = np.concatenate(g.pieces["l"])
    rng = np.random.default_rng(7)
    To = pose_to_transform([0.1, 0.0, 0.0, 0.2, 0.0, 0.3])
    surf = _primitive_surface(kind, rng, 3000) @ To[:3, :3].T + To[:3, 3]
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    hull_support = (pts @ dirs.T).max(axis=0)
    shape_support = (surf @ dirs.T).max(axis=0)
    centre = To[:3, 3] @ dirs.T
    # containment: the pieces' support exceeds the primitive's in every direction ...
    assert np.all(hull_support >= shape_support - 1e-9)
    # ... without inflating it beyond `reach` (circumscribed polygons, not boxes)
    assert np.all(hull_support - centre <= reach * (shape_support - centre) + 1e-6)


# ── rasterization against the per-triangle truth ────────────────────────────


@pytest.mark.parametrize("q", [Q_A, Q_B, Q_C])
@pytest.mark.parametrize("camera", ["isaac", "isaac_side"])
def test_link_mask_covers_the_true_silhouette_with_bounded_bloat(synth_geometry, q, camera):
    T = _cam(camera)
    t = 50.0
    truth = _fill(_posed(TRUTH_TRIS, _synth_poses(_state(q, t))[0]), T)
    assert truth.sum() > 2000
    exact = _link_mask(synth_geometry, lambda: _state(q, t), dilate_px=0).mask_for(_blank(t), T)
    m0 = _metrics(exact, truth)
    assert m0["coverage"] >= 0.995          # 1 by construction, minus rasterisation at the edge
    assert m0["bloat"] <= 0.06
    shipped = _link_mask(synth_geometry, lambda: _state(q, t))   # default dilation
    m2 = _metrics(shipped.mask_for(_blank(t), T), truth)
    assert m2["coverage"] == 1.0
    assert m2["bloat"] <= 0.15 and m2["iou"] >= 0.85
    assert shipped.last["reason"] == "link_mask" and shipped.last["pixels"] > 0


def test_cells_bound_the_bloat_a_per_link_hull_would_add(synth_urdf):
    lm = _lm()
    cells = lm.LinkGeometry.from_urdf(synth_urdf)                # cell_m 0.05 (shipped)
    hulls = lm.LinkGeometry.from_urdf(synth_urdf, cell_m=10.0)   # one convex piece per link
    T, t = _cam("isaac_side"), 50.0
    truth = _fill(_posed(TRUTH_TRIS, _synth_poses(_state(Q_A, t))[0]), T)
    bc = _metrics(_link_mask(cells, lambda: _state(Q_A, t), dilate_px=0).mask_for(_blank(t), T), truth)
    bh = _metrics(_link_mask(hulls, lambda: _state(Q_A, t), dilate_px=0).mask_for(_blank(t), T), truth)
    assert bc["coverage"] >= 0.995 and bh["coverage"] >= 0.995
    # one hull per link fills in the L of the upper link and the U of the hand
    assert bc["bloat"] < 0.5 * bh["bloat"]


def test_the_hull_reduction_changes_the_cost_not_the_silhouette(synth_urdf, monkeypatch):
    lm = _lm()
    reduced = lm.LinkGeometry.from_urdf(synth_urdf)
    monkeypatch.setattr(lm, "_hull_vertices", lambda pts: np.unique(np.asarray(pts, dtype=float), axis=0))
    every_vertex = lm.LinkGeometry.from_urdf(synth_urdf)
    count = lambda g: sum(len(p) for pieces in g.pieces.values() for p in pieces)   # noqa: E731
    assert count(every_vertex) > count(reduced)
    ring = np.ones((3, 3), np.uint8)
    for q in (Q_A, Q_B):
        for camera in ("isaac", "isaac_side"):
            T, t = _cam(camera), 9.0
            a = _link_mask(reduced, lambda: _state(q, t)).mask_for(_blank(t), T)
            b = _link_mask(every_vertex, lambda: _state(q, t)).mask_for(_blank(t), T)
            # The same polygons; the 2D hull's vertex lists may differ by
            # (near-)collinear points, whose 1/16 px rounding flips a few edge
            # pixels (measured: 2-8 of 15-20k). Nothing else may differ.
            assert (a ^ b).sum() <= 0.001 * a.sum()
            assert not (a & ~cv2.dilate(b.astype(np.uint8), ring).astype(bool)).any()
            assert not (b & ~cv2.dilate(a.astype(np.uint8), ring).astype(bool)).any()


def test_pieces_behind_through_or_beside_the_camera():
    lm = _lm()
    T = np.eye(4)    # camera frame == base frame: the camera looks along +z

    def box(lo, hi):
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    behind = box((-0.1, -0.1, -1.2), (0.1, 0.1, -0.8))
    aside = box((4.9, -0.1, 0.9), (5.1, 0.1, 1.1))
    assert not lm.rasterize_pieces([behind], K, T, (H, W)).any()
    assert not lm.rasterize_pieces([aside], K, T, (H, W)).any()
    # through the camera plane, on the optical axis: what is in front fills the view
    through = lm.rasterize_pieces([box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))], K, T, (H, W))
    assert through.dtype == bool and through.shape == (H, W) and through.all()
    # through the camera plane but all at x > 0: only image columns right of cx
    right = lm.rasterize_pieces([box((0.05, -0.1, -0.5), (0.25, 0.1, 0.5))], K, T, (H, W))
    cols = np.nonzero(right.any(axis=0))[0]
    assert right.any() and cols.min() > K[0, 2] and cols.max() == W - 1
    # dilation is a real knob, and an empty piece list is an empty mask
    small = lm.rasterize_pieces([box((-0.05, -0.05, 0.9), (0.05, 0.05, 1.1))], K, T, (H, W))
    big = lm.rasterize_pieces([box((-0.05, -0.05, 0.9), (0.05, 0.05, 1.1))], K, T, (H, W), dilate_px=3)
    assert big.sum() > small.sum() > 0 and not (small & ~big).any()
    assert not lm.rasterize_pieces([], K, T, (H, W)).any()


# ── through the real fusion paths ───────────────────────────────────────────


def _fuse(path, frame, dets, link=None):
    if path == "watcher":
        return _fuse_watcher(frame, dets, link)
    return _fuse_update(frame, dets, link, primary=path == "get_observation")


@pytest.mark.parametrize("path", ["watcher", "get_observation", "reobserve_other_camera"])
def test_an_arm_detection_is_dropped_with_the_link_mask_exactly_as_with_a_render_mask(synth_geometry, path):
    frame, dets, _ = _scene(T_on_frame=path != "get_observation")
    rendered, rdets, _ = _scene(t=frame.t, T_on_frame=path != "get_observation", render_mask=True)
    link = _link_mask(synth_geometry, lambda: _state(Q_A, time.monotonic()))
    with_link = _fuse(path, frame, dets, link)
    with_render = _fuse(path, rendered, rdets)
    assert "biplane" not in with_link and "cube" in with_link
    assert _beliefs_snapshot(with_link) == _beliefs_snapshot(with_render)
    assert link.last["reason"] == "link_mask"
    assert frame.robot_mask is None    # the camera's frame is never written


def test_detections_away_from_the_arm_are_unaffected(synth_geometry):
    frame, dets, _ = _scene()
    link = _link_mask(synth_geometry, lambda: _state(Q_A, time.monotonic()))
    off = _fuse_watcher(frame, dets)
    on = _fuse_watcher(frame, dets, link)
    assert _beliefs_snapshot({"cube": on["cube"]}) == _beliefs_snapshot({"cube": off["cube"]})
    assert np.array_equal(on["cube"].points, off["cube"].points)
    assert not (link.mask_for(frame, _cam("isaac_side")) & dets[1].mask).any()


def test_attach_returns_a_fusion_copy_and_leaves_the_frame_alone(synth_geometry):
    frame, _, robot = _scene()
    link = _link_mask(synth_geometry, lambda: _state(Q_A, time.monotonic()))
    out = link.attach(frame, _cam("isaac_side"))
    assert out is not frame and frame.robot_mask is None
    assert out.robot_mask.dtype == bool and out.robot_mask.shape == (H, W)
    assert out.t == frame.t and out.frame_id == frame.frame_id and out.depth_m is frame.depth_m
    assert out.payload_mask is None
    assert _metrics(out.robot_mask, robot)["coverage"] == 1.0


def test_reobserve_fuses_with_the_link_mask_and_samples_before_inference(synth_geometry):
    from cascade.skills.runtime import SkillRuntime
    frame, dets, _ = _scene(T_on_frame=False)
    log = []

    def state_fn():
        log.append("state")
        return _state(Q_A, time.monotonic())

    rt = _partial_runtime(frame, dets, _link_mask(synth_geometry, state_fn), log)
    SkillRuntime._reobserve(rt, frames=1)
    seen = _by_label(rt.beliefs)
    assert "biplane" not in seen and "cube" in seen
    assert log.index("state") < log.index("detect")


def test_reobserve_other_cameras_get_the_link_mask_too(synth_geometry):
    from cascade.skills.runtime import SkillRuntime
    primary, dets, _ = _scene(T_on_frame=False, render_mask=True)   # Isaac-like: rendered
    other, _, _ = _scene(t=primary.t)                               # real-rig-like: none
    log = []

    def state_fn():
        log.append("state")
        return _state(Q_A, time.monotonic())

    link = _link_mask(synth_geometry, state_fn)
    rt = _partial_runtime(primary, dets, link, log)
    side = SimpleNamespace(stream=SimpleNamespace(name="side", get_frame=Mock(return_value=other)),
                           depth=SimpleNamespace(ensure_depth=lambda f: f),
                           extrinsics=SimpleNamespace(cam_to_base=lambda: _cam("isaac_side")), fuse=True)
    rt.watcher = SimpleNamespace(_cams=[SimpleNamespace(fuse=True), side])
    SkillRuntime._reobserve(rt, frames=1)
    seen = _by_label(rt.beliefs)
    assert "biplane" not in seen and "cube" in seen
    # the rendered primary frame never reaches the link mask; the other camera's
    # frame is sampled for before inference and masked once
    assert log == ["detect", "state", "detect"]
    assert link.counts == {"link_mask": 1}
    assert other.robot_mask is None and primary.robot_mask is not None


def test_get_observation_samples_the_arm_before_inference_and_reports_no_arm(synth_geometry):
    from cascade.skills.runtime import SkillRuntime
    frame, dets, _ = _scene(T_on_frame=False)
    log = []

    def state_fn():
        log.append("state")
        return _state(Q_A, time.monotonic())

    rt = _partial_runtime(frame, dets, _link_mask(synth_geometry, state_fn), log)
    out = SkillRuntime._describe_observation(rt, frame)
    assert {o["label"] for o in out["objects_visible"]} == {"cube"}
    assert log[0] == "state" and "detect" in log


def test_watcher_samples_before_inference_never_while_paused_and_leaves_occupancy_its_frame(synth_geometry):
    frame, dets, _ = _scene()
    log = []

    def state_fn():
        log.append("state")
        return _state(Q_A, time.monotonic())

    def detect(f, classes=None):
        log.append("detect")
        return dets

    occupancy = SimpleNamespace(refresh=Mock())
    watch, cam, beliefs = _watch(dets, link_mask=_link_mask(synth_geometry, state_fn), occupancy=occupancy)
    watch._detector.detect.side_effect = detect
    cam.stream.latest.return_value = frame
    watch._tick(cam)
    assert log.index("state") < log.index("detect")
    seen = _by_label(beliefs)
    assert "biplane" not in seen and "cube" in seen
    # the occupancy map keeps its own body masking: it gets the camera's frame
    assert occupancy.refresh.call_args[0][0] is frame and frame.robot_mask is None
    # while the arm moves (fusion paused) the watcher never reads the arm
    log.clear()
    moving, _, _ = _scene()
    moving.frame_id = 2
    cam.stream.latest.return_value = moving
    with watch.paused():
        watch._tick(cam)
    assert log == ["detect"]


# ── time alignment: missing evidence stays missing ──────────────────────────


@pytest.mark.parametrize("skew, built", [(-1.0, False), (1.0, False), (0.16, False), (-0.16, False),
                                         (0.14, True), (-0.14, True)])
def test_the_joint_sample_must_be_close_to_the_frame(synth_geometry, skew, built):
    frame, dets, _ = _scene(t=1000.0)
    state_fn = Mock(side_effect=lambda: _state(Q_A, 1000.0 + skew))
    link = _link_mask(synth_geometry, state_fn)
    out = link.attach(frame, _cam("isaac_side"))
    assert link.last["arms"]["arm0"]["skew_s"] == pytest.approx(skew)
    if built:
        assert out.robot_mask is not None and link.last["reason"] == "link_mask"
        return
    assert out is frame and frame.robot_mask is None
    assert link.last["reason"] == "stale_joint_state" and link.counts["stale_joint_state"] == 1
    # fusion is then exactly the flag-off behaviour: the phantom stays
    _assert_golden(_fuse_watcher(frame, dets, link), "watcher")


@pytest.mark.parametrize("state_fn, why", [
    (lambda: None, None),
    (Mock(side_effect=RuntimeError("CAN feedback lost (3 consecutive mechPos read failures)")), "CAN feedback lost"),
    (lambda: _state([np.nan, 0.0], time.monotonic()), None),
])
def test_no_usable_joint_state_builds_no_mask_and_never_raises(synth_geometry, state_fn, why):
    frame, dets, _ = _scene(t=1000.0)
    link = _link_mask(synth_geometry, state_fn)
    assert link.attach(frame, _cam("isaac_side")) is frame
    assert link.last["reason"] == "no_joint_state"
    if why:
        assert why in link.last["arms"]["arm0"]["error"]
    _assert_golden(_fuse_watcher(frame, dets, link), "watcher")


@pytest.mark.parametrize("path", ["watcher", "get_observation"])
def test_a_frame_with_a_render_mask_is_untouched(synth_geometry, path):
    frame, dets, _ = _scene(render_mask=True, T_on_frame=path == "watcher")
    render = frame.robot_mask
    # a different pose: had the link mask been consulted, it would not match
    state_fn = Mock(side_effect=lambda: _state(Q_B, time.monotonic()))
    link = _link_mask(synth_geometry, state_fn)
    assert link.attach(frame, _cam("isaac_side")) is frame and frame.robot_mask is render
    assert link.last["reason"] == "render_mask"
    plain, pdets, _ = _scene(t=frame.t, render_mask=True, T_on_frame=path == "watcher")
    assert _beliefs_snapshot(_fuse(path, frame, dets, link)) == _beliefs_snapshot(_fuse(path, plain, pdets))
    assert state_fn.call_count == 0     # the arm is not even read
    assert frame.robot_mask is render


def test_the_joint_sample_nearest_the_frame_poses_the_mask(synth_geometry):
    t = 500.0
    T = _cam("isaac_side")
    samples = iter([_state(Q_B, t - 0.12), _state(Q_A, t + 0.05), _state(Q_C, t + 0.40)])
    link = _link_mask(synth_geometry, lambda: next(samples, None), dilate_px=0)
    for _ in range(3):
        link.sample()
    m = link.mask_for(_blank(t), T)
    truth = {q: _fill(_posed(TRUTH_TRIS, _synth_poses(_state(q, t))[0]), T) for q in (Q_A, Q_B, Q_C)}
    assert link.last["arms"]["arm0"]["skew_s"] == pytest.approx(0.05)
    assert _metrics(m, truth[Q_A])["coverage"] >= 0.995
    assert _metrics(m, truth[Q_A])["iou"] > _metrics(m, truth[Q_B])["iou"] + 0.2


def test_joint_samples_keep_a_bounded_history_and_reject_unusable_states():
    js = _lm().JointSamples(maxlen=3)
    for k in range(5):
        assert js.record(_state([k, k], 10.0 + k))
    assert len(js) == 3
    state, skew = js.nearest(11.2)          # 10 and 11 were evicted
    assert state.q[0] == 2 and skew == pytest.approx(0.8)
    state, skew = js.nearest(13.4)
    assert state.q[0] == 3 and skew == pytest.approx(-0.4)
    assert not js.record(None)
    assert not js.record(_state([np.nan, 0.0], 20.0))
    assert not js.record(SimpleNamespace(q=np.zeros(2), t=float("nan")))
    assert not js.record(SimpleNamespace(q=None, t=21.0))
    assert len(js) == 3
    assert _lm().JointSamples().nearest(1.0) is None


def test_a_second_arm_and_a_base_pose(synth_geometry):
    t = 70.0
    T = _cam("isaac_side")
    shift = _T(np.eye(3), (0.0, -0.25, 0.0))
    left = _fill(_posed(TRUTH_TRIS, _synth_poses(_state(Q_A, t))[0]), T)
    moved = {k: shift @ v for k, v in _synth_poses(_state(Q_A, t))[0].items()}
    right = _fill(_posed(TRUTH_TRIS, moved), T)
    lm = _lm().LinkSelfMask(dilate_px=0)
    lm.add_arm("left", synth_geometry, _synth_poses, lambda: _state(Q_A, t))
    lm.add_arm("right", synth_geometry, _synth_poses, lambda: _state(Q_A, t - 5.0), base_T=shift)
    one = lm.mask_for(_blank(t), T)
    assert lm.last["reason"] == "link_mask"
    assert lm.last["arms"]["right"]["reason"] == "stale_joint_state"
    assert _metrics(one, left)["coverage"] >= 0.995 and _metrics(one, left)["bloat"] <= 0.06
    both = _lm().LinkSelfMask(dilate_px=0)
    both.add_arm("left", synth_geometry, _synth_poses, lambda: _state(Q_A, t))
    both.add_arm("right", synth_geometry, _synth_poses, lambda: _state(Q_A, t), base_T=shift)
    two = both.mask_for(_blank(t), T)
    assert _metrics(two, left | right)["coverage"] >= 0.995
    assert (right & ~left).sum() > 1000 and _metrics(two, right & ~left)["coverage"] >= 0.995


# ── configuration and wiring ────────────────────────────────────────────────


def test_the_shipped_config_leaves_the_link_mask_off_and_builds_nothing():
    from cascade.config import load_demo_config
    lm = _lm()
    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    block = cfg.get("workspace_filter").get("link_self_mask")
    assert block.get("enabled") is False
    assert block.get("dilate_px") == 2
    assert block.get("max_skew_s") == pytest.approx(0.15) and block.get("cell_m") == pytest.approx(0.05)
    assert lm.link_self_mask_settings(cfg) is None
    arm, kin = Mock(), Mock()
    assert lm.build_link_self_mask(cfg, [("arm0", cfg.arm, kin, arm)]) is None
    assert arm.method_calls == [] and kin.method_calls == []   # no read, no model load


@pytest.mark.parametrize("path", ["watcher", "get_observation"])
def test_flag_off_through_the_config_is_the_golden(path):
    from cascade.config import load_demo_config
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    link = _lm().build_link_self_mask(cfg, [("arm0", cfg.arm, Mock(), Mock())])
    assert link is None
    frame, dets, _ = _scene(t=1000.0, T_on_frame=path == "watcher")
    _assert_golden(_fuse(path, frame, dets, link), path)


def test_settings_when_enabled_and_when_the_gate_is_off():
    s = _lm().link_self_mask_settings
    assert s({"workspace_filter": {"link_self_mask": {"enabled": True}}}) == {
        "dilate_px": 2, "max_skew_s": 0.15, "cell_m": 0.05, "capsule_radius_m": None}
    assert s({"workspace_filter": {"link_self_mask": {
        "enabled": True, "dilate_px": 0, "max_skew_s": 0.3, "cell_m": 0.02, "capsule_radius_m": 0.04}}}) == {
        "dilate_px": 0, "max_skew_s": 0.3, "cell_m": 0.02, "capsule_radius_m": 0.04}
    # the mask only feeds the B32a gate: with that gate off there is nothing to build
    assert s({"workspace_filter": {"self_mask": False, "link_self_mask": {"enabled": True}}}) is None
    assert s({"workspace_filter": {}}) is None and s({}) is None


@pytest.mark.parametrize("block, field", [
    ({"enabled": "yes"}, "enabled"),
    ({"enabled": True, "dilate_px": -1}, "dilate_px"),
    ({"enabled": True, "dilate_px": 1.5}, "dilate_px"),
    ({"enabled": True, "dilate_px": True}, "dilate_px"),
    ({"enabled": True, "max_skew_s": 0}, "max_skew_s"),
    ({"enabled": True, "max_skew_s": float("nan")}, "max_skew_s"),
    ({"enabled": True, "cell_m": 0}, "cell_m"),
    ({"enabled": True, "capsule_radius_m": -0.1}, "capsule_radius_m"),
    ("on", "link_self_mask"),
])
def test_bad_settings_are_refused(block, field):
    with pytest.raises(ValueError, match=field):
        _lm().link_self_mask_settings({"workspace_filter": {"link_self_mask": block}})


def test_without_pinocchio_the_link_mask_is_skipped_and_says_why(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pinocchio", None)     # `import pinocchio` -> ImportError
    arm, kin = Mock(), Mock()
    cfg = {"workspace_filter": {"link_self_mask": {"enabled": True}}}
    assert _lm().build_link_self_mask(cfg, [("arm0", {"model": str(URDF)}, kin, arm)]) is None
    assert "pinocchio" in capsys.readouterr().err
    assert arm.method_calls == [] and kin.method_calls == []


@needs_pin
@pytest.mark.parametrize("enabled", [True, False])
def test_build_runtime_wires_one_link_mask_into_both_fusion_paths(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45101")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "45102")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45103")
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert 45100 <= int(cfg.grasp.graspgenx.port) <= 45199
    assert 45100 <= int(cfg.occupancy.port) <= 45199
    cfg._data["workspace_filter"]["link_self_mask"]["enabled"] = enabled
    rt, arm = build_runtime(cfg, tmp_path / "run")
    try:
        link = rt._link_self_mask
        if not enabled:
            assert link is None and rt.watcher._link_mask is None
            return
        assert isinstance(link, _lm().LinkSelfMask) and rt.watcher._link_mask is link
        assert [a.name for a in link.arms] == list(rt.arm_rig.names)   # one per rig arm, same names
        frame = rt.observe()
        mask = link.mask_for(frame, rt.extrinsics.cam_to_base())
        assert mask is not None and mask.shape == frame.rgb.shape[:2]
        assert link.last["reason"] == "link_mask"
    finally:
        shutdown_runtime(rt, arm)


# ── the real reBot RS model (Pinocchio FK; meshes need git-lfs) ─────────────


def _rs_meshes_available() -> bool:
    probe = URDF.parents[1] / "meshes" / "link2.STL"
    try:
        return probe.exists() and not probe.read_bytes()[:64].startswith(b"version https://git-lfs")
    except OSError:
        return False


needs_rs_meshes = pytest.mark.skipif(not _rs_meshes_available(), reason="RS meshes not fetched (git-lfs)")

RS_POSES = {"home": [0.0, 1.2, 1.2, 0.0, 0.0, 0.0], "handover": [0.5, 1.2, 1.2, 0.0, 0.75, 0.0],
            "reach": [0.3, 0.6, 1.6, 0.2, -0.4, 0.5], "low": [-0.4, 1.6, 0.9, -0.3, 0.6, -1.0]}


def _rs_truth_triangles() -> dict:
    """The RS collision meshes, read by the test itself (binary STL)."""
    import xml.etree.ElementTree as ET
    from cascade.types import pose_to_transform
    root = ET.parse(URDF).getroot()
    pkg = URDF.parents[1]
    out = {}
    for link in root.iter("link"):
        tris = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            fn = col.find("geometry/mesh").get("filename").split("/", 3)[-1]
            data = (pkg / fn).read_bytes()
            n = struct.unpack("<I", data[80:84])[0]
            rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            T = pose_to_transform(xyz + rpy)
            tris.append(rec["v"].reshape(-1, 3, 3).astype(float) @ T[:3, :3].T + T[:3, 3])
        if tris:
            out[link.get("name")] = np.concatenate(tris)
    return out


def _pin_link_poses(kin, q, finger_m: float) -> dict:
    """Every RS link's placement by Pinocchio directly (fingers at `finger_m`)."""
    import pinocchio as pin
    data = kin.model.createData()
    qf = np.zeros(kin.model.nq)
    qf[:6] = q
    qf[6:] = finger_m
    pin.forwardKinematics(kin.model, data, qf)
    pin.updateFramePlacements(kin.model, data)
    return {kin.model.frames[i].name: np.array(data.oMf[i].homogeneous)
            for i in range(len(kin.model.frames)) if kin.model.frames[i].type == pin.FrameType.BODY}


@needs_pin
@needs_rs_meshes
def test_rs_arm_link_mask_against_per_triangle_truth():
    from cascade.control.kinematics import Kinematics
    lm = _lm()
    kin = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    geometry = lm.LinkGeometry.from_urdf(URDF)
    assert set(geometry.sources.values()) == {"mesh"} and len(geometry.links) == 10
    poses = lm.KinematicsLinkPoses(kin, geometry.links, gripper_open=1.0, gripper_closed=0.0)
    tris = _rs_truth_triangles()
    w1, h1 = 1280, 720
    fx = 18.0 / 20.955 * w1
    K1 = np.array([[fx, 0.0, w1 / 2], [0.0, fx, h1 / 2], [0.0, 0.0, 1.0]])
    rows = []
    for name, q in RS_POSES.items():
        state = _state(q, 100.0, gripper_pos=0.5)       # fingers at half travel
        truth_poses = _pin_link_poses(kin, q, 0.025)
        for camera in ("isaac", "isaac_side"):
            T = _cam(camera)
            truth = _fill(_posed(tris, truth_poses), T, (h1, w1), K1)
            frame = Frame(rgb=np.zeros((h1, w1, 3), np.uint8), depth_m=None, K=K1.copy(), t=100.0)
            link = _link_mask(geometry, lambda s=state: s, poses=poses)
            m = _metrics(link.mask_for(frame, T), truth)
            rows.append((name, camera, round(m["coverage"], 5), round(m["bloat"], 3), round(m["iou"], 3),
                         round(link.last["ms"], 2)))
            assert m["coverage"] == 1.0
            assert m["bloat"] <= 0.20 and m["iou"] >= 0.82
    print("\nRS link mask vs per-triangle truth (pose, camera, coverage, bloat, IoU, ms):")
    for row in rows:
        print("  ", row)


@needs_pin
def test_link_poses_speak_the_local_joint_convention():
    from cascade.control.kinematics import Kinematics
    lm = _lm()
    local = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    asset = Kinematics(str(URDF), "gripper_end", 6, None)
    q = np.array([0.3, 0.9, 1.1, -0.2, 0.4, 0.6])
    links = ["link2", "link3", "link5", "gripper_end"]
    got = lm.KinematicsLinkPoses(local, links)(_state(q, 0.0, gripper_valid=False))
    ref = lm.KinematicsLinkPoses(asset, links)(_state(-q, 0.0, gripper_valid=False))
    for a, b in zip(got, ref, strict=True):
        for n in links:
            assert np.allclose(a[n], b[n], atol=1e-9)
    assert np.allclose(got[0]["gripper_end"], local.fk(q), atol=1e-9)
    mirrored = lm.KinematicsLinkPoses(local, links)(_state(-q, 0.0, gripper_valid=False))
    assert not np.allclose(mirrored[0]["link3"], got[0]["link3"], atol=1e-3)


@needs_pin
def test_finger_links_follow_the_measured_opening_and_are_swept_when_unknown():
    from cascade.control.kinematics import Kinematics
    lm = _lm()
    kin = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    links = ["gripper_left", "gripper_right", "link6"]
    q = [0.0, 1.2, 1.2, 0.0, 0.0, 0.0]

    def gap(p):
        return float(np.linalg.norm(p["gripper_left"][:3, 3] - p["gripper_right"][:3, 3]))

    isaac = lm.KinematicsLinkPoses(kin, links, gripper_open=1.0, gripper_closed=0.0)  # the bridge's fraction
    for frac, want in ((0.0, 0.0), (0.5, 0.05), (1.0, 0.10), (1.4, 0.10)):
        out = isaac(_state(q, 0.0, gripper_pos=frac))
        assert len(out) == 1 and gap(out[0]) == pytest.approx(want, abs=1e-3)
    real = lm.KinematicsLinkPoses(kin, links, gripper_open=6.2, gripper_closed=0.0)   # the RS motor, rad
    assert gap(real(_state(q, 0.0, gripper_pos=3.1))[0]) == pytest.approx(0.05, abs=1e-3)
    # unknown opening (failed gripper read, or no open/closed mapping): both ends, swept
    for poses, state in ((isaac, _state(q, 0.0, gripper_pos=0.5, gripper_valid=False)),
                         (lm.KinematicsLinkPoses(kin, links), _state(q, 0.0, gripper_pos=0.5))):
        out = poses(state)
        assert sorted(round(gap(p), 3) for p in out) == [0.0, 0.1]
        assert all(np.allclose(p["link6"], out[0]["link6"]) for p in out)



# ── scripts/compare_link_self_mask.py: the parent's live comparison, offline ─


def _compare_script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import compare_link_self_mask
    finally:
        sys.path.pop(0)
    return compare_link_self_mask


@needs_pin
@needs_rs_meshes
def test_compare_script_scores_recorded_frames_against_the_render_mask(tmp_path, monkeypatch):
    """The live comparison's offline half on a synthetic recording: the
    'render mask' is the per-triangle truth, so the script must report the
    same IoU/coverage the RS test above pins, agree with fusion's gate on a
    robot and a prop detection, and refuse to score a frame whose capture-time
    joints are missing."""
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45111")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "45112")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45113")
    from cascade.control.kinematics import Kinematics
    from cascade.config import load_demo_config
    cmp = _compare_script()
    kin = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    q = RS_POSES["reach"]
    T = _cam("isaac_side")
    truth = _fill(_posed(_rs_truth_triangles(), _pin_link_poses(kin, q, 0.025)), T)
    prop = _fill(_box_tris((0.05, 0.05, 0.05), PROP_CENTER), T)
    assert not (prop & truth).any() and truth.sum() > 2000
    frame = Frame(rgb=np.zeros((H, W, 3), np.uint8), depth_m=None, K=K.copy(), t=321.0)
    robot_det = truth.copy()
    robot_det[: int(np.nonzero(truth.any(axis=1))[0].mean())] = False    # the lower half of the arm
    dets = [SimpleNamespace(label="biplane", mask=robot_det), SimpleNamespace(label="cube", mask=prop)]
    state = _state(q, 321.0, gripper_pos=0.5)
    cmp.save_record(tmp_path / "a.npz", frame=frame, camera="side", T=T, render_self=truth,
                    capture_state=state, runtime_mask=truth, runtime_info={"reason": "link_mask"}, dets=dets)
    cmp.save_record(tmp_path / "b.npz", frame=frame, camera="side", T=T, render_self=truth)
    cfg = load_demo_config(cameras=["isaac", "isaac_side"], arm="isaac", llm="mock")
    cfg._data["workspace_filter"]["link_self_mask"]["enabled"] = True
    report = cmp.score_records(sorted(tmp_path.glob("*.npz")), cfg)
    side = report["summary"]["side"]
    assert side["frames"] == 1 and report["unscored"] == 1
    assert side["coverage_min"] == 1.0 and side["iou_min"] >= 0.82 and side["bloat_max"] <= 0.20
    a, b = report["rows"]
    assert a["runtime"]["iou"] == 1.0 and a["runtime_reason"] == "link_mask"
    assert [(d["label"], d["drop_render"], d["drop_link"]) for d in a["detections"]] == [
        ("biplane", True, True), ("cube", False, False)]
    assert side["gate_agree"] == 2 and side["dropped_link"] == 1
    assert b["capture"] is None and "no capture-time joint snapshot" in b["why"]
    # the CLI: a different dilation is a re-score of the same recording
    out = tmp_path / "score.json"
    assert cmp.main(["score", str(tmp_path), "--json", str(out), "--dilate-px", "0"]) == 0
    tight = json.loads(out.read_text())
    assert tight["settings"]["dilate_px"] == 0
    assert tight["summary"]["side"]["bloat_max"] < side["bloat_max"]



def test_compare_script_record_plumbing_with_a_fake_runtime(tmp_path, monkeypatch, synth_geometry):
    """`record` without Isaac: a stand-in runtime with the real link mask,
    gate and camera objects. Every pose goes through `rt.arm.move_joints`
    (the SafeArm); a refused pose is logged and skipped, never forced; each
    saved frame carries the render mask fusion uses, the capture-time joints
    and the runtime link mask built for the same frame without its render mask."""
    monkeypatch.setenv("CASCADE_GRASPGENX_PORT", "45121")
    monkeypatch.setenv("CASCADE_OCCUPANCY_PORT", "45122")
    monkeypatch.setenv("CASCADE_BRIDGE_PORT", "45123")
    import cascade.apps.demo as demo
    import cascade.control.isaac_arm as isaac_arm
    from cascade.types import SafetyViolation
    cmp = _compare_script()
    frames = []

    def get_frame():
        frame, _, _ = _scene(render_mask=True)
        frames.append(frame)
        return frame

    _, dets, robot = _scene(render_mask=True)
    link = _link_mask(synth_geometry, lambda: _state(Q_A, time.monotonic()))
    stream = SimpleNamespace(name="side", get_frame=get_frame)
    cam = SimpleNamespace(stream=stream, depth=SimpleNamespace(ensure_depth=lambda f: f), fuse=True,
                          extrinsics=SimpleNamespace(cam_to_base=lambda: _cam("isaac_side")))
    moves = []

    def move_joints(q, duration_s):
        moves.append([float(v) for v in q])
        if q[0] > 1.0:
            raise SafetyViolation("joint 1 outside its limits")
        return True

    rt = SimpleNamespace(_link_self_mask=link, watcher=SimpleNamespace(_cams=[cam]), _workspace=WorkspaceFilter(),
                         detector=SimpleNamespace(detect=Mock(return_value=dets)), _default_classes=None,
                         arm=SimpleNamespace(move_joints=move_joints))
    teardown = Mock()
    monkeypatch.setattr(demo, "build_runtime", lambda cfg, run_dir, view=False, serve=False: (rt, "raw"))
    monkeypatch.setattr(demo, "shutdown_runtime", teardown)

    class Observer:
        def __init__(self, acfg):
            assert int(acfg.get("bridge_port")) == 45123     # the private port reached the profile
            self.acfg = acfg

        def state_from_frame(self, frame):
            return _state(Q_A, frame.t, gripper_pos=0.4)

    monkeypatch.setattr(isaac_arm, "IsaacArm", Observer)
    out = tmp_path / "rec"
    assert cmp.main(["record", "--out", str(out), "--frames", "2", "--interval-s", "0", "--settle-s", "0",
                     "--pose", "home", "--pose", "1.4,0,0,0,0,0"]) == 0
    assert moves == [[0.0, 1.2, 1.2, 0.0, 0.0, 0.0], [1.4, 0.0, 0.0, 0.0, 0.0, 0.0]]
    teardown.assert_called_once_with(rt, "raw")
    log = json.loads((out / "record.json").read_text())
    assert log["frames"] == 2 and log["skipped"] == []
    assert log["poses"][0] == {"pose": "home", "moved": True}
    assert log["poses"][1]["moved"] is False and "outside its limits" in log["poses"][1]["error"]
    ports = dict(log["ports"])
    assert not set(ports.values()) & cmp.SHARED_PORTS
    assert {ports[k] for k in ("arm.bridge_port", "grasp.graspgenx.port", "occupancy.port")} == {45121, 45122, 45123}
    recs = [cmp.load_record(p) for p in sorted(out.glob("*.npz"))]
    assert [r["camera"] for r in recs] == ["side", "side"]
    for rec, frame in zip(recs, frames, strict=True):
        assert np.array_equal(rec["render_self"], robot) and rec["t"] == frame.t
        assert rec["capture_q"].tolist() == list(Q_A) and float(rec["capture_gripper_pos"]) == 0.4
        assert rec["runtime_info"]["reason"] == "link_mask"
        assert _metrics(rec["runtime_mask"], robot)["coverage"] == 1.0
        assert rec["det_masks"].shape == (2, H, W) and rec["det_labels"].tolist() == ["biplane", "cube"]
        assert frame.robot_mask is not None    # the recorded frame itself is untouched



def test_a_pose_source_that_raises_builds_no_mask_and_fusion_is_the_golden(synth_geometry, capsys):
    """Any failure inside the mask (here: FK refusing the joint vector) is NO
    mask, counted and logged once: fusion falls back to the base cylinder."""
    frame, dets, _ = _scene(t=1000.0)

    def poses(state):
        raise ValueError("joint state has 1 finite values, the arm commands 2")

    link = _link_mask(synth_geometry, lambda: _state(Q_A, 1000.0), poses=poses)
    assert link.attach(frame, _cam("isaac_side")) is frame and frame.robot_mask is None
    assert link.last["reason"] == "error" and "arm commands 2" in link.last["error"]
    _assert_golden(_fuse_watcher(frame, dets, link), "watcher")
    assert link.counts["error"] == 2
    assert capsys.readouterr().err.count("[link-self-mask] no mask") == 1   # state changes, not spam


def test_huge_polygons_are_clipped_before_the_fixed_point_fill():
    """A piece crossing the near plane far off-axis projects to coordinates
    past int32 at 1/16 px; it is clipped to the canvas first."""
    lm = _lm()
    T = np.eye(4)
    wall = np.array([[x, y, z] for x in (-5000.0, 5000.0) for y in (-5000.0, 5000.0) for z in (-0.5, 0.5)])
    assert lm.rasterize_pieces([wall], K, T, (H, W)).all()
    side = np.array([[x, y, z] for x in (0.2, 5000.0) for y in (-5000.0, 5000.0) for z in (-0.5, 0.5)])
    m = lm.rasterize_pieces([side], K, T, (H, W))
    cols = np.nonzero(m.any(axis=0))[0]
    assert m.any() and cols.min() > K[0, 2] and cols.max() == W - 1


@needs_pin
def test_build_takes_the_capsule_radius_from_the_profile_and_never_wakes_a_standby_arm(tmp_path, capsys):
    """Unreadable meshes -> capsules as thick as the occupancy body mask
    (`body_mask_radius_m`) unless the block overrides it; a standby LazyArm
    (`connected` False) is never read: no state, no mask."""
    from cascade.control.kinematics import Kinematics
    lm = _lm()
    urdf = tmp_path / "rs_nomesh.urdf"
    urdf.write_text(URDF.read_text())          # package:// meshes do not resolve from here
    kin = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    standby = SimpleNamespace(connected=False, get_state=Mock())

    def built(block, profile):
        cfg = {"workspace_filter": {"link_self_mask": {"enabled": True, **block}}}
        return lm.build_link_self_mask(cfg, [("arm0", {"model": str(urdf), **profile}, kin, standby)])

    def thickness(link):
        g = link.arms[0].geometry
        assert set(g.sources) and all(s.startswith("capsule") for s in g.sources.values())
        pts = np.concatenate(g.pieces["base_link"])
        return float(-pts[:, 0].min())        # the capsule starts at the link origin

    assert thickness(built({}, {"body_mask_radius_m": 0.03})) == pytest.approx(0.03, rel=0.2)
    assert thickness(built({}, {})) == pytest.approx(0.06, rel=0.2)
    assert thickness(built({"capsule_radius_m": 0.045}, {"body_mask_radius_m": 0.03})) == pytest.approx(0.045, rel=0.2)
    link = built({}, {})
    assert "capsule fallback" in capsys.readouterr().err
    assert link.mask_for(_blank(5.0), _cam("isaac_side")) is None
    assert link.last["reason"] == "no_joint_state" and standby.get_state.call_count == 0



def test_a_sample_too_old_for_the_frame_is_replaced_by_an_on_demand_read(synth_geometry):
    """The watcher's last read can be older than max_skew_s by the time a
    frame is fused (a slow detector pass): the arm is read once more, and the
    mask is posed at that read, not given up."""
    clock = iter([100.0, 101.0])
    state_fn = Mock(side_effect=lambda: _state(Q_A, next(clock)))
    link = _link_mask(synth_geometry, state_fn)
    link.sample()                                  # 1 s before the frame
    assert link.mask_for(_blank(101.0), _cam("isaac_side")) is not None
    assert link.last["reason"] == "link_mask" and state_fn.call_count == 2
    assert link.last["arms"]["arm0"]["skew_s"] == pytest.approx(0.0)


def test_with_the_gate_off_neither_path_reads_the_arm(synth_geometry):
    """`workspace_filter.self_mask: false` is B32a's A/B baseline: nothing
    consumes robot pixels, so neither fusion path reads the arm or builds a
    mask, even when handed a link mask."""
    frame, dets, _ = _scene()
    state_fn = Mock(side_effect=lambda: _state(Q_A, time.monotonic()))
    link = _link_mask(synth_geometry, state_fn)
    watch, cam, beliefs = _watch(dets, link_mask=link)
    watch._workspace = WorkspaceFilter(self_mask=False)
    cam.stream.latest.return_value = frame
    watch._tick(cam)
    assert "biplane" in _by_label(beliefs)
    from cascade.skills.runtime import SkillRuntime
    obs_frame, odets, _ = _scene(T_on_frame=False)
    rt = _partial_runtime(obs_frame, odets, link)
    rt._workspace = WorkspaceFilter(self_mask=False)
    out = SkillRuntime._describe_observation(rt, obs_frame)
    assert "biplane" in {o["label"] for o in out["objects_visible"]}
    assert state_fn.call_count == 0 and dict(link.counts) == {}
