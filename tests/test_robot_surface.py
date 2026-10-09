"""Robot surface model: the arm's own meshes, posed by the SAME signed
kinematics the safety layer uses (calibration/robot_surface.py).

Markerless hand-eye calibration and the extrinsic drift monitor both compare
depth against this model, so two properties are load-bearing and pinned
here: the meshes are in metres (a millimetre STL would put the arm 1000x too
far away and every fit would fail quietly), and a link's points move exactly
with kin.fk(q) -- the model is built on kin's own (joint-sign-baked) model,
never on a second, separately loaded copy that could disagree about signs.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from conftest import JOINT_SIGNS, URDF, needs_pin

from cascade.calibration.robot_surface import (
    RobotSurface,
    exterior_mask,
    parse_collision_meshes,
    read_stl,
    sample_triangles,
)

# ── STL / sampling primitives (no pinocchio) ─────────────────────────────


def _tetra():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    return np.stack([v[list(f)] for f in faces])


def _write_binary(path, tris):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0, 0, 0))
            f.write(struct.pack("<9f", *t.ravel()))
            f.write(b"\0\0")


def _write_ascii(path, tris):
    lines = ["solid t"]
    for t in tris:
        lines += ["facet normal 0 0 0", " outer loop"]
        lines += [f"  vertex {x:.6f} {y:.6f} {z:.6f}" for x, y, z in t]
        lines += [" endloop", "endfacet"]
    lines.append("endsolid t")
    path.write_text("\n".join(lines))


def test_stl_reader_reads_binary_and_ascii_identically(tmp_path):
    tris = _tetra()
    _write_binary(tmp_path / "b.stl", tris)
    _write_ascii(tmp_path / "a.stl", tris)
    for p in (tmp_path / "b.stl", tmp_path / "a.stl"):
        got = read_stl(p)
        assert got.shape == (4, 3, 3)
        assert np.allclose(got, tris, atol=1e-6)


def test_a_truncated_binary_stl_is_refused(tmp_path):
    _write_binary(tmp_path / "b.stl", _tetra())
    data = (tmp_path / "b.stl").read_bytes()
    (tmp_path / "cut.stl").write_bytes(data[:-30])
    with pytest.raises(ValueError):
        read_stl(tmp_path / "cut.stl")


def test_sampling_is_area_weighted_on_the_surface_with_outward_normals():
    # Unit cube, outward-wound: a 2x1x1 box has faces of area 2 and 1.
    box = _box_tris([0, 0, 0], [2, 1, 1])
    pts, nrm = sample_triangles(box, 20000, np.random.default_rng(0))
    # Every point lies on the box surface ...
    on = (np.isclose(pts[:, 0], 0) | np.isclose(pts[:, 0], 2) | np.isclose(pts[:, 1], 0)
          | np.isclose(pts[:, 1], 1) | np.isclose(pts[:, 2], 0) | np.isclose(pts[:, 2], 1))
    assert on.all()
    # ... normals are unit and point away from the centre ...
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    assert np.all(np.einsum("ij,ij->i", nrm, pts - [1.0, 0.5, 0.5]) > 0)
    # ... and the +-x faces (area 1 each of total 10) get ~20 % of the points.
    frac_x = np.mean(np.abs(nrm[:, 0]) > 0.9)
    assert frac_x == pytest.approx(0.2, abs=0.02)
    # Deterministic for a seed.
    again, _ = sample_triangles(box, 20000, np.random.default_rng(0))
    assert np.array_equal(pts, again)


def _box_tris(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[lo[0] if i & 1 == 0 else hi[0], lo[1] if i & 2 == 0 else hi[1],
                   lo[2] if i & 4 == 0 else hi[2]] for i in range(8)])
    quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
    tris = []
    for a, b, cc, d in quads:
        tris += [c[[a, b, cc]], c[[a, cc, d]]]
    tris = np.stack(tris)
    # orient outward
    centre = (lo + hi) / 2
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", n, tris.mean(axis=1) - centre) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def test_exterior_filter_drops_geometry_enclosed_by_the_shell():
    """CAD meshes carry internal parts (bores, screws, ribs) that no camera
    can ever see; left in the model they pull the fit behind the surface."""
    outer = _box_tris([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05])
    inner = _box_tris([-0.02, -0.02, -0.02], [0.02, 0.02, 0.02])
    rng = np.random.default_rng(1)
    po, no = sample_triangles(outer, 4000, rng)
    pi, ni = sample_triangles(inner, 1000, rng)
    pts, nrm = np.vstack([po, pi]), np.vstack([no, ni])
    keep = exterior_mask(pts, nrm)
    assert keep[:4000].mean() > 0.97
    assert keep[4000:].mean() < 0.02


# ── the reBot arm model ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def kin():
    pytest.importorskip("pinocchio")
    from cascade.control.kinematics import Kinematics

    return Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)


@pytest.fixture(scope="module")
def surface(kin):
    return RobotSurface.from_kinematics(kin, URDF)


def test_collision_meshes_resolve_package_uris_to_files():
    meshes = parse_collision_meshes(URDF)
    assert set(meshes) >= {"base_link", "link1", "link2", "link3", "link4", "link5", "link6",
                           "gripper_end", "gripper_left", "gripper_right"}
    for parts in meshes.values():
        for T, path, scale in parts:
            assert path.is_file() and T.shape == (4, 4) and np.all(np.asarray(scale) > 0)


@needs_pin
def test_finger_links_are_excluded_because_their_joints_are_not_measured(surface):
    """kin pads the passive finger joints with zeros: FK cannot say where the
    jaws are, so their geometry must not be in the model."""
    assert "gripper_left" not in surface.links and "gripper_right" not in surface.links
    assert {"base_link", "link2", "link3", "gripper_end"} <= set(surface.links)


@needs_pin
def test_mesh_units_are_metres_consistent_with_link_lengths(kin, surface):
    q = np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    joints = np.asarray(kin.link_positions(q))
    cloud = surface.points_at(q)
    # The whole arm fits in a ~1 m box and spans its own chain.
    span = cloud.points.max(axis=0) - cloud.points.min(axis=0)
    assert np.all(span < 1.0), span
    chain = np.linalg.norm(np.ptp(np.vstack([joints, kin.fk(q)[:3, 3]]), axis=0))
    assert 0.6 * chain < np.linalg.norm(span) < 2.0 * chain + 0.2
    # The upper arm (link2) is as long as the joint2 -> joint3 distance.
    i2 = surface.links.index("link2")
    l2 = cloud.points[cloud.link_ids == i2]
    upper = np.linalg.norm(joints[2] - joints[1])
    assert upper > 0.2                       # the RS upper arm is ~0.26 m
    ext = np.linalg.norm(np.ptp(l2, axis=0))
    assert 0.9 * upper < ext < 1.6 * upper + 0.05


@needs_pin
def test_gripper_points_move_rigidly_with_fk(kin, surface):
    rng = np.random.default_rng(3)
    lo, hi = kin.joint_limits
    ig = surface.links.index("gripper_end")
    local = None
    for _ in range(5):
        q = rng.uniform(lo + 0.1, hi - 0.1)
        cloud = surface.points_at(q)
        sel = cloud.link_ids == ig
        T = kin.fk(q)
        here = (cloud.points[sel] - T[:3, 3]) @ T[:3, :3]
        nloc = cloud.normals[sel] @ T[:3, :3]
        if local is None:
            local, local_n = here, nloc
        assert np.allclose(here, local, atol=1e-9)
        assert np.allclose(nloc, local_n, atol=1e-9)


@needs_pin
def test_base_link_never_moves(kin, surface):
    ib = surface.links.index("base_link")
    a = surface.points_at(np.zeros(6))
    b = surface.points_at(np.array([1.0, 0.5, 0.7, -0.3, 0.4, 1.0]))
    assert np.allclose(a.points[a.link_ids == ib], b.points[b.link_ids == ib])


@needs_pin
def test_model_is_kins_own_signed_model():
    """Same physical pose: q on the signed (local) model == -q on the raw
    mirrored asset. The surface follows whichever model kin carries."""
    from cascade.control.kinematics import Kinematics

    signed = Kinematics(str(URDF), "gripper_end", 6, JOINT_SIGNS)
    raw = Kinematics(str(URDF), "gripper_end", 6, None)
    q = np.array([0.4, 1.0, 0.8, -0.5, 0.6, 0.3])
    a = RobotSurface.from_kinematics(signed, URDF).points_at(q).points
    b = RobotSurface.from_kinematics(raw, URDF).points_at(-q).points
    assert np.allclose(a, b, atol=1e-9)


@needs_pin
def test_sampling_is_cached_and_deterministic(kin):
    a = RobotSurface.from_kinematics(kin, URDF)
    b = RobotSurface.from_kinematics(kin, URDF)
    assert a.local_points is b.local_points          # cached, not re-read
    c = RobotSurface.from_kinematics(kin, URDF, points_per_m2=a.points_per_m2 / 2)
    assert len(c.points_at(np.zeros(6)).points) < len(a.points_at(np.zeros(6)).points)
