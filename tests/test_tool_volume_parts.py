"""Tool volume parts from measurements; the shipped values stay today's slab.

Spec docs/superpowers/specs/2026-10-16-nvblox-object-aware-design.md §3.4, §8.
The wrist-camera box is rebuilt from the owner's calipers (every face moves
OUTWARD to the next 5 mm, a missing number keeps today's face), next to two
URDF parts: the `gripper_end` housing and both fingers over the full stroke.
Until the calipers arrive nothing ships: the reBot samples today's
14 x 15 x 8 cm slab plus the jaw gap, byte for byte, and the parts ship
together with the first measured face (one change, one refusal count).
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin

from cascade.safety.tool_volume import (
    TODAY_SLAB,
    build_tool_volume,
    camera_faces_from_calipers,
    uncovered_camera_faces,
)
from cascade.skills import relative_motion as rm
from cascade.skills.spatial_moves import (
    DEFAULT_WRIST_CAMERA,
    CameraDims,
    wrist_camera_dims,
)
from cascade.types import SkillError

ROOT = Path(__file__).resolve().parents[1]
CALIPERS = ROOT / "docs" / "evidence" / "wrist-camera-calipers" / "measurement.json"
FACES = ("front_behind_tip", "rear_behind_tip", "minus_y", "plus_y", "bottom_above_axis", "top_above_axis")

#: URDF part boxes, (approach, jaw, camera-up) in the TCP frame, rounded UP to
#: 5 mm (spec §3.4.1 (b), (c); hull AABBs -0.1572..-0.0732, +-0.0920,
#: +-0.0410 and -0.0886..0, +-0.0848, +-0.0192)
HOUSING = ((-.160, -.095, -.045), (-.070, .095, .045))
FINGERS = ((-.090, -.085, -.020), (0., .085, .020))

#: sha256 of today's reBot volume (local rounded to 1 nm, part, names,
#: spacing, band), computed with the PRE-CHANGE builder at abcc372
SHIPPED_REBOT_DIGEST = "ec5ffe2cc5e23b3066825a2c1ac1cc357355153d381e2f8e431c730e2693f7b1"


def _cfg(arm="rebot_rs"):
    from cascade.config import load_demo_config

    return load_demo_config(arm=arm, camera="mock", llm="mock")


def _digest(vol):
    h = hashlib.sha256()
    h.update((np.round(vol.local, 9) + 0.0).astype("<f8").tobytes())
    h.update(vol.part.astype("<i8").tobytes())
    h.update(repr((vol.names, vol.spacing_m, vol.table_band_m)).encode())
    return h.hexdigest()


def _part(vol, name):
    return vol.local[vol.part == vol.names.index(name)]


def _box_block(box):
    lo, hi = box
    return {"min": list(lo), "max": list(hi)}


def _with_parts(cfg, **parts):
    cfg.safety._data["tool_volume"].update(parts)
    return cfg


def shipped_rebot_dims():
    dims, _ = wrist_camera_dims(_cfg().relative_motion)
    return dims


def committed_calipers():
    return json.loads(CALIPERS.read_text())


def required_from_json():
    return camera_faces_from_calipers(committed_calipers()["faces_mm"])


# ── calipers -> camera faces: outward to 5 mm, a missing number keeps today ─


def test_faces_round_outward_to_5mm():
    got = camera_faces_from_calipers({"front_behind_tip": 81, "rear_behind_tip": 136, "minus_y": 74,
                                      "plus_y": 71, "bottom_above_axis": 37, "top_above_axis": 77})
    assert got == CameraDims(.08, .14, .075, .08, .035)


def test_faces_on_the_5mm_grid_stay_and_a_fraction_past_it_moves_out():
    exact = {"front_behind_tip": 80, "rear_behind_tip": 140, "minus_y": 75, "plus_y": 75,
             "bottom_above_axis": 35, "top_above_axis": 80}
    assert camera_faces_from_calipers(exact) == CameraDims(.08, .14, .075, .08, .035)
    past = {"front_behind_tip": 79.9, "rear_behind_tip": 140.1, "minus_y": 75.1, "plus_y": 60,
            "bottom_above_axis": 34.9, "top_above_axis": 80.1}
    assert camera_faces_from_calipers(past) == CameraDims(.085, .145, .08, .075, .03)


def test_a_missing_number_keeps_todays_face():
    assert camera_faces_from_calipers({"rear_behind_tip": 146}) == TODAY_SLAB._replace(behind_m=.15)
    assert camera_faces_from_calipers({}) == TODAY_SLAB
    assert camera_faces_from_calipers(dict.fromkeys(FACES)) == TODAY_SLAB
    assert camera_faces_from_calipers({"front_behind_tip": 92}) == TODAY_SLAB._replace(front_m=.09)
    assert camera_faces_from_calipers({"bottom_above_axis": 21}) == TODAY_SLAB._replace(bottom_m=.02)
    assert camera_faces_from_calipers({"top_above_axis": 91}) == TODAY_SLAB._replace(protrusion_m=.095)


def test_one_measured_side_never_narrows_the_other():
    """The box is symmetric across the jaws: one side measured narrower keeps
    today's other side; a wider one widens both."""
    assert camera_faces_from_calipers({"plus_y": 61}) == TODAY_SLAB
    assert camera_faces_from_calipers({"minus_y": 61, "plus_y": None}) == TODAY_SLAB
    assert camera_faces_from_calipers({"plus_y": 81}) == TODAY_SLAB._replace(half_width_m=.085)
    assert camera_faces_from_calipers({"minus_y": 61, "plus_y": 66}) == TODAY_SLAB._replace(half_width_m=.07)


def test_front_and_bottom_never_go_below_zero():
    got = camera_faces_from_calipers({"front_behind_tip": 3, "bottom_above_axis": 4.9})
    assert got == TODAY_SLAB and got.front_m == 0.0 and got.bottom_m == 0.0
    assert camera_faces_from_calipers({"front_behind_tip": 0, "bottom_above_axis": 0}) == TODAY_SLAB


@pytest.mark.parametrize("face", ["front_behind_tip", "bottom_above_axis"])
def test_a_camera_reaching_past_the_tip_or_below_the_axis_is_refused(face):
    """The box starts at the fingertip plane and the finger axis at the
    earliest: a face in front of / below them cannot be covered, so it must
    not silently clamp to 0."""
    with pytest.raises(ValueError, match="calipers: .* cannot cover it"):
        camera_faces_from_calipers({face: -3})


def test_the_rounding_starts_from_the_given_today():
    today = CameraDims(.1, .2, .1, .05, .02)
    assert camera_faces_from_calipers({}, today=today) == today
    assert camera_faces_from_calipers({"rear_behind_tip": 151}, today=today) == today._replace(behind_m=.155)


@pytest.mark.parametrize("bad", [
    {"rear_behind_tip": float("nan")}, {"top_above_axis": float("inf")}, {"plus_y": "71"},
    {"minus_y": True}, {"rear_behind_tip": -5}, {"top_above_axis": 0}, {"minus_y": -1},
    {"rear": 140},                                            # a typo must not keep today silently
    {"front_behind_tip": 150, "rear_behind_tip": 140},        # front behind the rear face
    {"bottom_above_axis": 90, "top_above_axis": 80},          # bottom above the top
    {"rear_behind_tip": 400},                                 # beyond the 0.30 m camera limit
])
def test_bad_caliper_numbers_raise(bad):
    with pytest.raises(ValueError, match="calipers"):
        camera_faces_from_calipers(bad)


# ── the committed record vs the shipped box ────────────────────────────────


def test_the_committed_caliper_record_is_well_formed():
    data = committed_calipers()
    assert data["schema"] == "cascade.wrist_camera_calipers/1"
    assert data["units"] == "mm"
    assert tuple(data["faces_mm"]) == FACES
    for v in data["faces_mm"].values():
        assert v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
    assert isinstance(data["photos"], list)
    assert "fingertip plane" in data["reference_frame"] and "gripper_end" in data["reference_frame"]


def test_shipped_config_covers_the_committed_calipers():      # all null today -> today's slab
    assert uncovered_camera_faces(shipped_rebot_dims(), required_from_json()) == []


def test_the_coverage_check_is_red_against_the_old_slab():
    assert uncovered_camera_faces(TODAY_SLAB, camera_faces_from_calipers({"rear_behind_tip": 146})) == ["rear"]


@pytest.mark.parametrize("measured, face", [
    ({"front_behind_tip": 74}, None),                 # front measured nearer: today's front 0 covers it
    ({"rear_behind_tip": 141}, "rear"), ({"plus_y": 76}, "sides"), ({"minus_y": 76}, "sides"),
    ({"top_above_axis": 81}, "top"), ({"rear_behind_tip": 139, "top_above_axis": 79}, None),
])
def test_each_face_the_shipped_slab_misses_is_named(measured, face):
    assert uncovered_camera_faces(TODAY_SLAB, camera_faces_from_calipers(measured)) == ([face] if face else [])


def test_a_cut_front_or_bottom_must_not_reach_past_the_measured_face():
    shipped = TODAY_SLAB._replace(front_m=.09, bottom_m=.04)
    assert uncovered_camera_faces(shipped, TODAY_SLAB._replace(front_m=.085, bottom_m=.04)) == ["front"]
    assert uncovered_camera_faces(shipped, TODAY_SLAB._replace(front_m=.09, bottom_m=.035)) == ["bottom"]
    assert uncovered_camera_faces(shipped, shipped) == []
    assert uncovered_camera_faces(TODAY_SLAB, TODAY_SLAB._replace(behind_m=.15, protrusion_m=.09)) == ["rear", "top"]


def test_calipers_and_parts_ship_together():
    """Spec §3.4.3: the housing and finger parts ship with the first measured
    face, never before or after it (one change, one refusal count)."""
    measured = any(v is not None for v in committed_calipers()["faces_mm"].values())
    block = _cfg().safety.tool_volume
    has_parts = block.get("housing") is not None and block.get("fingers") is not None
    assert has_parts == measured


def test_today_slab_is_the_shipped_and_the_default_box():
    assert TODAY_SLAB == CameraDims(0.08, 0.14, 0.075, 0.0, 0.0)
    assert shipped_rebot_dims() == TODAY_SLAB
    assert wrist_camera_dims(_cfg("mock").relative_motion)[0] == TODAY_SLAB
    d = DEFAULT_WRIST_CAMERA
    assert CameraDims(d["protrusion_m"], d["behind_tip_m"], d["half_width_m"], d["front_m"], d["bottom_m"]) == TODAY_SLAB
    assert wrist_camera_dims({})[0] == TODAY_SLAB      # no key at all: the same slab


# ── the shipped reBot volume is unchanged ──────────────────────────────────


def test_shipped_rebot_volume_is_unchanged():               # same samples as before the change
    cfg = _cfg()
    vol = build_tool_volume(cfg, cfg.arm)
    assert vol.names == ("wrist camera", "jaws")
    assert _digest(vol) == SHIPPED_REBOT_DIGEST
    cam, jaws = _part(vol, "wrist camera"), _part(vol, "jaws")
    assert (len(cam), len(jaws)) == (360, 48)
    np.testing.assert_allclose(cam.min(axis=0), [-.14, -.075, 0.], atol=1e-12)
    np.testing.assert_allclose(cam.max(axis=0), [0., .075, .08], atol=1e-12)


def test_null_parts_are_off():
    cfg = _cfg()
    base = build_tool_volume(cfg, cfg.arm)
    vol = build_tool_volume(_with_parts(cfg, housing=None, fingers=None), cfg.arm)
    assert vol.names == base.names
    np.testing.assert_array_equal(vol.local, base.local)
    np.testing.assert_array_equal(vol.part, base.part)


# ── the parts, when configured ─────────────────────────────────────────────


def test_housing_and_fingers_are_sampled_when_configured():
    cfg = _cfg()
    shipped = build_tool_volume(cfg, cfg.arm)
    vol = build_tool_volume(_with_parts(cfg, housing=_box_block(HOUSING), fingers=_box_block(FINGERS)), cfg.arm)
    assert vol.names == ("wrist camera", "housing", "fingers", "jaws")
    spacing = float(cfg.safety.tool_volume.spacing_m)
    # reBot TCP frame (down_open): x approach, y jaws, z camera up
    for name, (lo, hi) in (("housing", HOUSING), ("fingers", FINGERS)):
        pts = _part(vol, name)
        np.testing.assert_allclose(pts.min(axis=0), lo, atol=1e-12)
        np.testing.assert_allclose(pts.max(axis=0), hi, atol=1e-12)
        for axis in range(3):
            steps = np.diff(np.unique(np.round(pts[:, axis], 9)))
            assert steps.size and steps.max() <= spacing + 1e-9
    # adding parts changes nothing else
    for name in ("wrist camera", "jaws"):
        np.testing.assert_array_equal(_part(vol, name), _part(shipped, name))


def test_parts_follow_the_tool_axis_order():
    """(approach, jaw, camera-up) map through the arm's tool_axis_order, like
    the jaws: open_down = approach z, jaws x, third y."""
    cfg = _cfg()
    cfg.safety._data["tool_volume"]["wrist_camera"] = False   # its up axis is the open_down approach
    _with_parts(cfg, housing=_box_block(HOUSING))
    vol = build_tool_volume(cfg, {"tool_axis_order": "open_down"})
    pts = _part(vol, "housing")
    a, j, u = rm._TOOL_COLUMNS["open_down"]
    np.testing.assert_allclose(pts.min(axis=0)[[a, j, u]], HOUSING[0], atol=1e-12)
    np.testing.assert_allclose(pts.max(axis=0)[[a, j, u]], HOUSING[1], atol=1e-12)


def test_a_block_with_only_parts_samples_them():
    cfg = _cfg()
    cfg.safety._data["tool_volume"] = {"wrist_camera": False, "jaws": False, "fingers": _box_block(FINGERS)}
    vol = build_tool_volume(cfg, cfg.arm)
    assert vol.names == ("fingers",)


@pytest.mark.parametrize("part", ["housing", "fingers"])
@pytest.mark.parametrize("bad", [
    {"min": [0.0, 0.0, 0.0]},                                    # no max
    {"min": [0.0, 0.0], "max": [0.1, 0.1]},                      # 2 numbers
    {"min": [0.0, 0.0, 0.0], "max": [0.1, 0.0, 0.1]},            # min == max on one axis
    {"min": [0.1, 0.0, 0.0], "max": [0.0, 0.1, 0.1]},            # min > max
    {"min": [0.0, 0.0, float("nan")], "max": [0.1, 0.1, 0.1]},
    {"min": [0.0, 0.0, 0.0], "max": [0.1, 0.1, float("inf")]},
    {"min": [-0.31, 0.0, 0.0], "max": [0.1, 0.1, 0.1]},          # |v| > 0.30 m
    {"min": [0.0, 0.0, 0.0], "max": [0.1, True, 0.1]},
    {"min": [0.0, False, 0.0], "max": [0.1, 0.1, 0.1]},           # a bool is no length, even in range
    {"min": [0.0, 0.0, 0.0], "max": [0.1, "0.1", 0.1]},
    {"min": [0.0, 0.0, 0.0], "max": [0.1, 0.1, 0.1], "pad": 0.01},   # unknown key
    [0.0, 0.1], "box", True,
])
def test_a_bad_part_block_is_refused_loudly(part, bad):
    cfg = _with_parts(_cfg(), **{part: bad})
    with pytest.raises(ValueError, match=f"tool_volume.{part}"):
        build_tool_volume(cfg, cfg.arm)


def test_a_part_at_the_limit_is_accepted():
    cfg = _with_parts(_cfg(), housing={"min": [-0.30, -0.30, -0.30], "max": [0.30, 0.30, 0.30]})
    assert "housing" in build_tool_volume(cfg, cfg.arm).names


# ── the cut camera box: front and bottom, guarded by the parts ─────────────


def _cut_camera(cfg, **faces):
    cfg.relative_motion._data["wrist_camera"].update(faces)
    return cfg


def test_front_without_fingers_is_refused():
    with pytest.raises(ValueError, match="open fingers unsampled"):
        cfg = _cut_camera(_cfg(), front_m=0.08)
        build_tool_volume(cfg, cfg.arm)
    with pytest.raises(ValueError, match="open fingers unsampled"):
        cfg = _with_parts(_cut_camera(_cfg(), front_m=0.08), fingers=None, housing=_box_block(HOUSING))
        build_tool_volume(cfg, cfg.arm)


def test_bottom_without_housing_is_refused():
    """Raising the slab's bottom leaves the housing's top under the camera
    unsampled, the same hole as a front without fingers."""
    with pytest.raises(ValueError, match="housing unsampled"):
        cfg = _with_parts(_cut_camera(_cfg(), bottom_m=0.035), fingers=_box_block(FINGERS))
        build_tool_volume(cfg, cfg.arm)


def test_a_cut_camera_with_its_parts_spans_front_to_rear_and_bottom_to_top():
    cfg = _with_parts(_cut_camera(_cfg(), front_m=0.08, bottom_m=0.035),
                      housing=_box_block(HOUSING), fingers=_box_block(FINGERS))
    vol = build_tool_volume(cfg, cfg.arm)
    cam = _part(vol, "wrist camera")
    np.testing.assert_allclose(cam.min(axis=0), [-.14, -.075, .035], atol=1e-12)
    np.testing.assert_allclose(cam.max(axis=0), [-.08, .075, .08], atol=1e-12)
    spacing = float(cfg.safety.tool_volume.spacing_m)
    for axis in range(3):
        assert np.diff(np.unique(np.round(cam[:, axis], 9))).max() <= spacing + 1e-9


def test_the_parser_reads_front_and_bottom():
    block = {"wrist_camera": {"protrusion_m": 0.08, "behind_tip_m": 0.14, "half_width_m": 0.075,
                              "front_m": 0.08, "bottom_m": 0.035}}
    dims, up = wrist_camera_dims(block)
    assert dims == CameraDims(0.08, 0.14, 0.075, 0.08, 0.035) and up == (0.0, 0.0, 1.0)
    assert (dims.front_m, dims.bottom_m) == (0.08, 0.035)


@pytest.mark.parametrize("faces", [{"front_m": 0.14}, {"front_m": 0.2}, {"bottom_m": 0.08}, {"front_m": -0.01},
                                   {"bottom_m": float("nan")}, {"front_m": "0.05"}, {"bottom_m": True}])
def test_the_parser_refuses_a_cut_past_the_opposite_face(faces):
    block = {"wrist_camera": {"protrusion_m": 0.08, "behind_tip_m": 0.14, "half_width_m": 0.075, **faces}}
    with pytest.raises(SkillError, match="wrist_camera"):
        wrist_camera_dims(block)


def test_camera_box_defaults_are_todays_slab_bit_for_bit():
    T = np.eye(4)
    T[:3, :3] = rm.rotation_matrix([0.3, -0.5, 0.8], 1.1)
    T[:3, 3] = [0.31, -0.12, 0.27]
    old = rm.camera_box(T, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=0.08, behind_m=0.14, half_width_m=0.075)
    new = rm.camera_box(T, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=0.08, behind_m=0.14, half_width_m=0.075,
                        front_m=0.0, bottom_m=0.0)
    for a, b in ((old.center, new.center), (old.axes, new.axes), (old.half, new.half)):
        np.testing.assert_array_equal(a, b)
    # the reference slab: approach -behind..0, up 0..protrusion
    np.testing.assert_allclose(old.half, [0.07, 0.075, 0.04], atol=1e-15)


def test_camera_box_spans_minus_behind_to_minus_front_and_bottom_to_top():
    box = rm.camera_box(np.eye(4), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=0.08, behind_m=0.14,
                        half_width_m=0.075, front_m=0.08, bottom_m=0.035)
    c = box.corners()
    np.testing.assert_allclose(c.min(axis=0), [-0.14, -0.075, 0.035], atol=1e-12)
    np.testing.assert_allclose(c.max(axis=0), [-0.08, 0.075, 0.08], atol=1e-12)
    assert box.min_z() == pytest.approx(0.035)


@pytest.mark.parametrize("kw", [{"front_m": 0.14}, {"front_m": 0.15}, {"bottom_m": 0.08}, {"bottom_m": 0.09},
                                {"front_m": -1e-3}, {"bottom_m": -1e-3}, {"front_m": float("nan")},
                                {"bottom_m": float("inf")}, {"front_m": True}, {"bottom_m": None}])
def test_camera_box_refuses_a_front_or_bottom_past_the_opposite_face(kw):
    with pytest.raises(SkillError, match="wrist_camera"):
        rm.camera_box(np.eye(4), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=0.08, behind_m=0.14,
                      half_width_m=0.075, **kw)


def test_no_cut_adds_no_rule_a_zero_depth_box_stays_legal():
    """front/bottom 0 (no cut) must not change which sizes parse: a box of
    zero depth or height was legal before and stays legal."""
    box = rm.camera_box(np.eye(4), [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], protrusion_m=0.0, behind_m=0.0,
                        half_width_m=0.075)
    np.testing.assert_array_equal(box.half, [0.0, 0.075, 0.0])


def test_the_profile_comment_states_the_tested_part_boxes():
    """configs/arms/rebot_rs.yaml documents the part boxes the calipers PR
    will switch on; they must be the boxes pinned against the URDF here."""
    import yaml

    text = (ROOT / "configs" / "arms" / "rebot_rs.yaml").read_text()
    stated = {}
    for line in text.splitlines():
        body = line.strip().lstrip("#").strip()
        for name in ("housing", "fingers"):
            if body.startswith(f"{name}: {{min:"):
                stated[name] = yaml.safe_load(body)[name]
    assert stated == {"housing": _box_block(HOUSING), "fingers": _box_block(FINGERS)}


def test_spatial_tools_check_the_cut_box_too():
    """One parser, one geometry function, two layers: a TCP pointing straight
    down 1 cm over the table puts today's slab under the 2 cm floor; with the
    front cut at 8 cm the camera starts 9 cm up and the move is fine."""
    from cascade.skills import spatial_moves

    down = np.eye(4)
    down[:3, :3] = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])   # approach -z, camera up +x
    down[:3, 3] = [0.3, 0.0, 0.01]
    limits = SimpleNamespace(table_z=0.0, table_clearance=0.02, keep_out=[])
    rt = SimpleNamespace(_tool_axis_order="down_open",
                         arm=SimpleNamespace(harness=SimpleNamespace(limits=limits, occupancy=None)))
    slab = spatial_moves.settings(SimpleNamespace(cfg=_cfg()))
    assert slab.wrist_camera == TODAY_SLAB
    found = spatial_moves._camera_volume(rt, [down], slab)
    assert found is not None and found[0] == "wrist_camera" and "table" in found[1]
    cut = spatial_moves.settings(SimpleNamespace(cfg=_cut_camera(_cfg(), front_m=0.08)))
    assert cut.wrist_camera == TODAY_SLAB._replace(front_m=0.08)
    assert spatial_moves._camera_volume(rt, [down], cut) is None
    # level, 1 cm over the table: today's slab bottom sits at the finger axis;
    # raised to 5 cm it clears the 2 cm floor
    level = np.eye(4)
    level[:3, 3] = [0.3, 0.0, 0.01]                                             # approach +x, camera up +z
    found = spatial_moves._camera_volume(rt, [level], slab)
    assert found is not None and "table" in found[1]
    raised = spatial_moves.settings(SimpleNamespace(cfg=_cut_camera(_cfg(), bottom_m=0.05)))
    assert raised.wrist_camera == TODAY_SLAB._replace(bottom_m=0.05)
    assert spatial_moves._camera_volume(rt, [level], raised) is None


# ── the URDF part boxes cover the real hulls ───────────────────────────────


@needs_pin
def test_urdf_part_boxes_cover_the_hulls_closed_and_fully_open():
    """The housing box covers the `gripper_end` hull and the fingers box both
    finger hulls at every jaw gap, each face at most 5 mm outside the hull's
    extreme (rounded up, not padded)."""
    from cascade.control.kinematics import Kinematics
    from cascade.grasping.gripper_clearance import ApproachClearance, load_geometry

    arm = _cfg().arm
    kin = Kinematics(arm.model, arm.get("ee_frame", "gripper_end"), 6, joint_signs=arm.get("joint_signs"))
    clearance = ApproachClearance(load_geometry(arm.model), kin)
    fully_open = 2.0 * max(hi for _, _, hi in clearance.geometry.finger_joints)
    rng = np.random.default_rng(5)
    home = np.asarray(arm.home_q, dtype=float)
    lo_hi = {"housing": [np.full(3, np.inf), np.full(3, -np.inf)],
             "fingers": [np.full(3, np.inf), np.full(3, -np.inf)]}
    for q in (home, *(home + rng.uniform(-0.6, 0.6, 6) for _ in range(3))):
        T = kin.fk(q)
        for gap in (0.0, 0.03, 0.09, fully_open):
            placed = clearance.placed(q, gap)
            local = {n: (v - T[:3, 3]) @ T[:3, :3] for n, v in placed.items()}
            for name, pts in (("housing", local["gripper_end"]),
                              ("fingers", np.vstack([local["gripper_left"], local["gripper_right"]]))):
                lo_hi[name][0] = np.minimum(lo_hi[name][0], pts.min(axis=0))
                lo_hi[name][1] = np.maximum(lo_hi[name][1], pts.max(axis=0))
    for name, (lo, hi) in (("housing", HOUSING), ("fingers", FINGERS)):
        seen_lo, seen_hi = lo_hi[name]
        # 1 um: the URDF finger stroke is not exactly square to the approach,
        # its fully open tip vertex sits 0.18 um past the fingertip plane
        assert np.all(np.asarray(lo) <= seen_lo + 1e-6) and np.all(seen_hi <= np.asarray(hi) + 1e-6), name
        assert np.all(seen_lo - np.asarray(lo) < 0.005) and np.all(np.asarray(hi) - seen_hi < 0.005), name
    # fully open is the widest: the URDF fingers reach +-8.48 cm at the roots
    assert lo_hi["fingers"][1][1] == pytest.approx(0.0848, abs=5e-4)
