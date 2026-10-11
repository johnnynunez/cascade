"""Pure frame/magnitude resolution behind the natural-language motion tools.

`cascade.skills.relative_motion` turns what a chat model passes for a vague
instruction ("a bit to the right", "bring it toward me", "tilt it down") into
ONE base-frame displacement or rotation plus a sentence saying how it was
read. Nothing here touches an arm: these tests pin the interpretation, the
caps and the input validation, which are the parts where a wrong guess is a
safety problem (the same word means opposite directions in two frames).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cascade.skills import relative_motion as rm
from cascade.types import SkillError

# The shipped mock camera: 0.6 m above the table looking straight down,
# cam x -> base -y, cam y -> base -x, cam z -> base -z (configs/cameras/mock.yaml).
TOP_DOWN = np.array([
    [0.0, -1.0, 0.0, 0.28],
    [-1.0, 0.0, 0.0, 0.00],
    [0.0, 0.0, -1.0, 0.60],
    [0.0, 0.0, 0.0, 1.00],
])

# The reference rig's scene camera (configs/cameras/d455f_scene.yaml inline
# guess): on the robot's left (+y), looking back across the table (-y) and
# down. Image right is roughly robot -x.
OBLIQUE = np.array([
    [-0.99151, 0.12855, -0.01949, 0.23383],
    [0.06771, 0.38250, -0.92147, 0.54274],
    [-0.11100, -0.91497, -0.38796, 0.46806],
    [0.0, 0.0, 0.0, 1.0],
])


# ── magnitudes ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("amount, metres", [("tiny", 0.01), ("small", 0.02),
                                            ("medium", 0.05), ("large", 0.10)])
def test_vague_amount_words_map_to_fixed_distances(amount, metres):
    distance, source = rm.resolve_distance(None, amount)
    assert distance == pytest.approx(metres)
    assert amount in source


def test_no_size_given_means_the_medium_default():
    distance, source = rm.resolve_distance(None, None)
    assert distance == pytest.approx(rm.AMOUNTS_M["medium"])
    assert "default" in source


def test_an_explicit_distance_wins_over_an_amount_word_and_says_so():
    distance, source = rm.resolve_distance(0.03, "large")
    assert distance == pytest.approx(0.03)
    assert "distance_m" in source and "ignored" in source


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.02, 0.0, 0.001, 0.16, 5.0,
                                 "far", True])
def test_absurd_or_out_of_cap_distances_are_refused(bad):
    with pytest.raises(SkillError):
        rm.resolve_distance(bad, None)


def test_the_cap_itself_is_allowed():
    assert rm.resolve_distance(rm.MAX_STEP_M, None)[0] == pytest.approx(0.15)


def test_unknown_amount_word_is_refused():
    with pytest.raises(SkillError, match="amount"):
        rm.resolve_distance(None, "enormous")


@pytest.mark.parametrize("bad", [float("nan"), -5.0, 0.0, 0.5, 31.0, 90.0])
def test_rotation_angles_are_capped_and_validated(bad):
    with pytest.raises(SkillError):
        rm.resolve_angle(bad, None)


def test_rotation_amount_words():
    assert rm.resolve_angle(None, "small")[0] == pytest.approx(10.0)
    assert rm.resolve_angle(None, None)[0] == pytest.approx(rm.AMOUNTS_DEG["medium"])
    assert rm.resolve_angle(30.0, None)[0] == pytest.approx(30.0)


# ── viewer frame from the scene camera ──────────────────────────────────────


def test_top_down_camera_viewer_axes():
    axes = rm.viewer_axes(TOP_DOWN, camera="mock")
    np.testing.assert_allclose(axes.right, [0, -1, 0], atol=1e-9)
    # up x right: for a top-down view "away" is "up on the screen"
    np.testing.assert_allclose(axes.away, [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(axes.up, [0, 0, 1], atol=1e-9)
    assert axes.camera == "mock"


def test_oblique_scene_camera_right_is_the_robots_back():
    """The reference rig: camera on the robot's left looking back across the
    table. Screen-right is (almost) robot -x and toward-the-camera is +y --
    the reason a silent frame guess would be dangerous."""
    axes = rm.viewer_axes(OBLIQUE, camera="d455f_scene")
    assert axes.right[0] < -0.99 and abs(axes.right[2]) < 1e-9
    assert axes.away[1] < -0.99            # away from the camera = robot -y
    assert np.dot(axes.right, axes.away) == pytest.approx(0.0, abs=1e-9)
    # right-handed: right x away = up
    np.testing.assert_allclose(np.cross(axes.right, axes.away), [0, 0, 1], atol=1e-9)


def test_a_camera_whose_image_x_is_vertical_has_no_screen_right():
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # x_cam = +z
    with pytest.raises(SkillError, match="viewer"):
        rm.viewer_axes(T)


def test_an_uncalibrated_camera_transform_is_refused():
    with pytest.raises(SkillError, match="viewer"):
        rm.viewer_axes(np.full((4, 4), np.nan))


# ── translations ────────────────────────────────────────────────────────────


def _no_viewer():
    raise SkillError("no calibrated scene camera")


def _tool():
    return rm.tool_axes(np.eye(3), "down_open")  # approach = +x (mock home)


def test_bare_right_defaults_to_the_robot_base_frame():
    move = rm.resolve_translation("right", distance=0.05, viewer=_no_viewer, tool=_tool)
    assert move.frame == "base" == rm.DEFAULT_FRAME
    np.testing.assert_allclose(move.delta, [0, -0.05, 0])
    assert "robot" in move.words and "-y" in move.words


def test_viewer_right_on_the_reference_rig_is_robot_back():
    move = rm.resolve_translation("right", frame="viewer", distance=0.05,
                                  viewer=lambda: rm.viewer_axes(OBLIQUE, camera="scene"), tool=_tool)
    assert move.frame == "viewer"
    assert move.delta[0] < -0.049
    assert "viewer" in move.words and "scene" in move.words


def test_toward_the_viewer_implies_the_viewer_frame():
    move = rm.resolve_translation("toward_viewer", distance=0.02,
                                  viewer=lambda: rm.viewer_axes(OBLIQUE), tool=_tool)
    assert move.frame == "viewer"
    assert move.delta[1] > 0.0199          # the camera is on the robot's left


@pytest.mark.parametrize("direction, frame", [("toward_viewer", "base"), ("away_from_viewer", "tool"),
                                              ("in", "base"), ("out", "viewer"),
                                              ("left", "tool"), ("forward", "tool")])
def test_contradictory_frame_and_direction_are_refused(direction, frame):
    with pytest.raises(SkillError, match="frame"):
        rm.resolve_translation(direction, frame=frame, distance=0.02,
                               viewer=lambda: rm.viewer_axes(OBLIQUE), tool=_tool)


def test_tool_in_follows_the_gripper_approach():
    move = rm.resolve_translation("in", distance=0.03, viewer=_no_viewer, tool=_tool)
    assert move.frame == "tool"
    np.testing.assert_allclose(move.delta, [0.03, 0, 0])


def test_up_is_world_vertical_in_every_frame():
    for frame in rm.FRAMES:
        move = rm.resolve_translation("up", frame=frame, distance=0.02,
                                      viewer=lambda: rm.viewer_axes(OBLIQUE), tool=_tool)
        np.testing.assert_allclose(move.delta, [0, 0, 0.02])


def test_a_free_vector_in_the_base_frame():
    move = rm.resolve_translation(vector=[0.0, 0.03, 0.04], viewer=_no_viewer, tool=_tool)
    np.testing.assert_allclose(move.delta, [0.0, 0.03, 0.04])
    assert move.distance == pytest.approx(0.05)


def test_a_free_vector_in_the_viewer_frame_is_right_away_up():
    axes = rm.viewer_axes(OBLIQUE)
    move = rm.resolve_translation(vector=[0.03, 0.0, 0.01], frame="viewer",
                                  viewer=lambda: axes, tool=_tool)
    np.testing.assert_allclose(move.delta, 0.03 * axes.right + [0, 0, 0.01], atol=1e-12)


@pytest.mark.parametrize("vector", [[0.0, float("nan"), 0.0], [0.1, 0.1, 0.1], [0.001, 0, 0],
                                    [0.01, 0.01], "left", [True, 0, 0]])
def test_bad_free_vectors_are_refused(vector):
    with pytest.raises(SkillError):
        rm.resolve_translation(vector=vector, viewer=_no_viewer, tool=_tool)


def test_vector_and_direction_together_are_ambiguous():
    with pytest.raises(SkillError, match="either"):
        rm.resolve_translation("left", vector=[0, 0.02, 0], viewer=_no_viewer, tool=_tool)
    with pytest.raises(SkillError, match="vector"):
        rm.resolve_translation(vector=[0, 0.02, 0], distance=0.05, viewer=_no_viewer, tool=_tool)


def test_go_back_is_refused_as_ambiguous_with_a_pointer_to_undo():
    with pytest.raises(SkillError, match="undo_move"):
        rm.resolve_translation("go back", distance=0.02, viewer=_no_viewer, tool=_tool)


def test_viewer_frame_without_a_camera_says_to_use_the_base_frame():
    with pytest.raises(SkillError, match="frame=base|frame='base'|base frame"):
        rm.resolve_translation("left", frame="viewer", distance=0.02, viewer=_no_viewer, tool=_tool)


@pytest.mark.parametrize("word, canonical", [("towards me", "toward_viewer"), ("Backwards", "back"),
                                             ("away from me", "away_from_viewer"), ("LEFT", "left")])
def test_unambiguous_aliases_are_normalized(word, canonical):
    assert rm.normalize_direction(word) == canonical


def test_unknown_frame_and_direction_are_refused():
    with pytest.raises(SkillError, match="frame"):
        rm.resolve_translation("left", frame="world", distance=0.02, viewer=_no_viewer, tool=_tool)
    with pytest.raises(SkillError, match="direction"):
        rm.resolve_translation("sideways", distance=0.02, viewer=_no_viewer, tool=_tool)


def test_the_screen_description_names_what_the_viewer_sees():
    axes = rm.viewer_axes(OBLIQUE, camera="scene")
    # robot right (-y) is "away from the camera" on the reference rig
    text = rm.describe_on_screen(np.array([0.0, -0.05, 0.0]), axes)
    assert "away from the camera" in text
    assert "right" in rm.describe_on_screen(np.array([-0.05, 0.0, 0.0]), axes)


# ── orientation nudges ──────────────────────────────────────────────────────


def _down_tool():
    # a top-down reBot gripper: approach (col0) = -z
    R = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    return rm.tool_axes(R, "down_open")


def _forward_tool():
    return rm.tool_axes(np.eye(3), "down_open")  # approach = +x


def test_roll_clockwise_is_a_positive_turn_about_the_approach():
    """Clockwise as seen looking along the jaws: for a top-down gripper that
    is clockwise seen from above, i.e. NEGATIVE about +z."""
    rot = rm.resolve_rotation("clockwise", angle_deg=10.0, tool=_down_tool, viewer=_no_viewer)
    assert rot.kind == "roll"
    np.testing.assert_allclose(rot.axis, [0, 0, -1], atol=1e-12)
    assert "clockwise" in rot.words and "10" in rot.words


def test_tilt_down_swings_a_forward_pointing_gripper_toward_the_table():
    rot = rm.resolve_rotation("down", angle_deg=15.0, tool=_forward_tool, viewer=_no_viewer)
    assert rot.kind == "tilt"
    R = rm.rotation_matrix(rot.axis, math.radians(15.0))
    new_approach = R @ np.array([1.0, 0.0, 0.0])
    assert new_approach[2] == pytest.approx(-math.sin(math.radians(15.0)))


def test_tilting_toward_where_the_gripper_already_points_is_refused():
    with pytest.raises(SkillError, match="already"):
        rm.resolve_rotation("down", angle_deg=10.0, tool=_down_tool, viewer=_no_viewer)


def test_a_tilt_past_the_target_direction_is_refused_with_the_room_left():
    # 20 deg below horizontal; tilting down 30 deg would overshoot nothing,
    # but tilting a gripper that is 5 deg from straight down by 10 deg would
    # sweep past vertical -- the room left is reported
    a = np.array([math.sin(math.radians(5.0)), 0.0, -math.cos(math.radians(5.0))])
    R = np.column_stack([a, [0, 1, 0], np.cross(a, [0, 1, 0])])
    with pytest.raises(SkillError, match="5 deg"):
        rm.resolve_rotation("down", angle_deg=10.0, tool=lambda: rm.tool_axes(R, "down_open"),
                            viewer=_no_viewer)


def test_tilt_in_the_viewer_frame_uses_the_screen_direction():
    axes = rm.viewer_axes(OBLIQUE)
    rot = rm.resolve_rotation("toward_viewer", angle_deg=10.0, tool=_down_tool, viewer=lambda: axes)
    assert rot.frame == "viewer"
    R = rm.rotation_matrix(rot.axis, math.radians(10.0))
    tip = R @ np.array([0.0, 0.0, -1.0])
    assert float(tip @ -axes.away) > 0.17   # the jaws swing toward the camera


@pytest.mark.parametrize("direction", ["sideways", "go back", 3])
def test_unknown_rotation_words_are_refused(direction):
    with pytest.raises(SkillError):
        rm.resolve_rotation(direction, angle_deg=10.0, tool=_down_tool, viewer=_no_viewer)


def test_roll_takes_no_frame():
    with pytest.raises(SkillError, match="frame"):
        rm.resolve_rotation("clockwise", frame="viewer", angle_deg=10.0, tool=_down_tool,
                            viewer=_no_viewer)


# ── history ─────────────────────────────────────────────────────────────────


def test_history_is_bounded_and_last_in_first_out():
    h = rm.MoveHistory(maxlen=3)
    for i in range(5):
        h.push(rm.MoveRecord(tool="move_relative", words=f"m{i}", waypoints=[np.zeros(6), np.ones(6)],
                             tcp_end=np.eye(4)))
    assert len(h) == 3
    assert h.last().words == "m4"
    assert h.pop().words == "m4" and h.last().words == "m3"
    h.clear()
    assert len(h) == 0 and h.last() is None


def test_pose_match_tolerances_for_a_stale_history():
    T = np.eye(4)
    near = T.copy()
    near[:3, 3] = [0.005, 0, 0]
    far = T.copy()
    far[:3, 3] = [0.05, 0, 0]
    turned = T.copy()
    turned[:3, :3] = rm.rotation_matrix(np.array([0.0, 0.0, 1.0]), math.radians(20.0))
    assert rm.pose_matches(T, near)
    assert not rm.pose_matches(T, far)
    assert not rm.pose_matches(T, turned)


# ── wrist camera must stay on top (user safety requirement) ─────────────────


def test_rolling_upside_down_is_refused_as_a_word():
    with pytest.raises(SkillError, match="camera must stay on top"):
        rm.normalize_rotation_direction("upside down")


def test_camera_up_check_refuses_a_path_that_turns_the_tcp_z_below_horizontal():
    up = np.array([0.0, 0.0, 1.0])  # reBot: TCP +z is the wrist camera's up
    start = np.eye(3)
    rolled_120 = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(120.0))
    reason = rm.camera_up_violation([start, rolled_120], up, min_z=0.0)
    assert reason is not None and "camera must stay on top" in reason
    rolled_60 = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(60.0))
    assert rm.camera_up_violation([start, rolled_60], up, min_z=0.0) is None


def test_camera_up_check_samples_the_whole_path_not_only_the_end():
    up = np.array([0.0, 0.0, 1.0])
    mid = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(100.0))
    assert rm.camera_up_violation([np.eye(3), mid, np.eye(3)], up, min_z=0.0) is not None


def test_camera_up_check_lets_an_already_low_camera_only_come_back_up():
    """Escape rule (mirrors the harness's): from a pose already below the
    limit, a move that raises the camera is allowed, one that lowers it is not."""
    up = np.array([0.0, 0.0, 1.0])
    low = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(100.0))
    better = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(95.0))
    worse = rm.rotation_matrix(np.array([1.0, 0.0, 0.0]), math.radians(110.0))
    assert rm.camera_up_violation([low, better], up, min_z=0.0) is None
    assert rm.camera_up_violation([low, worse], up, min_z=0.0) is not None


# ── wrist camera volume (it sticks out 6-8 cm above the gripper) ────────────

_DOWN_OPEN_APPROACH = np.array([1.0, 0.0, 0.0])   # reBot: approach = TCP x
_CAM_UP = np.array([0.0, 0.0, 1.0])               # reBot: camera up = TCP z


def _box_at(R, p=(0.0, 0.0, 0.0), protrusion=0.08, behind=0.12, half_width=0.04):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return rm.camera_box(T, _DOWN_OPEN_APPROACH, _CAM_UP, protrusion_m=protrusion,
                         behind_m=behind, half_width_m=half_width)


def test_camera_box_at_home_sits_on_top_and_behind_the_fingertips():
    box = _box_at(np.eye(3), p=(0.45, 0.0, 0.40))
    c = box.corners()
    np.testing.assert_allclose(c.min(axis=0), [0.33, -0.04, 0.40], atol=1e-12)
    np.testing.assert_allclose(c.max(axis=0), [0.45, 0.04, 0.48], atol=1e-12)
    assert box.min_z() == pytest.approx(0.40)


def test_pointing_down_the_camera_never_hangs_below_the_fingertips():
    down = np.column_stack([[0, 0, -1], [0, 1, 0], [1, 0, 0]])   # approach -z, camera up +x
    for roll in (0.0, 45.0, 90.0, 170.0):
        R = rm.rotation_matrix([0, 0, 1], math.radians(roll)) @ down
        assert _box_at(R, p=(0.3, 0.0, 0.05)).min_z() == pytest.approx(0.05, abs=1e-12)


def test_a_rolled_horizontal_gripper_drops_one_side_of_the_camera():
    """Rolled 80 deg the camera-up axis still points up (orientation rule
    passes) but the camera's side now hangs ~4 cm below the TCP."""
    R = rm.rotation_matrix([1, 0, 0], math.radians(80.0))
    box = _box_at(R, p=(0.45, 0.0, 0.10))
    assert float((R @ _CAM_UP)[2]) > 0.17
    assert box.min_z() == pytest.approx(0.10 - 0.04 * math.sin(math.radians(80.0)), abs=1e-12)


def test_camera_box_rejects_bad_dimensions():
    for kw in ({"protrusion": float("nan")}, {"behind": -0.01}, {"half_width": float("inf")},
               {"protrusion": 0.5}):
        with pytest.raises(SkillError, match="wrist_camera"):
            _box_at(np.eye(3), **kw)


def test_box_overlap_separated_overlapping_and_contained():
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    I, e = np.eye(3), np.array([0.5, 0.5, 0.5])
    assert not rm.box_overlaps_aabb(np.array([1.6, 0.0, 0.0]), I, e, lo, hi)
    assert rm.box_overlaps_aabb(np.array([1.4, 0.0, 0.0]), I, e, lo, hi)
    assert rm.box_overlaps_aabb(np.zeros(3), I, e, lo, hi)                     # inside
    assert rm.box_overlaps_aabb(np.zeros(3), I, np.array([5.0, 5.0, 5.0]), lo, hi)  # engulfs


def test_box_overlap_is_exact_for_a_rotated_box_near_a_corner():
    """45-deg box beside the AABB's corner: their axis-aligned bounds
    overlap, the boxes do not (separated along the box's own axis)."""
    lo, hi = -np.ones(3), np.ones(3)
    A = rm.rotation_matrix([0, 0, 1], math.radians(45.0))
    c = np.array([2.2, 2.2, 0.0])
    assert not rm.box_overlaps_aabb(c, A, np.ones(3), lo, hi)
    assert rm.box_overlaps_aabb(np.array([1.6, 1.6, 0.0]), A, np.ones(3), lo, hi)


def _dense_points(c, A, e, n=25):
    g = np.linspace(-1.0, 1.0, n)
    local = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3) * e
    return c + local @ A.T


def test_box_overlap_finds_an_edge_to_edge_separation():
    """No face axis of either box separates these two; only an edge-cross
    axis does (found by search). Dense sampling confirms no part of the box
    is inside the AABB."""
    raw = np.array([[-0.853829, 0.08784, -0.513088],
                    [0.500891, 0.406974, -0.763859],
                    [0.141716, -0.909207, -0.391484]])
    U, _, Vt = np.linalg.svd(raw)
    A = U @ Vt
    c, e = np.array([0.5645, -1.5168, -1.5986]), np.array([0.4203, 0.6494, 0.5197])
    lo, hi = -np.ones(3), np.ones(3)
    pts = _dense_points(c, A, e)
    assert not np.any(np.all((pts >= lo) & (pts <= hi), axis=1))
    assert not rm.box_overlaps_aabb(c, A, e, lo, hi)


def test_box_overlap_finds_a_separation_only_the_box_face_shows():
    """Neither a world axis nor an edge cross product separates these; only
    one of the oriented box's own face normals does (found by search)."""
    raw = np.array([[-0.615732, -0.671355, 0.4125],
                    [-0.503363, 0.737899, 0.449589],
                    [-0.606218, 0.06919, -0.792283]])
    U, _, Vt = np.linalg.svd(raw)
    A = U @ Vt
    c, e = np.array([-1.318, -1.1239, -1.5623]), np.array([0.4117, 0.4804, 0.6612])
    lo, hi = -np.ones(3), np.ones(3)
    pts = _dense_points(c, A, e)
    assert not np.any(np.all((pts >= lo) & (pts <= hi), axis=1))
    assert not rm.box_overlaps_aabb(c, A, e, lo, hi)


def test_box_overlap_finds_a_separation_only_a_world_axis_shows():
    """Only the AABB's own (world) face normals separate these (found by
    search): the test is exact, not merely conservative."""
    raw = np.array([[0.651153, -0.6296, -0.423796],
                    [-0.340817, -0.741506, 0.577939],
                    [-0.678117, -0.23189, -0.697412]])
    U, _, Vt = np.linalg.svd(raw)
    A = U @ Vt
    c, e = np.array([0.1413, -0.003, -1.8294]), np.array([0.3102, 0.6935, 0.531])
    lo, hi = -np.ones(3), np.ones(3)
    pts = _dense_points(c, A, e)
    assert not np.any(np.all((pts >= lo) & (pts <= hi), axis=1))
    assert not rm.box_overlaps_aabb(c, A, e, lo, hi)


def test_box_overlap_never_misses_a_sampled_intersection():
    rng = np.random.default_rng(7)
    lo, hi = -np.ones(3), np.ones(3)
    for _ in range(300):
        A = rm.rotation_matrix(rng.normal(size=3), rng.uniform(0, math.pi))
        e, c = rng.uniform(0.1, 0.8, 3), rng.uniform(-2.2, 2.2, 3)
        sampled = np.any(np.all((_dense_points(c, A, e, 9) >= lo) & (_dense_points(c, A, e, 9) <= hi),
                                axis=1))
        if sampled:
            assert rm.box_overlaps_aabb(c, A, e, lo, hi)


def test_camera_points_cover_the_box_at_the_requested_spacing():
    box = _box_at(np.eye(3))
    pts = box.points(spacing_m=0.02)
    assert len(pts) == 7 * 5 * 5                   # 12 cm / 8 cm / 8 cm at <= 2 cm
    np.testing.assert_allclose(pts.min(axis=0), box.corners().min(axis=0), atol=1e-12)
    np.testing.assert_allclose(pts.max(axis=0), box.corners().max(axis=0), atol=1e-12)
