"""Tests for the VIA-style annotated view (perception/visual_interface.py).

ROADMAP open follow-up: `annotated_view` surfaced a stale 4th "cube" mark --
the belief store correctly never forgets a sighting (object permanence is
deliberate, see memory/beliefs.py), but the annotated view drew every belief
with equal confidence, so a single stale misdetection looked exactly like a
repeatedly-confirmed object to both the LLM agent and a human watching the
booth feed. These pin that a mark seen only once and not currently visible
renders/describes as UNCONFIRMED instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.visual_interface import (
    VisualInterface, configured_grasp_band, configured_reach_x,
)

# ── synthetic pinhole camera looking straight down (mirrors test_probe.py) ─

W, H = 64, 48
K = np.array([[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]])
T_BASE_CAM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _frame():
    return SimpleNamespace(rgb=np.zeros((H, W, 3), dtype=np.uint8), K=K, T_base_cam=T_BASE_CAM)


def _belief(label, position, last_seen_t, observations):
    return SimpleNamespace(
        label=label,
        position=np.asarray(position, dtype=float),
        last_seen_t=last_seen_t,
        observations=observations,
    )


def _vi():
    return VisualInterface(workspace={"min": [-1, -1, -1], "max": [1, 1, 1]})


def test_recently_seen_belief_is_confirmed_regardless_of_observation_count():
    b = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=1)
    _, marks = _vi().render(_frame(), beliefs=[b], now=100.0)
    assert len(marks) == 1
    assert marks[0].confirmed is True


def test_repeatedly_confirmed_belief_stays_confirmed_after_going_stale():
    b = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=3)
    _, marks = _vi().render(_frame(), beliefs=[b], now=110.0)  # 10 s stale
    assert marks[0].confirmed is True


def test_single_stale_sighting_is_unconfirmed_not_a_phantom_solid_mark():
    """The exact bug: one sighting, never re-observed, long gone -- still a
    permanent belief-store entry, but must not read as a real object."""
    b = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=1)
    _, marks = _vi().render(_frame(), beliefs=[b], now=110.0)  # 10 s stale
    assert len(marks) == 1
    assert marks[0].confirmed is False
    assert marks[0].as_dict()["confirmed"] is False


def test_describe_flags_unconfirmed_marks_in_the_llm_facing_text():
    b = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=1)
    _, marks = _vi().render(_frame(), beliefs=[b], now=110.0)
    text = VisualInterface.describe(marks)
    assert "UNCONFIRMED" in text


def test_age_s_reflects_real_staleness_not_a_constant_zero():
    """Previously read `getattr(b, "age", 0.0)`, a field ObjectBelief never
    sets -- age_s was silently always 0.0, hiding staleness entirely."""
    b = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=1)
    _, marks = _vi().render(_frame(), beliefs=[b], now=110.0)
    assert marks[0].as_dict()["age_s"] == 10.0


def test_default_view_does_not_draw_an_invented_grasp_band():
    view = _vi()
    actual, _ = view.render(_frame(), grid=False)
    without_envelope, _ = view.render(_frame(), grid=False, envelope=False)

    assert np.array_equal(actual, without_envelope)
    assert view.reach_x is None


@pytest.mark.parametrize("config,expected", [
    ({}, None),
    ({"reach_x_min": .2}, None),
    ({"reach_x_min": .3, "reach_x_max": .2}, None),
    ({"reach_x_min": .2, "reach_x_max": float("nan")}, None),
    ({"reach_x_min": .2, "reach_x_max": .3}, (.2, .3)),
])
def test_display_band_requires_explicit_finite_ordered_bounds(config, expected):
    assert configured_grasp_band(config) == expected


def test_marks_report_workspace_membership_without_certifying_reachability():
    belief = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=2)
    for top, inside in [(1.0, True), (.8, False)]:
        view = VisualInterface(workspace={"min": [-1, -1, -1], "max": [1, 1, top]})
        _, marks = view.render(_frame(), beliefs=[belief], now=100.0)
        result = marks[0].as_dict()

        assert result["reachable"] is None
        assert result["in_workspace"] is inside
        assert result["ik_checked"] is False
        assert "do not verify inverse kinematics" in VisualInterface.describe(marks)


@pytest.mark.parametrize("u,v,inside", [
    (32., 24., True), (-.2, 24., False), (64., 24., False),
    (32., -.2, False), (32., 48., False), (32., 62., False),
])
def test_offscreen_projections_are_not_drawn_or_claimed_as_visible_badges(u, v, inside):
    # At z=.9 the synthetic camera is .1 m above the point.
    position = [(u - W / 2) * .1 / 50, -(v - H / 2) * .1 / 50, .9]
    belief = _belief("cube", position, last_seen_t=100., observations=2)
    frame = _frame()
    image = frame.rgb.copy()
    marks = _vi()._draw_marks(image, [belief], T_BASE_CAM, K, now=100.)

    assert len(marks) == 1
    assert marks[0].as_dict()["projected_in_image"] is inside
    assert np.any(image != frame.rgb) == inside
    key = VisualInterface.describe(marks)
    assert ("projects outside this camera image; no badge" in key) == (not inside)
    assert "does not establish current visibility" in key


def test_offscreen_entries_keep_their_indices_and_count_separate_from_badges(monkeypatch):
    beliefs = [_belief("offscreen", [.09, -.07, .9], 100., 2),
               _belief("onscreen", [0., 0., .9], 100., 2)]
    labels = []
    monkeypatch.setattr("cascade.perception.visual_interface._put_label",
                        lambda image, text, *_a, **_kw: labels.append(text))

    _, marks = _vi().render(_frame(), beliefs=beliefs, now=100., grid=False)

    assert [mark.index for mark in marks] == [1, 2]
    assert [mark.extra["projected_in_image"] for mark in marks] == [False, True]
    assert "1 badges | 2 tracked objects | grid=5 cm at base Z=0.00m" in labels
    assert "1" not in labels and "2" in labels


def test_empty_projection_key_does_not_claim_an_empty_scene():
    key = VisualInterface.describe([])
    assert "no tracked-object projections" in key
    assert "does not establish that the scene is empty" in key

def test_fr3_has_no_invented_topdown_band_and_explicit_band_still_draws():
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="d435i", arm="fr3", llm="mock")
    assert configured_reach_x(cfg) is None
    vi = _vi()
    without_band, _ = vi.render(_frame(), grid=False)
    green = lambda img: (img[:, :, 1] > img[:, :, 0]) & (img[:, :, 1] > img[:, :, 2])
    assert not green(without_band).any()

    vi.reach_x = (0.155, 0.185)
    with_band, _ = vi.render(_frame(), grid=False)
    assert green(with_band).any()


def test_grid_labels_identify_base_coordinates_and_actual_plane_spacing(monkeypatch):
    import cascade.perception.visual_interface as module

    labels = []
    monkeypatch.setattr(module, "_put_label", lambda img, text, *a, **kw: labels.append(text))
    vi = VisualInterface(workspace={"min": [-0.1, -0.1, -0.2], "max": [0.1, 0.1, 0.5]},
                         grid_step_m=0.025, table_z=-0.10)
    vi.render(_frame())
    assert "X=0.00m" in labels
    assert any("grid=2.5 cm at base Z=-0.10m" in label for label in labels)
    assert any("X forward | Y left | Z up" in label for label in labels)


def test_workspace_status_checks_height_without_claiming_ik_reachability():
    vi = VisualInterface(workspace={"min": [-1, -1, -0.01], "max": [1, 1, 0.5]})
    belief = _belief("cube", [0.1, 0.0, 0.9], last_seen_t=100.0, observations=2)
    _, marks = vi.render(_frame(), beliefs=[belief], now=100.0)
    assert marks[0].reachable is None
    assert marks[0].extra["in_workspace"] is False
    assert marks[0].extra["ik_checked"] is False
    assert "outside the workspace AABB" in vi.describe(marks)


def test_dashboard_agent_view_uses_selected_camera_geometry():
    import cv2
    from cascade.apps.stream_server import StreamServer
    from cascade.config import Cfg

    wide_k = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    frame = SimpleNamespace(rgb=np.zeros((480, 640, 3), dtype=np.uint8), K=wide_k,
                            T_base_cam=None)
    over = SimpleNamespace(name="over", latest=lambda: frame)
    side = SimpleNamespace(name="side", latest=lambda: frame)
    side_t = T_BASE_CAM.copy()
    side_t[0, 3] = 0.20
    over_ext = SimpleNamespace(cam_to_base=lambda: T_BASE_CAM)
    side_ext = SimpleNamespace(cam_to_base=lambda: side_t)
    belief = _belief("cube", [0.1, 0., 0.1], last_seen_t=100., observations=2)
    rig = SimpleNamespace(primary=over, get=lambda name: {"over": over, "side": side}[name])
    runtime = SimpleNamespace(
        cfg=Cfg({"safety": {"workspace": {}, "table_z": 0.}, "grasp": {}}),
        extrinsics=over_ext, rig=rig,
        watcher=SimpleNamespace(_cams=[SimpleNamespace(stream=over, extrinsics=over_ext),
                                       SimpleNamespace(stream=side, extrinsics=side_ext)]),
        beliefs=SimpleNamespace(all=lambda: [belief]), _tcp=lambda: None,
    )
    server = StreamServer(rig, runtime_fn=lambda: runtime)
    rgb = cv2.imdecode(np.frombuffer(server.annotated_view_jpeg("side"), dtype=np.uint8), cv2.IMREAD_COLOR)
    # The side camera sees x=.10 from its own x=.20 position: u≈264.
    # Borrowing the primary camera's pose puts the mark at u≈376 instead.
    # Use the lower ring, below the text that extends right from the mark.
    patch = lambda u: rgb[248:256, u-15:u+16].astype(int)
    selected = patch(264)
    wrong = patch(376)
    assert ((selected[:, :, 1] - selected[:, :, 2]) > 40).any()
    assert not ((wrong[:, :, 1] - wrong[:, :, 2]) > 40).any()
