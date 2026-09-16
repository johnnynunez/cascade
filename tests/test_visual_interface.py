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

from cascade.perception.visual_interface import VisualInterface, configured_grasp_band

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
    assert "1 badges | 2 tracked objects | grid = 5 cm" in labels
    assert "1" not in labels and "2" in labels


def test_empty_projection_key_does_not_claim_an_empty_scene():
    key = VisualInterface.describe([])
    assert "no tracked-object projections" in key
    assert "does not establish that the scene is empty" in key
