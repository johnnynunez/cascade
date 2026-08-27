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

from cascade.perception.visual_interface import VisualInterface

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
