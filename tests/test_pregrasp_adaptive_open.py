"""Opt-in adaptive pre-grasp opening: open to the grasp width + a margin.

Ported from Seeed WRC commit a2d5950 ("add gripper control loop and adaptive
opening"): `skill_grasp_object` opened the jaws to `grasp.width_m + 10 mm`
instead of fully, so the fingers sweep less of the scene on the descent. In
cascade it is an arm-profile tunable, `gripper.pregrasp_open_margin_m`, and
OFF unless a profile sets it: the opening is computed with the framework's
LINEAR angle->width map, and on the RS that map is not yet re-measured (WRC
maps 5.0 rad -> 0.09-0.095 m, cascade's profile maps 6.2 rad -> 0.09 m), so
an opening computed from a wrong scale could be narrower than the object.

Invariants pinned here: the commanded opening is never narrower than the
grasp width (margin >= 0, else a config error), never wider than full open,
polarity-aware, and an unset margin is byte-identical to the old full open.
"""

from __future__ import annotations

import math

import pytest

from conftest import needs_pin

from cascade.grasping.force import pregrasp_open_position
from cascade.types import SkillError


def _opening_m(pos, open_pos, closed_pos, max_width):
    return (pos - closed_pos) / (open_pos - closed_pos) * max_width


def test_unset_margin_opens_fully():
    assert pregrasp_open_position(0.03, None, 6.2, 0.0, 0.09) == 6.2


@pytest.mark.parametrize("open_pos, closed_pos", [(6.2, 0.0), (0.0, 1.0), (-6.8, 0.0)])
def test_opens_to_width_plus_margin_on_either_polarity(open_pos, closed_pos):
    pos = pregrasp_open_position(0.03, 0.01, open_pos, closed_pos, 0.09)
    assert _opening_m(pos, open_pos, closed_pos, 0.09) == pytest.approx(0.04)
    lo, hi = min(open_pos, closed_pos), max(open_pos, closed_pos)
    assert lo <= pos <= hi


def test_capped_at_full_open():
    assert pregrasp_open_position(0.085, 0.01, 6.2, 0.0, 0.09) == pytest.approx(6.2)


def test_zero_margin_is_exactly_the_grasp_width():
    pos = pregrasp_open_position(0.05, 0.0, 6.2, 0.0, 0.09)
    assert _opening_m(pos, 6.2, 0.0, 0.09) == pytest.approx(0.05)


@pytest.mark.parametrize("margin", [-0.005, math.nan, math.inf, "wide"])
def test_a_margin_that_could_close_on_the_object_is_refused(margin):
    with pytest.raises(SkillError, match="pregrasp_open_margin_m"):
        pregrasp_open_position(0.03, margin, 6.2, 0.0, 0.09)


@pytest.mark.parametrize("width", [math.nan, -0.01, 0.0, None])
def test_unknown_width_falls_back_to_full_open(width):
    """No usable width = no basis for a partial opening: open fully."""
    assert pregrasp_open_position(width, 0.01, 6.2, 0.0, 0.09) == 6.2


# ── wiring: skill_grasp_object uses it for the open phase ───────────────────


def _grasp_and_record(demo_cfg, tmp_path, margin):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    if margin is not None:
        demo_cfg._data["arm"]["gripper"]["pregrasp_open_margin_m"] = margin
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        arm.object_stop_frac = 0.5
        calls = []
        original = runtime.arm.set_gripper

        def record(pos, effort=1.0, **kw):
            calls.append(float(pos))
            return original(pos, effort, **kw)

        runtime.arm.set_gripper = record
        import time

        deadline = time.monotonic() + 10.0
        while runtime.beliefs.find("red cube") is None and time.monotonic() < deadline:
            time.sleep(0.05)
        with runtime.watcher.paused():
            result = runtime.execute("grasp_object", {"label": "red cube", "material": "rigid"})
        g = demo_cfg.arm.gripper
        return result, calls, float(g.open_pos), float(g.closed_pos), float(g.max_width_m)
    finally:
        shutdown_runtime(runtime, arm)


@needs_pin
def test_grasp_object_opens_fully_without_the_tunable(demo_cfg, tmp_path):
    result, calls, open_pos, _, _ = _grasp_and_record(demo_cfg, tmp_path, None)
    assert result["ok"], result
    assert calls[0] == pytest.approx(open_pos)


@needs_pin
def test_grasp_object_opens_to_width_plus_margin_with_the_tunable(demo_cfg, tmp_path):
    result, calls, open_pos, closed_pos, max_w = _grasp_and_record(demo_cfg, tmp_path, 0.01)
    assert result["ok"], result
    expected = min(float(result["grasp_width_m"]) + 0.01, max_w)
    # grasp_width_m is reported rounded to 1 mm
    assert _opening_m(calls[0], open_pos, closed_pos, max_w) == pytest.approx(expected, abs=1e-3)
    assert _opening_m(calls[0], open_pos, closed_pos, max_w) >= float(result["grasp_width_m"]) - 1e-3
