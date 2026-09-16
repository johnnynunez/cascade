"""The verifier must score against the destination, and must not be looser
than the task's own success criterion.

Two bugs found on LIBERO-Pro `libero_spatial_swap` (task 3), where the
position-swap perturbation moves the destination after the belief was seeded:

1. `_check_relocated` compared the object's final pose against the point the
   SKILL SAID it aimed at. A stale aim therefore confirmed itself: placement
   was 0.9 cm from the aim, so the check passed, while the object sat 4.3 cm
   from the actual plate and LIBERO scored the episode false.

2. After scoring against the destination, it still confirmed, because the
   threshold was `SAME_PLACE_M * 2` = 10 cm while LIBERO's On() predicate
   requires 3 cm. A verifier three times looser than the success criterion
   cannot catch a near miss, which is exactly the failure worth catching.

Measured effect before the fix: verified false claims were 13/100 on the swap
suite versus 5/100 on suites where nothing moves.
"""

from __future__ import annotations

import numpy as np

from cascade.agent.effects import (
    DEST_TOLERANCE_M,
    REFUTED,
    SAME_PLACE_M,
    UNVERIFIED,
    PostconditionChecker,
    annotate_result,
)


def _checker(poses: dict):
    return PostconditionChecker(object_pose=lambda name: poses.get(name))


def _run(checker, args, result, before):
    return checker.verify("pick_and_place", args, result, before)


def test_place_is_refuted_when_it_misses_the_named_destination():
    """The exact episode that used to confirm: LIBERO-Pro swap, task 3."""
    poses = {
        "bowl": np.array([0.024, -0.263, 0.918]),
        "plate": np.array([-0.008, -0.291, 0.902]),
    }
    pc = _run(
        _checker(poses),
        {"object": "bowl", "destination": "plate"},
        # The skill aimed here and landed 0.9 cm away: excellent placement,
        # stale target.
        {"ok": True, "placed_at": [0.016, -0.267, 0.945]},
        {"label": "bowl", "pose": [0.4, 0.1, 0.9]},
    )
    assert pc.status == REFUTED, (
        f"expected REFUTED, got {pc.status}: {pc.evidence}"
    )
    assert "plate" in pc.evidence
    assert pc.measured["dest_err_m"] > DEST_TOLERANCE_M


def test_place_is_confirmed_when_it_reaches_the_destination():
    poses = {
        "bowl": np.array([0.001, -0.290, 0.918]),
        "plate": np.array([0.0, -0.291, 0.902]),
    }
    pc = _run(
        _checker(poses),
        {"object": "bowl", "destination": "plate"},
        {"ok": True, "placed_at": [0.0, -0.291, 0.945]},
        {"label": "bowl", "pose": [0.4, 0.1, 0.9]},
    )
    assert pc.status != REFUTED, pc.evidence
    assert pc.measured["dest_err_m"] < DEST_TOLERANCE_M


def test_destination_tolerance_is_not_looser_than_liberos_predicate():
    """LIBERO's On() needs < 3 cm. A verifier looser than the benchmark it is
    evaluated against will confirm episodes the benchmark fails."""
    assert DEST_TOLERANCE_M <= 0.03


def test_destination_tolerance_is_independent_of_same_place():
    """They answer different questions.

    SAME_PLACE_M asks "did the object move at all", where being generous is
    correct. The destination check asks "did it arrive", where it is not.
    Sharing one constant is what made the check 10 cm wide.
    """
    assert DEST_TOLERANCE_M < SAME_PLACE_M


def test_no_destination_falls_back_to_the_aim_point():
    """place_at has no named destination; behaviour there is unchanged."""
    poses = {"cube": np.array([0.30, 0.10, 0.92])}
    pc = _run(
        _checker(poses),
        {"object": "cube"},
        {"ok": True, "placed_at": [0.30, 0.10, 0.95]},
        {"label": "cube", "pose": [0.0, 0.0, 0.9]},
    )
    assert "dest_err_m" not in pc.measured
    assert pc.status != REFUTED, pc.evidence


def test_unresolvable_destination_does_not_break_the_check():
    """A destination the pose channel cannot see must degrade, not crash."""
    poses = {"bowl": np.array([0.024, -0.263, 0.918])}
    pc = _run(
        _checker(poses),
        {"object": "bowl", "destination": "ghost"},
        {"ok": True, "placed_at": [0.020, -0.260, 0.945]},
        {"label": "bowl", "pose": [0.4, 0.1, 0.9]},
    )
    assert "dest_err_m" not in pc.measured
    assert pc.status in {"confirmed", "unverified", REFUTED}


def test_marked_square_is_not_resolved_as_the_green_cube_itself():
    calls = []

    def pose(name):
        calls.append(name)
        return np.array([.145, -.275, .03])

    pc = _run(PostconditionChecker(object_pose=pose),
        {"object": "green cube", "destination": "green square"},
        {"ok": True, "picked": "green cube", "placed_at": [.14, -.27, .1],
         "destination": "green square", "destination_kind": "configured_point"},
        {"label": "green cube", "pose": [.18, -.03, .03], "channel": "physics"})
    assert "green square" not in calls
    assert "dest_err_m" not in pc.measured
    assert pc.measured["target_err_m"] > 0
    assert pc.measured["destination"] == "green square"
    assert "green square center (physics)" in pc.evidence
    assert pc.status == UNVERIFIED


def test_missed_marked_square_still_fails_the_physical_target_check():
    pc = _run(_checker({"green cube": np.array([.3, -.27, .03])}),
        {"object": "green cube", "destination": "green square"},
        {"ok": True, "picked": "green cube", "placed_at": [.14, -.27, .1],
         "destination": "green square", "destination_kind": "configured_point"},
        {"label": "green cube", "pose": [.18, -.03, .03], "channel": "physics"})
    assert pc.status == REFUTED


def test_configured_center_cannot_confirm_containment_outside_the_green_square():
    pc = _run(_checker({"orange": np.array([.23, -.27, .03])}),
        {"object": "orange", "destination": "green square"},
        {"ok": True, "picked": "orange", "placed_at": [.14, -.27, .1],
         "destination": "green square", "destination_kind": "configured_point"},
        {"label": "orange", "pose": [.3, .1, .03], "channel": "physics"})
    assert pc.status == UNVERIFIED
    assert pc.measured["target_err_m"] == .09
    assert "containment and release are not established" in pc.evidence
    result = annotate_result({"ok": True}, pc)
    assert result["ok"] is True
    assert result["verified"] is False


def test_open_box_center_alone_cannot_confirm_a_release_inside_the_box():
    pc = _run(_checker({"orange": np.array([.30, -.14, .13])}),
        {"object": "orange", "destination": "open box"},
        {"ok": True, "picked": "orange", "placed_at": [.30, -.14, .104],
         "destination": "open box", "destination_kind": "configured_point"},
        {"label": "orange", "pose": [.2, .1, .03], "channel": "physics"})
    assert pc.status == UNVERIFIED
    assert pc.measured["target_err_m"] == 0
    assert "open box center (physics)" in pc.evidence
