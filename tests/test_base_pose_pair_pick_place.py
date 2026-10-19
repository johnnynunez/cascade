"""Acceptance on the mock so101 pair (base_pose spec §0.8, §7; plan Task 8 Step 3; runs in CI).

`--arms so101_left,so101_right --camera mock_small`, the camera fixed in the
table frame and shifted 10 cm in -y, so the red cube stands at W (0.198, -0.10),
inside so101_right's reach. Two sequences on `arm="so101_right"`:

1. pick_and_place(object="red cube"): the object point at the release (the
   lowest commanded TCP of the place, lifted by the right arm's base_pose, plus
   the held offset rotated into the table frame, R16) lies within 5 mm of the
   right arm's drop zone in the table frame, (0.20, -0.34);
2. grasp_object, then place_at(x=0.22, y=-0.30): the same check against
   (0.22, -0.30).

In both, the cube's belief ends within 5 mm of the target, the verifier never
refutes the place (the mock stack has no independent channel: UNVERIFIED; the
MuJoCo acceptance asserts CONFIRMED by physics), and so101_left sends no
command. On main before Option 2
(#352) both sequences were refused on the right arm; before #352 the place
landed 22 cm off (spec §1).
"""

from __future__ import annotations

import numpy as np
import pytest
from base_pose_oracle import SHIPPED_RIGHT, lift, make_runtime, raw_arm, transform  # noqa: F401
from test_base_pose_runtime_frame import (  # noqa: F401  (gate_off: a fixture by name)
    _belief_xy,
    _fusion_held,
    _lowest,
    _offset_spy,
    _pair_world_rig,
    gate_off,
)

RIGHT_DROP_W = np.array([0.20, -0.34])      # so101_right's drop_zone [0.20, -0.12] (its base frame) in W
PLACE_W = np.array([0.22, -0.30])
TOL_M = 0.005


def _place_spy(rt, monkeypatch, right):
    """The right arm's command count at the start of each place_at."""
    starts, real = [], rt.skill_place_at

    def place_at(*a, **k):
        starts.append(len(raw_arm(right).commands))
        return real(*a, **k)

    monkeypatch.setattr(rt, "skill_place_at", place_at)
    return starts


def _object_point(right, start, offsets, T):
    """(lifted release TCP, object point): the place's lowest commanded TCP by
    the right arm's own Kinematics, lifted by T, plus R . held offset (R16)."""
    commands = raw_arm(right).commands[start:]
    tcp = _lowest(lift(T, np.array([right.harness.kin.fk(np.asarray(q, dtype=float))[:3, 3] for q in commands])))
    offset = offsets[-1] if offsets and offsets[-1] is not None else np.zeros(3)
    return tcp, tcp + np.asarray(T)[:3, :3] @ offset


def _not_refuted(result):
    post = result.get("postcondition") or {}
    return str(post.get("status", "")).lower() in ("confirmed", "unverified")


@pytest.mark.parametrize("sequence", ["pick_and_place", "grasp_object + place_at"])
def test_the_pair_places_at_the_world_target_on_the_right_arm(make_runtime, gate_off, monkeypatch,  # noqa: F811
                                                               sequence):
    T = transform(SHIPPED_RIGHT)
    rt = make_runtime(_pair_world_rig())
    left, right = rt.arm_rig.get("so101_left"), rt.arm_rig.get("so101_right")
    assert np.allclose(right.__dict__["harness"].base_pose, T)
    l0 = len(raw_arm(left).commands)
    if sequence == "pick_and_place":
        target = RIGHT_DROP_W
        offsets, starts = _offset_spy(rt, monkeypatch), _place_spy(rt, monkeypatch, right)
        with _fusion_held(rt):
            result = rt.execute("pick_and_place", {"object": "red cube", "arm": "so101_right"})
            belief = _belief_xy(rt, "red cube")
        reported = result["placed_at"][:2]
    else:
        target = PLACE_W
        grasped = rt.execute("grasp_object", {"label": "red cube", "arm": "so101_right"})
        assert grasped["ok"] is True and grasped["held"] == "red cube", grasped
        offsets, starts = _offset_spy(rt, monkeypatch), _place_spy(rt, monkeypatch, right)
        with _fusion_held(rt):
            result = rt.execute("place_at", {"x": float(PLACE_W[0]), "y": float(PLACE_W[1]), "arm": "so101_right"})
            belief = _belief_xy(rt, "red cube")
        reported = result["at"][:2]
    assert result.get("ok", True) is True, result
    assert _not_refuted(result), result.get("postcondition")
    assert starts, "place_at ran"
    tcp, object_point = _object_point(right, starts[-1], offsets, T)
    assert np.hypot(*(object_point[:2] - target)) <= TOL_M, (object_point, tcp, target)
    assert np.hypot(*(belief - target)) <= TOL_M, (belief, target)
    np.testing.assert_allclose(reported, target, rtol=0.0, atol=1.1e-3)
    assert len(raw_arm(left).commands) == l0, "so101_left never moved"
    assert rt.held_object is None and rt._held_arm is None


def test_the_watcher_keeps_every_arms_watchdog_alive(make_runtime):  # noqa: F811
    """The perception watchdog of EVERY arm on a rig is fed by the watcher's
    fresh frames (the cameras watch the shared table). Before, only the
    primary's was: on the MuJoCo pair so101_right's pick stopped "perception
    watchdog: last observation 7.7s old" between two of its own observes, and
    the shutdown park of the right arm was skipped for the same reason. Pre-
    existing on main for every multi-arm rig, base_pose or not."""
    import time

    rt = make_runtime(_pair_world_rig())
    assert rt.watcher is not None
    harnesses = [s.__dict__["harness"] for s in rt.arm_rig.arms.values()]
    stale = time.monotonic() - 100.0
    for h in harnesses:
        h._last_heartbeat = stale
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not all(h._last_heartbeat > stale for h in harnesses):
        time.sleep(0.05)
    assert [h._last_heartbeat > stale for h in harnesses] == [True, True], "every arm's watchdog was fed"


def test_a_single_arm_watcher_heartbeats_its_own_harness(make_runtime):  # noqa: F811
    """One arm: the watcher keeps heartbeating that harness object itself."""
    from base_pose_oracle import moved_cfg

    rt = make_runtime(moved_cfg("so101_right", None))
    assert rt.watcher._harness is rt.arm.__dict__["harness"]
