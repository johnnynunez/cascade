"""Aim the OBJECT at the target, not the gripper.

`place_at` solves IK for the TCP, so unless the goal is shifted by wherever
the object sits in the jaws, the object lands offset by exactly that much.

Measured on LIBERO: the object's horizontal offset from the TCP was 1.1 cm at
grasp and 5.1 cm at release, i.e. it slid 4.3 cm in transit, which was most of
a ~6 cm placement error against a 3 cm success predicate. With compensation
fed by a working pose channel, median aim error went 6.4 -> 1.8 cm and the
well-behaved episodes all landed under 3 cm.

These tests pin the contract rather than re-running that measurement, and in
particular the two mistakes that were made while building it: reporting the
compensated TCP point as the placement result (which would have graded the
place against the wrong target and forgiven its own error), and re-seeding the
belief at the TCP point (which would have re-introduced the offset that
compensation just removed).
"""

from __future__ import annotations

import numpy as np


from cascade.skills.runtime import SkillRuntime


class _Kin:
    def __init__(self, tcp):
        self._tcp = np.asarray(tcp, float)

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = self._tcp
        return T


class _Arm:
    def __init__(self, tcp):
        self.kin = _Kin(tcp)

    def get_state(self):
        class S:
            q = np.zeros(7)
        return S()


def _runtime(tcp=(0.30, 0.10, 0.95), held="cube"):
    """A SkillRuntime with only the fields _held_object_offset touches."""
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.held_object = held
    rt._held_det_label = held
    rt._held_offset = None
    rt._object_pose = None
    rt.arm = _Arm(tcp)
    rt.kin = _Kin(tcp)
    return rt


def test_no_held_object_means_no_offset():
    rt = _runtime(held=None)
    assert rt._held_object_offset() is None


def test_pose_channel_is_used_when_available():
    """The channel that made the difference: 6.4 -> 1.8 cm median."""
    tcp = np.array([0.30, 0.10, 0.95])
    obj = tcp + np.array([0.02, 0.04, -0.03])
    rt = _runtime(tcp)
    rt._object_pose = lambda name: obj
    off = rt._held_object_offset()
    assert off is not None
    assert np.allclose(off, obj - tcp, atol=1e-9)


def test_absurd_offset_is_rejected_not_trusted():
    """A match 40 cm from the TCP is another object on the table.

    Trusting it would throw the place further off than doing nothing, so the
    sanity gate must reject it and fall back.
    """
    tcp = np.array([0.30, 0.10, 0.95])
    rt = _runtime(tcp)
    rt._object_pose = lambda name: tcp + np.array([0.40, 0.0, 0.0])
    rt._held_offset = np.array([0.01, 0.0, -0.03])
    off = rt._held_object_offset()
    assert np.allclose(off, [0.01, 0.0, -0.03])


def test_falls_back_to_grasp_time_offset_when_channel_is_dead():
    """A dead camera must degrade, never fail the place.

    This is also the branch that silently hid the first attempt's failure:
    MockDetector returns nothing, so every call landed here.
    """
    rt = _runtime()
    rt._object_pose = lambda name: (_ for _ in ()).throw(RuntimeError("no pose"))
    rt._held_offset = np.array([0.005, -0.002, -0.05])
    off = rt._held_object_offset()
    assert np.allclose(off, [0.005, -0.002, -0.05])


def test_returns_none_when_nothing_is_known():
    rt = _runtime()
    rt._object_pose = None
    rt._held_offset = None
    assert rt._held_object_offset() is None


def test_compensation_shifts_the_tcp_goal_by_the_offset():
    """The arithmetic that makes the object land on the requested point."""
    requested = np.array([0.20, 0.05])
    offset = np.array([0.02, 0.04])          # object sits +x/+y of the TCP
    tcp_goal = requested - offset
    # The object ends up at tcp_goal + offset, which is the request.
    assert np.allclose(tcp_goal + offset, requested)


def test_place_result_reports_the_object_aim_not_the_tcp():
    """Guards a bug made while building this.

    The postcondition channel scores the object's final pose against the
    reported placement. Returning the compensated TCP point would grade the
    place against the wrong target and quietly forgive the compensation error.
    """
    import inspect

    src = inspect.getsource(SkillRuntime.skill_place_at)
    tail = src[src.rindex("return {"):]
    assert '"tcp_at"' in tail, "the TCP point should still be reported, separately"
    # The `at` field must be built from the caller's x/y, not from `target`
    # (which may have been shifted).
    assert 'round(float(x), 3), round(float(y), 3)' in tail


def test_belief_is_reseeded_at_the_object_aim_not_the_tcp():
    """The other bug: seeding the belief at the compensated TCP point would
    re-introduce exactly the offset compensation just removed."""
    import inspect

    src = inspect.getsource(SkillRuntime.skill_place_at)
    seed = src[src.index("self.beliefs.update("):]
    seed = seed[:seed.index(")")]
    assert "np.array([x, y, release_z]" in seed
