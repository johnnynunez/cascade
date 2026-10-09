"""The hand-eye collection loop: every pose vetted, every motion via SafeArm.

Ported from WRC tests/test_calib_session.py (the `_collect_samples` state
machine) onto cascade's safety path. What changed and is pinned here:

* WRC called `safe_arm.move_joints` after a home-grown joint-limit check;
  here every preset is vetted with `harness.vet_pose` when the sweep is
  planned AND again immediately before the arm moves, and the motion itself
  is `SafeArm.move_planned` (route vetting + per-waypoint `approve()`).
* Speed is a FRACTION of the harness's own `max_joint_vel` (default 0.5);
  asking for more than the cap is refused, never clamped upward.
* A sample's FK comes from the MEASURED joints after settling, not the
  commanded ones (settling error would fold straight into the extrinsic).
* The perception watchdog is fed only by frames that actually arrived.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.calibration.dataset import MarkerSpec
from cascade.calibration.handeye import EYE_TO_HAND
from cascade.calibration.session import CollectionSession, SessionConfig, order_by_joint_distance
from cascade.control.kinematics import IKResult
from cascade.types import Frame, RobotState, SafetyViolation, pose_to_transform


class Log(list):
    def add(self, *ev):
        self.append(ev)


class FakeHarness:
    def __init__(self, log, veto=()):
        self.log = log
        self.veto = {tuple(np.round(v, 6)) for v in veto}
        self.limits = SimpleNamespace(max_joint_vel=0.8, joint_margin=0.02)
        self.estopped = False
        self.heartbeats = 0

    def vet_pose(self, q):
        key = tuple(np.round(np.asarray(q, dtype=float), 6))
        self.log.add("vet", key)
        return "TCP outside workspace" if key in self.veto else None

    def heartbeat(self):
        self.heartbeats += 1
        self.log.add("heartbeat")


class FakeSafeArm:
    def __init__(self, log, harness, q0, *, settle_error=0.0, refuse=()):
        self.log, self.harness = log, harness
        self.q = np.asarray(q0, dtype=float)
        self.n_joints = 6
        self.settle_error = settle_error
        self.refuse = {tuple(np.round(v, 6)) for v in refuse}
        self.moves = []

    def get_state(self):
        return RobotState(q=self.q.copy(), dq=np.zeros(6), tau=np.zeros(6))

    def move_planned(self, q, duration_s=3.0):
        key = tuple(np.round(np.asarray(q, dtype=float), 6))
        self.log.add("move", key, float(duration_s))
        if key in self.refuse:
            raise SafetyViolation("joint 2 waypoint outside margin")
        self.moves.append((np.asarray(q, dtype=float).copy(), float(duration_s)))
        self.q = np.asarray(q, dtype=float) + self.settle_error
        return True


class FakeKin:
    """IK/FK on the pose vector itself: q == (x, y, z, r, p, y)."""

    def __init__(self, unreachable=()):
        self.unreachable = {tuple(p) for p in unreachable}
        self.fk_calls = []

    def ik(self, T, q_init, **kw):
        pose = self._pose_of(T)
        ok = tuple(np.round(pose, 6)) not in self.unreachable
        return IKResult(q=pose, success=ok, error=0.0 if ok else 0.03, iterations=3)

    def fk(self, q):
        self.fk_calls.append(np.asarray(q, dtype=float).copy())
        return pose_to_transform(list(np.asarray(q, dtype=float)))

    @staticmethod
    def _pose_of(T):
        R = T[:3, :3]
        # Inverse of the URDF rpy convention used by pose_to_transform.
        pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.round(np.array([*T[:3, 3], roll, pitch, yaw]), 6)


class FakeCamera:
    def __init__(self, log, fail=()):
        self.log, self.fail, self.n = log, set(fail), 0

    def get_frame(self):
        i, self.n = self.n, self.n + 1
        if i in self.fail:
            from cascade.perception.camera_base import CameraError
            self.log.add("frame_error")
            raise CameraError("usb hiccup")
        self.log.add("frame")
        return Frame(rgb=np.zeros((4, 4, 3), np.uint8), depth_m=None, K=np.eye(3))


class FakeAruco:
    """Detects the marker in every frame unless told otherwise."""

    def __init__(self, pattern=None):
        self.pattern = pattern  # callable(call_index) -> bool (seen?)
        self.n = 0

    def detect_frame(self, frame, size_m, *, D=None, target_id=None):
        i, self.n = self.n, self.n + 1
        if self.pattern is not None and not self.pattern(i):
            return None
        T = np.eye(4)
        T[:3, 3] = [0.01, 0.02, 0.6]
        return SimpleNamespace(marker_id=0, T_marker2cam=T, reprojection_px=0.4,
                               corners_px=np.zeros((4, 2)), ambiguity=0.1)


POSES = [
    [0.30, -0.05, 0.28, 0.0, 0.0, -0.5],
    [0.30, -0.05, 0.28, 0.0, 0.2, 0.0],
    [0.25, 0.00, 0.25, 0.3, 0.2, 0.5],
    [0.35, -0.05, 0.30, -0.3, 0.0, 0.3],
]
HOME = np.array([0.30, 0.0, 0.30, 0.0, 0.0, 0.0])


def _session(tmp_path, *, veto=(), unreachable=(), refuse=(), aruco=None, settle_error=0.0,
             fail_frames=(), **cfg):
    log = Log()
    h = FakeHarness(log, veto=veto)
    arm = FakeSafeArm(log, h, HOME, settle_error=settle_error, refuse=refuse)
    kin = FakeKin(unreachable)
    sc = SessionConfig(mode=EYE_TO_HAND, marker=MarkerSpec(), settle_s=0.0,
                       marker_timeout_s=0.05, stable_frames=2, **cfg)
    s = CollectionSession(safe_arm=arm, kin=kin, camera=FakeCamera(log, fail_frames),
                          aruco=aruco or FakeAruco(), config=sc, home_q=HOME,
                          trace_path=tmp_path / "trace.jsonl", log=lambda *_: None,
                          sleep=lambda _s: None)
    return s, log, arm, h, kin


def _events(tmp_path):
    return [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]


def test_auto_collects_one_sample_per_reachable_pose(tmp_path):
    s, log, arm, h, kin = _session(tmp_path)
    samples = s.run_auto(POSES)
    assert len(samples) == 4 and len(arm.moves) == 4
    ev = _events(tmp_path)
    assert ev[0]["event"] == "session_start" and ev[0]["mode"] == "eye_to_hand"
    assert ev[-1]["event"] == "session_end" and ev[-1]["n_collected"] == 4
    assert sum(e["event"] == "sample_recorded" for e in ev) == 4


def test_every_motion_is_vetted_immediately_before_it_is_commanded(tmp_path):
    vetoed = POSES[1]
    s, log, arm, h, kin = _session(tmp_path, veto=[vetoed])
    s.run_auto(POSES)
    moved = [e[1] for e in log if e[0] == "move"]
    assert tuple(np.round(vetoed, 6)) not in moved
    for i, e in enumerate(log):
        if e[0] == "move":
            assert log[i - 1] == ("vet", e[1]), f"move {e[1]} not preceded by its own vet"
    skips = [e for e in _events(tmp_path) if e["event"] == "pose_skipped"]
    assert [x["reason"] for x in skips] == ["vetoed"]
    assert "workspace" in skips[0]["detail"]


def test_harness_refusal_mid_route_skips_the_pose_not_the_session(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, refuse=[POSES[2]])
    samples = s.run_auto(POSES)
    assert len(samples) == 3
    skips = [e for e in _events(tmp_path) if e["event"] == "pose_skipped"]
    assert skips[0]["reason"] == "refused" and "margin" in skips[0]["detail"]


def test_unreachable_pose_is_skipped(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, unreachable=[tuple(POSES[0])])
    assert len(s.run_auto(POSES)) == 3
    skips = [e for e in _events(tmp_path) if e["event"] == "pose_skipped"]
    assert skips[0]["reason"] == "ik_failed"


def test_motion_speed_is_a_fraction_of_the_harness_cap(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, speed_frac=0.5, min_move_s=1.0)
    s.run_auto(POSES)
    q_prev = HOME
    for q, dur in arm.moves:
        dq = float(np.max(np.abs(q - q_prev)))
        # min-jerk peak velocity is 1.875 dq / T
        assert 1.875 * dq / dur <= 0.5 * h.limits.max_joint_vel + 1e-9
        assert dur >= 1.0
        q_prev = q


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.01, 2.0])
def test_speed_above_the_cap_or_nonpositive_is_refused(frac):
    with pytest.raises(ValueError, match="speed"):
        SessionConfig(mode=EYE_TO_HAND, marker=MarkerSpec(), speed_frac=frac)


def test_large_joint_jumps_are_skipped(tmp_path):
    far = [0.30, -0.05, 0.28, 0.0, 0.0, 2.9]
    s, log, arm, h, kin = _session(tmp_path, max_joint_step_rad=1.0)
    samples = s.run_auto([POSES[0], far])
    assert len(samples) == 1
    skips = [e for e in _events(tmp_path) if e["event"] == "pose_skipped"]
    assert skips[0]["reason"] == "joint_jump"


def test_samples_use_measured_joints_not_commanded(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, settle_error=0.004)
    samples = s.run_auto(POSES[:1])
    commanded = arm.moves[0][0]
    assert np.allclose(samples[0].q, commanded + 0.004)
    assert np.allclose(samples[0].T_gripper2base, kin.fk(commanded + 0.004))


def test_no_marker_skips_with_a_trace_event(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, aruco=FakeAruco(lambda i: False))
    assert s.run_auto(POSES[:2]) == []
    skips = [e for e in _events(tmp_path) if e["event"] == "sample_skipped"]
    assert len(skips) == 2 and all(e["reason"] == "no_marker" for e in skips)


def test_stability_requires_consecutive_detections(tmp_path):
    """WRC's alternating-detection fake: the counter must reset on a miss."""
    s, log, arm, h, kin = _session(tmp_path, aruco=FakeAruco(lambda i: i % 2 == 0))
    assert s.run_auto(POSES[:1]) == []


def test_watchdog_is_fed_only_by_frames_that_arrived(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, fail_frames={0, 1, 2})
    s.run_auto(POSES[:1])
    n_frames = sum(1 for e in log if e[0] == "frame")
    assert h.heartbeats == n_frames > 0


def test_poses_are_visited_in_greedy_joint_order(tmp_path):
    qs = [np.array([0, 0, 0, 0, 0, 3.0]), np.array([0, 0, 0, 0, 0, 1.0]),
          np.array([0, 0, 0, 0, 0, 2.0])]
    assert order_by_joint_distance(np.zeros(6), qs) == [1, 2, 0]


def test_manual_mode_is_enter_per_pose(tmp_path):
    answers = iter(["", "s", "", "q"])
    s, log, arm, h, kin = _session(tmp_path)
    samples = s.run_manual(POSES, prompt=lambda _msg: next(answers))
    assert len(samples) == 2 and len(arm.moves) == 2
    ev = _events(tmp_path)
    assert ev[-1]["finish_reason"] == "operator finished"
    assert any(e["event"] == "pose_skipped" and e["reason"] == "operator_skip" for e in ev)


def test_manual_mode_eof_finishes_cleanly(tmp_path):
    def eof(_msg):
        raise EOFError
    s, log, arm, h, kin = _session(tmp_path)
    assert s.run_manual(POSES, prompt=eof) == []
    assert arm.moves == []


def test_go_home_is_vetted_and_speed_limited(tmp_path):
    s, log, arm, h, kin = _session(tmp_path, speed_frac=0.25, min_move_s=1.0)
    arm.q = HOME + 0.8
    s.go_home()
    (q, dur), = arm.moves
    assert np.allclose(q, HOME)
    assert log[-2] == ("vet", tuple(np.round(HOME, 6)))
    assert 1.875 * 0.8 / dur <= 0.25 * h.limits.max_joint_vel + 1e-9
    assert any(e["event"] == "home" for e in _events(tmp_path))


def test_start_home_moves_home_inside_the_session(tmp_path):
    s, log, arm, h, kin = _session(tmp_path)
    arm.q = HOME + 0.3
    s.run_auto(POSES[:2], start_home=True)
    assert np.allclose(arm.moves[0][0], HOME) and len(arm.moves) == 3
    ev = [e["event"] for e in _events(tmp_path)]
    assert ev[:2] == ["session_start", "home"] and ev[-1] == "session_end"


def test_a_vetoed_home_ends_the_session_with_a_trace(tmp_path):
    from cascade.types import SkillError

    s, log, arm, h, kin = _session(tmp_path, veto=[HOME])
    with pytest.raises(SkillError):
        s.run_auto(POSES, start_home=True)
    assert arm.moves == []
    end = _events(tmp_path)[-1]
    assert end["event"] == "session_end" and "SkillError" in end["finish_reason"]


def test_go_home_refuses_a_vetoed_home(tmp_path):
    from cascade.types import SkillError

    s, log, arm, h, kin = _session(tmp_path, veto=[HOME])
    with pytest.raises(SkillError, match="home"):
        s.go_home()
    assert arm.moves == []
