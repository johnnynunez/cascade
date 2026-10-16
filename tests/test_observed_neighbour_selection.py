"""Grasp choice next to an observed neighbour, and the next candidate after a
pre-contact refusal (Isaac kitchen early E2E, attempt C3).

C3: the chosen grasp closed toward a lemon ~8 cm away. Its PLANNED finger
volumes cleared the lemon's observed surface by millimetres, so the existing
pre-vet admitted it; the arm then settled a few millimetres off the plan
(Isaac settle tolerance 0.02 rad per joint) and the live observed-finger gate
refused "measured finger intersects observed surface"; the pre-close retreat
was refused too and no other candidate was tried.

These tests run the REAL observed-finger gate (ObservedScene +
ObservedFingerGate + the PhysX finger geometry) on a depth image ray-cast
from a top-down camera: table, target cube, neighbouring sphere. Only the
transport (an Isaac bridge double) and a 4-joint kinematics double
(x, y, z, yaw; tool pointing down) are fakes. A settle offset in the bridge
reproduces C3's tracking error. The gate and every margin stay unchanged.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.grasping import observed_scene
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.grasping.observed_scene import (OPEN_TRACKING_ENVELOPE_M, PHYSX_GEOMETRY, FingerGeometry,
                                             ObservedFingerGate, ObservedScene)
from cascade.safety.harness import SafeArm, SafetyHarness
from cascade.skills.runtime import OBSERVED_RANK_LATERAL_M, SkillRuntime
from cascade.types import Detection, Frame, Grasp, MotionHalted, ObjectFix, SafetyViolation, SkillError

GEOMETRY = FingerGeometry(PHYSX_GEOMETRY, expected_engine="physx")
TARGET = np.array([.30, 0., .03])      # centre of a 4 x 4 x 6 cm box on the table (z = 0)
HALF = .02
NEIGHBOUR_R = .025
# Sphere centre offset along +x (the closing axis of candidate A). Its
# observed surface clears A's PLANNED open finger by ~3 mm (asserted below).
NEIGHBOUR_DX = .0905
HOME = np.array([.30, 0., .25, 0.])
SAG = np.array([.005, 0., 0., 0.])     # settle offset toward the neighbour near the bottom
SAG_BELOW_Z = .06
SOURCE = ("test", 0)


def yaw_of(R):
    return float(np.arctan2(R[1, 1], R[0, 1]))


class Kin:
    """q = (x, y, z, yaw) of a tool pointing down (reBot `down_open` frame)."""

    ee_frame = "gripper_end"

    def fk(self, q):
        T = np.eye(4)
        T[:3, :3] = _yaw_rotation(float(q[3]))
        T[:3, 3] = q[:3]
        return T

    def ik(self, pose, seed):
        q = np.r_[pose[:3, 3], yaw_of(pose[:3, :3])]
        ok = bool(np.allclose(self.fk(q)[:3, :3], pose[:3, :3], rtol=0, atol=1e-9))
        return SimpleNamespace(q=q, success=ok, error=0. if ok else 1.)


def render(neighbour=True, size=240, height=1., fov_m=.4, half=(HALF, HALF), neighbour_dx=None):
    """Top-down pinhole depth of table + box + optional sphere (base frame)."""
    f = size / fov_m
    K = np.array([[f, 0., size / 2], [0., f, size / 2], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1., -1., -1.])       # camera looks straight down
    T[:3, 3] = [TARGET[0], TARGET[1], height]
    rows, cols = np.mgrid[0:size, 0:size]
    rays_cam = np.stack([(cols - K[0, 2]) / f, (rows - K[1, 2]) / f, np.ones_like(cols, float)], -1)
    d = rays_cam @ T[:3, :3].T                # base-frame direction, camera-z component 1
    o = T[:3, 3]
    t_table = np.full(d.shape[:2], o[2] / -d[..., 2])
    lo = np.array([TARGET[0] - half[0], TARGET[1] - half[1], 0.])
    hi = np.array([TARGET[0] + half[0], TARGET[1] + half[1], 2 * TARGET[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - o) / d, (hi - o) / d
    tnear = np.nanmax(np.minimum(t0, t1), -1)
    tfar = np.nanmin(np.maximum(t0, t1), -1)
    t_cube = np.where((tnear <= tfar) & (tnear > 0), tnear, np.inf)
    t_sphere = np.full(d.shape[:2], np.inf)
    if neighbour:
        c = TARGET + [NEIGHBOUR_DX if neighbour_dx is None else neighbour_dx, 0., 0.]
        oc = o - c
        a = np.sum(d * d, -1)
        b = 2 * d @ oc
        disc = b * b - 4 * a * (oc @ oc - NEIGHBOUR_R ** 2)
        hit = disc >= 0
        t_sphere[hit] = ((-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit]))
    depth = np.minimum(np.minimum(t_table, t_cube), t_sphere).astype(np.float32)
    target = (t_cube <= t_table) & (t_cube <= t_sphere)
    capture = dict(backend="isaac", source=SOURCE, camera="cam", t=15., proprioception=dict(
        version=1, backend="isaac", robot_id="/robot", joint_convention="asset",
        producer_epoch="test", t=15., time_source="physics_loop_monotonic"))
    frame = Frame(np.zeros((size, size, 3), np.uint8), depth, K, frame_id=42, depth_source="sensor",
                  T_base_cam=T, capture=capture, robot_mask=np.zeros((size, size), bool))
    rays = np.c_[cols[target], rows[target], np.ones(target.sum())] @ np.linalg.inv(K).T
    points = (rays * depth[target][:, None]) @ T[:3, :3].T + T[:3, 3]
    ys, xs = np.nonzero(target)
    det = Detection("cube", .95, np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32), mask=target)
    fix = ObjectFix("cube", TARGET.copy(), points, det, np.array([.06, 2 * half[0], 2 * half[1]]), np.eye(3))
    return frame, fix


def candidate(yaw, quality, label, width=.04):
    return Grasp(TARGET.copy(), _yaw_rotation(yaw), width, np.array([0., 0., -1.]), quality, label)


A = (0., .95, "A: closes along +x, toward the neighbour")
B = (np.pi / 2, .90, "B: closes along y")


def build(monkeypatch, *, neighbour=True, sag=None, grasps=(A, B), half=(HALF, HALF), grasp_cfg=None,
          neighbour_dx=None):
    clock = [0.]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    calls = []
    upper = [float(v) for v in GEOMETRY.upper]

    class Bridge:
        _addr = SOURCE
        step = 0
        q = HOME.copy()
        gripper_pos = 1.

        def state(self, *, timeout_s=None):
            self.step += 1
            clock[0] += .02
            calls.append(("read", self.q.tolist(), self.gripper_pos))
            stroke = [u * self.gripper_pos for u in upper]
            return {"q": self.q.copy(), "dq": np.zeros(4), "gripper_pos": self.gripper_pos, "t": clock[0],
                    "physics_clock": {"version": 1, "engine": "physx", "clock": "SimulationManager",
                                      "robot_id": "/robot", "epoch": "test", "sim_time": self.step * .02,
                                      "physics_step": self.step, "physics_dt_s": .02},
                    "gripper_joints": {"version": 1, "names": ["joint_left", "joint_right"],
                                       "position_m": stroke, "lower_m": [0., 0.], "upper_m": upper}}

        def set_joints(self, q, *, timeout_s=None):
            calls.append(("joints", np.asarray(q).tolist()))
            self.q = np.asarray(q, float).copy()
            if sag is not None and self.q[2] < SAG_BELOW_Z:
                self.q = self.q + sag   # the arm settles off the plan, inside settle_tol

        def gripper(self, pos, effort):
            calls.append(("gripper", pos, effort))
            self.gripper_pos = max(.35, pos)   # deterministic object contact double

    raw = IsaacArm(Cfg({"n_joints": 4, "bridge_robot_id": "/robot"}))
    raw._client = Bridge()
    harness = SimpleNamespace(
        limits=SimpleNamespace(workspace_min=[0, -.3, 0], table_z=0., max_joint_vel=10.),
        estopped=False, vet_pose=lambda *a, **k: None, vet_step=lambda *a, **k: None,
        _halt_generation=0, check_stream_start=lambda **k: None, check_contact_episode=lambda **k: None,
        begin_motion=lambda **k: None, end_motion=lambda: None, approve=lambda *a, **k: None,
        allow_grasp_descent=lambda *a, **k: None, clear_grasp_exemption=lambda: None)
    harness._check_halt_generation = SafetyHarness._check_halt_generation.__get__(harness)
    harness._pending_release_episode = None
    harness.check_release_episode = SafetyHarness.check_release_episode.__get__(harness)
    harness._pending_model_withdrawal = None
    harness.check_model_withdrawal = SafetyHarness.check_model_withdrawal.__get__(harness)
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.arm, rt.kin = SafeArm(raw, harness), Kin()
    rt.cfg = Cfg({"arm": {"home_q": HOME.tolist(), "gripper": {}},
                  "grasp": {"pregrasp_offset_m": .1, "move_duration_s": .1,
                            "descend_duration_s": .1, "close_feedback_timeout_s": 2.,
                            **(grasp_cfg or {})}})
    rt.held_object = None
    rt._max_width, rt._grip_open, rt._grip_closed = .09, 1., 0.
    rt.memory = SimpleNamespace(add=lambda *a, **k: None)
    rt.beliefs = SimpleNamespace(mark_removed=lambda *a, **k: None)
    rt.grasp_memory = SimpleNamespace(record=lambda *a, **k: None, prior=lambda *a, **k: None)
    planned = [candidate(*g) for g in grasps]
    plans = []

    def plan(*a, **k):
        plans.append(1)
        return copy.deepcopy(planned)
    rt._plan_grasps = plan
    frame, fix = render(neighbour, half=half, neighbour_dx=neighbour_dx)
    gates, refusals = [], []

    def real_gate(runtime, frame_, fix_, state):
        scene = ObservedScene(frame_, fix_.detection.mask, frame_.T_base_cam, source=SOURCE,
                              robot_id="/robot", epoch="test")
        gate = ObservedFingerGate(scene, GEOMETRY, runtime.kin, state, uncertainty_m=OPEN_TRACKING_ENVELOPE_M)
        for name in ("feedback", "require_closing", "require_profile"):
            method = getattr(gate, name)

            def recorded(*a, _method=method, _name=name, **k):
                try:
                    return _method(*a, **k)
                except SafetyViolation as exc:
                    refusals.append((_name, str(exc)))
                    raise
            setattr(gate, name, recorded)
        gates.append(gate)
        return gate

    monkeypatch.setattr(observed_scene, "for_runtime", real_gate)
    return SimpleNamespace(rt=rt, calls=calls, frame=frame, fix=fix, gates=gates, refusals=refusals,
                           bridge=raw._client, harness=harness, plans=plans)


def grasp(env):
    return env.rt.skill_grasp_object("cube", _fix=env.fix, _frame=env.frame)


def joint_targets(env):
    return [np.asarray(c[1]) for c in env.calls if c[0] == "joints"]


def descents(env):
    """Yaw of every candidate that descended to the grasp height, in order."""
    bottom = [q for q in joint_targets(env) if np.isclose(q[2], TARGET[2], rtol=0, atol=1e-9)]
    yaws = []
    for q in bottom:
        if not yaws or not np.isclose(q[3], yaws[-1], rtol=0, atol=1e-9):
            yaws.append(float(q[3]))
    return yaws


def lowest_yaw(env):
    """Yaw of the last candidate that descended to the grasp height."""
    return descents(env)[-1]


def _bare_gate(env):
    scene = ObservedScene(env.frame, env.fix.detection.mask, env.frame.T_base_cam, source=SOURCE,
                          robot_id="/robot", epoch="test")
    state = env.rt.arm.get_state()
    return ObservedFingerGate(scene, GEOMETRY, Kin(), state, uncertainty_m=OPEN_TRACKING_ENVELOPE_M)


def test_scene_reproduces_c3_geometry(monkeypatch):
    """Preconditions, measured with the real gate: candidate A's planned
    fingers clear the neighbour (the exact pre-vet admits it), the C3-sized
    settle offset puts A's measured finger into the neighbour, and B stays
    clear with the same offset."""
    env = build(monkeypatch)
    gate = _bare_gate(env)
    qa = np.r_[TARGET, 0.]
    qb = np.r_[TARGET, np.pi / 2]
    for q in (qa, qb):
        assert gate.pose(q) is None and gate.closing_pose(q) is None
    exact = GEOMETRY.envelopes(GEOMETRY.upper, GEOMETRY.upper)
    hit = gate.scene.conflict(Kin().fk(qa + SAG), exact)
    assert hit is not None and hit["surface"] == "other observed surface"
    assert gate.scene.conflict(Kin().fk(qb + SAG), exact) is None
    # A's planned clearance is a few millimetres: 2 mm more offset is still clear.
    assert gate.scene.conflict(Kin().fk(qa + SAG * .4), exact) is None


def test_grasp_next_to_neighbour_picks_the_candidate_the_live_gate_accepts(monkeypatch):
    """C3 reproduced: top-ranked A clears the neighbour only by millimetres;
    the arm settles 5 mm toward it. The pre-vet ranks A last and B passes
    the live gate: 0 gate refusals, the cube is held."""
    env = build(monkeypatch, sag=SAG)
    result = grasp(env)
    assert env.refusals == []
    assert result["held"] == "cube" and result["grip_verified"]
    assert lowest_yaw(env) == pytest.approx(np.pi / 2)
    selection = result["observed_selection"]
    assert [r["candidate"] for r in selection["ranked_last"]] == [0]
    assert "observed neighbour" in selection["ranked_last"][0]["reason"]
    assert selection["attempts"] == [{"candidate": 1, "outcome": "grasped"}]
    assert selection["succeeded_attempt"] == 1


def test_no_neighbour_keeps_the_top_candidate_and_the_command_count(monkeypatch):
    """Pin: without a neighbour the choice and every command equal main
    (top-ranked A; same count of reads/targets/jaw commands as base)."""
    env = build(monkeypatch, neighbour=False, sag=SAG)
    result = grasp(env)
    assert result["held"] == "cube" and env.refusals == []
    assert lowest_yaw(env) == pytest.approx(0.)
    assert result.get("observed_selection", {}).get("ranked_last", []) == []
    # Measured on main (ad4786f) with this same scene and test double.
    assert len(joint_targets(env)) == 54
    assert [c for c in env.calls if c[0] == "gripper"] == [
        ("gripper", 1., .8), ("gripper", .5, pytest.approx(.49)), ("gripper", pytest.approx(.15), .7)]
    assert len(env.calls) == 271


def _precommand(message, conflict):
    """The gate's pre-command veto (tagged on this branch; a plain veto on main)."""
    make = getattr(observed_scene, "precommand_refusal", None)
    if make is None:
        return SafetyViolation(f"{message}: {conflict}")
    return make(message, conflict)


def _refuse_first_close(monkeypatch, *, retreat="completes"):
    """Inject a live closing veto after the pre-vet (the gate refuses before
    any jaw command). `retreat` decides whether the withdrawal is vetted."""
    original = observed_scene.for_runtime
    closes = []

    def make(*args):
        gate = original(*args)
        require_closing = gate.require_closing
        require_profile = gate.require_profile
        refused_retreats = []

        def close(state, **kw):
            closes.append(state.q.copy())
            if len(closes) == 1:
                raise _precommand("closing fingers intersects observed non-target surface",
                                  {"finger": "joint_left", "surface": "other observed surface",
                                   "injected": True})
            return require_closing(state, **kw)

        def profile(*a, **kw):
            # The withdrawal's own route veto (the profile right after the
            # vetoed close); every later route is the gate's real answer.
            if retreat == "refused" and len(closes) == 1 and not refused_retreats:
                refused_retreats.append(1)
                raise _precommand("finger trajectory intersects observed surface",
                                  {"finger": "joint_left", "injected": True})
            return require_profile(*a, **kw)
        gate.require_closing = close
        gate.require_profile = profile
        return gate

    monkeypatch.setattr(observed_scene, "for_runtime", make)
    return closes


def test_refusal_before_contact_with_vetted_withdrawal_tries_next_candidate(monkeypatch):
    env = build(monkeypatch, neighbour=False)
    closes = _refuse_first_close(monkeypatch)
    result = grasp(env)
    assert result["held"] == "cube" and result["grip_verified"]
    assert len(closes) == 3                         # refused A, then B's two close stages
    assert descents(env) == pytest.approx([0., np.pi / 2])
    selection = result["observed_selection"]
    assert [a["candidate"] for a in selection["attempts"]] == [0, 1]
    assert selection["attempts"][0]["outcome"].startswith("refused before contact: closing fingers")
    assert "withdrew to pregrasp" in selection["attempts"][0]["outcome"]
    assert selection["attempts"][1]["outcome"] == "grasped"
    assert selection["succeeded_attempt"] == 2
    # Withdrawal to A's pregrasp, then home, B's pregrasp and B's descent.
    pre_a = np.r_[TARGET + [0, 0, .1], 0.]
    targets = joint_targets(env)
    first_close = next(i for i, c in enumerate(env.calls) if c[0] == "gripper" and c[1] < 1.)
    joints = [(i, np.asarray(c[1])) for i, c in enumerate(env.calls) if c[0] == "joints"]
    retreat = [i for i, q in joints if np.allclose(q, pre_a)]
    assert retreat and retreat[-1] < first_close   # A withdrew before any jaw closed
    home = [i for i, q in joints if np.allclose(q, HOME) and retreat[-1] < i < first_close]
    pre_b = [i for i, q in joints if np.allclose(q, np.r_[TARGET + [0, 0, .1], np.pi / 2])]
    assert home and pre_b and home[-1] < pre_b[0]  # back home over its vetted route, then B
    assert np.allclose(targets[-1][:3], TARGET + [0, 0, .1])   # lift back to B's pregrasp
    assert len(env.gates) == 1 and env.plans == [1]   # same observed scene, no new plan


def test_refused_withdrawal_stops_as_today_with_no_extra_command(monkeypatch):
    env = build(monkeypatch, neighbour=False)
    closes = _refuse_first_close(monkeypatch, retreat="refused")
    with pytest.raises(SafetyViolation, match="pre-close retreat refused") as err:
        grasp(env)
    assert "closing fingers intersects" in str(err.value)
    assert len(closes) == 1
    # No jaw closed and nothing streamed after the descent: home, pregrasp and
    # descent of A only, exactly as main.
    assert [c for c in env.calls if c[0] == "gripper"] == [("gripper", 1., .8)]
    assert len(joint_targets(env)) == 51   # measured on main (ad4786f)
    assert np.allclose(joint_targets(env)[-1], np.r_[TARGET, 0.])


def test_next_candidate_retries_are_bounded_to_two(monkeypatch):
    yaws = [(k * np.pi / 6, .95 - .01 * k, f"c{k}") for k in range(5)]
    env = build(monkeypatch, neighbour=False, grasps=yaws)
    original = observed_scene.for_runtime
    closes = []

    def make(*args):
        gate = original(*args)

        def close(state, **kw):
            closes.append(state.q.copy())
            raise _precommand("closing fingers occluded by non-target depth",
                              {"finger": "joint_right", "injected": True})
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, "for_runtime", make)
    with pytest.raises(SafetyViolation, match="closing fingers occluded") as err:
        grasp(env)
    assert len(closes) == 3                          # the first candidate + at most 2 more
    assert "3 grasp candidates refused before contact" in str(err.value)
    assert not [c for c in env.calls if c[0] == "gripper" and c[1] < 1.]


def test_measured_finger_refusal_is_not_retried(monkeypatch):
    """A finger that already met an observed surface may have touched it;
    the scene the next candidate would be vetted against is then stale."""
    env = build(monkeypatch, neighbour=False)
    original = observed_scene.for_runtime
    seen, fired = [], []

    def make(*args):
        gate = original(*args)
        feedback = gate.feedback

        def guard(state):
            seen.append(state.q.copy())
            if state.q[2] < SAG_BELOW_Z and not fired:
                fired.append(1)
                raise SafetyViolation("measured finger intersects observed surface: {'injected': True}")
            return feedback(state)
        gate.feedback = guard
        return gate
    monkeypatch.setattr(observed_scene, "for_runtime", make)
    with pytest.raises(SafetyViolation, match="measured finger.*pre-close retreat completed"):
        grasp(env)
    assert not [c for c in env.calls if c[0] == "gripper" and c[1] < 1.]
    assert not [q for q in joint_targets(env) if not np.isclose(q[3], 0.)]   # B never commanded


def test_halt_after_withdrawal_ends_the_skill_without_another_candidate(monkeypatch):
    env = build(monkeypatch, neighbour=False)
    _refuse_first_close(monkeypatch)
    move = env.rt._move_joints
    moves = []

    def moved(*a, **k):
        result = move(*a, **k)
        moves.append(1)
        if len(moves) == 4:                          # home, pregrasp, descent, withdrawal
            env.harness._halt_generation += 1        # operator halt right after the withdrawal
        return result
    env.rt._move_joints = moved
    with pytest.raises(MotionHalted):
        grasp(env)
    assert len(moves) == 4
    assert [c for c in env.calls if c[0] == "gripper"] == [("gripper", 1., .8)]


def test_rank_lateral_validation_fails_closed_before_motion(monkeypatch):
    env = build(monkeypatch)
    env.rt.cfg = Cfg({**env.rt.cfg.as_dict(), "grasp": {**env.rt.cfg.as_dict()["grasp"],
                                                        "observed_rank_lateral_m": float("nan")}})
    with pytest.raises(SkillError, match="observed_rank_lateral_m"):
        grasp(env)
    assert all(c[0] == "read" for c in env.calls)


def test_tight_fit_on_the_target_alone_does_not_reorder(monkeypatch):
    """Pin: only NON-target surfaces rank a candidate last. A closes across
    the 9 cm side with its fingers ~5 mm from the target; it stays first."""
    tight_a = (0., .95, "A: closes across the long side", .09)
    env = build(monkeypatch, neighbour=False, half=(.045, .02), grasps=(tight_a, B))
    result = grasp(env)
    assert result["held"] == "cube" and env.refusals == []
    assert descents(env) == pytest.approx([0.])
    assert result.get("observed_selection", {}).get("ranked_last", []) == []


def test_next_candidate_never_asks_the_planner_for_a_new_batch(monkeypatch):
    """With bounded GraspGen-X replanning configured, the next candidate
    still comes only from the plan already vetted: B fails the vet, nothing
    is left, the original veto is reported and nothing moves after the
    withdrawal."""
    env = build(monkeypatch, neighbour=False, grasp_cfg={
        "backend": "graspgenx", "graspgenx": {"feasibility_max_batches": 3, "feasibility_timeout_s": 8.}})
    _refuse_first_close(monkeypatch)
    vet = env.harness.vet_pose
    env.harness.vet_pose = lambda q, **k: ("blocked yaw" if np.isclose(abs(q[3]), np.pi / 2)
                                          else vet(q, **k))   # B and its jaws-swapped twin
    with pytest.raises(SafetyViolation, match="closing fingers intersects") as err:
        grasp(env)
    assert "pre-close retreat completed" in str(err.value)
    assert "no further candidate (next candidate: no executable grasp" in str(err.value)
    assert env.plans == [1]
    assert descents(env) == pytest.approx([0.])
    assert np.allclose(joint_targets(env)[-1], np.r_[TARGET + [0, 0, .1], 0.])   # ends withdrawn


def _refusal_state(monkeypatch, *, conflict=True, recovery="completed", episode=None,
                   attempts=1, remaining=True, deadline=None, provisional=None):
    from cascade.skills import runtime as module
    rt = SkillRuntime.__new__(SkillRuntime)
    rt._held_provisional = provisional
    rt._task_deadline = deadline
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.)
    selection = module._ObservedSelection()
    planned = [candidate(k * .3, .9, f"c{k}") for k in range(attempts + 1)]
    selection.add_batch(planned)
    for g in planned[:attempts]:
        selection.attempt(g)
    exc = (_precommand("closing fingers occluded by non-target depth", {"injected": True}) if conflict
           else SafetyViolation("measured finger intersects observed surface"))
    grasps = planned if remaining else planned[:attempts]
    return SkillRuntime._observed_next_candidate_refusal(rt, exc, recovery, episode, selection, grasps, None)


@pytest.mark.parametrize("case,expected", [
    ({}, (None, True)),
    ({"conflict": False}, ("not a pre-command observed-finger veto", False)),
    ({"recovery": "refused: route vetoed"}, ("withdrawal not completed", False)),
    ({"recovery": "did not settle"}, ("withdrawal not completed", False)),
    ({"episode": object()}, ("jaw command attempted", False)),
    ({"provisional": ("cube", "cube", None)}, ("jaw command attempted", False)),
    ({"attempts": 3}, ("limit of 2 further candidates reached", True)),
    ({"attempts": 2}, (None, True)),
    ({"deadline": 99.}, ("grasp time budget spent", True)),
    ({"deadline": 101.}, (None, True)),
    ({"remaining": False}, ("no other candidate in this plan", True)),
])
def test_next_candidate_rule_on_its_own(monkeypatch, case, expected):
    """Each condition decides alone (the call sites pre-filter some)."""
    assert _refusal_state(monkeypatch, **case) == expected


def test_horizontal_dilation_contains_every_horizontal_shift_and_never_grows_down():
    from cascade.grasping.observed_scene import _horizontal_dilation
    rng = np.random.default_rng(4)
    exact = GEOMETRY.envelopes(GEOMETRY.upper, GEOMETRY.upper)
    lateral = .01
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    shifts = np.concatenate([np.zeros((1, 3))] + [
        np.c_[r * np.cos(angles), r * np.sin(angles), np.zeros(16)] for r in (.0025, .005, .0075, .01)])

    def inside(parts, p):
        return any(np.all(parts_planes[:, :3] @ p + parts_planes[:, 3] <= 1e-12)
                   for _, _, parts_planes, *_ in parts)
    for yaw, tilt in ((0., 0.), (.7, 0.), (2.1, .2)):
        R = _yaw_rotation(yaw)
        if tilt:
            c, s_ = np.cos(tilt), np.sin(tilt)
            R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]) @ R
        dilated = _horizontal_dilation(exact, R, lateral)
        lo = np.min([e[3] for e in exact], axis=0)
        hi = np.max([e[4] for e in exact], axis=0)
        for _ in range(300):
            p_tcp = rng.uniform(lo - .012, hi + .012)
            p_base = R @ p_tcp
            hit = any(inside(exact, R.T @ (p_base - s)) for s in shifts)
            if hit:   # conservative: every horizontally displaced finger point is inside
                assert inside(dilated, p_tcp)
        if not tilt:
            # A vertical tool: nothing below the finger tip (base z) is ever added.
            bottom = min(float((R @ v)[2]) for e in exact for v in (e[3], e[4]))
            for _ in range(200):
                p_base = R @ rng.uniform(lo - .012, hi + .012)
                p_base[2] = bottom - rng.uniform(1e-4, .01)
                assert not inside(dilated, R.T @ p_base)


def test_only_vetoes_issued_before_their_command_are_tagged(monkeypatch):
    """The next candidate relies on this tag: the real gate's planned-profile
    and closing vetoes carry it (nothing of that step moved), and so does the
    runtime's planned-curve veto; a MEASURED-feedback refusal does not."""
    env = build(monkeypatch)
    moves = []
    move = env.rt._move_joints

    def spy(*a, **k):
        moves.append(k)
        return move(*a, **k)
    env.rt._move_joints = spy
    assert grasp(env)["held"] == "cube"
    gate = env.gates[0]
    inside = np.r_[TARGET, 0.] + 2 * SAG          # A's fingers inside the neighbour
    env.bridge.gripper_pos = 1.                   # open jaws, as during an approach
    state = env.rt.arm.get_state()                # a fresh physics step
    state.q = inside
    vetoes = []
    for call in (lambda: gate.require_closing(state),
                 lambda: gate.require_profile(inside, inside, .1, 30.),
                 lambda: moves[-1]["_trajectory_preflight"](
                     inside, [SimpleNamespace(checks=((inside, inside, .02),))], lambda: None)):
        with pytest.raises(SafetyViolation, match="intersects observed") as err:
            call()
        vetoes.append(observed_scene.precommand_conflict(err.value))
    assert all(v is not None and v["surface"] == "other observed surface" for v in vetoes)
    state = env.rt.arm.get_state()                # the next physics step
    state.q = inside
    with pytest.raises(SafetyViolation, match="measured finger intersects observed surface") as err:
        gate.feedback(state)
    assert observed_scene.precommand_conflict(err.value) is None


def test_a_candidate_ranked_last_is_still_tried_when_nothing_else_fits(monkeypatch):
    """Ranking never refuses: with only A in the plan (beside the neighbour,
    the arm on plan), A is executed exactly as on main."""
    env = build(monkeypatch, grasps=(A,))
    result = grasp(env)
    assert result["held"] == "cube" and env.refusals == []
    assert descents(env) == pytest.approx([0.])
    selection = result["observed_selection"]
    assert [r["candidate"] for r in selection["ranked_last"]] == [0]
    assert selection["attempts"] == [{"candidate": 0, "outcome": "grasped"}]


def test_ranking_checks_both_jaw_orientations_the_selector_may_execute():
    """The selector may execute a candidate's jaws-swapped twin (yaw + pi
    about the approach). A candidate whose TWIN alone comes near a neighbour
    is ranked last too (a gate double: only the twin's rotation conflicts)."""
    from cascade.grasping.observed_scene import neighbour_ranked
    from cascade.grasping.selector import _flip_twin
    near, far = candidate(0., .9, "near twin"), candidate(np.pi / 4, .8, "far")
    twin = _flip_twin(near).rotation
    seen = []

    def proximity(T, lateral_m, *, closing=False):
        seen.append(T[:3, :3].copy())
        return {"surface": "other observed surface"} if np.allclose(T[:3, :3], twin, atol=1e-9) else None
    gate = SimpleNamespace(neighbour_proximity=proximity)
    ranked = []
    order = list(neighbour_ranked([near, far], gate, lateral_m=.01, pregrasp_offset_m=.1,
                                  ranked_last=lambda i, g, why: ranked.append((i, why))))
    assert [g.label for g in order] == ["far", "near twin"]
    assert ranked and ranked[0][0] == 0 and ranked[0][1].startswith("pregrasp fingers within 10 mm")
    assert any(np.allclose(R, near.rotation) for R in seen)


def test_float32_learned_rotations_are_ranked_not_refused(monkeypatch):
    """GraspGen-X rotations arrive in float32 (rigid to ~1e-7, the scene's
    conflict test wants 1e-9). Ranking projects them for its own check only:
    A still goes last beside the neighbour, and both candidates are yielded
    exactly as planned (the selector and IK see them unchanged)."""
    from cascade.grasping.observed_scene import neighbour_ranked
    env = build(monkeypatch)
    assert grasp(env)["held"] == "cube"
    c, s_ = np.cos(1e-3), np.sin(1e-3)
    about_approach = np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])  # 1 mrad about base z
    planned = [candidate(*g) for g in (A, B)]
    for g in planned:
        g.rotation = (about_approach @ g.rotation).astype(np.float32)
        R = g.rotation.astype(float)
        assert not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=0)        # not rigid to 1e-9
    order = list(neighbour_ranked(planned, env.gates[0], lateral_m=.01, pregrasp_offset_m=.1))
    assert [g.label for g in order] == [B[2], A[2]]
    assert all(g.rotation.dtype == np.float32 for g in order)


def test_an_attempted_candidate_is_never_retried_as_its_jaws_swapped_twin():
    from cascade.skills.runtime import _ObservedSelection
    from cascade.grasping.selector import _flip_twin
    a, b = candidate(0., .9, "A"), candidate(np.pi / 2, .8, "B")
    selection = _ObservedSelection()
    selection.add_batch([a, b])
    selection.attempt(_flip_twin(a))     # the selector executed A's twin
    assert selection.attempts == [{"candidate": 0, "outcome": "started"}]
    assert selection.attempted(a) and not selection.attempted(b)


def test_the_choice_record_names_candidates_the_vet_dropped(monkeypatch):
    env = build(monkeypatch, neighbour=False)
    vet = env.harness.vet_pose
    env.harness.vet_pose = lambda q, **k: ("blocked yaw" if np.isclose(q[3], 0.) or np.isclose(abs(q[3]), np.pi)
                                          else vet(q, **k))   # A and its jaws-swapped twin
    result = grasp(env)
    assert result["held"] == "cube" and descents(env) == pytest.approx([np.pi / 2])
    selection = result["observed_selection"]
    assert [d["candidate"] for d in selection["dropped"]] == [0] and selection["dropped_count"] == 1
    assert "blocked yaw" in selection["dropped"][0]["reason"]
    assert selection["attempts"] == [{"candidate": 1, "outcome": "grasped"}]


@pytest.mark.parametrize("lateral", [OBSERVED_RANK_LATERAL_M, 0.])
def test_planned_finger_inside_the_neighbour_never_reaches_the_live_gate(monkeypatch, lateral):
    """The neighbour 2 cm closer: A's PLANNED fingers enter it. With ranking
    A goes last and is never vetted (B is executable); with ranking off the
    static pre-vet (the same closing envelope as the live gate, no motion)
    drops A and records why. Either way B runs and the gate never refuses."""
    env = build(monkeypatch, neighbour_dx=NEIGHBOUR_DX - .02, grasp_cfg={"observed_rank_lateral_m": lateral})
    assert _bare_gate(env).closing_pose(np.r_[TARGET, 0.]) is not None   # precondition
    result = grasp(env)
    assert result["held"] == "cube" and env.refusals == []
    assert descents(env) == pytest.approx([np.pi / 2])
    selection = result["observed_selection"]
    if lateral:
        assert [r["candidate"] for r in selection["ranked_last"]] == [0] and selection["dropped"] == []
    else:
        assert selection["ranked_last"] == [] and [d["candidate"] for d in selection["dropped"]] == [0]
        assert "observed non-target surface" in selection["dropped"][0]["reason"]
    assert selection["attempts"] == [{"candidate": 1, "outcome": "grasped"}]
    assert selection["succeeded_attempt"] == 1
