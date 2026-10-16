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
from scipy.spatial.transform import Rotation

from cascade.config import Cfg
from cascade.control.isaac_arm import IsaacArm
from cascade.grasping import observed_scene
from cascade.grasping.obb_grasp import _yaw_rotation
from cascade.grasping.observed_scene import (OPEN_TRACKING_ENVELOPE_M, PHYSX_GEOMETRY, FingerGeometry,
                                             ObservedFingerGate, ObservedScene)
from cascade.safety.harness import SafeArm, SafetyHarness
from cascade.skills import runtime as runtime_mod
from cascade.skills.runtime import SkillRuntime
from cascade.types import Detection, Frame, Grasp, MotionHalted, ObjectFix, SafetyViolation, SkillError

OBSERVED_RANK_LATERAL_M = getattr(runtime_mod, "OBSERVED_RANK_LATERAL_M", .01)
# Fix round 1 bounds (read through the module so this file still imports on
# the commit before them and every new test fails on its own).
RANK_MAX_CANDIDATES = getattr(runtime_mod, "OBSERVED_RANK_MAX_CANDIDATES", 32)
RANK_TIME_SLICE_S = getattr(runtime_mod, "OBSERVED_RANK_TIME_SLICE_S", .3)

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


class TiltKin:
    """q = (x, y, z, rx, ry, rz): the TCP position and the rotation vector of
    the tool relative to pointing straight down, so learned-style TILTED
    candidates (kitchen: up to 18 degrees from vertical) solve."""

    ee_frame = "gripper_end"
    R0 = _yaw_rotation(0.)

    def fk(self, q):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(np.asarray(q[3:6], float)).as_matrix() @ self.R0
        T[:3, 3] = q[:3]
        return T

    def ik(self, pose, seed):
        q = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3] @ self.R0.T).as_rotvec()]
        ok = bool(np.allclose(self.fk(q)[:3, :3], pose[:3, :3], rtol=0, atol=1e-9))
        return SimpleNamespace(q=q, success=ok, error=0. if ok else 1.)


TILT_HOME = np.r_[HOME[:3], 0., 0., 0.]


def tilted(k, *, tilt_deg, yaw, tilt_axis, z, width=.03, quality=None):
    """Candidate k: tool tilted `tilt_deg` about a horizontal axis, approach = tool +x."""
    axis = np.array([np.cos(tilt_axis), np.sin(tilt_axis), 0.])
    R = Rotation.from_rotvec(axis * np.deg2rad(tilt_deg)).as_matrix() @ _yaw_rotation(yaw)
    return Grasp(np.r_[TARGET[:2], z], R, width, R[:, 0].copy(), .95 - .001 * k if quality is None else quality,
                 f"t{k}")


def tilted_set(n=32, *, z=.010, seed=7):
    """`n` learned-style candidates on a low box: tilts 5-18 deg, any yaw."""
    rng = np.random.default_rng(seed)
    return [tilted(k, tilt_deg=rng.uniform(5., 18.), yaw=rng.uniform(-np.pi, np.pi),
                   tilt_axis=rng.uniform(0., 2 * np.pi), z=z) for k in range(n)]


def render(neighbour=True, size=240, height=1., fov_m=.4, half=(HALF, HALF), neighbour_dx=None,
           box_top=2 * TARGET[2], ring=None):
    """Top-down pinhole depth of table + box + optional sphere (base frame).

    `size` is square or (rows, cols) (720 x 1280 = the kitchen camera, 921k
    observed points). `ring=(r_in, r_out, z)` adds a flat observed annulus
    around the target at height z: a neighbour on every side."""
    rows_n, cols_n = (size, size) if np.isscalar(size) else size
    f = cols_n / fov_m
    K = np.array([[f, 0., cols_n / 2], [0., f, rows_n / 2], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1., -1., -1.])       # camera looks straight down
    T[:3, 3] = [TARGET[0], TARGET[1], height]
    rows, cols = np.mgrid[0:rows_n, 0:cols_n]
    rays_cam = np.stack([(cols - K[0, 2]) / f, (rows - K[1, 2]) / f, np.ones_like(cols, float)], -1)
    d = rays_cam @ T[:3, :3].T                # base-frame direction, camera-z component 1
    o = T[:3, 3]
    t_table = np.full(d.shape[:2], o[2] / -d[..., 2])
    lo = np.array([TARGET[0] - half[0], TARGET[1] - half[1], 0.])
    hi = np.array([TARGET[0] + half[0], TARGET[1] + half[1], box_top])
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
    if ring is not None:
        r_in, r_out, z_ring = ring
        t_flat = (o[2] - z_ring) / -d[..., 2]
        xy = o[:2] + t_flat[..., None] * d[..., :2]
        r = np.linalg.norm(xy - TARGET[:2], axis=-1)
        t_sphere = np.minimum(t_sphere, np.where((r >= r_in) & (r <= r_out), t_flat, np.inf))
    depth = np.minimum(np.minimum(t_table, t_cube), t_sphere).astype(np.float32)
    target = (t_cube <= t_table) & (t_cube <= t_sphere)
    capture = dict(backend="isaac", source=SOURCE, camera="cam", t=15., proprioception=dict(
        version=1, backend="isaac", robot_id="/robot", joint_convention="asset",
        producer_epoch="test", t=15., time_source="physics_loop_monotonic"))
    frame = Frame(np.zeros((rows_n, cols_n, 3), np.uint8), depth, K, frame_id=42, depth_source="sensor",
                  T_base_cam=T, capture=capture, robot_mask=np.zeros((rows_n, cols_n), bool))
    rays = np.c_[cols[target], rows[target], np.ones(target.sum())] @ np.linalg.inv(K).T
    points = (rays * depth[target][:, None]) @ T[:3, :3].T + T[:3, 3]
    ys, xs = np.nonzero(target)
    det = Detection("cube", .95, np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32), mask=target)
    fix = ObjectFix("cube", np.r_[TARGET[:2], box_top / 2], points, det,
                    np.array([box_top, 2 * half[0], 2 * half[1]]), np.eye(3))
    return frame, fix


def candidate(yaw, quality, label, width=.04):
    return Grasp(TARGET.copy(), _yaw_rotation(yaw), width, np.array([0., 0., -1.]), quality, label)


A = (0., .95, "A: closes along +x, toward the neighbour")
B = (np.pi / 2, .90, "B: closes along y")


def build(monkeypatch, *, neighbour=True, sag=None, grasps=(A, B), half=(HALF, HALF), grasp_cfg=None,
          neighbour_dx=None, kin=None, home=HOME, real_time=False, size=240, box_top=2 * TARGET[2], ring=None):
    """`grasps`: (yaw, quality, label[, width]) tuples or Grasp objects.
    `real_time=True` keeps the real clock (timing tests); otherwise time is
    a fake clock that advances only with bridge reads and sleeps."""
    clock = [0.]
    if not real_time:
        monkeypatch.setattr("time.monotonic", lambda: clock[0])
        monkeypatch.setattr("time.sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    calls = []
    upper = [float(v) for v in GEOMETRY.upper]
    n_joints = len(home)

    class Bridge:
        _addr = SOURCE
        step = 0
        q = np.asarray(home, float).copy()
        gripper_pos = 1.

        def state(self, *, timeout_s=None):
            self.step += 1
            clock[0] += .02
            calls.append(("read", self.q.tolist(), self.gripper_pos))
            stroke = [u * self.gripper_pos for u in upper]
            return {"q": self.q.copy(), "dq": np.zeros(n_joints), "gripper_pos": self.gripper_pos, "t": clock[0],
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

    raw = IsaacArm(Cfg({"n_joints": n_joints, "bridge_robot_id": "/robot"}))
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
    rt.arm, rt.kin = SafeArm(raw, harness), (kin or Kin())
    rt.cfg = Cfg({"arm": {"home_q": np.asarray(home, float).tolist(), "gripper": {}},
                  "grasp": {"pregrasp_offset_m": .1, "move_duration_s": .1,
                            "descend_duration_s": .1, "close_feedback_timeout_s": 2.,
                            **(grasp_cfg or {})}})
    rt.held_object = None
    rt._max_width, rt._grip_open, rt._grip_closed = .09, 1., 0.
    rt.memory = SimpleNamespace(add=lambda *a, **k: None)
    rt.beliefs = SimpleNamespace(mark_removed=lambda *a, **k: None)
    rt.grasp_memory = SimpleNamespace(record=lambda *a, **k: None, prior=lambda *a, **k: None)
    planned = [g if isinstance(g, Grasp) else candidate(*g) for g in grasps]
    plans = []

    def plan(*a, **k):
        plans.append(1)
        return copy.deepcopy(planned)
    rt._plan_grasps = plan
    frame, fix = render(neighbour, half=half, neighbour_dx=neighbour_dx, size=size, box_top=box_top, ring=ring)
    gates, refusals, proximity = [], [], []

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
        near = getattr(gate, "neighbour_proximity", None)

        def counted(T, *a, **k):
            proximity.append(np.asarray(T, float).copy())
            return near(T, *a, **k)
        if near is not None:
            gate.neighbour_proximity = counted
        gates.append(gate)
        return gate

    monkeypatch.setattr(observed_scene, "for_runtime", real_gate)
    return SimpleNamespace(rt=rt, calls=calls, frame=frame, fix=fix, gates=gates, refusals=refusals,
                           bridge=raw._client, harness=harness, plans=plans, proximity=proximity, clock=clock)


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
    # The final veto carries the whole choice record and names the first refusal.
    assert "; first refusal: closing fingers occluded by non-target depth" in str(err.value)
    selection = err.value.observed_selection
    assert [a["candidate"] for a in selection["attempts"]] == [0, 1, 2]
    assert all(a["outcome"].startswith("refused before contact") for a in selection["attempts"])
    assert selection["first_refusal"].startswith("closing fingers occluded by non-target depth")


def _count_motions(env):
    """The harness double counts arm motions as SafetyHarness does
    (`motions_begun`: once per motion, at its first approved command), and
    returns the runtime's actuation counts the effects verifier diffs."""
    h = env.harness
    h.motions_begun, h.uncounted = 0, False

    def begin(**k):
        h.uncounted = True

    def end():
        h.uncounted = False

    def approve(*a, **k):
        if h.uncounted:
            h.uncounted = False
            h.motions_begun += 1
    h.begin_motion, h.end_motion, h.approve = begin, end, approve
    return lambda: {"arm_motions": h.motions_begun, "gripper_closes": env.rt._gripper_closes_commanded,
                    "gripper_opens": env.rt._gripper_opens_completed}


def _always_refuse_close(monkeypatch):
    original = observed_scene.for_runtime

    def make(*args):
        gate = original(*args)

        def close(state, **kw):
            raise _precommand("closing fingers occluded by non-target depth", {"injected": True})
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, "for_runtime", make)


@pytest.mark.parametrize("case", ["next_candidate_grasps", "every_candidate_refused"])
def test_a_next_candidate_keeps_the_verifier_counts_of_the_call_honest(monkeypatch, case):
    """The effects verifier diffs the runtime's counts around ONE call
    (arm motions started, gripper closes completed). Across the next-candidate
    re-entry: a vetoed close was never sent and is not counted; the next
    candidate's close stages and the motions of every candidate are. When
    every candidate is refused before contact the arm moved but no jaw
    closed: the verdict is neither "no grasp was attempted" nor "the jaw
    closed"."""
    from cascade.agent.effects import CONFIRMED, REFUTED, PostconditionChecker
    single = build(monkeypatch, neighbour=False)
    single_counts = _count_motions(single)
    assert grasp(single)["held"] == "cube"
    yaws = [(k * np.pi / 6, .95 - .01 * k, f"c{k}") for k in range(5)]
    env = build(monkeypatch, neighbour=False, **({} if case == "next_candidate_grasps" else {"grasps": yaws}))
    if case == "next_candidate_grasps":
        _refuse_first_close(monkeypatch)
    else:
        _always_refuse_close(monkeypatch)
    poses = {"cube": TARGET.tolist()}
    checker = PostconditionChecker(belief_pose=lambda label: poses.get(label),
                                   gripper_frac=lambda: env.bridge.gripper_pos, actuation=_count_motions(env))
    before = checker.snapshot("cube")
    if case == "next_candidate_grasps":
        result = grasp(env)
        poses["cube"] = (TARGET + [0., 0., .1]).tolist()
    else:
        with pytest.raises(SafetyViolation, match="3 grasp candidates refused before contact") as err:
            grasp(env)
        result = {"ok": False, "error": f"SafetyViolation: {err.value}"}
    acted = checker._actuation_delta(before)
    sent = [c for c in env.calls if c[0] == "gripper" and c[1] < 1.]
    assert acted["gripper_closes"] == len(sent) == (2 if case == "next_candidate_grasps" else 0)
    assert acted["arm_motions"] > single_counts()["arm_motions"] > 0   # the refused candidates moved too
    pc = checker.verify("grasp_object", {"label": "cube"}, result, before=before)
    assert "no grasp was attempted" not in pc.evidence
    if case == "next_candidate_grasps":
        assert pc.status == CONFIRMED
    else:
        assert pc.status == REFUTED and "did not rise" in pc.evidence and "closed" not in pc.evidence


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
    with pytest.raises(MotionHalted, match=r"^halt received during route planning or execution$"):
        grasp(env)
    assert len(moves) == 4
    assert [c for c in env.calls if c[0] == "gripper"] == [("gripper", 1., .8)]


def _with_grasp_cfg(env, **values):
    env.rt.cfg = Cfg({**env.rt.cfg.as_dict(), "grasp": {**env.rt.cfg.as_dict()["grasp"], **values}})


@pytest.mark.parametrize("key,value", [
    ("observed_rank_lateral_m", float("nan")), ("observed_rank_lateral_m", 0.),
    ("observed_rank_lateral_m", -.01), ("observed_rank_lateral_m", .0301), ("observed_rank_lateral_m", .5),
    ("observed_rank_lateral_m", True), ("observed_rank_lateral_m", "0.01"),
    ("observed_rank_max_candidates", -1), ("observed_rank_max_candidates", 1.5),
    ("observed_rank_max_candidates", True), ("observed_rank_max_candidates", "32"),
])
def test_rank_settings_validation_fails_closed_before_motion(monkeypatch, key, value):
    """Ranking distance: 0 < value <= 0.03 m (a larger one would rank every
    candidate last). Candidate limit: an integer >= 0 (0 = no ranking)."""
    env = build(monkeypatch)
    _with_grasp_cfg(env, **{key: value})
    with pytest.raises(SkillError, match=key):
        grasp(env)
    assert all(c[0] == "read" for c in env.calls)


@pytest.mark.parametrize("values", [{"observed_rank_lateral_m": .03}, {"observed_rank_max_candidates": 0},
                                    {"observed_rank_lateral_m": .005, "observed_rank_max_candidates": 1}])
def test_rank_settings_at_their_bounds_are_accepted(monkeypatch, values):
    env = build(monkeypatch)
    _with_grasp_cfg(env, **values)
    assert grasp(env)["held"] == "cube"


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


def _part_corners(planes):
    """Test oracle, independent of the code under test: the corners of one
    envelope part (intersection of its halfspaces), TCP frame."""
    from scipy.optimize import linprog
    from scipy.spatial import HalfspaceIntersection
    A_, b_ = planes[:, :3], planes[:, 3]
    centre = linprog(np.r_[0., 0., 0., -1.], A_ub=np.c_[A_, np.linalg.norm(A_, axis=1)], b_ub=-b_,
                     bounds=[(None, None)] * 3 + [(0, None)])
    assert centre.status == 0 and centre.x[3] > 1e-6
    return HalfspaceIntersection(planes, centre.x[:3]).intersections


_CORNERS = {}


def _finger_corners(envelopes):
    key = id(envelopes)
    if key not in _CORNERS:
        _CORNERS[key] = (envelopes, [_part_corners(e[2]) for e in envelopes])
    return _CORNERS[key][1]


def test_horizontal_dilation_contains_every_horizontal_shift_and_never_grows_down():
    from cascade.grasping.observed_scene import _envelope_vertices, _horizontal_dilation
    rng = np.random.default_rng(4)
    exact = GEOMETRY.envelopes(GEOMETRY.upper, GEOMETRY.upper)
    corners = _finger_corners(exact)
    lateral = .01
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    shifts = np.concatenate([np.zeros((1, 3))] + [
        np.c_[r * np.cos(angles), r * np.sin(angles), np.zeros(16)] for r in (.0025, .005, .0075, .01)])

    def inside(parts, p):
        return any(np.all(parts_planes[:, :3] @ p + parts_planes[:, 3] <= 1e-12)
                   for _, _, parts_planes, *_ in parts)
    for yaw, tilt in ((0., 0.), (.7, 0.), (2.1, .2), (-1.3, np.deg2rad(18.)), (.4, np.deg2rad(9.))):
        R = _yaw_rotation(yaw)
        if tilt:
            c, s_ = np.cos(tilt), np.sin(tilt)
            R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]) @ R
        dilated = _horizontal_dilation(exact, R, lateral, _envelope_vertices(exact))
        lo = np.min([e[3] for e in exact], axis=0)
        hi = np.max([e[4] for e in exact], axis=0)
        for _ in range(180):
            p_tcp = rng.uniform(lo - .012, hi + .012)
            p_base = R @ p_tcp
            hit = any(inside(exact, R.T @ (p_base - s)) for s in shifts)
            if hit:   # conservative: every horizontally displaced finger point is inside
                assert inside(dilated, p_tcp)
        # Tilted or not, nothing below a part's own lowest point (base z) is
        # added: the table under a low grasp is not a neighbour. Probe right
        # under each part's lowest corner, inside the horizontal reach.
        for (name, part, *_), dilated_part, part_corners in zip(exact, dilated, corners):
            low = part_corners[np.argmin(part_corners @ R[2])]
            for _ in range(25):
                r, a = lateral * np.sqrt(rng.uniform()), rng.uniform(0., 2 * np.pi)
                p_base = R @ low + [r * np.cos(a), r * np.sin(a), -rng.uniform(1e-5, .004)]
                assert not inside([dilated_part], R.T @ p_base), (name, part, yaw, tilt)


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


@pytest.mark.parametrize("near, ranked", [("neither", False), ("planned", False), ("twin", False),
                                           ("both", True)])
def test_ranking_uses_the_better_jaw_orientation(near, ranked):
    """The executor may run either jaw orientation (the jaws-swapped twin,
    yaw + pi about the approach), so a candidate is ranked by the BETTER one:
    it goes last only when BOTH orientations come near a neighbour. A tilted
    kitchen candidate whose planned orientation dips 0.2 mm under the table
    plane while its twin clears it (main executed the twin) keeps its place.
    A planned orientation that is clear needs no twin queries."""
    from cascade.grasping.observed_scene import neighbour_ranked
    from cascade.grasping.selector import _flip_twin
    cand, far = candidate(0., .9, "cand"), candidate(np.pi / 4, .8, "far")
    rotations = {"planned": [cand.rotation], "twin": [_flip_twin(cand).rotation], "neither": [],
                 "both": [cand.rotation, _flip_twin(cand).rotation]}[near]
    seen = []

    def proximity(T, lateral_m, *, closing=False):
        seen.append(T[:3, :3].copy())
        if any(np.allclose(T[:3, :3], R, atol=1e-9) for R in rotations):
            return {"surface": "other observed surface"}
        return None
    ranked_last = []
    order = list(neighbour_ranked([cand, far], SimpleNamespace(neighbour_proximity=proximity), lateral_m=.01,
                                  pregrasp_offset_m=.1, ranked_last=lambda i, g, why: ranked_last.append((i, why))))
    assert [g.label for g in order] == (["far", "cand"] if ranked else ["cand", "far"])
    assert [i for i, _ in ranked_last] == ([0] if ranked else [])
    if ranked:
        assert ranked_last[0][1].startswith("pregrasp fingers within 10 mm of an observed neighbour in both "
                                            "jaw orientations")
    mine = seen[:-4]                   # `far` is clear: its planned orientation's 4 queries
    twin_queried = any(np.allclose(R, _flip_twin(cand).rotation, atol=1e-9) for R in mine)
    assert twin_queried == (near in ("planned", "both"))   # a near planned orientation consults the twin
    # Queries before a verdict: clear planned orientation 4; near planned (first query) + twin 4;
    # both near: one query each.
    assert len(mine) == {"neither": 4, "twin": 4, "planned": 5, "both": 2}[near]


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


@pytest.mark.parametrize("ranking", [True, False])
def test_planned_finger_inside_the_neighbour_never_reaches_the_live_gate(monkeypatch, ranking):
    """The neighbour 2 cm closer: A's PLANNED fingers enter it. With ranking
    A goes last and is never vetted (B is executable); with ranking off
    (`observed_rank_max_candidates: 0`) the static pre-vet (the same closing
    envelope as the live gate, no motion) drops A and records why. Either way
    B runs and the gate never refuses."""
    env = build(monkeypatch, neighbour_dx=NEIGHBOUR_DX - .02,
                grasp_cfg={} if ranking else {"observed_rank_max_candidates": 0})
    assert _bare_gate(env).closing_pose(np.r_[TARGET, 0.]) is not None   # precondition
    result = grasp(env)
    assert result["held"] == "cube" and env.refusals == []
    assert descents(env) == pytest.approx([np.pi / 2])
    selection = result["observed_selection"]
    if ranking:
        assert [r["candidate"] for r in selection["ranked_last"]] == [0] and selection["dropped"] == []
    else:
        assert selection["ranked_last"] == [] and [d["candidate"] for d in selection["dropped"]] == [0]
        assert "observed non-target surface" in selection["dropped"][0]["reason"]
        assert env.proximity == []
    assert selection["attempts"] == [{"candidate": 1, "outcome": "grasped"}]
    assert selection["succeeded_attempt"] == 1



# ── fix round 1 (review deleg_12d67c17) ──────────────────────────────────────


def _tilted_scene_gate(monkeypatch, cands):
    """No neighbour, a 2 x 2 x 2 cm box, the real gate (tilting kinematics)."""
    env = build(monkeypatch, neighbour=False, grasps=cands, kin=TiltKin(), home=TILT_HOME, half=(.01, .01),
                box_top=.02)
    scene = ObservedScene(env.frame, env.fix.detection.mask, env.frame.T_base_cam, source=SOURCE,
                          robot_id="/robot", epoch="test")
    gate = ObservedFingerGate(scene, GEOMETRY, TiltKin(), env.rt.arm.get_state(),
                              uncertainty_m=OPEN_TRACKING_ENVELOPE_M)
    return env, gate


def _lowest_finger_z(gate, grasp_):
    """Oracle: lowest base z of the open or closing finger volume at the
    grasp pose, in the candidate's BETTER jaw orientation (the higher of the
    two orientations' lowest points: the ranking judges the better one)."""
    from cascade.grasping.selector import _flip_twin
    best = []
    for g in (grasp_, _flip_twin(grasp_)):
        R = np.asarray(g.rotation, float)
        best.append(min(float(np.min(c @ R[2])) for envelopes in (gate.envelopes, gate.closing_envelopes)
                        for c in _finger_corners(envelopes)))
    return max(best) + float(grasp_.position[2])


def test_table_is_not_a_neighbour_for_tilted_low_grasps(monkeypatch):
    """Reviewer probe: no neighbour, a 2 cm box, 32 learned-style candidates
    tilted 5-18 deg with the TCP 10 mm above the table. Moving the finger
    planes out horizontally also reached up to lateral * sin(2 tilt) BELOW a
    tilted finger, so the table ranked such candidates last. Every candidate
    whose fingers stay above the table in either jaw orientation keeps its
    planned place (t0: planned orientation 0.2 mm under the table plane, its
    twin above it)."""
    cands = tilted_set(32, z=.010)
    env, gate = _tilted_scene_gate(monkeypatch, cands)
    ranked = []
    list(observed_scene.neighbour_ranked(cands, gate, lateral_m=OBSERVED_RANK_LATERAL_M, pregrasp_offset_m=.1,
                                         ranked_last=lambda i, g, why: ranked.append(i)))
    above = [i for i, g in enumerate(cands) if _lowest_finger_z(gate, g) > 0.]
    assert len(above) >= 20                      # precondition: most of the set clears the table
    assert 0 in above                            # precondition: t0, the better orientation clears it
    assert sorted(set(ranked) & set(above)) == []


# Measured on main (ad4786f, again on c9f8465) with the same scene and
# doubles: the candidate executed, all bridge calls and the joint targets
# streamed. Main executes t0 as its jaws-swapped twin (t0's planned
# orientation dips 0.2 mm under the table plane, the twin clears it).
_TILTED_MAIN = {0: ("t0", 275, 55), 1: ("t1", 287, 58), 12: ("t12", 271, 54), 15: ("t15", 271, 54),
                18: ("t18", 279, 56)}


@pytest.mark.parametrize("top", sorted(_TILTED_MAIN))
def test_tilted_low_grasp_without_neighbour_keeps_main_choice_and_commands(monkeypatch, top):
    """Pin (the reviewer's 6/6 divergence): no neighbour, 32 tilted
    candidates on a 2 cm box, the planned list starting at `top`. The choice,
    every bridge call and the jaw sequence equal main's."""
    cands = tilted_set(32, z=.010)
    order = cands[top:] + cands[:top]
    env = build(monkeypatch, neighbour=False, grasps=order, kin=TiltKin(), home=TILT_HOME, half=(.01, .01),
                box_top=.02)
    result = grasp(env)
    label, n_calls, n_targets = _TILTED_MAIN[top]
    assert result["held"] == "cube" and env.refusals == []
    selection = result["observed_selection"]
    assert selection["ranked_last"] == []
    assert order[selection["attempts"][0]["candidate"]].label == label
    assert len(env.calls) == n_calls and len(joint_targets(env)) == n_targets
    assert [c for c in env.calls if c[0] == "gripper"] == [
        ("gripper", 1., .8), ("gripper", .5, pytest.approx(.49)), ("gripper", pytest.approx(.15), .7)]


def _near_copies(n):
    """`n` copies of A (beside the neighbour), 0.1 mm apart along y."""
    return [Grasp(TARGET + [0., 1e-4 * k, 0.], _yaw_rotation(0.), .04, np.array([0., 0., -1.]),
                  .95 - 1e-3 * k, f"A{k}") for k in range(n)]


def test_ranking_stops_after_the_candidate_limit_and_keeps_planned_order(monkeypatch):
    """More candidates beside the neighbour than the limit: the first
    `observed_rank_max_candidates` go last, the next keeps its planned place
    and is chosen; B (clear, planned after them) is not reached. An unbounded
    ranking would reach B."""
    env = build(monkeypatch, grasps=_near_copies(RANK_MAX_CANDIDATES + 8) + [candidate(*B)])
    result = grasp(env)
    selection = result["observed_selection"]
    assert result["held"] == "cube" and env.refusals == []
    assert selection["ranking"]["ranked"] == RANK_MAX_CANDIDATES
    assert selection["ranking"]["stopped"] == f"limit of {RANK_MAX_CANDIDATES} ranked candidates"
    assert selection["attempts"] == [{"candidate": RANK_MAX_CANDIDATES, "outcome": "grasped"}]
    assert descents(env) == pytest.approx([0.])


def test_ranking_stops_after_its_time_slice():
    """The ranking's own work is capped (checked before every query): with
    50 ms per query and a 0.3 s slice, six queries run. c0 is clear in its
    planned jaw orientation after four (ranked; its twin is not queried), c1
    is unfinished after two more: it and every later one keep their planned
    order unranked."""
    from cascade.grasping.observed_scene import NeighbourRanking
    now = [0.]

    def proximity(T, lateral_m, *, closing=False):
        now[0] += .05
        return None
    ranking = NeighbourRanking(SimpleNamespace(neighbour_proximity=proximity), lateral_m=.01,
                               pregrasp_offset_m=.1, max_candidates=32, time_slice_s=.3, clock=lambda: now[0])
    planned = [candidate(k * .1, .9, f"c{k}") for k in range(5)]
    assert [g.label for g in ranking.order(planned)] == [g.label for g in planned]
    assert ranking.summary() == {"ranked": 1, "queries": 6, "ranking_s": .3, "stopped": "time slice of 0.3 s"}


def test_next_candidate_reuses_the_first_ranking(monkeypatch):
    """A (beside the neighbour) is ranked once, in the first call. After B's
    pre-contact refusal the re-entry orders the rest from that ranking (A
    stays last) and queries only the candidate it had not ranked yet (C)."""
    C = (2 * np.pi / 3, .85, "C: closes at 120 degrees")
    env = build(monkeypatch, grasps=(A, B, C))
    marks = []
    original = observed_scene.for_runtime

    def make(*args):
        gate = original(*args)
        require_closing = gate.require_closing

        def close(state, **kw):
            if not marks:
                marks.append(len(env.proximity))
                raise _precommand("closing fingers intersects observed non-target surface", {"injected": True})
            return require_closing(state, **kw)
        gate.require_closing = close
        return gate
    monkeypatch.setattr(observed_scene, "for_runtime", make)
    result = grasp(env)
    assert result["held"] == "cube"
    assert [round(y % np.pi, 6) for y in descents(env)] == [round(np.pi / 2, 6), round(2 * np.pi / 3, 6)]
    selection = result["observed_selection"]
    assert [a["candidate"] for a in selection["attempts"]] == [1, 2]
    assert [r["candidate"] for r in selection["ranked_last"]] == [0]
    first, again = env.proximity[:marks[0]], env.proximity[marks[0]:]
    assert {_jaw_axis(T) for T in first} == {0., round(np.pi / 2, 6)}
    assert again and {_jaw_axis(T) for T in again} == {round(2 * np.pi / 3, 6)}


def _jaw_axis(T):
    """Yaw of the closing axis mod pi (a grasp and its jaws-swapped twin agree)."""
    a = yaw_of(np.asarray(T)[:3, :3]) % np.pi
    return 0. if np.isclose(a, 0., atol=1e-6) or np.isclose(a, np.pi, atol=1e-6) else round(a, 6)


class _Selected(Exception):
    """Raised at the first actuation: the choice is made."""


def _select_only(monkeypatch, grasps, grasp_cfg, **scene):
    env = build(monkeypatch, real_time=True, size=(720, 1280), grasps=grasps, grasp_cfg=grasp_cfg, **scene)

    def first_actuation(*a, **k):
        raise _Selected()
    env.rt._set_gripper = first_actuation
    with pytest.raises(_Selected) as err:
        grasp(env)
    return err.value.observed_selection


_RING = (.0725, .10, .05)   # inner edge 7.25 cm from the target: exact-clear, within 1 cm


@pytest.mark.parametrize("case", ["near_ring", "ik_failures"])
def test_bounded_ranking_stays_inside_the_approach_budget_at_kitchen_resolution(monkeypatch, case):
    """Real clock, 720 x 1280 depth (921k observed points), 400 candidates:
    all beside a ring-shaped neighbour, or 399 that fail IK before the one
    that solves. Unbounded, ranking took seconds and the approach deadline
    (PLAN_BUDGET_S) refused every route; bounded, the choice costs at most
    0.4 s more than with ranking off."""
    if case == "near_ring":
        grasps = [(k * np.pi / 400, .95 - 1e-4 * k, f"r{k}") for k in range(400)]
        scene = {"neighbour": False, "ring": _RING}
    else:
        rng = np.random.default_rng(3)
        grasps = [tilted(k, tilt_deg=rng.uniform(5., 18.), yaw=rng.uniform(-np.pi, np.pi),
                         tilt_axis=rng.uniform(0., 2 * np.pi), z=TARGET[2], width=.04) for k in range(399)]
        grasps.append(candidate(0., .5, "the one that solves"))
        scene = {"neighbour": False}
    off = _select_only(monkeypatch, grasps, {"observed_rank_max_candidates": 0}, **scene)
    on = _select_only(monkeypatch, grasps, {}, **scene)
    assert off.get("ranking") is None
    assert 0 < on["ranking"]["ranked"] <= RANK_MAX_CANDIDATES
    assert on["ranking"]["ranking_s"] <= RANK_TIME_SLICE_S + .1
    assert on["selection_s"] - off["selection_s"] < .4, (on["selection_s"], off["selection_s"], on["ranking"])
    if case == "near_ring":
        assert on["ranked_last"]                  # the ring was seen


@pytest.mark.parametrize("blocked_yaw,executed_yaw", [(np.pi, 0.), (0., np.pi)])
def test_a_refused_jaw_orientation_is_not_a_drop_when_the_other_passes(monkeypatch, blocked_yaw, executed_yaw):
    """A drop is recorded only when NEITHER jaw orientation passes the vet."""
    env = build(monkeypatch, neighbour=False)
    vet = env.harness.vet_pose
    env.harness.vet_pose = lambda q, **k: ("blocked orientation" if np.isclose(abs(q[3]), blocked_yaw)
                                          else vet(q, **k))
    result = grasp(env)
    assert result["held"] == "cube" and abs(descents(env)[0]) == pytest.approx(executed_yaw)
    selection = result["observed_selection"]
    assert selection["dropped"] == [] and selection["dropped_count"] == 0
    assert selection["attempts"] == [{"candidate": 0, "outcome": "grasped"}]


def test_no_executable_grasp_carries_the_choice_record(monkeypatch):
    from cascade.grasping.selector import NoExecutableGrasp
    env = build(monkeypatch, neighbour=False)
    env.harness.vet_pose = lambda q, **k: "blocked"
    with pytest.raises(NoExecutableGrasp) as err:
        grasp(env)
    selection = err.value.observed_selection
    assert [d["candidate"] for d in selection["dropped"]] == [0, 1] and selection["dropped_count"] == 2
    assert selection["attempts"] == []
    assert all(c[0] == "read" for c in env.calls)


@pytest.mark.parametrize("failure", ["route budget", "home did not settle"])
def test_a_failed_next_candidate_names_the_first_refusal(monkeypatch, failure):
    """After a pre-contact refusal and withdrawal, any failure of the next
    candidate keeps its own error and adds the first refusal, the count of
    refused candidates and the choice record."""
    env = build(monkeypatch, neighbour=False)
    closes = _refuse_first_close(monkeypatch)
    if failure == "route budget":
        vet = env.harness.vet_pose
        late = []

        def slow_vet(q, **k):
            if closes and not late:
                late.append(1)
                env.clock[0] += 4.            # the next candidate's route vet outlives PLAN_BUDGET_S
            return vet(q, **k)
        env.harness.vet_pose = slow_vet
        expected, attempts = "route preflight exceeded its time budget", 1
    else:
        move = env.rt._move_joints
        moves = []

        def moved(*a, **k):
            moves.append(1)
            return False if len(moves) == 5 else move(*a, **k)   # the next candidate's home move
        env.rt._move_joints = moved
        expected, attempts = "did not settle at required observed-finger home pose", 2
    with pytest.raises(SkillError, match=expected) as err:
        grasp(env)
    assert ("; after 1 earlier grasp candidate refused before contact and withdrew "
            "(first refusal: closing fingers intersects observed non-target surface") in str(err.value)
    selection = err.value.observed_selection
    assert len(selection["attempts"]) == attempts
    assert selection["attempts"][0]["outcome"].startswith("refused before contact")
    if attempts == 2:
        assert selection["attempts"][1]["outcome"].startswith(f"failed: {expected}")
    assert not [c for c in env.calls if c[0] == "gripper" and c[1] < 1.]


def _refuse_every_close(monkeypatch):
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
    return closes


def test_pick_and_place_failure_result_carries_the_choice_record(monkeypatch):
    yaws = [(k * np.pi / 6, .95 - .01 * k, f"c{k}") for k in range(5)]
    env = build(monkeypatch, neighbour=False, grasps=yaws)
    env.rt._localize = lambda label, spatial_hint=None: (env.frame, env.fix)
    closes = _refuse_every_close(monkeypatch)
    result = SkillRuntime.skill_pick_and_place(env.rt, "cube")
    assert result["ok"] is False and result["stage"] == "grasp" and len(closes) == 3
    assert "3 grasp candidates refused before contact" in result["error"]
    selection = result["observed_selection"]
    assert [a["candidate"] for a in selection["attempts"]] == [0, 1, 2]
    assert selection["first_refusal"].startswith("closing fingers occluded")


def test_grasp_object_failure_result_carries_the_choice_record(monkeypatch, tmp_path):
    from owned_server import held_dead_port

    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config
    record = {"ranked_last": [], "dropped": [{"candidate": 0, "reason": "blocked"}], "dropped_count": 1,
              "attempts": []}
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        runtime, arm = build_runtime(load_demo_config(camera="mock", arm="mock", llm="mock"), tmp_path / "run")
        try:
            def refused(label, **kw):
                exc = SafetyViolation("closing fingers intersects observed non-target surface: {}")
                exc.observed_selection = record
                raise exc
            runtime.skill_grasp_object = refused
            result = runtime.execute("grasp_object", {"label": "red cube"})
        finally:
            shutdown_runtime(runtime, arm)
    assert result["ok"] is False and "closing fingers intersects" in result["error"]
    assert result["observed_selection"] == record


@pytest.mark.parametrize("offset_m,yaw_deg,skipped", [(.003, 5., True), (.003, 20., False), (.012, 5., False)])
def test_a_near_copy_of_the_refused_candidate_is_not_tried_next(monkeypatch, offset_m, yaw_deg, skipped):
    """Learned planners propose near-copies of one grasp; after A's
    pre-contact refusal a copy with its TCP within 1 cm and its jaws within
    15 deg (approach included) would be refused the same way, so the retry
    skips it. A grasp at the same centre with the jaws turned (B) is not a
    copy: it puts the fingers elsewhere."""
    copy_a = Grasp(TARGET + [offset_m, 0., 0.], _yaw_rotation(np.deg2rad(yaw_deg)), .04, np.array([0., 0., -1.]),
                   .94, "A': a copy of A?")
    env = build(monkeypatch, neighbour=False, grasps=(A, copy_a, B))
    _refuse_first_close(monkeypatch)
    result = grasp(env)
    assert result["held"] == "cube"
    attempts = [a["candidate"] for a in result["observed_selection"]["attempts"]]
    if skipped:
        assert attempts == [0, 2] and descents(env) == pytest.approx([0., np.pi / 2])
    else:
        assert attempts == [0, 1] and descents(env) == pytest.approx([0., np.deg2rad(yaw_deg)])


def test_the_same_grasp_descending_along_another_approach_is_not_a_copy(monkeypatch):
    """Same TCP and jaws as A but approached 20 deg off vertical: its
    pregrasp and descent (along `approach`) sweep the fingers elsewhere, so
    it stays a candidate after A's refusal."""
    tilt = np.deg2rad(20.)
    slanted = Grasp(TARGET.copy(), _yaw_rotation(0.), .04, np.array([np.sin(tilt), 0., -np.cos(tilt)]), .94,
                    "A along a slanted approach")
    env = build(monkeypatch, neighbour=False, grasps=(A, slanted))
    _refuse_first_close(monkeypatch)
    result = grasp(env)
    assert result["held"] == "cube"
    assert [a["candidate"] for a in result["observed_selection"]["attempts"]] == [0, 1]
    pre_slanted = TARGET - .1 * slanted.approach
    assert any(np.allclose(q[:3], pre_slanted) for q in joint_targets(env))   # its own pregrasp


def test_a_halt_during_the_pre_close_retreat_ends_the_skill_as_a_halt(monkeypatch):
    """Pre-existing on the edited block: the withdrawal's halt refusal was
    folded into the veto's SafetyViolation text. It now propagates as the
    MotionHalted it is, before anything of the withdrawal streams."""
    env = build(monkeypatch, neighbour=False)
    _refuse_first_close(monkeypatch)
    move = env.rt._move_joints
    moves = []

    def moved(*a, **k):
        moves.append(1)
        if len(moves) == 4:                          # home, pregrasp, descent, then the withdrawal
            env.harness._halt_generation += 1        # operator halt as the withdrawal starts
        return move(*a, **k)
    env.rt._move_joints = moved
    with pytest.raises(MotionHalted, match=r"^halt received during route planning or execution$"):
        grasp(env)
    assert len(moves) == 4
    assert [c for c in env.calls if c[0] == "gripper"] == [("gripper", 1., .8)]
    assert len(joint_targets(env)) == 51             # home, pregrasp, descent; the withdrawal never streamed
