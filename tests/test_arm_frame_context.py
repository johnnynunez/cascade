"""The per-call arm frame (spec 2026-10-17 base_pose runtime frame, §3.2; Q4 a, D1-D5).

`execute(arm=X)` binds X's own Kinematics, resolved profile and base_pose for
that one call, thread-locally, beside the arm selection; unbound reads (other
threads, between calls) see the primary. Pre-flight rulings pinned here:
R3 (the bind window covers the pre-state snapshot and the verifier), R4 (the
verifier's table height is read per call), R6 (hand-built rigs register
bindings), R7 (#357's running-profile lookup agrees with the bound profile),
R8 (every arm's grasp section is validated before any arm is made), R13 (rig
grasp services stay top-level), R17 (sketch fixes).
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import yaml
from base_pose_oracle import bind_like_primary

from cascade.config import Cfg, load_demo_config

REPO = Path(__file__).resolve().parents[1]

SHIPPED_RIGHT = [0.0, -0.22, 0.0, 0.0, 0.0, 0.0]
YAWED = [0.03, -0.22, 0.0, 0.0, 0.0, 0.4]
RAISED = [0.0, -0.22, 0.03, 0.0, 0.0, 0.0]
IDENTITY = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

#: The so101 jaw mapping (configs/arms/so101_*.yaml gripper).
SO101_OPEN, SO101_CLOSED = 0.60, -0.15


def _cfg(arms, poses=None, camera="mock_small"):
    if len(arms) == 1:
        cfg = load_demo_config(arm=arms[0], camera=camera, llm="mock")
    else:
        cfg = load_demo_config(arms=arms, camera=camera, llm="mock")
    entries = [cfg._data["arm"], *(cfg._data.get("arms") or [])]
    for name, pose in (poses or {}).items():
        hit = [e for e in entries if e.get("name") == name]
        assert hit, name
        for entry in hit:
            entry["base_pose"] = list(pose)
    return cfg


@pytest.fixture
def make_runtime(tmp_path, monkeypatch):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    monkeypatch.setenv("CASCADE_GRASP_BACKEND", "obb")
    built = []

    def make(arms, poses=None, *, lazy=False, cfg=None):
        cfg = cfg if cfg is not None else _cfg(arms, poses)
        rt, arm = build_runtime(cfg, tmp_path / f"run{len(built)}", view=False, serve=False, lazy_arm=lazy)
        built.append((rt, arm))
        return rt

    try:
        yield make
    finally:
        for rt, arm in built:
            shutdown_runtime(rt, arm)


def _raw(safe):
    raw = safe.raw
    return raw.__dict__.get("_arm", raw)


def _record_inside(rt, monkeypatch, read):
    """Patch list_objects to record `read()` inside each call."""
    seen = []

    def spy():
        seen.append(read())
        return {"ok": True}

    monkeypatch.setattr(rt, "skill_list_objects", spy)
    return seen


# ── 1. the helpers: no transform without a base_pose ──────────────────────


def _fix(T_points=None):
    from cascade.types import Detection, ObjectFix

    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.1, 0.3, size=(20, 3)) if T_points is None else T_points
    axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return ObjectFix(label="cube", position=pts.mean(axis=0), points=pts,
                     detection=Detection(label="cube", conf=0.9, bbox=np.zeros(4, np.float32)),
                     extent=np.array([0.05, 0.04, 0.03]), axes=axes)


def _frame(T_base_cam=None):
    from cascade.types import Frame

    return Frame(rgb=np.zeros((4, 4, 3), np.uint8), depth_m=None, K=np.eye(3), T_base_cam=T_base_cam)


def test_helpers_return_their_input_without_a_base_pose():
    from cascade.skills import arm_frame as af

    p = np.array([0.2, -0.1, 0.05])
    pts = np.ones((4, 3))
    M = np.eye(4)
    M[:3, 3] = [0.5, 0.0, 0.4]
    frame, fix = _frame(M), _fix()
    ext = SimpleNamespace(cam_to_base=lambda: pytest.fail("no transform needs no camera pose"))
    assert af.to_arm(None, p) is p and af.to_world(None, p) is p and af.to_world(None, pts) is pts
    assert af.dir_to_arm(None, p) is p and af.dir_to_world(None, p) is p
    assert af.cam_to_arm(None, M) is M
    assert af.frame_to_arm(None, frame, ext) is frame
    assert af.fix_to_world(None, fix) is fix and af.fix_to_arm(None, fix) is fix
    assert af.table_z_in_world(None, -0.03) == -0.03


def test_helpers_invert_each_other_on_a_yawed_pose():
    from cascade.skills import arm_frame as af
    from cascade.types import pose_to_transform

    T = pose_to_transform(YAWED)
    R = T[:3, :3]
    P = np.random.default_rng(0).uniform(-1, 1, size=(50, 3))
    np.testing.assert_allclose(af.to_world(T, af.to_arm(T, P)), P, atol=1e-12)
    np.testing.assert_allclose(af.to_world(T, np.zeros(3)), [0.03, -0.22, 0.0], atol=1e-12)
    np.testing.assert_allclose(af.to_arm(T, af.to_world(T, P[0])), P[0], atol=1e-12)
    v = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(af.dir_to_world(T, v), R @ v, atol=1e-12)
    np.testing.assert_allclose(af.dir_to_arm(T, R @ v), v, atol=1e-12)
    C = pose_to_transform([0.4, 0.1, 0.6, 3.0, 0.1, -0.2])
    np.testing.assert_allclose(af.cam_to_arm(T, C), np.linalg.inv(T) @ C, atol=1e-12)
    framed = af.frame_to_arm(T, _frame(C), SimpleNamespace(cam_to_base=lambda: pytest.fail("frame T wins")))
    np.testing.assert_allclose(framed.T_base_cam, np.linalg.inv(T) @ C, atol=1e-12)
    static = af.frame_to_arm(T, _frame(None), SimpleNamespace(cam_to_base=lambda: C))
    np.testing.assert_allclose(static.T_base_cam, np.linalg.inv(T) @ C, atol=1e-12)
    fix = _fix()
    world = af.fix_to_world(T, fix)
    np.testing.assert_allclose(world.axes, R @ fix.axes, atol=1e-12)
    np.testing.assert_allclose(world.points, fix.points @ R.T + T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(world.position, R @ fix.position + T[:3, 3], atol=1e-12)
    assert world.extent is fix.extent and world.detection is fix.detection and world.label == fix.label
    back = af.fix_to_arm(T, world)
    np.testing.assert_allclose(back.points, fix.points, atol=1e-12)
    np.testing.assert_allclose(back.axes, fix.axes, atol=1e-12)
    assert af.table_z_in_world(pose_to_transform(RAISED), -0.03) == pytest.approx(0.0, abs=1e-12)


class _Harness:
    def __init__(self, pose):
        self.base_pose = pose


class _Arm:
    def __init__(self, harness):
        self.harness = harness


def _world_case(case):
    from cascade.types import pose_to_transform

    scaled = pose_to_transform(SHIPPED_RIGHT)
    scaled[:3, :3] *= 1.01
    nan = np.eye(4)
    nan[0, 3] = np.nan
    mirror = pose_to_transform(SHIPPED_RIGHT) @ np.diag([1.0, 1.0, -1.0, 1.0])   # orthonormal, det -1
    projective = pose_to_transform(SHIPPED_RIGHT)
    projective[3, 2] = 0.1
    return {
        "no harness": object(),
        "no pose": _Arm(_Harness(None)),
        "identity": _Arm(_Harness(pose_to_transform(IDENTITY))),
        "Mock arm": mock.Mock(),
        "Mock harness": _Arm(mock.Mock()),
        "shipped": _Arm(_Harness(pose_to_transform(SHIPPED_RIGHT))),
        "string": _Arm(_Harness("not a matrix")),
        "3x3": _Arm(_Harness(np.eye(3))),
        "nan": _Arm(_Harness(nan)),
        "scaled R": _Arm(_Harness(scaled)),
        "mirror": _Arm(_Harness(mirror)),
        "projective": _Arm(_Harness(projective)),
    }[case]


@pytest.mark.parametrize("case", ["no harness", "no pose", "identity", "Mock arm", "Mock harness",
                                  "shipped", "string", "3x3", "nan", "scaled R", "mirror", "projective"])
def test_world_from_base_reads_only_a_real_rigid_pose(case):
    from cascade.skills.arm_frame import ArmFrameError, world_from_base
    from cascade.types import SkillError, pose_to_transform

    arm = _world_case(case)
    if case in ("no harness", "no pose", "identity", "Mock arm", "Mock harness"):
        assert world_from_base(arm) is None
    elif case == "shipped":
        np.testing.assert_array_equal(world_from_base(arm), pose_to_transform(SHIPPED_RIGHT))
    else:
        with pytest.raises(ArmFrameError, match="base_pose is not a rigid transform"):
            world_from_base(arm)
        assert issubclass(ArmFrameError, SkillError)


def test_world_from_base_never_materializes_a_lazy_arm():
    from cascade.control.lazy_arm import LazyArm
    from cascade.safety.harness import SafeArm
    from cascade.skills.arm_frame import world_from_base
    from cascade.types import pose_to_transform

    made = []
    lazy = LazyArm(lambda: made.append(1))
    assert world_from_base(lazy) is None
    np.testing.assert_array_equal(world_from_base(SafeArm(lazy, _Harness(pose_to_transform(YAWED)))),
                                  pose_to_transform(YAWED))
    assert world_from_base(SafeArm(lazy, _Harness(None))) is None
    assert lazy.connected is False and made == []


def test_level_and_tilt_are_read_from_the_rotation():
    from cascade.skills.arm_frame import is_level, tilt_deg
    from cascade.types import pose_to_transform

    assert is_level(pose_to_transform(YAWED))
    assert not is_level(pose_to_transform([0, -0.22, 0, 0.1, 0, 0]))
    roll, pitch = tilt_deg(pose_to_transform([0, -0.22, 0, 0.1, 0.0, 0.4]))
    assert roll == pytest.approx(np.degrees(0.1)) and pitch == pytest.approx(0.0, abs=1e-9)
    roll, pitch = tilt_deg(pose_to_transform([0, -0.22, 0, 0.0, 0.1, 0.4]))
    assert pitch == pytest.approx(np.degrees(0.1)) and roll == pytest.approx(0.0, abs=1e-9)


# ── 2. execute(arm=) binds the arm's kinematics, profile and pose (Q4 a) ──


def test_arm_selection_binds_that_arms_kinematics_profile_and_pose(make_runtime, monkeypatch):
    from cascade.types import pose_to_transform

    rt = make_runtime(["so101_left", "so101_right"])
    left, right = rt.arm_rig.get("so101_left"), rt.arm_rig.get("so101_right")
    primary_kin = rt.kin
    assert primary_kin is left.harness.kin
    seen = _record_inside(rt, monkeypatch, lambda: {
        "kin": rt.kin, "drop": list(rt.cfg.grasp.drop_zone), "name": rt.cfg.arm.get("name"),
        "T": rt.T_world_base, "width": rt._max_width, "arm": rt.arm})
    assert rt.execute("list_objects", {"arm": "so101_right"})["ok"] is True
    assert rt.execute("list_objects", {})["ok"] is True
    inside_right, inside_primary = seen
    assert inside_right["kin"] is right.harness.kin and inside_right["kin"] is not primary_kin
    assert inside_right["drop"] == [0.20, -0.12] and inside_right["name"] == "so101_right"
    np.testing.assert_allclose(inside_right["T"], pose_to_transform(SHIPPED_RIGHT), atol=1e-12)
    assert inside_right["width"] == 0.055 and inside_right["arm"] is right
    assert inside_primary["kin"] is primary_kin and inside_primary["drop"] == [0.20, 0.12]
    assert inside_primary["name"] == "so101_left" and inside_primary["arm"] is left
    # after both calls: unbound reads are the primary's
    assert rt.kin is primary_kin and rt.cfg.arm.get("name") == "so101_left" and rt.arm is left
    np.testing.assert_allclose(rt.T_world_base, pose_to_transform([0.0, 0.22, 0.0, 0.0, 0.0, 0.0]), atol=1e-12)


def test_the_runtime_helpers_use_the_bound_frame(make_runtime, monkeypatch):
    from cascade.types import pose_to_transform

    rt = make_runtime(["so101_left", "so101_right"], {"so101_right": RAISED})
    T = pose_to_transform(RAISED)
    p = np.array([0.2, -0.1, 0.05])
    seen = _record_inside(rt, monkeypatch, lambda: {
        "to_arm": rt._to_arm(p), "to_world": rt._to_world(p), "dir": rt._dir_to_world(p),
        "cam": rt._cam_to_arm(np.eye(4)), "table": rt._table_z_world(),
        "frame": rt._frame_in_arm(_frame(np.eye(4))).T_base_cam,
        "fix": rt._fix_to_world(_fix()).position, "back": rt._fix_to_arm(_fix()).position})
    rt.execute("list_objects", {"arm": "so101_right"})
    got = seen[0]
    np.testing.assert_allclose(got["to_arm"], p - T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(got["to_world"], p + T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(got["dir"], p, atol=1e-12)
    np.testing.assert_allclose(got["cam"], np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(got["frame"], np.linalg.inv(T), atol=1e-12)
    np.testing.assert_allclose(got["fix"], _fix().position + T[:3, 3], atol=1e-12)
    np.testing.assert_allclose(got["back"], _fix().position - T[:3, 3], atol=1e-12)
    # so101_right's own table_z (its base frame) lifted by its 3 cm-up base
    table_b = float(rt.arm_rig.get("so101_right").harness.limits.table_z)
    assert got["table"] == pytest.approx(table_b + 0.03, abs=1e-12)


def test_a_mixed_model_rig_solves_with_the_selected_arms_model(make_runtime, monkeypatch):
    rt = make_runtime(["so101_mock", "mock"])
    primary_kin = rt.kin
    seen = _record_inside(rt, monkeypatch, lambda: {
        "name": rt.cfg.arm.get("name"), "kin": rt.kin, "width": rt._max_width,
        "grip": (rt._grip_open, rt._grip_closed)})
    rt.execute("list_objects", {"arm": "mock"})
    rt.execute("list_objects", {})
    on_mock, on_primary = seen
    assert on_mock["name"] == "mock" and on_mock["kin"] is not primary_kin
    assert on_mock["kin"] is rt.arm_rig.get("mock").harness.kin
    assert on_mock["width"] == 0.09 and on_mock["grip"] == (0.0, 1.0)       # configs/arms/mock.yaml
    assert on_primary["name"] == "so101_mock" and on_primary["kin"] is primary_kin
    assert on_primary["width"] == 0.055 and on_primary["grip"] == (0.60, -0.15)
    assert rt._max_width == 0.055


def test_the_frame_is_thread_local_and_restored_after_a_raise(make_runtime, monkeypatch):
    from cascade.types import SkillError

    rt = make_runtime(["so101_left", "so101_right"])
    primary_kin = rt.kin
    other = {}

    def spy():
        def read():
            other.update(kin=rt.kin, name=rt.cfg.arm.get("name"), arm=rt.arm)

        worker = threading.Thread(target=read)
        worker.start()
        worker.join(5)
        other["mine"] = rt.kin
        raise SkillError("boom")

    monkeypatch.setattr(rt, "skill_list_objects", spy)
    result = rt.execute("list_objects", {"arm": "so101_right"})
    assert result["ok"] is False and "boom" in result["error"]
    assert other["kin"] is primary_kin and other["name"] == "so101_left"
    assert other["arm"] is rt.arm_rig.get("so101_left")
    assert other["mine"] is rt.arm_rig.get("so101_right").harness.kin
    assert getattr(rt._arm_override, "frame", None) is None
    assert getattr(rt._arm_override, "arm", None) is None
    assert rt.kin is primary_kin


def test_the_held_binding_is_stable_per_arm(make_runtime, monkeypatch):
    from cascade.skills import held_observation

    rt = make_runtime(["so101_left", "so101_right"])
    seen = _record_inside(rt, monkeypatch, lambda: held_observation.binding(rt))
    rt.execute("list_objects", {"arm": "so101_right"})
    rt.execute("list_objects", {"arm": "so101_right"})
    rt.execute("list_objects", {"arm": "so101_left"})
    first, second, left = seen
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert any(a is not b for a, b in zip(first, left, strict=True))
    assert first[-1] is not left[-1]                     # each arm's own Cfg


# ── 3. fail closed (D4) ───────────────────────────────────────────────────


def _hand_built_runtime(tmp_path):
    from cascade.agent.task_effects import TaskEffects
    from cascade.agent.trace import TraceLogger
    from cascade.control.arm_rig import ArmRig
    from cascade.skills.runtime import SkillRuntime

    rt = SkillRuntime.__new__(SkillRuntime)
    rt._task_effects = TaskEffects()
    rt.cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    rt.cfg._data["arms"] = [{**rt.cfg.arm.as_dict(), "name": name} for name in ("left", "right")]
    left = SimpleNamespace(name="left", harness=SimpleNamespace(violations=[]))
    right = SimpleNamespace(name="right", harness=SimpleNamespace(violations=[]))
    rt._arm = left
    rt.arm_rig = ArmRig([left, right], ["left", "right"])
    rt._arm_override = threading.local()
    rt.held_object = None
    rt.last_frame = rt.watcher = rt._motion_t0 = rt.effects = None
    rt.current_tier = "test"
    rt.memory = SimpleNamespace(memory_frames=lambda _: [], add=lambda *a, **kw: None)
    rt.envelope = SimpleNamespace(record=lambda *a, **kw: None)
    rt.observe = lambda **kwargs: pytest.fail("a refused call observed the scene")
    rt._show_status = lambda _: pytest.fail("a refused call showed a status line")
    rt.trace = TraceLogger(tmp_path / "hand")
    return rt


def test_a_selected_arm_without_a_binding_is_refused_before_any_work(tmp_path):
    from cascade.skills.runtime import FAILED_INSTRUCTION

    rt = _hand_built_runtime(tmp_path)
    ran = []
    rt.skill_move_home = lambda: ran.append("move_home") or {}
    rt.skill_list_objects = lambda: ran.append("list_objects") or {"ok": True}
    for skill in ("move_home", "list_objects"):
        result = rt.execute(skill, {"arm": "right"})
        assert result["ok"] is False, result
        assert result["error"] == (f"{skill} refused on arm 'right': it has no kinematics/profile bound to "
                                   "this runtime; no motion was sent")
        assert result["next_action"].count(FAILED_INSTRUCTION) == 1
        assert "arm_left_in_place" not in result and "move_home before" not in result["next_action"]
    assert ran == []
    # the primary needs no registration, and a registered arm runs
    rt._show_status = lambda _: None
    rt.observe = lambda **kwargs: None
    assert rt.execute("list_objects", {"arm": "left"})["ok"] is True
    bind_like_primary(rt, "right")
    assert rt.execute("list_objects", {"arm": "right"})["ok"] is True
    assert ran == ["list_objects", "list_objects"]


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_an_unreadable_base_pose_refuses_every_skill_on_that_arm(make_runtime, lazy):
    from cascade.skills.runtime import FAILED_INSTRUCTION

    rt = make_runtime(["so101_left", "so101_right"], lazy=lazy)
    right = rt.arm_rig.get("so101_right")
    right.harness.__dict__["base_pose"] = np.full((4, 4), np.nan)
    raws = [a.raw for a in rt.arm_rig.arms.values()]
    n0 = [len(_raw(a).commands) for a in rt.arm_rig.arms.values()] if not lazy else None
    for skill, args in (("move_home", {}), ("rotate_joint", {"joint": 1, "degrees": 5}),
                        ("list_objects", {}), ("grasp_object", {"label": "red cube"})):
        result = rt.execute(skill, {**args, "arm": "so101_right"})
        assert result["ok"] is False, result
        assert result["error"].startswith(f"{skill} refused on arm 'so101_right': its base_pose is not a rigid "
                                          "transform"), result["error"]
        assert result["error"].endswith("; no motion was sent")
        assert result["next_action"].count(FAILED_INSTRUCTION) == 1
        assert "arm_left_in_place" not in result
    if lazy:
        assert [r.connected for r in raws] == [False, False]
    else:
        assert [len(_raw(a).commands) for a in rt.arm_rig.arms.values()] == n0
    # the primary is unaffected
    assert rt.execute("list_objects", {})["ok"] is True


def test_a_halt_is_never_refused_by_an_unreadable_pose(make_runtime):
    rt = make_runtime(["so101_left", "so101_right"])
    right = rt.arm_rig.get("so101_right")
    right.harness.__dict__["base_pose"] = np.full((4, 4), np.nan)
    result = rt.execute("halt_motion", {"reason": "operator", "arm": "so101_right"})
    assert result["ok"] is True, result
    assert right.harness.halted == "operator"


def test_a_halt_is_never_refused_for_a_missing_binding(tmp_path):
    rt = _hand_built_runtime(tmp_path)
    rt._show_status = lambda _: None
    rt.observe = lambda **kwargs: None
    halted = {}
    for safe in rt.arm_rig.arms.values():
        safe.harness.halt = lambda reason, _n=safe.name: halted.setdefault(_n, reason)
    notes = []
    rt.memory.add = lambda *a, **kw: notes.append(a)
    assert "right" not in (getattr(rt, "_arm_bindings", None) or {})      # never registered
    result = rt.execute("halt_motion", {"reason": "operator", "arm": "right"})
    assert result["ok"] is True, result
    assert halted == {"right": "operator"}                  # the selected arm's harness, not the primary's
    assert notes[0] == ("note", "halted motion: operator")


def _every_camera_dict(cfg):
    data = cfg._data
    out = [data["camera"], *(data.get("cameras") or [])]
    for prof in [data["arm"], *(data.get("arms") or [])]:
        view = prof.get("resolved") or {}
        out += [c for c in [view.get("camera"), *(view.get("cameras") or [])] if c]
    return out


class _RecordingArms:
    def __init__(self, monkeypatch):
        from cascade.apps import demo
        from cascade.control.mock_arm import MockArm

        self.made, self.connected = [], []

        class Recording(MockArm):
            def connect(inner):
                self.connected.append(inner)
                return super().connect()

        def make(acfg, kinematics=None, *a, **k):
            arm = Recording(acfg, kinematics=kinematics)
            self.made.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make)


@pytest.mark.parametrize("rig", ["pair", "so101_right alone"])
def test_an_eye_in_hand_camera_on_a_base_pose_rig_refuses_to_build(tmp_path, monkeypatch, rig):
    from cascade.apps.demo import build_runtime

    cfg = _cfg(["so101_left", "so101_right"] if rig == "pair" else ["so101_right"])
    for cam in _every_camera_dict(cfg):
        cam.setdefault("extrinsics", {})["mode"] = "eye_in_hand"
    arms = _RecordingArms(monkeypatch)
    with pytest.raises(ValueError, match="eye_in_hand") as err:
        build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    assert "base_pose" in str(err.value) and repr(str(cfg.camera.get("name"))) in str(err.value)
    assert arms.made == [] and arms.connected == []


def test_an_eye_in_hand_camera_without_a_base_pose_is_not_refused_here(tmp_path, monkeypatch):
    """No base_pose (or the identity): the build is today's, whatever it decides."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    cfg = _cfg(["so101_left", "so101_right"], {"so101_left": IDENTITY, "so101_right": IDENTITY})
    for cam in _every_camera_dict(cfg):
        cam.setdefault("extrinsics", {})["mode"] = "eye_in_hand"
    _RecordingArms(monkeypatch)
    try:
        rt, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    except ValueError as exc:
        assert "base_pose" not in str(exc)
    else:
        shutdown_runtime(rt, arm)


def test_every_arms_grasp_section_is_validated_before_any_arm_is_made(tmp_path, monkeypatch):   # R8
    from cascade.apps.demo import build_runtime

    cfg = _cfg(["so101_left", "so101_right"])
    cfg._data["arms"][1]["resolved"]["grasp"]["open_wait_timeout_s"] = -1
    arms = _RecordingArms(monkeypatch)
    with pytest.raises(ValueError, match="open_wait_timeout_s") as err:
        build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    assert "'so101_right'" in str(err.value)
    assert arms.made == [] and arms.connected == []


# ── 4. the bind window covers the snapshot and the verifier (R3, R4) ──────


def _jaw_frac(safe):
    pos = float(safe.get_state().gripper_pos)
    return (pos - SO101_CLOSED) / (SO101_OPEN - SO101_CLOSED)


def test_the_verifier_reads_the_running_arms_jaws(make_runtime):
    """Pre-flight jaw.log: on main close_gripper(arm=so101_right) was CONFIRMED
    from the LEFT jaw (open, 1.00) and open_gripper(arm=so101_right) at the
    left's closed 0.15."""
    rt = make_runtime(["so101_left", "so101_right"])
    left, right = rt.arm_rig.get("so101_left"), rt.arm_rig.get("so101_right")
    rt.execute("close_gripper", {"arm": "so101_left"})
    opened = rt.execute("open_gripper", {"arm": "so101_right"})
    assert _jaw_frac(left) < 0.3 and _jaw_frac(right) > 0.9
    assert opened["postcondition"]["measured"]["gripper_frac"] == pytest.approx(_jaw_frac(right), abs=1e-3)
    rt.execute("open_gripper", {"arm": "so101_left"})
    closed = rt.execute("close_gripper", {"arm": "so101_right"})
    assert _jaw_frac(left) > 0.9 and _jaw_frac(right) < 0.3
    assert closed["postcondition"]["measured"]["gripper_frac"] == pytest.approx(_jaw_frac(right), abs=1e-3)
    assert "1.00" not in closed["postcondition"]["evidence"]


def test_the_pre_state_snapshot_reads_the_running_arms_jaws(make_runtime, monkeypatch):
    rt = make_runtime(["so101_left", "so101_right"])
    right = rt.arm_rig.get("so101_right")
    rt.execute("close_gripper", {"arm": "so101_left"})
    rt.execute("open_gripper", {"arm": "so101_right"})
    seen = []
    original = rt.effects.snapshot

    def snapshot(*a, **k):
        snap = original(*a, **k)
        seen.append(snap)
        return snap

    monkeypatch.setattr(rt.effects, "snapshot", snapshot)
    rt.execute("open_gripper", {"arm": "so101_right"})
    assert seen[0]["gripper_frac"] == pytest.approx(_jaw_frac(right), abs=1e-3)


def test_the_verifier_table_height_is_the_bound_arms_world_table(make_runtime, monkeypatch):   # R4
    rt = make_runtime(["so101_left", "so101_right"], {"so101_right": RAISED})
    seen = _record_inside(rt, monkeypatch, lambda: (rt.effects.table_z, rt._table_z_world()))
    rt.execute("list_objects", {"arm": "so101_right"})
    rt.execute("list_objects", {})
    right_table_b = float(rt.arm_rig.get("so101_right").harness.limits.table_z)
    left_table_b = float(rt.arm_rig.get("so101_left").harness.limits.table_z)
    assert seen[0] == (pytest.approx(right_table_b + 0.03, abs=1e-12),) * 2
    assert seen[1] == (pytest.approx(left_table_b, abs=1e-12),) * 2
    assert rt.effects.table_z == pytest.approx(left_table_b, abs=1e-12)      # unbound: the primary


# ── 5. #357's running-arm profile agrees with the bound profile (R7) ──────


@pytest.mark.parametrize("arms", [["so101_left", "so101_right"], ["so101_mock", "mock"]],
                         ids=["pair", "mixed"])
def test_the_running_arm_profile_is_the_bound_profile(make_runtime, monkeypatch, arms):
    rt = make_runtime(arms)
    seen = _record_inside(rt, monkeypatch, lambda: (rt._running_arm_profile(), rt.cfg.arm))
    for name in arms:
        rt.execute("list_objects", {"arm": name})
    for name, (profile, bound) in zip(arms, seen, strict=True):
        assert profile is not None, name
        assert profile.get("name") == bound.get("name") == name
        assert profile.get("type") == bound.get("type")


# ── 6. which sections are per arm (D3, R13) ───────────────────────────────


def test_arm_scoped_sections_cover_every_shipped_override_key():
    from cascade.skills.arm_frame import (
        ARM_SCOPED_SECTIONS,
        RIG_GRASP_SERVICE_KEYS,
        RIG_SCOPED_OVERRIDES,
    )

    assert ARM_SCOPED_SECTIONS == ("arm", "grasp", "safety", "relative_motion")
    assert RIG_SCOPED_OVERRIDES == {"workspace_filter", "memory", "perception_loop", "detector",
                                    "object_descriptions", "occupancy"}
    assert RIG_GRASP_SERVICE_KEYS == {"backend", "executor", "graspgenx", "hug", "vla"}
    seen, services = set(), {}
    for path in sorted((REPO / "configs" / "arms").glob("*.yaml")):
        overrides = (yaml.safe_load(path.read_text()) or {}).get("overrides") or {}
        seen |= set(overrides)
        assert set(overrides) <= set(ARM_SCOPED_SECTIONS) | RIG_SCOPED_OVERRIDES, path.name
        overridden = set(overrides.get("grasp") or {}) & RIG_GRASP_SERVICE_KEYS
        if overridden:
            services[path.name] = overridden
    assert {"grasp", "safety", "workspace_filter"} <= seen
    # R13: a non-primary arm's view takes these grasp keys from the TOP level (the
    # services the rig built), so its own override of one is dropped there. Every
    # profile that overrides one is listed: a new one forces a decision.
    assert services == {
        "fr3.yaml": {"backend"},
        "isaac_kitchen_gpu.yaml": {"backend", "graspgenx"},
        "isaac_kitchen_hug.yaml": {"backend", "hug"},
        "piper.yaml": {"graspgenx"},       # grasp.graspgenx.sweep
        "so101.yaml": {"graspgenx"},       # grasp.graspgenx.sweep (so101_left/_right inherit it)
    }


def test_rig_grasp_services_stay_top_level(make_runtime, monkeypatch):            # R13
    from cascade.skills.arm_frame import RIG_GRASP_SERVICE_KEYS

    rt = make_runtime(["so101_mock", "mock"])
    top = rt.cfg.as_dict()["grasp"]
    mock_view = rt.cfg.as_dict()["arms"][1]["resolved"]["grasp"]
    assert mock_view.get("graspgenx") != top.get("graspgenx")      # the case the ruling is about
    seen = _record_inside(rt, monkeypatch, lambda: rt.cfg.as_dict()["grasp"])
    rt.execute("list_objects", {"arm": "mock"})
    bound = seen[0]
    for key in RIG_GRASP_SERVICE_KEYS:
        assert bound.get(key) == top.get(key), key
    for key in set(mock_view) - RIG_GRASP_SERVICE_KEYS:
        assert bound[key] == mock_view[key], key                  # everything else is mock's own


def test_arm_view_cfg_keeps_the_rig_and_takes_the_arms_sections():
    from cascade.skills.arm_frame import arm_view_cfg

    top = Cfg({"arm": {"name": "a"}, "arms": [{"name": "a"}, {"name": "b"}], "grasp": {"drop_zone": [1, 2],
               "backend": "obb"}, "memory": {"x": 1}, "workspace_filter": {"y": 2}})
    acfg = Cfg({"name": "b", "resolved": {"arm": {"name": "b"}, "grasp": {"drop_zone": [3, 4], "backend": "hug",
                                                                        "hug": {"port": 1}},
                                          "memory": {"x": 9}, "safety": {"table_z": -0.01}}})
    view = arm_view_cfg(top, acfg)
    assert view.arm.get("name") == "b" and view.safety.get("table_z") == -0.01
    assert view.grasp.get("drop_zone") == [3, 4]
    assert view.grasp.get("backend") == "obb" and view.grasp.get("hug") is None
    assert view.memory.get("x") == 1 and view.arms == top.arms
    assert view.workspace_filter.get("y") == 2


# ── 7. no thread inside skills sees the wrong frame (Review Focus 1) ──────


def _thread_starts(source: str, filename: str = "<src>") -> list[str]:
    """Calls that run code on another thread: a Thread / Timer (or a subclass
    named so), an executor's submit / map, asyncio.to_thread, a loop's
    run_in_executor, _thread.start_new_thread. The bound frame is
    thread-local, so code started that way would read the primary's."""
    found = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if (name.endswith(("Thread", "Timer"))
                or name in ("to_thread", "run_in_executor", "start_new_thread")
                or (isinstance(func, ast.Attribute) and name in ("submit", "map"))):
            found.append((node.lineno, node.col_offset, f"{filename}:{node.lineno} {ast.unparse(func)}"))
    return [hit for *_, hit in sorted(found)]


def test_the_thread_census_sees_every_way_to_start_a_thread():
    lines = [
        "threading.Thread(target=f).start()",
        "Thread(target=f)",
        "threading.Timer(1.0, f).start()",
        "pool.submit(f)",
        "pool.map(f, xs)",
        "asyncio.to_thread(f)",
        "loop.run_in_executor(None, f)",
        "_thread.start_new_thread(f, ())",
        "_WorkerThread()",
        # not thread starts
        "class _W(threading.Thread): pass",
        "list(map(f, xs))",
        "self.map.add_pose(p)",
        "ThreadContract(x)",
    ]
    source = "\n".join(lines)
    assert [hit.split(" ", 1)[1] for hit in _thread_starts(source)] == [
        "threading.Thread", "Thread", "threading.Timer", "pool.submit", "pool.map", "asyncio.to_thread",
        "loop.run_in_executor", "_thread.start_new_thread", "_WorkerThread"]


def test_no_skill_or_grasp_module_starts_a_thread():
    allowed = {REPO / "src" / "cascade" / "skills" / "mobile_runtime.py"}
    found = []
    for pkg in ("skills", "grasping"):
        for path in sorted((REPO / "src" / "cascade" / pkg).rglob("*.py")):
            if path in allowed:
                continue
            found += _thread_starts(path.read_text(), path.name)
    assert found == []
