"""Held08 retained geometry: an aiming estimate must never authorize opening."""
import copy
from pathlib import Path
from types import SimpleNamespace as S

import numpy as np
import pytest

from cascade.skills.held_observation import measure
from cascade.skills.runtime import SkillRuntime
from cascade.types import RobotState, SkillError
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.control.isaac_arm import IsaacArm


SOURCE = ("127.0.0.1", 8697)


def clock(step=20, epoch="retained-08"):
    return dict(version=1, engine="physx", clock="SimulationManager", epoch=epoch,
        robot_id="/robot", sim_time=step/120, physics_step=step,
        physics_dt_s=1/120, source=SOURCE)


def runtime():
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.held_object = "orange"
    rt._held_det_label = "orange fruit"
    rt._held_offset = np.array([-.0111206318, .0358323786, -.1029361486])
    rt._held_observation_floor = clock(10)
    rt._held_observation_floor_q = np.array([.19, .08, .13, 0, 0, 0])
    rt._object_pose = None
    rt._grip_open = 1.
    rt.cfg = S(arm={"bridge_host": SOURCE[0], "bridge_port": SOURCE[1],
        "bridge_robot_id": "/robot", "joint_signs": [-1]*6},
        grasp={"slip_drop_m": .06}, safety={"table_z": 0.})
    state = RobotState(q=np.array([.19, .08, .13, 0, 0, 0]), physics_clock=clock())
    harness = SafetyHarness(SafetyLimits(np.full(3, -2.), np.full(3, 2.)))
    backend = IsaacArm(rt.cfg.arm)
    rt.arm = S(get_state=lambda **kw: state, harness=harness, raw=backend)
    def fk(q):
        t = np.eye(4); t[:3, 3] = q[:3]; return t
    rt.kin = S(fk=fk)
    rt.observe = lambda: (_ for _ in ()).throw(RuntimeError("camera unavailable"))
    rt.detector = rt.extrinsics = None
    rt._localization_workspace_bounds = lambda: None
    rt.memory = S(add=lambda *a, **kw: None)
    return rt


def atomic(position=(.19, .08, .12), step=22):
    return dict(version=1, source=SOURCE, physics_clock=clock(step),
        q_asset=[-.19, -.08, -.13, 0., 0., 0.],
        joint_convention="asset", joint_indices=list(range(6)),
        joint_names=[f"joint{i}" for i in range(1, 7)],
        position_m=list(position), resolved_name="orange", resolved_path="/World_Props/orange",
        server_monotonic=100., pose_frame="world", meters_per_unit=1.,
        base_position_world=[0., 0., 0.], base_orientation_wxyz=[1., 0., 0., 0.])


def no_localize(*a, **kw):
    raise RuntimeError("unavailable image")


def test_retained_negative_cache_is_aiming_only():
    rt = runtime()
    measured = measure(rt, no_localize)
    np.testing.assert_allclose(measured.offset, rt._held_offset)
    assert measured.channel == "cached_aim" and not measured.release_authority


def test_legacy_pose_far_below_tcp_is_not_release_authority():
    rt = runtime(); rt._object_pose = lambda label: [.19, .08, .02]
    result = measure(rt, no_localize)
    assert result.offset[2] < -.06 and result.channel == "legacy_pose"
    assert not result.release_authority


def test_atomic_retained_lift_geometry_and_exact_asset_signs():
    rt = runtime(); sample = atomic()
    rt._object_pose = S(observation=lambda labels: copy.deepcopy(sample))
    result = measure(rt, no_localize)
    assert result.release_authority and result.channel == "atomic_physics"
    np.testing.assert_allclose(result.offset, [0, 0, -.01], atol=1e-12)


def test_retained_08_lift_with_real_kinematics_does_not_report_a_drop():
    from cascade.control.kinematics import Kinematics
    rt = runtime()
    model = Path(__file__).resolve().parents[1] / "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf"
    rt.kin = Kinematics(str(model), "gripper_end", 6, [-1]*6)
    q_asset = np.array([.1090253517, -1.320039749, -1.1606731415,
                       1.409919858, -.3135627508, -1.020200968])
    q = -q_asset
    rt.arm.get_state = lambda **kw: RobotState(q=q, physics_clock=clock(21958))
    rt._held_observation_floor = clock(21956)
    rt._held_observation_floor_q = q.copy()
    sample = atomic(position=(.1895442456, .0847463161, .11995345355), step=21960)
    sample["q_asset"] = q_asset.tolist()
    rt._object_pose = S(observation=lambda labels: sample)
    result = measure(rt, no_localize)
    assert result.release_authority
    assert result.offset[2] == pytest.approx(-.0094820135, abs=1e-8)
    assert result.offset[2] > -.06


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(source=("wrong", 8697)),
    lambda s: s["physics_clock"].update(epoch="new"),
    lambda s: s["physics_clock"].update(robot_id="/other"),
    lambda s: s.update(q_asset=[float("nan")]*6),
    lambda s: s.update(q_asset=[False]*6),
    lambda s: s.update(position_m=[float("nan"), 0, 0]),
    lambda s: s.update(resolved_name="lemon", resolved_path="/World_Props/lemon"),
    lambda s: s.update(resolved_path="/World_Props/lemon"),
    lambda s: s.update(meters_per_unit=.01),
    lambda s: s.update(base_orientation_wxyz=[0., 0., 0., 0.]),
    lambda s: s.update(server_monotonic=float("nan")),
    lambda s: s.update(q_asset=[0.]*8),
    lambda s: s.update(joint_convention="local"),
    lambda s: s["joint_names"].reverse(),
    lambda s: s.update(joint_indices=[True, 1, 2, 3, 4, 5]),
    lambda s: s.update(joint_indices=[0]*6),
])
def test_invalid_atomic_measurements_do_not_authorize_open(mutation):
    rt = runtime(); sample = atomic(position=(.19, .08, .02)); mutation(sample)
    rt._object_pose = S(observation=lambda labels: sample)
    assert not measure(rt, no_localize).release_authority


def test_atomic_same_physics_step_with_different_q_is_rejected():
    rt = runtime(); sample = atomic(step=20); sample["q_asset"][2] = -.14
    rt._object_pose = S(observation=lambda labels: sample)
    result = measure(rt, no_localize)
    assert not result.release_authority
    assert any("same physical step" in s for s in result.evidence["rejected"])


def test_base_translation_and_rotation_are_applied_to_world_pose():
    rt = runtime(); sample = atomic()
    sample.update(base_position_world=[1., 2., 3.],
        base_orientation_wxyz=[np.sqrt(.5), 0., 0., np.sqrt(.5)],
        position_m=[1.-.08, 2.+.19, 3.+.12])
    rt._object_pose = S(observation=lambda labels: sample)
    result = measure(rt, no_localize)
    assert result.release_authority
    np.testing.assert_allclose(result.offset, [0, 0, -.01], atol=1e-12)


def image(rt, *, step=22, epoch="retained-08", target="orange"):
    q = np.array([.19, .08, .04, 0, 0, 0])
    mask = np.ones((2, 2), bool)
    frame = S(capture={"source": SOURCE, "t": 100.,
        "proprioception": {"robot_id": "/robot", "producer_epoch": epoch},
        "render_reference": {"version": 1, "source": "rpFabricTime", "producer_epoch": epoch,
            "history_physics_step": step, "history_simulation_time": step/120,
            "snapshot_started_monotonic": 100.}},
        prop_masks={"/World_Props/"+target: mask})
    if isinstance(rt.arm, SafeArm):
        rt.arm.raw.state_from_frame = lambda f: RobotState(q=q)
    else:
        rt.arm.raw = S(state_from_frame=lambda f: RobotState(q=q))
    rt.observe = lambda: frame
    fix = S(position=np.array([.19, .08, .03]), detection=S(mask=mask.copy()))
    return frame, fix


def test_historical_lift_image_uses_its_own_tcp_and_cannot_open():
    rt = runtime(); frame, fix = image(rt, step=18)
    result = measure(rt, lambda *a, **kw: fix)
    np.testing.assert_allclose(result.offset, [0, 0, -.01])
    assert not result.release_authority  # It was captured before this check.


@pytest.mark.parametrize("kind", ["previous_lift", "old_epoch", "neighbor", "empty", "partial"])
def test_image_requires_new_capture_and_exact_measured_identity(kind):
    rt = runtime(); frame, fix = image(rt,
        step=8 if kind == "previous_lift" else 22,
        epoch="other" if kind == "old_epoch" else "retained-08",
        target="lemon" if kind == "neighbor" else "orange")
    if kind == "empty": fix.detection.mask[:] = False
    if kind == "partial": fix.detection.mask[0, 0] = False
    assert not measure(rt, lambda *a, **kw: fix).release_authority


def test_image_same_step_inconsistent_joints_is_rejected():
    rt = runtime(); frame, fix = image(rt, step=20)
    result = measure(rt, lambda *a, **kw: fix)
    assert not result.release_authority
    assert result.invalidated and "same physical step" in result.evidence["rejected"][0]


def test_fresh_identified_image_can_observe_genuine_drop():
    rt = runtime(); frame, fix = image(rt)
    fix.position[2] = -.05  # 9 cm below that capture's TCP, not the current TCP.
    result = measure(rt, lambda *a, **kw: fix)
    assert result.release_authority and result.offset[2] == pytest.approx(-.09)


def test_genuine_atomic_drop_keeps_threshold_and_generation_for_open():
    rt = runtime(); rt._object_pose = S(observation=lambda labels: atomic(position=(.19, .08, .02)))
    opened = []
    rt.arm.set_gripper = lambda pos, **kw: opened.append((pos, kw))
    with pytest.raises(SkillError, match="slipped out of the gripper"):
        rt.skill_place_at(.2, -.12)
    assert opened == [(1., {"effort": .6, "_halt_generation": 0})]
    assert rt.held_object is None


def test_missing_channels_keep_held_without_opening_before_planning():
    rt = runtime(); opened = []
    rt.arm.set_gripper = lambda *a, **kw: opened.append((a, kw))
    rt.kin.ik = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("planning boundary"))
    with pytest.raises(RuntimeError, match="planning boundary"):
        rt.skill_place_at(.2, -.12)
    assert not opened and rt.held_object == "orange"


def composite(rt):
    rt._reconcile_held = lambda: None
    rt.cfg.grasp.update(max_pick_attempts=3, persist_seconds=120.)
    events = []
    rt.skill_move_home = lambda **kw: events.append("home")
    rt._reobserve = lambda: events.append("reobserve")
    raw = rt.arm.raw
    raw.get_state = rt.arm.get_state
    raw.set_gripper = lambda *a, **kw: events.append("open")
    rt.arm = SafeArm(raw, rt.arm.harness)
    rt.kin.ik = lambda *a, **kw: events.append("plan")
    return events


@pytest.mark.parametrize("where", ["during_observation", "before_open"])
def test_pick_and_place_halt_resume_never_opens_retries_or_goes_home(where):
    rt = runtime(); events = composite(rt)
    def stop_resume():
        rt.arm.harness.halt("cancel this attempt")
        rt.arm.harness.clear_halt()
    def observe(labels):
        if where == "during_observation": stop_resume()
        return atomic(position=(.19, .08, .02))
    rt._object_pose = S(observation=observe)
    if where == "before_open":
        actual = rt.arm.set_gripper
        def late_open(*a, **kw):
            stop_resume(); return actual(*a, **kw)
        rt.arm.set_gripper = late_open
    result = rt.skill_pick_and_place("orange")
    assert result["ok"] is False and result["stage"] == "held_observation"
    assert result["place_attempts"] == 1 and result["home_skipped"] is True
    assert rt.held_object == "orange" and events == []


@pytest.mark.parametrize("kind", ["state_epoch", "state_regression", "state_dt", "state_engine",
                                  "truth_source", "truth_epoch", "image_epoch", "arm_swap", "joint_signs"])
def test_pick_and_place_changed_identity_never_retries_or_moves(kind):
    rt = runtime(); events = composite(rt)
    if kind == "state_epoch":
        rt.arm.get_state().physics_clock["epoch"] = "changed"
    elif kind == "state_regression":
        rt.arm.get_state().physics_clock.update(physics_step=8, sim_time=8/120)
    elif kind == "state_dt":
        rt.arm.get_state().physics_clock["physics_dt_s"] = 1/60
    elif kind == "state_engine":
        rt.arm.get_state().physics_clock.update(engine="newton", clock="newton_stage")
    elif kind == "image_epoch":
        image(rt, epoch="changed")
    else:
        def observe(labels):
            sample = atomic()
            if kind == "truth_source": sample["source"] = ("other", 8697)
            if kind == "truth_epoch": sample["physics_clock"]["epoch"] = "changed"
            if kind == "arm_swap": rt.arm = SafeArm(rt.arm.raw, rt.arm.harness)
            if kind == "joint_signs": rt.cfg.arm["joint_signs"] = [1]*6
            return sample
        rt._object_pose = S(observation=observe)
    result = rt.skill_pick_and_place("orange")
    assert result["stage"] == "held_observation" and result["place_attempts"] == 1
    assert not result["ok"] and rt.held_object == "orange" and events == []


def test_observation_result_records_authority_and_exact_geometry_without_extra_reads():
    rt = runtime(); reads = []
    def observe(labels):
        reads.append(labels); return atomic(position=(.19, .08, .02))
    rt._object_pose = S(observation=observe)
    notes = []
    rt.memory = S(add=lambda *a, **kw: notes.append(a[2] if len(a) > 2 else kw.get("data")))
    rt.arm.set_gripper = lambda *a, **kw: None
    with pytest.raises(SkillError, match="slipped out"):
        rt.skill_place_at(.2, -.12)
    assert len(reads) == 1
    decision = next(n["held_observation"] for n in notes if n)
    assert decision["channel"] == "atomic_physics" and decision["release_authority"]
    assert decision["offset_m"][2] == pytest.approx(-.11)
    assert decision["evidence"]["resolved_path"] == "/World_Props/orange"


def test_elapsed_bound_rejects_a_slow_atomic_read_and_isaac_state_is_one_second(monkeypatch):
    rt = runtime(); rt.cfg.arm["type"] = "isaac"
    calls = []; original = rt.arm.get_state
    rt.arm.get_state = lambda **kw: (calls.append(kw), original())[1]
    elapsed = [0.]
    monkeypatch.setattr("cascade.skills.held_observation.time.monotonic", lambda: elapsed[0])
    def slow(labels):
        elapsed[0] = 2.001; return atomic(position=(.19, .08, .02))
    rt._object_pose = S(observation=slow)
    result = measure(rt, no_localize)
    assert not result.release_authority and calls == [{"timeout_s": 1.}]
