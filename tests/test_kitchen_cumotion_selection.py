"""CPU doubles at ordinary planner/stream boundaries; no native admission."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import load_demo_config, load_profile
from cascade.planning.cumotion import CumotionPlanner, MotionPlan
from cascade.planning.runtime import RuntimeMotionPlanner
from cascade.planning.trajectory import TrajectoryProfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kitchen_backends", ROOT / "benchmark/diagnostics/kitchen_backends.py")
backends = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backends
spec.loader.exec_module(backends)
spec = importlib.util.spec_from_file_location(
    "kitchen_selection_campaign", ROOT / "benchmark/diagnostics/kitchen_acceptance.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def configuration(profile="isaac_kitchen_cumotion"):
    return load_demo_config(arm=profile, cameras=["isaac", "isaac_side", "isaac_proof"], llm="mock")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture(name, renderer="ovrtx"):
    state = dict(version=1, backend="isaac", robot_id="robot", producer_epoch="epoch",
                 time_source="physics_loop_monotonic", t=10., q=[0.]*6, joint_convention="asset",
                 gripper_joints=dict(version=1, names=["joint_left", "joint_right"],
                                     position_m=[.03, .03], lower_m=[0., 0.], upper_m=[.04, .04]))
    ref = dict(version=1, product="/camera/"+name,
               source="ovrtx_snapshot" if renderer == "ovrtx" else "rpFabricTime",
               proprioception_sha256=digest(state), snapshot_sha256="a"*64,
               scene_sha256="b"*64, producer_epoch="epoch", snapshot_started_monotonic=10.,
               snapshot_finished_monotonic=10.01, history_physics_step=100,
               history_simulation_time=1., renderer_epoch="renderer-epoch",
               renderer=dict(ordinal=2, step_start_s=1., sensor_start_s=1.5,
                             sensor_end_s=1.5, step_end_s=2.))
    if renderer == "isaac":
        ref.update(numerator=60, denominator=60, render_simulation_time=1.)
    return dict(backend="isaac", source=("127.0.0.1", 8611), camera=name,
                renderer=renderer, t=10., proprioception=state, render_reference=ref)


@pytest.fixture
def subject(tmp_path):
    cfg = configuration()
    cfg.arm._data["bridge_robot_id"] = "robot"
    config = cfg.arm.motion_planner.as_dict()
    for key in ("urdf", "xrdf"):
        path = tmp_path / key
        path.write_text("synthetic " + key)
        config[key] = str(path)
    cfg.arm._data["motion_planner"] = config
    native = object.__new__(CumotionPlanner)
    native._closed = native._poisoned = False
    native.model_sha256 = digest({k: Path(config[k]).read_text() for k in ("urdf", "xrdf")})
    native.scene_sha256 = digest({"base_frame": config["base_frame"], "obstacles": []})
    native.joint_names = tuple(config["joint_names"])
    native.base_frame, native.tool_frame = config["base_frame"], config["tool_frame"]
    planner = object.__new__(RuntimeMotionPlanner)
    planner.config, planner._planner, planner._closed = copy.deepcopy(config), native, False
    calls = []
    def stream(profile, **kwargs):
        calls.append((profile, kwargs))
        return True
    raw = SimpleNamespace(stream_profile=stream)
    frames = [SimpleNamespace(capture=capture(name), rgb=np.zeros((2, 2, 3), np.uint8),
                              depth_m=np.ones((2, 2)), K=np.eye(3))
              for name in ("cam0", "side", "proof")]
    rig = [SimpleNamespace(latest=lambda f=f: f) for f in frames]
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=raw, motion_planner=planner), rig=rig)
    plan = MotionPlan(native.joint_names, (0., 1.), ((0.,)*6, (.1,)*6),
                      ((.1,)*6, (.1,)*6), native.base_frame, native.tool_frame,
                      native.model_sha256, native.scene_sha256, "c"*64)
    profile = TrajectoryProfile.from_curve(lambda t: np.full(6, .1*t), 1., 30., plan)
    state = SimpleNamespace(q=np.zeros(6), physics_clock={"epoch": "epoch", "physics_step": 100})
    return SimpleNamespace(cfg=cfg, runtime=runtime, native=native, planner=planner,
                           frames=frames, profile=profile, state=state, calls=calls)


def admit(s, renderer="ovrtx"):
    return backends.BackendEvidence(s.runtime, s.cfg, arm_profile="isaac_kitchen_cumotion",
                                    renderer=renderer)


def test_profile_preserves_complete_kitchen_contract_and_existing_cumotion_recipe():
    before, after = configuration("isaac_kitchen_gpu"), configuration()
    expected = load_profile("arms", "isaac_cumotion")
    assert after.arm.motion_planner.as_dict() == expected.motion_planner.as_dict()
    assert after.arm.park_q == expected.park_q
    for key in ("grasp", "safety", "occupancy", "perception_loop", "detector", "camera", "cameras"):
        a, b = before.get(key), after.get(key)
        assert (a.as_dict() if hasattr(a, "as_dict") else a) == (b.as_dict() if hasattr(b, "as_dict") else b)
    for key, value in before.arm.as_dict().items():
        if key not in ("name", "resolved"):
            assert after.arm.as_dict()[key] == value
    assert before.arm.get("motion_planner") is None


@pytest.mark.parametrize("profile,renderer", [("isaac", "ovrtx"), ("mock", None),
                                             ("isaac_kitchen_cumotion", None),
                                             ("isaac_kitchen_gpu", "fake")])
def test_invalid_selection_refuses_before_runtime(profile, renderer):
    with pytest.raises(ValueError):
        backends.validate_selection(profile, renderer)


def test_default_selection_requires_no_sdk_or_camera_reads():
    cfg = configuration("isaac_kitchen_gpu")
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=object(), motion_planner=None))
    evidence = backends.BackendEvidence(runtime, cfg, arm_profile="isaac_kitchen_gpu", renderer=None)
    assert evidence.report()["pass"]


def test_original_curve_and_all_safety_callbacks_reach_executor_unchanged(subject):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    callbacks = {key: lambda *args: None for key in ("approve", "preflight", "before_stream", "feedback_guard")}
    assert s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state, **callbacks)
    assert s.calls == [(s.profile, dict(planned_state=s.state, **callbacks))]
    report = evidence.report()
    assert report["pass"] and report["physical_acceptance"] is False
    row = report["curves"][0]
    assert row["plan"]["execution_authorized"] is False
    assert row["executor_called"] and row["completed"]
    assert row["target_count"] == 30
    assert row["planned_state_clock"] == s.state.physics_clock
    report["curves"][0]["plan"]["model_sha256"] = "tampered"
    assert evidence.report()["curves"][0]["plan"]["model_sha256"] == s.native.model_sha256


def test_no_curves_cannot_claim_selected_cumotion_execution(subject):
    assert not admit(subject).report()["pass"]


def test_native_renderer_selection_uses_captured_reference(subject):
    s = subject
    for frame in s.frames:
        frame.capture = capture(frame.capture["camera"], renderer="isaac")
    evidence = admit(s, renderer="isaac")
    evidence.phase = "pick"
    assert s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
    assert evidence.report()["pass"]


@pytest.mark.parametrize("fault", ["token", "rgb", "depth", "jaw"])
def test_native_renderer_label_alone_is_not_capture_evidence(subject, fault):
    s = subject
    for frame in s.frames:
        frame.capture = capture(frame.capture["camera"], renderer="isaac")
    frame = s.frames[0]
    if fault == "token": frame.capture["render_reference"]["denominator"] = 0
    elif fault == "rgb": frame.rgb = None
    elif fault == "depth": frame.depth_m = None
    elif fault == "jaw": frame.capture["proprioception"].pop("gripper_joints")
    with pytest.raises(RuntimeError): admit(s, renderer="isaac")
    assert not s.calls


def test_missing_native_candidate_cannot_send_or_gain_credit(subject):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    with pytest.raises(RuntimeError, match="lacks its native candidate"):
        s.runtime.arm.raw.stream_profile(replace(s.profile, plan=None), planned_state=s.state)
    assert not s.calls and not evidence.report()["pass"]


def test_renderer_error_stays_in_evidence_after_later_healthy_capture(subject):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    original = copy.deepcopy(s.frames[0].capture)
    s.frames[0].capture = None
    with pytest.raises(RuntimeError): evidence.check()
    s.frames[0].capture = original
    assert s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
    assert not evidence.report()["pass"]


@pytest.mark.parametrize("fault", ["owner", "raw_arm", "model", "scene"])
def test_changed_owner_binding_before_stream_rejected(subject, fault):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    entry = s.runtime.arm.raw.stream_profile
    if fault == "owner": s.planner._planner = copy.copy(s.native)
    elif fault == "raw_arm": s.runtime.arm.raw = object()
    elif fault == "model": s.native.model_sha256 = "0"*64
    elif fault == "scene": s.native.scene_sha256 = "0"*64
    with pytest.raises(RuntimeError): entry(s.profile, planned_state=s.state)
    assert not s.calls and not evidence.report()["pass"]


@pytest.mark.parametrize("fault", ["absent", "wrong_renderer", "wrong_source", "wrong_robot",
                                  "wrong_epoch", "missing_reference", "invalid_digest", "missing_camera",
                                  "missing_timestamp"])
def test_camera_evidence_failure_refuses_before_any_stream(subject, fault):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    c = s.frames[0].capture
    if fault == "absent": s.frames[0].capture = None
    elif fault == "wrong_renderer": c["renderer"] = "isaac"
    elif fault == "wrong_source": c["source"] = ("127.0.0.1", 9999)
    elif fault == "wrong_robot": c["proprioception"]["robot_id"] = "other"
    elif fault == "wrong_epoch": c["render_reference"]["producer_epoch"] = "other"
    elif fault == "missing_reference": c["render_reference"] = None
    elif fault == "invalid_digest": c["render_reference"]["proprioception_sha256"] = "0"*64
    elif fault == "missing_camera": s.runtime.rig.pop()
    elif fault == "missing_timestamp": c.pop("t")
    with pytest.raises((RuntimeError, ValueError)):
        s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
    assert not s.calls and not evidence.report()["pass"]


@pytest.mark.parametrize("field,value", [("model_sha256", "0"*64), ("scene_sha256", "0"*64),
                                        ("sdk_version", "9.9"), ("request_sha256", "invalid"),
                                        ("joint_names", ("wrong",)*6), ("base_frame", "wrong")])
def test_wrong_candidate_refuses_before_executor(subject, field, value):
    s = subject; evidence = admit(s); evidence.phase = "pick"
    profile = replace(s.profile, plan=replace(s.profile.plan, **{field: value}))
    with pytest.raises(RuntimeError, match="candidate"):
        s.runtime.arm.raw.stream_profile(profile, planned_state=s.state)
    assert not s.calls and not evidence.report()["pass"]


@pytest.mark.parametrize("fault", ["absent", "fake_owner", "closed", "poisoned", "changed_model"])
def test_config_flag_cannot_substitute_effective_planner(subject, fault):
    s = subject
    if fault == "absent": s.runtime.arm.motion_planner = None
    elif fault == "fake_owner": s.planner._planner = SimpleNamespace()
    elif fault == "closed": s.planner._closed = True
    elif fault == "poisoned": s.native._poisoned = True
    elif fault == "changed_model": s.native.model_sha256 = "0"*64
    with pytest.raises(RuntimeError): admit(s)
    assert not s.calls


@pytest.mark.parametrize("outcome", [False, RuntimeError("ordinary executor refused")])
def test_failed_stream_never_becomes_backend_acceptance(subject, outcome):
    s = subject
    def stream(*args, **kwargs):
        if isinstance(outcome, Exception): raise outcome
        return outcome
    s.runtime.arm.raw.stream_profile = stream
    evidence = admit(s); evidence.phase = "pick"
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match="ordinary executor refused"):
            s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state)
    else:
        assert s.runtime.arm.raw.stream_profile(s.profile, planned_state=s.state) is False
    assert not evidence.report()["pass"]


def test_phase_refusal_does_not_call_skill_or_settle(subject):
    s = subject; evidence = admit(s)
    s.frames[0].capture = None
    s.runtime.execute = lambda *args: pytest.fail("skill called after backend refusal")
    observer = SimpleNamespace(mark=lambda name: None,
        settle=lambda **kw: pytest.fail("settle called after refusal"))
    receipt = {"pass": False, "errors": []}
    campaign.phase(observer, s.runtime, "pick_and_place", {}, "pick", receipt, evidence)
    assert receipt["errors"] and "pick_result" not in receipt


def test_cli_requires_renderer_for_new_profile_before_creating_output(tmp_path):
    with pytest.raises(SystemExit) as exc:
        campaign.main(["--port", "8692", "--engine", "physx", "--arm-profile", "isaac_kitchen_cumotion",
                       "--output", str(tmp_path/"never-created")])
    assert exc.value.code == 2 and not (tmp_path/"never-created").exists()


def test_selected_manifest_adds_actual_urdf_xrdf_and_helper():
    manifests = [campaign.frozen_sources(ROOT/"demo/scene/kitchen_config.json", profile)
                 for profile in backends.ARM_PROFILES]
    config = load_profile("arms", "isaac_kitchen_cumotion").motion_planner
    for key in ("urdf", "xrdf"):
        path = Path(config.get(key)); relative = str(path.relative_to(ROOT))
        assert manifests[1][relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(manifests[0]).issubset(manifests[1])
    assert "benchmark/diagnostics/kitchen_backends.py" in manifests[0]
