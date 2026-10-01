"""Atomic contact loss terminates transport, never authorizes a recovery."""
import copy
from types import SimpleNamespace as S

import numpy as np
import pytest

from cascade.control.isaac_arm import IsaacArm
from cascade.control.simulation_motion import SimulationMotion
from cascade.skills import carry_attachment as carry
from cascade.skills.runtime import SkillRuntime
from cascade.types import RobotState, SafetyViolation
from test_held_observation import runtime, composite, clock, SOURCE
from test_isaac_simulation_motion import Sim

PATH = "/World_Props/object-not-a-hardcoded-color"


def jaws():
    return dict(version=1, names=["joint_left", "joint_right"], position_m=[.02, .021],
                lower_m=[0., 0.], upper_m=[.05, .05])


def add_attachment(state, signs):
    c = state.physics_clock
    state.gripper_joints = jaws()
    state.attachment = dict(version=1, backend="isaac", source=c["source"], robot_id=c["robot_id"],
        producer_epoch=c["epoch"], physics_step=c["physics_step"], sim_time=c["sim_time"],
        joint_convention="asset", q=(state.q * signs).tolist(), gripper_joints=jaws(),
        tracking=True, error=None, paths=[PATH], channel="completed_update_bilateral_contact",
        sensor_channel="physx_gpu_contact_tensor", history_reference=[c["physics_step"], 120])
    return state


def episode(rt, state):
    c = state.physics_clock
    rt.arm.harness.occupancy = S(scene_reset_generation=0, tracks_payload=True)
    return dict(halt_generation=rt.arm.harness._halt_generation, expected_paths=(PATH,),
        barrier={"integrated": [dict(camera=name, source=c["source"], robot_id=c["robot_id"],
                                     clock="physics_loop_monotonic", producer_epoch=c["epoch"])
                                for name in ["front", "side", "proof"]]})


def armed():
    rt = runtime()
    state = add_attachment(rt.arm.get_state(), rt.arm.raw._signs)
    carry.arm(rt, episode(rt, state), state)
    return rt, state


def next_state(state):
    state = copy.deepcopy(state)
    state.physics_clock["physics_step"] += 1
    state.physics_clock["sim_time"] += state.physics_clock["physics_dt_s"]
    state.attachment.update(physics_step=state.physics_clock["physics_step"],
                            sim_time=state.physics_clock["sim_time"])
    return state


def test_aligned_feedback_never_uses_lagging_camera_or_map_contact_cache():
    rt, state = armed()
    rt.arm.harness.occupancy._contact_paths = []  # Delayed map is not the state authority.
    second = next_state(state)
    second.q[0] += .01
    second.attachment["q"] = (second.q * rt.arm.raw._signs).tolist()
    carry.observe(rt, second)
    assert carry.active(rt).status == "attached"
    # Retained evidence owns arrays/dicts independently of transport buffers.
    second.attachment["paths"].clear()
    assert carry.active(rt).last["paths"] == [PATH]


@pytest.mark.parametrize("change", [
    lambda s: setattr(s, "attachment", None),
    lambda s: s.attachment.update(tracking=False),
    lambda s: s.attachment.update(error="sensor unavailable"),
    lambda s: s.attachment.update(paths=None),
    lambda s: s.attachment.update(paths=[PATH, PATH]),
    lambda s: s.attachment.update(source=("other", 1)),
    lambda s: s.attachment.update(robot_id="/other"),
    lambda s: s.attachment.update(producer_epoch="new"),
    lambda s: s.physics_clock.update(epoch="new"),
    lambda s: s.attachment.update(physics_step=s.attachment["physics_step"] - 1),
    lambda s: s.attachment.update(joint_convention="local"),
    lambda s: s.attachment.update(q=[0.]),
    lambda s: s.attachment.update(sensor_channel="other"),
    lambda s: s.attachment.update(gripper_joints=jaws() | {"position_m": [.02, .022]}),
])
def test_unknown_or_mismatched_metadata_latches_without_followup_action(change):
    rt, state = armed(); bad = next_state(state); change(bad)
    with pytest.raises(carry.AttachmentInvalid): carry.observe(rt, bad)
    with pytest.raises(carry.AttachmentInvalid): carry.observe(rt, next_state(bad if bad.attachment else state))
    assert rt.held_object == "orange"
    result = rt.skill_reset_scene()
    assert result["stage"] == "carry_attachment" and not result["grip_verified"]
    assert result["home_skipped"] and result["retry_allowed"] is False


def test_same_step_cannot_change_q_or_jaws_even_with_matching_attachment_copy():
    for field in ("q", "jaws"):
        rt, state = armed(); bad = copy.deepcopy(state)
        if field == "q":
            bad.q[0] += .001; bad.attachment["q"] = (bad.q * rt.arm.raw._signs).tolist()
        else:
            bad.gripper_joints["position_m"][0] += .001
            bad.attachment["gripper_joints"] = copy.deepcopy(bad.gripper_joints)
        with pytest.raises(carry.AttachmentInvalid, match="without a physical step"):
            carry.observe(rt, bad)


@pytest.mark.parametrize("change", [
    {"names": ["joint_right", "joint_left"]}, {"upper_m": [0., .05]},
    {"position_m": [True, .02]}, {"upper_m": [.051, .05]},
])
def test_jaw_schema_and_retained_limits_are_authority_not_only_two_values(change):
    rt, state = armed(); bad = next_state(state)
    bad.gripper_joints.update(change)
    bad.attachment["gripper_joints"] = copy.deepcopy(bad.gripper_joints)
    with pytest.raises(carry.AttachmentInvalid): carry.observe(rt, bad)


@pytest.mark.parametrize("error", [TypeError("callback internal"), SafetyViolation("cancelled")])
def test_safe_arm_stretch_read_validates_existing_callback_before_begin_or_stream(error):
    from cascade.safety.harness import SafeArm
    events = []
    raw = S(get_state=lambda: events.append("read") or RobotState(q=np.zeros(2)),
            stream_to=lambda *a, **kw: events.append("stream"))
    harness = S(check_release_episode=lambda **kw: None, begin_motion=lambda **kw: events.append("begin"))
    arm = SafeArm(raw, harness)
    def veto(state):
        events.append("feedback"); raise error
    with pytest.raises(SafetyViolation, match="no unguarded retry|cancelled") as caught:
        arm.move_joints(np.ones(2), feedback_guard=veto)
    if isinstance(error, TypeError): assert caught.value.__cause__ is error
    assert events == ["read", "feedback"]


@pytest.mark.parametrize("where", ["before_first_target", "during_carry", "settle"])
def test_real_executor_veto_prevents_every_target_after_loss(monkeypatch, where):
    sim = Sim(monkeypatch)
    sim._signs = np.ones(2)
    sim.attachment_feedback_version = 1
    rt = runtime(); rt.cfg.arm.update(type="isaac", bridge_host="fake", bridge_port=1,
                                     bridge_robot_id="/robot", joint_signs=[1, 1])
    rt.arm.raw = sim
    original = sim.get_state
    loss = []
    def read(**kw):
        state = add_attachment(original(**kw), sim._signs)
        count = len(sim.sent)
        if ((where == "before_first_target" and armed_flag[0])
                or (where == "during_carry" and count >= 3)
                or (where == "settle" and count >= 10)):
            state.attachment["paths"] = []; loss.append(count)
        return state
    armed_flag = [False]
    sim.get_state = read
    initial = sim.get_state(timeout_s=.1)
    carry.arm(rt, episode(rt, initial), initial)
    armed_flag[0] = True
    guard = carry.active(rt)
    with pytest.raises(carry.AttachmentInvalid, match="lost"):
        SimulationMotion(sim, feedback_guard=guard.observe, before_stream=guard.guard).stream(
            np.array([.1, -.1]), .2, 50., .045, 1., None)
    assert len(sim.sent) == loss[0] == {"before_first_target": 0, "during_carry": 3, "settle": 10}[where]
    assert guard.status == "lost"


def test_real_decoder_checks_asset_q_before_signs_and_binds_source():
    rt, state = armed(); backend = rt.arm.raw
    wire = dict(q=state.attachment["q"], dq=[0.] * 6, gripper_joints=state.gripper_joints,
                attachment=copy.deepcopy(state.attachment), physics_clock=state.physics_clock)
    wire["attachment"]["source"] = ("untrusted", 1)
    decoded = backend._decode_state(wire)
    assert decoded.attachment["source"] == SOURCE
    np.testing.assert_array_equal(decoded.q, state.q)
    carry.observe(rt, decoded)
    wire["attachment"]["q"][0] += .01
    assert decoded.attachment["q"] != wire["attachment"]["q"]
    bad = backend._decode_state(wire)
    with pytest.raises(carry.AttachmentInvalid): carry.observe(rt, bad)


def test_non_payload_or_legacy_backend_never_requires_isaac_metadata():
    rt = runtime(); carry.arm(rt, None, RobotState(q=np.zeros(6)))
    assert carry.active(rt) is None
    rt.arm.raw = S()
    carry.arm(rt, {"unused": True}, RobotState(q=np.zeros(6)))
    assert carry.active(rt) is None


def test_rejected_initial_barrier_keeps_failure_on_the_next_call():
    rt = runtime(); events = composite(rt)
    state = add_attachment(rt.arm.get_state(), rt.arm.raw._signs)
    invalid = episode(rt, state); invalid["barrier"]["integrated"] = []
    with pytest.raises(carry.AttachmentInvalid): carry.arm(rt, invalid, state)
    for _ in range(2):
        result = rt.skill_pick_and_place("orange")
        assert result["stage"] == "carry_attachment" and not result["grip_verified"]
        assert rt.skill_reset_scene()["home_skipped"]
    assert events == []


def test_malformed_initial_barrier_and_late_transport_error_also_latch():
    rt = runtime()
    state = add_attachment(rt.arm.get_state(), rt.arm.raw._signs)
    invalid = episode(rt, state); invalid["barrier"]["integrated"] = None
    with pytest.raises(carry.AttachmentInvalid): carry.arm(rt, invalid, state)
    with pytest.raises(carry.AttachmentInvalid): carry.arm(rt, episode(rt, state), state)
    rt, state = armed()
    rt.arm.get_state = lambda: (_ for _ in ()).throw(RuntimeError("state deadline"))
    with pytest.raises(carry.AttachmentInvalid): rt._gripper_width_frac()
    assert rt.skill_pick_and_place("orange")["stage"] == "carry_attachment"


def test_first_post_close_read_timeout_is_latched_before_any_motion():
    rt = runtime(); events = composite(rt)
    state = add_attachment(rt.arm.get_state(), rt.arm.raw._signs)
    barrier = episode(rt, state)
    rt.arm.raw.get_state = lambda **kw: (_ for _ in ()).throw(RuntimeError("post-close state timeout"))
    with pytest.raises(carry.AttachmentInvalid, match="post-close state timeout"):
        carry.arm(rt, barrier)
    assert carry.active(rt).failure and carry.active(rt).last is None
    assert rt.skill_pick_and_place("orange")["grip_verified"] is False
    assert rt.skill_reset_scene()["home_skipped"] and not events


@pytest.mark.parametrize("key", ["before_stream", "feedback_guard", "_halt_generation"])
def test_callbacks_and_cancellation_tokens_cannot_be_silently_replaced(key):
    rt, state = armed()
    rt.arm.move_joints = lambda *_a, **_k: pytest.fail("ambiguous callback must not run")
    with pytest.raises(carry.AttachmentInvalid):
        carry.move(rt, state.q, **{key: 1 if key == "_halt_generation" else lambda *_: None})


@pytest.mark.parametrize("leg", [0, 1, 2])
def test_each_native_place_leg_keeps_terminal_type_no_later_leg_or_open(leg):
    from test_pre_carry_lift import enabled
    rt, moves, opens, _ = enabled()
    base, initial = armed()
    state = copy.deepcopy(initial)
    state.q = rt.arm.get_state().q.copy()
    add_attachment(state, base.arm.raw._signs)
    # Keep real IK/placement control flow, use an explicit state/stream double.
    rt.arm.raw = base.arm.raw
    rt.cfg.arm.update(base.cfg.arm)
    rt.arm.harness._halt_generation = 0
    rt.arm.harness._check_halt_generation = base.arm.harness._check_halt_generation
    rt.arm.get_state = lambda **kw: state
    carry.arm(rt, episode(rt, state), state)
    def stream(q, **kw):
        kw["before_stream"]()
        sample = next_state(state)
        if len(moves) == leg: sample.attachment["paths"] = []
        kw["feedback_guard"](sample)
        moves.append(q.copy()); state.q = q.copy()
        state.physics_clock = sample.physics_clock
        add_attachment(state, rt.arm.raw._signs)
        return True
    rt.arm.move_joints = stream
    with pytest.raises(carry.AttachmentInvalid): rt.skill_place_at(.14, -.27)
    assert len(moves) == leg and not opens and rt.held_object == "tomato can"


@pytest.mark.parametrize("failure", ["lift", "air_heuristic", "none"])
def test_native_grasp_arms_before_initial_lift_and_preserves_terminal_marker(monkeypatch, failure):
    from test_grasp_evidence import runtime as grasp_runtime
    from cascade.skills import contact_episode
    rt, calls, fix, frame = grasp_runtime(monkeypatch)
    raw = rt.arm.raw
    original_read = raw.get_state
    after_arm_targets = []
    original_target = raw._client.set_joints
    def send(q, **kw):
        original_target(q, **kw)
        if carry.active(rt) is not None: after_arm_targets.append(q.copy())
    raw._client.set_joints = send
    def read(**kw):
        state = add_attachment(original_read(**kw), raw._signs)
        if failure == "lift" and after_arm_targets: state.attachment["paths"] = []
        return state
    raw.get_state = read
    # Isolate the already separately tested three-camera barrier. The real
    # runtime, SafeArm, Isaac executor, initial lift and jaw heuristic run.
    monkeypatch.setattr(contact_episode, "begin", lambda r, q: episode(r, read()))
    monkeypatch.setattr(contact_episode, "wait_geometry", lambda r, e: e["barrier"])
    monkeypatch.setattr(contact_episode, "finish", lambda *a, **kw: None)
    if failure == "air_heuristic":
        width = rt._gripper_width_frac
        rt._gripper_width_frac = lambda: 0. if carry.active(rt) is not None else width()
    if failure == "none":
        result = rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
        assert result["held"] == "orange" and result["grip_verified"]
        assert carry.active(rt).status == "attached" and len(after_arm_targets) > 1
        return
    with pytest.raises(carry.AttachmentInvalid): rt.skill_grasp_object("orange", _fix=fix, _frame=frame)
    count = len(calls)
    assert rt._held_provisional[0] == "orange"
    assert rt.skill_pick_and_place("orange")["grip_verified"] is False
    assert len(calls) == count  # No re-observation/open/home/retry after latch.
    if failure == "lift": assert len(after_arm_targets) == 1


@pytest.mark.parametrize("change", ["halt_resume", "map", "scene_reset", "backend", "signs"])
def test_retained_identity_or_cancellation_never_rearms(change):
    rt, state = armed()
    if change == "halt_resume": rt.arm.harness.halt("stop"); rt.arm.harness.clear_halt()
    elif change == "map": rt.arm.harness.occupancy = S(scene_reset_generation=0)
    elif change == "scene_reset": rt.arm.harness.occupancy.scene_reset_generation += 1
    elif change == "backend": rt.arm.raw = S()
    else: rt.cfg.arm["joint_signs"] = [1] * 6
    with pytest.raises(carry.AttachmentInvalid): carry.observe(rt, state)


def test_composite_stops_before_retry_home_or_open_and_does_not_claim_hold():
    rt = runtime(); events = composite(rt)
    state = add_attachment(rt.arm.get_state(), rt.arm.raw._signs)
    carry.arm(rt, episode(rt, state), state)
    bad = next_state(state); bad.attachment["paths"] = []
    rt.arm.raw.get_state = lambda **kw: bad
    result = rt.skill_pick_and_place("orange")
    assert result["stage"] == "carry_attachment" and result["place_attempts"] == 1
    assert result["grip_verified"] is False and result["retry_allowed"] is False
    assert "holding" not in result and rt.held_object == "orange" and not events
    assert rt.skill_pick_and_place("orange")["place_attempts"] == 0
    with pytest.raises(carry.AttachmentInvalid): SkillRuntime.skill_move_home(rt)
    assert rt.skill_reset_scene()["stage"] == "carry_attachment" and not events
