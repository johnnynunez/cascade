"""CPU software fixtures ONLY: scripted states are NOT physical receipts.

Even fixtures labelled measurement_kind=physics only exercise the verifier's
schema/decision branches. Physical class separation is a separate live gate.
"""
from __future__ import annotations

import math
import threading
import time

import pytest

from cascade.control.mobile_base import BaseState


def limits(**updates):
    # Wall-clock slack for CI schedulers; synthetic time windows and motion
    # tolerances below remain unchanged. Socket fixtures supply their own RTT budget.
    result = {
        "sample_interval_s": 0.002, "read_timeout_s": 0.04,
        "max_wall_duration_s": 3., "settle_timeout_s": 0.3,
        "settle_window_s": 0.04, "min_motion_samples": 2,
        "min_settle_samples": 3, "max_samples": 180, "max_history": 3,
        "max_state_age_s": 0.2, "max_sample_gap_s": 0.1,
        "max_position_abs_m": 10., "max_linear_speed_m_s": 1.,
        "max_angular_speed_rad_s": 5., "min_height_m": 0.1,
        "max_tilt_rad": 0.6, "min_progress_ratio": 0.5,
        "max_progress_ratio": 1.5, "translation_tolerance_m": 0.001,
        "rotation_tolerance_rad": 0.03, "max_lateral_drift_m": 0.003,
        "max_heading_drift_rad": 0.04, "stop_linear_speed_m_s": 0.01,
        "stop_angular_speed_rad_s": 0.02, "stop_drift_m": 0.001,
        "stop_drift_rad": 0.01,
    }
    result.update(updates)
    return result


def fixture_support_contract():
    return dict(version=1, model_identity_sha256="e" * 64,
                robot_shapes=["/Fixture/Robot/left_sole", "/Fixture/Robot/right_sole", "/Fixture/Robot/body"],
                foot_shapes=["/Fixture/Robot/left_sole", "/Fixture/Robot/right_sole"],
                ground_shapes=["/Fixture/Ground"], gravity_world_m_s2=[0., 0., -9.81])


def fixture_support(n, *, sim_time_s=None, epoch="fixture-epoch", model_identity_sha256="e" * 64):
    # Synthetic unit force for schema/decision testing; NOT a physical receipt.
    return dict(version=1, status="known", reason="", epoch=epoch, step=n,
                sim_time_s=n * .02 if sim_time_s is None else sim_time_s,
                model_identity_sha256=model_identity_sha256,
                contacts=[dict(shape_a_id=0, shape_b_id=1,
                               shape_a="/Fixture/Ground", shape_b="/Fixture/Robot/left_sole",
                               force_on_b_world_n=[0., 0., 1.], normal_force_n=1.,
                               normal_a_to_b_world=[0., 0., 1.], point_world_m=[0., 0., 0.])])


def state(n, **updates):
    values = dict(
        robot_id="synthetic-microduck", source="scripted-software-fixture",
        epoch="fixture-epoch", step=n, sim_time_s=n * 0.02,
        received_monotonic_s=time.monotonic(), producer_age_s=0.,
        position_world=(0., 0., 0.3), orientation_wxyz=(1., 0., 0., 0.),
        linear_velocity_world=(0., 0., 0.), angular_velocity_body=(0., 0., 0.),
        joint_names=("fixture_joint",), joint_positions=(0.,), joint_velocities=(0.,),
        controller_status="ready", generation=0, contacts=("fixture-foot",),
        fallen=False, latched=False, measurement_kind="physics", model_identity_sha256="e" * 64,
    )
    values.update(updates)
    if "support" not in values:
        values["support"] = fixture_support(values["step"], sim_time_s=values["sim_time_s"],
            epoch=values["epoch"], model_identity_sha256=values["model_identity_sha256"])
    return BaseState(**values)


class ScriptedReader:
    def __init__(self, make_state=None):
        self.make_state = make_state or (lambda n: state(n))
        self.calls = 0
        self.closed = False
        self.condition = threading.Condition()

    def __call__(self):
        with self.condition:
            self.calls += 1
            value = self.make_state(self.calls)
            self.condition.notify_all()
            return value

    def wait(self, calls):
        with self.condition:
            assert self.condition.wait_for(lambda: self.calls >= calls, timeout=2.)

    def close(self):
        self.closed = True


def checker_for(reader, **updates):
    from cascade.agent.base_effects import BasePostconditionChecker
    return BasePostconditionChecker(reader, limits=limits(**updates), support_contract=fixture_support_contract())


def test_pre_admission_travel_never_confirms_an_inert_command():
    admitted, stopped = threading.Event(), threading.Event()
    reader = ScriptedReader(lambda n: state(
        n, position_world=(min(n - 1, 5) * .002, 0., .3),
        generation=2 if stopped.is_set() else 1 if admitted.is_set() else 0))
    checker = checker_for(reader)
    try:
        token = checker.begin("walk_velocity", dict(vx=.1, vy=0., wz=0., duration_s=.1))
        reader.wait(8)
        start = reader.calls * .02
        admitted.set()
        reader.wait(reader.calls + 8)
        stopped.set()
        result = {"execution_ok": True,
                  "ack": {"ok": True, "accepted": True, "latched": False, "generation": 1,
                          "robot_id": "synthetic-microduck", "source": "scripted-software-fixture",
                          "epoch": "fixture-epoch", "model_identity_sha256": "e" * 64, "start_sim_time_s": start, "end_sim_time_s": start + .1},
                  "stop_ack": {"ok": True, "latched": False, "generation": 2,
                               "robot_id": "synthetic-microduck", "source": "scripted-software-fixture",
                               "epoch": "fixture-epoch", "model_identity_sha256": "e" * 64}}
        verdict = checker.finish(token, result)
        assert verdict["status"] == "refuted", verdict["reason"]
        assert verdict["metrics"]["body_displacement_m"] == [0., 0.]
        assert any(s["state"]["position_world"][0] == 0. for s in verdict["evidence"]["samples"])
    finally:
        checker.close()


def test_actor_measured_and_ack_cannot_replace_missing_truth():
    reader = ScriptedReader(lambda n: None)
    checker = checker_for(reader)
    try:
        token = checker.begin("walk_velocity", dict(vx=0.1, vy=0., wz=0., duration_s=0.1))
        actor = {"ok": True, "execution_ok": True,
                 "measured": {"before": state(1).as_dict(),
                              "after": state(2, position_world=(1., 0., 0.3)).as_dict()},
                 "ack": {"ok": True}}
        verdict = checker.finish(token, actor)
        assert verdict["status"] == "unverified"
        assert verdict["evidence"]["samples"] == []
        assert verdict["metrics"] == {}
        assert checker.history() == [verdict]
        assert "unverified" in checker.digest()
    finally:
        checker.close()
    assert reader.closed


def fixture_motion_receipt():
    # Explicit synthetic protocol schedule for the scripted decision fixtures.
    identity = dict(robot_id="synthetic-microduck", source="scripted-software-fixture", epoch="fixture-epoch", model_identity_sha256="e" * 64)
    return {"execution_ok": True,
            "ack": dict(ok=True, accepted=True, latched=False, generation=1,
                        start_sim_time_s=.02, end_sim_time_s=.12, **identity),
            "stop_ack": dict(ok=True, latched=False, generation=2, **identity)}


def bind_fixture_motion(reader):
    from dataclasses import replace
    make = reader.make_state
    def bound(n):
        value = make(n)
        if isinstance(value, BaseState) and value.generation == 0:
            return replace(value, generation=0 if n == 1 else 1 if n <= 8 else 2)
        return value  # preserve deliberately corrupted/custom fences
    reader.make_state = bound


def run_window(reader, skill="walk_velocity", args=None, result=None, **updates):
    args = args if args is not None else dict(vx=0.1, vy=0., wz=0., duration_s=0.1)
    if skill in {"walk_velocity", "turn"}:
        bind_fixture_motion(reader)
        result = {**fixture_motion_receipt(), **(result or {})}
    checker = checker_for(reader, **updates)
    try:
        token = checker.begin(skill, args)
        reader.wait(8)
        return checker.finish(token, result or {"execution_ok": True})
    finally:
        checker.close()


@pytest.mark.parametrize("scale,expected", [(0., "refuted"), (1., "confirmed"),
                                              (-1., "refuted"), (3., "refuted")])
def test_walk_requires_measured_progress_of_correct_sign_and_size(scale, expected):
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * 0.002 * scale, 0., 0.3)))
    verdict = run_window(reader)
    assert verdict["status"] == expected
    assert verdict["metrics"]["body_displacement_m"][0] == pytest.approx(0.01 * scale)
    assert {item["phase"] for item in verdict["evidence"]["samples"]} == {"before", "during", "after"}
    assert verdict["evidence"]["provenance"]["source"] == "scripted-software-fixture"


@pytest.mark.parametrize("sign", [-1., 1.])
def test_body_frame_walk_is_not_world_x_or_actor_q(sign):
    # Facing +Y: body forward/reverse is world +Y/-Y, not world X.
    reader = ScriptedReader(lambda n: state(
        n, position_world=(0., min(n - 1, 5) * 0.002 * sign, 0.3),
        orientation_wxyz=(math.sqrt(0.5), 0., 0., math.sqrt(0.5))))
    verdict = run_window(reader, args=dict(vx=0.1 * sign, vy=0., wz=0., duration_s=0.1))
    assert verdict["status"] == "confirmed"


@pytest.mark.parametrize("change,reason", [
    ({"fallen": True}, "fallen"),
    ({"position_world": (0., 0., 0.295)}, "height"),
    ({"orientation_wxyz": (math.cos(0.305), math.sin(0.305), 0., 0.)}, "tilt"),
    ({"controller_status": "fault"}, "controller"),
])
def test_walk_refutes_observed_fall_even_if_final_state_recovers(change, reason):
    # Cross posture thresholds slowly enough to respect numeric plausibility.
    def make(n):
        values = {"position_world": (min(n - 1, 5) * 0.002, 0., 0.3),
                  "orientation_wxyz": (math.cos(0.295), math.sin(0.295), 0., 0.)}
        if n == 4:
            values.update(change)
        return state(n, **values)
    verdict = run_window(ScriptedReader(make), min_height_m=0.297)
    assert verdict["status"] == "refuted"
    assert reason in verdict["reason"]


@pytest.mark.parametrize("change,reason", [
    ({"source": "different-source"}, "source"),
    ({"robot_id": "different-robot"}, "robot_id"),
    ({"epoch": "reset"}, "epoch"),
    ({"measurement_kind": "hardware"}, "measurement_kind"),
    ({"producer_age_s": 10.}, "stale"),
    ({"received_monotonic_s": 0.}, "stale"),
    ({"received_monotonic_s": 1e12}, "future"),
    ({"sim_time_s": 0.}, "clock"),
    ({"step": 0}, "clock"),
    ({"sim_time_s": 3.}, "gap"),
    ({"generation": 2}, "generation"),
    ({"position_world": (1e8, 0., 0.3)}, "plausibility"),
    ({"linear_velocity_world": (1e4, 0., 0.)}, "plausibility"),
    ({"angular_velocity_body": (0., 0., 1e4)}, "plausibility"),
])
def test_one_corrupted_sample_poisoning_provenance_cannot_be_hidden_by_final_pose(change, reason):
    def make(n):
        values = {"position_world": (min(n - 1, 5) * 0.002, 0., 0.3)}
        if n == 4:
            values.update(change)
        return state(n, **values)
    verdict = run_window(ScriptedReader(make))
    assert verdict["status"] == "unverified"
    assert reason in verdict["reason"]


@pytest.mark.parametrize("kind", ["kinematic_mock", "physics"])
def test_mock_never_confirms_while_explicit_physics_fixture_exercises_positive_branch(kind):
    reader = ScriptedReader(lambda n: state(n, measurement_kind=kind,
                                           position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    verdict = run_window(reader)
    assert verdict["status"] == ("unverified" if kind == "kinematic_mock" else "confirmed")


@pytest.mark.parametrize("invalid", [None, {"ok": True}, "raise", "zero_quaternion", "nan"])
def test_missing_or_invalid_truth_in_middle_is_unknown_not_normalized(invalid):
    def make(n):
        value = state(n, position_world=(min(n - 1, 5) * 0.002, 0., 0.3))
        if n != 4:
            return value
        if invalid == "raise":
            raise RuntimeError("synthetic sensor outage")
        if invalid == "zero_quaternion":
            object.__setattr__(value, "orientation_wxyz", (0., 0., 0., 0.))
            return value
        if invalid == "nan":
            object.__setattr__(value, "position_world", (float("nan"), 0., 0.3))
            return value
        return invalid
    verdict = run_window(ScriptedReader(make))
    assert verdict["status"] == "unverified"


def test_duplicate_steps_add_no_evidence_weight_and_no_motion_verdict():
    reader = ScriptedReader(lambda n: state(1))
    verdict = run_window(reader)
    assert verdict["status"] == "unverified"
    assert len(verdict["evidence"]["samples"]) == 1
    assert verdict["evidence"]["duplicates"] > 2


def test_same_step_with_different_pose_is_conflicting_not_a_completed_step():
    reader = ScriptedReader(lambda n: state(1, position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    verdict = run_window(reader)
    assert verdict["status"] == "unverified"
    assert "conflict" in verdict["reason"]


def test_a_motion_without_observations_during_execution_is_unknown():
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    checker = checker_for(reader, sample_interval_s=0.03)
    try:
        token = checker.begin("walk_velocity", dict(vx=0.1, vy=0., wz=0., duration_s=0.1))
        verdict = checker.finish(token, {"execution_ok": True})
        assert verdict["status"] == "unverified"
        assert "during" in verdict["reason"]
    finally:
        checker.close()


@pytest.mark.parametrize("angle", [-0.2, 0.2])
@pytest.mark.parametrize("scale,expected", [(0., "refuted"), (-1., "refuted"),
                                              (1., "confirmed"), (2., "refuted")])
def test_turn_uses_signed_unwrapped_measured_yaw_not_integrated_wz(angle, scale, expected):
    # Cross the -pi/pi cut in either direction.
    start = math.copysign(3.1, angle)
    def make(n):
        yaw = start + min(n - 1, 5) / 5 * angle * scale
        return state(n, orientation_wxyz=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)))
    verdict = run_window(ScriptedReader(make), "turn", {"angle_rad": angle})
    assert verdict["status"] == expected
    assert verdict["metrics"]["yaw_change_rad"] == pytest.approx(angle * scale)


@pytest.mark.parametrize("skill", ["stop", "stop_navigation", "emergency_stop"])
@pytest.mark.parametrize("velocity,expected", [(0., "confirmed"), (0.1, "refuted")])
def test_stop_ack_never_substitutes_for_advancing_low_velocity_window(skill, velocity, expected):
    reader = ScriptedReader(lambda n: state(n, linear_velocity_world=(velocity, 0., 0.)))
    verdict = run_window(reader, skill, {}, {"ok": True})
    assert verdict["status"] == expected
    assert verdict["metrics"]["settle_samples"] >= 3
    assert verdict["metrics"]["settle_sim_duration_s"] >= limits()["settle_window_s"]


def test_settling_window_starts_after_result_not_at_begin():
    # Long quiet period before finish cannot cover a moving post-command base.
    moving = threading.Event()
    reader = ScriptedReader(lambda n: state(n, linear_velocity_world=(0.1 if moving.is_set() else 0., 0., 0.)))
    checker = checker_for(reader)
    try:
        token = checker.begin("stop_navigation", {})
        reader.wait(10)
        moving.set()
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "refuted"
        assert "settle" in verdict["reason"]
    finally:
        checker.close()


def test_walking_success_with_residual_rotation_is_refuted():
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * 0.002, 0., 0.3),
                                           angular_velocity_body=(0., 0., 0.1)))
    verdict = run_window(reader)
    assert verdict["status"] == "refuted"
    assert "settle" in verdict["reason"]


def test_repeated_snapshot_after_stop_cannot_prove_rest():
    reader = ScriptedReader(lambda n: state(min(n, 8)))
    checker = checker_for(reader)
    try:
        token = checker.begin("stop_navigation", {})
        reader.wait(8)
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert "after" in verdict["reason"]
    finally:
        checker.close()


def test_stop_detects_drift_even_when_velocity_field_claims_zero():
    reader = ScriptedReader(lambda n: state(n, position_world=(n * 0.002, 0., 0.3)))
    verdict = run_window(reader, "stop_navigation", {})
    assert verdict["status"] == "refuted"
    assert "settle" in verdict["reason"]


def test_old_producer_snapshots_received_after_finish_do_not_prove_settling():
    reader = ScriptedReader(lambda n: state(n, producer_age_s=0.15))
    # The age must exceed this test's whole after-outcome window.
    verdict = run_window(reader, "stop_navigation", {}, settle_timeout_s=.1)
    assert verdict["status"] == "unverified"
    assert "after" in verdict["reason"]


def test_failed_execution_is_preserved_despite_final_favorable_physics():
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    verdict = run_window(reader, result={"execution_ok": False, "error": "controller timed out"})
    assert verdict["status"] != "confirmed"
    assert verdict["execution_failed"] is True
    assert "execution" in verdict["reason"]


@pytest.mark.parametrize("key", list(limits()))
@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), -1., 0., "1"])
def test_explicit_limits_reject_missing_or_nonpositive_nonfinite_values(key, bad):
    values = limits()
    if bad is None:
        del values[key]
    else:
        values[key] = bad
    from cascade.agent.base_effects import BasePostconditionChecker
    with pytest.raises(ValueError):
        BasePostconditionChecker(lambda: None, limits=values)


@pytest.mark.parametrize("changes", [
    {"extra": 1.}, {"min_motion_samples": 1}, {"min_settle_samples": 1},
    {"min_motion_samples": 2.5}, {"max_samples": 5}, {"sample_interval_s": 0.2},
    {"max_wall_duration_s": 0.05}, {"min_progress_ratio": 1.1},
    {"max_progress_ratio": 0.9}, {"max_tilt_rad": math.pi},
    {"rotation_tolerance_rad": math.pi}, {"max_heading_drift_rad": math.pi},
    {"stop_linear_speed_m_s": 2.}, {"stop_angular_speed_rad_s": 6.},
])
def test_limits_fail_closed_on_inconsistent_relations(changes):
    with pytest.raises(ValueError):
        checker_for(lambda: None, **changes)


@pytest.mark.parametrize("skill,args", [
    ("turn", {"angle_rad": float("nan")}), ("turn", {"angle_rad": True}),
    ("turn", {"angle_rad": "0.2"}), ("turn", {"angle_rad": 10.}),
    ("walk_velocity", {"vx": float("nan"), "vy": 0., "wz": 0., "duration_s": 0.1}),
    ("walk_velocity", {"vx": True, "vy": 0., "wz": 0., "duration_s": 0.1}),
    ("walk_velocity", {"vx": 0., "vy": 0., "wz": 0., "duration_s": -1.}),
    ("walk_velocity", {"vx": 1e308, "vy": 0., "wz": 0., "duration_s": 1e308}),
])
def test_invalid_intent_can_never_turn_into_a_positive_comparison(skill, args):
    verdict = run_window(ScriptedReader(), skill, args)
    assert verdict["status"] == "unverified"
    assert "intent" in verdict["reason"]


def test_history_limits_and_begin_args_are_defensive_snapshots():
    reader = ScriptedReader()
    configured = limits(max_history=2)
    from cascade.agent.base_effects import BasePostconditionChecker
    checker = BasePostconditionChecker(reader, limits=configured, support_contract=fixture_support_contract())
    try:
        for index in range(4):
            args = {"base": f"fixture-{index}"}
            token = checker.begin("stop_navigation", args)
            args["base"] = "mutated-after-begin"
            configured["stop_drift_m"] = 999.
            verdict = checker.finish(token, {"ok": True})
            assert verdict["status"] == "confirmed"
            assert verdict["args"]["base"] == f"fixture-{index}"
            assert verdict["limits"]["stop_drift_m"] == 0.001
            verdict["reason"] = "mutated-result"
            returned = checker.history()
            returned[0]["evidence"]["samples"].clear()
        assert len(checker.history()) == 2
        assert checker.history()[0]["evidence"]["samples"]
        assert all(v["reason"] != "mutated-result" for v in checker.history())
        assert "confirmed=2" in checker.digest()
    finally:
        checker.close()


def test_sample_quota_and_deadline_are_not_completion_evidence():
    reader = ScriptedReader()
    checker = checker_for(reader, max_samples=8)
    try:
        token = checker.begin("stop_navigation", {})
        reader.wait(8)
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert "sample_limit" in verdict["reason"]
        assert verdict["evidence"]["attempts"] <= 8
        assert len(verdict["evidence"]["samples"]) <= 8
    finally:
        checker.close()


def test_lifecycle_refuses_overlap_replay_and_post_close_reads():
    reader = ScriptedReader()
    checker = checker_for(reader)
    token = checker.begin("stop_navigation", {})
    with pytest.raises(RuntimeError):
        checker.begin("turn", {"angle_rad": 0.1})
    with pytest.raises(ValueError):
        checker.finish("not-a-token", {})
    checker.finish(token, {"ok": True})
    with pytest.raises(ValueError):
        checker.finish(token, {"ok": True})
    checker.close()
    checker.close()
    with pytest.raises(RuntimeError):
        checker.begin("stop_navigation", {})
    assert not any(t.name == "base-postcondition-sampler" for t in threading.enumerate())


def test_stuck_reader_is_quarantined_without_unbounded_thread_spawning():
    released, entered, completed = (threading.Event() for _ in range(3))
    def reader():
        entered.set()
        released.wait()  # Only test cleanup can release this genuinely stuck read.
        return None
    checker = checker_for(reader)
    results, errors = [], []
    def observe():
        try:
            token = checker.begin("stop_navigation", {})
            results.append(checker.finish(token, {"ok": True}))
        except BaseException as exc:
            errors.append(exc)
        finally:
            completed.set()
    worker = threading.Thread(target=observe, name="stuck-reader-test-driver")
    worker.start()
    try:
        assert entered.wait(2.)
        # Completion must precede releasing the reader; a short stopwatch bound
        # measured thread startup/CI scheduling rather than this property.
        assert completed.wait(2.), "verifier waited for the blocked reader"
        assert not errors, errors
        assert not released.is_set()
        verdict, = results
        assert verdict["status"] == "unverified"
        assert "timeout" in verdict["reason"]
        with pytest.raises(RuntimeError):
            checker.begin("stop_navigation", {})
        assert sum(t.name == "base-postcondition-sampler" for t in threading.enumerate()) == 1
    finally:
        released.set()
        checker.close()
        worker.join(2.)
        assert not worker.is_alive()
    assert not any(t.name == "base-postcondition-sampler" for t in threading.enumerate())


@pytest.fixture
def truth_server():
    """Real loopback TCP, scripted JSON fixture; NEVER a simulator."""
    import json
    import socketserver
    from cascade.sim.mobile_identity import support_contract_digest
    profile = dict(robot_id="synthetic-microduck", source="scripted-software-fixture",
                   engine="physx", device="cpu", asset_sha256="a" * 64,
                   policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=fixture_support_contract(), bridge_host="127.0.0.1", timeout_s=0.15)
    hello = dict(ok=True, protocol=1, kind="microduck", epoch="fixture-epoch",
                 capabilities=["state", "velocity", "stop"], measurement_kind="physics",
                 physics_dt=0.005, policy_dt=0.02,
                 support_contract_sha256=support_contract_digest(profile["support_contract"]),
                 **{k: profile[k] for k in ("robot_id", "source", "engine", "device", "asset_sha256", "policy_sha256", "model_identity_sha256")})
    requests = []
    payload = {"hello": hello, "state": {"ok": True, "state": state(1).as_dict()}}

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                message = json.loads(line)
                requests.append(message)
                response = payload[message["op"]]
                if callable(response):
                    response = response()
                self.wfile.write(json.dumps(response).encode() + b"\n")

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True

    server = Server(("127.0.0.1", 0), Handler)
    profile["bridge_port"] = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.005})
    thread.start()
    try:
        yield profile, payload, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.)
        assert not thread.is_alive()


@pytest.mark.parametrize("key,bad", [("protocol", True), ("kind", "rebot"),
                                    ("robot_id", "different"), ("source", "different"),
                                    ("engine", "newton"), ("device", "cuda:0"),
                                    ("asset_sha256", "c" * 64), ("policy_sha256", "d" * 64),
                                    ("measurement_kind", "kinematic_mock"), ("epoch", ""),
                                    ("capabilities", []), ("physics_dt", 0.1)])
def test_truth_reader_refuses_wrong_hello_without_control_ownership(truth_server, key, bad):
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, requests = truth_server
    payload["hello"][key] = bad
    reader = BaseTruthReader(profile)
    try:
        assert requests == []  # constructing a verifier never dials or owns control
        assert reader() is None
        assert reader.last_error
        assert requests == [{"op": "hello", "role": "reader"}]
    finally:
        reader.close()
    reader.close()
    assert reader() is None
    assert len(requests) == 1


@pytest.mark.parametrize("key,bad", [("bridge_port", None), ("bridge_port", True),
                                    ("bridge_host", "8.8.8.8"), ("timeout_s", 0.),
                                    ("timeout_s", float("nan")), ("asset_sha256", "A" * 64),
                                    ("robot_id", " padded ")])
def test_truth_reader_rejects_incomplete_or_unsafe_profile_without_network(key, bad):
    from cascade.sim.base_truth import BaseTruthReader
    profile = dict(robot_id="fixture", source="fixture", engine="physx", device="cpu",
                   asset_sha256="a" * 64, policy_sha256="b" * 64,
                   bridge_host="127.0.0.1", bridge_port=43210, timeout_s=0.1)
    if bad is None:
        profile.pop(key)
    else:
        profile[key] = bad
    with pytest.raises(ValueError):
        BaseTruthReader(profile)


def test_truth_reader_uses_actual_transport_parser_and_only_hello_state(truth_server, monkeypatch):
    import cascade.control.isaac_base as transport
    from cascade.sim.base_truth import BaseTruthReader
    def prohibited(*args, **kwargs):
        raise AssertionError("truth reader tried to instantiate an actuator")
    monkeypatch.setattr(transport, "IsaacBase", prohibited)
    profile, payload, requests = truth_server
    payload["state"]["state"]["received_monotonic_s"] = 1e12  # remote clock is NOT local
    payload["state"]["state"]["producer_age_s"] = 0.001
    reader = BaseTruthReader(profile)
    try:
        start = time.monotonic()
        first = reader()
        assert first is not None, reader.last_error
        assert start <= first.received_monotonic_s <= time.monotonic()
        assert first.producer_age_s >= 0.001
        assert reader().step == first.step  # observing never manufactures a tick
        assert requests == [{"op": "hello", "role": "reader"}, {"op": "state"}, {"op": "state"}]
        assert reader.last_error is None
    finally:
        reader.close()
    assert all(r["op"] in ("hello", "state") and "owner" not in r for r in requests)


@pytest.mark.parametrize("key,bad", [("source", "other"), ("robot_id", "other"),
                                    ("epoch", "new-epoch"), ("measurement_kind", "kinematic_mock"),
                                    ("orientation_wxyz", [0., 0., 0., 0.]),
                                    ("position_world", [float("nan"), 0., 0.3])])
def test_truth_reader_rejects_malformed_or_misattributed_state(truth_server, key, bad):
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, requests = truth_server
    payload["state"]["state"][key] = bad
    reader = BaseTruthReader(profile)
    try:
        assert reader() is None
        assert reader.last_error
        assert [r["op"] for r in requests] == ["hello", "state"]
    finally:
        reader.close()


def test_truth_reader_epoch_binding_survives_read_failure_and_reconnect(truth_server):
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, requests = truth_server
    reader = BaseTruthReader(profile)
    try:
        assert reader() is not None
        payload["state"] = {"ok": False, "error": "synthetic outage"}
        assert reader() is None
        payload["hello"]["epoch"] = "reset-epoch"
        payload["state"] = {"ok": True, "state": state(1, epoch="reset-epoch").as_dict()}
        assert reader() is None
        assert "epoch" in reader.last_error
        assert requests[-1] == {"op": "hello", "role": "reader"}
    finally:
        reader.close()


def test_truth_reader_timeout_is_bounded_and_closes_without_commands(truth_server):
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, requests = truth_server
    entered = threading.Event()
    release = threading.Event()
    def delayed():
        entered.set()
        release.wait(1.)
        return {"ok": True, "state": state(1).as_dict()}
    payload["state"] = delayed
    profile["timeout_s"] = 0.025
    reader = BaseTruthReader(profile)
    try:
        start = time.monotonic()
        assert reader() is None
        assert entered.is_set()
        assert time.monotonic() - start < 0.3
        assert "deadline" in reader.last_error or "timed out" in reader.last_error
    finally:
        release.set()
        reader.close()
    assert [r["op"] for r in requests] == ["hello", "state"]


def test_truth_reader_rejects_truthy_nonboolean_hello_ok(truth_server):
    from cascade.sim.base_truth import BaseTruthReader
    profile, payload, _ = truth_server
    payload["hello"]["ok"] = "true"
    reader = BaseTruthReader(profile)
    try:
        assert reader() is None
        assert "ok" in reader.last_error
    finally:
        reader.close()


@pytest.mark.parametrize("change", [
    {"position_world": (0.5, 0., 0.3)},
    {"orientation_wxyz": (math.cos(0.6), 0., 0., math.sin(0.6))},
])
def test_impossible_interstep_pose_jump_is_unknown_not_a_motion_or_fall(change):
    reader = ScriptedReader(lambda n: state(n, **(change if n == 4 else {})))
    verdict = run_window(reader)
    assert verdict["status"] == "unverified"
    assert "plausibility" in verdict["reason"]


def test_insufficient_after_window_does_not_refute_motion_from_partial_history():
    reader = ScriptedReader(lambda n: state(min(n, 8)))
    bind_fixture_motion(reader)
    checker = checker_for(reader)
    try:
        token = checker.begin("walk_velocity", dict(vx=0.1, vy=0., wz=0., duration_s=0.1))
        reader.wait(8)
        verdict = checker.finish(token, fixture_motion_receipt())
        assert verdict["status"] == "unverified"
        assert "after" in verdict["reason"]
    finally:
        checker.close()


def test_zero_twist_must_stay_balanced_not_wander_out_and_back():
    def make(n):
        x = 0.004 * min(max(n - 1, 0), max(7 - n, 0))
        return state(n, position_world=(x, 0., 0.3))
    verdict = run_window(ScriptedReader(make), args=dict(vx=0., vy=0., wz=0., duration_s=0.1))
    assert verdict["status"] == "refuted"
    assert "drift" in verdict["reason"]


def test_zero_twist_does_not_confirm_if_velocity_nonzero_during_balancing_then_stops():
    reader = ScriptedReader(lambda n: state(n, linear_velocity_world=(0.1 if n < 7 else 0., 0., 0.)))
    verdict = run_window(reader, args=dict(vx=0., vy=0., wz=0., duration_s=0.1))
    assert verdict["status"] == "refuted"
    assert "balance" in verdict["reason"]


def test_expired_wall_window_cannot_use_new_after_outcome_snapshots():
    reader = ScriptedReader()
    checker = checker_for(reader, max_wall_duration_s=0.1, settle_timeout_s=0.1, max_samples=180)
    try:
        token = checker.begin("stop_navigation", {})
        time.sleep(0.13)
        verdict = checker.finish(token, {"ok": True})
        assert verdict["status"] == "unverified"
        assert "wall" in verdict["reason"]
    finally:
        checker.close()


def test_close_during_sampling_is_not_a_favorable_final_state():
    reader = ScriptedReader()
    checker = checker_for(reader)
    token = checker.begin("stop_navigation", {})
    reader.wait(3)
    checker.close()
    verdict = checker.finish(token, {"ok": True})
    assert verdict["status"] == "unverified"
    assert "closed" in verdict["reason"]
    assert not any(t.name == "base-postcondition-sampler" for t in threading.enumerate())


def test_checker_crash_is_a_receipt_and_does_not_wedge_the_next_token(monkeypatch):
    from cascade.agent.base_effects import BasePostconditionChecker
    reader = ScriptedReader()
    checker = checker_for(reader)
    def crash(*args):
        raise RuntimeError("synthetic analysis bug")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(BasePostconditionChecker, "_measure", crash)
            token = checker.begin("stop_navigation", {})
            verdict = checker.finish(token, {"ok": True})
            assert verdict["status"] == "unverified"
            assert "verifier_error" in verdict["reason"]
        token = checker.begin("stop_navigation", {})
        assert checker.finish(token, {"ok": True})["status"] == "confirmed"
    finally:
        checker.close()


@pytest.mark.parametrize("scale,expected", [(0., "refuted"), (1., "confirmed")])
def test_independent_checker_through_actual_mobile_rpc_with_scripted_publisher(scale, expected):
    """Integration software fixture: publisher scripts values, NOT physics."""
    from cascade.agent.base_effects import BasePostconditionChecker
    from cascade.sim.base_truth import BaseTruthReader
    from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
    identity = dict(robot_id="synthetic-microduck", source="scripted-software-fixture",
                    engine="physx", device="cpu", asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=fixture_support_contract())
    controller = MobileBridgeController(**identity, max_linear_speed=0.2, max_angular_speed=1.,
                                        max_duration_s=1., lease_s=1., max_state_age_s=1.)
    def publish(n):
        support = fixture_support(n, sim_time_s=n * .005, epoch=controller.hello()["epoch"])
        controller.publish(dict(step=n, sim_time=n * 0.005,
                                position=[min(max(n - 1, 0), 20) * 0.0005 * scale, 0., 0.3],
                                orientation_wxyz=[1., 0., 0., 0.], linear_velocity=[0., 0., 0.],
                                angular_velocity=[0., 0., 0.], q=[0.] * 14, dq=[0.] * 14,
                                joint_names=[f"fixture_joint_{j}" for j in range(14)],
                                contacts=["fixture-foot"], fallen=False, support=support))
    publish(1)
    server = MobileBridgeServer(controller, port=0)
    server.start()
    reader = BaseTruthReader({**identity, "bridge_host": server.address[0],
                              "bridge_port": server.address[1], "timeout_s": 0.25})
    checker = BasePostconditionChecker(reader, limits=limits(
        read_timeout_s=.25, settle_timeout_s=.8, max_samples=600),
        support_contract=fixture_support_contract())
    release = threading.Event()
    finished_script = threading.Event()
    def producer():
        n = 1
        while not release.wait(0.004):
            n += 1
            controller.control_at(n * .005)
            publish(n)
            if n >= 25:
                finished_script.set()
    worker = threading.Thread(target=producer, name="synthetic-completed-step-publisher")
    try:
        first = reader()
        assert first is not None, reader.last_error
        assert reader().step == first.step  # actual server state is passive
        token = checker.begin("walk_velocity", dict(vx=0.1, vy=0., wz=0., duration_s=0.1))
        ack = controller.command_velocity(dict(vx=.1, vy=0., wz=0., duration_s=.1,
            robot_id=identity["robot_id"], source=identity["source"], epoch=controller.hello()["epoch"],
            generation=0, model_identity_sha256=identity["model_identity_sha256"], owner="fixture-owner", command_id="fixture-command"))
        worker.start()
        assert finished_script.wait(2.)
        stop_ack = controller.stop(latch=False)
        verdict = checker.finish(token, {"execution_ok": True, "ack": ack, "stop_ack": stop_ack})
        assert verdict["status"] == expected, verdict["reason"]
        assert verdict["evidence"]["provenance"]["epoch"] == controller.hello()["epoch"]
        # The independent sampler need not observe both published endpoints.
        # Check the known scripted geometry over its ACTUAL admitted interval;
        # never credit an unobserved prefix/suffix or infer it from the target.
        interval = verdict["evidence"]["effect_interval"]
        samples = {row["state"]["step"]: row["state"]
                   for row in verdict["evidence"]["samples"]}
        baseline, last = (samples[interval[key]] for key in ("baseline_step", "last_step"))
        assert (ack["start_sim_time_s"] <= baseline["sim_time_s"]
                < last["sim_time_s"] <= ack["end_sim_time_s"])
        expected_positions = [min(max(s["step"] - 1, 0), 20) * .0005 * scale
                              for s in (baseline, last)]
        assert [s["position_world"][0] for s in (baseline, last)] == pytest.approx(expected_positions)
        assert verdict["metrics"]["body_displacement_m"][0] == pytest.approx(
            expected_positions[1] - expected_positions[0])
        assert verdict["metrics"]["settle_samples"] >= 3
        fence = controller.hello()["generation"]
        checker.close()
        assert controller.hello()["generation"] == fence  # reader teardown is not stop
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(timeout=1.)
        checker.close()
        server.close()
    assert not worker.is_alive()


@pytest.mark.parametrize("field,bad", [
    ("ack", None), ("stop_ack", None), ("execution_ok", "true"), ("delivery_uncertain", True),
])
def test_motion_missing_or_uncertain_receipts_never_confirm(field, bad):
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * .002, 0., .3)))
    verdict = run_window(reader, result={field: bad})
    assert verdict["status"] != "confirmed"


@pytest.mark.parametrize("receipt,key,bad", [
    ("ack", "generation", True), ("ack", "generation", 2), ("stop_ack", "generation", 3),
    ("ack", "start_sim_time_s", float("nan")), ("ack", "end_sim_time_s", .5),
    ("ack", "source", "different"), ("stop_ack", "epoch", "other"),
    ("ack", "accepted", "true"), ("stop_ack", "latched", "false"),
    ("ack", "error", "uncertain admission"), ("stop_ack", "delivery_uncertain", True),
])
def test_bad_motion_receipts_cannot_borrow_valid_independent_travel(receipt, key, bad):
    result = fixture_motion_receipt()
    result[receipt][key] = bad
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * .002, 0., .3)))
    verdict = run_window(reader, result=result)
    assert verdict["status"] != "confirmed"


def test_missing_near_admission_baseline_abstains_without_interpolation():
    reader = ScriptedReader(lambda n: state(n + 20, generation=1 if n <= 8 else 2,
        position_world=(min(n - 1, 5) * .002, 0., .3)))
    result = fixture_motion_receipt()
    verdict = run_window(reader, result=result)
    assert verdict["status"] == "unverified"
    assert "baseline" in verdict["reason"]
    assert verdict["metrics"] == {}


def test_pre_admission_anomaly_is_retained_without_crediting_its_travel():
    admitted, stopped = threading.Event(), threading.Event()
    reader = ScriptedReader(lambda n: state(n, fallen=n == 4,
        generation=2 if stopped.is_set() else 1 if admitted.is_set() else 0))
    checker = checker_for(reader)
    try:
        token = checker.begin("walk_velocity", dict(vx=0., vy=0., wz=0., duration_s=.1))
        reader.wait(8)
        start = reader.calls * .02
        admitted.set()
        reader.wait(reader.calls + 8)
        stopped.set()
        result = fixture_motion_receipt()
        result["ack"].update(start_sim_time_s=start, end_sim_time_s=start + .1)
        verdict = checker.finish(token, result)
        assert verdict["status"] == "refuted" and "fallen" in verdict["reason"]
        assert verdict["metrics"]["path_length_m"] == 0.
        assert any(e["state"]["fallen"] for e in verdict["evidence"]["samples"])
    finally:
        checker.close()


def test_extra_generation_is_ambiguous_even_when_pose_matches():
    reader = ScriptedReader(lambda n: state(n, generation=4 if n >= 4 else 0,
                                           position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    verdict = run_window(reader)
    assert verdict["status"] == "unverified"
    assert "generation" in verdict["reason"]


def test_completion_and_stop_generation_changes_are_not_new_physics_evidence():
    reader = ScriptedReader(lambda n: state(n, generation=0 if n == 1 else (1 if n < 6 else 2),
                                           position_world=(min(n - 1, 5) * 0.002, 0., 0.3)))
    verdict = run_window(reader)
    assert verdict["status"] == "confirmed"


def test_tiny_nonzero_progress_cannot_exploit_a_negative_lower_error_bound():
    reader = ScriptedReader(lambda n: state(n, position_world=(min(n - 1, 5) * 0.000001, 0., 0.3)))
    verdict = run_window(reader, args=dict(vx=0.011, vy=0., wz=0., duration_s=0.1))
    assert verdict["status"] == "unverified"
    assert "resolution" in verdict["reason"]


def test_active_controller_is_not_a_completed_stop_even_with_zero_measured_velocity():
    verdict = run_window(ScriptedReader(lambda n: state(n, controller_status="active")),
                         "stop_navigation", {})
    assert verdict["status"] == "refuted"
    assert "settle" in verdict["reason"]


def test_late_reader_reply_cannot_supply_settle_evidence_past_deadline():
    class LateReader(ScriptedReader):
        def __init__(self):
            super().__init__()
            self.after = False
        def __call__(self):
            if self.after:
                time.sleep(0.035)
            return super().__call__()
    reader = LateReader()
    checker = checker_for(reader, settle_timeout_s=0.04, min_settle_samples=2, settle_window_s=0.01)
    try:
        token = checker.begin("stop_navigation", {})
        reader.wait(3)
        reader.after = True
        verdict = checker.finish(token, {"ok": True})
        # At most one fresh call completes within the 40 ms settle budget.
        assert verdict["status"] == "unverified"
        assert sum(e["phase"] == "after" for e in verdict["evidence"]["samples"]) <= 1
    finally:
        checker.close()


def test_fresh_but_regressing_local_receipt_clock_is_unknown():
    stamps = []
    def make(n):
        timestamp = time.monotonic()
        if n == 4:
            timestamp = stamps[-1] - 0.001
        stamps.append(timestamp)
        return state(n, received_monotonic_s=timestamp,
                     position_world=(min(n - 1, 5) * 0.002, 0., 0.3))
    verdict = run_window(ScriptedReader(make))
    assert verdict["status"] == "unverified"
    assert "receipt" in verdict["reason"]


def test_duplicate_step_with_fault_cannot_hide_failure_behind_earlier_healthy_window():
    reader = ScriptedReader(lambda n: state(min(n, 12), controller_status="ready" if n <= 12 else "fault"))
    verdict = run_window(reader, "stop_navigation", {})
    assert verdict["status"] == "unverified"
    assert "controller" in verdict["reason"]
