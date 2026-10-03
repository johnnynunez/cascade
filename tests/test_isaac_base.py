"""CPU transport exercises real loopback sockets, never Kit or physics."""
from __future__ import annotations

import copy
import threading
import time

import pytest
from mobile_support_fixture import support_contract
from mobile_tick_fixture import healthy_episode_gc  # noqa: F401

from cascade.control.mobile_base import BaseState, VelocityCommand
from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
from test_mobile_bridge import publish, wait_until


@pytest.fixture(autouse=True)
def no_mobile_transport_thread_leaks():
    before = set(threading.enumerate())
    yield
    def remaining():
        return [t for t in threading.enumerate() if t not in before and t.name.startswith("mobile-")]
    wait_until(lambda: not remaining())


def profile(port=45678, **changes):
    return {"robot_id": "duck", "source": "isolated-bridge", "engine": "physx",
            "device": "cuda:0", "asset_sha256": "a" * 64, "policy_sha256": "b" * 64, "model_identity_sha256": "e" * 64, "support_contract": support_contract(),
            "bridge_host": "127.0.0.1", "bridge_port": port, "timeout_s": 0.3, **changes}


@pytest.fixture
def bridge():
    c = MobileBridgeController(
        robot_id="duck", source="isolated-bridge", engine="physx", device="cuda:0",
        asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=support_contract(),
        max_linear_speed=0.2, max_angular_speed=0.8, max_duration_s=5.,
        lease_s=0.3, max_state_age_s=1., max_action_wall_s=2.,
    )
    server = MobileBridgeServer(c, port=0)
    server.start()
    try:
        yield c, server
    finally:
        server.close()


def test_construct_metadata_and_capabilities_do_not_dial(monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata must not connect")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    raw = IsaacBase(profile())
    assert raw.metadata == {"robot_id": "duck", "source": "isolated-bridge", "measurement_kind": "physics", "model_identity_sha256": "e" * 64}
    assert raw.capabilities == frozenset({"walk_velocity", "turn", "stop_navigation"})
    assert not raw.connected
    with pytest.raises(RuntimeError):
        raw.get_state()


@pytest.mark.parametrize("field", list(profile()))
def test_profile_requires_explicit_connection_and_identity_fields(field):
    from cascade.control.isaac_base import IsaacBase
    data = profile()
    del data[field]
    with pytest.raises(ValueError, match=field):
        IsaacBase(data)


@pytest.mark.parametrize("changes", [
    {"bridge_port": 0}, {"bridge_port": True}, {"bridge_port": "123"},
    {"bridge_host": "0.0.0.0"}, {"bridge_host": "example.com"},
    {"timeout_s": float("nan")}, {"timeout_s": 0}, {"timeout_s": True},
    {"asset_sha256": "bad"}, {"policy_sha256": "F" * 64},
    {"robot_id": " duck"}, {"engine": "mujoco"},
])
def test_profile_fails_closed_without_coercion(changes):
    from cascade.control.isaac_base import IsaacBase
    with pytest.raises(ValueError):
        IsaacBase(profile(**changes))


def test_state_decode_uses_local_receipt_and_conservative_rtt(bridge):
    from cascade.control.isaac_base import state_from_wire
    c, _ = bridge
    publish(c, q=list(range(14)))
    payload = c.state()
    payload["state"]["received_monotonic_s"] = 99999999.
    payload["state"]["producer_age_s"] = 0.25
    original = copy.deepcopy(payload)
    result = state_from_wire(payload, received_monotonic_s=10., round_trip_s=0.125)
    assert isinstance(result, BaseState)
    assert result.received_monotonic_s == 10.
    assert result.producer_age_s == 0.375
    assert result.joint_positions == tuple(range(14))
    assert payload == original
    assert state_from_wire(payload["state"], received_monotonic_s=10., round_trip_s=0.125) == result
    for field, value in (("producer_age_s", -1.), ("joint_positions", [0.]), ("step", True)):
        bad = copy.deepcopy(payload)
        bad["state"][field] = value
        with pytest.raises(ValueError):
            state_from_wire(bad, received_monotonic_s=10., round_trip_s=0.)
    with pytest.raises(ValueError):
        state_from_wire(payload, received_monotonic_s=10., round_trip_s=-0.1)


def test_real_client_socket_command_stop_reset_and_passive_state(bridge):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    raw = IsaacBase(profile(server.address[1]))
    try:
        raw.connect()
        with pytest.raises(RuntimeError, match="feedback|state"):
            raw.get_state()
        publish(c)
        before = raw.get_state()
        for _ in range(3):
            assert raw.get_state().step == before.step
        ack = raw.command_velocity(VelocityCommand(.1, 0., 0., .05), generation=before.generation)
        assert ack["generation"] == before.generation + 1
        assert raw.get_state().controller_status == "active"
        c.control_at(.055)
        assert raw.get_state().generation == ack["generation"]
        stopped = raw.stop(latch=True)
        assert stopped["latched"]
        assert c.control_at(.06) == (0., 0., 0.)
        assert raw.reset_stop()["latched"] is False
        assert c.control_at(.065) == (0., 0., 0.)
    finally:
        raw.disconnect()
    assert not raw.connected


@pytest.mark.parametrize("field,bad", [
    ("protocol", 2), ("kind", "rebot"), ("robot_id", "other"), ("source", "other"),
    ("engine", "newton"), ("device", "cpu"), ("asset_sha256", "c" * 64),
    ("policy_sha256", "c" * 64), ("model_identity_sha256", "f" * 64), ("support_contract_sha256", "f" * 64), ("physics_dt", .01), ("policy_dt", .01),
])
def test_every_channel_requires_identity_attestation(bridge, monkeypatch, field, bad):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    original = server.dispatch
    count = 0
    def corrupt(request):
        nonlocal count
        result = original(request)
        if request.get("op") == "hello":
            count += 1
            if count == 2:  # command hello was good; independent stop channel is not
                result[field] = bad
        return result
    monkeypatch.setattr(server, "dispatch", corrupt)
    raw = IsaacBase(profile(server.address[1]))
    with pytest.raises((ValueError, RuntimeError), match=field):
        raw.connect()
    raw.disconnect()
    assert not raw.connected
    assert c.hello()["generation"] == 0


@pytest.mark.parametrize("wall_tick_s", [.005, .025], ids=["normal-producer", "slow-producer"])
@pytest.mark.usefixtures("healthy_episode_gc")
def test_safe_base_uses_real_socket_renewal_and_two_advancing_preflight_reads(
        bridge, monkeypatch, wall_tick_s):
    from types import SimpleNamespace
    from cascade.control import isaac_base
    from cascade.control.isaac_base import IsaacBase
    from cascade.safety import base_harness
    from cascade.safety.base_harness import SafeBase
    c, server = bridge
    # This is a protocol/ownership test, not a host scheduling benchmark. Each
    # requested synthetic solve completes before its passive TCP observation;
    # the fixture clock advances by the normal/slow producer interval. Real
    # socket timeouts, renewal worker and 5 s emergency timer remain in place.
    # The real-wall delayed-state test below separately pins the .2 s veto.
    c.max_action_wall_s = 5.
    now, renew_due = [1.], [None]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(isaac_base, "time", clock)
    monkeypatch.setattr(base_harness, "time", clock)
    monkeypatch.setattr(c, "_clock", clock.monotonic)
    renewed = threading.Condition()
    renewals = []
    dispatch = server.dispatch
    def record(request):
        response = dispatch(request)
        with renewed:
            if request["op"] == "command_velocity" and response.get("ok"):
                renew_due[0] = now[0] + c.lease_s / 3
            if request["op"] == "renew" and response.get("ok"):
                renewals.append(response)
                renew_due[0] = now[0] + c.lease_s / 3
                renewed.notify_all()
        return response
    monkeypatch.setattr(server, "dispatch", record)
    raw = IsaacBase(profile(server.address[1]))
    limits = {"max_vx": .2, "max_vy": 0., "max_wz": .8, "max_duration_s": 1.,
              "max_state_age_s": .2, "max_no_progress_s": .2, "max_wall_duration_s": 5.,
              "poll_interval_s": .01, "turn_speed_rad_s": .3, "turn_tolerance_rad": .02,
              "max_turn_angle_rad": 1.}
    safe = SafeBase(raw, limits)
    halt = threading.Event()
    requested, published = threading.Event(), threading.Event()
    errors = []
    completed = []
    def completed_steps():
        try:
            step = 0
            while not halt.is_set():
                if not requested.wait(.05):
                    continue
                requested.clear()
                # Advance only after the actual background worker has renewed
                # over its own TCP channel. No forged ACK or repeated solve.
                with renewed:
                    assert renewed.wait_for(lambda: renew_due[0] is None or
                        now[0] + wall_tick_s < renew_due[0], timeout=2.), "renewal worker stalled"
                    now[0] += wall_tick_s
                step += 1
                c.control_at(step * .005)
                publish(c, step=step, sim_time=step * .005)
                completed.append((step, step * .005))
                published.set()
        except Exception as e:
            errors.append(e)
            published.set()
    get_state = raw.get_state
    def after_completed_solve():
        published.clear()
        requested.set()
        assert published.wait(2.), "synthetic producer stalled"
        assert not errors, errors
        return get_state()
    monkeypatch.setattr(raw, "get_state", after_completed_solve)
    producer = threading.Thread(target=completed_steps)
    try:
        safe.connect()
        producer.start()
        # Both fixture wall cadences span the .3 s lease during .45 s simulated.
        result = safe.walk_velocity(.1, 0., 0., .45)
        assert result["execution_ok"], (
            f"error={result.get('error')}; successful renewals={len(renewals)}; "
            f"wall tick={wall_tick_s}; "
            f"completed solves={len(completed)}; producer errors={errors}")
        assert result["ok"] is False and result["outcome"] == "unverified"
        samples = result["measured"]["samples"]
        assert samples[1]["step"] > samples[0]["step"]
        assert result["ack"]["generation"] == samples[1]["generation"] + 1
        assert result["stop_ack"]["generation"] == result["ack"]["generation"] + 1
        assert len(renewals) >= 2
        assert sum(ack["active"] for ack in renewals) >= 2
        assert all(ack["generation"] == result["ack"]["generation"] for ack in renewals)
        assert samples[-1]["received_monotonic_s"] - samples[1]["received_monotonic_s"] > c.lease_s
        solves = tuple(completed)
        assert len(solves) >= 90
        assert solves == tuple((step, step * .005) for step in range(1, len(solves) + 1))
        assert samples[-1]["sim_time_s"] >= result["ack"]["end_sim_time_s"]
        assert all(s["position_world"] == [0., 0., .12] for s in samples)
        assert not errors
    finally:
        halt.set()
        producer.join(1.)
        safe.disconnect()
    assert not producer.is_alive()
    assert not raw._renew_thread.is_alive()


@pytest.mark.parametrize("wall_tick_s", [.005, .025], ids=["normal-producer", "slow-producer"])
@pytest.mark.usefixtures("healthy_episode_gc")
def test_gc_isolation_does_not_admit_a_delayed_socket_state(bridge, monkeypatch, wall_tick_s):
    """Real consumer delay still vetoes freshness while ticks/renewals advance."""
    import gc
    from cascade.control.isaac_base import IsaacBase
    from cascade.safety.base_harness import SafeBase
    from mobile_tick_fixture import scheduled_tick_steps
    c, server = bridge
    c.max_action_wall_s = 5.
    raw = IsaacBase(profile(server.address[1]))
    safe = SafeBase(raw, {
        "max_vx": .2, "max_vy": 0., "max_wz": .8, "max_duration_s": 1.,
        "max_state_age_s": .2, "max_no_progress_s": .2, "max_wall_duration_s": 5.,
        "poll_interval_s": .01, "turn_speed_rad_s": .3, "turn_tolerance_rad": .02,
        "max_turn_angle_rad": 1.,
    })
    halt = threading.Event()
    completed, errors, captured, renewals, commands, stops = [], [], [], [], [], []
    dispatch, get_state = server.dispatch, raw.get_state
    def record(request):
        response = dispatch(request)
        if request['op'] == 'renew' and response.get('ok'):
            renewals.append(response)
        if request['op'] == 'command_velocity' and response.get('ok'):
            commands.append(response)
        if request['op'] == 'stop' and response.get('ok'):
            stops.append(response)
        return response
    def delayed_state():
        value = get_state()  # genuine socket read with original capture/receipt
        if value.generation == 1 and not captured:
            captured.append(value.as_dict())
            assert not gc.isenabled()
            threading.Event().wait(.25)  # exceeds the unchanged .2 s freshness
            assert value.as_dict() == captured[0]
        return value
    def produce():
        try:
            for step in scheduled_tick_steps(halt, first_step=1, wall_interval_s=wall_tick_s):
                c.control_at(step * .005)
                publish(c, step=step, sim_time=step * .005)
                completed.append(step)
        except Exception as exc:
            errors.append(exc)
    monkeypatch.setattr(server, 'dispatch', record)
    monkeypatch.setattr(raw, 'get_state', delayed_state)
    producer = threading.Thread(target=produce)
    try:
        safe.connect()
        producer.start()
        wait_until(lambda: c.state()['feedback_available'])
        result = safe.walk_velocity(.1, 0., 0., .45)
        assert result['execution_ok'] is False and result['ok'] is False, result
        assert result['error'] == 'stale or future-dated feedback', result
        assert len(commands) == 1 and len(captured) == 1
        assert captured[0]['generation'] == commands[0]['generation']
        assert completed[-1] > captured[0]['step']  # healthy producer did not freeze
        assert renewals and all(row['generation'] == commands[0]['generation'] for row in renewals)
        assert len(stops) == 1 and stops[0]['generation'] == commands[0]['generation'] + 1
        assert safe.latched and c.state()['latched']
        assert not errors
    finally:
        halt.set()
        producer.join(1.)
        safe.disconnect()
    assert not producer.is_alive() and not raw._renew_thread.is_alive()


def test_stop_is_independent_of_both_blocked_command_and_state(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    raw = IsaacBase(profile(server.address[1], timeout_s=.6))
    raw.connect()
    generation = raw.get_state().generation
    original = server.dispatch
    command_entered, state_entered, release = threading.Event(), threading.Event(), threading.Event()
    def blocked(request):
        if request["op"] in {"command_velocity", "state"}:
            (command_entered if request["op"] == "command_velocity" else state_entered).set()
            assert release.wait(1.)
        return original(request)
    monkeypatch.setattr(server, "dispatch", blocked)
    results, errors = [], []
    def reader():
        try:
            raw.get_state()
        except Exception as e:
            errors.append(e)
    mover = threading.Thread(target=lambda: results.append(raw.command_velocity(
        VelocityCommand(.1, 0., 0., .1), generation=generation)))
    observer = threading.Thread(target=reader)
    try:
        mover.start()
        observer.start()
        assert command_entered.wait(.5) and state_entered.wait(.5)
        started = time.monotonic()
        ack = raw.stop()
        assert time.monotonic() - started < .2
        assert ack["ok"] and c.state()["latched"]
        assert not release.is_set()
    finally:
        release.set()
        mover.join(1.)
        observer.join(1.)
        raw.disconnect()
    assert not mover.is_alive() and not observer.is_alive()
    assert results[0]["ok"] is False and results[0]["delivery_uncertain"]
    assert c.control_at(.1) == (0., 0., 0.)
    assert not errors


def test_renew_delivery_failure_remains_latched_even_if_later_state_is_good(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    original = server.dispatch
    renew_seen = threading.Event()
    def refuse(request):
        if request["op"] == "renew":
            renew_seen.set()
            return {"ok": False, "error": "renew lost"}
        return original(request)
    monkeypatch.setattr(server, "dispatch", refuse)
    raw = IsaacBase(profile(server.address[1]))
    try:
        raw.connect()
        assert raw.command_velocity(VelocityCommand(.1, 0., 0., .5), generation=0)["ok"]
        assert renew_seen.wait(.5)
        wait_until(lambda: c.state()["latched"])
        assert c.control_at(.2) == (0., 0., 0.)
        again = raw.command_velocity(VelocityCommand(.1, 0., 0., .1), generation=c.hello()["generation"])
        assert again["ok"] is False
    finally:
        raw.disconnect()


def test_stop_before_connect_survives_startup_without_motion(bridge):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    raw = IsaacBase(profile(server.address[1]))
    result = raw.stop()
    assert result["ok"] is False
    try:
        raw.connect()
        publish(c)
        assert raw.get_state().latched
        assert raw.command_velocity(VelocityCommand(.1, 0., 0., .1), generation=1)["ok"] is False
        assert raw.reset_stop()["ok"]
        assert c.control_at(.01) == (0., 0., 0.)
    finally:
        raw.disconnect()


def test_reset_during_active_motion_is_rejected_without_dispatching_reset(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    raw = IsaacBase(profile(server.address[1]))
    calls = []
    original = server.dispatch
    def record(request):
        calls.append(request["op"])
        return original(request)
    monkeypatch.setattr(server, "dispatch", record)
    try:
        raw.connect()
        assert raw.command_velocity(VelocityCommand(.1, 0., 0., .5), generation=0)["ok"]
        result = raw.reset_stop()
        assert result["ok"] is False
        assert "reset_stop" not in calls
    finally:
        raw.disconnect()


def test_lost_command_response_is_uncertain_not_retried_or_revived(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    raw = IsaacBase(profile(server.address[1], timeout_s=.05))
    original = server.dispatch
    admitted, release = threading.Event(), threading.Event()
    count = 0
    def lose_ack(request):
        nonlocal count
        result = original(request)
        if request["op"] == "command_velocity":
            count += 1
            admitted.set()
            assert release.wait(.5)
        return result
    monkeypatch.setattr(server, "dispatch", lose_ack)
    try:
        raw.connect()
        begin = time.monotonic()
        result = raw.command_velocity(VelocityCommand(.1, 0., 0., .5), generation=0)
        assert time.monotonic() - begin < .3
        assert admitted.is_set()
        assert result["ok"] is False and result["delivery_uncertain"]
        assert c.state()["latched"]
        assert c.control_at(.01) == (0., 0., 0.)
        release.set()
        raw.disconnect()
        raw.connect()
        assert raw.get_state().latched
        assert count == 1
    finally:
        release.set()
        raw.disconnect()


@pytest.mark.parametrize("field,bad", [
    ("robot_id", "wrong"), ("source", "wrong"), ("epoch", "old"), ("generation", True),
    ("generation", 0), ("latched", "false"), ("accepted", False),
    ("model_identity_sha256", "f" * 64), ("model_identity_sha256", None),
    ("end_sim_time_s", .005), ("start_sim_time_s", float("nan")),
])
def test_malformed_command_ack_never_authorizes_retry(bridge, monkeypatch, field, bad):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    original = server.dispatch
    commands = []
    def corrupt(request):
        result = original(request)
        if request["op"] == "command_velocity":
            commands.append(request)
            result[field] = bad
        return result
    monkeypatch.setattr(server, "dispatch", corrupt)
    raw = IsaacBase(profile(server.address[1], timeout_s=.05))
    try:
        raw.connect()
        result = raw.command_velocity(VelocityCommand(.1, 0., 0., .1), generation=0)
        assert result["ok"] is False and result["delivery_uncertain"]
        assert len(commands) == 1
        wait_until(lambda: c.state()["latched"])
        assert c.control_at(.01) == (0., 0., 0.)
    finally:
        raw.disconnect()


@pytest.mark.parametrize('hello_index', [1, 2, 3, 4])
@pytest.mark.parametrize('bad', [None, 'f'*64])
def test_effective_recipe_bound_on_every_connection(bridge, monkeypatch, hello_index, bad):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    dispatch = server.dispatch
    count = 0
    def changed(request):
        nonlocal count
        result = dispatch(request)
        if request['op'] == 'hello':
            count += 1
            if count == hello_index:
                result['model_identity_sha256'] = bad
        return result
    monkeypatch.setattr(server, 'dispatch', changed)
    raw = IsaacBase(profile(server.address[1]))
    try:
        with pytest.raises(RuntimeError, match='model_identity_sha256'):
            raw.connect()
        assert not raw.connected
        assert c.hello()['generation'] == 0
    finally:
        raw.disconnect()


@pytest.mark.parametrize('consumer', ['actor', 'truth'])
def test_state_cannot_change_effective_recipe_after_a_valid_hello(bridge, monkeypatch, consumer):
    from cascade.control.isaac_base import IsaacBase
    from cascade.sim.base_truth import BaseTruthReader
    c, server = bridge
    publish(c)
    dispatch = server.dispatch
    def changed(request):
        result = dispatch(request)
        if request['op'] == 'state':
            result['state']['model_identity_sha256'] = 'f'*64
            result['state']['support']['model_identity_sha256'] = 'f'*64
        return result
    monkeypatch.setattr(server, 'dispatch', changed)
    if consumer == 'actor':
        raw = IsaacBase(profile(server.address[1]))
        try:
            raw.connect()
            with pytest.raises(RuntimeError, match='identity'):
                raw.get_state()
        finally:
            raw.disconnect()
    else:
        reader = BaseTruthReader(profile(server.address[1]))
        try:
            assert reader() is None
            assert 'identity' in reader.last_error
        finally:
            reader.close()


def test_stop_ack_requires_new_invalidation_not_a_replayed_generation(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    publish(c)
    original = server.dispatch
    def stale(request):
        result = original(request)
        if request["op"] == "stop":
            result["generation"] = 0
        return result
    monkeypatch.setattr(server, "dispatch", stale)
    raw = IsaacBase(profile(server.address[1]))
    try:
        raw.connect()
        assert raw.command_velocity(VelocityCommand(.1, 0., 0., .1), generation=0)["ok"]
        result = raw.stop()
        assert result["ok"] is False and result["delivery_uncertain"]
    finally:
        raw.disconnect()


@pytest.mark.parametrize("cancel", ["stop", "disconnect"])
def test_cancellation_crossing_connection_startup_is_not_lost(bridge, monkeypatch, cancel):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    raw = IsaacBase(profile(server.address[1], timeout_s=.5))
    entered, release = threading.Event(), threading.Event()
    original = server.dispatch
    def blocked(request):
        if request["op"] == "hello" and request.get("role") == "control":
            entered.set()
            assert release.wait(1.)
        return original(request)
    monkeypatch.setattr(server, "dispatch", blocked)
    errors = []
    def connect():
        try:
            raw.connect()
        except Exception as e:
            errors.append(e)
    worker = threading.Thread(target=connect)
    try:
        worker.start()
        assert entered.wait(.5)
        getattr(raw, cancel)()
        release.set()
        worker.join(1.)
        assert not worker.is_alive()
        if cancel == "stop":
            assert not errors
            assert c.state()["latched"]
        else:
            assert len(errors) == 1 and "cancel" in str(errors[0])
            assert not raw.connected
            wait_until(lambda: not server._owners)
    finally:
        release.set()
        worker.join(1.)
        raw.disconnect()


def test_reset_racing_priority_stop_cannot_unlatch(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    c, server = bridge
    raw = IsaacBase(profile(server.address[1], timeout_s=.5))
    raw.connect()
    raw.stop()
    entered, release = threading.Event(), threading.Event()
    original = server.dispatch
    def blocked(request):
        if request["op"] == "reset_stop":
            entered.set()
            assert release.wait(1.)
        return original(request)
    monkeypatch.setattr(server, "dispatch", blocked)
    results = []
    worker = threading.Thread(target=lambda: results.append(raw.reset_stop()))
    try:
        worker.start()
        assert entered.wait(.5)
        assert raw.stop()["ok"]
        release.set()
        worker.join(1.)
        assert not worker.is_alive()
        assert results[0]["ok"] is False
        assert c.state()["latched"]
    finally:
        release.set()
        worker.join(1.)
        raw.disconnect()


def test_independent_truth_reader_shares_only_the_pure_wire_decoder(bridge):
    from cascade.control.isaac_base import IsaacBase
    from cascade.sim.base_truth import BaseTruthReader
    c, server = bridge
    raw = IsaacBase(profile(server.address[1]))
    reader = BaseTruthReader(profile(server.address[1]))
    try:
        raw.connect()
        publish(c)
        snapshot = reader()
        assert snapshot is not None, reader.last_error
        assert snapshot.step == 1 and snapshot.source == "isolated-bridge"
        assert len(server._owners) == 1
        ack = raw.command_velocity(VelocityCommand(.1, 0., 0., .1), generation=0)
        assert ack["ok"]
        reader.close()
        assert c.control_at(.01) == (.1, 0., 0.)
        assert not c.state()["latched"]
    finally:
        reader.close()
        raw.disconnect()


def test_live_socket_with_frozen_physics_cannot_authorize_safe_base(bridge, monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    from cascade.safety.base_harness import SafeBase
    c, server = bridge
    publish(c)
    raw = IsaacBase(profile(server.address[1]))
    safe = SafeBase(raw, {"max_vx": .2, "max_vy": 0., "max_wz": .8, "max_duration_s": 1.,
                         "max_state_age_s": .2, "max_no_progress_s": .05, "max_wall_duration_s": .4,
                         "poll_interval_s": .005, "turn_speed_rad_s": .3, "turn_tolerance_rad": .02,
                         "max_turn_angle_rad": 1.})
    original, commands = server.dispatch, []
    def record(request):
        if request["op"] == "command_velocity":
            commands.append(request)
        return original(request)
    monkeypatch.setattr(server, "dispatch", record)
    try:
        safe.connect()
        result = safe.walk_velocity(.1, 0., 0., .1)
        assert result["execution_ok"] is False
        assert "did not advance" in result["error"]
        assert commands == []
    finally:
        safe.disconnect()
