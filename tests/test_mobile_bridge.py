"""The simulator-side lease must survive loss of its client, not its command."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
from mobile_support_fixture import support_contract


MODULE = Path(__file__).resolve().parents[1] / "src/cascade/sim/mobile_bridge.py"


def _module():
    spec = importlib.util.spec_from_file_location("mobile_bridge_under_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


@pytest.fixture
def control():
    module = _module()
    clock = Clock()
    c = module.MobileBridgeController(
        robot_id="duck", source="isolated-bridge", engine="physx", device="cuda:0",
        asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64, support_contract=support_contract(),
        max_linear_speed=0.2, max_angular_speed=0.8, max_duration_s=10.0,
        lease_s=0.5, max_state_age_s=0.5, clock=clock,
    )
    return c, clock


def publish(c, *, step=1, sim_time=0.005, **changes):
    from mobile_support_fixture import support
    state = {
        "step": step, "sim_time": sim_time,
        "position": [0.0, 0.0, 0.12], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity": [0.0, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.0],
        "q": [0.0] * 14, "dq": [0.0] * 14, "fallen": False,
        "joint_names": [f"joint_{i}" for i in range(14)], "contacts": ["left_foot"],
        "support": support(step, sim_time),
    }
    state.update(changes)
    c.publish(state)


def command(c, **changes):
    request = {
        "robot_id": "duck", "source": "isolated-bridge", "epoch": c.hello()["epoch"],
        "generation": c.hello()["generation"], "owner": "test-client", "command_id": "move-1",
        "vx": 0.1, "vy": 0.0, "wz": 0.0, "duration_s": 1.0,
    }
    request.update(changes)
    return c.command_velocity(request)


def test_latched_permission_does_not_disable_explicit_live_balance(control):
    """Synthetic software snapshot: actual rest still needs physical windows."""
    c, _ = control
    publish(c, balance_active=True)
    command(c)
    ack = c.stop(latch=True)
    state = c.state()["state"]
    assert ack["physical_stop_verified"] is False
    assert state["latched"] is True
    assert state["controller_status"] == "ready"
    assert c.control_at(0.005) == (0., 0., 0.)
    with pytest.raises(ValueError, match="latched"):
        command(c, command_id="must-not-revive")


def test_missing_balance_attestation_remains_disabled_after_latching(control):
    c, _ = control
    publish(c)
    c.stop(latch=True)
    assert c.state()["state"]["controller_status"] == "disabled"


def test_explicit_inactive_balance_rejects_motion_even_without_latch(control):
    c, _ = control
    publish(c, balance_active=False)
    assert c.state()["state"]["controller_status"] == "disabled"
    with pytest.raises(ValueError, match="balance"):
        command(c)
    assert c.control_at(0.005) == (0., 0., 0.)


def test_fault_overrides_live_balance_attestation(control):
    c, _ = control
    publish(c, balance_active=True)
    c.fault("failed policy")
    assert c.state()["state"]["controller_status"] == "fault"


@pytest.mark.parametrize("value", [None, 0, 1, "yes", {}, [True]])
def test_balance_attestation_is_strict_boolean_when_provided(control, value):
    c, _ = control
    with pytest.raises(ValueError, match="balance_active"):
        publish(c, balance_active=value)
    assert c.state()["latched"] is True


def test_introspection_does_not_start_motion(control):
    c, _ = control
    assert c.hello()["kind"] == "microduck"
    assert c.hello()["protocol"] == 1
    assert c.control_at(0.0) == (0.0, 0.0, 0.0)
    assert c.state()["controller"] == "disarmed"
    with pytest.raises(ValueError, match="state"):
        command(c)


def test_command_has_simulated_duration_and_zero_after_completion(control):
    c, _ = control
    publish(c)
    result = command(c)
    assert result["accepted"] is True
    assert result["completed"] is False
    assert c.control_at(0.5) == (0.1, 0.0, 0.0)
    assert c.control_at(1.005) == (0.0, 0.0, 0.0)
    assert c.state()["controller"] == "balancing"


def test_wall_deadman_expires_while_simulation_is_paused(control):
    c, clock = control
    publish(c)
    command(c)
    clock.now += 0.501
    assert c.control_at(0.005) == (0.0, 0.0, 0.0)
    assert c.state()["latched"] is True
    assert "lease" in c.state()["fault"]
    with pytest.raises(ValueError):
        command(c, command_id="move-2")


def test_renew_does_not_extend_simulated_duration(control):
    c, clock = control
    publish(c)
    command(c)
    for i in range(1, 4):
        clock.now += 0.2
        publish(c, step=i + 1, sim_time=0.2 * i)
        c.renew({"owner": "test-client", "command_id": "move-1", "epoch": c.hello()["epoch"],
                 "generation": c.hello()["generation"]})
    assert c.control_at(1.005) == (0.0, 0.0, 0.0)


def test_wrong_owner_cannot_renew_or_replace_command(control):
    c, _ = control
    publish(c)
    command(c)
    with pytest.raises(ValueError, match="owner|active"):
        command(c, owner="other", command_id="move-2")
    with pytest.raises(ValueError, match="owner"):
        c.renew({"owner": "other", "command_id": "move-1", "epoch": c.hello()["epoch"],
                 "generation": c.hello()["generation"]})


def test_stop_invalidates_pending_generation_and_normal_stop_cannot_clear_latch(control):
    c, _ = control
    publish(c)
    old = c.hello()["generation"]
    c.stop(latch=True)
    c.stop(latch=False)
    assert c.state()["latched"] is True
    with pytest.raises(ValueError, match="latched|generation"):
        command(c, generation=old)
    c.reset_stop()
    assert c.control_at(0.1) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="generation"):
        command(c, generation=old)


def test_epoch_reset_invalidates_old_commands_and_policy_history(control):
    c, _ = control
    publish(c)
    old_epoch = c.hello()["epoch"]
    command(c)
    c.begin_epoch()
    assert c.hello()["epoch"] != old_epoch
    assert c.control_at(0.0) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="epoch"):
        command(c, epoch=old_epoch)


@pytest.mark.parametrize("field,value", [
    ("vx", math.nan), ("vx", math.inf), ("vy", True), ("wz", "0.1"),
    ("vx", 0.201), ("wz", 0.801), ("duration_s", 0.0),
    ("duration_s", -1.0), ("duration_s", 10.1), ("duration_s", True),
    ("generation", True), ("owner", ""), ("command_id", ""), ("robot_id", "rebot"),
])
def test_invalid_command_is_rejected_without_actuation(control, field, value):
    c, _ = control
    publish(c)
    with pytest.raises(ValueError):
        command(c, **{field: value})
    assert c.control_at(0.005) == (0.0, 0.0, 0.0)


def test_repeated_or_regressing_feedback_is_not_refreshed(control):
    c, clock = control
    publish(c)
    command(c)
    clock.now += 0.3
    with pytest.raises(ValueError, match="advance|step"):
        publish(c)
    assert c.state()["latched"] is True
    assert c.control_at(0.005) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("change", [
    {"position": [0.0, math.nan, 0.12]}, {"orientation_wxyz": [0.0] * 4},
    {"q": [0.0] * 13}, {"dq": [math.inf] * 14}, {"fallen": True},
    {"sim_time": -0.01}, {"step": True},
])
def test_invalid_physics_feedback_latches_fault(control, change):
    c, _ = control
    publish(c)
    command(c)
    with pytest.raises(ValueError):
        publish(c, **{"step": 2, "sim_time": 0.01, **change})
    assert c.state()["latched"] is True
    assert c.control_at(0.01) == (0.0, 0.0, 0.0)


def test_reads_are_copies_and_never_advance_physics(control):
    c, _ = control
    publish(c)
    state = c.state()
    state["position"][0] = 500.0
    assert c.state()["position"] == [0.0, 0.0, 0.12]
    assert c.state()["step"] == 1


def test_policy_fault_is_not_silently_reset(control):
    c, _ = control
    publish(c)
    command(c)
    c.fault("policy inference failed")
    assert c.control_at(0.2) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="fault"):
        c.reset_stop()
    assert c.state()["fault"] == "policy inference failed"


def test_mobile_admission_consumes_fence_but_completion_does_not(control):
    c, _ = control
    publish(c)
    before = c.hello()["generation"]
    ack = command(c)
    assert ack["generation"] == before + 1
    assert {k: ack[k] for k in ("robot_id", "source", "latched")} == {
        "robot_id": "duck", "source": "isolated-bridge", "latched": False}
    assert ack["start_sim_time_s"] == 0.005
    assert ack["end_sim_time_s"] == 1.005
    assert c.state()["state"]["controller_status"] == "active"
    c.control_at(1.005)
    state = c.state()["state"]
    assert state["generation"] == before + 1
    assert state["controller_status"] == "ready"
    assert state["step"] == 1
    assert state["sim_time_s"] == 0.005
    for ack in (c.stop(latch=False), c.reset_stop()):
        assert ack["robot_id"] == "duck"
        assert ack["source"] == "isolated-bridge"
        assert ack["latched"] is False


def test_mobile_snapshot_preserves_measured_fields_and_age(control):
    from cascade.control.mobile_base import BaseState
    c, clock = control
    assert c.state().get("state") is None
    publish(c, q=list(range(14)), dq=[0.2] * 14)
    clock.now += 0.2
    snapshot = BaseState.from_dict(c.state()["state"])
    assert snapshot.joint_names == tuple(f"joint_{i}" for i in range(14))
    assert snapshot.joint_positions == tuple(range(14))
    assert snapshot.joint_velocities == (0.2,) * 14
    assert snapshot.contacts == ("left_foot",)
    assert snapshot.producer_age_s == pytest.approx(0.2)
    assert snapshot.measurement_kind == "physics"
    assert snapshot.position_world == (0.0, 0.0, 0.12)
    c.stop()
    assert c.state()["state"]["step"] == snapshot.step
    assert c.state()["state"]["controller_status"] == "disabled"


@pytest.mark.parametrize("changes", [
    {"joint_names": []}, {"joint_names": ["duplicate"] * 14},
    {"contacts": None}, {"source": "other"},
])
def test_mobile_rejects_incomplete_or_foreign_snapshot(control, changes):
    c, _ = control
    with pytest.raises(ValueError):
        publish(c, **changes)
    assert c.state()["latched"]


@pytest.fixture
def server(control):
    c, _ = control
    server = _module().MobileBridgeServer(c, port=0)
    server.start()
    try:
        yield server
    finally:
        server.close()


def channel(server, role="reader", owner=None):
    from cascade.sim.bridge_client import BridgeClient
    client = BridgeClient(*server.address, timeout_s=0.5)
    client.connect()
    try:
        client.request({"op": "hello", "role": role, **({"owner": owner} if owner else {})})
    except BaseException:
        client.close()
        raise
    return client


def wire_command(c, **changes):
    return {"op": "command_velocity", "robot_id": "duck", "source": "isolated-bridge",
            "epoch": c.hello()["epoch"], "generation": c.hello()["generation"],
            "owner": "socket-owner", "command_id": "socket-move",
            "vx": 0.1, "vy": 0., "wz": 0., "duration_s": 1., **changes}


def wait_until(predicate):
    import time
    end = time.monotonic() + 1.5
    while not predicate():
        assert time.monotonic() < end, "bounded wait expired"
        time.sleep(0.002)


def test_socket_reader_disconnect_does_not_estop_owner(server, control):
    c, _ = control
    publish(c)
    owner = channel(server, "control", "socket-owner")
    reader = channel(server)
    try:
        ack = owner.request(wire_command(c))
        assert ack["generation"] == 1
        assert reader.request({"op": "state"})["state"]["step"] == 1
        reader.close()
        assert c.control_at(0.1) == (0.1, 0., 0.)
        owner.close()
        wait_until(lambda: c.state()["latched"])
        assert c.control_at(0.2) == (0., 0., 0.)
    finally:
        reader.close()
        owner.close()


def test_server_watchdog_expires_without_physics_or_state_reads(server, control):
    c, clock = control
    publish(c)
    owner = channel(server, "control", "socket-owner")
    try:
        owner.request(wire_command(c))
        clock.now += 0.6
        wait_until(lambda: c._latched)  # not a state call that could hide a missing watchdog
        assert c.state()["state"]["generation"] == 2
        assert "lease" in c.state()["fault"]
        assert c.state()["state"]["step"] == 1
    finally:
        owner.close()


def test_reader_cannot_actuate_and_arbitrary_ops_never_dispatch(server, control):
    from cascade.sim.bridge_client import BridgeError
    c, _ = control
    publish(c)
    reader = channel(server)
    try:
        with pytest.raises(BridgeError, match="role|channel"):
            reader.request(wire_command(c))
        for op in ("exec", "set_joints", "reset_scene", "begin_epoch", "frame"):
            result = server.dispatch({"op": op})
            assert result["ok"] is False
        assert c.hello()["generation"] == 0
    finally:
        reader.close()


def test_stop_channel_passes_blocked_state_request(server, control, monkeypatch):
    import threading
    c, _ = control
    publish(c)
    owner = channel(server, "control", "socket-owner")
    reader = channel(server)
    stopper = channel(server, "stop", "socket-owner")
    entered, release = threading.Event(), threading.Event()
    original = c.state
    def blocked():
        entered.set()
        assert release.wait(1.)
        return original()
    monkeypatch.setattr(c, "state", blocked)
    errors = []
    def read():
        try:
            reader.request({"op": "state"})
        except Exception as e:
            errors.append(e)
    worker = threading.Thread(target=read)
    try:
        owner.request(wire_command(c))
        worker.start()
        assert entered.wait(0.5)
        assert stopper.request({"op": "stop", "latch": True})["latched"]
        assert not release.is_set()
    finally:
        release.set()
        worker.join(1.)
        for client in (owner, reader, stopper):
            client.close()
    assert not worker.is_alive()
    assert not errors


def test_rpc_reset_cannot_clear_a_stop_that_crossed_its_fence(server, control):
    c, _ = control
    publish(c)
    old = {"op": "reset_stop", "epoch": c.hello()["epoch"], "generation": 0}
    c.stop(latch=True)
    assert server.dispatch(old)["ok"] is False
    assert c.state()["latched"]
    assert server.dispatch({**old, "generation": 1})["ok"]
    assert c.state()["state"]["controller_status"] == "ready"


def test_rpc_request_age_includes_queue_before_controller_admission(server, control, monkeypatch):
    c, clock = control
    publish(c)
    original = c.command_velocity
    def delayed(request):
        clock.now += 0.6
        publish(c, step=2, sim_time=0.01)
        return original(request)
    monkeypatch.setattr(c, "command_velocity", delayed)
    result = server.dispatch(wire_command(c))
    assert result["ok"] is False
    assert c.control_at(0.01) == (0., 0., 0.)


def test_completed_renew_is_safe_without_extending_motion(control):
    c, _ = control
    publish(c)
    ack = command(c)
    request = {"owner": "test-client", "command_id": "move-1",
               "epoch": ack["epoch"], "generation": ack["generation"]}
    c.control_at(1.005)
    done = c.renew(request)
    assert done["ok"] and done["active"] is False
    assert c.state()["state"]["generation"] == ack["generation"]
    with pytest.raises(ValueError):
        c.renew({**request, "owner": "not-owner"})


def test_renewal_cannot_extend_absolute_action_wall_budget(control):
    c, clock = control
    c.max_action_wall_s = 0.75
    publish(c)
    ack = command(c)
    req = {"owner": "test-client", "command_id": "move-1", "epoch": ack["epoch"],
           "generation": ack["generation"]}
    for step in (2, 3, 4):
        clock.now += 0.2
        publish(c, step=step, sim_time=step * 0.005)
        c.renew(req)
    clock.now += 0.151
    c.watchdog()
    assert c.state()["latched"]
    with pytest.raises(ValueError):
        c.renew(req)


@pytest.mark.parametrize("host", ["0.0.0.0", "example.com", "192.0.2.1", "::"])
def test_server_refuses_non_loopback_before_binding(control, host):
    with pytest.raises(ValueError, match="loopback"):
        _module().MobileBridgeServer(control[0], host=host, port=0)


def test_request_and_connection_bounds_and_cleanup(control):
    import socket
    c, _ = control
    server = _module().MobileBridgeServer(c, port=0)
    server.MAX_CONNECTIONS = 2
    server._slots = __import__("threading").BoundedSemaphore(2)
    server.IO_TIMEOUT_S = 0.05
    server.start()
    sockets = []
    try:
        for _ in range(2):
            s = socket.create_connection(server.address, timeout=0.5)
            sockets.append(s)
        wait_until(lambda: len(server._clients) == 2)
        third = socket.create_connection(server.address, timeout=0.5)
        sockets.append(third)
        assert third.recv(1) == b""
        sockets[0].sendall(b"{" + b" " * server.MAX_REQUEST_BYTES)
        assert sockets[0].recv(1) == b""
        sockets[1].sendall(b'{"op":')
        assert sockets[1].recv(1) == b""  # partial request has absolute deadline
    finally:
        server.close()
        for s in sockets:
            s.close()
    assert not server._workers
    assert not server._accept_thread.is_alive()
    assert not server._watchdog_thread.is_alive()


def test_source_binding_is_rejected(control):
    c, _ = control
    publish(c)
    with pytest.raises(ValueError, match="source"):
        c.command_velocity(wire_command(c, source="another-producer"))


def test_active_reset_is_rejected(control):
    c, _ = control
    publish(c)
    ack = c.command_velocity(wire_command(c))
    with pytest.raises(ValueError, match="active"):
        c.reset_stop({"epoch": ack["epoch"], "generation": ack["generation"]})
    assert c.state()["state"]["controller_status"] == "active"


def test_stale_renew_channel_cannot_outlive_its_control_owner(server, control):
    from cascade.sim.bridge_client import BridgeError
    c, _ = control
    publish(c)
    owner = channel(server, "control", "socket-owner")
    renew = channel(server, "renew", "socket-owner")
    ack = owner.request(wire_command(c))
    c.control_at(1.005)
    owner.close()
    wait_until(lambda: not server._owners)
    try:
        with pytest.raises(BridgeError, match="owner|channel"):
            renew.request({"op": "renew", "epoch": ack["epoch"], "generation": ack["generation"],
                           "owner": "socket-owner", "command_id": "socket-move"})
    finally:
        renew.close()


def test_valid_fallen_measurement_is_preserved_even_though_control_faults(control):
    c, _ = control
    publish(c)
    with pytest.raises(ValueError, match="fallen"):
        publish(c, step=2, sim_time=.01, fallen=True, position=[.01, 0., .02])
    state = c.state()["state"]
    assert state["fallen"] is True
    assert state["controller_status"] == "fault"
    assert state["position_world"] == [.01, 0., .02]
    assert state["step"] == 2 and state["sim_time_s"] == .01


def test_frame_callback_is_optional_cached_data_without_controller_lock(control):
    c, _ = control
    server = _module().MobileBridgeServer(c, port=0, frame_callback=lambda req: {"camera": req["camera"], "pixels": [1]})
    server.start()
    reader = channel(server)
    try:
        assert "frame" in server.dispatch({"op": "hello"})["capabilities"]
        frame = reader.request({"op": "frame", "camera": "proof"})
        assert frame == {"ok": True, "camera": "proof", "pixels": [1]}
        assert c.state()["feedback_available"] is False
    finally:
        reader.close()
        server.close()


def test_server_close_cannot_join_a_registered_but_unstarted_worker(control, monkeypatch):
    import socket
    import threading
    import time
    module = _module()
    server = module.MobileBridgeServer(control[0], port=0)
    original = threading.Thread.start
    entered, release = threading.Event(), threading.Event()
    def delayed_start(thread):
        if thread.name == "mobile-rpc-client":
            entered.set()
            assert release.wait(1.)
        return original(thread)
    monkeypatch.setattr(threading.Thread, "start", delayed_start)
    server.start()
    conn = socket.create_connection(server.address, timeout=.5)
    errors = []
    def close():
        try:
            server.close()
        except Exception as e:
            errors.append(e)
    closer = threading.Thread(target=close)
    try:
        assert entered.wait(.5)
        closer.start()
        time.sleep(.02)
    finally:
        release.set()
        closer.join(1.)
        conn.close()
    assert not errors
    assert not closer.is_alive()
    wait_until(lambda: not server._workers)
    assert not server._accept_thread.is_alive()
    assert not server._watchdog_thread.is_alive()


def test_renew_requires_real_owner_and_command_even_when_no_active_command(control):
    c, _ = control
    with pytest.raises(ValueError):
        c.renew({"epoch": c.hello()["epoch"], "generation": 0})
