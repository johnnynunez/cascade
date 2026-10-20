"""A held arm is never dropped by the next start (#424 review, Important 1 and 2, Minors 1-7).

Since #424 a teardown whose park did not bring a torque-cutting arm to rest keeps it POWERED and HOLDING
(so101, rebot_rs, rebot_rs_mb), and every MCP SIGTERM ends that way. The next process could not see it:
its standby LazyArm reported `arm_connected: false`, and the reBot drivers' connect() sends
`clear_motor_faults` (a type-4 STOP frame: no torque) before enabling, so the next start, arm-check or
calibrate dropped the raised arm. Now the teardown writes a durable per-arm hold marker
(`cascade.control.arm_hold`); every connect of a held RS arm refuses BEFORE any frame until an operator
confirms at that moment (or the startup check measures the arm folded at rest), and the dashboard /state
reports the hold.

Also here: the demo CLI's exit park beside a dashboard motion (Important 2: the park's begin_motion cleared
the halt and both streamed, interleaved), a chat step that starts after the MCP teardown's halt (Minor 2), the
bounded teardown wait (follow-up 3, Minor 1), the gate-on home check before the jaw opens (Minor 3), the
confirmation read at the moment of the cut with a command that works for MCP operators (Minor 4) and the one
stage naming the park and the hold share (Minor 5).
"""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from base_pose_oracle import make_runtime, raw_arm  # noqa: F401
from conftest import observed
from test_agent_results_compose import _gate_on
from test_rebot_clear_error import _install_fake_motorbridge, _install_fake_sdk, _Log, _mb_arm, _rs_arm, no_sleep  # noqa: F401
from test_safety import limits
from test_shutdown_drive_state import hardware
from test_watchdog_frame_age import _jaw_counter

from cascade.apps import demo, mcp_server
from cascade.apps.demo import shutdown_runtime
from cascade.config import Cfg, load_demo_config
from cascade.control.arm_rig import ArmRig
from cascade.safety.harness import SafeArm, SafetyHarness

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_env_confirmation(monkeypatch):
    monkeypatch.delenv("CASCADE_TORQUE_OFF_UNPARKED", raising=False)


@pytest.fixture
def fast(monkeypatch):
    """The 2 s park streams unpaced (test_shutdown_drive_state's clock)."""
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _seconds: None)


def runtime_for(*arms, names=None):
    names = names or [str(i) for i in range(len(arms))]
    return SimpleNamespace(arm=arms[0], arm_rig=ArmRig(list(arms), list(names)), watcher=None, rig=None,
                           camera=SimpleNamespace(close=lambda: None), stream_server=None, viewer=None,
                           beliefs_path=None)


def _torque_offs(events):
    return [e for e in events if e[0] == "torque_off"]


def _dead_harness_safe(raw):
    """`raw` behind a harness whose camera never delivered: the exit park is refused (#422)."""
    harness = SafetyHarness(limits())
    harness.park_q = np.zeros(int(raw.n_joints))
    return SafeArm(raw, harness)


def _hold_a_rebot(log, make=_rs_arm):
    """Process 1: a connected reBot over the fake bus; its exit park is refused (dead camera), so its
    teardown keeps it powered and holding (#424)."""
    raw = make()
    raw.connect()
    safe = _dead_harness_safe(raw)
    receipt = shutdown_runtime(runtime_for(safe), raw)
    assert receipt["torque_held"] == {"0": "the shutdown park was refused (SafetyViolation: perception "
                                           "watchdog: no observation yet)"}, receipt
    assert not any(e[0] in ("disconnect", "disable_all", "close_bus") for e in log), log
    return raw


# ── Important 1: the restart (a new process) never drops the held arm ─────────


def test_a_restart_never_sends_a_frame_to_a_rebot_arm_its_last_teardown_held(monkeypatch, no_sleep):  # noqa: F811
    """Main: the next RebotRSArm.connect sent clear_error (the STOP frame: no torque) to all 7 motors
    of the held, raised arm, then enabled it. Now: refused before the SDK is even constructed."""
    log = _Log()
    fake = _install_fake_sdk(monkeypatch, log)
    _hold_a_rebot(log)
    n0, built = len(log), len(fake.instances)
    refused = None
    try:
        _rs_arm().connect()
    except RuntimeError as exc:
        refused = str(exc)
    sent = log[n0:]
    assert [e for e in sent if e[0] == "clear_error"] == [], f"the restart dropped the held arm: {sent}"
    assert sent == [] and len(fake.instances) == built, "a refused connect opens nothing"
    assert refused is not None and "POWERED and HOLDING" in refused and "confirm" in refused, refused
    assert "arm_hold.py" not in refused and "holding the arm" not in refused, "agent-facing: no command"


def test_a_restart_never_sends_a_frame_to_a_motorbridge_rebot_its_last_teardown_held(monkeypatch, no_sleep):  # noqa: F811
    """rebot_rs_mb: same arm, same bus (can0). Main: ensure_mode, clear_error, enable_all on the held arm."""
    log = _Log()
    controllers = []
    Controller = _install_fake_motorbridge(monkeypatch, log)
    real_init = Controller.__init__
    monkeypatch.setattr(Controller, "__init__", lambda self, channel=None: (controllers.append(channel),
                                                                            real_init(self, channel))[1])
    _hold_a_rebot(log, make=_mb_arm)
    n0, opened = len(log), len(controllers)
    with pytest.raises(RuntimeError, match="POWERED and HOLDING"):
        _mb_arm().connect()
    assert log[n0:] == [] and len(controllers) == opened, log[n0:]


def test_a_held_rebot_arm_survives_a_restart_of_the_mcp_runtime(monkeypatch, tmp_path, no_sleep):  # noqa: F811
    """The rig's own path: the MCP server restarts (`cascade-demo.sh reset`, `systemctl stop` + start)
    with a lazy rebot_rs arm. Its /state says the arm is held (main: `arm_connected: false`, the
    guard read 'standby'), and the first motion is refused with nothing sent (main: the LazyArm
    materialized and the connect's STOP frame dropped the arm)."""
    from cascade.control.rebot_rs_arm import RebotRSArm

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the profile's sdk_path insert stays in this test
    _hold_a_rebot(log)
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    occ = cfg._data.get("occupancy")
    if isinstance(occ, dict) and occ.get("required"):
        occ["required"] = False
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: RebotRSArm(acfg))
    rt, arm = demo.build_runtime(cfg, tmp_path / "run2", view=False, serve=False, lazy_arm=True)
    try:
        state = demo._runtime_state(rt)
        assert state["arm_connected"] is True, "a held arm is reported powered, never standby"
        held = state.get("arm_held")
        assert held and held[0]["key"] == "can0" and "refused" in held[0]["reason"], state.get("arm_held")
        n0 = len(log)
        out = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        assert [e for e in log[n0:] if e[0] == "clear_error"] == [], f"the restart dropped the arm: {log[n0:]}"
        assert log[n0:] == [] and out.get("ok") is False, out
        assert "POWERED and HOLDING" in str(out.get("error")), out
        assert not rt.arm.raw.connected
    finally:
        shutdown_runtime(rt, arm)


def test_the_operators_fresh_confirmation_lets_the_next_connect_through_and_clears_the_hold(
        monkeypatch, no_sleep):  # noqa: F811
    from cascade.control import arm_hold

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    _hold_a_rebot(log)
    assert arm_hold.read("can0") is not None
    arm_hold.confirm("holding the arm")  # what the operator's `arm_hold.py confirm` writes
    n0 = len(log)
    _rs_arm().connect()
    sent = [e[0] for e in log[n0:]]
    assert "clear_error" in sent and sent.index("clear_error") < sent.index("enable")
    assert arm_hold.read("can0") is None, "the confirmed connect cleared the hold"
    arm_hold.record("can0", arm="0", reason="held again")
    with pytest.raises(RuntimeError, match="POWERED and HOLDING"):  # used once: a new hold needs a new one
        _rs_arm().connect()


def test_a_stale_or_misspelled_confirmation_never_lets_a_held_arm_connect(monkeypatch, no_sleep):  # noqa: F811
    from cascade.control import arm_hold

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    _hold_a_rebot(log)
    with pytest.raises(ValueError):
        arm_hold.confirm("yes")
    arm_hold.confirm("  Holding the ARM ")  # the phrase, case and spaces aside

    class Later:
        def time(self):
            return time.time() + arm_hold.CONFIRM_TTL_S + 1.0

        def __getattr__(self, name):
            return getattr(time, name)

    monkeypatch.setattr(arm_hold, "time", Later())
    n0 = len(log)
    with pytest.raises(RuntimeError, match="POWERED and HOLDING"):
        _rs_arm().connect()
    assert log[n0:] == [] and arm_hold.read("can0") is not None


def test_an_unreadable_hold_marker_counts_as_held(monkeypatch, no_sleep):  # noqa: F811
    from cascade.control import arm_hold

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    arm_hold.hold_dir().mkdir(parents=True, exist_ok=True)
    arm_hold.marker_path("can0").write_text("{not json")
    with pytest.raises(RuntimeError, match="POWERED and HOLDING"):
        _rs_arm().connect()
    assert log == []


def test_the_motorbridge_connect_never_drops_an_enabled_arm_without_a_confirmation(monkeypatch, no_sleep):  # noqa: F811
    """rebot_rs_mb's connect, on motors an earlier session left ENABLED (no marker: a bring-up script,
    MotorBridge Studio, a crash): its run-mode write fails, and main then `disable_all()`ed (torque off,
    a printed 'SUPPORT THE ARM') with nobody confirming. Now it refuses and releases the bus (no
    disable frame); with the operator's fresh confirmation it drops torque as before."""
    from cascade.control import arm_hold

    log = _Log()
    Controller = _install_fake_motorbridge(monkeypatch, log)
    bus = {"enabled": True}
    real_add = Controller.add_robstride_motor

    def add(self, motor_id, feedback_id, model):
        motor = real_add(self, motor_id, feedback_id, model)

        def ensure_mode(mode, *a):
            log.append(("ensure_mode", motor_id))
            if bus["enabled"]:
                raise RuntimeError("ensure_mode timeout: 0x7005 never answered")
        motor.ensure_mode = ensure_mode
        return motor

    monkeypatch.setattr(Controller, "add_robstride_motor", add)
    monkeypatch.setattr(Controller, "disable_all", lambda self: (log.append(("disable_all", None)),
                                                                 bus.update(enabled=False)))
    arm = _mb_arm()
    with pytest.raises(RuntimeError, match="confirm"):
        arm.connect()
    assert ("disable_all", None) not in log, "the connect cut torque on an enabled arm unconfirmed"
    assert ("close_bus", None) in log and ("close", None) in log and arm._ctrl is None
    arm_hold.confirm("holding the arm")
    arm.connect()
    events = [e[0] for e in log]
    assert events.count("disable_all") == 1 and events.index("disable_all") < events.index("enable_all")


def test_the_motorbridge_disconnect_sends_no_disable_frame(monkeypatch, no_sleep):  # noqa: F811
    """Minor 6, from the code: rebot_rs_mb's disconnect only releases the host's handle (close_bus +
    close), no disable frame; #424's table said it leaves the motors limp. What the motors do once the
    host is gone is their firmware's CAN timeout (not verified on the rig), so teardown still treats it
    as a disconnect that may cut torque and parks first."""
    log = _Log()
    _install_fake_motorbridge(monkeypatch, log)
    arm = _mb_arm()
    arm.connect()
    n0 = len(log)
    arm.disconnect()
    assert log[n0:] == [("close_bus", None), ("close", None)]
    assert arm.disconnect_cuts_torque is True and arm.hold_key == "can0"


def test_an_so101_connect_never_cuts_the_torque_of_a_held_arm(monkeypatch):
    """Backend table: FeetechArm.connect writes mode, torque limit, acceleration and torque ENABLE;
    nothing turns a servo's torque off or resets it, so connecting over a held so101 drops nothing
    (the servos keep their last goal). It is not refused; its hold is still reported and stays until a
    parked torque-off."""
    from cascade.control import arm_hold, feetech_arm
    from cascade.control.feetech import ADDR_TORQUE_ENABLE

    writes = []

    class Bus:
        def __init__(self, port, endian=None):
            pass

        def ping(self, sid):
            return True

        def write_reg(self, sid, address, value):
            writes.append((sid, address, value))

        def read_reg(self, sid, address):
            return 2048

    monkeypatch.setattr(feetech_arm, "ServoBus", Bus)
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=lambda *a, **k: SimpleNamespace(
        close=lambda: None)))
    arm_hold.record("/dev/ttyACM0", arm="0", reason="test hold")
    arm = feetech_arm.FeetechArm(Cfg({"n_joints": 5, "servo_ids": [1, 2, 3, 4, 5], "gripper_id": 6}))
    monkeypatch.setattr(arm, "_temperatures", lambda: {})
    arm.connect()
    assert [w for w in writes if w[1] == ADDR_TORQUE_ENABLE and w[2] == 0] == []
    assert arm.hold_key == "/dev/ttyACM0" and arm_hold.read("/dev/ttyACM0") is not None


def test_the_startup_check_clears_a_hold_only_when_it_measures_the_arm_folded_at_rest(monkeypatch, capsys):
    """`~/cascade-demo.sh up` runs scripts/check_rebot_gripper.py: it reads the arm with NO motor
    enabled. At park_q the hold is cleared (the connect's STOP frame then drops nothing); raised, it
    refuses as before and the hold stays."""
    from test_rebot_gripper_reference import _ArmFactory, _check_script

    import cascade.control.rebot_rs_arm as rs
    from cascade.control import arm_hold

    for q, code in (([0, 1.2, 1.2, 0, 0, 0], 2), ([0.001, 0.0, -0.002, 0.0, 0.0, 0.003], 0)):
        arm_hold.record("can0", arm="0", reason="the e-stop is latched, so the shutdown park never moved it")
        mod = _check_script()
        calls = []
        factory = _ArmFactory(SimpleNamespace(get_state=None))
        monkeypatch.setattr(mod, "read_rest_pose", lambda q=q: q)
        monkeypatch.setattr(mod, "run", lambda arm, park, **k: calls.append("run") or 0)
        monkeypatch.setattr(rs, "RebotRSArm", factory)
        assert mod.main(["--yes", "--reference-only"]) == code
        assert (arm_hold.read("can0") is None) is (code == 0), capsys.readouterr().out
        assert factory.calls == ([] if code else ["construct", "connect", "disconnect"])
        arm_hold.clear("can0", why="test")


def test_clear_if_at_rest_holds_its_own_tolerance_whatever_the_caller_checked():
    """The clear is its own gate (any caller, not only the startup check after rests_unpowered): a
    joint just past the tolerance, an unread joint, or a pose of the wrong length keeps the hold."""
    from cascade.control import arm_hold

    park, tol = [0.0] * 6, 0.05
    for q in ([0.0, 0.0, tol + 0.01, 0.0, 0.0, 0.0], [0.0, None, 0.0, 0.0, 0.0, 0.0], [0.0] * 5):
        arm_hold.record("can0", arm="0", reason="held")
        assert arm_hold.clear_if_at_rest("can0", q, park, tol, log=lambda *_: None) is False, q
        assert arm_hold.read("can0") is not None, q
    assert arm_hold.clear_if_at_rest("can0", [0.0, 0.0, tol - 0.01, 0.0, 0.0, 0.0], park, tol,
                                     log=lambda *_: None) is True
    assert arm_hold.read("can0") is None and arm_hold.clear_if_at_rest("can0", park, park, tol) is False


# ── the teardown writes the marker, reads the confirmation at the cut, names its command ─


def test_the_teardown_reads_the_confirmation_at_the_moment_of_the_cut_and_says_how(fast, capsys, tmp_path):
    """Minor 4: the confirmation was an env variable set when the process started (stale), and the
    server.log warning told MCP operators to set it although no registration forwards it. Now the
    warning names a command that works from any shell (absolute interpreter and script path; the
    operator types the phrase at its prompt, #426 review Important 4), and a confirmation written just
    before the cut lets that teardown cut torque."""
    from cascade.control import arm_hold

    events = []
    arm = hardware(events)
    arm.harness._last_heartbeat = -np.inf  # dead camera: the park is refused
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert _torque_offs(events) == [] and list(receipt["torque_held"]) == ["0"]
    assert arm_hold.read("/dev/ttyACM0")["reason"] == receipt["torque_held"]["0"]
    err = capsys.readouterr().err
    script = REPO / "scripts" / "arm_hold.py"
    command = f"{sys.executable} {script} confirm"
    assert command in err and "holding the arm" not in err.split(command)[1][:3], err
    arm_hold.confirm("holding the arm")  # what that command writes once the operator typed the phrase
    events2 = []
    arm2 = hardware(events2)
    arm2.harness._last_heartbeat = -np.inf
    receipt2 = shutdown_runtime(runtime_for(arm2), arm2.raw)
    assert len(_torque_offs(events2)) == 6 and "torque_held" not in receipt2
    assert arm_hold.read("/dev/ttyACM0") is None, "torque cut: nothing is held any more"
    assert "operator confirmed" in capsys.readouterr().err


def test_a_two_arm_rig_names_its_held_arm_as_the_park_named_it(fast):
    """Minor 5: one stage list for the park and the hold. Arms named left/right on their own ports:
    only `right`'s park is refused, only `right` is held and marked."""
    from cascade.control import arm_hold

    left, right = hardware([]), hardware(events := [])
    right.raw._port_name = "/dev/ttyACM1"
    right.harness._last_heartbeat = -np.inf
    receipt = shutdown_runtime(runtime_for(left, right, names=["left", "right"]), left.raw)
    assert list(receipt["torque_held"]) == ["right"] and _torque_offs(events) == []
    assert arm_hold.read("/dev/ttyACM1")["arm"] == "right" and arm_hold.read("/dev/ttyACM0") is None


# ── Important 2: the demo CLI's exit park never streams beside a dashboard motion ─


def _paced(raw, log):
    real = raw.send_joint_target

    def send(q):
        time.sleep(0.02)
        log.append((threading.current_thread().name, time.monotonic()))
        return real(q)

    raw.send_joint_target = send


@pytest.mark.parametrize("exit_kind", ["normal", "signal"])
def test_the_cli_exit_park_never_streams_beside_a_dashboard_motion(make_runtime, exit_kind):  # noqa: F811
    """The reviewer's probe: a dashboard task streams a slow move when the CLI exits (EOF, or a signal
    whose halt the park's own begin_motion cleared). Main: 104-105 task commands after the exit
    started, interleaved with the 102-command park, then 'mock arm not connected'. Now: the task's move
    halts at its next waypoint, the park waits for it, then runs alone."""
    rt = make_runtime(load_demo_config(arm="mock", camera="mock", llm="mock"))
    assert rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0}).get("ok")
    raw = raw_arm(rt.arm)
    log = []
    _paced(raw, log)
    streaming, result = threading.Event(), {}
    real_approve = rt.arm.harness.approve
    rt.arm.harness.approve = lambda *a, **k: (streaming.set(), real_approve(*a, **k))[1]
    task = threading.Thread(target=lambda: result.setdefault(
        "out", rt.execute("rotate_joint", {"joint": 1, "degrees": -60.0})), name="dashboard-task", daemon=True)
    task.start()
    assert streaming.wait(5.0)
    time.sleep(0.2)
    t_call = time.monotonic()
    if exit_kind == "signal":  # main()'s SignalRequest path
        for safe in list(rt.arm_rig):
            if demo._shutdown_park_required(safe.raw):
                safe.harness.halt("signal: graceful shutdown")
    receipt = shutdown_runtime(rt, rt.arm.raw)
    task.join(10.0)
    task_sends = [t for n, t in log if n == "dashboard-task"]
    after = [t for t in task_sends if t > t_call]
    park = [t for n, t in log if n != "dashboard-task" and t > t_call]
    assert park and min(park) > max(task_sends), "a task command after the park started: two motions"
    assert len(after) <= 1, f"{len(after)} task commands after the exit started"
    out = result["out"]
    assert out.get("ok") is False and "halted" in str(out.get("error")), out
    assert receipt["ok"] is True and not rt.arm.harness.estopped


def test_a_harness_closed_for_teardown_refuses_every_other_threads_motion_and_jaw():
    """The gate itself: after close_for_teardown only the closing thread (the park) may begin a motion,
    pass a pre-motion or stream-start check, or command the jaws; any other thread gets MotionHalted
    before its first command, and the halt it raised is not cleared by that refusal."""
    from cascade.control.mock_arm import MockArm
    from cascade.types import MotionHalted

    raw = MockArm(Cfg({"n_joints": 6}))
    raw.connect()
    jaws = []
    raw.set_gripper = lambda *a, **k: jaws.append(a)
    safe = SafeArm(raw, observed(SafetyHarness(limits())))
    h = safe.harness
    h.begin_motion()  # a motion streaming on this thread
    assert h.close_for_teardown("shutting down") is True and h.halted == "shutting down"
    h.end_motion()
    seen = {}

    def other():
        for name, call in (("begin", h.begin_motion), ("before", h.check_before_motion),
                           ("stream", h.check_stream_start), ("jaw", lambda: safe.set_gripper(0.0))):
            try:
                call()
                seen[name] = "allowed"
            except MotionHalted as exc:
                seen[name] = str(exc)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join(5.0)
    assert seen == dict.fromkeys(("begin", "before", "stream", "jaw"), "halted: shutting down"), seen
    assert h.halted == "shutting down" and jaws == []
    h.begin_motion()  # the closing thread (the park) still may
    h.end_motion()
    safe.set_gripper(0.0)
    assert len(jaws) == 1
    # Closed while nothing streams (no halt to rely on), or after the park's own begin_motion cleared the
    # halt: the gate alone still refuses every other thread, the stream-start check included.
    for closed_idle in (True, False):
        h2 = observed(SafetyHarness(limits()))
        if closed_idle:
            assert h2.close_for_teardown("shutting down") is False and h2.halted is None
        else:
            h2.begin_motion()
            h2.close_for_teardown("shutting down")
            h2.end_motion()
            h2.clear_halt()
            assert h2.halted is None
        seen.clear()
        thread = threading.Thread(target=lambda: [seen.__setitem__(n, _refused(c)) for n, c in
                                                  (("begin", h2.begin_motion), ("before", h2.check_before_motion),
                                                   ("stream", h2.check_stream_start))])
        thread.start()
        thread.join(5.0)
        assert seen == dict.fromkeys(("begin", "before", "stream"), "halted: shutting down"), (closed_idle, seen)


def _refused(call):
    from cascade.types import MotionHalted

    try:
        call()
        return "allowed"
    except MotionHalted as exc:
        return str(exc)


# ── Minor 2: a chat step that starts after the MCP teardown's halt sends nothing ─


def test_a_chat_motion_that_starts_after_the_teardown_halt_sends_nothing(make_runtime, tmp_path,  # noqa: F811
                                                                           monkeypatch):
    """The chat holds `_exec_lock` in a non-motion step, then starts a motion after the teardown halted.
    Main: that motion's begin_motion cleared the halt and it ran (281 commands) before the park."""
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    rt = make_runtime(load_demo_config(arm="mock", camera="mock", llm="mock"), lazy=True)
    assert rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0}).get("ok")
    log = []
    _paced(raw_arm(rt.arm), log)
    server = mcp_server.McpSkillServer()
    server._runtime, server._arm = rt, rt.arm.raw
    holding, result = threading.Event(), {}

    def chat():
        with server._exec_lock:
            holding.set()
            time.sleep(0.5)
            result["out"] = rt.execute("rotate_joint", {"joint": 1, "degrees": -30.0})

    thread = threading.Thread(target=chat, name="chat", daemon=True)
    thread.start()
    assert holding.wait(5.0)
    t_call = time.monotonic()
    receipt = server.shutdown()
    thread.join(10.0)
    assert [t for n, t in log if n == "chat" and t > t_call] == []
    assert result["out"].get("ok") is False and "halted" in str(result["out"].get("error")), result
    assert [t for n, t in log if n != "chat" and t > t_call], "the park ran"
    assert receipt["ok"] is True


# ── follow-up 3 + Minor 1: the teardown's wait is bounded, then the e-stop latches ─


def _server_with_a_held_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    events = []
    arm = hardware(events)  # a live camera: only the wait can stop this park
    server = mcp_server.McpSkillServer()
    server._runtime, server._arm = runtime_for(arm), arm.raw
    release, holding = threading.Event(), threading.Event()

    def holder():  # a dashboard call that never returns
        with server._exec_lock:
            holding.set()
            release.wait(30.0)

    threading.Thread(target=holder, name="stuck-call", daemon=True).start()
    assert holding.wait(5.0)
    return server, arm, events, release


def _shutdown_within(server, seconds):
    box = {}

    def run():
        box["receipt"] = server.shutdown()

    thread = threading.Thread(target=run, name="teardown", daemon=True)
    t0 = time.monotonic()
    thread.start()
    thread.join(seconds)
    return box, thread, time.monotonic() - t0


@pytest.mark.parametrize("why", ["bound", "signal", "second_signal"])
def test_the_mcp_teardown_waits_a_bounded_time_then_latches_the_estop_and_holds(tmp_path, monkeypatch,
                                                                                capsys, why):
    """A call holding `_exec_lock` never returns. Main: the teardown waited forever; one SIGTERM was
    deferred and did nothing; a second one raised KeyboardInterrupt out of the wait (no receipt, arm
    left powered). Now: after the bound (or at once on a stop signal) the e-stop latches (motion stops,
    the arm keeps its torque), the park is skipped and the teardown finishes with its receipt."""
    server, arm, events, release = _server_with_a_held_lock(tmp_path, monkeypatch)
    monkeypatch.setattr(demo, "TEARDOWN_WAIT_S", 0.3 if why == "bound" else 30.0, raising=False)
    if why == "signal":  # StopSignals defers a first SIGINT/SIGTERM during teardown: signum is all it sets
        server._signals = SimpleNamespace(signum=None)
        threading.Timer(0.3, lambda: setattr(server._signals, "signum", signal.SIGTERM)).start()
    if why == "second_signal":  # a second SIGINT/SIGTERM raises KeyboardInterrupt inside the wait
        real = server._exec_lock

        class Interrupted:
            def acquire(self, blocking=True, timeout=-1):
                if not blocking:
                    return real.acquire(False)
                time.sleep(0.2)
                raise KeyboardInterrupt

            def release(self):
                real.release()

        server._exec_lock = Interrupted()
    try:
        box, thread, took = _shutdown_within(server, 8.0)
        assert not thread.is_alive(), "the teardown is still waiting for the stuck call"
    finally:
        release.set()
    receipt = box["receipt"]
    assert took < 5.0 and arm.harness.estopped
    assert _torque_offs(events) == [] and not [e for e in events if e[0] == "joint_target"]
    saved = json.loads((tmp_path / "run" / "teardown.json").read_text())
    stage = next(s for s in saved["stages"] if s["stage"] == "runtime")
    assert stage["result"]["torque_held"] == {"0": "the e-stop is latched, so the shutdown park never moved it"}
    assert receipt["ok"] is False
    assert "e-stop latched" in capsys.readouterr().err


def test_the_cli_teardown_waits_a_bounded_time_for_a_stuck_motion_then_latches_and_holds(monkeypatch):
    """shutdown_runtime (every CLI exit) beside a motion that never reaches its next waypoint. Main:
    the park started at once beside it. Now: halt, wait (bounded), then the e-stop: no park, torque kept."""
    events = []
    arm = hardware(events)
    monkeypatch.setattr(demo, "TEARDOWN_WAIT_S", 0.3, raising=False)
    inside, stuck = threading.Event(), threading.Event()
    arm.raw.stream_to = lambda *a, **k: (inside.set(), stuck.wait(20.0), False)[2]
    target = np.asarray(arm.get_state().q, dtype=float) + np.array([0.0, 0.1, 0.0, 0.0, 0.0])
    motion = threading.Thread(target=lambda: arm.move_joints(target, duration_s=2.0), name="stuck", daemon=True)
    motion.start()
    assert inside.wait(5.0)
    box = {}
    teardown = threading.Thread(target=lambda: box.setdefault("r", shutdown_runtime(runtime_for(arm), arm.raw)),
                                daemon=True)
    teardown.start()
    teardown.join(8.0)
    alive = teardown.is_alive()
    stuck.set()
    teardown.join(10.0)
    motion.join(10.0)
    assert not alive, "the teardown waited past its bound"
    assert arm.harness.estopped and _torque_offs(events) == []
    assert box["r"]["torque_held"] == {"0": "the e-stop is latched, so the shutdown park never moved it"}


# ── Minor 3: with the observed-finger gate on, the home move is vetted before the jaws open ─


def test_with_the_gate_on_a_refused_home_segment_costs_no_jaw_command(make_runtime, monkeypatch):  # noqa: F811
    """Isaac reBot sim, CASCADE_OBSERVED_FINGER_GATE=1: the home move's segment was vetted inside its
    own preflight, after the pre-grasp jaw opening, so a refusal there still cost 1 jaw command."""
    from cascade.safety import trajectory

    rt = make_runtime(load_demo_config(arm="mock", camera="mock", llm="mock"))
    _gate_on(monkeypatch, raw_arm(rt.arm))
    home = np.asarray(rt.cfg.arm.home_q, dtype=float)
    real_vet = trajectory.vet_segment

    def vet(harness, start, target, *a, **k):
        if np.allclose(np.asarray(target, dtype=float), home):
            return "injected: the home segment crosses an obstacle"
        return real_vet(harness, start, target, *a, **k)

    monkeypatch.setattr(trajectory, "vet_segment", vet)
    jaws = _jaw_counter(rt)
    out = rt.execute("grasp_object", {"label": "red cube"})
    assert out.get("ok") is False and "home segment" in str(out.get("error")), out
    assert sum(jaws.values()) == 0, f"{jaws} jaw command(s) before the refused home segment"
