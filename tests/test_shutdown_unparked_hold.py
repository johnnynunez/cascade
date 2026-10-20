"""A refused shutdown park never cuts torque on a raised arm (#422 review follow-up 2).

Teardown parks every arm, then disconnects it, and on a torque-cutting backend (so101's serial
torque-off, the reBot SDK's `disable_all`, and motorbridge's `close_bus`, which sends no disable frame
but leaves the motors to their firmware's CAN timeout, unverified: treated as a possible drop) the
disconnect drops an arm that is not at rest. A dead camera refuses the park (the
perception watchdog), and so does any other gate; a latched e-stop skips it (every MCP SIGTERM:
`launch.sh --down`, `openshell_rig.py teardown`, `systemctl stop` latch the e-stop first). Main then
disconnected anyway and the raised arm fell.

Ruling (the SAFE option; no gate is relaxed): the park stays refused, and an arm whose park did not
bring it to rest keeps its torque on and holds; the receipt (`torque_held`) and stderr say so per
arm. Cutting torque there needs the operator's explicit confirmation,
`CASCADE_TORQUE_OFF_UNPARKED='holding the arm'`, never anything automatic. A parked arm, a
controller-held arm (ros2), a simulator (mujoco, isaac), the Unitree arm SDK (its weight ramp hands
the arm back to the robot's own controller) and the mock disconnect as before.

The MCP server's teardown park also runs under `_exec_lock` (#422 review follow-up 1): a motion
still running on the dashboard thread is halted and waited for, so two motions never share one
harness.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import observed
from test_safety import limits
from test_shutdown_drive_state import hardware

from cascade.apps import demo, mcp_server
from cascade.apps.demo import TORQUE_OFF_CONFIRM_ENV, shutdown_runtime
from cascade.config import Cfg
from cascade.control.arm_rig import ArmRig
from cascade.control.feetech_arm import FeetechArm
from cascade.control.isaac_arm import IsaacArm
from cascade.control.lazy_arm import LazyArm
from cascade.control.mock_arm import MockArm
from cascade.control.mujoco_arm import MujocoArm
from cascade.control.rebot_rs_arm import RebotRSArm
from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm
from cascade.control.ros2_arm import Ros2Arm
from cascade.control.unitree_arm import UnitreeArm
from cascade.safety.harness import SafeArm, SafetyHarness
from cascade.types import MotionHalted, SafetyViolation


@pytest.fixture(autouse=True)
def _no_confirmation(monkeypatch):
    monkeypatch.delenv(TORQUE_OFF_CONFIRM_ENV, raising=False)


@pytest.fixture
def fast(monkeypatch):
    """The 2 s park streams unpaced (test_shutdown_drive_state's clock)."""
    monkeypatch.setattr("cascade.control.arm_base.time.sleep", lambda _seconds: None)


def runtime_for(*arms):
    return SimpleNamespace(arm=arms[0], arm_rig=ArmRig(list(arms), [str(i) for i in range(len(arms))]),
                           watcher=None, rig=None, camera=SimpleNamespace(close=lambda: None),
                           stream_server=None, viewer=None, beliefs_path=None)


def feetech(events, *, fresh):
    """A real FeetechArm (so101) with its register I/O in memory; `fresh`: its camera delivered."""
    arm = hardware(events)
    if not fresh:
        arm.harness._last_heartbeat = -np.inf  # the camera never delivered (a dead camera)
    return arm


def _torque_offs(events):
    return [e for e in events if e[0] == "torque_off"]


# ── one so101 arm, the whole real path (park refused by the gate, then the arms stage) ──


@pytest.mark.parametrize("why", ["watchdog", "estop", "midway"])
def test_a_refused_park_keeps_a_torque_cutting_arm_powered_and_says_so(fast, capsys, why):
    """The park is refused (a dead camera's watchdog; the e-stop latched by a SIGTERM; a gate that
    stops it midway, after 4 commands) and the so101's torque stays ON: no torque-off write, the
    port stays open, the receipt and stderr name the arm and why. Main: 6 torque-off writes, the
    raised arm dropped."""
    events = []
    arm = feetech(events, fresh=why != "watchdog")
    if why == "estop":
        arm.stop()  # SafeArm.stop(): what the MCP server's stop_now does on SIGTERM
    if why == "midway":
        real, n = arm.harness.approve, []

        def approve(*a, **k):
            n.append(1)
            if len(n) == 5:
                raise SafetyViolation("TCP outside workspace (5th command)")
            return real(*a, **k)

        arm.harness.approve = approve
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert _torque_offs(events) == [] and ("port_closed",) not in events, events
    assert arm.raw._connected is True
    assert receipt["ok"] is False and receipt["complete"] is False
    reason = receipt["torque_held"]["0"]
    assert reason == {
        "watchdog": "the shutdown park was refused (SafetyViolation: perception watchdog: no observation yet)",
        "estop": "the e-stop is latched, so the shutdown park never moved it",
        "midway": "the shutdown park was refused (SafetyViolation: TCP outside workspace (5th command))"}[why]
    assert len([e for e in events if e[0] == "joint_target"]) == (4 if why == "midway" else 0)
    arms = next(s for s in receipt["stages"] if s["stage"] == "arms")
    assert arms["result"]["stages"] == [{"stage": "0", "ok": False, "complete": False, "holding": True,
                                          "errors": [{"type": "TorqueHeld", "message": reason}]}]
    err = capsys.readouterr().err
    assert f"arm 0 is POWERED and HOLDING its pose: {reason}" in err, err
    assert "CASCADE_TORQUE_OFF_UNPARKED='holding the arm'" in err


def test_a_parked_arm_still_has_its_torque_cut(fast):
    """Control: a live camera, the park completes (100 joint targets) and the torque goes off after it,
    exactly as on main; nothing is held."""
    events = []
    arm = feetech(events, fresh=True)
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    targets = [i for i, e in enumerate(events) if e[0] == "joint_target"]
    offs = [i for i, e in enumerate(events) if e[0] == "torque_off"]
    assert len(targets) == 100 and len(offs) == 6 and max(targets) < min(offs)
    assert "torque_held" not in receipt and receipt["ok"] is True


@pytest.mark.parametrize("value, cut", [("holding the arm", True), ("  Holding the Arm ", True), ("1", False),
                                        ("yes", False), ("", False), ("holding", False)])
def test_only_the_operators_exact_confirmation_cuts_torque_on_an_unparked_arm(fast, monkeypatch, capsys, value, cut):
    events = []
    arm = feetech(events, fresh=False)
    monkeypatch.setenv(TORQUE_OFF_CONFIRM_ENV, value)
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert len(_torque_offs(events)) == (6 if cut else 0), events
    assert ("torque_held" in receipt) is not cut
    err = capsys.readouterr().err
    assert ("cutting its torque anyway: the operator confirmed" in err) is cut, err


def test_a_two_arm_rig_holds_only_the_arm_whose_park_was_refused(fast, capsys):
    """Arm 0's camera is live and it parks; arm 1's never delivered: arm 0 is disconnected (torque
    off after its park), arm 1 keeps its torque."""
    parked, refused = [], []
    a0, a1 = feetech(parked, fresh=True), feetech(refused, fresh=False)
    receipt = shutdown_runtime(runtime_for(a0, a1), a0.raw)
    assert len(_torque_offs(parked)) == 6 and parked[-1] == ("port_closed",)
    assert _torque_offs(refused) == [] and a1.raw._connected is True
    assert list(receipt["torque_held"]) == ["1"]
    arms = next(s for s in receipt["stages"] if s["stage"] == "arms")["result"]["stages"]
    assert [(s["stage"], s["ok"], s.get("holding", False)) for s in arms] == [("0", True, False), ("1", False, True)]


def test_a_park_stage_that_failed_whole_still_holds(fast, monkeypatch):
    """Conservative side: the park stage itself raised (no per-arm receipt): nothing says the arm is at
    rest, so a torque-cutting arm stays powered."""
    events = []
    arm = feetech(events, fresh=True)

    def broken(_runtime):
        raise RuntimeError("park bookkeeping failed")

    monkeypatch.setattr(demo, "_park_arm", broken)
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert _torque_offs(events) == [] and receipt["torque_held"] == {"0": "the shutdown park did not run"}


# ── every backend: who is held, who is disconnected ───────────────────────


def _backend(kind, events):
    """`kind`'s real class, never connected (object.__new__), with its disconnect's own I/O faked so a
    call is recorded; the real disconnect() runs."""
    rec = lambda name: (lambda *a, **k: events.append(name))  # noqa: E731
    if kind == "so101":
        return hardware(events).raw
    if kind == "rebot_rs":
        raw = object.__new__(RebotRSArm)
        raw._arm = SimpleNamespace(disconnect=rec("sdk disconnect: disable_all"))
    elif kind == "rebot_rs_mb":
        raw = object.__new__(RebotRSMotorBridgeArm)
        raw._ctrl, raw._motors = SimpleNamespace(close_bus=rec("close_bus: no disable frame, motors to the "
                                                             "firmware's CAN timeout"), close=rec("close")), {}
    elif kind == "mujoco":
        raw = object.__new__(MujocoArm)
        raw._engine, raw._connected = SimpleNamespace(close=rec("engine closed")), True
    elif kind == "unitree_arm":
        raw = object.__new__(UnitreeArm)
        raw.connected, raw._pub, raw._executor, raw._spin_thread = True, None, None, None
        raw._node = SimpleNamespace(destroy_node=rec("node destroyed (weight ramp hands the arm back)"))
    elif kind == "ros2":
        raw = object.__new__(Ros2Arm)
        raw._motion_active, raw._arm_goal, raw._executor, raw._spin_thread = False, None, None, None
        raw._node = SimpleNamespace(destroy_node=rec("node destroyed (controller holds)"))
    elif kind == "isaac":
        raw = object.__new__(IsaacArm)
        raw._client = SimpleNamespace(close=rec("bridge client closed"))
    elif kind == "mock":
        raw = MockArm(Cfg({"n_joints": 6}))
        raw.connect()
        raw.disconnect = rec("mock disconnected")
    return raw


BACKENDS = {"so101": True, "rebot_rs": True, "rebot_rs_mb": True, "mujoco": False, "unitree_arm": False,
            "ros2": False, "isaac": False, "mock": False}


@pytest.mark.parametrize("kind", list(BACKENDS))
def test_every_backend_declares_whether_its_disconnect_cuts_torque(kind):
    raw = _backend(kind, [])
    assert raw.disconnect_cuts_torque is BACKENDS[kind]
    lazy = LazyArm(lambda: pytest.fail("teardown must never materialize an arm"))
    assert lazy.disconnect_cuts_torque is False  # standby: never powered
    lazy._arm = raw
    assert lazy.disconnect_cuts_torque is BACKENDS[kind]


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("kind", list(BACKENDS))
def test_a_refused_park_holds_exactly_the_torque_cutting_backends(kind, lazy):
    """The park refused by the watchdog on each backend's real class: rebot_rs / rebot_rs_mb / so101
    stay connected and holding; the others disconnect as on main (ros2's park is never attempted:
    its controller keeps the hold)."""
    events = []
    raw = _backend(kind, events)
    if lazy:
        wrapper = LazyArm(lambda: pytest.fail("teardown must never materialize an arm"))
        wrapper._arm = raw
        raw = wrapper
    events.clear()

    def refused(*_a, **_k):  # the real harness's refusal of a dead camera's park
        raise SafetyViolation("perception watchdog: no observation yet")

    safe = SimpleNamespace(harness=SimpleNamespace(estopped=False, park_q=[0.0] * 6, kin=None), raw=raw,
                           n_joints=6, move_joints=refused)
    rt = runtime_for(safe)
    rt.arm_rig = None
    receipt = shutdown_runtime(rt, raw)
    if BACKENDS[kind]:
        assert events == [] and receipt["torque_held"] == {
            "arm": "the shutdown park was refused (SafetyViolation: perception watchdog: no observation yet)"}
    else:
        assert len(events) >= 1 and "torque_held" not in receipt, (events, receipt)


# ── the MCP server: SIGTERM's e-stop, and the park under the exec lock ────


def test_mcp_sigterm_estop_then_teardown_holds_a_raised_so101(fast, tmp_path, monkeypatch):
    """`systemctl stop` / `launch.sh --down` / `openshell_rig.py teardown`: SIGTERM -> stop_now()
    latches the e-stop -> shutdown(): the park is skipped (estopped) and, now, the arm keeps its
    torque; teardown.json carries `torque_held`. Main: torque off on the e-stopped arm."""
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    events = []
    arm = feetech(events, fresh=True)
    server = mcp_server.McpSkillServer()
    server._runtime, server._arm = runtime_for(arm), arm.raw
    server.stop_now()
    receipt = server.shutdown()
    assert arm.harness.estopped and _torque_offs(events) == [], events
    saved = json.loads((tmp_path / "teardown.json").read_text())
    runtime_stage = next(s for s in saved["stages"] if s["stage"] == "runtime")
    assert runtime_stage["result"]["torque_held"] == {"0": "the e-stop is latched, so the shutdown park never moved it"}
    assert receipt["ok"] is False


def _mock_safe():
    """A MockArm paced like a real 50 Hz stream (MockArm itself streams unpaced)."""
    raw = MockArm(Cfg({"n_joints": 6}))
    raw.connect()
    real_send = raw.send_joint_target
    raw.send_joint_target = lambda q: (time.sleep(0.02), real_send(q))[1]
    harness = observed(SafetyHarness(limits(max_joint_vel=0.5)))
    harness.park_q = np.zeros(6)
    return SafeArm(raw, harness)


def test_the_mcp_teardown_park_never_shares_the_harness_with_a_running_motion(tmp_path, monkeypatch):
    """The dashboard's reflex chat holds `_exec_lock` and streams a slow move when the host closes
    stdin: the teardown halts it (MotionHalted at its next waypoint), waits for its call, then parks.
    Every park command comes after the chat's last one, and the chat's move ends halted, not e-stopped.
    Main parked at once beside the running move: two motions on one harness, commands interleaved."""
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))  # real pacing: the overlap is in wall-clock time
    safe = _mock_safe()
    order, outcome = [], {}
    real_approve = safe.harness.approve

    def approve(*a, **k):
        out = real_approve(*a, **k)
        order.append(threading.current_thread().name)
        return out

    safe.harness.approve = approve
    server = mcp_server.McpSkillServer()
    server._runtime, server._arm = runtime_for(safe), safe.raw
    streaming, ended = threading.Event(), []

    def chat():
        with server._exec_lock:
            start = np.array(safe.get_state().q, dtype=float)
            target = start + np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
            safe.harness.approve = lambda *a, **k: (streaming.set(), approve(*a, **k))[1]
            try:
                outcome["moved"] = safe.move_joints(target, duration_s=4.0)
            except MotionHalted as exc:
                outcome["halted"] = str(exc)
            finally:
                safe.harness.approve = approve
                ended.append(time.monotonic())

    thread = threading.Thread(target=chat, name="chat")
    thread.start()
    assert streaming.wait(5.0), "test setup: the chat's move is streaming"
    t0 = time.monotonic()
    receipt = server.shutdown()
    thread.join(10.0)
    assert not thread.is_alive()
    assert outcome == {"halted": "halted: server shutting down"}, outcome
    assert "chat" in order and order.count("MainThread") > 0, order
    first_park = order.index("MainThread")
    assert "chat" not in order[first_park:], "a chat command after the park started: two motions on one harness"
    assert ended and ended[0] - t0 < 1.5, "halted at its next waypoint, not waited out"
    assert not safe.harness.estopped and receipt["ok"] is True, receipt
    assert server._exec_lock.acquire(blocking=False), "the teardown released the exec lock"
    server._exec_lock.release()


# ── the calibration CLI's cleanup: the same rule ──────────────────────────


def test_the_calibration_cleanup_holds_an_arm_its_park_left_raised(fast, capsys):
    from cascade.calibration import cli

    events = []
    arm = feetech(events, fresh=False)
    dead = SimpleNamespace(get_frame=lambda: (_ for _ in ()).throw(RuntimeError("camera unplugged")),
                           close=lambda: events.append(("camera_closed",)))
    cli._cleanup(SimpleNamespace(camera=dead, safe_arm=arm, raw_arm=arm.raw), park=True)
    assert _torque_offs(events) == [] and events == [("camera_closed",)], events
    assert "arm arm is POWERED and HOLDING its pose" in capsys.readouterr().err


def test_the_calibration_cleanup_still_releases_a_parked_arm(fast):
    from cascade.calibration import cli

    events = []
    arm = feetech(events, fresh=False)
    frame = SimpleNamespace(t=time.monotonic())
    cli._cleanup(SimpleNamespace(camera=SimpleNamespace(get_frame=lambda: frame, close=lambda: None),
                                 safe_arm=arm, raw_arm=arm.raw), park=True)
    assert len(_torque_offs(events)) == 6


def test_demo_keeps_the_names_the_docs_and_the_pr_give_the_operator():
    assert (demo.TORQUE_OFF_CONFIRM_ENV, demo.TORQUE_OFF_CONFIRMATION) == ("CASCADE_TORQUE_OFF_UNPARKED",
                                                                          "holding the arm")
    assert FeetechArm.disconnect_cuts_torque.fget is not None  # the ArmBase default, not an override
