"""Every path that leaves an arm held remembers it, and a held arm is never dropped, never called safe
to power off and never released by an agent (#426 review: Important 1-4, Minors 1-9).

- Important 1: the calibration CLI's cleanup, the arm-check's soft stops, the startup check's and the
  bring-up scripts' exits that leave an arm powered now write the hold marker the runtime teardown
  writes, so the next connect sends nothing. And with no marker at all (a killed process, an older
  version) a reBot connect measures the arm with parameter reads BEFORE its mode write and fault clear:
  not folded at rest, it is refused like a marker. The SDK reBot's marker names the channel of its
  hardware YAML (Minor 6).
- Important 2: `move_to_rest` on a held standby arm reports it held, never `safe_to_power_off`.
- Important 3: a confirmed connect that fails keeps the marker and the confirmation; a successful one
  uses the confirmation, so a later hold needs a new one.
- Important 4: no agent-facing text names the confirm command, and `arm_hold.py confirm` reads the
  phrase at the controlling terminal of an interactive session outside an agent's environment.
- Minors: a malformed confirmation never aborts a teardown (1); `list` fails closed (2); an SO-101's
  hold is not announced as refused (3); world_state says what arm_connected means for a held arm (4);
  the start-time variable no longer cuts torque (5, tests/test_shutdown_unparked_hold.py); the
  arm-check over a hold exits cleanly (7); the tool error names the hold (8); the reBot bulk jaw close
  honours the teardown gate (9).
"""

from __future__ import annotations

import json
import os
import select
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from test_held_arm_across_restart import _dead_harness_safe, _hold_a_rebot, _torque_offs, fast, runtime_for  # noqa: F401
from test_rebot_clear_error import (_install_fake_motorbridge, _install_fake_sdk, _Log, _mb_arm, _Motor,  # noqa: F401
                                    _rs_arm, no_sleep)
from test_shutdown_drive_state import hardware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))  # home_rebot_rs, the bring-up scripts

from cascade.apps import demo, mcp_server
from cascade.apps.demo import shutdown_runtime
from cascade.config import Cfg, load_demo_config
from cascade.control import arm_hold

REPO = Path(__file__).resolve().parent.parent
#: what can drop a held arm: the fault clear (a STOP frame), a disable, the SDK's disconnect (disable_all +
#: shutdown), a controller shutdown, an e-stop
TORQUE = ("clear_error", "disable_all", "disable", "estop", "shutdown", "disconnect")


@pytest.fixture(autouse=True)
def _no_env_confirmation(monkeypatch, tmp_path):
    monkeypatch.delenv("CASCADE_TORQUE_OFF_UNPARKED", raising=False)
    assert str(arm_hold.hold_dir()).startswith(str(tmp_path)), "never the host's hold directory"
    yield
    assert str(arm_hold.hold_dir()).startswith(str(tmp_path)), "a test undid the hold-dir pin"


def _joint(name) -> int | None:
    if isinstance(name, int):
        return name if 1 <= name <= 6 else None
    if isinstance(name, str) and name.startswith("joint"):
        return int(name[5:])
    return None


def _reads(monkeypatch, log, table):
    """Every motor's mechPos parameter read, logged; `table`: joint number -> rad (None: no answer)."""
    def read(self, param, *a):
        log.append(("param_read", self.name))
        value = table.get(_joint(self.name), 0.0)
        if value is None:
            raise RuntimeError("parameter 0x7019 not received")
        return float(value)

    monkeypatch.setattr(_Motor, "robstride_get_param_f32", read)


def _fake_bus(monkeypatch, backend, table=None):
    log = _Log()
    sdk = _install_fake_sdk(monkeypatch, log)
    ctl = _install_fake_motorbridge(monkeypatch, log)
    _reads(monkeypatch, log, table or {})
    return log, sdk, ctl, (_rs_arm if backend == "rebot_rs" else _mb_arm)


def _dropped(events):
    return [e for e in events if e[0] in TORQUE]


# ── Important 1: every holder outside shutdown_runtime writes the marker ──────────────────────────


def test_the_calibration_cleanup_hold_is_remembered_and_the_next_connect_sends_nothing(monkeypatch, no_sleep):  # noqa: F811
    """The probe's RED row: calibrate's cleanup with a dead camera kept the arm held (no torque cut) but
    wrote no marker, so the next calibrate / arm-check / start sent 7 STOP frames to the raised arm."""
    from cascade.calibration import cli

    log, _sdk, _ctl, make = _fake_bus(monkeypatch, "rebot_rs")
    raw = make()
    raw.connect()  # the calibration moved the arm: connected, powered

    class DeadCamera:
        def get_frame(self):
            raise RuntimeError("camera unplugged")

        def close(self):
            pass

    rig = cli.Rig(safe_arm=_dead_harness_safe(raw), raw_arm=raw, kin=None, camera=DeadCamera(),
                  home_q=np.zeros(6), ee_frame="")
    n = len(log)
    cli._cleanup(rig, park=True)
    assert _dropped(log[n:]) == [], "the refused park keeps the torque"
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "calibrate" and "refused" in marker["reason"], marker
    n0 = len(log)
    with pytest.raises(arm_hold.ArmHeldError, match="POWERED and HOLDING"):
        make().connect()  # the next process
    assert log[n0:] == [], f"the next connect sent {log[n0:]}"


@needs_pin
def test_the_arm_checks_soft_stop_is_remembered_and_the_next_connect_sends_nothing(monkeypatch, no_sleep, capsys):  # noqa: F811
    """The probe's RED row: home_rebot_rs.py refused mid-way up soft-stops (torque ON, arm raised) and
    exits; it wrote no marker, so the next connect sent 7 STOP frames."""
    import home_rebot_rs as hr
    from test_home_rebot_rs import RIG_REST, FakeCamera, _builder, _main

    from cascade.control.mock_arm import MockArm

    monkeypatch.setattr(MockArm, "hold_key", property(lambda self: "can0"))  # the arm on the rig's bus
    events = []
    assert _main(["--yes", "--hold-s", "0"], _builder(RIG_REST, events, refuse_on_move=2), FakeCamera()) == 1
    assert ("stop",) in events and ("disconnect",) not in events  # held, torque ON
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "home_rebot_rs.py", marker
    assert "SAFETY VIOLATION" in marker["reason"] and hr.REST_TOL == arm_hold.REST_TOL_RAD
    log, _sdk, _ctl, make = _fake_bus(monkeypatch, "rebot_rs")
    with pytest.raises(arm_hold.ArmHeldError):
        make().connect()
    assert log == []


def test_an_arm_check_that_ends_off_its_rest_remembers_the_hold(monkeypatch):
    """home_rebot_rs.py's other held exit: the way down tracked, but the arm is not MEASURED at rest
    afterwards, so its torque stays ON and the process exits."""
    import home_rebot_rs as hr

    park, home = np.zeros(6), np.array([0.0, 1.2, 1.2, 0.0, 0.0, 0.0])
    reads = iter([park, park + [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]])  # start at rest; after the way down: off
    safe = SimpleNamespace(raw=SimpleNamespace(hold_key="can0"), harness=SimpleNamespace(
        limits=SimpleNamespace(max_joint_vel=0.4)), disconnect=lambda: pytest.fail("torque cut off its rest"))
    safe.get_state = lambda: SimpleNamespace(q=next(reads, home))
    monkeypatch.setattr(hr, "_leg", lambda *a, **k: True)
    kin = SimpleNamespace(fk=lambda q: np.eye(4))
    assert hr.home_and_back(safe, kin, home, park, hold_s=0, log=lambda m: None) == 1
    marker = arm_hold.read("can0")
    assert marker is not None and "not measured at rest" in marker["reason"], marker


def test_the_startup_check_over_a_hold_says_so_without_a_traceback(monkeypatch, capsys):
    from test_rebot_gripper_reference import _check_script

    import cascade.control.rebot_rs_arm as rs

    def refuse():
        raise arm_hold.ArmHeldError("the arm on can0 is not folded at rest (test)")

    mod = _check_script()
    monkeypatch.setattr(mod, "read_rest_pose", lambda: [0.0] * 6)
    monkeypatch.setattr(rs, "RebotRSArm", lambda cfg: SimpleNamespace(connect=refuse, hold_key="can0"))
    assert mod.main(["--yes", "--reference-only"]) == 2
    assert "the arm was NOT connected: the arm on can0 is not folded at rest" in capsys.readouterr().out


def test_a_standby_lazy_arm_says_whether_its_connect_cuts_torque_without_connecting():
    from cascade.control.lazy_arm import LazyArm

    def never():
        raise AssertionError("materialized")

    assert LazyArm(never, profile_type="rebot_rs").connect_cuts_torque is True
    assert LazyArm(never, profile_type="rebot_rs_mb").connect_cuts_torque is True
    assert LazyArm(never, profile_type="so101").connect_cuts_torque is False
    assert LazyArm(never, profile_type="mock").connect_cuts_torque is False


def test_the_startup_checks_hold_after_connecting_is_remembered(monkeypatch, capsys):
    """check_rebot_gripper.py: an arm measured off its rest after the connect is held (stop, torque ON)
    and the process exits: that hold is remembered too."""
    from test_rebot_gripper_reference import _check_script

    import cascade.control.rebot_rs_arm as rs

    calls = []
    arm = SimpleNamespace(connect=lambda: calls.append("connect"), stop=lambda: calls.append("stop"),
                          disconnect=lambda: calls.append("disconnect"), hold_key="can0")
    mod = _check_script()
    monkeypatch.setattr(mod, "read_rest_pose", lambda: [0.0] * 6)
    monkeypatch.setattr(mod, "run", lambda arm, park, **k: 2)
    monkeypatch.setattr(rs, "RebotRSArm", lambda cfg: arm)
    assert mod.main(["--yes", "--reference-only"]) == 2
    assert calls == ["connect", "stop"]
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "check_rebot_gripper.py", marker


@pytest.mark.parametrize("script", ["jog_rebot_mb", "sign_check_rebot_mb"])
def test_a_bring_up_script_that_enabled_the_motors_remembers_the_hold(monkeypatch, script):
    """jog_rebot_mb.py / sign_check_rebot_mb.py leave the motors enabled by design (and on any error
    after the enable): every later process must know."""
    import test_rebot_bringup_scripts_clear as B

    mod = __import__(script)
    log = []
    B._fake_motorbridge(monkeypatch, log)
    monkeypatch.setattr(mod, "_sdk_motors", lambda: dict(B._SDK))
    monkeypatch.setattr(mod, "_profile_gains", lambda mid: (5.0, 0.5))
    monkeypatch.setattr(mod, "_read", lambda m, p, timeout_ms=300: 0.5)
    monkeypatch.setattr(mod, "_urdf_local_limits", B._limits)
    argv = ["jog", "--joint", "1", "--hold-only", "--yes"] if script == "jog_rebot_mb" else ["sc", "--joints", "1", "--yes"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(B._Energized):
        mod.main()
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == f"{script}.py", marker


# ── Important 1, second layer: no marker, but the arm measures raised ─────────────────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
@pytest.mark.parametrize("table", [{2: 1.2, 3: 1.2}, {2: 0.13}, {3: None}, {1: float("nan")}],
                         ids=["raised", "just_out", "j3_unread", "j1_nan"])
def test_a_raised_rebot_with_no_marker_is_refused_on_parameter_reads_alone(monkeypatch, no_sleep, backend, table):  # noqa: F811
    """A killed process, an older cascade or a bring-up script can leave the motors ENABLED with no
    marker. No RobStride read reports the enable state, so the connect measures the arm (mechPos
    parameter reads) BEFORE its mode write (motorbridge v0.5.6's ensure_mode disables a motor not in
    MIT) and its fault clear: not folded at rest, it is refused and the bus released with no frame
    that can drop it. Main: mode write, 7 STOP frames, enable."""
    log, _sdk, _ctl, make = _fake_bus(monkeypatch, backend, table)
    with pytest.raises(arm_hold.ArmHeldError, match="not folded at rest") as err:
        make().connect()
    kinds = [e[0] for e in log]
    assert _dropped(log) == [] and not {"mode_mit", "ensure_mode", "enable", "enable_all"} & set(kinds), log
    assert set(kinds) <= {"connect", "param_read", "close_bus", "close"}, log
    assert kinds.count("param_read") == 6 and kinds[-2:] == ["close_bus", "close"], log
    assert "arm_hold.py" not in str(err.value) and "holding the arm" not in str(err.value)
    why = {"raised": "rad from park_q", "just_out": "rad from park_q", "j3_unread": "joint(s) [3] did not answer",
           "j1_nan": "joint(s) [1] gave an implausible reading"}
    assert any(w in str(err.value) for w in why.values()) and str(err.value).count("joint(s)") <= 1, err.value
    if None in table.values():
        assert why["j3_unread"] in str(err.value), err.value
    elif any(v != v for v in table.values()):
        assert why["j1_nan"] in str(err.value), err.value
    assert arm_hold.read("can0") is None, "a measurement refuses; it records nothing"


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_rebot_folded_at_rest_connects_as_before_after_reading_it(monkeypatch, no_sleep, backend):  # noqa: F811
    log, _sdk, _ctl, make = _fake_bus(monkeypatch, backend, {2: 0.11})  # inside the startup check's tolerance
    make().connect()
    kinds = [e[0] for e in log]
    mode = kinds.index("mode_mit" if backend == "rebot_rs" else "ensure_mode")
    assert kinds[:mode].count("param_read") == 6, "the rest is read before the mode write"
    assert kinds.count("clear_error") == 7 and "close_bus" not in kinds


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_raised_rebot_connects_on_the_operators_confirmation_once(monkeypatch, no_sleep, backend):  # noqa: F811
    log, _sdk, _ctl, make = _fake_bus(monkeypatch, backend, {2: 1.2, 3: 1.2})
    arm_hold.confirm("holding the arm")
    make().connect()
    assert [e[0] for e in log].count("clear_error") == 7
    assert arm_hold.pending_confirmations() == [], "used by the connect that succeeded"
    n0 = len(log)
    with pytest.raises(arm_hold.ArmHeldError):
        make().connect()
    assert _dropped(log[n0:]) == []


def test_a_raised_connect_whose_fault_clear_fails_keeps_the_confirmation(monkeypatch, no_sleep):  # noqa: F811
    log = _Log()
    _install_fake_sdk(monkeypatch, log, failures={"joint3": [RuntimeError("control ack timeout: comm_type=4")] * 3})
    _reads(monkeypatch, log, {2: 1.2})
    arm_hold.confirm("holding the arm")
    with pytest.raises(RuntimeError, match="joint3"):
        _rs_arm().connect()
    assert len(arm_hold.pending_confirmations()) == 1, "a failed connect leaves it for the retry"


# ── Minor 6: the SDK reBot's marker names the channel its hardware YAML opens ─────────────────────


def test_the_sdk_rebots_hold_key_is_the_channel_of_its_hardware_yaml(monkeypatch, tmp_path, no_sleep):  # noqa: F811
    from cascade.control.rebot_rs_arm import RebotRSArm

    monkeypatch.setattr(sys, "path", list(sys.path))  # RebotRSArm inserts sdk_path
    sdk = tmp_path / "reBotArm_control_py"
    (sdk / "config").mkdir(parents=True)
    (sdk / "config" / "left.yaml").write_text("channel: can1\nrate: 500\n")
    (sdk / "config" / "right.yaml").write_text("channel: can0\nrate: 500\n")
    (tmp_path / "abs.yaml").write_text("channel: can7\n")

    def cfg(**kw):
        return Cfg({"n_joints": 6, "sdk_path": str(sdk), **kw})

    left, right = cfg(hw_yaml="left.yaml", channel="can0"), cfg(hw_yaml="right.yaml")
    assert arm_hold.hold_key("rebot_rs", left) == RebotRSArm(left).hold_key == "can1", "the YAML's, not the profile's"
    assert arm_hold.hold_key("rebot_rs", right) == "can0"
    assert arm_hold.hold_key("rebot_rs", cfg(hw_yaml=str(tmp_path / "abs.yaml"))) == "can7"
    assert arm_hold.hold_key("rebot_rs", cfg(hw_yaml="missing.yaml", channel="can3")) == "can3"
    assert arm_hold.hold_key("rebot_rs", Cfg({"hw_yaml": "x.yaml"})) == "can0"
    assert arm_hold.hold_key("rebot_rs_mb", cfg(hw_yaml="left.yaml", channel="can2")) == "can2"
    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    arm_hold.record("can1", arm="left", reason="held")
    with pytest.raises(arm_hold.ArmHeldError):
        RebotRSArm(left).connect()
    assert log == []
    RebotRSArm(right).connect()  # another bus: its own arm
    assert [e[0] for e in log].count("clear_error") == 7


# ── Important 2 / 4 / Minor 8: a held standby arm on the MCP runtime ──────────────────────────────


@pytest.fixture
def held_runtime(monkeypatch, tmp_path, no_sleep):  # noqa: F811
    from cascade.control.rebot_rs_arm import RebotRSArm

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _hold_a_rebot(log)
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    occ = cfg._data.get("occupancy")
    if isinstance(occ, dict) and occ.get("required"):
        occ["required"] = False
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: RebotRSArm(acfg))
    rt, arm = demo.build_runtime(cfg, tmp_path / "run2", view=False, serve=False, lazy_arm=True)
    yield rt, log
    shutdown_runtime(rt, arm)


@needs_pin
def test_move_to_rest_never_calls_a_held_standby_arm_safe_to_power_off(held_runtime):
    """Main and #426: 0 frames sent, then "torque released, safe_to_power_off: true ... safe to switch the
    robot off" about an arm an earlier session left powered and holding."""
    rt, log = held_runtime
    for args in ({}, {"keep_torque": True}):
        n0 = len(log)
        out = rt.execute("move_to_rest", args)
        assert log[n0:] == [], f"move_to_rest{args} sent {log[n0:]}"
        assert out["safe_to_power_off"] is False and out["torque"] == "on", out
        assert out["at"] == "held" and out["ok"] is False and "NOT safe" in out["error"], out
    rt.arm.stop()  # e-stop latched: the refusal still says the torque is on
    out = rt.execute("move_to_rest", {})
    assert out["torque"] == "on" and out["safe_to_power_off"] is False, out


@needs_pin
def test_no_agent_facing_text_names_the_confirm_command(held_runtime):
    """Important 4: the tool error and world_state handed the agent `<python> arm_hold.py confirm
    'holding the arm'`, and host MCP clients have a shell. Now they ask for the operator. Minor 8:
    the tool error names the hold, not an `unexpected ArmHeldError`."""
    rt, log = held_runtime
    out = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
    assert out["ok"] is False and out["error"].startswith("ArmHeldError: ") and "unexpected" not in out["error"]
    assert out.get("refused") == "arm_held", out
    texts = {"tool error": json.dumps(out), "/state": json.dumps(demo._runtime_state(rt), default=str),
             "world_state": json.dumps(mcp_server.McpSkillServer()._world_state(rt), default=str),
             "move_to_rest": json.dumps(rt.execute("move_to_rest", {}))}
    for where, text in texts.items():
        assert "arm_hold.py" not in text and "holding the arm" not in text and "confirm_command" not in text, where
        assert "operator" in text, where
    assert log[-1][0] != "clear_error"


def test_world_states_description_says_what_arm_connected_means_for_a_held_arm():
    tool = next(t for t in mcp_server._EXTRA_TOOLS if t["name"] == "world_state")
    assert "earlier session left it powered and holding" in tool["description"]
    assert "arm_held" in tool["description"] and "safe to power off" in tool["description"]


# ── Important 3: a confirmed connect that fails keeps the hold ────────────────────────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_confirmed_connect_that_fails_keeps_the_hold_and_the_confirmation(monkeypatch, no_sleep, backend):  # noqa: F811
    """The probe's RED row: confirm, then connect with can0 down: the marker was cleared and the
    confirmation used up before the bus opened, so the later retry (nobody re-confirmed) sent 7 STOP
    frames with nobody holding the arm."""
    log, sdk, ctl, make = _fake_bus(monkeypatch, backend)
    _hold_a_rebot(log, make=make)
    arm_hold.confirm("holding the arm")
    bus_down = pytest.MonkeyPatch()  # private: undoing it never touches conftest's hold-dir pin
    if backend == "rebot_rs":
        bus_down.setattr(sdk, "connect", lambda self: (_ for _ in ()).throw(OSError("can0: No such device")))
    else:
        def broken(self, channel=None):
            raise OSError("can0: No such device")
        bus_down.setattr(ctl, "__init__", broken)
    try:
        with pytest.raises(OSError):
            make().connect()
    finally:
        bus_down.undo()  # the bus is up again
    assert arm_hold.read("can0") is not None, "the failed connect keeps the hold"
    assert len(arm_hold.pending_confirmations()) == 1, "and the confirmation, for a retry in its window"
    n_retry = len(log)
    make().connect()  # the retry within the window: the operator still holds the arm
    assert [e[0] for e in log[n_retry:]].count("clear_error") == 7
    assert arm_hold.read("can0") is None and arm_hold.pending_confirmations() == []
    arm_hold.record("can0", arm="0", reason="held again")
    n0 = len(log)
    with pytest.raises(arm_hold.ArmHeldError):  # never after a successful connect
        make().connect()
    assert log[n0:] == []


# ── Important 4: only the operator, at a terminal, confirms ───────────────────────────────────────


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if k not in arm_hold.AGENT_ENV and not k.startswith(arm_hold.AGENT_ENV_PREFIXES)}
    env.update(extra)
    return env


#: The child makes the pseudo-terminal on its stdout its controlling terminal, then runs the script (no
#: fork in the test process itself: pytest runs threads).
_IN_A_TERMINAL = ("import fcntl, os, sys, termios; os.setsid(); fcntl.ioctl(1, termios.TIOCSCTTY, 0); "
                  "os.execv(sys.executable, [sys.executable] + sys.argv[1:])")


def _confirm_in_a_terminal(env, typed: str | None, *args, timeout_s=60.0, piped: str | None = None):
    """`scripts/arm_hold.py confirm` with a pseudo-terminal as its controlling terminal (and stdin,
    unless `piped` text is its stdin instead); types `typed` at its prompt. (exit code, output)."""
    import pty

    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-c", _IN_A_TERMINAL, str(REPO / "scripts" / "arm_hold.py"),
                             "confirm", *args], stdin=slave if piped is None else subprocess.PIPE, stdout=slave,
                            stderr=slave, env=env, close_fds=True)
    os.close(slave)
    if piped is not None:
        proc.stdin.write(piped.encode())
        proc.stdin.close()
    out, sent, deadline = b"", False, time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError:  # EIO: the child closed the terminal
                    break
                if not chunk:
                    break
                out += chunk
                if typed is not None and not sent and b"Type 'holding the arm'" in out:
                    os.write(master, typed.encode() + b"\n")
                    sent = True
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
    return code, out.decode(errors="replace")


def test_the_operator_confirms_by_typing_the_phrase_at_the_terminal():
    code, out = _confirm_in_a_terminal(_clean_env(), "  Holding the ARM ")
    assert code == 0 and "confirmed: you are supporting the arm" in out, out
    assert len(arm_hold.pending_confirmations()) == 1
    arm_hold.revoke_confirmations()
    code, out = _confirm_in_a_terminal(_clean_env(), "yes", "--arm", "can0")
    assert code == 1 and "not confirmed" in out and arm_hold.pending_confirmations() == [], out


def test_confirm_refuses_a_piped_stdin_even_with_a_terminal_to_type_on():
    """An agent can feed a pipe but not type at the operator's terminal: stdin must be the terminal too,
    whatever /dev/tty would answer."""
    code, out = _confirm_in_a_terminal(_clean_env(), "holding the arm", piped="holding the arm\n")
    assert code == 2 and "needs an interactive terminal" in out, out
    assert arm_hold.pending_confirmations() == []


@pytest.mark.parametrize("marker", ["AI_AGENT", "HERMES_SESSION_ID", "CLAUDECODE", "CASCADE_MCP_TOKEN",
                                    "OPENCLAW_GATEWAY_TOKEN"])
def test_confirm_refuses_in_an_agents_environment_even_at_a_terminal(marker):
    code, out = _confirm_in_a_terminal(_clean_env(**{marker: "1"}), "holding the arm")
    assert code == 2 and f"REFUSED: {marker} is set" in out and "agent" in out, out
    assert arm_hold.pending_confirmations() == []


def test_confirm_refuses_without_a_terminal_and_never_takes_the_phrase_as_an_argument(tmp_path):
    with pytest.raises(SystemExit):  # the phrase is typed, never passed
        arm_hold.main(["confirm", "holding the arm"])
    assert arm_hold.main(["confirm"]) == 2  # pytest's stdin is no terminal
    run = subprocess.run([sys.executable, str(REPO / "scripts" / "arm_hold.py"), "confirm"], cwd=tmp_path,
                         input="holding the arm\n", capture_output=True, text=True, timeout=60,
                         env=_clean_env(), start_new_session=True)  # piped, no controlling terminal
    assert run.returncode == 2 and "REFUSED" in run.stderr, run
    assert arm_hold.pending_confirmations() == []


def test_list_and_revoke(capsys):
    assert arm_hold.main(["list"]) == 0 and "no arm is held" in capsys.readouterr().out
    arm_hold.record("can0", arm="0", reason="held for the test")
    arm_hold.confirm("holding the arm")
    assert arm_hold.main(["list"]) == arm_hold.LIST_HELD_RC
    out = capsys.readouterr().out
    assert "HELD: arm on can0: held for the test" in out and "CONFIRMED: any arm" in out, out
    assert arm_hold.main(["revoke"]) == arm_hold.LIST_HELD_RC
    assert arm_hold.pending_confirmations() == [] and "1 pending confirmation(s) revoked" in capsys.readouterr().out


# ── Minors 1, 2, 3, 7, 9 ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("at", ['"now"', "null", "[1]", "true", '"nan"', "1e400"])
def test_a_malformed_confirmation_never_aborts_the_teardown(fast, at):  # noqa: F811
    """Minor 1: `{"at": "now"}` raised ValueError inside the teardown's `_disconnect_arms`, before it
    wrote the markers: the next start could not see the hold."""
    arm_hold.hold_dir().mkdir(parents=True, exist_ok=True)
    (arm_hold.hold_dir() / "_any.confirm.json").write_text(f'{{"at": {at}}}')
    events = []
    arm = hardware(events)
    arm.harness._last_heartbeat = -np.inf  # dead camera: the park is refused
    receipt = shutdown_runtime(runtime_for(arm), arm.raw)
    assert _torque_offs(events) == [] and list(receipt["torque_held"]) == ["0"], receipt
    assert receipt["hold_markers"]["0"]["key"] == "/dev/ttyACM0" and arm_hold.read("/dev/ttyACM0") is not None
    assert arm_hold.take_confirmation(None) is None


def test_a_malformed_confirmation_never_lets_a_held_rebot_connect(monkeypatch, no_sleep):  # noqa: F811
    log, _sdk, _ctl, make = _fake_bus(monkeypatch, "rebot_rs")
    arm_hold.record("can0", arm="0", reason="held")
    arm_hold.hold_dir().joinpath("_any.confirm.json").write_text('{"at": "now"}')
    with pytest.raises(arm_hold.ArmHeldError):
        make().connect()
    assert log == []


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads any directory")
def test_list_fails_closed_when_the_hold_directory_cannot_be_read(capsys):
    """Minor 2: with an unreadable hold directory `list` said "no arm is held" while every connect
    refused."""
    arm_hold.record("can0", arm="0", reason="held")
    root = arm_hold.hold_dir()
    os.chmod(root, 0)
    try:
        held = arm_hold.all_held()
        code = arm_hold.main(["list"])
    finally:
        os.chmod(root, stat.S_IRWXU)
    assert held and held[0]["unreadable"] is True, held
    out = capsys.readouterr().out
    assert code == arm_hold.LIST_HELD_RC and "HELD:" in out and "no arm is held" not in out, out


def test_list_fails_closed_when_the_hold_directory_is_a_file(capsys):
    root = arm_hold.hold_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text("not a directory")
    assert arm_hold.main(["list"]) == arm_hold.LIST_HELD_RC
    assert "HELD:" in capsys.readouterr().out
    assert arm_hold.read("can0") is not None, "and every connect refuses"


def test_an_so101_hold_is_never_announced_as_refused(fast, capsys, monkeypatch, no_sleep):  # noqa: F811
    """Minor 3: the warning said "the next start refuses" for an SO-101, whose connect writes no
    torque-off and is never refused."""
    events = []
    arm = hardware(events)
    arm.harness._last_heartbeat = -np.inf
    shutdown_runtime(runtime_for(arm), arm.raw)
    so101 = capsys.readouterr().err
    assert "POWERED and HOLDING" in so101 and "refuses" not in so101 and "writes no torque-off" in so101, so101
    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    _hold_a_rebot(log)
    rebot = capsys.readouterr().err
    assert "refuses to connect it" in rebot and "arm_hold.py confirm" in rebot, rebot


@needs_pin
def test_the_arm_check_over_a_hold_exits_cleanly_with_nothing_connected(monkeypatch, no_sleep, capsys):  # noqa: F811
    """Minor 7: home_rebot_rs.py over a hold died with an uncaught ArmHeldError traceback."""
    import home_rebot_rs as hr
    from test_home_rebot_rs import FakeCamera

    from cascade.control.rebot_rs_arm import RebotRSArm

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _hold_a_rebot(log)
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: RebotRSArm(acfg))
    n0 = len(log)
    code = hr.main(["--yes", "--hold-s", "0", "--camera", "mock", "--camera-wait-s", "3"],
                   make_camera=lambda cfg: FakeCamera())
    out = capsys.readouterr().out
    assert code == 3 and "the arm was NOT connected" in out and "POWERED and HOLDING" in out, out
    assert log[n0:] == []


def test_the_rebot_bulk_jaw_close_honours_the_teardown_gate():
    """Minor 9: the reBot drivers' own two-stage close bypasses SafeArm.set_gripper; it checked the halt
    generation but not the teardown gate, so a dashboard thread could close the jaws beside the park."""
    from test_safety import limits

    from cascade.grasping.force import GRIP_PROFILES
    from cascade.safety.harness import SafetyHarness
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import MotionHalted

    harness = SafetyHarness(limits())
    closer = threading.Thread(target=harness.close_for_teardown, args=("shutting down",))
    closer.start()
    closer.join()  # closed by the teardown's thread: every other thread is refused
    closes = []
    rt = SimpleNamespace(arm=SimpleNamespace(harness=harness, raw=SimpleNamespace(
        close_gripper_two_stage=lambda **kw: closes.append(kw))), _grip_open=6.2, _grip_closed=0.0,
        _force_contact_close_enabled=lambda: False, _mark_motion_started=lambda: closes.append("started"),
        _gripper_closes_commanded=0)
    profile = next(iter(GRIP_PROFILES.values()))
    with pytest.raises(MotionHalted, match="shutting down"):
        SkillRuntime._close_stages(rt, profile, _halt_generation=None, _before_close=None, _never_past=None)
    assert closes == []
