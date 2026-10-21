"""#431 re-review Important 1, and its follow-up: the operator's 'go' is typed at the robot's own terminal.

A reBot connect over a held (or raised) arm that went ahead on the operator's confirmation only HOLDS,
their hands on the arm. A command-line tool then asks them to let go and type 'go' before its first
motion (`arm_hold.wait_for_let_go`: the startup check `scripts/check_rebot_gripper.py`, the arm-check
`scripts/home_rebot_rs.py`, calibrate `cascade.calibration.cli`). It read `input()`: a piped 'go'
(`echo go | ...`) passed, in an agent's environment too, and the jaw (startup check) or the arm
(arm-check, calibrate) moved with nobody saying they had let go. Now it applies `confirm`'s rules: an
interactive session (stdin and stdout on a terminal), the answer read from the controlling terminal
(/dev/tty, never stdin), never in an agent's or the robot server's environment (`arm_hold.AGENT_ENV`,
never named). Anything else stops there with the arm held: the startup check exits 3 with its marker
kept, the arm-check and calibrate refuse with no motion.

The follow-up: the demo CLI builds its arm EAGERLY (no LazyArm latch), so a connect there on a
confirmation went straight on to the agent's first motion. Now it asks for the same 'go' first, with
the arm's e-stop latched until the operator types it.

No test touches CAN, the real SDK, motorbridge or sudo: the bus is a recording fake, and the terminal
rows run the startup check in a child process on a pseudo-terminal (never `pty.fork()` in pytest).
"""

from __future__ import annotations

import builtins
import io
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_pin
from test_held_arm_across_restart import _hold_a_rebot
from test_held_arm_every_path import _clean_env
from test_held_arm_followups import _bus, _operator_shell, _private_stores  # noqa: F401
from test_rebot_clear_error import no_sleep  # noqa: F401

from cascade.control import arm_hold

REPO = Path(__file__).resolve().parent.parent
LET_GO = "connected and holding; let go of the arm and retry"

# ── the startup check in a child process, its stdin / stdout / controlling terminal as given ───────

#: argv: the fd to make the controlling terminal, then the command. A new session first (no fork in
#: the test process itself: pytest runs threads).
_WITH_TERMINAL = ("import fcntl, os, sys, termios; os.setsid(); fcntl.ioctl(int(sys.argv[1]), termios.TIOCSCTTY, 0); "
                  "os.execv(sys.executable, [sys.executable] + sys.argv[2:])")
#: The startup check (`--yes --reference-only`) over a marker the 0.05 rule keeps (the arm 0.08 off park,
#: within the rig's 0.12) and the operator's fresh confirmation, on the recording fakes; the gripper
#: reference itself is a stub that records it ran. argv: src, tests, the private dir, the result file.
_CHECK = r'''
import importlib.abc, importlib.util, json, os, sys
src, tests, private, out = sys.argv[1:5]
assert os.environ["CASCADE_ARM_HOLD_DIR"].startswith(private) and os.environ["HOME"].startswith(private)


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("motorbridge", "reBotArm_control_py", "can", "serial", "pyrealsense2"):
            raise ImportError(f"{name}: the real driver never loads in this test")
        return None


sys.meta_path.insert(0, _Blocker())
sys.path[:0] = [src, tests]
import pytest
from test_rebot_clear_error import _ConnectClock, _install_fake_motorbridge, _install_fake_sdk, _Log
from cascade.config import Cfg
from cascade.control import arm_hold, rebot_rs_arm, robstride

mp, log = pytest.MonkeyPatch(), _Log()
_install_fake_sdk(mp, log)
_install_fake_motorbridge(mp, log)
for module in (robstride, rebot_rs_arm):
    mp.setattr(module, "time", _ConnectClock())
arm_hold.record("can0", arm="0", reason="the shutdown park stopped 0.08 rad short of park_q")
arm_hold.confirm("holding the arm")
spec = importlib.util.spec_from_file_location("check_rebot_gripper", os.path.join(src, "..", "scripts",
                                                                                   "check_rebot_gripper.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
q = [0.0, 0.0, 0.08, 0.0, 0.0, 0.0]
mp.setattr(mod, "read_rest_pose", lambda *a, **k: (list(q), list(q)))
ran = []
mp.setattr(mod, "run", lambda arm, park, **k: ran.append(1) or 0)
real = rebot_rs_arm.RebotRSArm
mp.setattr(rebot_rs_arm, "RebotRSArm", lambda cfg: real(Cfg({"hw_yaml": "rebotarm_rs.yaml", "n_joints": 6})))
code = mod.main(["--yes", "--reference-only"])
kinds = [e[0] for e in log]
marker = arm_hold.read("can0")
with open(out, "w") as f:
    json.dump({"rc": code, "ran": bool(ran), "clear_error": kinds.count("clear_error"),
               "disconnect": "disconnect" in kinds, "marker": None if marker is None else marker.get("source")}, f)
'''


def _startup_check(tmp_path, *, stdin="tty", stdout="tty", terminal=True, typed="", piped="go\n", env=None,
                   timeout_s=120.0):
    """Run the startup check in a child process. `stdin`: "tty" (the pseudo-terminal), "pipe" (`echo go |`)
    or "other_tty" (a second pseudo-terminal that is NOT the controlling one) -- the last two carry
    `piped`; `stdout`: "tty" or "pipe" (`| tee`); `terminal`: the pseudo-terminal is the child's
    controlling terminal (else a new session with none). `typed`: what the operator types at the
    pseudo-terminal (buffered from the start). (the child's row, everything the terminal showed)."""
    import pty

    private = tmp_path / "child"
    private.mkdir()
    out = private / "row.json"
    master, slave = pty.openpty()
    close, other_master = [slave], None
    if stdin == "tty":
        child_in = slave
    elif stdin == "pipe":
        child_in = subprocess.PIPE
    else:
        other_master, other_slave = pty.openpty()
        close.append(other_slave)
        child_in = other_slave
        os.write(other_master, piped.encode())
    child_out = slave if stdout == "tty" else open(private / "stdout.txt", "wb")  # noqa: SIM115 -- closed below
    argv = ["-c", _CHECK, str(REPO / "src"), str(REPO / "tests"), str(private), str(out)]
    argv = [sys.executable, "-c", _WITH_TERMINAL, "2", *argv] if terminal else [sys.executable, *argv]
    child_env = {**(env if env is not None else _clean_env()), "HOME": str(private),
                 "CASCADE_ARM_HOLD_DIR": str(private / "arm_holds"), "CASCADE_REST_LOG": str(private / "rest.jsonl"),
                 "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen(argv, stdin=child_in, stdout=child_out, stderr=slave, env=child_env, close_fds=True,
                            start_new_session=not terminal)
    for fd in close:
        os.close(fd)
    if child_out is not slave:
        child_out.close()
    if stdin == "pipe":
        proc.stdin.write(piped.encode())
        proc.stdin.close()
    if typed:
        os.write(master, typed.encode())
    shown, deadline = b"", time.monotonic() + timeout_s
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
                shown += chunk
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
        if other_master is not None:
            os.close(other_master)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    text = shown.decode(errors="replace")
    if stdout == "pipe":
        text += (private / "stdout.txt").read_text(errors="replace")
    assert out.exists(), f"the child did not finish (exit {proc.returncode}):\n{text}"
    return json.loads(out.read_text()), text


def _held(row, text):
    """Stopped before the gripper check: exit 3, nothing moved, torque on, the hold remembered."""
    assert row["clear_error"] == 7, "connected on the confirmation"
    assert row["rc"] == 3 and row["ran"] is False, (row, text)
    assert row["disconnect"] is False and row["marker"] == "check_rebot_gripper.py", (row, text)
    assert "stopped before the gripper check" in text and "HOLDING" in text, text


def test_the_operator_types_go_at_the_terminal_and_the_check_runs(tmp_path):
    """The operator's own terminal: 'go' typed there lets the gripper reference run."""
    row, text = _startup_check(tmp_path, typed="go\n")
    assert row["rc"] == 0 and row["ran"] is True and row["disconnect"] is True, (row, text)
    assert row["marker"] is None and LET_GO in text and "type 'go'" in text, text


def test_anything_but_go_at_the_terminal_stops_holding(tmp_path):
    row, text = _startup_check(tmp_path, typed="no\n")
    _held(row, text)


def test_a_piped_go_with_no_terminal_stops_holding(tmp_path):
    """The re-review's row: `echo go | check_rebot_gripper.py --yes --reference-only` (no terminal at all)
    exited 0, ran the gripper reference and released the torque."""
    row, text = _startup_check(tmp_path, stdin="pipe", stdout="pipe", terminal=False)
    _held(row, text)
    assert "interactive terminal" in text, text


def test_a_piped_go_in_a_terminal_stops_holding(tmp_path):
    """`echo go | ...` typed in a terminal: stdin is the pipe. Not an interactive session (confirm's rule),
    even when 'go' is typed at the terminal as well."""
    row, text = _startup_check(tmp_path, stdin="pipe", typed="go\n")
    _held(row, text)


def test_go_with_the_output_piped_stops_holding(tmp_path):
    """`... | tee up.log`: stdout is not the terminal; confirm refuses it, and so does the let-go."""
    row, text = _startup_check(tmp_path, stdout="pipe", typed="go\n")
    _held(row, text)


def test_the_answer_is_read_from_the_controlling_terminal_never_stdin(tmp_path):
    """stdin a terminal that is NOT the controlling one, carrying 'go'; the operator types 'no' at the
    controlling terminal: their answer counts."""
    row, text = _startup_check(tmp_path, stdin="other_tty", piped="go\n", typed="no\n")
    _held(row, text)


def test_no_controlling_terminal_stops_holding(tmp_path):
    """stdin and stdout on a terminal, but no controlling terminal (a new session): /dev/tty cannot be
    opened, so nobody can be asked."""
    row, text = _startup_check(tmp_path, terminal=False, typed="go\n")
    _held(row, text)
    assert "controlling terminal" in text, text


@pytest.mark.parametrize("marker", ["HERMES_SESSION_ID", "AI_AGENT", "CLAUDECODE", "OPENCLAW_GATEWAY_TOKEN"])
def test_an_agents_environment_never_lets_go(tmp_path, marker):
    """The re-review's row: the same `echo go |` with HERMES_SESSION_ID and AI_AGENT set exited 0. Now even
    a terminal and a typed 'go' do not count in an agent's (or the robot server's) environment, and the
    refusal names no variable to unset."""
    row, text = _startup_check(tmp_path, typed="go\n", env=_clean_env(**{marker: "1"}))
    _held(row, text)
    assert "agent" in text and marker not in text and " is set" not in text, text


# ── wait_for_let_go itself ──────────────────────────────────────────────────────────────────────────


def test_the_let_go_never_reads_stdin(monkeypatch):
    def no_input(prompt=""):
        raise AssertionError("input() read the let-go answer from stdin")

    monkeypatch.setattr(builtins, "input", no_input)
    monkeypatch.setattr(sys, "stdin", io.StringIO("go\n"))
    said = []
    assert arm_hold.wait_for_let_go("can0", log=said.append) is False
    assert any(LET_GO in s for s in said) and any("interactive terminal" in s for s in said), said


def test_the_let_go_asks_no_one_in_an_agents_environment(monkeypatch):
    asked = []
    monkeypatch.setenv("HERMES_SESSION_ID", "s-1")
    monkeypatch.setattr(arm_hold, "_answer_at_terminal", lambda prompt: asked.append(prompt) or "go", raising=False)
    said = []
    assert arm_hold.wait_for_let_go("can0", log=said.append) is False
    assert asked == [], "refused before anything was asked"
    assert any("agent" in s for s in said) and not any("HERMES_SESSION_ID" in s for s in said), said


@pytest.mark.parametrize(("typed", "go"), [("go\n", True), ("  GO \n", True), ("no\n", False), ("", False),
                                          ("go on\n", False)])
def test_only_go_typed_at_the_terminal_lets_go(monkeypatch, typed, go):
    asked = []
    monkeypatch.setattr(arm_hold, "_answer_at_terminal", lambda prompt: asked.append(prompt) or typed, raising=False)
    assert arm_hold.wait_for_let_go("can0", log=lambda *_: None) is go
    assert len(asked) == 1 and "let go of the arm" in asked[0].lower() and "'go'" in asked[0], asked


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, OSError])
def test_a_terminal_that_fails_mid_answer_stops_holding(monkeypatch, error):
    def fails(prompt):
        raise error()

    monkeypatch.setattr(arm_hold, "_answer_at_terminal", fails, raising=False)
    assert arm_hold.wait_for_let_go("can0", log=lambda *_: None) is False


# ── the arm-check and calibrate: a 'go' from stdin, or in an agent's environment, moves nothing ──────


@needs_pin
@pytest.mark.parametrize("how", ["piped", "agent"])
def test_the_arm_check_moves_nothing_on_a_go_it_did_not_read_at_the_terminal(monkeypatch, capsys, how):
    from test_held_arm_followups import _confirmed_arm_check

    monkeypatch.setattr(sys, "stdin", io.StringIO("go\n"))
    if how == "agent":
        monkeypatch.setenv("AI_AGENT", "1")
        monkeypatch.setattr(arm_hold, "_answer_at_terminal", lambda prompt: "go", raising=False)
    code, events = _confirmed_arm_check(monkeypatch, "go", at="stdin")
    assert code == 3 and not [e for e in events if e[0] == "target"], events
    assert ("disconnect",) not in events, "the arm stays held"
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "home_rebot_rs.py", marker


@needs_pin
@pytest.mark.parametrize("how", ["piped", "agent"])
def test_calibrate_sweeps_nothing_on_a_go_it_did_not_read_at_the_terminal(monkeypatch, tmp_path, capsys, how):
    from test_held_arm_followups import _calibrate

    if how == "agent":
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr(arm_hold, "_answer_at_terminal", lambda prompt: "go", raising=False)
    rc, rig, calls = _calibrate(monkeypatch, tmp_path, arm_hold.connected_and_holding("can0"), answers=["go"] * 3,
                                at="stdin")
    assert rc == 1 and rig.raw_arm.commands == [] and "disconnect" not in calls, (rc, calls)
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "calibrate", marker


# ── the follow-up: the demo CLI's eager build ───────────────────────────────────────────────────────


def _demo_cli(monkeypatch, tmp_path, *, answer, signal_in_run=None):
    """`python -m cascade.apps.demo --arm rebot_rs` over a reBot an earlier session left holding, with the
    operator's fresh confirmation: its EAGER build connects the arm on that confirmation. The agent run
    (`_run_demo`) is a stub whose first act is a motion (move_home), then raises `signal_in_run` if given.
    `answer`: what the operator types at the terminal (None: no terminal; a callable: called instead)."""
    from test_confirmed_connect_holds import _commands, _mark_connects

    from cascade.apps import demo
    from cascade.control.rebot_rs_arm import RebotRSArm

    monkeypatch.setattr(sys, "path", list(sys.path))
    log, make = _bus(monkeypatch, "rebot_rs")
    _hold_a_rebot(log, make)
    assert arm_hold.read("can0") is not None, "an earlier session left the arm holding"
    real_load = demo.load_demo_config

    def load(**kw):
        cfg = real_load(**kw)
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False  # conftest pins CASCADE_OCCUPANCY=0
        return cfg

    monkeypatch.setattr(demo, "load_demo_config", load)
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: RebotRSArm(acfg))
    _mark_connects(monkeypatch, log)
    runs, asked = [], []

    def run_demo(args, cfg, runtime, mobile):
        n0 = len(log)
        out = runtime.execute("move_home", {})
        runs.append({"out": out, "commands": _commands(log[n0:])})
        if signal_in_run is not None:
            signal.raise_signal(signal_in_run)
        return 0

    def at_terminal(prompt):
        asked.append(prompt)
        return answer() if callable(answer) else answer

    monkeypatch.setattr(demo, "_run_demo", run_demo)
    if answer is not None:
        monkeypatch.setattr(arm_hold, "_answer_at_terminal", at_terminal, raising=False)
    arm_hold.confirm("holding the arm")
    n0 = len(log)
    rc = demo.main(["--arm", "rebot_rs", "--camera", "mock", "--llm", "mock", "--no-view", "--no-serve",
                    "--run-dir", str(tmp_path / "run")])
    return rc, runs, asked, log[n0:], _commands


@needs_pin
def test_the_demo_cli_moves_a_confirmed_eager_connect_only_after_go(monkeypatch, tmp_path, no_sleep, capsys):  # noqa: F811
    rc, runs, asked, sent, _commands = _demo_cli(monkeypatch, tmp_path, answer="go")
    assert [e[0] for e in sent].count("clear_error") == 7, "connected on the confirmation"
    assert len(asked) == 1 and "let go of the arm" in asked[0].lower(), asked
    # (on the fakes the move ends on the contact monitor: the joints read 0 and never follow; it moved)
    assert rc == 0 and len(runs) == 1 and runs[0]["commands"], runs
    assert "ConnectedAndHolding" not in str(runs[0]["out"]) and "e-stop" not in str(runs[0]["out"]), runs
    assert LET_GO in capsys.readouterr().out


@needs_pin
@pytest.mark.parametrize("answer", [None, "no"], ids=["no_terminal", "no"])
def test_the_demo_cli_runs_nothing_after_a_confirmed_eager_connect_without_go(monkeypatch, tmp_path, no_sleep,  # noqa: F811
                                                                             capsys, answer):
    """Main: the eager build connected on the confirmation and the agent's first motion moved the arm with
    the operator's hands on it. Now nothing runs: exit 3, no command at all (the teardown's park is
    refused by the latched e-stop), torque on, the hold remembered for the next process."""
    rc, runs, _asked, sent, _commands = _demo_cli(monkeypatch, tmp_path, answer=answer)
    assert [e[0] for e in sent].count("clear_error") == 7, "connected on the confirmation"
    assert rc == 3 and runs == [], (rc, runs)
    assert _commands(sent) == [], f"written after the connect: {_commands(sent)}"
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "teardown", marker
    assert "retry" not in marker["reason"], marker
    out = capsys.readouterr().out
    assert LET_GO in out and "HOLDING" in out, out


@needs_pin
@pytest.mark.parametrize(("signum", "rc"), [(signal.SIGINT, 3), (signal.SIGTERM, 128 + signal.SIGTERM)],
                         ids=["SIGINT", "SIGTERM"])
def test_a_signal_at_the_eager_go_prompt_never_reads_as_success(monkeypatch, tmp_path, no_sleep, capsys,  # noqa: F811
                                                                 signum, rc):
    """#435 review Minor 2: Ctrl+C at the prompt exited 0 with nothing run and the arm left powered and
    holding, so an `up`-style wrapper read success. Now STOPPED_HOLDING (3), as a 'no'; SIGTERM keeps
    143. Either way nothing moved and the hold is remembered."""
    def interrupted():
        signal.raise_signal(signum)
        pytest.fail("the signal did not interrupt the prompt")

    got, runs, asked, sent, _commands = _demo_cli(monkeypatch, tmp_path, answer=interrupted)
    assert len(asked) == 1 and got == rc and runs == [], (got, runs)
    assert _commands(sent) == [], f"written after the connect: {_commands(sent)}"
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "teardown", marker


@needs_pin
def test_ctrl_c_after_the_go_still_exits_0_as_before(monkeypatch, tmp_path, no_sleep, capsys):  # noqa: F811
    """Control: once the operator typed 'go', a Ctrl+C during the run is the CLI's ordinary stop (0)."""
    got, runs, _asked, _sent, _commands = _demo_cli(monkeypatch, tmp_path, answer="go", signal_in_run=signal.SIGINT)
    assert got == 0 and len(runs) == 1, (got, runs)


def test_the_eager_let_go_skips_a_lazy_arm_and_an_unconfirmed_one(monkeypatch):
    """Only an eager backend that connected on a confirmation is asked about; a LazyArm keeps its own latch
    and is never materialized to find out."""
    from types import SimpleNamespace

    from cascade.apps import demo
    from cascade.control.lazy_arm import LazyArm

    def factory():
        raise AssertionError("materialized")

    lazy = LazyArm(factory, profile_type="rebot_rs", hold_key="can0")
    plain = SimpleNamespace(connected_on_confirmation=False)
    asked = []
    monkeypatch.setattr(arm_hold, "_answer_at_terminal", lambda prompt: asked.append(prompt) or "go", raising=False)
    for raw in (lazy, plain):
        harness = SimpleNamespace(estopped=False, estop=lambda *a: pytest.fail("latched"))
        runtime = SimpleNamespace(arm_rig=None, arm=SimpleNamespace(raw=raw, harness=harness))
        assert demo._hold_confirmed_connects(runtime) == []
    assert asked == []
    np.testing.assert_equal(lazy.connected, False)
