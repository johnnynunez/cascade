"""Follow-ups to #426 (its fix-round re-review): the held-arm rules, tightened.

- Important 1: `arm_hold.py confirm` also needs the operator's sudo PASSWORD (`sudo -k -v` at the
  terminal; a sudo that validates without one, NOPASSWD, is refused: fail closed). Its refusal in an
  agent's environment no longer names the variable to unset. Any same-user process can still delete a
  marker or write a confirmation file: the boundary is the OpenShell sandbox (documented).
- Important 2: the startup check keeps a marker unless every joint is within the runtime's own release
  tolerance (0.05 rad, `skills.rest_pose.REST_TOL_RAD`); every joint is read twice and the two reads
  must agree (Minor 2); every rest check is logged (`~/.cascade/logs/rest_readings.jsonl`, bounded) so
  the no-marker tolerance, now a per-rig profile value (`rest_check_tol_rad`, default 0.12, may only
  tighten), can be tightened from rig data (`scripts/rest_tolerance_report.py`).
- Minors: the operator ask comes first in every hold refusal (1: next_action keeps 240 characters); the
  off-park refusal names the joints and says lowering the arm is enough (3); the bring-up scripts read
  the marker before their mode write and `--yes` never skips their 'supported' prompt over one (4); a
  connect made on a confirmation connects and holds, and the motion that triggered it waits for the
  next call (5); calibrate over a raised arm prints a clean refusal (6).

No test touches CAN, the real SDK, motorbridge or the real sudo: the bus is a recording fake, and
`confirm` runs on a pseudo-terminal with a stub sudo set in the child (`arm_hold.SUDO`).
"""

from __future__ import annotations

import builtins
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from test_held_arm_across_restart import _hold_a_rebot  # noqa: F401
from test_held_arm_every_path import _IN_A_TERMINAL, _clean_env, _dropped, _joint
from test_rebot_clear_error import (_install_fake_motorbridge, _install_fake_sdk, _Log, _mb_arm, _Motor,  # noqa: F401
                                    _rs_arm, no_sleep)

from cascade.config import Cfg, load_demo_config
from cascade.control import arm_hold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
ASK = "Do not retry on your own: ask the operator"
#: next_action quotes this many characters of the tool error ("ArmHeldError: " + the refusal)
QUOTED = 240 - len("ArmHeldError: ")
#: a motion command: the driver's joint target (its connect's own hold commands are not motions)
TARGET = "target"


@pytest.fixture(autouse=True)
def _private_stores(tmp_path):
    def pinned():
        assert str(arm_hold.hold_dir()).startswith(str(tmp_path)), "never the host's hold directory"
        log_path = getattr(arm_hold, "rest_log_path", None)  # (absent before this change: RED rows fail alone)
        assert log_path is None or str(log_path()).startswith(str(tmp_path)), "never the host's rest log"
        sudo = getattr(arm_hold, "SUDO", None)
        assert sudo is None or str(sudo).startswith(str(tmp_path)), "never the real sudo in-process"

    pinned()
    yield
    pinned()


def _reads(monkeypatch, log, table):
    """mechPos parameter reads, logged. `table`: joint -> a value, or a list served in order (its last
    value repeats); None = no answer."""
    left = {j: (list(v) if isinstance(v, (list, tuple)) else [v]) for j, v in table.items()}

    def read(self, param, *a):
        log.append(("param_read", self.name))
        vals = left.get(_joint(self.name))
        if not vals:
            return 0.0
        value = vals.pop(0) if len(vals) > 1 else vals[0]
        if value is None:
            raise RuntimeError("parameter 0x7019 not received")
        return float(value)

    monkeypatch.setattr(_Motor, "robstride_get_param_f32", read)


def _bus(monkeypatch, backend, table=None):
    """Both reBot transports over the recording fakes; every joint target a driver is sent (a motion)
    is logged as ("target",)."""
    from cascade.control.rebot_rs_arm import RebotRSArm
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    log = _Log()
    _install_fake_sdk(monkeypatch, log)
    _install_fake_motorbridge(monkeypatch, log)
    _reads(monkeypatch, log, table or {})
    for cls in (RebotRSArm, RebotRSMotorBridgeArm):
        def target(self, q, _orig=cls.send_joint_target):
            log.append((TARGET,))
            return _orig(self, q)

        monkeypatch.setattr(cls, "send_joint_target", target)
    return log, (_rs_arm if backend == "rebot_rs" else _mb_arm)


def _rows():
    path = arm_hold.rest_log_path()
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ── Important 1: confirm needs the operator's sudo password ───────────────────────────────────────

#: The child runs arm_hold.main with `arm_hold.SUDO` set to the stub (argv: src, stub, args...).
_LAUNCH = ("import sys; sys.path.insert(0, sys.argv[1]); from cascade.control import arm_hold; "
           "arm_hold.SUDO = sys.argv[2]; sys.exit(arm_hold.main(sys.argv[3:]))")


def _stub_sudo(tmp_path, *, nopasswd=False, password="operator-pw"):
    """A stand-in for /usr/bin/sudo: logs its arguments; `-n` succeeds only when `nopasswd`; otherwise
    it asks for a password at the terminal and accepts `password` only."""
    calls = tmp_path / "sudo_calls.log"
    stub = tmp_path / "stub-sudo"
    stub.write_text("#!/bin/sh\n"
                    f"echo \"$*\" >> '{calls}'\n"
                    "case \" $* \" in *\" -n \"*) exit " + ("0" if nopasswd else "1") + ";; esac\n"
                    "printf '[sudo stub] password for the operator: ' > /dev/tty\n"
                    "IFS= read -r pw < /dev/tty || exit 1\n"
                    f"[ \"$pw\" = '{password}' ] && exit 0\n"
                    "echo 'Sorry, try again.' > /dev/tty\n"
                    "exit 1\n")
    stub.chmod(0o755)
    return stub, calls


def _dialog(env, sudo, answers, *args, timeout_s=60.0):
    """`arm_hold confirm` on a pseudo-terminal (its controlling terminal, stdin and stdout), with
    `sudo` as arm_hold.SUDO. `answers`: (prompt bytes, typed text) in order. (exit code, output)."""
    import pty

    master, slave = pty.openpty()
    proc = subprocess.Popen([sys.executable, "-c", _IN_A_TERMINAL, "-c", _LAUNCH, str(REPO / "src"), str(sudo),
                             "confirm", *args], stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    out, todo, deadline = b"", list(answers), time.monotonic() + timeout_s
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
                if todo and todo[0][0] in out:
                    os.write(master, todo.pop(0)[1].encode() + b"\n")
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


PASSWORD_PROMPT = b"[sudo stub] password"
PHRASE_PROMPT = b"Type 'holding the arm'"


def test_confirm_asks_for_the_operators_sudo_password_then_the_phrase(tmp_path):
    """The r426b probe's RED row: an agent's shell with its marker variables unset, under script(1),
    piped the phrase and confirmed (rc 0). The phrase alone is no proof of a person: the operator's
    sudo password is, checked with the cached credentials ignored (`-k`)."""
    stub, calls = _stub_sudo(tmp_path)
    code, out = _dialog(_clean_env(), stub, [(PASSWORD_PROMPT, "operator-pw"), (PHRASE_PROMPT, "holding the arm")])
    assert code == 0 and "confirmed: you are supporting the arm" in out, out
    assert len(arm_hold.pending_confirmations()) == 1
    lines = calls.read_text().splitlines()
    assert len(lines) == 2 and "-n" in lines[0].split() and "-k" in lines[0].split(), lines
    assert {"-k", "-v"} <= set(lines[1].split()) and "-n" not in lines[1].split(), lines


@pytest.mark.parametrize("typed", ["wrong-password", "holding the arm"], ids=["wrong", "phrase_as_password"])
def test_confirm_refuses_without_the_password(tmp_path, typed):
    """The agent's shape (env scrubbed, a pty, the phrase typed at whatever prompt comes) now meets
    the password prompt first: nothing is written."""
    stub, _calls = _stub_sudo(tmp_path)
    code, out = _dialog(_clean_env(), stub, [(PASSWORD_PROMPT, typed), (PHRASE_PROMPT, "holding the arm")])
    assert code == 2 and "REFUSED" in out and "sudo" in out, out
    assert "Type 'holding the arm'" not in out, "the phrase is asked only after the password"
    assert arm_hold.pending_confirmations() == []


def test_confirm_refuses_a_sudo_that_needs_no_password(tmp_path):
    """NOPASSWD: sudo proves nothing about who types. Fail closed and say what is needed."""
    stub, calls = _stub_sudo(tmp_path, nopasswd=True)
    code, out = _dialog(_clean_env(), stub, [(PASSWORD_PROMPT, "operator-pw"), (PHRASE_PROMPT, "holding the arm")])
    assert code == 2 and "password-protected sudo" in out, out
    assert len(calls.read_text().splitlines()) == 1, "the interactive validation never ran"
    assert arm_hold.pending_confirmations() == []


def test_confirm_refuses_without_sudo(tmp_path):
    code, out = _dialog(_clean_env(), tmp_path / "no-such-sudo", [(PHRASE_PROMPT, "holding the arm")])
    assert code == 2 and "password-protected sudo" in out, out
    assert arm_hold.pending_confirmations() == []


@pytest.mark.parametrize("marker", ["AI_AGENT", "HERMES_SESSION_ID", "CLAUDECODE", "OPENCLAW_GATEWAY_TOKEN"])
def test_the_agent_refusal_names_no_variable_to_unset(tmp_path, marker):
    stub, calls = _stub_sudo(tmp_path)
    code, out = _dialog(_clean_env(**{marker: "1"}), stub, [(PASSWORD_PROMPT, "operator-pw")])
    assert code == 2 and "agent" in out and marker not in out and " is set" not in out, out
    assert not calls.exists(), "refused before sudo"
    assert arm_hold.pending_confirmations() == []


# ── Important 2 (a): the startup check clears a marker only within the runtime's 0.05 ───────────


def test_the_marker_clear_tolerance_is_the_runtimes_release_tolerance():
    from cascade.skills import rest_pose

    assert arm_hold.MARKER_CLEAR_TOL_RAD == rest_pose.REST_TOL_RAD == 0.05
    park = [0.0] * 6
    for off, cleared in ((0.06, False), (0.11, False), (0.04, True)):
        arm_hold.record("can0", arm="0", reason="held")
        q = [0.0, 0.0, off, 0.0, 0.0, 0.0]
        assert arm_hold.clear_if_at_rest("can0", q, park, again=q, log=lambda *_: None) is cleared, off
        assert (arm_hold.read("can0") is None) is cleared, off
        arm_hold.clear("can0", why="test")


@pytest.mark.parametrize(("off", "code"), [(0.08, 2), (0.04, 0)], ids=["0.08_kept", "0.04_cleared"])
def test_the_startup_check_keeps_a_marker_unless_every_joint_is_within_005(monkeypatch, no_sleep, capsys,  # noqa: F811
                                                                            off, code):
    """#426: the startup check cleared a marker anywhere within 0.12 rad, and the connect after it
    sent 7 STOP frames to an arm the runtime had refused to release (0.05-0.12 off: still holding)."""
    from test_rebot_gripper_reference import _check_script

    import cascade.control.rebot_rs_arm as rs

    log, _make = _bus(monkeypatch, "rebot_rs", {3: off})
    arm_hold.record("can0", arm="0", reason="the shutdown park stopped 0.08 rad short of park_q")
    mod = _check_script()
    q = [0.0, 0.0, off, 0.0, 0.0, 0.0]
    monkeypatch.setattr(mod, "read_rest_pose", lambda: (list(q), list(q)))
    monkeypatch.setattr(mod, "run", lambda arm, park, **k: 0)
    real = rs.RebotRSArm
    monkeypatch.setattr(rs, "RebotRSArm", lambda cfg: real(Cfg({"hw_yaml": "rebotarm_rs.yaml", "n_joints": 6})))
    assert mod.main(["--yes", "--reference-only"]) == code
    out = capsys.readouterr().out
    if code:
        assert arm_hold.read("can0") is not None and log == [], "kept: the connect sent nothing"
        assert "the hold stays" in out and "joint 3" in out and "the arm was NOT connected" in out, out
    else:
        assert arm_hold.read("can0") is None and "hold cleared" in out, out
        assert [e[0] for e in log].count("clear_error") == 7


# ── Important 2 (b) / Minor 2: every joint read twice, the reads must agree ───────────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_one_plausible_glitch_per_raised_joint_no_longer_lets_a_held_arm_through(monkeypatch, no_sleep, backend):  # noqa: F811
    """r426b probe row d2: a held arm raised at j2 0.6 / j3 0.9 whose FIRST read of each glitched to a
    near-rest value connected (7 drops). The second read disagrees: refused on reads alone."""
    log, make = _bus(monkeypatch, backend, {2: [0.0, 0.6], 3: [0.01, 0.9]})
    with pytest.raises(arm_hold.ArmHeldError, match="disagree") as err:
        make().connect()
    kinds = [e[0] for e in log]
    assert _dropped(log) == [] and set(kinds) <= {"connect", "param_read", "close_bus", "close"}, log
    assert kinds.count("param_read") == 12, "each joint read twice"
    assert "joint 2" in str(err.value) and "joint 3" in str(err.value)


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_two_close_reads_of_a_folded_arm_connect_as_before(monkeypatch, no_sleep, backend):  # noqa: F811
    log, make = _bus(monkeypatch, backend, {2: [0.10, 0.11], 5: [-0.004, 0.003]})
    make().connect()
    kinds = [e[0] for e in log]
    mode = kinds.index("mode_mit" if backend == "rebot_rs" else "ensure_mode")
    assert kinds[:mode].count("param_read") == 12 and kinds.count("clear_error") == 7


def test_two_reads_that_disagree_near_rest_are_no_measurement():
    """Both reads lie within the tolerance but 0.05 rad apart: one of them is a glitch, so the arm is
    not measured (the agreement check, not the tolerance, refuses it)."""
    park = [0.0] * 6
    problem = arm_hold.rest_problem([0.0] * 6, park, 0.12, again=[0.0, 0.0, 0.05, 0.0, 0.0, 0.0])
    assert problem is not None and "disagree" in problem and "joint 3" in problem, problem
    assert arm_hold.rest_problem([0.0] * 6, park, 0.12, again=[0.0, 0.0, 0.019, 0.0, 0.0, 0.0]) is None


def test_both_reads_must_be_within_the_tolerance():
    q, again = [0.0, 0.0, 0.11, 0.0, 0.0, 0.0], [0.0, 0.0, 0.125, 0.0, 0.0, 0.0]  # agree; the second is out
    for first, second in ((q, again), (again, q)):
        problem = arm_hold.rest_problem(first, [0.0] * 6, 0.12, again=second)
        assert problem is not None and "joint 3 is 0.125 rad from park_q" in problem, problem


def test_the_startup_check_reads_every_joint_twice(monkeypatch, capsys):
    from test_rebot_gripper_reference import _check_script

    import cascade.control.rebot_rs_arm as rs

    mod = _check_script()
    import diag_rebot_mb as diag

    served = {mid: [0.0, 0.0] for mid in diag.ARM_IDS}
    served[3] = [0.002, 0.7]  # a glitch on the first read; the arm is really up
    closed = []
    monkeypatch.setattr(diag, "_open_motors", lambda channel, model: (SimpleNamespace(
        close_bus=lambda: closed.append("close_bus"), close=lambda: closed.append("close")),
        {mid: mid for mid in diag.ARM_IDS}))
    monkeypatch.setattr(diag, "_read", lambda m, param, timeout_ms=300: served[m].pop(0))
    first, second = mod.read_rest_pose()
    assert first[2] == 0.002 and second[2] == 0.7 and closed == ["close_bus", "close"]
    served.update({mid: [0.0, 0.0] for mid in diag.ARM_IDS})
    served[3] = [0.002, 0.7]
    monkeypatch.setattr(rs, "RebotRSArm", lambda cfg: pytest.fail("connected over disagreeing reads"))
    assert mod.main(["--yes", "--reference-only"]) == 2
    out = capsys.readouterr().out
    assert "disagree" in out and "Nothing energized" in out, out


# ── Important 2 (c): every rest check is logged, bounded; the tolerance is per rig ────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_every_connect_logs_its_rest_reading(monkeypatch, no_sleep, backend):  # noqa: F811
    log, make = _bus(monkeypatch, backend, {3: [0.03, 0.031]})
    make().connect()
    _reads(monkeypatch, log, {2: [0.5, 0.5]})
    with pytest.raises(arm_hold.ArmHeldError):
        make().connect()
    rows = _rows()
    assert [r["source"] for r in rows] == [f"connect:{backend}"] * 2, rows
    ok, refused = rows
    assert ok["key"] == "can0" and ok["marker"] is False and ok["tol_rad"] == arm_hold.REST_TOL_RAD
    assert ok["q"][2] == 0.03 and ok["again"][2] == 0.031 and ok["park"] == [0.0] * 6
    assert ok["dev"][2] == pytest.approx(0.031) and ok["max_dev"] == pytest.approx(0.031)
    assert ok["verdict"] == "rest" and "at" in ok and "at_text" in ok
    assert refused["verdict"] != "rest" and "joint 2" in refused["verdict"], refused


def test_the_startup_check_logs_its_reading(monkeypatch, capsys):
    from test_rebot_gripper_reference import _check_script

    import cascade.control.rebot_rs_arm as rs

    mod = _check_script()
    q = [0.0, 0.004, 0.0, 0.0, 0.0, 0.0]
    monkeypatch.setattr(mod, "read_rest_pose", lambda: (list(q), list(q)))
    monkeypatch.setattr(mod, "run", lambda arm, park, **k: 0)
    monkeypatch.setattr(rs, "RebotRSArm", lambda cfg: SimpleNamespace(connect=lambda: None, disconnect=lambda: None,
                                                                         hold_key="can0"))
    assert mod.main(["--yes", "--reference-only"]) == 0
    rows = _rows()
    assert [r["source"] for r in rows] == ["startup_check"] and rows[0]["verdict"] == "rest", rows
    assert rows[0]["dev"][1] == pytest.approx(0.004)


def test_the_rest_log_stays_bounded(monkeypatch):
    monkeypatch.setattr(arm_hold, "REST_LOG_MAX_BYTES", 4096)
    for i in range(200):
        arm_hold.log_rest_reading(source="test", key="can0", q=[0.001 * i] * 6, again=[0.001 * i] * 6,
                                  park=[0.0] * 6, tol=0.12, marker=False, verdict="rest")
    path = arm_hold.rest_log_path()
    older = path.with_name(path.name + ".1")
    assert path.stat().st_size <= 4096 + 1024 and older.exists() and older.stat().st_size <= 4096 + 1024
    assert len(list(path.parent.iterdir())) == 2, "one current file and one older one, never more"


def test_a_rest_log_that_cannot_be_written_never_stops_a_connect(monkeypatch, no_sleep, tmp_path):  # noqa: F811
    (tmp_path / "a_directory").mkdir()
    monkeypatch.setenv(arm_hold.REST_LOG_ENV, str(tmp_path / "a_directory"))
    log, make = _bus(monkeypatch, "rebot_rs")
    make().connect()
    assert [e[0] for e in log].count("clear_error") == 7


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_the_no_marker_tolerance_is_a_per_rig_profile_value(monkeypatch, no_sleep, backend):  # noqa: F811
    log, make = _bus(monkeypatch, backend, {3: 0.08})
    make().connect()  # the default 0.12: a normal cold start connects as before
    arm = make()
    arm._cfg._data["rest_check_tol_rad"] = 0.05
    arm = type(arm)(arm._cfg)
    with pytest.raises(arm_hold.ArmHeldError, match="joint 3"):
        arm.connect()
    assert _rows()[-1]["tol_rad"] == 0.05


@pytest.mark.parametrize("bad", [0.2, 0.0, -0.1, float("nan"), "x", True])
def test_the_per_rig_tolerance_may_only_tighten(bad):
    with pytest.raises(ValueError, match="rest_check_tol_rad"):
        arm_hold.rest_tolerance(Cfg({"rest_check_tol_rad": bad}))
    assert arm_hold.rest_tolerance(Cfg({})) == arm_hold.REST_TOL_RAD == 0.12
    assert arm_hold.rest_tolerance(Cfg({"rest_check_tol_rad": 0.06})) == 0.06


def test_the_shipped_rebot_profiles_name_their_tolerance():
    for arm in ("rebot_rs", "rebot_rs_mb", "rebot_rs_reach"):
        cfg = load_demo_config(camera="mock", arm=arm, llm="mock").arm
        assert cfg.get("rest_check_tol_rad") == 0.12, arm


def _report(*argv):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "rest_tolerance_report.py"), *argv],
                          capture_output=True, text=True, timeout=120, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})


class _Clock:
    """`arm_hold.time` with a wall clock the test sets (every other name is the real module's)."""

    def __init__(self, t: float = 1.79e9):
        self.t = t

    def time(self) -> float:
        return self.t

    def __getattr__(self, name):
        return getattr(time, name)


def _clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(arm_hold, "time", clock)
    return clock


def _q(j3: float, rng=None) -> list:
    q = list(rng.uniform(-0.02, 0.02, 6)) if rng is not None else [0.0] * 6
    q[2] = j3
    return q


def _row(clock, q, *, source="connect:rebot_rs", tol=0.12, marker=False, again=None):
    """One logged rest check, with the verdict the check itself gave (`rest_problem` at `tol`)."""
    problem = arm_hold.rest_problem(q, [0.0] * 6, tol, again=q if again is None else again)
    arm_hold.log_rest_reading(source=source, key="can0", q=q, again=q if again is None else again,
                              park=[0.0] * 6, tol=tol, marker=marker, verdict="rest" if problem is None else problem)
    clock.t += 2.0


def _up(clock, q, *, tol=0.12):
    """One `~/cascade-demo.sh up`: the startup check logs its reading, then its own connect logs the
    same rest seconds later (>= 2 rows per start); the next start is an hour later."""
    _row(clock, q, source="startup_check", tol=tol)
    _row(clock, q, source="connect:rebot_rs", tol=tol)
    clock.t += 3600.0


def _report_json(*argv) -> dict:
    run = _report("--log", str(arm_hold.rest_log_path()), "--json", *argv)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_the_rest_tolerance_report_recommends_from_the_folded_unpowered_starts(monkeypatch):
    clock, rng = _clock(monkeypatch), np.random.default_rng(0)
    for _ in range(30):
        _up(clock, _q(0.031, rng))  # joint 3: the spread's worst
    clock.t += 3600.0
    # rows the report must not learn from: a held arm (marker), two reads that disagree, a raised arm
    _row(clock, _q(0.09), source="startup_check", marker=True)
    clock.t += 3600.0
    _row(clock, [0.0] * 6, again=[0.0, 0.6, 0, 0, 0, 0])
    clock.t += 3600.0
    _row(clock, [0.0, 0.6, 0, 0, 0, 0])
    out = _report_json("--current", "0.12")
    assert out["samples"] == 30 and out["readings"] == 60, out  # starts, not log rows
    assert out["max_dev_rad"] == pytest.approx(0.031, abs=1e-6) and out["worst_joint"] == 3, out
    assert out["refused"] == 2 and out["ignored"] == 1, out
    assert [r["source"] for r in out["refused_rows"]] == ["connect:rebot_rs"] * 2, out["refused_rows"]
    assert out["recommended_rad"] == 0.06, out  # 0.031 + 0.02 margin, up to the next 0.01, >= 0.05
    text = _report("--log", str(arm_hold.rest_log_path()), "--current", "0.12")
    assert "rest_check_tol_rad: 0.06" in text.stdout and "refused" in text.stdout, text.stdout


def test_the_report_never_learns_a_refused_row(monkeypatch):
    """#431 re-review: current 0.05, 30 folded starts at 0.01, and ONE no-marker arm held 0.10 off
    park that the 0.05 connect refused: main learned it as rest spread and said 'set 0.12'."""
    clock = _clock(monkeypatch)
    for _ in range(30):
        _up(clock, _q(0.01))
    _row(clock, _q(0.10), tol=0.05)
    out = _report_json("--current", "0.05")
    assert out["samples"] == 30 and out["max_dev_rad"] == pytest.approx(0.01, abs=1e-6), out
    assert out["recommended_rad"] == 0.05 and "0.12" not in out["advice"], out
    assert out["refused"] == 1 and out["refused_rows"][0]["verdict"].startswith("joint 3 is 0.100 rad"), out


def test_the_report_never_advises_loosening_the_current_value(monkeypatch):
    """#431 re-review: every start passed at 0.05 with up to 0.035 off; + the 0.02 margin main said
    'set 0.06', above the value that let them all pass. The report never recommends above it."""
    clock = _clock(monkeypatch)
    for _ in range(30):
        _up(clock, _q(0.035), tol=0.05)
    out = _report_json("--current", "0.05")
    assert out["recommended_rad"] is None and out["current_rad"] == 0.05, out
    assert "keep rest_check_tol_rad: 0.05" in out["advice"] and "0.06" not in out["advice"], out
    tighter = _report_json("--current", "0.04")
    assert tighter["recommended_rad"] is None and "keep rest_check_tol_rad: 0.04" in tighter["advice"], tighter


def test_the_report_reads_the_current_value_from_the_profile(monkeypatch):
    clock = _clock(monkeypatch)
    for _ in range(20):
        _up(clock, _q(0.01))
    out = _report_json()
    assert out["current_rad"] == 0.12 and out["recommended_rad"] == 0.05, out  # the shipped rebot_rs.yaml


@pytest.mark.parametrize(("ups", "enough"), [(19, False), (20, True)])
def test_the_report_counts_starts_not_log_rows(monkeypatch, ups, enough):
    """Every `up` logs >= 2 rows (the startup check and its connect): 19 ups are 38 rows but 19 starts."""
    clock = _clock(monkeypatch)
    for _ in range(ups):
        _up(clock, _q(0.01))
    out = _report_json("--current", "0.12")
    assert out["samples"] == ups and out["readings"] == 2 * ups, out
    assert (out["recommended_rad"] is not None) is enough, out
    if not enough:
        assert f"not enough starts ({ups} of 20)" in out["advice"], out


@pytest.mark.parametrize("argv", [("--min-samples", "0"), ("--min-samples", "1"), ("--min-samples", "9"),
                                  ("--margin", "-0.05"), ("--margin", "nan"), ("--current", "0.5"),
                                  ("--current", "0")], ids=lambda a: " ".join(a))
def test_the_report_refuses_options_that_would_weaken_its_advice(argv):
    run = _report("--log", str(arm_hold.rest_log_path()), "--json", *argv)
    assert run.returncode == 2 and "Traceback" not in run.stderr and argv[0] in run.stderr, run.stderr


@pytest.mark.parametrize("kw", [{"min_samples": 0}, {"min_samples": 1}, {"margin": -0.05}, {"current": 0.2}])
def test_the_report_analysis_refuses_the_same_values(kw):
    import rest_tolerance_report as report

    args = {"margin": 0.02, "min_samples": 20, **kw}
    with pytest.raises(ValueError):
        report.analyse([], **args)


def test_the_rest_tolerance_report_keeps_the_value_without_enough_readings(monkeypatch):
    clock = _clock(monkeypatch)
    for _ in range(3):
        _up(clock, _q(0.0))
    out = _report_json("--current", "0.12")
    assert out["recommended_rad"] is None and "not enough" in out["advice"], out
    empty = _report("--log", str(arm_hold.rest_log_path()) + ".missing", "--json", "--current", "0.12")
    assert empty.returncode == 0 and json.loads(empty.stdout)["samples"] == 0, empty.stdout


# ── Minors 1 and 3: the ask comes first; the off-park refusal names the joints ────────────────────


def test_every_hold_refusal_starts_with_the_operator_ask():
    marker = {"reason": "the shutdown park was refused (SafetyViolation: perception watchdog: no observation "
                        "yet)", "pid": 4242, "at_text": "2026-10-21 02:03:04"}
    raised = arm_hold.raised_refusal("can0", arm_hold.rest_problem([0, 0.45, 0.2, 0, 0, 0], [0.0] * 6))
    for text in (arm_hold.refusal(marker, "can0"), raised):
        assert text.startswith(ASK), text
        cut = text[:QUOTED]
        assert "operator" in cut and "an agent cannot confirm" in cut, cut


def test_the_off_park_refusal_names_the_joints_and_says_lowering_is_enough():
    problem = arm_hold.rest_problem([0.0, 0.45, 0.2, 0.0, 0.03, 0.0], [0.0] * 6, 0.12)
    assert "joint 2" in problem and "joint 3" in problem and "joint 5" not in problem, problem
    text = arm_hold.raised_refusal("can0", problem)
    assert "lower the arm to rest" in text and "no confirmation" in text, text
    assert "or to support it and confirm" in text, text


@needs_pin
def test_next_action_keeps_the_operator_ask(monkeypatch, tmp_path, no_sleep):  # noqa: F811
    """#426: next_action quotes the first 240 characters of the error, and the ask came last: the
    agent read the hold but never the instruction to ask the operator."""
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
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
    try:
        out = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        assert out["ok"] is False and out["refused"] == "arm_held", out
        assert "ask the operator" in out["next_action"] and "(truncated" in out["next_action"], out["next_action"]
        # move_to_rest's own held refusal (not an ArmHeldError): the same ask, first
        out = rt.execute("move_to_rest", {})
        assert out["refused"] == "arm_held" and out["error"].startswith(ASK), out["error"]
        assert "an agent cannot confirm" in out["next_action"], out["next_action"]
    finally:
        shutdown_runtime(rt, arm)


# ── Minor 4: the bring-up scripts read the marker; --yes never skips 'supported' over one ─────────


def _bringup(monkeypatch, script, log, *, yes=True):
    import test_rebot_bringup_scripts_clear as B

    mod = __import__(script)
    opened = []
    B._fake_motorbridge(monkeypatch, log)
    real = sys.modules["motorbridge"].Controller.__init__

    def opening(self, channel=None):
        opened.append(channel)
        real(self, channel)

    monkeypatch.setattr(sys.modules["motorbridge"].Controller, "__init__", opening)
    monkeypatch.setattr(mod, "_sdk_motors", lambda: dict(B._SDK))
    monkeypatch.setattr(mod, "_profile_gains", lambda mid: (5.0, 0.5))
    monkeypatch.setattr(mod, "_read", lambda m, p, timeout_ms=300: 0.5)
    monkeypatch.setattr(mod, "_urdf_local_limits", B._limits)
    argv = ["jog", "--joint", "1", "--hold-only"] if script == "jog_rebot_mb" else ["sc", "--joints", "1"]
    monkeypatch.setattr(sys, "argv", argv + (["--yes"] if yes else []))
    return mod, opened, B._Energized


@pytest.mark.parametrize("script", ["jog_rebot_mb", "sign_check_rebot_mb"])
def test_a_bring_up_script_over_a_marker_sends_nothing_without_a_confirmation(monkeypatch, capsys, script):
    log = []
    mod, opened, _energized = _bringup(monkeypatch, script, log)
    arm_hold.record("can0", arm="0", reason="held by the robot server's teardown")
    assert mod.main() == 3
    assert log == [] and opened == [], "refused before the bus opened"
    out = capsys.readouterr().out
    assert "POWERED and HOLDING" in out and "NOT" in out, out
    assert arm_hold.read("can0") is not None


@pytest.mark.parametrize("answer", ["go", "no"])
@pytest.mark.parametrize("script", ["jog_rebot_mb", "sign_check_rebot_mb"])
def test_yes_never_skips_the_supported_prompt_over_a_marker(monkeypatch, capsys, script, answer):
    log = []
    mod, _opened, energized = _bringup(monkeypatch, script, log)
    arm_hold.record("can0", arm="0", reason="held by the robot server's teardown")
    arm_hold.confirm("holding the arm")
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt="": asked.append(prompt) or answer)
    if answer == "go":
        with pytest.raises(energized):
            mod.main()
        assert {"ensure_mode", "clear_error"} <= {e[0] for e in log}
    else:
        assert mod.main() == 0
        assert not {"ensure_mode", "clear_error", "enable_all"} & {e[0] for e in log}, log
        assert len(arm_hold.pending_confirmations()) == 1, "not used: kept for a retry in its window"
        assert arm_hold.read("can0") is not None
    assert len(asked) == 1, "--yes did not skip the prompt"
    assert "POWERED and HOLDING" in capsys.readouterr().out


# ── Minor 5: a connect on a confirmation connects and holds; the next call moves ──────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_lazy_connect_on_a_confirmation_moves_nothing_until_the_next_call(monkeypatch, no_sleep, backend):  # noqa: F811
    from cascade.control.lazy_arm import LazyArm

    log, make = _bus(monkeypatch, backend, {2: 0.6, 3: 0.9})  # raised, held
    arm_hold.record("can0", arm="0", reason="held")
    arm_hold.confirm("holding the arm")
    woke = []
    lazy = LazyArm(make, profile_type=backend, hold_key="can0", on_materialize=lambda: woke.append(1))
    q = np.array([0.0, 0.6, 0.9, 0.0, 0.0, 0.0])
    with pytest.raises(arm_hold.ConnectedAndHolding, match="connected and holding; let go of the arm and retry"):
        lazy.send_joint_target(q)
    kinds = [e[0] for e in log]
    assert lazy.connected and woke == [1], "materialized, and its hook ran"
    assert kinds.count("clear_error") == 7 and TARGET not in kinds, "connected; the triggering motion was not sent"
    assert arm_hold.read("can0") is None and arm_hold.pending_confirmations() == []
    n0 = len(log)
    with pytest.raises(arm_hold.ConnectedAndHolding):
        lazy.send_joint_target(q)  # nobody has told the operator to let go yet (#431 review Important A)
    lazy.told_to_let_go()  # the caller answered 'let go of the arm and retry'
    lazy.next_call()
    lazy.send_joint_target(q)  # the next call moves
    assert [e[0] for e in log[n0:]].count(TARGET) == 1 and "clear_error" not in [e[0] for e in log[n0:]], log[n0:]


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_an_unconfirmed_lazy_connect_moves_at_once_as_before(monkeypatch, no_sleep, backend):  # noqa: F811
    from cascade.control.lazy_arm import LazyArm

    log, make = _bus(monkeypatch, backend)
    lazy = LazyArm(make, profile_type=backend, hold_key="can0")
    lazy.send_joint_target(np.zeros(6))
    assert [e[0] for e in log].count(TARGET) == 1


@needs_pin
def test_an_mcp_motion_on_a_confirmation_connects_holds_and_asks_to_let_go(monkeypatch, tmp_path, no_sleep):  # noqa: F811
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.control.rebot_rs_arm import RebotRSArm

    log, _make = _bus(monkeypatch, "rebot_rs")
    monkeypatch.setattr(sys, "path", list(sys.path))
    _hold_a_rebot(log)
    cfg = load_demo_config(camera="mock", arm="rebot_rs", llm="mock")
    occ = cfg._data.get("occupancy")
    if isinstance(occ, dict) and occ.get("required"):
        occ["required"] = False
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: RebotRSArm(acfg))
    rt, arm = demo.build_runtime(cfg, tmp_path / "run2", view=False, serve=False, lazy_arm=True)
    try:
        _reads(monkeypatch, log, {2: 1.2, 3: 1.2})  # held up where the last session left it
        arm_hold.confirm("holding the arm")
        n0 = len(log)
        out = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        sent = [e[0] for e in log[n0:]]
        assert out["ok"] is False and out["error"].startswith("ConnectedAndHolding: connected and holding; let go "
                                                              "of the arm and retry"), out
        assert out.get("refused") == "connected_on_confirmation" and "let go" in out["next_action"], out
        assert sent.count("clear_error") == 7 and "enable" in sent and TARGET not in sent, sent
        n1 = len(log)
        again = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        sent = [e[0] for e in log[n1:]]
        assert again.get("refused") is None and TARGET in sent and "clear_error" not in sent, (again, sent)
    finally:
        shutdown_runtime(rt, arm)


def _confirmed_arm_check(monkeypatch, answer):
    from test_home_rebot_rs import RIG_REST, FakeCamera, _builder, _main

    from cascade.control.mock_arm import MockArm

    monkeypatch.setattr(MockArm, "hold_key", property(lambda self: "can0"))
    events = []
    inner = _builder(RIG_REST, events)

    def build(cfg, dry_run):
        raw, safe, kin = inner(cfg, dry_run)
        raw.connected_on_confirmation = True
        return raw, safe, kin

    def ask(prompt=""):
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr(builtins, "input", ask)
    return _main(["--yes", "--hold-s", "0"], build, FakeCamera()), events


@needs_pin
def test_the_arm_check_on_a_confirmation_moves_only_after_the_operator_lets_go(monkeypatch, capsys):
    code, events = _confirmed_arm_check(monkeypatch, "go")
    assert code == 0 and [e for e in events if e[0] == "target"], events
    assert "connected and holding; let go of the arm" in capsys.readouterr().out


@needs_pin
@pytest.mark.parametrize("answer", [None, "no"], ids=["eof", "no"])
def test_the_arm_check_on_a_confirmation_holds_when_the_operator_does_not_go_on(monkeypatch, capsys, answer):
    code, events = _confirmed_arm_check(monkeypatch, answer)
    assert code == 3 and not [e for e in events if e[0] == "target"], events
    assert ("disconnect",) not in events, "the arm stays held"
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "home_rebot_rs.py", marker


# ── Minors 5 and 6: calibrate ─────────────────────────────────────────────────────────────────────


def _calibrate(monkeypatch, tmp_path, first_error, answers=()):
    """cli.main over the dry-run rig whose FIRST arm read raises `first_error` (what the LazyArm's
    connect raises at the first sweep motion)."""
    from cascade.calibration import cli
    from cascade.control.mock_arm import MockArm

    monkeypatch.setattr(MockArm, "hold_key", property(lambda self: "can0"))
    made, calls = {}, []

    def build(cfg, args):
        dr = cli.dry_run_rig(args.arm, cfg.camera.extrinsics.mode, noise_px=0.3, seed=0)
        rig = cli.Rig(safe_arm=dr.safe_arm, raw_arm=dr.raw_arm, kin=dr.kin, camera=dr.camera, home_q=dr.home_q,
                      ee_frame=dr.ee_frame, camera_serial=cfg.camera.get("serial", ""))
        raw, fired = rig.raw_arm, []
        orig = raw.get_state

        def get_state(*a, **kw):
            if not fired:
                fired.append(1)
                raise first_error
            return orig(*a, **kw)

        raw.get_state = get_state
        orig_disc = raw.disconnect
        raw.disconnect = lambda: (calls.append("disconnect"), orig_disc())[1]
        made["rig"] = rig
        return rig

    monkeypatch.setattr(cli, "build_rig", build)
    it = iter(answers)
    monkeypatch.setattr(cli, "_prompt", lambda msg="": calls.append(("prompt", msg)) or next(it))
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes", "--settle-time", "0",
                   "--stable-frames", "2", "--out", str(tmp_path / "out.json"), "--run-dir", str(tmp_path / "run")])
    return rc, made["rig"], calls


@needs_pin
def test_calibrate_over_a_raised_arm_prints_a_clean_refusal(monkeypatch, tmp_path, capsys):
    refusal = arm_hold.ArmHeldError(arm_hold.raised_refusal("can0", "joint 3 is 0.900 rad from park_q (> 0.12)"))
    rc, rig, calls = _calibrate(monkeypatch, tmp_path, refusal)
    out = capsys.readouterr()
    assert rc == 3 and "the arm was NOT connected" in out.out and "Traceback" not in out.out + out.err, out
    assert rig.raw_arm.commands == [] and "disconnect" not in calls, "nothing moved, no park, no torque cut"
    assert arm_hold.read("can0") is None


@needs_pin
def test_calibrate_on_a_confirmation_sweeps_only_after_the_operator_lets_go(monkeypatch, tmp_path, capsys):
    rc, rig, calls = _calibrate(monkeypatch, tmp_path, arm_hold.connected_and_holding("can0"), answers=["go"])
    prompts = [c for c in calls if isinstance(c, tuple)]
    assert len(prompts) == 1 and "let go of the arm" in prompts[0][1].lower(), calls
    assert rig.raw_arm.commands, "the sweep ran after 'go'"
    assert rc == 0 and calls[-1] == "disconnect", (rc, calls)


@needs_pin
def test_calibrate_on_a_confirmation_holds_when_the_operator_does_not_go_on(monkeypatch, tmp_path, capsys):
    rc, rig, calls = _calibrate(monkeypatch, tmp_path, arm_hold.connected_and_holding("can0"), answers=["no"])
    assert rc == 1 and rig.raw_arm.commands == [] and "disconnect" not in calls, calls
    marker = arm_hold.read("can0")
    assert marker is not None and marker["source"] == "calibrate", marker
    assert "POWERED and HOLDING" in capsys.readouterr().err


@needs_pin
@pytest.mark.parametrize("answer", ["go", "no"])
def test_calibrate_over_a_lazy_arm_moves_only_once_the_operator_says_go(monkeypatch, tmp_path, capsys, answer):
    """The real LazyArm: after its confirmed connect it refuses every command until the operator lets go
    (#431 review Important A). Calibrate's 'go' says so; without it the sweep never moves."""
    from cascade.calibration import cli
    from cascade.control.lazy_arm import LazyArm
    from cascade.control.mock_arm import MockArm
    from cascade.safety.harness import SafeArm

    monkeypatch.setattr(MockArm, "hold_key", property(lambda self: "can0"))
    made, answers = {}, iter([answer])

    def build(cfg, args):
        dr = cli.dry_run_rig(args.arm, cfg.camera.extrinsics.mode, noise_px=0.3, seed=0)
        dr.raw_arm.connected_on_confirmation = True  # what a reBot connect on a confirmation reports
        lazy = LazyArm(lambda: dr.raw_arm, n_joints=dr.raw_arm.n_joints, profile_type="rebot_rs", hold_key="can0")
        safe = SafeArm(lazy, dr.safe_arm.harness, motion_planner=dr.safe_arm.motion_planner)
        made["mock"] = dr.raw_arm
        return cli.Rig(safe_arm=safe, raw_arm=lazy, kin=dr.kin, camera=dr.camera, home_q=dr.home_q,
                       ee_frame=dr.ee_frame, camera_serial=cfg.camera.get("serial", ""))

    monkeypatch.setattr(cli, "build_rig", build)
    monkeypatch.setattr(cli, "_prompt", lambda msg="": next(answers))
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes", "--settle-time", "0",
                   "--stable-frames", "2", "--out", str(tmp_path / "out.json"), "--run-dir", str(tmp_path / "run")])
    if answer == "go":
        assert rc == 0 and made["mock"].commands, "the sweep ran after 'go'"
    else:
        assert rc == 1 and made["mock"].commands == [], "stopped before any motion"
