"""#434 re-review minors (CPU only, fake bridges on ephemeral ports, no live port).

2. SIGUSR1, the operator's reset channel, reports in server.log (stderr): the stop it cleared and why it was
   latched, or that the reset was refused and why (the worker used to discard ``reset_now``'s result);
4. the fleet's stopped-robot refusal names no tool, like the composed runtime's;
5. ``scripts/replay_gate_jsonl.py`` exit codes: 0 = actions replayed and none refused, 1 = a channel refused,
   2 = no action replayed (empty or admission-only record) or any error, never a traceback;
6. a JSON integer too big for a float (``float()`` raises OverflowError) is a STOP row, not an exception.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from cascade.skills import humanoid_locomanip as hl

from test_humanoid_gate_hardening import (CARRY, REPO, ScriptedBridge, _mcp_server, _payload, _recorded_segment,
                                          _replay_module, composed_runtime)
from test_humanoid_locomanip import IN_BIN, FakeBridge, action, make_gate, profile, state, walk_script

HUGE = 10 ** 400  # a JSON integer float() cannot hold: OverflowError
LIVE_PORTS = {5555, 5556, 5557, 5558, 5566, 8080, 8081, 8091, 8092, 8093, 8611, 18481, 18482, 18741, 17710, 17810,
              17910}


@pytest.fixture(autouse=True)
def _no_live_port(monkeypatch):
    """Socket guard: the fakes listen on ephemeral ports; a live service port is never reached."""
    real = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and len(address) >= 2 and address[1] in LIVE_PORTS:
            raise OSError(f"test socket guard: connect to {address!r} refused")
        return real(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


# ------------------------------------------------------- 6. an integer too big for a float is a STOP row
def _set(data, key, index, value):
    if index is None:
        data[key] = value
    else:
        data[key][index] = value
    return data


@pytest.mark.parametrize("key, index", [("joint_targets", 2), ("joint_targets", 1), ("joint_targets", 4), ("nav", 0),
                                        ("torso_rpy", 2), ("base_height", None)],
                         ids=["arm", "knee_wbc", "finger", "nav", "torso", "base_height"])
def test_a_command_integer_too_big_for_a_float_is_a_stop_row(key, index):
    gate = make_gate()
    d = gate.check(1, state(1), _set(action(), key, index, HUGE))
    assert d.verdict == "STOP" and d.clamps == () and gate.latched is not None
    assert [r.name for r in d.rows if not r.ok] == ["state_finite"] and "malformed step" in d.reason


@pytest.mark.parametrize("key, index", [("joint_pos", 2), ("joint_vel", 1), ("pelvis_pos", 2),
                                        ("pelvis_quat_wxyz", 0), ("floor_z", None), ("sim_time_s", None)])
def test_a_measured_integer_too_big_for_a_float_is_a_stop_row_in_check_and_admit(key, index):
    for decide in (lambda g, st: g.check(1, st, action()), lambda g, st: g.admit(1, st)):
        gate = make_gate()
        d = decide(gate, _set(state(1), key, index, HUGE))
        assert d.verdict == "STOP" and "malformed step" in d.reason and gate.latched is not None


def test_a_huge_integer_mid_segment_stops_at_that_step_and_keeps_the_bridge_session(tmp_path):
    """Through the domain the step is answered STOP (hold steps run, the segment ends normally): the bridge
    session stays open, so the operator's reset clears it and the robot runs again."""
    def bad(i, step):
        if i == 2:
            step = json.loads(json.dumps(step))
            step["action"]["joint_targets"][2] = HUGE
        return step

    fake = ScriptedBridge(walk_script(5), bad, final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    try:
        out = domain.execute("carry_object", CARRY)
        assert out["stopped"] and out["gate"]["first_stop"]["step"] == 2, out
        assert fake.verdicts == ["pass", "pass", "stop", "stop", "stop"] and fake.applied == 2
        assert domain._session_closed is None and "malformed step" in domain.gate.latched
        assert domain.reset_stop()["ok"] is True and domain.gate.latched is None
    finally:
        domain.close()


# ---------------------------------------------------- 4. the fleet's refusal names no tool
def test_the_fleet_refusal_for_a_stopped_robot_matches_the_runtime_and_names_no_tool(tmp_path):
    from cascade.apps.fleet_mcp import FleetMcpServer
    from cascade.robotics.fleet import FleetRuntime

    fake = FakeBridge(walk_script(2))
    humanoid = composed_runtime(fake, tmp_path / "g1")
    robot = humanoid.cfg.robot_id
    fleet = FleetRuntime({robot: humanoid})
    output = queue.Queue()
    server = FleetMcpServer(fleet, output.put, max_workers=2)
    try:
        humanoid.execute("emergency_stop", {})
        own = humanoid.execute("carry.carry_object", CARRY)["error"]  # the composed runtime's own refusal
        assert own == "ValueError: robot stopped: ask the operator to clear the stop"
        assert fleet.stop(robot)["ok"]
        direct = fleet.execute(robot, "carry.carry_object", CARRY)
        assert direct == {"robot_id": robot, "ok": False, "error": own.removeprefix("ValueError: ")}
        server.accept({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "fleet.execute", "arguments": {"robot_id": robot, "episode_id": humanoid.episode_id,
                                                   "tool": "carry.carry_object", "arguments": CARRY}}})
        reply = json.loads(output.get(timeout=5)["result"]["content"][-1]["text"])
        assert reply["ok"] is False and reply["error"] == own.removeprefix("ValueError: ")
        assert "reset_stop" not in json.dumps(direct) and "reset_stop" not in reply["error"]
        assert fake.segments == [] and humanoid.domains["carry"].runtime.gate.latched is not None
    finally:
        assert server.close()["complete"]


# --------------------------------------------------------------------- 5. replay exit codes
def _session_beside(path):
    return path.parent.parent.parent / "session.json"


@pytest.mark.parametrize("keep", ["empty", "blank_lines", "admission_only"])
def test_replay_of_a_record_without_an_action_exits_2_never_a_false_all_clear(tmp_path, capsys, keep):
    path = _recorded_segment(tmp_path)
    lines = path.read_text().splitlines()
    path.write_text({"empty": "", "blank_lines": "\n\n", "admission_only": lines[0] + "\n"}[keep])
    assert _session_beside(path).exists()
    module = _replay_module()
    assert module.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert "0 recorded actions replayed" in captured.out and "no channel refused" not in captured.out
    assert "no recorded action replayed: nothing was checked" in captured.out + captured.err
    assert module.main([str(path), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["actions"] == 0


def test_replay_of_a_huge_integer_action_is_a_refusal_not_a_crash(tmp_path, capsys):
    path = _recorded_segment(tmp_path)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    next(line for line in lines if line.get("step") == 3 and line.get("phase") == "policy")["action"]["nav"][0] = HUGE
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    assert _replay_module().main([str(path), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["channels"]["state_finite"]["first"]["step"] == 3
    assert report["channels"]["state_finite"]["refused_steps"] == 1


def test_replay_reports_an_error_inside_the_replay_as_exit_2_without_a_traceback(tmp_path, capsys, monkeypatch):
    path = _recorded_segment(tmp_path)
    module = _replay_module()

    def broken(*_a, **_k):
        raise RuntimeError("replay broke")

    monkeypatch.setattr(module, "replay", broken)
    assert module.main([str(path)]) == 2
    err = capsys.readouterr().err
    assert "RuntimeError: replay broke" in err and "Traceback" not in err


def test_replay_exit_code_for_any_error_is_2_in_a_real_process(tmp_path):
    """The script's own exit status (``sys.exit(main())``): input the JSON decoder cannot even nest exits 2
    with one line on stderr, never 1 (the code of a refusal) with a traceback."""
    path = _recorded_segment(tmp_path)
    path.write_text("[" * 100_000 + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "replay_gate_jsonl.py"), str(path)],
                          capture_output=True, text=True, timeout=120, env=env, cwd=str(tmp_path))
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-400:])
    assert "Traceback" not in proc.stderr and proc.stderr.startswith("replay_gate_jsonl: ")


# -------------------------------------------------- 2. SIGUSR1 tells the operator what it did
def _signal_worker(server, script):
    """The MCP server's real serial worker on the main thread inside the real StopSignals(staff_reset=True);
    ``script(feed, responses, resets)`` runs on another thread and sends the signal from there."""
    from cascade.apps.mcp_server import _admit, _worker_loop
    from cascade.apps.signal_stop import StopSignals

    assert threading.current_thread() is threading.main_thread()
    inbox, eof, responses, resets, errors = queue.Queue(), object(), {}, [], []
    real_reset = server.reset_now

    def recording_reset(*a, **k):
        out = real_reset(*a, **k)
        resets.append(_payload(out))
        return out

    server.reset_now = recording_reset

    def send(resp):
        responses[resp.get("id")] = resp

    def feed(frame):
        if not _admit(server, frame, send):
            inbox.put((frame, send))

    def run():
        try:
            script(feed, responses, resets)
        except BaseException as exc:  # noqa: BLE001 -- reported by the assertion below
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            inbox.put(eof)

    with StopSignals(staff_reset=True) as signals:
        server._signals = signals
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        rc = _worker_loop(server, signals, inbox, eof)
        thread.join(15)
    assert rc == 0 and errors == [] and not thread.is_alive()


def _wait(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not pred() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pred()


def _call(ident, name, args):
    return {"jsonrpc": "2.0", "id": ident, "method": "tools/call", "params": {"name": name, "arguments": args}}


def _sigusr1_then_list(feed, responses, resets, ident):
    """Send SIGUSR1 from this (non-main) thread, then queue a tools/list behind it: its answer proves the
    worker finished the signal's stop + reset + log line."""
    n = len(resets)
    os.kill(os.getpid(), signal.SIGUSR1)
    _wait(lambda: len(resets) > n)
    feed({"jsonrpc": "2.0", "id": ident, "method": "tools/list"})
    _wait(lambda: ident in responses)


def _usr1_lines(err):
    return [line for line in err.splitlines() if "SIGUSR1" in line]


@pytest.mark.parametrize("stop", ["emergency_stop", "gate_stop"])
def test_sigusr1_logs_the_stop_it_cleared(tmp_path, monkeypatch, capsys, stop):
    fake = FakeBridge(walk_script(6, tilt_at=3) if stop == "gate_stop" else walk_script(2),
                      final=state(99, box_pos=IN_BIN))
    server = _mcp_server(fake, tmp_path, monkeypatch)
    rows = {}

    def script(feed, responses, resets):
        feed(_call(1, "carry.describe_scene", {}))
        _wait(lambda: 1 in responses)
        feed(_call(2, "emergency_stop" if stop == "emergency_stop" else "carry.carry_object",
                   {} if stop == "emergency_stop" else CARRY))
        _wait(lambda: 2 in responses)
        rows["latched"] = server._runtime.domains["carry"].runtime.gate.latched
        _sigusr1_then_list(feed, responses, resets, 3)
        rows["resets"] = list(resets)
        rows["after"] = server._runtime.domains["carry"].runtime.gate.latched

    try:
        capsys.readouterr()
        _signal_worker(server, script)
        err = capsys.readouterr().err
    finally:
        server.shutdown()
    reason = "emergency stop requested" if stop == "emergency_stop" else "pelvis_tilt_deg"
    assert rows["latched"] is not None and reason in rows["latched"]
    assert [r["ok"] for r in rows["resets"]] == [True] and rows["after"] is None
    lines = _usr1_lines(err)
    assert len(lines) == 1, err[-800:]
    assert lines[0].startswith("[cascade-mcp] SIGUSR1 operator reset: stop cleared")
    assert f"carry: {rows['latched']}" in lines[0] and reason in lines[0]


def test_sigusr1_logs_why_the_reset_was_refused_and_the_stop_stays_latched(tmp_path, monkeypatch, capsys):
    fake = ScriptedBridge(walk_script(5), lambda i, s: "this is not json" if i == 2 else s,
                          final=state(99, box_pos=IN_BIN))
    server = _mcp_server(fake, tmp_path, monkeypatch)
    rows = {}

    def script(feed, responses, resets):
        feed(_call(1, "carry.describe_scene", {}))
        _wait(lambda: 1 in responses)
        feed(_call(2, "carry.carry_object", CARRY))
        _wait(lambda: 2 in responses)
        _sigusr1_then_list(feed, responses, resets, 3)
        rows["resets"] = list(resets)
        domain = server._runtime.domains["carry"].runtime
        rows["after"] = (domain.gate.latched, domain._session_closed, server._runtime.stopped)

    try:
        capsys.readouterr()
        _signal_worker(server, script)
        err = capsys.readouterr().err
    finally:
        server.shutdown()
    assert [r["ok"] for r in rows["resets"]] == [False]
    latched, closed, stopped = rows["after"]
    assert latched is not None and closed is not None and stopped
    lines = _usr1_lines(err)
    assert len(lines) == 1, err[-800:]
    assert lines[0].startswith("[cascade-mcp] SIGUSR1 operator reset REFUSED, the robot stays stopped: ")
    assert "carry: bridge session closed mid-segment" in lines[0] and "restarts the robot runtime" in lines[0]


def test_sigusr1_before_the_runtime_exists_logs_the_refusal(tmp_path, monkeypatch, capsys):
    fake = FakeBridge(walk_script(2))
    server = _mcp_server(fake, tmp_path, monkeypatch)
    rows = {}

    def script(feed, responses, resets):
        feed({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})  # the worker loop is running (no build)
        _wait(lambda: 1 in responses)
        _sigusr1_then_list(feed, responses, resets, 2)
        rows["resets"] = list(resets)

    try:
        capsys.readouterr()
        _signal_worker(server, script)
        err = capsys.readouterr().err
    finally:
        server.shutdown()
    assert [r["ok"] for r in rows["resets"]] == [False] and server._runtime is None and fake.segments == []
    assert _usr1_lines(err) == ["[cascade-mcp] SIGUSR1 operator reset REFUSED, the robot stays stopped: "
                                "runtime starting or input closed; reset refused"]


@pytest.mark.parametrize("result, says", [
    (None, "unreadable reset result (TypeError)"),
    ({"content": []}, "unreadable reset result (IndexError)"),
    ({"content": [{"text": "not json"}]}, "unreadable reset result (JSONDecodeError)"),
    ({"content": [{"text": "[1]"}]}, "reset answered [1]"),
    ({"content": [{"text": json.dumps({"ok": False, "domains": {"a": "x", "b": {"error": "e"}, "c": {"ok": True}}})}]},
     "b: e"),
    ({"content": [{"text": json.dumps({"ok": False})}]}, "no reason given"),
])
def test_the_operator_reset_line_never_raises_and_always_says_refused_or_cleared(result, says):
    from cascade.apps.mcp_server import _operator_reset_line

    line = _operator_reset_line(result)
    assert line == "[cascade-mcp] SIGUSR1 operator reset REFUSED, the robot stays stopped: " + says
    ok = {"content": [{"text": json.dumps({"ok": True, "domains": {"a": "x", "b": {"ok": True, "cleared": None}}})}]}
    assert _operator_reset_line(ok) == "[cascade-mcp] SIGUSR1 operator reset: stop cleared"


def test_the_reset_receipt_names_the_stop_it_cleared(tmp_path):
    fake = FakeBridge(walk_script(2))
    runtime = composed_runtime(fake, tmp_path)
    try:
        runtime.execute("emergency_stop", {})
        reset = runtime.execute("reset_stop", {}, operator=True)
        assert reset["ok"] is True and reset["domains"]["carry"] == {"ok": True, "cleared": "emergency stop requested"}
        again = runtime.execute("reset_stop", {}, operator=True)  # nothing latched: nothing cleared
        assert again["ok"] is True and again["domains"]["carry"] == {"ok": True, "cleared": None}
    finally:
        runtime.close()
