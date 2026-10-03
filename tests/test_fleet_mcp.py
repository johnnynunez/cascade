"""Real stdio fleet routing and cancellation; synthetic domains only."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from cascade.apps.fleet_mcp import FleetMcpServer
from cascade.robotics.fleet import FleetRuntime
from test_fleet_runtime import member


REPO = Path(__file__).resolve().parents[1]


def call(ident, name, arguments=None):
    return {"jsonrpc": "2.0", "id": ident, "method": "tools/call",
            "params": {"name": name, "arguments": {} if arguments is None else arguments}}


def unpack(response):
    assert "error" not in response, response
    return json.loads(response["result"]["content"][-1]["text"])


def move(server, robot, ident, **arguments):
    return call(ident, "fleet.execute", {"robot_id": robot, "episode_id": server.episodes[robot],
        "tool": "locomotion.move", "arguments": {"distance": .1, **arguments}})


@pytest.fixture
def fixture(tmp_path):
    pairs = {name: member(name, tmp_path) for name in ("first", "second")}
    fleet = FleetRuntime({name: value[0] for name, value in pairs.items()})
    output = queue.Queue()
    servers = []
    def create(**kwargs):
        server = FleetMcpServer(fleet, output.put, max_workers=kwargs.pop("max_workers", 2), **kwargs)
        servers.append(server)
        return server
    yield create, pairs, output
    for _, domain in pairs.values():
        domain.release.set()
    for server in servers:
        assert server.close()["complete"]
    if not servers:
        fleet.close()


def test_workers_reach_independent_domains_and_stop_only_selected_robot(fixture):
    create, pairs, output = fixture
    server = create()
    for _, domain in pairs.values():
        domain.hold = True
    server.accept(move(server, "first", 1))
    server.accept(move(server, "second", 2))
    assert all(domain.entered.wait(2) for _, domain in pairs.values())
    server.accept(call(3, "fleet.stop", {"robot_id": "first"}))
    received = {row["id"]: row for row in (output.get(timeout=2), output.get(timeout=2))}
    assert set(received) == {1, 3}
    assert not unpack(received[1])["ok"]
    assert set(unpack(received[3])["robots"]) == {"first"}
    assert not pairs["second"][1].stopped.is_set()
    server.accept(call(4, "fleet.stop"))
    rows = {row["id"]: row for row in (output.get(timeout=2), output.get(timeout=2))}
    assert set(rows) == {2, 4}
    assert not unpack(rows[2])["ok"]
    assert set(unpack(rows[4])["robots"]) == {"first", "second"}


def test_capacity_rejection_and_cancelled_queue_never_dispatch(fixture):
    create, pairs, output = fixture
    server = create(max_workers=1, max_pending=2)
    first = pairs["first"][1]
    first.hold = True
    server.accept(move(server, "first", 1))
    assert first.entered.wait(2)
    server.accept(move(server, "second", 2))
    server.accept(move(server, "second", 3))
    rejected = output.get(timeout=2)
    assert rejected["id"] == 3 and "capacity" in unpack(rejected)["error"]
    server.accept({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}})
    first.release.set()
    rows = {row["id"]: row for row in (output.get(timeout=2), output.get(timeout=2))}
    assert not unpack(rows[2])["ok"]
    assert pairs["second"][1].calls == []
    assert not first.stopped.is_set()


def test_reset_received_before_newer_stop_cannot_clear_it(fixture):
    create, pairs, output = fixture
    server = create(max_workers=1, max_pending=3)
    first = pairs["first"][1]
    first.hold = True
    server.accept(move(server, "first", 1))
    assert first.entered.wait(2)
    server.accept(call(2, "fleet.reset_stop", {"robot_id": "second"}))
    server.accept(call(3, "fleet.stop", {"robot_id": "second"}))
    assert output.get(timeout=2)["id"] == 3
    first.release.set()
    rows = {row["id"]: row for row in (output.get(timeout=2), output.get(timeout=2))}
    assert not unpack(rows[2])["ok"]
    assert "stale reset generation" in unpack(rows[2])["result"]["error"]
    assert "robot stopped" in pairs["second"][0].unverified_actions()


def test_original_deadline_stops_running_request_and_invalidates_result(fixture):
    create, pairs, output = fixture
    server = create(timeout_s=.1)
    pairs["first"][1].hold = True
    server.accept(move(server, "first", 1))
    assert pairs["first"][1].entered.wait(2)
    result = unpack(output.get(timeout=2))
    assert not result["ok"] and "deadline" in result["error"]
    assert pairs["first"][1].stopped.is_set()
    assert not pairs["second"][1].stopped.is_set()


def test_wrong_episode_and_hidden_reset_have_no_dispatch(fixture, monkeypatch):
    monkeypatch.setenv("CASCADE_HIDE_TOOLS", "reset_stop,fleet.stop")
    create, pairs, output = fixture
    server = create()
    request = move(server, "first", 1)
    request["params"]["arguments"]["episode_id"] = server.episodes["second"]
    server.accept(request)
    assert "episode mismatch" in unpack(output.get(timeout=2))["error"]
    assert not pairs["first"][1].calls
    server.accept(call(2, "fleet.reset_stop", {"robot_id": "first"}))
    assert "hidden" in unpack(output.get(timeout=2))["error"]
    server.accept(call(3, "fleet.stop"))
    assert unpack(output.get(timeout=2))["ok"]
    server.accept({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    names = {row["name"] for row in output.get(timeout=2)["result"]["tools"]}
    assert "fleet.stop" in names and "fleet.reset_stop" not in names


def test_duplicate_inflight_id_cannot_enqueue_second_action(fixture):
    create, pairs, output = fixture
    server = create()
    pairs["first"][1].hold = True
    server.accept(move(server, "first", 1))
    assert pairs["first"][1].entered.wait(2)
    server.accept(move(server, "second", 1))
    assert "already in flight" in output.get(timeout=2)["error"]["message"]
    assert not pairs["second"][1].calls
    pairs["first"][1].release.set()
    assert output.get(timeout=2)["id"] == 1


def test_old_timeout_snapshot_cannot_cancel_reused_id_on_another_robot(fixture, monkeypatch):
    create, pairs, output = fixture
    server = create(timeout_s=.08)
    for _, domain in pairs.values():
        domain.hold = True
    captured, resume, finished = threading.Event(), threading.Event(), threading.Event()
    original = server._cancel_records
    def cancel(*args, **kwargs):
        paused = threading.current_thread() is server.monitor and bool(args[0]) and not captured.is_set()
        if paused:
            captured.set()
            assert resume.wait(3)
        original(*args, **kwargs)
        if paused:
            finished.set()
    monkeypatch.setattr(server, "_cancel_records", cancel)
    try:
        server.accept(move(server, "first", 7))
        assert pairs["first"][1].entered.wait(2) and captured.wait(2)
        pairs["first"][1].release.set()
        assert not unpack(output.get(timeout=2))["ok"]
        deadline = time.monotonic() + 2
        while 7 in server._pending and time.monotonic() < deadline:
            time.sleep(.001)
        assert 7 not in server._pending
        server.timeout_s = 2
        server.accept(move(server, "second", 7))
        assert pairs["second"][1].entered.wait(2)
        resume.set()
        assert finished.wait(2)
        assert not pairs["second"][1].stopped.is_set()
        pairs["second"][1].release.set()
        assert unpack(output.get(timeout=2))["ok"]
    finally:
        resume.set()


@pytest.mark.parametrize("fault", ["stop_reset", "deadline"])
def test_validation_cannot_renew_received_authority_or_deadline(fixture, monkeypatch, fault):
    create, pairs, output = fixture
    server = create(timeout_s=.03 if fault == "deadline" else 2.)
    original = server._operation
    def operation(*args):
        result = original(*args)
        if fault == "deadline":
            time.sleep(.04)
        else:
            assert server.fleet.stop("first")["ok"]
            assert server.fleet.reset_stop("first")["ok"]
        return result
    monkeypatch.setattr(server, "_operation", operation)
    server.accept(move(server, "first", 1))
    assert not unpack(output.get(timeout=2))["ok"]
    assert not pairs["first"][1].calls


def test_stuck_stop_rpc_cannot_delay_later_robot_deadline(tmp_path, monkeypatch):
    pairs = {name: member(name, tmp_path) for name in ("first", "second", "unrelated")}
    fleet = FleetRuntime({name: value[0] for name, value in pairs.items()})
    entered, release = threading.Event(), threading.Event()
    original = pairs["first"][1].stop
    def blocked_stop():
        entered.set()
        assert release.wait(3)
        return original()
    monkeypatch.setattr(pairs["first"][1], "stop", blocked_stop)
    for _, domain in pairs.values():
        domain.hold = True
    output = queue.Queue()
    server = FleetMcpServer(fleet, output.put, max_workers=2, timeout_s=.08)
    try:
        server.accept(move(server, "first", 1))
        assert entered.wait(2)
        server.accept(move(server, "second", 2))
        assert pairs["second"][1].entered.wait(1)
        assert pairs["second"][1].stopped.wait(.3), "later deadline waited behind unrelated blocked stop RPC"
        assert not release.is_set()
        assert pairs["second"][0].cancellation_token > 0
        assert pairs["unrelated"][0].cancellation_token == 0
        assert not pairs["unrelated"][1].stopped.is_set()
    finally:
        release.set()
        for _, domain in pairs.values():
            domain.release.set()
        assert server.close()["complete"]


def test_reset_deadline_is_enforced_by_owner_before_and_after_domain_io(fixture, monkeypatch):
    create, pairs, _ = fixture
    server = create()
    runtime, domain = pairs["first"]
    calls = []
    monkeypatch.setattr(domain, "reset_stop", lambda: calls.append("reset") or {"ok": True})
    expired = server.fleet.reset_stop("first", deadline_monotonic_s=time.monotonic() - 1)
    assert not expired["ok"] and not calls
    assert server.fleet.stop("first")["ok"]
    entered, release = threading.Event(), threading.Event()
    def reset():
        entered.set()
        assert release.wait(2)
        return {"ok": True}
    monkeypatch.setattr(domain, "reset_stop", reset)
    result = {}
    deadline = time.monotonic() + .04
    worker = threading.Thread(target=lambda: result.update(server.fleet.reset_stop(
        "first", expected_generation=runtime.cancellation_token, deadline_monotonic_s=deadline)))
    worker.start()
    try:
        assert entered.wait(1)
        time.sleep(max(0., deadline - time.monotonic()) + .005)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not result["ok"]
    assert "robot stopped" in runtime.unverified_actions()


class Client:
    def __init__(self, tmp_path, *, count=2, blocked=False):
        self.directory = tmp_path
        profile = tmp_path / "fleet.yaml"
        profile.write_text(yaml.safe_dump({"version": 1, "max_workers": count, "deadline_s": 5,
            "robots": [{"profile": "microduck_conversation_mock", "mock_id": f"duck{i:02}"}
                       for i in range(count)]}))
        env = {key: value for key, value in os.environ.items() if not key.startswith("CASCADE_")}
        env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="-1")
        prefix = [sys.executable, "-m", "cascade.apps.fleet_mcp"]
        if blocked:
            bootstrap = '''
import os, pathlib, time
from cascade.apps import fleet_mcp
from cascade.skills.mobile_runtime import MobileSkillRuntime
original = MobileSkillRuntime.execute
def execute(self, name, args=None):
    if name == "walk_velocity" and self.cfg.bases[0]["robot_id"] == "duck00":
        root = pathlib.Path(os.environ["TEST_BLOCK_DIR"])
        (root / "entered").write_text("blocked before dispatch")
        deadline = time.monotonic() + 10
        while not (root / "release").exists() and time.monotonic() < deadline:
            time.sleep(.005)
    return original(self, name, args)
MobileSkillRuntime.execute = execute
raise SystemExit(fleet_mcp.main())
'''
            env["TEST_BLOCK_DIR"] = str(tmp_path)
            prefix = [sys.executable, "-c", bootstrap]
        self.proc = subprocess.Popen([*prefix, "--fleet", str(profile), "--run-dir", str(tmp_path / "run")],
            cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self.queue, self.errors = queue.Queue(), []
        self.threads = [threading.Thread(target=self.read, daemon=True),
            threading.Thread(target=lambda: self.errors.extend(self.proc.stderr), daemon=True)]
        for worker in self.threads:
            worker.start()

    def read(self):
        for line in self.proc.stdout:
            self.queue.put(json.loads(line))
        self.queue.put(None)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def receive(self):
        result = self.queue.get(timeout=10)
        assert result is not None, "".join(self.errors)
        return result

    def close(self):
        (self.directory / "release").touch()
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
            raise
        finally:
            for worker in self.threads:
                worker.join(2)
            self.proc.stdout.close()
            self.proc.stderr.close()


def execute(catalog, robot, ident, tool="locomotion.walk_velocity", arguments=None):
    return call(ident, "fleet.execute", {"robot_id": robot, "episode_id": catalog[robot]["episode_id"],
        "tool": tool, "arguments": arguments or {"vx": .03, "vy": 0, "wz": 0, "duration_s": .06}})


def test_twelve_robot_stdio_catalog_dispatch_and_private_traces(tmp_path):
    client = Client(tmp_path, count=12)
    try:
        client.send({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
        assert client.receive()["result"]["serverInfo"]["name"] == "cascade-fleet"
        client.send(call(1, "fleet.catalog"))
        info = unpack(client.receive())
        assert not info["physical_fleet_admission"]
        catalog = info["robots"]
        assert len(catalog) == len({row["episode_id"] for row in catalog.values()}) == 12
        for index, robot in enumerate(catalog):
            client.send(execute(catalog, robot, index + 2))
        results = [unpack(client.receive()) for _ in range(12)]
        assert {row["robot_id"] for row in results} == set(catalog)
        assert all(row["result"]["execution_ok"] and not row["ok"] for row in results)
        assert all(row["result"]["postcondition"]["status"] == "unverified" for row in results)
        client.send(execute(catalog, "duck00", 50, tool="locomotion.task_memory", arguments={"new_task": True}))
        assert "history cannot be reset" in unpack(client.receive())["error"]
        client.send(execute(catalog, "duck00", 51, tool="task_done", arguments={"success": True, "summary": "mock"}))
        assert not unpack(client.receive())["result"]["success"]
    finally:
        client.close()
    assert client.proc.returncode == 0, "".join(client.errors)
    assert len(list((tmp_path / "run/robots").glob("*/trace.jsonl"))) == 12
    assert json.loads((tmp_path / "run/mcp-report.json").read_text())["shutdown"]["complete"]


@pytest.mark.parametrize("kind", ["stop", "cancel"])
def test_stdio_stop_and_cancel_bypass_blocked_robot_and_preserve_peer(tmp_path, kind):
    client = Client(tmp_path, blocked=True)
    try:
        client.send(call(1, "fleet.catalog"))
        catalog = unpack(client.receive())["robots"]
        client.send(execute(catalog, "duck00", 2))
        deadline = time.monotonic() + 5
        while not (tmp_path / "entered").exists() and time.monotonic() < deadline:
            time.sleep(.005)
        assert (tmp_path / "entered").exists()
        if kind == "stop":
            client.send(call(3, "fleet.stop", {"robot_id": "duck00"}))
            stopped = client.receive()
            assert stopped["id"] == 3 and set(unpack(stopped)["robots"]) == {"duck00"}
        else:
            client.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}})
        client.send(execute(catalog, "duck01", 4))
        peer = client.receive()
        assert peer["id"] == 4 and unpack(peer)["result"]["execution_ok"]
        (tmp_path / "release").touch()
        cancelled = client.receive()
        assert cancelled["id"] == 2 and not unpack(cancelled)["ok"]
    finally:
        client.close()
    assert client.proc.returncode == 0, "".join(client.errors)


def test_signal_with_stdin_open_closes_owned_fleet(tmp_path):
    client = Client(tmp_path)
    try:
        client.send(call(1, "fleet.catalog"))
        assert unpack(client.receive())["ok"]
        client.proc.send_signal(signal.SIGTERM)
        client.proc.wait(timeout=10)
        assert client.proc.returncode == 128 + signal.SIGTERM
    finally:
        client.close()
    report = json.loads((tmp_path / "run/mcp-report.json").read_text())
    assert report["shutdown"]["complete"] and not report["physical_fleet_admission"]
