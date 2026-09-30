"""Always-on services are private, bounded and owned by one Spark checkout."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/spark_public.py"
spec = importlib.util.spec_from_file_location("spark_public", SCRIPT)
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


def test_user_units_restore_install_home_and_supervise_only_their_stack(tmp_path):
    settings = {"runtime_home": str(tmp_path / "private home")}
    units = public.units(tmp_path, settings)
    assert set(units) == {"paai-spark-demo", "paai-spark-visitor", "paai-spark-ngrok"}
    for contents in units.values():
        assert "KillMode=mixed" in contents
        assert "WantedBy=default.target" in contents
        assert 'Environment="HOME=' + settings["runtime_home"] + '"' in contents
        assert "sudo" not in contents and "8090" not in contents
        assert "gateway.token" not in contents
    assert "8093" in units["paai-spark-visitor"]
    assert "--auth-file" in units["paai-spark-visitor"]
    assert public.VISITOR_ORIGIN == "http://127.0.0.1:8093"


def unit_directives(contents):
    return dict(line.split("=", 1) for line in contents.splitlines() if "=" in line and not line.startswith("#"))


def test_demo_unit_never_relaunches_a_stack_that_failed_to_reach_ready(tmp_path):
    demo = unit_directives(public.units(tmp_path, {"runtime_home": str(tmp_path)})["paai-spark-demo"])
    assert demo["Restart"] == "on-failure"
    assert demo["RestartPreventExitStatus"] == str(public.LAUNCH_FAILED)
    assert int(demo["StartLimitBurst"]) <= 3 and int(demo["StartLimitIntervalSec"]) >= 3600
    for name in ("paai-spark-visitor", "paai-spark-ngrok"):
        light = unit_directives(public.units(tmp_path, {"runtime_home": str(tmp_path)})[name])
        assert light["Restart"] == "always" and light["StartLimitIntervalSec"] == "0"


def test_serve_reports_failed_launch_with_the_non_restarting_status(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(public, "browser_module", lambda repo: SimpleNamespace(ready=lambda repo: True))
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "launch" in command else 0)
    monkeypatch.setattr(public.subprocess, "run", fake_run)
    assert public.serve(tmp_path) == public.LAUNCH_FAILED
    assert calls[-1] == [str(tmp_path / "run.sh"), "down"]


def test_serve_health_failure_after_ready_is_a_restartable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "browser_module", lambda repo: SimpleNamespace(ready=lambda repo: False))
    monkeypatch.setattr(public.subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(public.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="health check failed"):
        public.serve(tmp_path)


def _user_manager_available():
    if shutil.which("systemd-run") is None or shutil.which("systemctl") is None:
        return False
    return subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True,
                          timeout=10).stdout.strip() in ("running", "degraded")


@pytest.mark.skipif(not _user_manager_available(), reason="no systemd user manager")
def test_real_user_manager_honours_the_demo_restart_policy(tmp_path):
    """Run the demo unit's own restart directives on a real transient unit."""
    demo = unit_directives(public.units(tmp_path, {"runtime_home": str(tmp_path)})["paai-spark-demo"])
    counter = tmp_path / "starts"
    for status, expected_starts in ((public.LAUNCH_FAILED, 1), (1, 2)):
        counter.write_text("")
        unit = f"paai-restart-policy-test-{os.getpid()}-{status}"
        properties = [f"Restart={demo['Restart']}", f"RestartPreventExitStatus={demo['RestartPreventExitStatus']}",
                      "RestartSec=1", "StartLimitIntervalSec=60", "StartLimitBurst=2"]
        command = ["systemd-run", "--user", "--quiet", "--collect", f"--unit={unit}",
                   *[item for p in properties for item in ("-p", p)],
                   "/bin/sh", "-c", f"echo x >> '{counter}'; exit {status}"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                state = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "--value"],
                                       capture_output=True, text=True, timeout=10).stdout.strip()
                if state in ("failed", "inactive") and len(counter.read_text().split()) >= expected_starts:
                    time.sleep(2.5)            # a restart would land within RestartSec
                    break
                time.sleep(.2)
            assert len(counter.read_text().split()) == expected_starts, (status, counter.read_text())
        finally:
            subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True, timeout=30)
            subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True, timeout=30)


def test_existing_unrelated_service_is_never_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "unit_directory", lambda: tmp_path)
    path = tmp_path / "paai-spark-demo.service"
    path.write_text("[Service]\nExecStart=/other/application\n")
    monkeypatch.setattr(public, "run", lambda *a, **kw: f"LoadState=loaded\nActiveState=active\nFragmentPath={path}\nDropInPaths=\n")
    with pytest.raises(ValueError, match="another installation"):
        public.owned_units(tmp_path / "checkout")
    assert path.read_text() == "[Service]\nExecStart=/other/application\n"


def test_service_overrides_require_owner_review(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "run", lambda *a, **kw: "LoadState=loaded\nDropInPaths=/another/unit.conf\n")
    with pytest.raises(ValueError, match="overrides"):
        public.owned_units(tmp_path)


def test_secrets_require_private_owner_files_and_never_enter_failed_command_output(tmp_path, monkeypatch):
    path = tmp_path / "token"
    path.write_text("private-token-fixture")
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        public.private_file(path)
    path.chmod(0o600)
    assert public.private_file(path) == path
    monkeypatch.setattr(public.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="private-token-fixture", stderr="private-token-fixture"))
    with pytest.raises(RuntimeError) as failure:
        public.run(["ngrok", "config", "check"])
    assert "private-token-fixture" not in str(failure.value)


def test_private_json_creates_and_atomically_replaces_real_private_files(tmp_path):
    path = tmp_path / "credentials.json"
    public.private_json(path, {"password": "first-fixture-password"})
    assert json.loads(path.read_text()) == {"password": "first-fixture-password"}
    assert path.stat().st_mode & 0o777 == 0o600
    with path.open() as previous:
        public.private_json(path, {"password": "second-fixture-password"})
        assert json.load(previous) == {"password": "first-fixture-password"}
    assert json.loads(path.read_text()) == {"password": "second-fixture-password"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("link_location", ["destination", "temporary"])
def test_private_json_refuses_symlinks_without_touching_the_target(tmp_path, link_location):
    victim = tmp_path / "unrelated.json"
    victim.write_text("leave unchanged")
    path = tmp_path / "credentials.json"
    link = path if link_location == "destination" else path.with_suffix(".tmp")
    link.symlink_to(victim)
    with pytest.raises((ValueError, OSError)):
        public.private_json(path, {"password": "private-fixture"})
    assert victim.read_text() == "leave unchanged"
    assert link.is_symlink()


@pytest.fixture
def public_enable(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    auth_file, token_file, executable = (tmp_path / name for name in ("auth.json", "token", "ngrok"))
    auth_file.write_text(json.dumps({"username": "visitor", "password": "fixture-password"}))
    token_file.write_text("fixture-ngrok-token")
    executable.write_text("#!/bin/sh\nexit 0\n")
    auth_file.chmod(0o600)
    token_file.chmod(0o600)
    executable.chmod(0o700)
    unit_directory = tmp_path / "user-units"
    monkeypatch.setattr(public, "unit_directory", lambda: unit_directory)
    monkeypatch.setattr(public, "runtime_home", lambda root: str(tmp_path))
    commands = []
    fixture = SimpleNamespace(repo=repo, unit_directory=unit_directory, commands=commands,
        before="inactive", after="inactive", reset_error=False, reloaded=False,
        args=SimpleNamespace(domain="visitor.example.invalid", auth_file=auth_file,
            ngrok=executable, token_file=token_file, ngrok_config=None))
    def external(command, **kwargs):
        commands.append(command)
        if command[0] == "loginctl":
            return "yes\n"
        if command[:3] == ["systemctl", "--user", "show"]:
            if "--value" in command:
                assert fixture.reloaded, "Failure state must be read after daemon-reload"
                return fixture.after + "\n"
            path = unit_directory / command[3]
            if path.exists():
                return f"LoadState=loaded\nActiveState={fixture.before}\nFragmentPath={path}\nDropInPaths=\n"
            return "LoadState=not-found\nActiveState=inactive\nFragmentPath=\nDropInPaths=\n"
        if command == ["systemctl", "--user", "daemon-reload"]:
            fixture.reloaded = True
        if command[:3] == ["systemctl", "--user", "reset-failed"]:
            assert fixture.after == "failed", "Fresh or inactive units must not be reset"
            if fixture.reset_error:
                raise RuntimeError("Required reset failed")
        return ""
    monkeypatch.setattr(public, "run", external)
    return fixture


def test_enable_writes_real_private_configuration_and_units(public_enable):
    fixture = public_enable
    repo, unit_directory, commands = fixture.repo, fixture.unit_directory, fixture.commands
    result = public.enable(repo, fixture.args)
    assert result["enabled"] is True
    private = repo / "runs/.install/public"
    for name in ("auth.json", "ngrok.json", "settings.json"):
        assert (private / name).stat().st_mode & 0o777 == 0o600
        assert isinstance(json.loads((private / name).read_text()), dict)
    assert json.loads((private / "ngrok.json").read_text())["agent"]["authtoken"] == "fixture-ngrok-token"
    assert {path.stem for path in unit_directory.glob("*.service")} == set(public.SERVICES)
    assert all("fixture-password" not in path.read_text() and "fixture-ngrok-token" not in path.read_text()
               for path in unit_directory.glob("*.service"))
    assert not any(command[2:3] == ["reset-failed"] for command in commands)
    assert ["systemctl", "--user", "enable", "--now", *[name + ".service" for name in public.SERVICES]] in commands


@pytest.mark.parametrize("before,after", [
    ("inactive", "inactive"), ("active", "active"), ("failed", "inactive"),
    ("failed", "failed"), ("active", "failed"),
])
def test_matching_enable_resets_only_current_failed_or_start_limited_unit(public_enable, before, after):
    fixture = public_enable
    fixture.before, fixture.after = before, after
    fixture.unit_directory.mkdir()
    for name, contents in public.units(fixture.repo, {"runtime_home": str(fixture.repo.parent)}).items():
        (fixture.unit_directory / (name + ".service")).write_text(contents)
    assert public.enable(fixture.repo, fixture.args)["enabled"]
    reset = ["systemctl", "--user", "reset-failed", "paai-spark-demo.service"]
    enable = ["systemctl", "--user", "enable", "--now", *[name + ".service" for name in public.SERVICES]]
    assert (reset in fixture.commands) == (after == "failed")
    if after == "failed":
        assert fixture.commands.index(reset) < fixture.commands.index(enable)


def test_required_failure_reset_error_prevents_starting_any_service(public_enable):
    fixture = public_enable
    fixture.after, fixture.reset_error = "failed", True
    with pytest.raises(RuntimeError, match="Required reset failed"):
        public.enable(fixture.repo, fixture.args)
    assert ["systemctl", "--user", "reset-failed", "paai-spark-demo.service"] in fixture.commands
    assert not any(command[2:3] in (["enable"], ["restart"]) for command in fixture.commands)


def test_disable_checks_ownership_then_stops_only_new_units(tmp_path, monkeypatch):
    owned = {name: {"installed": True, "active": True} for name in public.SERVICES}
    calls = []
    monkeypatch.setattr(public, "owned_units", lambda repo: owned)
    monkeypatch.setattr(public, "run", lambda command, **kw: calls.append(command))
    public.disable(tmp_path)
    assert calls == [["systemctl", "--user", "disable", "--now", "paai-spark-ngrok.service", "paai-spark-visitor.service", "paai-spark-demo.service"]]


def test_systemd_argument_escaping_never_interprets_percent_or_dollars():
    assert public.unit_arg('/tmp/a $name 100%') == '"/tmp/a $$name 100%%"'
    with pytest.raises(ValueError):
        public.unit_arg("/tmp/name\nExecStart=/other")


def test_systemd_path_directive_is_absolute_without_argument_quotes():
    assert public.unit_path("/tmp/checkout with spaces/100%") == "/tmp/checkout with spaces/100%%"
    for invalid in ("relative/path", "/tmp/line\nbreak", "/tmp/line\rbreak", "/tmp/nul\0"):
        with pytest.raises(ValueError):
            public.unit_path(invalid)


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze is unavailable")
def test_generated_user_units_pass_real_systemd_validation_with_spaces_and_percent(tmp_path):
    repo = tmp_path / "checkout with spaces 100%"
    repo.mkdir()
    generated = tmp_path / "unit-validation"
    generated.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    paths = []
    for name, contents in public.units(repo, {"runtime_home": str(tmp_path / "private home 100%")}).items():
        path = generated / (name + ".service")
        path.write_text(contents)
        paths.append(str(path))
    # This parses temporary files only. It neither installs nor starts a unit.
    result = subprocess.run([shutil.which("systemd-analyze"), "--user", "--man=no", "verify", *paths],
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
                                 "XDG_RUNTIME_DIR": str(runtime)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def public_check(tmp_path, monkeypatch):
    directory = public.private_root(tmp_path)
    (directory / "settings.json").write_text(json.dumps({"domain": "visitor.example.invalid"}))
    (directory / "auth.json").write_text(json.dumps({"username": "visitor", "password": "fixture-password"}))
    gateway = tmp_path / "runs/.launch/profile-cascade-demo/openclaw/openclaw.json"
    gateway.parent.mkdir(parents=True)
    gateway.write_text(json.dumps({"gateway": {"auth": {"token": "private-token-fixture"}}}))
    monkeypatch.setattr(public, "owned_units", lambda repo: {
        name: {"installed": True, "active": True} for name in public.SERVICES})
    monkeypatch.setattr(public, "browser_module", lambda repo: SimpleNamespace(ready=lambda repo: True))
    monkeypatch.setattr(public.time, "sleep", lambda seconds: None)

    def verify(chat_body):
        frame = 0
        requested = []
        def response(url, authorization=None):
            nonlocal frame
            path = public.urlsplit(url).path
            requested.append(path)
            if authorization is None:
                return 401, b"Authentication required"
            if path == "/api/chat":
                return 200, chat_body
            if path in ("/", "/visitor.js"):
                return 200, b"Visitor"
            if path == "/api/status":
                frame += 1
                return 200, json.dumps({"cameras": [{"name": name, "online": True, "frame_id": frame}
                    for name in ("kitchen", "worktop", "side")]}).encode()
            if path.startswith("/snapshot/"):
                return 200, b"\xff\xd8fixture\xff\xd9"
            return 404, b"Not found"
        monkeypatch.setattr(public, "public_get", response)
        result = public.check(tmp_path)
        assert {"/guide", "/guide/", "/openclaw", "/openclaw/"} <= set(requested)
        return result
    return verify


@pytest.mark.parametrize("chat", [
    {}, {"enabled": False, "ready": True, "agent": "cascade-demo"},
    {"enabled": True, "ready": False, "agent": "cascade-demo"},
    {"enabled": True, "ready": True, "agent": "personal"},
    {"enabled": "true", "ready": True, "agent": "cascade-demo"},
    {"enabled": True, "ready": 1, "agent": "cascade-demo"}, [],
])
def test_public_ready_rejects_wrong_or_unready_chat_even_with_http_200(public_check, chat):
    with pytest.raises(RuntimeError, match="attendee chat is not READY"):
        public_check(json.dumps(chat).encode())


def test_public_ready_rejects_invalid_chat_json_even_with_http_200(public_check):
    with pytest.raises(RuntimeError, match="attendee chat is not READY"):
        public_check(b"<html>Wrong application</html>")


def test_public_ready_requires_the_connected_attendee_agent(public_check):
    result = public_check(json.dumps({"enabled": True, "ready": True, "agent": "cascade-demo"}).encode())
    assert result["healthy"] is True
    assert result["attendee_chat_ready"] is True
    assert result["cameras_advancing"] is True


def test_service_recovery_stops_its_owned_demo_after_three_unready_chat_checks(tmp_path, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(public, "browser_module", lambda repo: SimpleNamespace(ready=lambda repo: False))
    monkeypatch.setattr(public.signal, "signal", lambda *args: None)
    monkeypatch.setattr(public.time, "sleep", sleeps.append)
    monkeypatch.setattr(public.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="restarting the owned stack"):
        public.serve(tmp_path)
    assert sleeps == [10, 10, 10]
    assert len(calls) == 2
    assert calls[0][1:3] == [str(tmp_path / "scripts/desktop.py"), "launch"]
    assert calls[1] == [str(tmp_path / "run.sh"), "down"]


@pytest.fixture
def visitor_process(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    repo.mkdir()
    directory = public.private_root(repo)
    public.private_json(directory / "settings.json", {"domain": "visitor.example.invalid", "ngrok": "/fixture/ngrok"})
    public.private_json(directory / "ngrok.json", {"agent": {"authtoken": "fixture-ngrok-token"}})
    public.private_json(directory / "auth.json", {"username": "visitor", "password": "fixture-password"})
    units = tmp_path / "units"
    units.mkdir()
    unit = units / "paai-spark-visitor.service"
    unit.write_text(f"# PAAI Spark checkout: {repo}\n[Service]\n")
    proc = tmp_path / "proc"
    process = proc / "12345"
    (process / "fd").mkdir(parents=True)
    (process / "net").mkdir()
    fields = ["S"] + ["0"] * 18 + ["98765"] + ["0"] * 4
    (process / "stat").write_text("12345 (visitor) " + " ".join(fields))
    (process / "cmdline").write_bytes(b"\0".join(part.encode() for part in public.visitor_command(repo)) + b"\0")
    (process / "cwd").symlink_to(repo)
    (process / "fd/3").symlink_to("socket:[111]")
    (process / "fd/4").symlink_to("socket:[222]")
    listener = "0: 0100007F:1F9D 00000000:0000 0A 0:0 00:0 0 1000 0 111"
    connected = "1: 0100007F:1F9D 0100007F:B26E 01 0:0 00:0 0 1000 0 222"
    (process / "net/tcp").write_text("header\n" + listener + "\n" + connected + "\n")
    monkeypatch.setattr(public, "PROC", proc)
    monkeypatch.setattr(public, "unit_directory", lambda: units)
    monkeypatch.setattr(public, "run", lambda *args, **kwargs:
        f"MainPID=12345\nLoadState=loaded\nActiveState=active\nFragmentPath={unit}\nDropInPaths=\n")
    return repo, process, (12345, "98765", "111")


def test_visitor_guard_proves_process_command_cwd_and_listener_inode(visitor_process):
    repo, process, identity = visitor_process
    assert public.visitor_owner(repo) == identity
    (process / "cmdline").write_bytes(b"/usr/bin/python3\0/another/server.py\0")
    with pytest.raises(RuntimeError, match="does not own"):
        public.visitor_owner(repo)


def test_foreign_port_never_receives_credentials_or_starts_ngrok(visitor_process, monkeypatch):
    repo, process, _ = visitor_process
    (process / "fd/3").unlink()  # Port 8093 remains in the table, owned elsewhere.
    monkeypatch.setattr(public, "owned_visitor_get", lambda *a, **kw: pytest.fail("Foreign listener was contacted"))
    monkeypatch.setattr(public.subprocess, "Popen", lambda *a, **kw: pytest.fail("A foreign listener was exposed"))
    with pytest.raises(RuntimeError, match="does not own"):
        public.ngrok_service(repo)


def test_authentication_waits_for_the_accepted_connection_owner(visitor_process, monkeypatch):
    repo, process, identity = visitor_process
    (process / "fd/4").unlink()  # Correct listener, but this connection belongs elsewhere.
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def getsockname(self): return ("127.0.0.1", 45678)
    monkeypatch.setattr(public.socket, "create_connection", lambda *a, **kw: Connection())
    ticks = iter((0, 3))
    monkeypatch.setattr(public.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(public.http.client, "HTTPConnection", lambda *a, **kw: pytest.fail("Credentials reached an unowned connection"))
    with pytest.raises(RuntimeError, match="accepted connection"):
        public.owned_visitor_get(repo, identity, "/api/chat", "Basic fixture-private")


def test_owned_connection_can_authenticate_without_reconnecting(visitor_process, monkeypatch):
    repo, _, identity = visitor_process
    requests = []
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def getsockname(self): return ("127.0.0.1", 45678)
    class Client:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, headers):
            with pytest.raises(RuntimeError, match="verified visitor connection ended"):
                self.connect()
            requests.append((method, path, headers))
        def getresponse(self): return SimpleNamespace(status=200, read=lambda limit: b'{"agent":"cascade-demo"}')
        def close(self): pass
    monkeypatch.setattr(public.socket, "create_connection", lambda *a, **kw: Connection())
    monkeypatch.setattr(public.http.client, "HTTPConnection", Client)
    assert public.owned_visitor_get(repo, identity, "/api/chat", "Basic fixture-private")[0] == 200
    assert requests == [("GET", "/api/chat", {"Authorization": "Basic fixture-private"})]


@pytest.mark.parametrize("anonymous_status,chat", [
    (200, {"enabled": True, "ready": True, "agent": "cascade-demo"}),
    (401, {"enabled": True, "ready": False, "agent": "cascade-demo"}),
    (401, {"enabled": True, "ready": True, "agent": "personal"}),
])
def test_auth_or_attendee_mismatch_never_starts_ngrok(visitor_process, monkeypatch, anonymous_status, chat):
    repo, _, _ = visitor_process
    monkeypatch.setattr(public, "owned_visitor_get", lambda repo, owner, path, authorization=None:
        (anonymous_status, b"entry") if authorization is None else (200, json.dumps(chat).encode()))
    monkeypatch.setattr(public.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unverified visitor was exposed"))
    with pytest.raises(RuntimeError):
        public.ngrok_service(repo)


def test_valid_owned_visitor_starts_ngrok_and_owner_loss_stops_that_child(visitor_process, monkeypatch):
    import io
    repo, process, _ = visitor_process
    monkeypatch.setattr(public, "owned_visitor_get", lambda repo, owner, path, authorization=None:
        (401, b"entry") if authorization is None else
        (200, b'{"enabled":true,"ready":true,"agent":"cascade-demo"}'))
    monkeypatch.setattr(public.signal, "signal", lambda *args: None)
    commands = []
    class Child:
        def __init__(self):
            self.stdout = io.StringIO("")
            self.returncode = None
            self.terminated = False
        def poll(self): return self.returncode
        def terminate(self):
            self.terminated = True
            self.returncode = 0
        def wait(self, **kwargs): return self.returncode
    child = Child()
    monkeypatch.setattr(public.subprocess, "Popen", lambda command, **kwargs: commands.append(command) or child)
    def ownership_loss(seconds):
        assert seconds == 1
        (process / "fd/3").unlink()
    monkeypatch.setattr(public.time, "sleep", ownership_loss)
    with pytest.raises(RuntimeError, match="does not own"):
        public.ngrok_service(repo)
    assert len(commands) == 1 and commands[0][:3] == ["/fixture/ngrok", "http", public.VISITOR_ORIGIN]
    assert child.terminated and child.stdout.closed


@pytest.mark.skipif(not Path("/proc/self/net/tcp").exists(), reason="Linux proc socket tables are unavailable")
def test_socket_inode_reader_matches_a_real_ephemeral_loopback_listener():
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        sockets, rows = public.visitor_sockets(os.getpid())
        address = "0100007F:" + format(listener.getsockname()[1], "04X")
        assert any(row[1] == address and row[3] == "0A" and row[9] in sockets for row in rows)
