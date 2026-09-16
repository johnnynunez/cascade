"""Real private process/socket tests; OpenClaw itself is a local CLI double."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from cascade.apps import process_owner as owners


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("test_private_gateway", ROOT / "scripts/openclaw_gateway.py")
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


@pytest.fixture
def private_gateway(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    state = repo / "runs/.launch/profile-cascade-demo"
    home = tmp_path / "empty-home"
    home.mkdir()
    owner = owners.load_owner(state, repo, "cascade-demo", create=True)
    cli = repo / ".openclaw-cli/bin/openclaw"
    cli.parent.mkdir(parents=True)
    cli.write_text(f"#!{sys.executable}\n" + '''import json,os,socket,subprocess,sys,time
from pathlib import Path
args=sys.argv[1:]
with Path(os.environ['CLI_CALLS']).open('a') as out: out.write(json.dumps(args)+'\\n')
if args[:1]==['--profile']: args=args[2:]
if args[:2]==['gateway','run']:
    if os.environ.get('FAIL_GATEWAY')=='1': raise SystemExit(23)
    if os.environ.get('PRIVATE_WORKER_PID'):
        worker=subprocess.Popen([sys.executable,'-c',
            'import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write(1,b"R");time.sleep(60)'],stdout=subprocess.PIPE)
        assert worker.stdout.read(1)==b'R'
        worker.stdout.close()
        Path(os.environ['PRIVATE_WORKER_PID']).write_text(str(worker.pid))
    Path(os.environ['CHILD_ENV']).write_text(json.dumps({key:os.environ.get(key) for key in
        ('HOME','OPENCLAW_STATE_DIR','OPENCLAW_CONFIG_PATH','NODE_COMPILE_CACHE')}))
    server=socket.socket();server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server.bind(('127.0.0.1',int(args[args.index('--port')+1])));server.listen()
    while True:
        connection,_=server.accept();connection.close()
elif args[:1]==['health']:
    assert os.environ['OPENCLAW_STATE_DIR'].endswith('/profile-cascade-demo/openclaw')
    assert os.environ['OPENCLAW_CONFIG_PATH']==os.environ['OPENCLAW_STATE_DIR']+'/openclaw.json'
    try:
        with socket.create_connection(('127.0.0.1',int(os.environ['TEST_PORT'])),timeout=.1): pass
    except OSError: raise SystemExit(1)
    print('{"ok":true}')
else: raise SystemExit('unexpected CLI/service-manager operation: '+repr(args))
''')
    cli.chmod(0o755)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(tmp_path / "personal-config.json"))
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "personal-state"))
    monkeypatch.setenv("CLI_CALLS", str(tmp_path / "calls.jsonl"))
    monkeypatch.setenv("CHILD_ENV", str(tmp_path / "child-env.json"))
    monkeypatch.setenv("TEST_PORT", str(port))
    result = {"repo": repo, "state": state, "owner": owner, "home": home, "port": port,
              "calls": tmp_path / "calls.jsonl", "child_env": tmp_path / "child-env.json"}
    yield result
    for record in owners.live_records(state, owner, role="gateway_child"):
        owners.stop_private_gateway(record, owner)
    for record in owners.records(state):
        if record.get("role") == "gateway_child":
            try:
                os.waitpid(record["pid"], 0)
            except ChildProcessError:
                pass


def test_empty_home_uses_project_state_and_no_service_manager(private_gateway):
    h = private_gateway
    gateway.prepare(h["state"], h["owner"], h["port"])
    record = gateway.start(h["repo"], h["state"], h["owner"], h["port"], wait_seconds=3)
    assert owners.is_live(record, h["owner"])
    assert record["process_group"] == record["pid"]
    env = json.loads(h["child_env"].read_text())
    assert env["HOME"] == str(h["home"])
    assert env["OPENCLAW_STATE_DIR"] == str(h["state"] / "openclaw")
    assert env["OPENCLAW_CONFIG_PATH"] == str(h["state"] / "openclaw/openclaw.json")
    assert not (h["home"] / ".config/systemd").exists()
    calls = [json.loads(line)[2:] for line in h["calls"].read_text().splitlines()]
    assert all(args[:2] == ["gateway", "run"] or args[:1] == ["health"] for args in calls)
    gateway.prepare(h["state"], h["owner"], h["port"])
    assert not owners.is_live(record, h["owner"])
    assert not gateway.port_open(h["port"])


def test_unowned_port_is_refused_before_launcher_configuration(private_gateway, monkeypatch):
    h = private_gateway
    foreign = socket.socket()
    foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    foreign.bind(("127.0.0.1", h["port"]))
    foreign.listen()
    try:
        with pytest.raises(RuntimeError, match="unowned"):
            gateway.prepare(h["state"], h["owner"], h["port"])
        text = (ROOT / "scripts/launch.sh").read_text()
        block = text.split("GATEWAY_OWNED=0\n", 1)[1].split("# register the MCP server", 1)[0]
        prelude = '''set -euo pipefail
GATEWAY_OWNED=0
ownerctl() { return 1; }
run() { "$@"; }
die() { echo "$*" >&2; exit 1; }
log() { :; }
warn() { :; }
oc() { echo forbidden-config-write >> "$CONFIG_CALLS"; }
'''
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PY": sys.executable,
               "REPO": str(ROOT), "STATE_DIR": str(h["state"]), "CASCADE_INSTALL_PROFILE": "spark",
               "CASCADE_OPENCLAW_PROFILE": "cascade-demo", "GATEWAY_PORT": str(h["port"]), "DRY": "0",
               "CONFIG_CALLS": str(h["repo"] / "config-calls")}
        # The real helper requires the exact source owner, not the fixture's
        # unrelated repo. Rebind this isolated receipt for the shell test.
        h["state"].joinpath("owner.json").unlink()
        h["owner"] = owners.load_owner(h["state"], ROOT, "cascade-demo", create=True)
        result = subprocess.run(["bash", "-c", prelude + block], env=env, text=True, capture_output=True, timeout=10)
        assert result.returncode != 0 and "unowned" in result.stderr
        assert not Path(env["CONFIG_CALLS"]).exists()
        assert gateway.port_open(h["port"])
    finally:
        foreign.close()


def test_gateway_start_failure_leaves_no_owned_service(private_gateway, monkeypatch):
    h = private_gateway
    monkeypatch.setenv("FAIL_GATEWAY", "1")
    with pytest.raises(RuntimeError, match="healthy|exited"):
        gateway.start(h["repo"], h["state"], h["owner"], h["port"], wait_seconds=.3)
    assert not owners.live_records(h["state"], h["owner"], role="gateway_child")
    assert not gateway.port_open(h["port"])


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_start_failure_preserves_error_when_exited_child_needs_reaping(private_gateway, monkeypatch, platform):
    """Darwin rejects signals to a group containing only an unreaped zombie."""
    h = private_gateway
    child = SimpleNamespace(pid=424242, reaped=False)

    def reap(*args, **kwargs):
        child.reaped = True
        return 23

    child.poll = child.wait = reap
    monkeypatch.setattr(gateway, "sys", SimpleNamespace(platform=platform), raising=False)
    monkeypatch.setattr(gateway.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(gateway.signal, "signal", lambda *args: gateway.signal.SIG_DFL)

    def unhealthy(*args, **kwargs):
        raise RuntimeError("gateway did not become healthy")

    def group_signal(pid, number):
        assert pid == child.pid
        if child.reaped:
            raise ProcessLookupError("empty process group")
        if platform == "darwin":
            raise PermissionError("zombie-only process group")

    monkeypatch.setattr(gateway, "wait_gateway", unhealthy)
    monkeypatch.setattr(gateway.os, "killpg", group_signal)
    with pytest.raises(RuntimeError, match="did not become healthy"):
        gateway.start(h["repo"], h["state"], h["owner"], h["port"])
    assert child.reaped


@pytest.mark.parametrize("platform,worker,denied,own_child", [
    ("darwin", False, False, True),
    ("darwin", True, False, True),
    ("darwin", True, False, False),
    ("darwin", True, True, True),
    ("linux", False, False, True),
    ("linux", True, False, True),
    ("linux", True, True, True),
])
def test_private_stop_handles_zombie_boundary_without_hiding_denial(monkeypatch, platform, worker, denied, own_child):
    import signal

    record = {"pid": 424242, "role": "gateway_child", "birth": "known", "command": "known"}
    state = {"alive": True, "reaped": False, "worker": worker}
    signals, reaps = [], []
    monkeypatch.setattr(owners, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(owners, "is_live", lambda *args: state["alive"])
    monkeypatch.setattr(owners, "process_identity", lambda pid: None)

    def waitpid(pid, flags):
        assert pid == record["pid"] and flags == os.WNOHANG
        reaps.append(pid)
        if not own_child:
            raise ChildProcessError("created by another launcher process")
        state["reaped"] = True
        return pid, 0

    def group_signal(pid, number):
        assert pid == record["pid"]
        signals.append(number)
        if number == signal.SIGTERM:
            state["alive"] = False
        elif denied:
            raise PermissionError("live worker signal denied")
        elif state["worker"]:
            state["worker"] = False
        elif platform == "darwin" and not state["reaped"]:
            raise PermissionError("zombie-only process group")
        else:
            raise ProcessLookupError("empty process group")

    monkeypatch.setattr(owners.os, "waitpid", waitpid)
    monkeypatch.setattr(owners.os, "killpg", group_signal)
    if denied:
        with pytest.raises(PermissionError, match="live worker signal denied"):
            owners.stop_private_gateway(record, {})
    else:
        owners.stop_private_gateway(record, {})
        assert not state["worker"]
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    if platform == "linux":
        assert not reaps


def test_owned_group_cleanup_reaps_a_worker_that_ignores_term(private_gateway, monkeypatch):
    h = private_gateway
    worker_pid = h["repo"] / "worker.pid"
    monkeypatch.setenv("PRIVATE_WORKER_PID", str(worker_pid))
    record = gateway.start(h["repo"], h["state"], h["owner"], h["port"], wait_seconds=3)
    worker = int(worker_pid.read_text())
    assert owners.process_identity(worker) is not None
    assert os.getpgid(worker) == record["process_group"]
    owners.stop_owned(h["state"], h["owner"])
    deadline = time.monotonic() + 3
    while owners.process_identity(worker) is not None and time.monotonic() < deadline:
        time.sleep(.02)
    assert owners.process_identity(worker) is None


def test_scoped_down_leaves_an_unrelated_process_running(private_gateway):
    h = private_gateway
    other = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    try:
        record = gateway.start(h["repo"], h["state"], h["owner"], h["port"], wait_seconds=3)
        assert owners.stop_owned(h["state"], h["owner"]) == [record["pid"]]
        assert other.poll() is None
        gateway.wait_closed(h["port"])
        assert not gateway.port_open(h["port"])
    finally:
        other.terminate()
        other.wait(timeout=5)


@pytest.mark.parametrize("ending,code", [("exit 23", 23), ("kill -TERM $$", 143)])
def test_actual_launcher_failure_trap_cleans_its_foreground_gateway(private_gateway, ending, code):
    h = private_gateway
    scripts = h["repo"] / "scripts"
    scripts.mkdir()
    for name in ("openclaw_gateway.py", "demo_proof.py"):
        (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    source = (ROOT / "scripts/launch.sh").read_text()
    first = source.split("GATEWAY_OWNED=0\n", 1)[1].split("# register the MCP server", 1)[0]
    marker = 'if [[ "${CASCADE_INSTALL_PROFILE:-}" == spark ]]; then\n    log "starting the checkout\'s foreground gateway'
    last = marker + source.split(marker, 1)[1].split("run oc config validate", 1)[0]
    prelude = '''set -euo pipefail
GATEWAY_OWNED=0
ownerctl() { return 1; }
run() { "$@"; }
die() { echo "$*" >&2; exit 1; }
log() { :; }
warn() { echo "$*" >&2; }
oc() { :; }
'''
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PY": sys.executable,
           "REPO": str(h["repo"]), "STATE_DIR": str(h["state"]), "CASCADE_INSTALL_PROFILE": "spark",
           "CASCADE_OPENCLAW_PROFILE": "cascade-demo", "GATEWAY_PORT": str(h["port"]), "DRY": "0", "MCP_NAME": "cascade"}
    result = subprocess.run(["bash", "-c", prelude + first + last + "\n" + ending],
                            env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == code, result.stdout + result.stderr
    assert not owners.live_records(h["state"], h["owner"], role="gateway_child")
    assert not gateway.port_open(h["port"])
