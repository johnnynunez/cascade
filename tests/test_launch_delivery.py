"""Exercise the actual launcher, without starting services or installing deps."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json
import shutil

import pytest
from conftest import start_sleeping_process

from test_demo_proof import model_http_boundary as _model_http_boundary
from test_isaac_bridge import bridge_port as _bridge_port

model_http_boundary = _model_http_boundary  # share only the real-HTTP boundary fixture
bridge_port = _bridge_port  # real TCP/client; the simulator is a test double

REPO = Path(__file__).resolve().parents[1]


def test_launch_isaac_probe_connects_the_real_wire_client(bridge_port, monkeypatch, capsys):
    import re

    # Execute the launcher's actual heredoc, not a reimplementation of its
    # probe. Only the remote simulator is doubled by the TCP fixture.
    blocks = re.findall(
        r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF",
        (REPO / "scripts/launch.sh").read_text(),
        flags=re.DOTALL,
    )
    probes = [block for block in blocks if "Isaac bridge answers:" in block]
    assert len(probes) == 1
    monkeypatch.setattr(sys, "argv", ["-", str(bridge_port[0])])

    exec(compile(probes[0], str(REPO / "scripts/launch.sh"), "exec"), {})

    assert "state_ok=True" in capsys.readouterr().out


def test_check_refuses_an_unreachable_explicit_cosmos(tmp_path):
    # Port 0 is deliberately not a model endpoint. The preflight must inspect
    # the selected brain even when other dependencies also happen to be absent.
    env = {
        **os.environ,
        "PY": sys.executable,
        "CASCADE_LAUNCH_STATE": str(tmp_path / "state"),
        "CASCADE_COSMOS_BASE_URL": "http://127.0.0.1:0/v1",
    }
    p = subprocess.run(
        ["bash", str(REPO / "scripts/launch.sh"), "--check", "--sim", "mujoco", "--brain", "cosmos"],
        env=env, text=True, capture_output=True, timeout=60,
    )
    assert "[MISSING] brain" in p.stdout + p.stderr, p.stdout + p.stderr
    assert p.returncode != 0
    assert "preflight OK" not in p.stdout


def test_check_and_dry_run_do_not_create_state(tmp_path):
    for flag in ("--check", "--dry-run"):
        state = tmp_path / flag.removeprefix("--") / "not-created"
        p = subprocess.run(
            ["bash", str(REPO / "scripts/launch.sh"), flag, "--sim", "mujoco", "--brain", "keep", "--no-open"],
            env={**os.environ, "PY": sys.executable, "CASCADE_LAUNCH_STATE": str(state)},
            text=True, capture_output=True, timeout=60,
        )
        assert not state.exists(), f"{flag} wrote state: {p.stdout} {p.stderr}"
        if flag == "--dry-run":
            assert "READY" not in p.stdout


def launcher_copy(tmp_path):
    """No test --down may scan/modify the real checkout or real chat service."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src/cascade/apps").mkdir(parents=True)
    for name in ("launch.sh", "demo_proof.py", "isaac_runtime.py"):
        shutil.copy2(REPO / "scripts" / name, repo / "scripts" / name)
    shutil.copy2(REPO / "src/cascade/apps/process_owner.py", repo / "src/cascade/apps/process_owner.py")
    return repo


@pytest.mark.parametrize("profile", ["demo-a", ""])
def test_down_stops_only_registered_verified_pids_in_selected_profile(tmp_path, profile):
    import select

    from cascade.apps import process_owner as owners

    repo = launcher_copy(tmp_path)
    root = tmp_path / "state"
    state = owners.profile_state_dir(root, profile)
    own = owners.load_owner(state, repo, profile, create=True)
    other_state = owners.profile_state_dir(root, "demo-b")
    other = owners.load_owner(other_state, repo, "demo-b", create=True)
    children = []

    def child(owner, role, register=True):
        process = subprocess.Popen(
            [sys.executable, "-c", "import os, time; os.write(1, b'R'); time.sleep(120)",
             str(repo), "cascade.apps.mcp_server", "--launch-owner", owner["owner"]],
            stdout=subprocess.PIPE,
        )
        children.append(process)
        # Popen can return during exec with /proc/<pid>/cmdline still empty.
        # Wait for Python itself, not a sleep or a relaxed ownership check.
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 10)[0], (
            f"fixture child {process.pid} did not become ready (exit={process.poll()})"
        )
        assert process.stdout.read(1) == b"R", (
            f"fixture child {process.pid} exited before readiness (exit={process.poll()})"
        )
        if register:
            owners.register_process(owner["state_dir"], owner, process.pid, role)
        return process

    try:
        a = child(own, "mcp")
        b = child(other, "mcp")
        unregistered = child(own, "mcp", register=False)
        cosmos = child(own, "cosmos")
        foreign_cosmos = child(other, "cosmos")
        stale = child(own, "occupancy_bridge")
        stale_file = state / "occupancy_bridge.pid"
        receipt = json.loads(stale_file.read_text())
        stale_file.write_text(json.dumps({**receipt, "birth": "previous-process-at-this-pid"}))
        # Bare shared legacy markers must not stop anything, even a matching PID.
        (root / "cosmos.pid").write_text(str(foreign_cosmos.pid))
        p = subprocess.run(["bash", str(repo / "scripts/launch.sh"), "--down"],
                           env={**os.environ, "PY": sys.executable, "CASCADE_LAUNCH_STATE": str(root), "CASCADE_OPENCLAW_PROFILE": profile},
                           text=True, capture_output=True, timeout=20)
        assert p.returncode == 0, p.stdout + p.stderr
        assert b.poll() is None, "down killed another profile's MCP"
        assert unregistered.poll() is None, "down swept an unregistered MCP in the same repo"
        assert foreign_cosmos.poll() is None
        assert stale.poll() is None, "down trusted a reused PID without matching birth"
        assert a.wait(timeout=5) != 0
        assert cosmos.wait(timeout=5) != 0
    finally:
        for process in children:
            try:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            finally:
                if process.stdout is not None:
                    process.stdout.close()


@pytest.mark.parametrize("version", ["6.1.0", "6.0.0"])
def test_discovered_source_python_exports_its_release_to_real_check(tmp_path, version):
    repo = launcher_copy(tmp_path)
    home = tmp_path / "home"
    release = home / "isaacsim"
    (release / "apps").mkdir(parents=True)
    (release / "apps/isaacsim.exp.full.newton.kit").write_text(f'[package]\nversion="{version}"\n')
    # This only doubles the unavailable embedded Python executable. The actual
    # isaac_runtime.py parses/validates the real test TOML; Kit is not imported.
    runner = release / "test_python.py"
    runner.write_text("import os,runpy,sys\nfrom pathlib import Path\n"
                      "Path(os.environ['SOURCE_ENV_LOG']).write_text(os.environ.get('ISAACSIM_PATH',''))\n"
                      f"sys.executable={str(release / 'kit/python/bin/python3')!r}\n"
                      "sys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
    wrapper = release / "python.sh"
    wrapper.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{runner}" "$@"\n')
    wrapper.chmod(0o755)
    env_log = tmp_path / "isaac-env"
    env = {**os.environ, "HOME": str(home), "PY": sys.executable, "SOURCE_ENV_LOG": str(env_log),
           "CASCADE_LAUNCH_STATE": str(tmp_path / "state"), "CASCADE_OPENCLAW_PROFILE": "", "CASCADE_COSMOS_BASE_URL": "http://127.0.0.1:0/v1"}
    env.pop("ISAACSIM_PATH", None)
    env.pop("ISAACSIM_PYTHON_EXE", None)
    p = subprocess.run(["bash", str(repo / "scripts/launch.sh"), "--check", "--sim", "isaac", "--brain", "cosmos"],
                       env=env, text=True, capture_output=True, timeout=60)
    assert env_log.read_text() == str(release), p.stdout + p.stderr
    if version == "6.1.0":
        assert '[ok]      Isaac Sim:' in p.stdout
        assert '"layout": "source"' in p.stdout
    else:
        assert '[MISSING] Isaac Sim:' in p.stdout
        assert 'declares' in p.stdout


@pytest.fixture
def launcher_boundary(tmp_path, model_http_boundary):
    """Full shell entrypoint; doubles only deps/runtime, model HTTP and host CLI."""
    repo = launcher_copy(tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (repo / "models").mkdir()
    (repo / "models/yoloe-11s-seg.pt").write_text("test placeholder; never loaded by a model")
    py = bindir / "python"
    py.write_text(f'#!{sys.executable}\n' + '''import os,subprocess,sys
args=sys.argv[1:]
if args and args[0]=='-':
    source=sys.stdin.read()
    if 'need = {' in source and 'find_spec' in source:
        print('')  # dependency boundary: never install into any real venv
        sys.exit(0)
    if 'build_runtime' in source:
        print('[launch] runtime builds: ORCHESTRATION TEST DOUBLE, no physics')
        sys.exit(0)
    sys.exit(subprocess.run([os.environ['REAL_PY'],*args],input=source,text=True).returncode)
if args==['-c','import torch']:
    sys.exit(0)
os.execv(os.environ['REAL_PY'],[os.environ['REAL_PY'],*args])
''')
    py.chmod(0o755)
    oc = bindir / "openclaw"
    oc.write_text(f'#!{sys.executable}\n' + '''import json,os,sys,signal,subprocess,select
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['HOST_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')
if args[:1]==['--profile']: args=args[2:]
cfg=Path(os.environ['HOST_CFG'])
if args==['--version']: print('2026.9.3')
elif args[:2]==['models','status']: print(json.dumps({'resolvedDefault':'local/cosmos3-edge'}))
elif args[:1]==['health']: print('{"ok":true}')
elif args==['gateway','status','--json']:
    print(json.dumps({'service':{'runtime':{'pid':int(Path(os.environ['GATEWAY_TEST_PID']).read_text())}}}))
elif args==['gateway','restart'] and os.environ.get('GATEWAY_TEST_PID'):
    pidfile=Path(os.environ['GATEWAY_TEST_PID'])
    os.kill(int(pidfile.read_text()),signal.SIGTERM)
    child=subprocess.Popen([sys.executable,'-c',"import os,time; os.write(1,b'R'); time.sleep(120)",'openclaw-test-gateway'],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    try:
        assert select.select([child.stdout],[],[],10)[0] and child.stdout.read(1)==b'R'
    except BaseException:
        child.kill(); child.wait(); raise
    finally:
        child.stdout.close()
    pidfile.write_text(str(child.pid))
elif args==['gateway','stop','--force']:
    os.kill(int(Path(os.environ['GATEWAY_TEST_PID']).read_text()),signal.SIGTERM)
elif args[:2]==['gateway','stop']:
    raise SystemExit('fixture requires explicit --force for managed gateway stop')
elif args[:2]==['mcp','probe']:
    print(json.dumps({'tools':[os.environ['CASCADE_MCP_NAME']+'__'+s for s in ['pick_and_place','get_observation','analyze_scene','world_state','reset_scene']]}))
elif args[:2]==['mcp','set']: Path(os.environ['MCP_CONFIG']).write_text(args[3])
elif args[:1]==['onboard']:
    cfg.write_text(json.dumps({'models':{'providers':{'local':{'models':[{'id':args[args.index('--custom-model-id')+1]}]}}}}))
elif args==['config','file']: print(cfg)
elif args[:3]==['config','set','models.providers']:
    cfg.write_text(json.dumps({'models':{'providers':json.loads(args[3])}}))
elif args[:1]==['agent']:
    print(json.dumps({'status':'ok','result':{'payloads':[{'text':'OK'}],'meta':{'agentMeta':{'provider':'local','model':'cosmos3-edge','sessionId':args[args.index('--session-id')+1]}}}}))
''')
    oc.chmod(0o755)
    curl = bindir / "curl"
    curl.write_text('#!/bin/bash\nexit 1\n')  # make hardcoded discovery probes fail deterministically
    curl.chmod(0o755)
    h = model_http_boundary
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "PY": str(py), "REAL_PY": sys.executable,
           "CASCADE_OPENCLAW_PROFILE": "isolated-test", "CASCADE_LAUNCH_STATE": str(tmp_path / "state"),
           "CASCADE_COSMOS_BASE_URL": h["base_url"], "CASCADE_QWEN_BASE_URL": "http://127.0.0.1:0/v1",
           "OPENCLAW_GATEWAY_PORT": h["base_url"].split(":")[-1].split("/")[0], "CASCADE_MCP_NAME": "robot-test",
           "HOST_LOG": str(tmp_path / "host.jsonl"), "HOST_CFG": str(tmp_path / "host-config.json"), "MCP_CONFIG": str(tmp_path / "mcp.json")}
    command = ["bash", str(repo / "scripts/launch.sh"), "--sim", "none", "--arm", "mock", "--cameras", "mock", "--brain", "auto",
               "--occupancy", "none", "--graspgenx", "none", "--no-open", "--no-robot-turn", "--no-judge"]
    return {"repo": repo, "env": env, "command": command, "http": h}


def test_launch_and_check_share_brain_resolution_and_custom_endpoint(launcher_boundary):
    h = launcher_boundary
    check = subprocess.run([*h["command"], "--check"], env=h["env"], capture_output=True, text=True, timeout=60)
    assert check.returncode == 0, check.stdout + check.stderr
    assert h["http"]["base_url"] in check.stdout
    launch = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60)
    assert launch.returncode == 0, launch.stdout + launch.stderr
    commands = [json.loads(line) for line in Path(h["env"]["HOST_LOG"]).read_text().splitlines()]
    onboard = [c for c in commands if "onboard" in c]
    assert len(onboard) == 1, launch.stdout + launch.stderr
    assert onboard[0][onboard[0].index("--custom-base-url") + 1] == h["http"]["base_url"]
    config = json.loads(Path(h["env"]["MCP_CONFIG"]).read_text())
    assert "--launch-owner" in config["args"]
    state = Path(h["env"]["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"
    assert config["args"][config["args"].index("--launch-state-dir") + 1] == str(state)
    assert config["env"]["CASCADE_OPENCLAW_PROFILE"] == "isolated-test"
    assert not (state / "gateway.started").exists(), "borrowed gateway acquired ownership"
    assert 'STARTED (UNVERIFIED' in launch.stdout
    assert '[launch] READY' not in launch.stdout


def test_owned_gateway_receipt_follows_our_restart_and_down_is_profile_scoped(launcher_boundary, tmp_path):
    from cascade.apps import process_owner as owners

    h = launcher_boundary
    state = owners.profile_state_dir(h["env"]["CASCADE_LAUNCH_STATE"], "isolated-test")
    owner = owners.load_owner(state, h["repo"], "isolated-test", create=True)
    initial = start_sleeping_process("openclaw-test-gateway")
    pidfile = tmp_path / "gateway.pid"
    pidfile.write_text(str(initial.pid))
    h["env"]["GATEWAY_TEST_PID"] = str(pidfile)
    try:
        owners.register_process(state, owner, initial.pid, "gateway")
        launch = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60)
        assert launch.returncode == 0, launch.stdout + launch.stderr
        receipt = json.loads((state / "gateway.started").read_text())
        assert receipt["pid"] == int(pidfile.read_text()), "our gateway restart left a stale ownership receipt"
        assert receipt["pid"] != initial.pid
        down = subprocess.run(["bash", str(h["repo"] / "scripts/launch.sh"), "--down"], env=h["env"], capture_output=True, text=True, timeout=30)
        assert down.returncode == 0, down.stdout + down.stderr
        assert not owners.is_live(receipt, owner)
        assert owners.process_identity(receipt["pid"]) is None, "managed stop left the fake gateway running"
        commands = [json.loads(line) for line in Path(h["env"]["HOST_LOG"]).read_text().splitlines()]
        stops = [c for c in commands if c[2:4] == ["gateway", "stop"]]
        assert stops == [["--profile", "isolated-test", "gateway", "stop", "--force"]]
    finally:
        if initial.poll() is None:
            initial.terminate()
        initial.wait(timeout=5)
        import signal
        try:
            os.kill(int(pidfile.read_text()), signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_standalone_proof_uses_same_profile_state_root_as_launcher(launcher_boundary):
    h = launcher_boundary
    p = subprocess.run([sys.executable, str(h["repo"] / "scripts/demo_proof.py"), "--run", "--repo", str(h["repo"]),
                        "--sim", "none", "--no-robot-turn"], env=h["env"], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stdout + p.stderr
    state = Path(h["env"]["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"
    assert (state / "proof.json").exists(), "proof default ignored the launch state/profile contract"
    assert json.loads((state / "proof.json").read_text())["profile"] == "isolated-test"
