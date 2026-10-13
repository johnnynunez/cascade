"""Hardware launcher contracts without ROS, camera drivers or an OpenClaw install."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    scripts = str(REPO / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


launcher = load_script("franka_openclaw")
companion = load_script("franka_camera_companion")


def test_configuration_uses_one_shared_http_runtime_and_vision(tmp_path):
    config, ports = launcher.configuration(tmp_path, "m" * 32, "g" * 32,
                                          context_window=16384, model_base="http://127.0.0.1:8082/v1")
    server = config["mcp"]["servers"]["cascade"]
    assert server["transport"] == "streamable-http"
    assert server["url"] == "http://127.0.0.1:18740/mcp"
    assert "command" not in server and "env" not in server
    assert server["requestTimeoutMs"] >= 300000
    assert server["supportsParallelToolCalls"] is False
    assert config["tools"]["toolSearch"]["enabled"] is False
    assert "cascade__emergency_stop" in config["tools"]["allow"]
    assert "cascade__pick_and_place" in config["tools"]["allow"]
    assert "cascade__rotate_joint" not in config["tools"]["allow"]
    assert config["cron"]["enabled"] is False
    assert config["agents"]["defaults"]["heartbeat"]["every"] == "0m"
    provider = config["models"]["providers"]["local-cosmos"]
    assert provider["models"][0]["input"] == ["text", "image"]
    assert provider["models"][0]["contextWindow"] == 16384
    assert config["gateway"]["bind"] == "loopback"
    assert ports == {"gateway": 18790, "mcp": 18740, "stream": 8090, "camera": 8091}


def test_complete_mcp_catalog_is_an_explicit_host_option(tmp_path):
    config, _ = launcher.configuration(tmp_path, "m" * 32, "g" * 32, context_window=32768,
        model_base="http://127.0.0.1:8082/v1", full_tool_catalog=True)
    assert config["tools"]["allow"] == ["cascade__*"]
    assert "toolFilter" not in config["mcp"]["servers"]["cascade"]


def test_additional_host_template_joins_shared_runtime_without_starting_one():
    template = json.loads((REPO / "configs/openclaw/franka-mcp.json").read_text())
    server = template["mcpServers"]["cascade"]
    assert server == {
        "type": "http", "url": "http://127.0.0.1:18740/mcp",
        "headers": {"Authorization": "Bearer ${CASCADE_MCP_TOKEN}"},
    }
    # Selecting the optional hardware registration must not change the default
    # checkout registration used by existing Isaac installations.
    default = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["cascade"]
    assert default["type"] == "stdio"
    assert default["env"]["CASCADE_ARM"] == "isaac"


@pytest.mark.parametrize("url", ["https://remote.example/v1", "http://127.0.0.1:8082/private",
                                 "http://user:pass@127.0.0.1:8082/v1"])
def test_model_config_cannot_redirect_camera_data(tmp_path, url):
    with pytest.raises(ValueError):
        launcher.configuration(tmp_path, "m" * 32, "g" * 32,
                               context_window=16384, model_base=url)


def test_configure_private_repeatable_and_preserves_operator_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "REPO", REPO)
    args = SimpleNamespace(context_window=32768, model_base="http://127.0.0.1:8082/v1", port_offset=0, mock=False)
    launcher.configure(args, tmp_path)
    token = (tmp_path / "mcp.token").read_text()
    assert len(token.strip()) >= 24
    for file in (tmp_path / "mcp.token", tmp_path / "gateway.token", tmp_path / "openclaw/openclaw.json"):
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
    prompt = tmp_path / "workspace/AGENTS.md"
    prompt.write_text(prompt.read_text() + "\nOperator addition: prefer short answers.\n")
    launcher.configure(args, tmp_path)
    assert (tmp_path / "mcp.token").read_text() == token
    assert "Operator addition" in prompt.read_text()
    assert len(list((tmp_path / "openclaw").glob("*.backup.json"))) == 1


def test_runtime_environment_latches_stop_without_injecting_holding(tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_START_HOLDING", "cork")
    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", "/personal/config")
    monkeypatch.setenv("ROS_DOMAIN_ID", "31")
    monkeypatch.setenv("PYTHONPATH", "/some/ros/python")
    env = launcher.environment(tmp_path, {"ports": {"stream": 8090}, "mock": False}, sys.executable)
    assert env["CASCADE_START_STOPPED"] == "1"
    assert "CASCADE_START_HOLDING" not in env
    assert env["ROS_DOMAIN_ID"] == "31"
    assert env["OPENCLAW_CONFIG_PATH"] == str(tmp_path / "openclaw/openclaw.json")
    assert env["PYTHONPATH"] == str(REPO / "src") + ":/some/ros/python"


@pytest.mark.parametrize("mock", [False, True])
def test_dedicated_launcher_removes_conflicting_robot_selectors(tmp_path, monkeypatch, mock):
    for key in ("CASCADE_ARMS", "CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_GROUNDING_LLM_PROFILE"):
        monkeypatch.setenv(key, "unrelated-rig")
    env = launcher.environment(tmp_path, {"ports": {"stream": 8090}, "mock": mock}, sys.executable)
    assert not {"CASCADE_ARMS", "CASCADE_ROBOT", "CASCADE_BASE", "CASCADE_GROUNDING_LLM_PROFILE"}.intersection(env)
    assert env["CASCADE_FRANKA_MOCK"] == ("1" if mock else "0")


@pytest.mark.parametrize("mock", [False, True])
def test_direct_wrapper_cannot_inherit_another_embodiment(tmp_path, mock):
    # Execute the actual shell wrapper with an inert interpreter stand-in.
    # It records the selected profiles; no Cascade, ROS or camera is imported.
    recorder = tmp_path / "record-interpreter"
    recorder.write_text(f"#!{sys.executable}\nimport json,os\n"
                       "print(json.dumps({k:v for k,v in os.environ.items() if k.startswith('CASCADE_')}))\n")
    recorder.chmod(0o700)
    setup = tmp_path / "inert-setup.bash"
    setup.write_text("export CASCADE_ARMS=wrong-setup-arm CASCADE_ROBOT=wrong-setup-robot CASCADE_BASE=wrong-setup-base\n")
    env = dict(os.environ, PY=str(recorder), CASCADE_FRANKA_MOCK="1" if mock else "0",
               CASCADE_ARM="unrelated-arm", CASCADE_ARMS="other-arm,second-arm",
               CASCADE_ROBOT="other-robot", CASCADE_BASE="other-base",
               CASCADE_GROUNDING_LLM_PROFILE="unrelated-model",
               CASCADE_ROS_SETUP=str(setup), CASCADE_FRANKA_SETUP=str(setup))
    result = subprocess.run(["bash", str(REPO / "scripts/run_franka_http_mcp.sh")],
                            env=env, check=True, capture_output=True, text=True, timeout=5)
    selected = json.loads(result.stdout)
    assert selected["CASCADE_ARM"] == ("fr3_mock" if mock else "fr3")
    assert selected["CASCADE_CAMERAS"] == ("mock" if mock else "d435i")
    assert selected.get("CASCADE_GROUNDING_LLM_PROFILE") == (None if mock else "local_cosmos")
    assert not {"CASCADE_ARMS", "CASCADE_ROBOT", "CASCADE_BASE"}.intersection(selected)
    assert selected["CASCADE_START_STOPPED"] == "1"


@pytest.mark.parametrize("wrapper", ["bringup_franka.sh", "run_franka_mcp.sh"])
def test_legacy_wrappers_accept_installed_ros_and_python_paths(tmp_path, wrapper):
    # Both entrypoints run against inert setup scripts and a command recorder;
    # this checks their real shell behavior without importing a robot driver.
    recorder = tmp_path / "ros2"
    recorder.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                       "keys = ('CASCADE_TEST_ROS_SETUP','CASCADE_TEST_FRANKA_SETUP',"
                       "'ROS_DOMAIN_ID','PYTHONPATH','CASCADE_ARM','CASCADE_ARMS',"
                       "'CASCADE_ROBOT','CASCADE_BASE')\n"
                       "print(json.dumps({'argv':sys.argv[1:], 'env':"
                       "{k:os.environ[k] for k in keys if k in os.environ}}))\n")
    recorder.chmod(0o700)
    ros_setup = tmp_path / "ros setup.bash"
    ros_setup.write_text('export CASCADE_TEST_ROS_SETUP=loaded\n')
    franka_setup = tmp_path / "franka setup.bash"
    franka_setup.write_text('export CASCADE_TEST_FRANKA_SETUP=loaded\n')
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ.get("PATH", ""),
               PY=str(recorder), CASCADE_ROS_SETUP=str(ros_setup),
               CASCADE_FRANKA_SETUP=str(franka_setup), ROS_DOMAIN_ID="47",
               CASCADE_ARMS="other,arm", CASCADE_ROBOT="other-robot", CASCADE_BASE="other-base")
    result = subprocess.run(["bash", str(REPO / "scripts" / wrapper), "test-argument"],
                            env=env, check=True, capture_output=True, text=True, timeout=5)
    recorded = json.loads(result.stdout)
    selected = recorded["env"]
    assert selected["CASCADE_TEST_ROS_SETUP"] == "loaded"
    assert selected["CASCADE_TEST_FRANKA_SETUP"] == "loaded"
    assert selected["ROS_DOMAIN_ID"] == "47"
    assert recorded["argv"][-1] == "test-argument"
    if wrapper == "bringup_franka.sh":
        assert recorded["argv"][:2] == ["launch", str(REPO / "scripts/franka_stack.launch.py")]
    else:
        assert recorded["argv"][:2] == ["-m", "cascade.apps.mcp_server"]
        assert selected["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "src")
        assert selected["CASCADE_ARM"] == "fr3"
        assert not {"CASCADE_ARMS", "CASCADE_ROBOT", "CASCADE_BASE"}.intersection(selected)


@pytest.mark.skipif(sys.platform == "win32", reason="launcher uses private POSIX process groups")
def test_failed_start_cleanup_stops_respawn_after_gateway_leader_exits(tmp_path):
    marker = tmp_path / "worker.json"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json,os,pathlib,signal,socket,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'port':listener.getsockname()[1]}))\n"
        "time.sleep(60)\n")
    parent = tmp_path / "parent.py"
    parent.write_text("import subprocess,sys\nsubprocess.Popen([sys.executable,*sys.argv[1:]])\n")
    child = launcher.spawn([sys.executable, str(parent), str(worker), str(marker)],
                           tmp_path, "gateway_child", os.environ.copy(), None)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                 start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        state = json.loads(marker.read_text())
        child.wait(timeout=5)  # real gateway wrappers may exit before their child
        assert launcher.port_open(state["port"])
        launcher.terminate_spawned(child, timeout=.15)
        deadline = time.monotonic() + 3
        while launcher.port_open(state["port"]) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not launcher.port_open(state["port"])
        assert unrelated.poll() is None
    finally:
        launcher.terminate_spawned(child, timeout=.05)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_failed_start_cleanup_refuses_reused_process_identity(monkeypatch):
    child = SimpleNamespace(pid=424242, _cascade_birth="original")
    monkeypatch.setattr(launcher, "process_identity", lambda _: {"birth": "replacement"})
    signalled = []
    monkeypatch.setattr(launcher.os, "killpg", lambda *args: signalled.append(args))
    with pytest.raises(RuntimeError, match="identity changed"):
        launcher.terminate_spawned(child)
    assert not signalled


def test_browser_uses_dedicated_profile_and_upstream_camera_extension(tmp_path):
    url = "http://127.0.0.1:18790/#token=private"
    command = launcher.browser_command("chromium", tmp_path, url)
    assert command[-1] == url
    assert "--user-data-dir=" + str(tmp_path / "browser") in command
    assert "--load-extension=" + str(REPO / "extensions/chrome") in command
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_native_model_probe_refuses_served_context_overflow(tmp_path, monkeypatch):
    config, _ = launcher.configuration(tmp_path, "m" * 32, "g" * 32,
                                       context_window=32768, model_base="http://127.0.0.1:8082/v1")
    (tmp_path / "openclaw").mkdir()
    (tmp_path / "openclaw/openclaw.json").write_text(json.dumps(config))
    called = []
    monkeypatch.setitem(sys.modules, "demo_proof", SimpleNamespace(
        request_json=lambda _: {"data": [{"id": "cosmos3-edge", "max_model_len": 16384}]},
        probe_native_tools=lambda *args: called.append(args)))
    with pytest.raises(RuntimeError, match="exceeds"):
        launcher.probe_model(tmp_path)
    assert not called


def test_readiness_requires_camera_and_latched_stop_and_resume(tmp_path, monkeypatch):
    views = iter([
        {"cameras": {"d435i": {"fps": 15, "error": None}}, "stopped": False, "resume_supported": True},
        {"cameras": {"d435i": {"fps": 15, "error": "disconnected"}}, "stopped": True, "resume_supported": True},
        {"cameras": {"d435i": {"fps": 15, "error": None}}, "stopped": True, "resume_supported": True},
    ])
    monkeypatch.setattr(launcher, "get_json", lambda _: next(views))
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    result = launcher.wait_runtime(tmp_path, {"ports": {"stream": 8090}}, SimpleNamespace(poll=lambda: None))
    assert result["stopped"] is True
    assert result["cameras"]["d435i"]["error"] is None


@contextmanager
def http_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_camera_companion_reuses_stream_and_is_read_only():
    requests = []
    jpeg = b"\xff\xd8real-unaltered-jpeg\xff\xd9"

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path == "/state":
                body = json.dumps({"cameras": {"d435i": {"fps": 15, "frame_id": 77,
                    "error": None, "depth": True}}, "active_command": "private user text",
                    "objects": [{"label": "box"}]}).encode()
                kind = "application/json"
            elif self.path == "/snapshot/d435i.jpg":
                body, kind = jpeg, "image/jpeg"
            elif self.path == "/stream/d435i":
                body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                kind = "multipart/x-mixed-replace; boundary=frame"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with http_server(Upstream) as (_, upstream), http_server(companion.CameraHandler) as (proxy, origin):
        proxy.source = companion.CameraSource(upstream)
        with urlopen(origin + "/api/status") as response:
            state = json.load(response)
        assert set(state) == {"cameras", "camera_discovery"}
        assert state["cameras"]["d435i"]["frame_id"] == 77
        assert state["cameras"]["d435i"]["online"] is True
        assert state["camera_discovery"]["cameras"] == [{"name": "d435i", "stream_url": "/stream/d435i"}]
        with urlopen(origin + "/snapshot/d435i.jpg") as response:
            assert response.read() == jpeg
        with urlopen(origin + "/stream/d435i") as response:
            assert jpeg in response.read()
        before = len(requests)
        for request in (Request(origin + "/resume", data=b"{}"), Request(origin + "/task", data=b"{}"),
                        Request(origin + "/cancel", data=b"{}")):
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == 405
        for path in ("/cancel", "/analyze", "/stream/../state", "/state?x=1"):
            with pytest.raises(HTTPError) as error:
                urlopen(origin + path)
            assert error.value.code == 404
        assert len(requests) == before


def test_camera_outage_returns_reconnect_status():
    with http_server(companion.CameraHandler) as (proxy, origin):
        proxy.source = SimpleNamespace(state=lambda: (_ for _ in ()).throw(OSError("disconnected")))
        with pytest.raises(HTTPError) as error:
            urlopen(origin + "/api/status")
        assert error.value.code == 503
        assert json.load(error.value) == {"error": "Camera is reconnecting"}


def test_camera_source_cannot_become_arbitrary_network_proxy():
    for url in ("http://192.168.1.11:80", "https://127.0.0.1:8090", "http://127.0.0.1:8090/admin",
                "http://127.0.0.1:8090?target=private", "http://token@localhost:8090"):
        with pytest.raises(ValueError):
            companion.CameraSource(url)
