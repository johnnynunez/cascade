"""Ownership, browser isolation and read-only camera HTTP boundaries; no GPU."""
from __future__ import annotations

import base64
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socketserver
import sys
import threading
import time
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snap_uses_owned_profile_and_real_user_bus_without_mutating_runtime(tmp_path, monkeypatch):
    browser = load("spark_browser")
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/snap/bin/chromium" if name == "chromium" else None)
    actual = tmp_path / "real-user"
    monkeypatch.setattr(browser.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(actual)))
    monkeypatch.setattr(browser.Path, "exists", lambda path: str(path).endswith("/bus"))
    incoming = {"HOME": str(tmp_path / "private-home"), "DISPLAY": ":98",
                "DBUS_SESSION_BUS_ADDRESS": "unix:path=/private/session-bus"}
    command, env, data, extension = browser.browser_plan(tmp_path / "checkout", incoming)
    assert env["HOME"] == str(actual)
    assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path=/run/user/{os.getuid()}/bus"
    assert data.is_relative_to(actual / "snap/chromium/common/paai")
    assert incoming["HOME"].endswith("private-home")
    assert env["DISPLAY"] == ":98"
    assert "--load-extension=" + str(extension) in command
    assert command[-1] == "http://127.0.0.1:8092/"
    assert not any("remote-debugging" in arg for arg in command)
    assert not any("token" in arg for arg in command)


def test_debug_port_is_explicit_and_loopback_only(tmp_path, monkeypatch):
    browser = load("spark_browser")
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/usr/bin/chromium")
    command, *_ = browser.browser_plan(tmp_path, {"PAAI_BROWSER_DEBUG": "1"})
    assert "--remote-debugging-port=0" in command
    assert "--remote-debugging-address=127.0.0.1" in command


def test_ready_rejects_a_dead_proof_world_even_when_ports_answer(tmp_path, monkeypatch):
    import cascade.apps.process_owner as owner_module
    browser = load("spark_browser")
    state = tmp_path / "state"
    state.mkdir()
    (state / "proof.json").write_text(json.dumps({"verified": True, "sim": "isaac", "process": {"pid": 55}}))
    monkeypatch.setattr(browser, "ownership", lambda repo: (state, {"owner": "owned"}))
    monkeypatch.setattr(owner_module, "is_live", lambda record, owner: False)
    monkeypatch.setattr(browser, "get_json", lambda url: pytest.fail("must check the proof's actual world first"))
    assert not browser.ready(tmp_path)


@pytest.mark.parametrize("connected", [False, True])
def test_ready_requires_connected_attendee_chat(tmp_path, monkeypatch, connected):
    import cascade.apps.process_owner as owner_module
    browser = load("spark_browser")
    (tmp_path / "proof.json").write_text(json.dumps({"verified": True, "sim": "isaac", "process": {"pid": 55}}))
    monkeypatch.setattr(browser, "ownership", lambda repo: (tmp_path, {}))
    monkeypatch.setattr(owner_module, "is_live", lambda *a: True)
    monkeypatch.setattr(owner_module, "live_records", lambda *a: [{"role": role} for role in
        ("qwen", "isaac_bridge", "gateway_child", "cameras", "visitor")])
    def response(url):
        if url.endswith("/state"):
            return {"cameras": {name: {"online": True} for name in ("kitchen", "worktop", "side")}}
        return {"enabled": True, "ready": connected, "agent": "cascade-demo"}
    monkeypatch.setattr(browser, "get_json", response)
    assert browser.ready(tmp_path) is connected


def test_camera_surface_caches_original_jpegs_and_refuses_raw_or_motion_routes():
    camera = load("spark_cameras")
    calls = []
    jpg = b"\xff\xd8camera-boundary\xff\xd9"

    class Bridge(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            calls.append(request)
            data = {"ok": True, "t": len(calls), "rgb_jpeg_b64": base64.b64encode(jpg).decode()}
            self.wfile.write(json.dumps(data).encode() + b"\n")

    bridge = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Bridge)
    bridge.daemon_threads = True
    http = ThreadingHTTPServer(("127.0.0.1", 0), camera.CameraHandler)
    http.frames = camera.Frames(bridge.server_address[1])
    for server in (bridge, http):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{http.server_address[1]}"
    try:
        first = json.load(urlopen(origin + "/state"))
        assert set(first["cameras"]) == {"kitchen", "worktop", "side"}
        assert all(row["online"] for row in first["cameras"].values())
        assert urlopen(origin + "/snapshot/worktop.jpg").read() == jpg
        assert len(calls) == 3, "snapshot should share the already fetched frame"
        assert calls == [{"op": "frame", "camera": name} for name in ("proof", "cam0", "side")]
        for path in ("/exec", "/config", "/snapshot/../../secret", "/stream/unknown"):
            with pytest.raises(HTTPError) as error:
                urlopen(origin + path)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(origin + "/state", data=b'{"op":"set_joints"}')
        assert error.value.code == 405
    finally:
        for server in (http, bridge):
            server.shutdown()
            server.server_close()


def test_repeat_launch_attaches_without_restarting_or_invalidating_proof(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        helper = load("install_support")
        import spark_browser
        import cascade.apps.process_owner as owner_module
        monkeypatch.setattr(helper, "eula_accepted", lambda repo: True)
        monkeypatch.setattr(helper, "spark_ready", lambda repo, **kw: True)
        monkeypatch.setattr(spark_browser, "ownership", lambda repo: (tmp_path, {}))
        monkeypatch.setattr(spark_browser, "runtime_home", lambda repo: str(tmp_path))
        monkeypatch.setattr(owner_module, "live_records", lambda *a, **kw: [{"model_binding": {"sha": "same"}}])
        monkeypatch.setattr(helper, "model_environment", lambda repo: {})
        monkeypatch.setattr(helper, "qwen_binding", lambda *a: {"sha": "same"})
        monkeypatch.setattr(helper, "model_health", lambda: True)
        calls = []
        monkeypatch.setattr(helper, "start_spark_surfaces", lambda *a: calls.append("surfaces"))
        monkeypatch.setattr(helper, "open_browser", lambda *a: calls.append("browser"))
        monkeypatch.setattr(helper, "_launch", lambda *a, **kw: pytest.fail("duplicated the stack"))
        assert helper.launch(tmp_path, "spark", "qwen") == 0
        assert calls == ["surfaces", "browser"]
        output = capsys.readouterr().out
        assert "READY: attached" in output
        assert "Demo UI: http://127.0.0.1:8092" in output
        assert "OpenClaw dashboard: ./run.sh dashboard" in output
    finally:
        sys.path.pop(0)


def test_surface_failure_stops_only_components_created_by_this_start(tmp_path, monkeypatch):
    from conftest import start_sleeping_process
    from test_spark_install import launch_fixture
    from cascade.apps.process_owner import load_owner, register_process
    helper, repo = launch_fixture(tmp_path, monkeypatch)
    state = repo / "runs/.launch/profile-cascade-demo"
    owner = load_owner(state, repo, "cascade-demo", create=True)
    prior = start_sleeping_process()
    new = []
    register_process(state, owner, prior.pid, "preserved")

    def broken_surfaces(repo, env):
        process = start_sleeping_process()
        new.append(process)
        register_process(state, owner, process.pid, "cameras")
        raise RuntimeError("camera startup boundary failure")

    monkeypatch.setattr(helper, "start_spark_surfaces", broken_surfaces)
    try:
        with pytest.raises(RuntimeError, match="camera startup boundary failure"):
            helper.launch(repo, "spark", "qwen", no_open=True)
        assert prior.poll() is None
        assert new[0].wait(timeout=5) != 0
        assert all(process.poll() is not None for process in helper._fixture_qwen)
        assert json.loads((state / "proof.json").read_text())["verified"] is False
    finally:
        prior.terminate()
        prior.wait(timeout=5)
        for process in [*new, *helper._fixture_qwen]:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


def test_surface_interpreter_ack_preserves_pid_arguments_imports_and_registration(tmp_path):
    from cascade.apps.process_owner import is_live, load_owner, register_process
    browser = load("spark_browser")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sibling.py").write_text("VALUE = 'script-local import'\n")
    script = scripts / "surface.py"
    result = tmp_path / "child.json"
    script.write_text("import json,os,sys,time\nfrom pathlib import Path\nimport sibling\n"
                      "target=Path(sys.argv[1]); pending=target.with_suffix('.tmp')\n"
                      "pending.write_text(json.dumps({'pid':os.getpid(), 'argv':sys.argv, "
                      "'cwd':os.getcwd(), 'sibling':sibling.VALUE}))\npending.replace(target)\ntime.sleep(120)\n")
    state = tmp_path / "state"
    owner = load_owner(state, tmp_path, "cascade-demo", create=True)
    with (tmp_path / "surface.log").open("ab") as log:
        process = browser._spawn_surface([sys.executable, str(script), str(result), "--value", "with spaces"],
                                         tmp_path, os.environ.copy(), log)
    try:
        receipt = register_process(state, owner, process.pid, "cameras")
        deadline = time.monotonic() + 10
        while not result.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(.01)
        assert json.loads(result.read_text()) == {
            "pid": process.pid, "argv": [str(script), str(result), "--value", "with spaces"],
            "cwd": str(tmp_path), "sibling": "script-local import"}
        assert is_live(receipt, owner)
    finally:
        process.terminate()
        process.wait(timeout=5)


@pytest.mark.parametrize("mode", ["timeout", "eof", "invalid_ack"])
def test_surface_interpreter_failure_reaps_its_exact_child_and_closes_pipes(tmp_path, monkeypatch, mode):
    browser = load("spark_browser")
    children, descriptors = [], []
    original_popen, original_pipe = browser.subprocess.Popen, browser.os.pipe
    def capture(command, **kwargs):
        child = original_popen(command, **kwargs)
        children.append(child)
        return child
    def pipe():
        pair = original_pipe()
        descriptors.extend(pair)
        return pair
    monkeypatch.setattr(browser.subprocess, "Popen", capture)
    monkeypatch.setattr(browser.os, "pipe", pipe)
    code = {"timeout": "import time; time.sleep(120)",
            "eof": "raise SystemExit(23)",
            "invalid_ack": "import os,sys,time; os.write(int(sys.argv[1]), b'X'); time.sleep(120)"}[mode]
    monkeypatch.setattr(browser, "_SURFACE_BOOTSTRAP", code)
    pattern = {"timeout": "readiness timed out", "eof": "before interpreter readiness",
               "invalid_ack": "Invalid surface interpreter"}[mode]
    with (tmp_path / "surface.log").open("ab") as log, pytest.raises(RuntimeError, match=pattern):
        browser._spawn_surface([sys.executable, str(tmp_path / "unused.py")], tmp_path,
                               os.environ.copy(), log, startup_timeout_s=.2 if mode == "timeout" else 10)
    assert len(children) == 1 and children[0].poll() is not None
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize("failure", ["registration", "readiness"])
def test_surface_startup_preserves_prior_owned_child_when_new_start_fails(tmp_path, monkeypatch, failure):
    from conftest import start_sleeping_process
    import cascade.apps.process_owner as owners
    browser = load("spark_browser")
    state = tmp_path / "state"
    owner = owners.load_owner(state, tmp_path, "cascade-demo", create=True)
    prior = start_sleeping_process()
    try:
        prior_receipt = owners.register_process(state, owner, prior.pid, "visitor")
        script = tmp_path / "scripts/spark_cameras.py"
        script.parent.mkdir()
        script.write_text("import time\ntime.sleep(120)\n")
        monkeypatch.setattr(browser, "ownership", lambda repo: (state, owner))
        def unused_port(*args, **kwargs):
            raise ConnectionRefusedError()
        monkeypatch.setattr(browser.socket, "create_connection", unused_port)
        original_popen = browser.subprocess.Popen
        created = []
        def capture(command, **kwargs):
            child = original_popen(command, **kwargs)
            created.append(child)
            return child
        monkeypatch.setattr(browser.subprocess, "Popen", capture)
        if failure == "registration":
            def reject_registration(*args, **kwargs):
                raise ValueError("exact owner identity rejected")
            monkeypatch.setattr(owners, "register_process", reject_registration)
        def fail_readiness(repo):
            raise RuntimeError("surface readiness rejected")
        monkeypatch.setattr(browser, "ready", fail_readiness)
        with pytest.raises((ValueError, RuntimeError), match="rejected"):
            browser.start_surfaces(tmp_path, os.environ.copy())
        assert len(created) == 1 and created[0].poll() is not None
        assert prior.poll() is None and owners.is_live(prior_receipt, owner)
    finally:
        prior.terminate()
        prior.wait(timeout=5)
