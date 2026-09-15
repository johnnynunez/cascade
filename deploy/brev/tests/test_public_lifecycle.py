"""Public service operations preserve unrelated workloads and unchanged tunnels."""
import importlib.util
import json
import os
from pathlib import Path
import pwd

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("visitor_lifecycle", HERE / "public.py")
visitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visitor)


@pytest.fixture
def host(tmp_path, monkeypatch):
    root = tmp_path / "deployment"
    state = root / "data/state"
    state.mkdir(parents=True)
    (state / "deployment.json").write_text("{}")
    executable = root / "tools/ngrok/ngrok"
    executable.parent.mkdir(parents=True)
    executable.write_text("fixture")
    token = tmp_path / "ngrok.token"
    token.write_text("fixture-ngrok-token")
    token.chmod(0o600)
    source = tmp_path / "source"
    source.mkdir()
    for name in (*visitor.VISITOR_FILES, "public.py"):
        (source / name).write_text("fixture " + name)
    units = tmp_path / "units"
    units.mkdir()
    monkeypatch.setattr(visitor, "UNIT_DIRECTORY", units)
    monkeypatch.setattr(visitor, "HERE", source)
    monkeypatch.setattr(visitor, "current_url", lambda: "https://visitor.example")
    monkeypatch.setattr(visitor, "verify_visitor", lambda *args: None)
    active = {"unrelated": True}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["systemctl", "show"]:
            name = command[2].removesuffix(".service")
            unit = units / (name + ".service")
            return (f"LoadState={'loaded' if unit.exists() else 'not-found'}\n"
                    f"ActiveState={'active' if active.get(name) else 'inactive'}\n"
                    f"FragmentPath={unit if unit.exists() else ''}\n")
        if command[:3] == ["sudo", "-n", "tee"]:
            Path(command[3]).write_text(kwargs["input"])
        elif command[:3] == ["sudo", "-n", "systemctl"]:
            operation = command[3]
            if operation in ("enable", "start", "restart", "stop"):
                for name in command[4:]:
                    if not name.startswith("--"):
                        active[name] = operation != "stop"
        return ""

    monkeypatch.setattr(visitor, "run", run)
    return {"root": root, "token": token, "source": source,
            "units": units, "active": active, "calls": calls}


def restarts(host):
    return [call[4:] for call in host["calls"] if call[:4] == ["sudo", "-n", "systemctl", "restart"]]


def test_install_reuses_active_services_and_private_credentials(host):
    first = visitor.install(host["root"], host["token"], local_auth=True)
    auth_file = Path(first["credentials_file"])
    auth = auth_file.read_bytes()
    config = host["root"] / "data/private/public-visitor/ngrok.yml"
    modified = config.stat().st_mtime_ns
    host["calls"].clear()
    second = visitor.install(host["root"], host["token"], local_auth=True)
    assert second["healthy"] and second["url"] == "https://visitor.example"
    assert restarts(host) == []
    assert auth_file.read_bytes() == auth and config.stat().st_mtime_ns == modified
    assert host["active"]["unrelated"]
    assert not any("daemon-reload" in call for call in host["calls"])
    assert "fixture-ngrok-token" not in json.dumps(host["calls"])
    assert json.loads(auth)["password"] not in json.dumps(second)


def test_changed_visitor_source_restarts_only_visitor(host):
    visitor.install(host["root"], host["token"])
    (host["source"] / "visitor.js").write_text("updated visitor")
    host["calls"].clear()
    result = visitor.install(host["root"], host["token"])
    assert restarts(host) == [["paai-visitor"]]
    assert result["restarted"] == ["paai-visitor"]


def test_changed_token_restarts_only_ngrok_without_command_line_secret(host):
    visitor.install(host["root"], host["token"])
    host["token"].write_text("changed-fixture-token")
    host["calls"].clear()
    result = visitor.install(host["root"], host["token"])
    assert restarts(host) == [["paai-ngrok"]]
    assert result["restarted"] == ["paai-ngrok"]
    assert "changed-fixture-token" not in json.dumps(host["calls"])


def test_changing_tunnel_auth_policy_never_removes_local_auth(host):
    visitor.install(host["root"], host["token"], local_auth=True)
    expected = "--auth-file=" + str(host["root"] / "data/private/public-visitor/auth.json")
    original_run = visitor.run
    def checked_run(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[:3] == ["sudo", "-n", "systemctl"]:
            unit = (host["units"] / "paai-visitor.service").read_text()
            assert expected in unit
        return result
    visitor.run = checked_run
    try:
        visitor.install(host["root"], host["token"], local_auth=False)
        visitor.install(host["root"], host["token"], local_auth=True)
    finally:
        visitor.run = original_run


def test_adopting_ngrok_native_config_preserves_unchanged_tunnel(host):
    visitor.install(host["root"], host["token"])
    config = host["root"] / "data/private/public-visitor/ngrok.yml"
    contents = 'version: "3"\nagent:\n    authtoken: fixture-ngrok-token\n'
    config.write_text(contents)
    host["calls"].clear()
    visitor.install(host["root"], host["token"])
    assert config.read_text() == contents
    assert restarts(host) == []


def test_semantically_unchanged_private_json_preserves_active_tunnel(host):
    visitor.install(host["root"], host["token"])
    private = host["root"] / "data/private/public-visitor"
    for name in ("policy.json", "ngrok.yml"):
        path = private / name
        path.write_text(json.dumps(json.loads(path.read_text()), separators=(",", ":")))
    before = {name: (private / name).read_bytes() for name in ("policy.json", "ngrok.yml")}
    host["calls"].clear()
    visitor.install(host["root"], host["token"])
    assert restarts(host) == []
    assert before == {name: (private / name).read_bytes() for name in before}


def test_service_metadata_change_reloads_units_without_restarting_processes(host):
    visitor.install(host["root"], host["token"])
    for name in visitor.SERVICES:
        path = host["units"] / (name + ".service")
        text = path.read_text().replace("Description=Physical Agentic AI", "Description=Earlier demo")
        text = text.replace("After=network-online.target paai-demo-docker.service", "After=network.target")
        path.write_text(text.replace("RestartSec=5", "RestartSec=9"))
    host["calls"].clear()
    visitor.install(host["root"], host["token"])
    assert restarts(host) == []
    assert ["sudo", "-n", "systemctl", "daemon-reload"] in host["calls"]


def test_service_security_change_restarts_only_the_affected_process(host):
    visitor.install(host["root"], host["token"])
    unit = host["units"] / "paai-ngrok.service"
    unit.write_text(unit.read_text().replace("NoNewPrivileges=true", "NoNewPrivileges=false"))
    host["calls"].clear()
    visitor.install(host["root"], host["token"])
    assert restarts(host) == [["paai-ngrok"]]


def test_service_name_collision_is_refused_before_install_writes(host):
    unit = host["units"] / "paai-visitor.service"
    unit.write_text("[Service]\nUser=" + pwd.getpwuid(os.getuid()).pw_name
                    + "\nExecStart=/usr/bin/python3 /other/visitor.py\n")
    before = unit.read_bytes()
    with pytest.raises(ValueError, match="another deployment"):
        visitor.install(host["root"], host["token"])
    assert unit.read_bytes() == before
    assert not (host["root"] / "data/private").exists()
    assert not any(call[0] == "sudo" for call in host["calls"])


def test_stop_start_and_status_touch_only_owned_public_services(host):
    visitor.install(host["root"], host["token"])
    stopped = visitor.service_action(host["root"], "stop")
    assert stopped["installed"] and not stopped["healthy"] and stopped["url"] is None
    assert host["active"]["unrelated"]
    started = visitor.service_action(host["root"], "start")
    assert started["healthy"] and started["url"] == "https://visitor.example"
    assert host["active"]["unrelated"]


def test_status_does_not_report_http_failure_as_healthy(host, monkeypatch):
    visitor.install(host["root"], host["token"])

    def unavailable(*args):
        raise RuntimeError("private response withheld")

    monkeypatch.setattr(visitor, "verify_visitor", unavailable)
    result = visitor.status(host["root"])
    assert not result["healthy"] and result["url"] == "https://visitor.example"
    assert "private response" not in json.dumps(result)


def test_video_flag_changes_only_the_visitor_unit(host, monkeypatch):
    visitor.install(host["root"], host["token"])
    monkeypatch.setattr(visitor.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    host["calls"].clear()
    visitor.install(host["root"], host["token"], video=True)
    assert restarts(host) == [["paai-visitor"]]
    assert " --video\n" in (host["units"] / "paai-visitor.service").read_text()
    assert not any("apt-get" in call for call in host["calls"])


def test_video_installs_only_missing_ffmpeg(host, monkeypatch):
    found = iter((None, "/usr/bin/ffmpeg"))
    monkeypatch.setattr(visitor.shutil, "which", lambda name: next(found))
    visitor.install(host["root"], host["token"], video=True)
    installs = [call for call in host["calls"] if "apt-get" in call]
    assert installs == [["sudo", "-n", "apt-get", "install", "-y", "--no-install-recommends", "ffmpeg"]]


def test_stop_marker_allows_stop_but_refuses_start(host):
    visitor.install(host["root"], host["token"])
    (host["root"] / "source").mkdir()
    (host["root"] / "source/STOP").touch()
    assert not visitor.service_action(host["root"], "stop")["healthy"]
    with pytest.raises(InterruptedError, match="STOP"):
        visitor.service_action(host["root"], "start")


def test_service_dropin_is_refused_before_lifecycle_mutation(host, monkeypatch):
    visitor.install(host["root"], host["token"])
    original = visitor.run

    def override(command, **kwargs):
        result = original(command, **kwargs)
        return result + "DropInPaths=/other/override.conf\n" if command[:2] == ["systemctl", "show"] else result

    monkeypatch.setattr(visitor, "run", override)
    host["calls"].clear()
    with pytest.raises(ValueError, match="ownership review"):
        visitor.service_action(host["root"], "stop")
    assert not any(call[0] == "sudo" for call in host["calls"])


@pytest.mark.parametrize("relative", ["data/private", "tools/visitor"])
def test_install_refuses_directories_outside_the_deployment(host, relative):
    outside = host["root"].parent / "unrelated"
    outside.mkdir()
    (host["root"] / relative).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the deployment"):
        visitor.install(host["root"], host["token"])
    assert list(outside.iterdir()) == []
