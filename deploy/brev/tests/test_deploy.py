from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("brev_deploy", HERE / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_access_failure_never_runs_remote_or_sync(monkeypatch):
    monkeypatch.setattr(deploy, "boundary", lambda: None)
    monkeypatch.setattr(deploy.access_gate, "probe", lambda: {
        "status": "WAITING_FOR_ACCESS", "reason": "No target peer"})
    monkeypatch.setattr(deploy, "update_gate", lambda *args: None)
    monkeypatch.setattr(deploy, "remote", lambda *args, **kwargs: pytest.fail("remote execution"))
    monkeypatch.setattr(deploy, "sync", lambda *args, **kwargs: pytest.fail("file transfer"))
    with pytest.raises(ConnectionError, match="No target peer"):
        deploy.require_access()


def test_transfer_rejects_escape_before_rsync(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "private.pem"
    outside.write_text("fixture")
    (source / "escape").symlink_to(outside)
    monkeypatch.setattr(deploy, "boundary", lambda: None)
    monkeypatch.setattr(deploy, "execute", lambda *args, **kwargs: pytest.fail("rsync execution"))
    for member in ("../private.pem", str(outside), "escape", "file\nprivate.pem"):
        with pytest.raises(ValueError):
            deploy.sync(source, "/dedicated/destination", [member])


def test_healthy_start_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "remote_preparation", lambda p: {"active": False})
    monkeypatch.setattr(deploy, "remote_installation", lambda p: {"status": "PREPARED"})
    monkeypatch.setattr(deploy, "runtime_image_matches", lambda *args: True)
    monkeypatch.setattr(deploy, "require_installed_bundle", lambda *args: None)
    monkeypatch.setattr(deploy, "running_rows", lambda p: [{"Service": "demo", "State": "running", "Health": "healthy"}])
    monkeypatch.setattr(deploy, "preflight", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(deploy, "compose", lambda *a, **kw: pytest.fail("healthy workload restarted"))
    monkeypatch.setattr(deploy, "status", lambda p: {"ready": False, "live_gates": "pending"})
    assert deploy.start({}, Path("profile.json"))["ready"] is False
    assert calls == [{"installed": True, "running": True}]


def test_completed_install_skips_transfers_builds_and_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "staged_bundle", lambda: (tmp_path, ["source.py"]))
    monkeypatch.setattr(deploy, "bundle_identity", lambda *a: "same")
    monkeypatch.setattr(deploy, "remote_installation", lambda p: {"status": "PREPARED", "bundle_identity": "same"})
    monkeypatch.setattr(deploy, "remote_preparation", lambda p: {"active": False})
    monkeypatch.setattr(deploy, "runtime_image_matches", lambda *args: True)
    monkeypatch.setattr(deploy, "start", lambda *a: {"reused": True})
    monkeypatch.setattr(deploy, "sync", lambda *a, **kw: pytest.fail("unnecessary transfer"))
    monkeypatch.setattr(deploy, "remote", lambda *a, **kw: pytest.fail("unnecessary download/build"))
    assert deploy.install({}, Path("profile.json"), {}) == {"reused": True}


def test_failed_start_stops_owned_restart_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "remote_preparation", lambda p: {"active": False})
    monkeypatch.setattr(deploy, "remote_installation", lambda p: {"status": "PREPARED"})
    monkeypatch.setattr(deploy, "runtime_image_matches", lambda *args: True)
    monkeypatch.setattr(deploy, "require_installed_bundle", lambda *args: None)
    monkeypatch.setattr(deploy, "running_rows", lambda p: [])
    monkeypatch.setattr(deploy, "preflight", lambda *a, **kw: {})
    monkeypatch.setattr(deploy, "stop_ownership", lambda p: {"containers": ["a" * 64]})
    monkeypatch.setattr(deploy, "stop_container_ids", lambda ids, log: calls.append(("owned_stop", *ids)))
    def compose(profile, *args, **kwargs):
        calls.append(args)
        if args[0] == "up":
            raise RuntimeError("Health deadline failed")
    monkeypatch.setattr(deploy, "compose", compose)
    with pytest.raises(RuntimeError, match="Health deadline failed"):
        deploy.start({}, Path("profile.json"))
    assert [call[0] for call in calls] == ["up", "logs", "owned_stop"]
    assert calls[-1] == ("owned_stop", "a" * 64)


def test_compose_status_supports_array_and_ndjson(monkeypatch):
    row = {"Name": "paai-demo-demo-1", "Service": "demo", "State": "running", "Health": "healthy"}
    for output in (json.dumps([row]), json.dumps(row) + "\n"):
        monkeypatch.setattr(deploy, "compose", lambda *a, value=output, **kw: value)
        assert deploy.running_rows({})[0]["Health"] == "healthy"


def test_disabled_public_visitor_never_contacts_or_changes_services(monkeypatch):
    monkeypatch.setattr(deploy, "remote", lambda *a, **kw: pytest.fail("remote public operation"))
    monkeypatch.setattr(deploy, "sync", lambda *a, **kw: pytest.fail("public source transfer"))
    for operation in ("install", "start", "status", "restart", "stop"):
        assert deploy.public_visitor({}, operation) == {"enabled": False}


def test_public_install_uses_external_host_token_path_and_selected_video(monkeypatch):
    calls, transfers = [], []
    profile = {"storage": {"root": "/srv/demo"}, "public_visitor": {
        "enabled": True, "token_file": "/secure/ngrok.token", "local_auth": True, "video": True}}

    def remote(argv, **kwargs):
        calls.append(argv)
        return json.dumps({"healthy": True, "url": "https://visitor.example"})

    monkeypatch.setattr(deploy, "remote", remote)
    monkeypatch.setattr(deploy, "sync", lambda *a, **kw: transfers.append(a))
    result = deploy.public_visitor(profile, "install")
    assert result["enabled"] and result["healthy"]
    assert calls[-1] == ["python3", "/srv/demo/control/visitor/public.py", "install", "--root",
                         "/srv/demo", "--token-file", "/secure/ngrok.token", "--local-auth", "--video"]
    assert transfers[0][1] == "/srv/demo/control/visitor"
    assert {"public.py", "visitor_video.py", "visitor-player.js"} <= set(transfers[0][2])


def test_public_status_reports_missing_installation_without_starting_it(monkeypatch):
    profile = {"storage": {"root": "/srv/demo"}, "public_visitor": {"enabled": True}}

    def absent(argv, **kwargs):
        assert argv == ["test", "-f", "/srv/demo/tools/visitor/public.py"]
        raise RuntimeError("missing")

    monkeypatch.setattr(deploy, "remote", absent)
    result = deploy.public_visitor(profile, "status")
    assert result == {"enabled": True, "installed": False, "healthy": False, "url": None}
    with pytest.raises(RuntimeError, match="run install first"):
        deploy.public_visitor(profile, "start")


def test_stop_closes_public_visitor_before_owned_application(monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "stop_ownership", lambda p: calls.append(("admit",)) or {
        "preparation": {"active": False}, "containers": ["a" * 64]})
    monkeypatch.setattr(deploy, "public_visitor", lambda p, op: calls.append(("public", op)) or {})
    monkeypatch.setattr(deploy, "stop_container_ids", lambda ids, log: calls.append(("owned_stop", *ids)))
    assert deploy.stop({})["status"] == "STOPPED"
    assert calls == [("admit",), ("public", "stop"), ("owned_stop", "a" * 64)]


def test_status_includes_measured_public_url_and_health(tmp_path, monkeypatch):
    access = tmp_path / "access"
    access.mkdir()
    (access / "PASSED.json").write_text(json.dumps({"peer": {"TailscaleIPs": ["100.64.0.1"]}}))
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(deploy, "running_rows", lambda p: [])
    monkeypatch.setattr(deploy, "remote_preparation", lambda p: {"active": False})
    monkeypatch.setattr(deploy, "public_visitor", lambda p, op: {
        "enabled": True, "healthy": True, "url": "https://visitor.example"})
    result = deploy.status({})
    assert result["urls"]["public_visitor"] == "https://visitor.example"
    assert result["public_visitor"]["healthy"] is True
    assert result["ready"] is False


@pytest.mark.parametrize("settings", [{"enabled": "yes"}, {"enabled": True, "token_file": "relative"},
                                      {"enabled": True, "token_file": "/secure/token", "video": "yes"}])
def test_invalid_public_configuration_fails_before_any_remote_change(settings, monkeypatch):
    monkeypatch.setattr(deploy, "remote", lambda *a, **kw: pytest.fail("invalid configuration changed host"))
    with pytest.raises(ValueError):
        deploy.public_visitor({"storage": {"root": "/srv/demo"}, "public_visitor": settings}, "install")


@pytest.fixture
def restart_runtime(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"name": "brev-rtx6000", "public_visitor": {"enabled": True}}))
    state = {"preparing": False, "receipt": {"status": "PREPARED", "bundle_identity": "selected"},
             "image_matches": True, "preflight_passes": True, "running": True, "events": []}
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(deploy, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(sys, "argv", ["deploy.py", "restart", "--profile", str(profile)])
    monkeypatch.setattr(deploy, "require_access", lambda profile: {})
    monkeypatch.setattr(deploy, "remote_preparation", lambda profile: {"active": state["preparing"]})
    monkeypatch.setattr(deploy, "remote_installation", lambda profile: state["receipt"])
    monkeypatch.setattr(deploy, "runtime_image_matches", lambda *args: state["image_matches"])
    monkeypatch.setattr(deploy, "staged_bundle", lambda: (tmp_path, ["source.py"]))
    monkeypatch.setattr(deploy, "bundle_identity", lambda *args: "selected")
    monkeypatch.setattr(deploy, "running_rows", lambda profile: [
        {"Service": "demo", "State": "running", "Health": "healthy"}] if state["running"] else [])
    monkeypatch.setattr(deploy, "stop_ownership", lambda profile: {
        "preparation": {"active": False}, "containers": ["a" * 64]})
    monkeypatch.setattr(deploy, "status", lambda profile: {"ready": False})

    def preflight(path, *, installed, running):
        assert path == profile and installed
        state["events"].append(("preflight", running))
        if not state["preflight_passes"]:
            raise ValueError("GPU ownership admission failed")

    def public(profile, operation):
        state["events"].append(("public", operation))
        return {"healthy": operation == "start"}

    def stop_containers(identities, log):
        assert identities == ["a" * 64]
        state["events"].append(("containers", "stop"))
        state["running"] = False

    def compose(profile, *args, **kwargs):
        assert args[0] == "up"
        state["events"].append(("containers", "start"))
        state["running"] = True

    monkeypatch.setattr(deploy, "preflight", preflight)
    monkeypatch.setattr(deploy, "public_visitor", public)
    monkeypatch.setattr(deploy, "stop_container_ids", stop_containers)
    monkeypatch.setattr(deploy, "compose", compose)
    return state


@pytest.mark.parametrize("failure", [
    "active_preparation", "missing_receipt", "incomplete_receipt",
    "replaced_image", "changed_bundle", "preflight",
])
def test_restart_rejects_incompatible_runtime_before_public_or_container_stop(restart_runtime, failure):
    state = restart_runtime
    if failure == "active_preparation":
        state["preparing"] = True
    elif failure == "missing_receipt":
        state["receipt"] = {}
    elif failure == "incomplete_receipt":
        state["receipt"]["status"] = "UPDATING"
    elif failure == "replaced_image":
        state["image_matches"] = False
    elif failure == "changed_bundle":
        state["receipt"]["bundle_identity"] = "previous-source-or-profile"
    else:
        state["preflight_passes"] = False
    assert deploy.main() == 1
    assert state["running"]
    assert not any(event[0] in {"public", "containers"} for event in state["events"])


def test_restart_validates_before_stop_and_checks_stopped_runtime_before_start(restart_runtime):
    assert deploy.main() == 0
    assert restart_runtime["running"]
    assert restart_runtime["events"] == [
        ("preflight", True), ("public", "stop"), ("containers", "stop"),
        ("preflight", False), ("containers", "start"), ("public", "start"),
    ]
