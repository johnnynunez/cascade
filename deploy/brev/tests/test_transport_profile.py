"""Profile selection and cleanup must preserve the admitted private transport."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("transport_deploy", HERE / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
access = deploy.access_gate


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    (campaign / "access").mkdir(parents=True)
    monkeypatch.setattr(deploy, "HERE", campaign)
    monkeypatch.setattr(deploy, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(access, "HERE", campaign)
    monkeypatch.setattr(access, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    wrapper = tmp_path / "transport"
    wrapper.write_text("#!/bin/sh\nexit 99\n")
    wrapper.chmod(0o700)
    monkeypatch.setattr(access, "WRAPPER", wrapper)
    for name in ("TARGET", "ALIAS", "PRIVATE_CONFIG"):
        monkeypatch.setattr(access, name, getattr(access, name))
    monkeypatch.setattr(deploy, "update_gate", lambda *args: None)
    return campaign


def profile(target="review-rtx6000"):
    return {"network": {"access": "tailscale", "tailscale_hostname": target}}


@pytest.mark.parametrize("target", ["invalid hostname", "-oProxyCommand=bad", "../host", "target\nHost *", "100.64.0.1", "host.example", "UPPER", "", None])
def test_invalid_profile_cannot_change_transport_or_probe(isolated, monkeypatch, target):
    before = access.TARGET, access.ALIAS, access.PRIVATE_CONFIG
    monkeypatch.setattr(access, "run", lambda *a, **k: pytest.fail("Invalid target probed"))
    with pytest.raises(ValueError):
        deploy.require_access(profile(target))
    assert (access.TARGET, access.ALIAS, access.PRIVATE_CONFIG) == before


def test_profile_selects_exact_peer_and_shared_wrapper_for_command_and_rsync(isolated, monkeypatch, tmp_path):
    wrapper = tmp_path / "custom transport"
    wrapper.write_text("#!/bin/sh\nexit 99\n")
    wrapper.chmod(0o700)
    monkeypatch.setenv("PAAI_TRANSPORT_WRAPPER", str(wrapper))
    configured_access = runpy.run_path(str(HERE / "access_gate.py"))
    monkeypatch.setattr(access, "WRAPPER", configured_access["WRAPPER"])
    assert access.WRAPPER == wrapper
    calls = []
    configured = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "tailscale":
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "BackendState": "Running", "Peer": {
                    "other": {"HostName": "other-demo", "TailscaleIPs": ["100.64.0.10"]},
                    "selected": {"HostName": "review-rtx6000", "Online": True, "TailscaleIPs": ["100.64.0.20"]}}}))
        return SimpleNamespace(returncode=0, stderr="", stdout="PAAI_ACCESS_GATE_PASSED\n")

    monkeypatch.setattr(access, "run", run)
    monkeypatch.setattr(access, "configure", lambda ip: configured.append(ip))
    proof = deploy.require_access(profile())
    assert configured == ["100.64.0.20"]
    assert proof["status"] == "PASS" and proof["target"] == "review-rtx6000"
    assert calls[1][:4] == [access.WRAPPER, "-F", tmp_path / ".ssh/config.d/review-rtx6000.conf", "review-rtx6000-tailnet"]
    operations = []
    monkeypatch.setattr(deploy, "execute", lambda argv, **kwargs: operations.append(argv))
    deploy.remote(["hostname"])
    source = tmp_path / "source"
    source.mkdir()
    (source / "file.py").write_text("# Fixture\n")
    deploy.sync(source, "/target", ["file.py"])
    assert operations[0][:4] == calls[1][:4]
    transfer = operations[1]
    assert transfer[0] == "rsync"
    assert shlex.split(transfer[transfer.index("-e") + 1]) == list(map(str, calls[1][:3]))
    assert transfer[-1] == "review-rtx6000-tailnet:/target/"


@pytest.mark.parametrize("override", [None, "~/bin/transport"])
def test_wrapper_default_and_home_relative_configuration(isolated, monkeypatch, tmp_path, override):
    if override is None:
        monkeypatch.delenv("PAAI_TRANSPORT_WRAPPER", raising=False)
        expected = tmp_path / ".local/bin/totally-not-ssh.sh"
    else:
        monkeypatch.setenv("PAAI_TRANSPORT_WRAPPER", override)
        expected = Path(os.environ["HOME"]) / "bin/transport"
    configured_access = runpy.run_path(str(HERE / "access_gate.py"))
    assert configured_access["WRAPPER"] == expected


@pytest.mark.parametrize("problem", ["missing", "not_executable", "directory", "relative", "empty"])
def test_invalid_wrapper_never_configures_or_executes_remote(isolated, monkeypatch, tmp_path, problem):
    wrapper = tmp_path / "invalid transport"
    if problem == "not_executable":
        wrapper.write_text("fixture\n")
        wrapper.chmod(0o600)
    elif problem == "directory":
        wrapper.mkdir()
    elif problem == "relative":
        wrapper = Path("relative-transport")
    elif problem == "empty":
        wrapper = ""
    monkeypatch.setenv("PAAI_TRANSPORT_WRAPPER", str(wrapper))
    configured_access = runpy.run_path(str(HERE / "access_gate.py"))
    monkeypatch.setattr(access, "WRAPPER", configured_access["WRAPPER"])

    def run(argv, **kwargs):
        assert argv == ["tailscale", "status", "--json"]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "BackendState": "Running", "Peer": {"selected": {
                "HostName": "review-rtx6000", "TailscaleIPs": ["100.64.0.20"]}}}))

    monkeypatch.setattr(access, "run", run)
    monkeypatch.setattr(access, "configure", lambda *a: pytest.fail("Invalid wrapper wrote configuration"))
    with pytest.raises(ConnectionError, match="PAAI_TRANSPORT_WRAPPER"):
        deploy.require_access(profile())
    assert not (isolated / "access/PASSED.json").exists()
    record = json.loads((isolated / "access/current.json").read_text())
    assert record["remote_command_executed"] is False


def test_absent_selected_peer_cannot_fall_back_to_previous_instance(isolated, monkeypatch):
    def run(argv, **kwargs):
        assert argv == ["tailscale", "status", "--json"]
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "BackendState": "Running", "Peer": {"old": {
                "HostName": "other-demo", "TailscaleIPs": ["100.64.0.10"]}}}))
    monkeypatch.setattr(access, "run", run)
    with pytest.raises(ConnectionError):
        deploy.require_access(profile())
    assert not (isolated / "access/PASSED.json").exists()


def test_stop_cancels_only_receipt_target_even_after_different_profile_selection(isolated, monkeypatch, tmp_path):
    (isolated / "access/PASSED.json").write_text(json.dumps({
        "status": "PASS", "target": "prior-demo", "alias": "prior-demo-tailnet"}))
    (isolated / "proof").mkdir(exist_ok=True)
    (isolated / "proof/PREPARATION_OWNER.json").write_text(json.dumps({
        "target": "prior-demo", "root": "/fixture"}))
    access.select_profile(profile())
    deploy.STOP.touch()
    calls = []
    monkeypatch.setattr(deploy.subprocess, "run", lambda argv, **kwargs: (
        calls.append(argv) or SimpleNamespace(returncode=0)))
    result = deploy.cancel_preparation()
    assert result["stopped"] is True and result["target"] == "prior-demo"
    assert calls[0][2:4] == [str(tmp_path / ".ssh/config.d/prior-demo.conf"), "prior-demo-tailnet"]
    assert "paai-demo-prepare.service" in calls[0][-1]


@pytest.mark.parametrize("receipt", [{}, {"status": "FAIL"},
    {"status": "PASS", "target": "invalid hostname", "alias": "paai-demo-tailnet"},
    {"status": "PASS", "target": "prior-demo", "alias": "different-tailnet"}])
def test_unbound_cleanup_receipt_cannot_launch_command(isolated, monkeypatch, receipt):
    (isolated / "access/PASSED.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(deploy.subprocess, "run", lambda *a, **k: pytest.fail("Unbound cleanup"))
    assert deploy.cancel_preparation()["attempted"] is False
