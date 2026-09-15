"""A stop request cannot cross deployment roots or stop uninspected services."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deploy


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root = tmp_path / "deployment"
    identity = "a" * 64
    labels = {"com.docker.compose.project": "paai-demo", "com.docker.compose.service": "demo",
              "com.docker.compose.project.working_dir": str(root / "deployment"),
              "com.docker.compose.project.config_files": str(root / "deployment/compose.yaml")}
    row = {"id": identity, "labels": labels, "mounts": [
        {"Type": "bind", "Source": str(root / source), "Destination": destination}
        for source, destination in (("source", "/workspace"), ("data", "/data"), ("models", "/data/models"))]}
    unit = {"LoadState": "loaded", "ActiveState": "active", "User": "ubuntu", "DropInPaths": "",
            "FragmentPath": "/run/systemd/transient/" + deploy.prepare_runtime.UNIT,
            "ExecStart": "{ argv[]=python3 " + deploy.CONTROL + "/prepare_runtime.py run --root "
                         + str(root) + " ; }"}
    fixture = {"root": root, "profile": {"storage": {"root": str(root)}}, "unit": unit,
               "rows": {identity: row}, "docker_root": str(root / "docker"),
               "identities": [identity], "mutations": [], "reads": []}

    def execute_guard(arguments):
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", ["-c", *arguments])
            output = StringIO()
            with redirect_stdout(output):
                try:
                    exec(deploy.PREPARATION_UNIT_GUARD, {})
                except SystemExit as error:
                    assert error.code == 0
            return output.getvalue()

    def subprocess_run(command, **kwargs):
        if command[:2] == ["systemctl", "show"]:
            fixture["reads"].append("preparation")
            return SimpleNamespace(returncode=0, stdout="\n".join(k + "=" + v for k, v in unit.items()))
        if command[:4] == ["sudo", "-n", "systemctl", "stop"]:
            fixture["mutations"].append(("preparation", command[-1]))
            unit["ActiveState"] = "inactive"
            return SimpleNamespace(returncode=0)
        assert command[0] == str(deploy.access_gate.WRAPPER)
        remote_command = shlex.split(command[-1])
        try:
            execute_guard(remote_command[5:])
            return SimpleNamespace(returncode=0)
        except ValueError:
            return SimpleNamespace(returncode=1)

    def remote(command, **kwargs):
        if command[:2] == ["python3", "-c"]:
            assert command[2] == deploy.PREPARATION_UNIT_GUARD
            return execute_guard(command[3:])
        assert command[:3] == ["docker", "--host", deploy.DOCKER_SOCKET]
        operation = command[3]
        fixture["reads"].append(operation)
        if operation == "info":
            return fixture["docker_root"]
        if operation == "ps":
            return "\n".join(fixture["identities"])
        if operation == "inspect":
            return json.dumps(fixture["rows"][command[-1]])
        if operation == "stop":
            fixture["mutations"].append(("containers", *command[6:]))
            return "\n".join(command[6:])
        pytest.fail("Unexpected Docker operation")

    def public(profile, operation):
        assert operation == "stop" and "inspect" in fixture["reads"]
        fixture["mutations"].append(("public", "stop"))
        return {"healthy": False}

    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(deploy, "remote", remote)
    monkeypatch.setattr(deploy.subprocess, "run", subprocess_run)
    monkeypatch.setattr(deploy, "public_visitor", public)
    return fixture


def test_owned_stop_checks_every_owner_then_stops_only_inspected_ids(owned):
    result = deploy.stop(owned["profile"])
    assert result["status"] == "STOPPED"
    assert owned["mutations"] == [("public", "stop"), ("preparation", deploy.prepare_runtime.UNIT),
                                    ("containers", "a" * 64)]


@pytest.mark.parametrize("key,value", [
    ("com.docker.compose.project", "other-project"),
    ("com.docker.compose.service", "unrelated"),
    ("com.docker.compose.project.working_dir", "/other/deployment"),
    ("com.docker.compose.project.config_files", "/other/deployment/compose.yaml"),
])
def test_mismatched_compose_identity_blocks_all_stop_actions(owned, key, value):
    owned["rows"]["a" * 64]["labels"][key] = value
    with pytest.raises(ValueError, match="another deployment"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


def test_mismatched_mount_blocks_all_stop_actions(owned):
    owned["rows"]["a" * 64]["mounts"][0]["Source"] = "/unrelated/source"
    with pytest.raises(ValueError, match="mismatched deployment mounts"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


def test_another_docker_root_blocks_all_stop_actions(owned):
    owned["docker_root"] = "/other/docker"
    with pytest.raises(ValueError, match="another deployment"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


@pytest.mark.parametrize("key,value", [
    ("LoadState", "masked"), ("User", "root"),
    ("DropInPaths", "/other/override.conf"), ("FragmentPath", "/etc/systemd/system/other.service"),
    ("ExecStart", "{ argv[]=python3 /other/prepare_runtime.py run --root /other ; }"),
])
def test_mismatched_preparation_unit_blocks_all_stop_actions(owned, key, value):
    owned["unit"][key] = value
    with pytest.raises(ValueError, match="another deployment or is masked"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


def test_preparation_root_must_match_even_with_the_correct_script(owned):
    owned["unit"]["ExecStart"] = owned["unit"]["ExecStart"].replace(str(owned["root"]), "/other")
    with pytest.raises(ValueError, match="another deployment"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


def test_inactive_missing_preparation_is_safe_and_does_not_get_stopped(owned):
    owned["unit"].update(LoadState="not-found", ActiveState="inactive", ExecStart="", FragmentPath="")
    deploy.stop(owned["profile"])
    assert owned["mutations"] == [("public", "stop"), ("containers", "a" * 64)]


def test_new_container_after_admission_is_not_in_the_stop_request(owned, monkeypatch):
    def public(profile, operation):
        owned["identities"].append("b" * 64)
        return {}
    monkeypatch.setattr(deploy, "public_visitor", public)
    deploy.stop(owned["profile"])
    assert ("containers", "a" * 64) in owned["mutations"]
    assert all("b" * 64 not in row for row in owned["mutations"])


def test_preparation_is_rechecked_immediately_before_its_stop(owned, monkeypatch):
    def public(profile, operation):
        owned["unit"]["ExecStart"] = owned["unit"]["ExecStart"].replace(str(owned["root"]), "/other")
        return {}
    monkeypatch.setattr(deploy, "public_visitor", public)
    with pytest.raises(ValueError, match="another deployment"):
        deploy.stop(owned["profile"])
    assert owned["mutations"] == []


def write_cancel_owner(owned, *, owner=True):
    (deploy.HERE / "access").mkdir()
    (deploy.HERE / "access/PASSED.json").write_text(json.dumps({
        "status": "PASS", "target": "owned-demo", "alias": "owned-demo-tailnet"}))
    if owner:
        (deploy.HERE / "proof").mkdir()
        (deploy.HERE / "proof/PREPARATION_OWNER.json").write_text(json.dumps({
            "target": "owned-demo", "root": str(owned["root"])}))


def test_stop_marker_cancellation_checks_the_live_preparation_root(owned):
    write_cancel_owner(owned)
    owned["unit"]["ExecStart"] = owned["unit"]["ExecStart"].replace(str(owned["root"]), "/other")
    result = deploy.cancel_preparation()
    assert result["attempted"] and not result["stopped"]
    assert owned["mutations"] == []


def test_stop_marker_cancellation_requires_a_root_and_target_owner_receipt(owned):
    write_cancel_owner(owned, owner=False)
    assert deploy.cancel_preparation()["attempted"] is False
    assert owned["mutations"] == []


def test_stop_marker_cancellation_stops_the_exact_owned_preparation(owned):
    write_cancel_owner(owned)
    result = deploy.cancel_preparation()
    assert result["stopped"] and result["target"] == "owned-demo"
    assert owned["mutations"] == [("preparation", deploy.prepare_runtime.UNIT)]
