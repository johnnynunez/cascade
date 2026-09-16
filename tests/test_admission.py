"""Admission must fail before remote commands or unrelated host mutations."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from types import SimpleNamespace

import pytest


CAMPAIGN = Path(__file__).resolve().parents[1] / "deploy/brev"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, CAMPAIGN / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


access = load("brev_admission_access", "access_gate.py")
preflight = load("preflight", "preflight.py")
host = load("brev_admission_host", "host_setup.py")
PROFILE = json.loads((CAMPAIGN / "profiles/brev-rtx6000.json").read_text())


@pytest.fixture
def private_access(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    key = home / "brev.pem"
    key.write_text("This is test fixture data, never a private key.\n")
    key.chmod(0o600)
    campaign = tmp_path / "campaign"
    monkeypatch.setattr(access, "HERE", campaign)
    monkeypatch.setattr(access, "STOP", tmp_path / "STOP")
    monkeypatch.setattr(access, "KEY", key)
    monkeypatch.setattr(access, "PRIVATE_CONFIG", home / ".ssh/config.d/rtx6000.conf")
    monkeypatch.setattr(access, "MAIN_CONFIG", home / ".ssh/config")
    monkeypatch.setattr(access, "WRAPPER", home / "never-execute-wrapper")
    return home, key


def tailnet_result(status):
    return SimpleNamespace(returncode=0, stdout=json.dumps(status), stderr="")


def test_missing_target_never_invokes_wrapper_or_public_route(private_access, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command == ["tailscale", "status", "--json"]
        return tailnet_result({"BackendState": "Running", "Peer": {
            "old": {"HostName": "paai-demo", "TailscaleIPs": ["100.64.0.10"]}
        }})

    monkeypatch.setattr(access, "run", run)
    result = access.probe()
    assert result["status"] == "WAITING_FOR_ACCESS"
    assert result["remote_command_executed"] is False
    assert len(calls) == 1
    assert not access.PRIVATE_CONFIG.exists()


@pytest.mark.parametrize("state,address", [("NeedsMachineAuth", "100.64.0.10"),
                                           ("Running", "203.0.113.10")])
def test_unapproved_or_public_peer_never_invokes_wrapper(private_access, monkeypatch, state, address):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "tailscale"
        return tailnet_result({"BackendState": state, "Peer": {
            "target": {"HostName": "paai-demo-rtx6000", "TailscaleIPs": [address]}
        }})

    monkeypatch.setattr(access, "run", run)
    result = access.probe()
    assert result["status"] == "WAITING_FOR_ACCESS"
    assert result["remote_command_executed"] is False
    assert len(calls) == 1
    assert not access.PRIVATE_CONFIG.exists()


@pytest.mark.parametrize("problem", ["mode", "symlink", "owner"])
def test_key_security_rejects_before_configuration_writes(private_access, monkeypatch, problem):
    home, key = private_access
    if problem == "mode":
        key.chmod(0o644)
    elif problem == "symlink":
        link = home / "key-link"
        link.symlink_to(key)
        monkeypatch.setattr(access, "KEY", link)
    else:
        monkeypatch.setattr(access.os, "getuid", lambda: key.lstat().st_uid + 1)
    with pytest.raises(ValueError, match="0600|regular file"):
        access.configure("100.64.0.10")
    assert not access.PRIVATE_CONFIG.exists()
    assert not access.MAIN_CONFIG.exists()


def test_private_alias_preserves_other_hosts_and_key(private_access):
    home, key = private_access
    access.MAIN_CONFIG.parent.mkdir()
    previous = "Host existing\n    HostName existing.invalid\n"
    access.MAIN_CONFIG.write_text(previous)
    key_bytes = key.read_bytes()
    access.configure("100.64.0.10")
    include = f"Include {access.PRIVATE_CONFIG}\n"
    assert access.MAIN_CONFIG.read_text() == include + previous
    config = access.PRIVATE_CONFIG.read_text()
    assert "HostName 100.64.0.10\n" in config
    assert "    User ubuntu\n" in config
    assert f"    IdentityFile {key}\n" in config
    assert key_bytes.decode() not in config
    assert key.read_bytes() == key_bytes
    for path in (access.MAIN_CONFIG, access.PRIVATE_CONFIG):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(access.PRIVATE_CONFIG.parent.stat().st_mode) == 0o700
    before = access.MAIN_CONFIG.stat().st_mtime_ns
    access.configure("100.64.0.10")
    assert access.MAIN_CONFIG.stat().st_mtime_ns == before


def test_config_symlink_never_overwrites_another_file(private_access):
    home, key = private_access
    access.PRIVATE_CONFIG.parent.mkdir(parents=True)
    unrelated = home / "unrelated-config"
    unrelated.write_text("preserve this file\n")
    access.PRIVATE_CONFIG.symlink_to(unrelated)
    with pytest.raises(ValueError, match="regular file"):
        access.configure("100.64.0.10")
    assert unrelated.read_text() == "preserve this file\n"
    assert not access.MAIN_CONFIG.exists()


@pytest.mark.parametrize("row", [
    "NVIDIA RTX PRO 6000, 96000, 10, 95990, 12.0, 595.91.07",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, nan, 97000, 12.0, 595.91.07",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 10, inf, 12.0, 595.91.07",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 0, 99999, 12.0, 595.91.07",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 40000, 57887, 12.0, 595.91.07",
])
def test_gpu_refuses_wrong_device_invalid_telemetry_and_start_reserve(row):
    with pytest.raises(ValueError):
        preflight.validate_gpu(PROFILE, row)


def test_running_gpu_still_requires_measured_headroom():
    with pytest.raises(ValueError, match="reserve"):
        preflight.validate_gpu(PROFILE, "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 90000, 7887, 12.0, 595.91.07",
                               running=True)
    profile = deepcopy(PROFILE)
    profile["accelerator"]["minimum_headroom_mib"] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        preflight.validate_gpu(profile, "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 1000, 96887, 12.0, 595.91.07",
                               running=True)


@pytest.fixture
def volume(tmp_path, monkeypatch):
    mount = tmp_path / "nvme"
    mount.mkdir()
    root = mount / "paai-demo"
    profile = deepcopy(PROFILE)
    profile["storage"].update(mount=str(mount), root=str(root))
    device_overrides = {}
    owner_overrides = {}
    original_stat = Path.stat

    def virtual_stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == Path("/") or path.is_relative_to(mount):
            fields = list(result)
            fields[2] = device_overrides.get(path, 11 if path == Path("/") else 42)
            if path.is_relative_to(mount):
                fields[4] = owner_overrides.get(path, 1000)
            result = os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", virtual_stat)
    monkeypatch.setattr(preflight.os.path, "ismount", lambda path: Path(path) == mount)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=(400 if Path(path) == mount else 20) * preflight.GIB))

    def findmnt(command, **kwargs):
        assert command[0] == "findmnt"
        return json.dumps({"filesystems": [{"target": str(mount), "source": "fixture-nvme",
                                            "fstype": "ext4"}]})

    monkeypatch.setattr(preflight, "output", findmnt)
    return SimpleNamespace(profile=profile, mount=mount, root=root, devices=device_overrides,
                           owners=owner_overrides, findmnt=findmnt)


def test_root_backed_volume_is_refused(volume):
    volume.devices[volume.mount] = 11
    with pytest.raises(ValueError, match="separately from the root"):
        preflight.validate_storage(volume.profile)


def test_root_backed_bind_mount_inside_nvme_is_refused(volume):
    volume.root.mkdir()
    volume.devices[volume.root] = 11
    with pytest.raises(ValueError, match="data-volume filesystem"):
        preflight.validate_storage(volume.profile)


def test_preflight_returns_runtime_uid_and_gid(volume, monkeypatch):
    monkeypatch.setattr(preflight.socket, "gethostname", lambda: "brev-fixture")
    monkeypatch.setattr(preflight.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(preflight.os, "getuid", lambda: 1000)
    monkeypatch.setattr(preflight.os, "getgid", lambda: 1001)
    monkeypatch.setattr(preflight.os, "cpu_count", lambda: 8)
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
                        "MemTotal: 65000000 kB\n" if path == Path("/proc/meminfo")
                        else original_read(path, *args, **kwargs))

    def output(command, **kwargs):
        if command[0] == "findmnt":
            return volume.findmnt(command)
        if command == ["id", "-un"]:
            return "ubuntu"
        if command == ["sudo", "-n", "true"]:
            return ""
        if command[0] == "nvidia-smi":
            return "" if "--query-compute-apps" in command[1] else (
                "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 100, 97787, 12.0, 595.91.07")
        if command[:2] == ["docker", "version"]:
            return "29.0.0"
        if command[:2] == ["docker", "info"]:
            return "/fixture/default-docker"
        pytest.fail(f"Unexpected preflight command: {command[0]}")

    monkeypatch.setattr(preflight, "output", output)
    result = preflight.check(volume.profile)
    assert result["status"] == "PASS"
    assert (result["uid"], result["gid"]) == (1000, 1001)
    assert result["dedicated_docker_required"] is True


@pytest.fixture
def host_calls(volume, tmp_path, monkeypatch):
    unit_directory = tmp_path / "systemd"
    unit_directory.mkdir()
    monkeypatch.setattr(host, "UNIT_DIRECTORY", unit_directory)
    home = tmp_path / "ubuntu-home"
    home.mkdir()
    monkeypatch.setattr(host, "HOME_DIRECTORY", home)
    monkeypatch.setattr(host.socket, "gethostname", lambda: "brev-fixture")
    monkeypatch.setattr(host.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(host.os, "getuid", lambda: 1000)
    monkeypatch.setattr(host.os, "getgid", lambda: 1000)
    calls = []

    def run(command, **kwargs):
        command = [str(part) for part in command]
        calls.append(command)
        if command == ["sudo", "-n", "true"]:
            return ""
        if command == ["id", "-un"]:
            return "ubuntu"
        if command[:4] == ["sudo", "-n", "install", "-d"]:
            assert command[command.index("-o") + 1] == "1000"
            assert command[command.index("-g") + 1] == "1000"
            Path(command[-1]).mkdir(mode=0o750, parents=True)
            return ""
        if command[:4] == ["sudo", "-n", "install", "-m"]:
            shutil.copyfile(command[-2], command[-1])
            return ""
        if command[:3] == ["sudo", "-n", "systemctl"]:
            assert command[3:] in (["daemon-reload"], ["enable", "--now", host.UNIT])
            return ""
        if command[:4] == ["docker", "--host", host.DOCKER_SOCKET, "info"]:
            return str(volume.root / "docker")
        pytest.fail(f"Unrelated or unexpected host command: {command[0]}")

    monkeypatch.setattr(host, "run", run)
    return calls


def test_host_setup_refuses_wrong_volume_before_mutation(volume, host_calls):
    volume.devices[volume.mount] = 11
    with pytest.raises(ValueError, match="separately from root"):
        host.install(volume.profile)
    assert host_calls == []
    assert not volume.root.exists()


def test_host_setup_refuses_existing_wrong_owner(volume, host_calls):
    volume.root.mkdir()
    volume.owners[volume.root] = 1001
    with pytest.raises(ValueError, match="owned by the ubuntu"):
        host.install(volume.profile)
    assert all("install" not in command and "systemctl" not in command for command in host_calls)
    assert not (volume.root / "deployment").exists()


def test_host_setup_refuses_unsafe_resolved_alias_before_mutation(volume, host_calls):
    target = volume.mount / "unsafe%unit"
    target.mkdir()
    volume.root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe for systemd"):
        host.install(volume.profile)
    assert host_calls == []
    assert not (target / "deployment").exists()


def test_host_setup_preserves_different_existing_unit(volume, host_calls):
    unit = host.UNIT_DIRECTORY / host.UNIT
    existing = "[Service]\nExecStart=/preserve/existing/daemon\n"
    unit.write_text(existing)
    with pytest.raises(ValueError, match="unit differs"):
        host.install(volume.profile)
    assert unit.read_text() == existing
    assert not volume.root.exists()
    assert all("install" not in command and "systemctl" not in command for command in host_calls)


def test_host_setup_isolates_storage_and_preserves_existing_daemon(volume, host_calls, tmp_path):
    default = tmp_path / "default-docker-daemon.json"
    default.write_text('{"data-root":"/var/lib/docker"}\n')
    result = host.install(volume.profile)
    assert result["status"] == "PASS"
    assert result["existing_docker_daemon_modified"] is False
    unit_path = host.UNIT_DIRECTORY / host.UNIT
    unit = unit_path.read_text()
    assert f"ExecStartPre=/usr/bin/mountpoint -q {volume.mount}\n" in unit
    assert f"RequiresMountsFor={volume.root}\n" in unit
    config_path = volume.root / "deployment/docker-daemon.json"
    config = json.loads(config_path.read_text())
    assert config["data-root"] == str(volume.root / "docker")
    assert config["hosts"] == [host.DOCKER_SOCKET]
    assert config["bridge"] == "none"
    assert config["iptables"] is False and config["ip6tables"] is False
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    unit_before = unit_path.read_bytes()
    config_before = config_path.read_bytes()
    first_commands = len(host_calls)
    assert host.install(volume.profile)["status"] == "PASS"
    assert unit_path.read_bytes() == unit_before
    assert config_path.read_bytes() == config_before
    assert all("install" not in command and "restart" not in command
               for command in host_calls[first_commands:])
    assert default.read_text() == '{"data-root":"/var/lib/docker"}\n'


def test_home_link_is_created_and_reused_without_replacement(volume, host_calls):
    volume.root.mkdir()
    created = host.ensure_home_link(volume.root)
    link = Path(created["home_link"])
    inode = link.lstat().st_ino
    assert link == host.HOME_DIRECTORY / "paai-demo"
    assert link.resolve() == volume.root
    reused = host.ensure_home_link(volume.root)
    assert reused["home_link_status"] == "reused"
    assert link.lstat().st_ino == inode


def test_home_link_uses_alternate_and_preserves_existing_directory(volume, host_calls):
    volume.root.mkdir()
    existing = host.HOME_DIRECTORY / "paai-demo"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("Keep the existing project.\n")
    result = host.ensure_home_link(volume.root)
    assert Path(result["home_link"]) == host.HOME_DIRECTORY / "cascade-demo"
    assert Path(result["home_link"]).resolve() == volume.root
    assert result["home_link_conflicts"] == [str(existing)]
    assert sentinel.read_text() == "Keep the existing project.\n"


def test_home_link_reports_conflicts_without_overwriting(volume, host_calls):
    volume.root.mkdir()
    primary = host.HOME_DIRECTORY / "paai-demo"
    alternate = host.HOME_DIRECTORY / "cascade-demo"
    primary.symlink_to(host.HOME_DIRECTORY / "different-target", target_is_directory=True)
    alternate.write_text("Preserve this launcher.\n")
    result = host.ensure_home_link(volume.root)
    assert result["home_link"] is None
    assert result["home_link_status"] == "conflict"
    assert result["home_link_conflicts"] == [str(primary), str(alternate)]
    assert primary.is_symlink()
    assert alternate.read_text() == "Preserve this launcher.\n"


def test_host_setup_receipt_includes_home_link(volume, host_calls):
    result = host.install(volume.profile)
    assert result["home_link_status"] == "created"
    assert Path(result["home_link"]).resolve() == volume.root
