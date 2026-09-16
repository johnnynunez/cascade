"""Admission replay and filesystem attack fixtures; no host firewall access."""

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import network_guard as guard


IDENTITY = {"boot_id": "641e886b-f851-467a-b817-5ac2b075aefb", "netns_inode": 4026531840}


def receipt():
    return {"schema": 1, "status": "PASS", "backend": "nftables",
            "ports": [8554, 8555, 8556], "rules_sha256": guard.RULESET_SHA256,
            "checked_monotonic": 100.0, **IDENTITY}


@pytest.mark.parametrize("now", [100.0, 115.0])
def test_current_receipt_is_bound_to_host_namespace_and_exact_ports(now):
    value = receipt()
    assert guard.validate_receipt(value, (8554, 8555, 8556), now=now, identity=IDENTITY) == value


@pytest.mark.parametrize("change", [
    {"boot_id": "a-different-boot"}, {"netns_inode": 4026532999},
    {"ports": [8554, 8555, 8889]}, {"rules_sha256": "0" * 64},
    {"backend": "iptables"}, {"status": "PENDING"},
])
def test_copied_or_other_profile_receipt_cannot_admit_a_listener(change):
    value = receipt() | change
    with pytest.raises(ValueError, match="host and profile"):
        guard.validate_receipt(value, (8554, 8555, 8556), now=101.0, identity=IDENTITY)


@pytest.mark.parametrize("ports", [(8554, 8555), (8554, 8555, 8556, 8557), (8556, 8555, 8554)])
def test_listener_port_drift_requires_new_admission(ports):
    with pytest.raises(ValueError, match="three admitted RTSP ports"):
        guard.validate_receipt(receipt(), ports, now=101.0, identity=IDENTITY)


@pytest.mark.parametrize("checked", [84.999, 100.001, float("nan"), float("inf"),
                                      float("-inf"), True, "100", None])
def test_stale_future_or_nonfinite_heartbeat_cannot_authorize_video(checked):
    value = receipt() | {"checked_monotonic": checked}
    with pytest.raises(ValueError, match="stale or invalid"):
        guard.validate_receipt(value, (8554, 8555, 8556), now=100.0, identity=IDENTITY)


@pytest.fixture
def protected_files(tmp_path, monkeypatch):
    directory = tmp_path / "run" / "paai-camera-network"
    directory.mkdir(parents=True)
    admission = directory / "admission.json"
    admission.write_text(json.dumps(receipt()))
    admission.chmod(0o644)
    monkeypatch.setattr(guard, "DIRECTORY", directory)
    monkeypatch.setattr(guard, "ADMISSION", admission)
    lstat, fstat = Path.lstat, os.fstat
    directory_info = {}
    state = SimpleNamespace(directory=directory, admission=admission,
                            readonly=True, file_uid=0, directory_info=directory_info)

    def fixture_lstat(path, *args, **kwargs):
        if path == directory or path in directory.parents:
            return directory_info.get(path, SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0))
        return lstat(path, *args, **kwargs)

    def fixture_fstat(fd):
        info = list(fstat(fd))
        info[4] = state.file_uid
        return os.stat_result(info)

    monkeypatch.setattr(Path, "lstat", fixture_lstat)
    monkeypatch.setattr(os, "fstat", fixture_fstat)
    monkeypatch.setattr(os, "statvfs", lambda path: SimpleNamespace(
        f_flag=os.ST_RDONLY if state.readonly else 0))
    return state


def test_readonly_root_owned_receipt_can_be_read_and_checked(protected_files, monkeypatch):
    monkeypatch.setattr(guard, "host_identity", lambda: deepcopy(IDENTITY))
    monkeypatch.setattr(guard.time, "monotonic", lambda: 105.0)
    assert guard.verify_isolation((8554, 8555, 8556)) == receipt()


def test_writable_container_mount_cannot_attest_host_rules(protected_files):
    protected_files.readonly = False
    with pytest.raises(ValueError, match="mounted read-only"):
        guard.read_protected_receipt()


@pytest.mark.parametrize("location,mode,uid", [
    ("directory", stat.S_IFDIR | 0o775, 0),
    ("parent", stat.S_IFDIR | 0o777, 0),
    ("directory", stat.S_IFDIR | 0o755, 1000),
    ("parent", stat.S_IFLNK | 0o777, 0),
])
def test_replaceable_or_redirected_receipt_ancestry_is_rejected(protected_files, location, mode, uid):
    path = protected_files.directory if location == "directory" else protected_files.directory.parent
    protected_files.directory_info[path] = SimpleNamespace(st_mode=mode, st_uid=uid)
    with pytest.raises(ValueError, match="directory must be protected"):
        guard.read_protected_receipt()


@pytest.mark.parametrize("kind", ["unprivileged-owner", "group-write", "symlink", "fifo", "empty", "oversize"])
def test_untrusted_receipt_objects_are_not_read_as_admission(protected_files, kind):
    path = protected_files.admission
    if kind == "unprivileged-owner":
        protected_files.file_uid = 1000
    elif kind == "group-write":
        path.chmod(0o664)
    elif kind == "symlink":
        target = path.with_name("forged.json")
        path.rename(target)
        path.symlink_to(target)
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "empty":
        path.write_bytes(b"")
    else:
        path.write_bytes(b" " * 8193)
    with pytest.raises((ValueError, OSError)):
        guard.read_protected_receipt()
