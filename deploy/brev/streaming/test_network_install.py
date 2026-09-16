"""Exercise protected service installation in temporary trees without host changes."""

from pathlib import Path
import shutil
import stat
from types import SimpleNamespace

import pytest

import install_network as installer


@pytest.fixture
def host_install(tmp_path, monkeypatch):
    root = tmp_path / "host"
    library = root / "usr/local/lib/paai-camera-network"
    units = root / "etc/systemd/system"
    library.mkdir(parents=True)
    units.mkdir(parents=True)
    monkeypatch.setattr(installer, "LIBRARY", library)
    monkeypatch.setattr(installer, "UNIT_PATH", units / installer.UNIT_NAME)
    monkeypatch.setattr(installer, "DROPIN", units / "paai-demo-docker.service.d/camera-network.conf")
    monkeypatch.setattr(installer.socket, "gethostname", lambda: "paai-demo-l40s")
    monkeypatch.setattr(installer.os, "getuid", lambda: 1000)
    monkeypatch.setattr(installer.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="ubuntu"))
    lstat, is_file = Path.lstat, Path.is_file
    state = SimpleNamespace(root=root, library=library, calls=[], unsafe={}, fail_enable=False)

    def fixture_lstat(path, *args, **kwargs):
        if path in state.unsafe:
            return state.unsafe[path]
        result = lstat(path, *args, **kwargs)
        if path == root or path.is_relative_to(root) or path in root.parents:
            mode = stat.S_IFDIR | 0o755 if stat.S_ISDIR(result.st_mode) else result.st_mode
            return SimpleNamespace(st_mode=mode, st_uid=0)
        return result

    def run(arguments, timeout=30):
        args = list(map(str, arguments))
        state.calls.append((args, timeout))
        if args[:4] == ["sudo", "-n", "install", "-d"]:
            destination = Path(args[-1])
            assert destination.is_relative_to(root)
            destination.mkdir(parents=True)
        elif args[:3] == ["sudo", "-n", "install"]:
            source, destination = map(Path, args[-2:])
            assert destination.is_relative_to(root)
            shutil.copyfile(source, destination)
            destination.chmod(0o644)
        elif args == ["sudo", "-n", "true"] or args == ["sudo", "-n", "systemctl", "daemon-reload"]:
            pass
        elif args == ["sudo", "-n", "systemctl", "enable", "--now", installer.UNIT_NAME]:
            if state.fail_enable:
                raise RuntimeError("fixture service start failed")
        elif args == ["systemctl", "is-active", installer.UNIT_NAME]:
            return "active"
        else:
            pytest.fail(f"Unexpected installation operation: {args}")
        return ""

    monkeypatch.setattr(Path, "lstat", fixture_lstat)
    monkeypatch.setattr(Path, "is_file", lambda path: True if path == Path("/usr/sbin/nft") else is_file(path))
    monkeypatch.setattr(installer, "run", run)
    return state


def test_identical_existing_service_is_reused_without_replacing_files(host_install):
    first = installer.install()
    assert first["status"] == "PASS_HOST_SETUP" and first["live_media_verified"] is False
    assert first["raw_rtsp_negative_probe_pending"] is True
    installed = {path: path.read_bytes() for path in host_install.root.rglob("*") if path.is_file()}
    host_install.calls.clear()
    assert installer.install() == first
    assert not any("install" in args or "daemon-reload" in args for args, _ in host_install.calls)
    assert {path: path.read_bytes() for path in installed} == installed


def test_different_existing_service_is_refused_and_kept_for_review(host_install):
    installer.install()
    installer.UNIT_PATH.write_text("[Service]\nExecStart=/unreviewed/program\n")
    host_install.calls.clear()
    with pytest.raises(ValueError, match="differs"):
        installer.install()
    assert "/unreviewed/program" in installer.UNIT_PATH.read_text()
    assert not any("systemctl" in args for args, _ in host_install.calls)


@pytest.mark.parametrize("mode", [stat.S_IFDIR | 0o775, stat.S_IFLNK | 0o777])
def test_replaceable_root_executable_ancestor_is_refused(host_install, mode):
    host_install.unsafe[host_install.library.parent] = SimpleNamespace(st_mode=mode, st_uid=0)
    with pytest.raises(ValueError, match="protected|directory|ancestor"):
        installer.install()
    assert not any("systemctl" in args for args, _ in host_install.calls)


def test_failed_service_start_cannot_return_setup_success(host_install):
    host_install.fail_enable = True
    with pytest.raises(RuntimeError, match="fixture service start failed"):
        installer.install()
