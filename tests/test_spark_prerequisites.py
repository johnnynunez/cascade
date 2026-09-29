"""Host preflight must report every missing prerequisite and fail closed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def host(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("spark_prerequisites", ROOT / "scripts/spark_prerequisites.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    browser = tmp_path / "chromium"
    browser.write_text("fixture")
    browser.chmod(0o755)
    values = {
        "getconf": "glibc 2.39",
        "git": "git-lfs/3.4.1 (linux arm64)",
        "nvidia-smi": "NVIDIA GB10, 580.142",
        "/fixture/nvcc": "Cuda compilation tools, release 13.0, V13.0.88",
    }
    missing = set()

    def which(command):
        if command in missing:
            return None
        if command == "chromium":
            return str(browser)
        return f"/fixture/{command}"

    def output(command):
        value = values[command[0]]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(module.shutil, "which", which)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: None)
    monkeypatch.setattr(module, "output", output)
    return SimpleNamespace(module=module, values=values, missing=missing, browser=browser)


def test_complete_host_succeeds_without_executing_browser(host, capsys):
    assert host.module.main() == 0
    output = capsys.readouterr()
    assert "PREREQUISITES_OK" in output.out
    assert "Chromium executable:" in output.out
    assert not output.err


def test_missing_components_are_all_reported_without_success(host, monkeypatch, capsys):
    host.values["getconf"] = "glibc 2.31"
    host.values["git"] = RuntimeError("command exited 1: git lfs version")
    host.values["nvidia-smi"] = OSError("not found")
    host.values["/fixture/nvcc"] = "Cuda compilation tools, release 12.9, V12.9"
    host.missing.update({"c++", "curl"})
    monkeypatch.setattr(host.module.shutil, "disk_usage", lambda _: SimpleNamespace(free=149 * 1024**3))
    assert host.module.main() == 2
    output = capsys.readouterr()
    for diagnostic in ("glibc 2.35", "Git LFS", "NVIDIA driver", "CUDA toolkit 13",
                       "Missing command: c++", "Missing command: curl", "150 GiB"):
        assert diagnostic in output.err
    assert "PREREQUISITES_OK" not in output.out


def test_browser_absence_cannot_pass_the_one_click_prerequisite(host, monkeypatch, capsys):
    monkeypatch.setenv("DISPLAY", ":0")
    host.browser.chmod(0o644)
    access = host.module.os.access
    monkeypatch.setattr(host.module.os, "access", lambda path, mode:
                        False if str(path) == "/snap/bin/chromium" else access(path, mode))
    assert host.module.main() == 2
    output = capsys.readouterr()
    assert "Chromium is missing" in output.err
    assert "PREREQUISITES_OK" not in output.out


@pytest.mark.parametrize("missing", ["chromium", "gio", "gnome-terminal"])
def test_headless_ssh_install_is_not_blocked_by_desktop_only_components(host, monkeypatch, capsys, missing):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    if missing == "chromium":
        host.browser.chmod(0o644)
        access = host.module.os.access
        monkeypatch.setattr(host.module.os, "access", lambda path, mode:
                            False if str(path) == "/snap/bin/chromium" else access(path, mode))
    else:
        host.missing.add(missing)
    assert host.module.main() == 0
    output = capsys.readouterr()
    assert "PREREQUISITES_OK" in output.out
    assert "WARNING:" in output.err and "desktop launcher only" in output.err
    assert "MISSING:" not in output.err


@pytest.mark.parametrize("missing", ["gio", "gnome-terminal"])
def test_desktop_session_still_requires_launcher_commands(host, monkeypatch, capsys, missing):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    host.missing.add(missing)
    assert host.module.main() == 2
    assert f"Missing command: {missing}" in capsys.readouterr().err


def test_unsupported_architecture_cannot_pass(host, monkeypatch, capsys):
    monkeypatch.setattr(host.module.platform, "machine", lambda: "x86_64")
    assert host.module.main() == 2
    assert "Linux aarch64" in capsys.readouterr().err
