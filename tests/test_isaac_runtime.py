from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_newton_experience_is_found_in_a_wheel_layout(tmp_path):
    spec = importlib.util.find_spec("isaac_runtime")
    assert spec is not None, "Isaac needs runtime discovery independent of a developer source checkout"
    import isaac_runtime

    package = tmp_path / "site-packages/isaacsim"
    kit = package / "apps/isaacsim.exp.full.newton.kit"
    kit.parent.mkdir(parents=True)
    kit.write_text("[package]\nversion='6.1.0'\n")
    assert isaac_runtime.find_experience("newton", package_roots=[package]) == kit
    with pytest.raises(FileNotFoundError, match="newton"):
        isaac_runtime.find_experience("newton", package_roots=[tmp_path / "missing"])


def test_managed_runtime_requires_the_requested_release(monkeypatch):
    import isaac_runtime

    check = getattr(isaac_runtime, "installation_info", None)
    assert callable(check), "An existing python executable does not prove Isaac 6.1 is installed"
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "version", lambda _: "6.0.0.1")
    with pytest.raises(RuntimeError, match="6.1.0.0"):
        check()


def _kit(root, version="6.1.0"):
    path = root / "apps/isaacsim.exp.full.newton.kit"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[package]\nversion="{version}"\n')
    return path


def test_source_release_is_checked_without_requiring_pip_metadata(tmp_path, monkeypatch):
    import isaac_runtime

    release = tmp_path / "release"
    kit = _kit(release)
    monkeypatch.setenv("ISAACSIM_PATH", str(release))
    monkeypatch.setattr(sys, "executable", str(release / "kit/python/bin/python3"))
    def no_package(_):
        raise isaac_runtime.importlib.metadata.PackageNotFoundError("isaacsim")
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "version", no_package)
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "distribution", no_package)
    info = isaac_runtime.installation_info()
    assert info["version"] == "6.1.0"
    assert info["layout"] == "source"
    assert info["newton_experience"] == str(kit)


@pytest.mark.parametrize("version", ["6.0.0", "6.1.0.0", "6.2.0"])
def test_experience_refuses_wrong_package_version_even_if_file_exists(tmp_path, version):
    import isaac_runtime

    _kit(tmp_path, version)
    with pytest.raises(RuntimeError, match="6.1.0"):
        isaac_runtime.find_experience("newton", release=tmp_path, package_roots=[])


def test_explicit_source_cannot_fall_through_to_a_different_wheel(tmp_path):
    import isaac_runtime

    package = tmp_path / "wheel"
    _kit(package)
    with pytest.raises(FileNotFoundError):
        isaac_runtime.find_experience("newton", release=tmp_path / "incomplete-source", package_roots=[package])


def test_source_does_not_certify_an_unrelated_python(tmp_path, monkeypatch):
    import isaac_runtime

    _kit(tmp_path)
    monkeypatch.setenv("ISAACSIM_PATH", str(tmp_path))
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "version", lambda _: "6.1.0.0")
    with pytest.raises(RuntimeError, match="Python|python"):
        isaac_runtime.installation_info()


@pytest.mark.parametrize("kit_version", ["6.0.0", "6.1.0"])
def test_wheel_checks_its_own_kit_and_exact_metadata(tmp_path, monkeypatch, kit_version):
    import isaac_runtime

    monkeypatch.delenv("ISAACSIM_PATH", raising=False)
    package = tmp_path / "site-packages/isaacsim"
    kit = _kit(package, kit_version)
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "version", lambda _: "6.1.0.0")
    class Distribution:
        def locate_file(self, name):
            assert name == "isaacsim"
            return package
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "distribution", lambda _: Distribution())
    if kit_version != "6.1.0":
        with pytest.raises(RuntimeError, match="6.1.0"):
            isaac_runtime.installation_info()
    else:
        info = isaac_runtime.installation_info()
        assert info["version"] == "6.1.0.0"
        assert info["newton_experience"] == str(kit)


def test_source_python_can_be_a_packman_symlink_but_not_an_arbitrary_python(tmp_path, monkeypatch):
    import isaac_runtime

    release = tmp_path / "release"
    _kit(release)
    shared_python = tmp_path / "packman/python3"
    shared_python.parent.mkdir()
    shared_python.touch()
    link = release / "kit/python/bin/python3"
    link.parent.mkdir(parents=True)
    link.symlink_to(shared_python)
    monkeypatch.setenv("ISAACSIM_PATH", str(release))
    monkeypatch.setattr(sys, "executable", str(shared_python))
    def no_package(_):
        raise isaac_runtime.importlib.metadata.PackageNotFoundError("isaacsim")
    monkeypatch.setattr(isaac_runtime.importlib.metadata, "version", no_package)
    assert isaac_runtime.installation_info()["layout"] == "source"
