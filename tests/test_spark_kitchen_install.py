"""Kitchen readiness must gate the public Spark entry points."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from test_spark_install import support_module


LFS_POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 123\n"


@pytest.fixture
def robot_assets(tmp_path, monkeypatch):
    support = support_module()
    assets = {
        "assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda": b"robot scene",
        "assets/usd/RS-rebot-dev-arm/geometry.usd": b"robot geometry",
        "assets/urdf/00-arm-rs_asm-v3/urdf/00-arm-rs_asm-v3.urdf": b"robot URDF",
        "assets/urdf/00-arm-rs_asm-v3/meshes/base_link.STL": b"robot mesh",
    }
    for name, data in assets.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    state = SimpleNamespace(repo=tmp_path, support=support, assets=assets, pulls=[], hydrate=True, status=0)

    def run(command, **kwargs):
        if command[:4] == ["git", "-C", str(tmp_path), "ls-files"]:
            assert command[-2:] == list(support.ROBOT_ASSET_DIRS)
            return SimpleNamespace(returncode=0, stdout=("\0".join(assets) + "\0").encode())
        if command[:5] == ["git", "-C", str(tmp_path), "lfs", "pull"]:
            state.pulls.append(command)
            if state.hydrate and not state.status:
                for name, data in assets.items():
                    if (tmp_path / name).read_bytes() == LFS_POINTER:
                        (tmp_path / name).write_bytes(data)
            return SimpleNamespace(returncode=state.status, stdout="", stderr="fixture access denied" if state.status else "")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(support.subprocess, "run", run)
    monkeypatch.setattr(support.shutil, "which", lambda name: "/fixture/git-lfs" if name == "git-lfs" else None)
    return state


@pytest.mark.parametrize("payload", ["assets/usd/RS-rebot-dev-arm/geometry.usd",
                                    "assets/urdf/00-arm-rs_asm-v3/meshes/base_link.STL"])
def test_robot_assets_hydrate_only_required_subtrees_and_reuse_valid_files(robot_assets, payload):
    h = robot_assets
    (h.repo / payload).write_bytes(LFS_POINTER)
    valid_before = {name: (h.repo / name).stat().st_mtime_ns for name in h.assets if name != payload}

    assert h.support.main(["robot-assets", "--repo", str(h.repo)]) == 0
    assert h.pulls == [["git", "-C", str(h.repo), "lfs", "pull",
                       "--include=assets/usd/RS-rebot-dev-arm/**,assets/urdf/00-arm-rs_asm-v3/**", "--exclude="]]
    assert h.support.scene_problems(h.repo) == []
    assert all((h.repo / name).read_bytes() == data for name, data in h.assets.items())
    assert valid_before == {name: (h.repo / name).stat().st_mtime_ns for name in valid_before}
    assert h.support.main(["robot-assets", "--repo", str(h.repo)]) == 0
    assert len(h.pulls) == 1


def test_robot_assets_missing_git_lfs_reports_actionable_prerequisite(robot_assets, monkeypatch, capsys):
    h = robot_assets
    payload = h.repo / "assets/usd/RS-rebot-dev-arm/geometry.usd"
    payload.write_bytes(LFS_POINTER)
    monkeypatch.setattr(h.support.shutil, "which", lambda name: None)

    assert h.support.main(["robot-assets", "--repo", str(h.repo)]) == 1
    assert "Install git-lfs" in capsys.readouterr().err
    assert payload.read_bytes() == LFS_POINTER
    assert h.pulls == []


@pytest.mark.parametrize("failure", ["download", "still_pointer"])
def test_robot_assets_recheck_hydration_before_later_asset_downloads(robot_assets, monkeypatch, failure):
    h = robot_assets
    (h.repo / "assets/usd/RS-rebot-dev-arm/geometry.usd").write_bytes(LFS_POINTER)
    h.hydrate = False
    h.status = 1 if failure == "download" else 0
    monkeypatch.setattr(h.support, "ensure_model_asset", lambda *args: pytest.fail("must finish robot hydration first"))
    monkeypatch.setattr(h.support, "prepare_kitchen", lambda *args: pytest.fail("must finish robot hydration first"))

    with pytest.raises(RuntimeError, match="Robot Git LFS download failed" if h.status else "unresolved Git LFS pointer"):
        h.support.prepare_assets(h.repo)


def test_install_check_reports_robot_pointers_without_hydrating(robot_assets, monkeypatch, capsys):
    h = robot_assets
    payload = h.repo / "assets/urdf/00-arm-rs_asm-v3/meshes/base_link.STL"
    payload.write_bytes(LFS_POINTER)
    package = h.repo / ".openclaw-cli/lib/node_modules/openclaw/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version":"2026.9.3"}')
    python = h.repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(h.support, "MODEL_ASSETS", {})
    monkeypatch.setattr(h.support, "kitchen_problems", lambda repo: [])

    assert h.support.main(["check", "--repo", str(h.repo)]) == 3
    assert "unresolved Git LFS pointer" in capsys.readouterr().out
    assert payload.read_bytes() == LFS_POINTER
    assert h.pulls == []


@pytest.mark.parametrize("download_fails", [False, True])
def test_model_pointer_uses_validated_release_download_and_survives_failure(tmp_path, monkeypatch, download_fails):
    support = support_module()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("archive/data.pkl", "fixture model")
    content = stream.getvalue()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(content).hexdigest()}\nsize {len(content)}\n".encode()
    dest = tmp_path / "detector.pt"
    dest.write_bytes(pointer)
    calls = []

    def fetch(url, stage, force):
        calls.append(url)
        assert dest.read_bytes() == pointer
        stage.write_bytes(content[:10] if download_fails else content)

    monkeypatch.setattr(support, "fetch", fetch)
    if download_fails:
        with pytest.raises(RuntimeError, match="incomplete/invalid release asset"):
            support.ensure_model_asset("https://fixture.invalid/model.pt", dest, len(content))
        assert dest.read_bytes() == pointer
    else:
        support.ensure_model_asset("https://fixture.invalid/model.pt", dest, len(content))
        assert dest.read_bytes() == content
        support.ensure_model_asset("https://fixture.invalid/model.pt", dest, len(content))
    assert len(calls) == 1
    assert not dest.with_suffix(".pt.download").exists()


def test_spark_check_reports_missing_kitchen(tmp_path, monkeypatch, capsys):
    support = support_module()
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    package = tmp_path / ".openclaw-cli/lib/node_modules/openclaw/package.json"
    package.parent.mkdir(parents=True)
    package.write_text('{"version":"2026.9.3"}')
    monkeypatch.setattr(support, "MODEL_ASSETS", {})
    monkeypatch.setattr(support, "scene_problems", lambda repo: [])
    monkeypatch.setattr(support, "kitchen_problems", lambda repo: [
        "missing kitchen asset: demo/scene/assets/background.usda"
    ])
    monkeypatch.setattr(support.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))

    assert support.main(["check", "--repo", str(tmp_path), "--profile", "spark"]) == 3
    output = capsys.readouterr().out
    assert "missing kitchen asset: demo/scene/assets/background.usda" in output
    assert "READY" not in output


def test_spark_launch_refuses_incomplete_kitchen_before_starting_services(tmp_path, monkeypatch):
    support = support_module()
    monkeypatch.delenv("CASCADE_LAUNCH_STATE", raising=False)
    monkeypatch.setattr(support, "eula_accepted", lambda repo: True)
    monkeypatch.setattr(support, "kitchen_problems", lambda repo: ["missing kitchen asset"])
    receipt = tmp_path / "runs/.launch/profile-cascade-demo/proof.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"verified": true}')
    calls = []
    monkeypatch.setattr(support.subprocess, "Popen", lambda *a, **kw: calls.append(a))

    with pytest.raises(RuntimeError, match="missing kitchen asset"):
        support.launch(tmp_path, "spark", "qwen")
    assert calls == []
    assert json.loads(receipt.read_text())["verified"] is False


def test_spark_asset_preparation_includes_verified_kitchen(tmp_path, monkeypatch):
    support = support_module()
    monkeypatch.setattr(support, "scene_problems", lambda repo: [])
    monkeypatch.setattr(support, "MODEL_ASSETS", {})
    prepared: list[Path] = []
    monkeypatch.setattr(support, "prepare_kitchen", prepared.append)

    support.prepare_assets(tmp_path)
    assert prepared == [tmp_path]
