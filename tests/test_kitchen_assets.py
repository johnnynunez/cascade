"""Kitchen archive integrity and extraction boundaries; no simulator required."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kitchen_assets", ROOT / "scripts/kitchen_assets.py")
kitchen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kitchen)


@pytest.fixture
def bundle(tmp_path):
    repo = tmp_path / "repo"
    manifest = repo / "deploy/brev/bundle_assets.json"
    manifest.parent.mkdir(parents=True)
    content = {
        "demo/scene/assets/background.usda": b"kitchen background",
        "demo/vendor-kitchen/LICENSE.txt": b"Lightwheel CC BY-NC 4.0",
    }
    manifest.write_text(json.dumps({"files": {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name, data in content.items()
    }}))
    return repo, content


@pytest.fixture
def archive_at(monkeypatch):
    def build(path, content, *, extra=None):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in content.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            if extra is not None:
                archive.addfile(extra, io.BytesIO(b""))
        monkeypatch.setattr(kitchen, "KITCHEN_ARCHIVE_BYTES", path.stat().st_size)
        monkeypatch.setattr(kitchen, "KITCHEN_ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
        return path
    return build


def test_manifest_pins_all_kitchen_files_and_notices():
    files = kitchen.kitchen_manifest(ROOT)
    assert len(files) == 166
    assert sum(entry["size_bytes"] for entry in files.values()) == 1_242_964_417
    assert {"demo/vendor-kitchen/LICENSE.txt", "demo/vendor-kitchen/SOURCE_README.md",
            "demo/scene/assets/orange70_LICENSE.txt"} <= files.keys()


def test_install_local_archive_and_reuse_without_download(bundle, tmp_path, monkeypatch, archive_at):
    repo, content = bundle
    archive = archive_at(tmp_path / "kitchen.tar.gz", content)
    monkeypatch.setenv("CASCADE_KITCHEN_ARCHIVE", str(archive))
    kitchen.prepare_kitchen(repo)
    assert kitchen.kitchen_problems(repo) == []
    for name, data in content.items():
        assert (repo / name).read_bytes() == data
    archive.unlink()
    kitchen.prepare_kitchen(repo)


def test_missing_and_same_size_corrupt_files_fail_check(bundle, tmp_path, archive_at):
    repo, content = bundle
    assert len(kitchen.kitchen_problems(repo)) == 2
    kitchen.prepare_kitchen(repo, archive=archive_at(tmp_path / "kitchen.tar.gz", content))
    asset = repo / next(iter(content))
    asset.write_bytes(b"x" * asset.stat().st_size)
    assert "checksum mismatch" in kitchen.kitchen_problems(repo)[0]


@pytest.mark.parametrize("fault", ["missing", "size", "checksum", "truncated"])
def test_invalid_archive_does_not_publish_any_files(bundle, tmp_path, fault, archive_at):
    repo, content = bundle
    changed = dict(content)
    name = next(iter(changed))
    if fault == "missing":
        del changed[name]
    elif fault == "size":
        changed[name] += b"x"
    elif fault == "checksum":
        changed[name] = b"x" * len(changed[name])
    archive = archive_at(tmp_path / "kitchen.tar.gz", changed)
    if fault == "truncated":
        archive.write_bytes(archive.read_bytes()[:50])
    with pytest.raises(RuntimeError, match="Kitchen installation failed"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert not (repo / "demo").exists()
    assert not list(repo.glob(".kitchen-install-*"))


@pytest.mark.parametrize("name,kind", [
    ("../escaped", tarfile.REGTYPE),
    ("/tmp/escaped", tarfile.REGTYPE),
    ("demo/scene/assets/../escaped", tarfile.REGTYPE),
    ("demo/vendor-kitchen/extra", tarfile.REGTYPE),
    ("demo/scene/assets/background.usda", tarfile.SYMTYPE),
    ("demo/scene/assets/background.usda", tarfile.LNKTYPE),
    ("demo/scene/assets/background.usda", tarfile.REGTYPE),
])
def test_unexpected_links_and_duplicate_members_are_rejected(bundle, tmp_path, name, kind, archive_at):
    repo, content = bundle
    extra = tarfile.TarInfo(name)
    extra.type = kind
    extra.linkname = "../../outside"
    archive = archive_at(tmp_path / "kitchen.tar.gz", content, extra=extra)
    with pytest.raises(RuntimeError, match="unsafe kitchen archive member"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert not (repo / "demo").exists()
    assert not (tmp_path / "escaped").exists()


def test_existing_symlink_cannot_redirect_install(bundle, tmp_path, archive_at):
    repo, content = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "demo").symlink_to(outside, target_is_directory=True)
    archive = archive_at(tmp_path / "kitchen.tar.gz", content)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert list(outside.iterdir()) == []


def test_default_release_download_installs_verified_files(bundle, tmp_path, monkeypatch, archive_at):
    repo, content = bundle
    archive = archive_at(tmp_path / "kitchen.tar.gz", content)
    calls = []

    def download(request, timeout):
        calls.append((request.full_url, timeout))
        return archive.open("rb")

    monkeypatch.delenv("CASCADE_KITCHEN_ARCHIVE", raising=False)
    monkeypatch.setattr(kitchen.urllib.request, "urlopen", download)
    kitchen.prepare_kitchen(repo)
    assert calls == [(kitchen.KITCHEN_URL, 60)]
    assert "/releases/download/kitchen-v1/" in calls[0][0]
    assert kitchen.kitchen_problems(repo) == []


@pytest.mark.parametrize("fault", ["size", "checksum"])
def test_archive_identity_fails_before_extraction(bundle, tmp_path, archive_at, monkeypatch, fault):
    repo, content = bundle
    archive = archive_at(tmp_path / "kitchen.tar.gz", content)
    if fault == "size":
        archive.write_bytes(archive.read_bytes() + b"x")
    else:
        archive.write_bytes(b"x" * archive.stat().st_size)
    monkeypatch.setattr(kitchen, "_unpack_verified", lambda *args: pytest.fail("unverified archive was unpacked"))
    with pytest.raises(RuntimeError, match="Kitchen archive (size|SHA-256) mismatch"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert not (repo / "demo").exists()
