"""Pinned release admission, hostile archives, and scene identity; no GPU needed."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kitchen_assets", ROOT / "scripts/kitchen_assets.py")
kitchen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kitchen)
import scene_identity as identity


@pytest.fixture
def source(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    for name in identity.SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Original fixture " + name + "\n")
    config = repo / "demo/scene/kitchen_config.json"
    config.write_text(json.dumps({"version": 1, "scene_name": identity.SCENE_NAME, "props": [
        {"name": "tomato_can", "asset": "props/tomato_can.usda"},
        {"name": "lemon", "asset": "props/lemon.usda"},
        {"name": "orange", "visual": "procedural_orange"}]}))
    for name, raw in {"cocina_asier.usdc": b"reviewed room", "geometry-audit.json": b"{}",
                      "NOTICE.md": b"Owner authorization and CC0 notices", "sources.json": b"{}",
                      "textures/wood.jpg": b"CC0 fixture"}.items():
        path = repo / identity.BUNDLE_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    kitchen.package_bundle(repo, tmp_path / "release.zip")
    monkeypatch.setattr(identity, "__file__", str(repo / "demo/scene_identity.py"))
    return repo, config


def test_shipped_sources_are_complete_and_verified():
    assert set(kitchen.kitchen_manifest(ROOT)) == set(identity.SOURCE_FILES)
    assert identity.source_problems(ROOT) == []
    # CI explicitly installs the public pinned release before scene-dependent
    # tests; this source test remains usable in an offline source-only clone.
    identity.bundle_manifest(ROOT)


def test_offline_prepare_leaves_all_bytes_and_mtimes_unchanged(source, monkeypatch):
    repo, _ = source
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in repo.rglob("*") if p.is_file()}
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network access"))
    kitchen.prepare_kitchen(repo)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("fault", ["missing", "size", "checksum", "symlink"])
@pytest.mark.parametrize("source_name", ["demo/own_kitchen.py", "scripts/isaac_python_spans.py"])
def test_tampered_source_is_rejected_before_use(source, tmp_path, fault, source_name):
    repo, config = source
    path = repo / source_name
    raw = path.read_bytes()
    if fault == "missing":
        path.unlink()
    elif fault == "size":
        path.write_bytes(raw + b"X")
    elif fault == "checksum":
        path.write_bytes(b"X" + raw[1:])
    else:
        external = tmp_path / "external.py"
        external.write_bytes(raw)
        path.unlink()
        path.symlink_to(external)
    with pytest.raises(RuntimeError, match="verification failed"):
        kitchen.prepare_kitchen(repo)
    with pytest.raises(ValueError, match="verification failed"):
        identity.scene_identity(config)


def test_manifest_cannot_admit_extra_unreviewed_assets(source):
    repo, _ = source
    manifest = repo / identity.MANIFEST
    value = json.loads(manifest.read_text())
    value["files"]["demo/scene/unreviewed.usd"] = {"sha256": "0" * 64, "size_bytes": 1}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exactly"):
        kitchen.kitchen_manifest(repo)


def test_reviewed_artwork_changes_content_identity_even_with_same_config(source):
    repo, config = source
    before = identity.scene_identity(config)
    path = repo / "demo/own_kitchen_props.py"
    path.write_text(path.read_text() + "Reviewed new orange geometry\n")
    (repo / identity.MANIFEST).write_text(json.dumps(identity.source_manifest(repo)))
    after = identity.scene_identity(config)
    assert before["scene_config_sha256"] == after["scene_config_sha256"]
    assert before["scene_content_sha256"] != after["scene_content_sha256"]
    assert before["scene_assets_sha256"] != after["scene_assets_sha256"]
    assert after["scene_name"] == identity.SCENE_NAME
    assert after["scene_config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_foreign_scene_identity_is_rejected(source):
    _, config = source
    config.write_text('{"version":1,"scene_name":"unknown"}')
    with pytest.raises(ValueError, match="Cocina Asier kitchen identity"):
        identity.scene_identity(config)


def test_unmanifested_visual_cannot_hide_behind_valid_source_hashes(source):
    repo, config = source
    value = json.loads(config.read_text())
    value["props"][0]["asset"] = "props/unreviewed.usda"
    (repo / "demo/scene/props/unreviewed.usda").write_text("unreviewed geometry")
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unreviewed visual"):
        identity.scene_identity(config)
    with pytest.raises(RuntimeError, match="unreviewed visual"):
        kitchen.prepare_kitchen(repo)


def test_check_on_missing_bundle_is_read_only_and_never_downloads(source, monkeypatch):
    repo, _ = source
    shutil.rmtree(repo / identity.BUNDLE_ROOT)
    before = {p: p.stat().st_mtime_ns for p in repo.rglob("*")}
    monkeypatch.setattr(kitchen.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Network in --check"))
    assert kitchen.main(["--repo", str(repo), "--check"]) == 1
    assert before == {p: p.stat().st_mtime_ns for p in repo.rglob("*")}


def test_missing_bundle_installs_from_pinned_archive_offline(source, tmp_path, monkeypatch):
    repo, config = source
    expected = identity.scene_identity(config)
    shutil.rmtree(repo / identity.BUNDLE_ROOT)
    monkeypatch.setattr(kitchen.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network"))
    kitchen.prepare_kitchen(repo, archive=tmp_path / "release.zip")
    assert identity.scene_identity(config) == expected


@pytest.mark.parametrize("fault", ["missing", "checksum", "extra", "symlink", "directory_symlink"])
def test_release_tampering_cannot_claim_the_reviewed_scene(source, tmp_path, fault):
    repo, config = source
    path = repo / identity.BUNDLE_ROOT / "cocina_asier.usdc"
    if fault == "missing":
        path.unlink()
    elif fault == "checksum":
        path.write_bytes(b"X" + path.read_bytes()[1:])
    elif fault == "extra":
        (path.parent / "unreviewed.usd").write_text("unreviewed")
    elif fault == "directory_symlink":
        (path.parent / "unreviewed").symlink_to(tmp_path, target_is_directory=True)
    else:
        external = tmp_path / "external.usdc"
        path.rename(external)
        path.symlink_to(external)
    with pytest.raises(ValueError, match="verification failed"):
        identity.scene_identity(config)
    with pytest.raises(RuntimeError, match="verification failed"):
        kitchen.prepare_kitchen(repo, check=True)


def test_failed_archive_verification_preserves_existing_scene(source, tmp_path):
    repo, _ = source
    path = repo / identity.BUNDLE_ROOT / "cocina_asier.usdc"
    path.write_bytes(b"invalid room")
    before = {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    archive = tmp_path / "release.zip"
    archive.write_bytes(b"X" + archive.read_bytes()[1:])
    with pytest.raises(ValueError, match="archive checksum"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert before == {p.relative_to(path.parent): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    assert not list(path.parent.parent.glob(".cocina-asier-*"))


@pytest.mark.parametrize("fault", ["traversal", "duplicate", "extra", "symlink", "checksum", "size"])
def test_hostile_archive_never_replaces_existing_files(source, tmp_path, fault):
    repo, _ = source
    bundle = identity.bundle_manifest(repo)
    root = repo / identity.BUNDLE_ROOT
    # A reviewed archive digest is necessary but not sufficient: independently
    # enforce the per-file inventory, paths, type, sizes, and content checks.
    archive = tmp_path / "hostile.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name in bundle["files"]:
            info = zipfile.ZipInfo(name)
            raw = (root / name).read_bytes()
            if name == "cocina_asier.usdc":
                if fault == "traversal":
                    info.filename = "../outside.usdc"
                elif fault == "symlink":
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif fault == "checksum":
                    raw = b"X" + raw[1:]
                elif fault == "size":
                    raw += b"X"
            output.writestr(info, raw)
        if fault == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                output.writestr("cocina_asier.usdc", b"reviewed room")
        elif fault == "extra":
            output.writestr("unreviewed.usdc", b"unreviewed")
    bundle["archive"].update(identity.file_receipt(archive))
    (repo / identity.BUNDLE_MANIFEST).write_text(json.dumps(bundle))
    (repo / identity.MANIFEST).write_text(json.dumps(identity.source_manifest(repo)))
    (root / "cocina_asier.usdc").write_text("force repair, preserve on failure")
    with pytest.raises(ValueError, match="inventory|regular file|checksum"):
        kitchen.prepare_kitchen(repo, archive=archive)
    assert (root / "cocina_asier.usdc").read_text() == "force repair, preserve on failure"
    assert not (root.parent / "outside.usdc").exists()


def test_reviewed_bundle_changes_content_identity_with_same_config(source, tmp_path):
    repo, config = source
    before = identity.scene_identity(config)
    (repo / identity.BUNDLE_ROOT / "cocina_asier.usdc").write_text("new reviewed room")
    kitchen.package_bundle(repo, tmp_path / "release-2.zip")
    after = identity.scene_identity(config)
    assert before["scene_config_sha256"] == after["scene_config_sha256"]
    assert before["scene_content_sha256"] != after["scene_content_sha256"]


def test_package_is_reproducible_and_does_not_overwrite_artifacts(source, tmp_path):
    repo, _ = source
    kitchen.package_bundle(repo, tmp_path / "release-2.zip")
    assert (tmp_path / "release.zip").read_bytes() == (tmp_path / "release-2.zip").read_bytes()
    with pytest.raises(ValueError, match="immutable"):
        kitchen.package_bundle(repo, tmp_path / "release.zip")


@pytest.mark.parametrize("fault", ["extra_member", "missing_notice", "missing_sources", "source_deleted", "artwork_changed"])
def test_invalid_package_never_replaces_either_existing_receipt(source, tmp_path, monkeypatch, fault):
    repo, _ = source
    receipts = {name: ((repo / name).read_bytes(), (repo / name).stat().st_mtime_ns)
                for name in (identity.BUNDLE_MANIFEST, identity.MANIFEST)}
    root = repo / identity.BUNDLE_ROOT
    if fault == "extra_member":
        (root / "unreviewed.md").write_text("not admitted")
    elif fault == "missing_notice":
        (root / "NOTICE.md").unlink()
    elif fault == "missing_sources":
        (root / "sources.json").unlink()
    elif fault == "source_deleted":
        (repo / "demo/cocina_asier.py").unlink()
    else:
        original = kitchen.unpack
        def change_after_unpack(*args):
            original(*args)
            (root / "cocina_asier.usdc").write_text("changed during packaging")
        monkeypatch.setattr(kitchen, "unpack", change_after_unpack)
    candidate = tmp_path / "invalid-candidate.zip"
    with pytest.raises(ValueError):
        kitchen.package_bundle(repo, candidate)
    assert not candidate.exists()
    assert receipts == {name: ((repo / name).read_bytes(), (repo / name).stat().st_mtime_ns)
                        for name in receipts}


@pytest.mark.parametrize("fault", ["oversize", "truncated", "checksum", "redirect", "content_length", "deadline"])
def test_download_rejects_size_checksum_origin_and_time_failures(tmp_path, monkeypatch, fault):
    raw = b"archive fixture"
    expected = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    body = raw + b"X" if fault == "oversize" else raw[:-1] if fault == "truncated" else raw
    if fault == "checksum":
        body = b"X" + body[1:]
    response = io.BytesIO(body)
    response.geturl = lambda: "https://example.invalid/asset" if fault == "redirect" else identity.BUNDLE_URL
    response.headers = {"Content-Length": "1"} if fault == "content_length" else {}
    monkeypatch.setattr(kitchen.urllib.request, "urlopen", lambda *a, **kw: response)
    if fault == "deadline":
        ticks = iter([0, kitchen.DOWNLOAD_BUDGET_S + 1])
        monkeypatch.setattr(kitchen.time, "monotonic", lambda: next(ticks))
    with pytest.raises((ValueError, TimeoutError)):
        kitchen.download(identity.BUNDLE_URL, tmp_path / "download.zip", expected)


def test_downloaded_release_is_verified_and_second_run_reuses_it(source, tmp_path, monkeypatch):
    repo, config = source
    expected = identity.scene_identity(config)
    raw = (tmp_path / "release.zip").read_bytes()
    shutil.rmtree(repo / identity.BUNDLE_ROOT)
    calls = []
    def fetch(request, timeout):
        calls.append(request.full_url)
        assert timeout == 30
        response = io.BytesIO(raw)
        response.geturl = lambda: "https://release-assets.githubusercontent.com/fixture.zip"
        response.headers = {"Content-Length": str(len(raw))}
        return response
    monkeypatch.setattr(kitchen.urllib.request, "urlopen", fetch)
    kitchen.prepare_kitchen(repo)
    kitchen.prepare_kitchen(repo)
    assert calls == [identity.BUNDLE_URL]
    assert identity.scene_identity(config) == expected
