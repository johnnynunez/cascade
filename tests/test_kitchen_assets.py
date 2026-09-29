"""Offline authored-source admission and identity; no simulator required."""
import hashlib
import importlib.util
import json
from pathlib import Path

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
    (repo / identity.MANIFEST).write_text(json.dumps(identity.source_manifest(repo)))
    config = repo / "demo/scene/kitchen_config.json"
    config.write_text(json.dumps({"version": 1, "scene_name": identity.SCENE_NAME, "props": [
        {"name": "tomato_can", "asset": "props/tomato_can.usda"},
        {"name": "lemon", "asset": "props/lemon.usda"},
        {"name": "orange", "visual": "procedural_orange"}]}))
    monkeypatch.setattr(identity, "__file__", str(repo / "demo/scene_identity.py"))
    return repo, config


def test_shipped_sources_are_complete_and_verified():
    assert set(kitchen.kitchen_manifest(ROOT)) == set(identity.SOURCE_FILES)
    assert kitchen.kitchen_problems(ROOT) == []


def test_offline_prepare_leaves_all_bytes_and_mtimes_unchanged(source, monkeypatch):
    repo, _ = source
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in repo.rglob("*") if p.is_file()}
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unexpected network access"))
    kitchen.prepare_kitchen(repo)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("fault", ["missing", "size", "checksum", "symlink"])
def test_tampered_source_is_rejected_before_use(source, tmp_path, fault):
    repo, config = source
    path = repo / "demo/own_kitchen.py"
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
    with pytest.raises(ValueError, match="authored kitchen identity"):
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
