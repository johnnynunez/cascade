"""Admission tests: offline check, explicit fetch, immutable bytes, safe paths."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/microduck_assets.py"


def module():
    assert SCRIPT.is_file(), "explicit asset admission CLI missing"
    spec = importlib.util.spec_from_file_location("microduck_assets_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def manifest():
    return {
        "schema_version": 1,
        "sources": {"model-source": {"kind": "github", "repository": "owner/robot", "revision": "a"*40}},
        "licenses": {"models": {"license": "Creative Commons BY-SA-NC", "version": None,
                                 "evidence": "https://example.org/license"}},
        "files": [{"source": "model-source", "source_path": "robot.xml", "path": "model-source/robot.xml",
                   "size": 6, "sha256": hashlib.sha256(b"<xml/>").hexdigest(), "license": "models"}],
    }


def save_manifest(tmp_path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return path


@pytest.mark.parametrize("badpath", ["../escape", "/tmp/escape", "a/../../x", "a\\..\\x", "C:/x",
    "a//b", "a/./b", "a/%2e%2e/b", "a?b", "a#b", "a\x00b"])
def test_rejects_path_traversal_before_network_or_writes(tmp_path, badpath):
    m = module()
    data = manifest()
    data["files"][0]["path"] = badpath
    with pytest.raises(ValueError): m.load_manifest(save_manifest(tmp_path, data))


@pytest.mark.parametrize("mutation", ["pin", "hash", "size", "bool_size", "unknown_license", "duplicate", "prefix", "source_path", "unknown_source", "empty"])
def test_rejects_malformed_manifests(tmp_path, mutation):
    m = module()
    data = manifest()
    row = data["files"][0]
    if mutation == "pin": data["sources"]["model-source"]["revision"] = "main"
    if mutation == "hash": row["sha256"] = "f"*63
    if mutation == "size": row["size"] = -1
    if mutation == "bool_size": row["size"] = True
    if mutation == "unknown_license": row["license"] = "Apache"
    if mutation == "duplicate": data["files"].append(copy.deepcopy(row))
    if mutation == "prefix":
        extra = copy.deepcopy(row)
        extra.update(path=row["path"]+"/child", source_path="robot.xml/child")
        data["files"].append(extra)
    if mutation == "source_path": row["source_path"] = "../outside"
    if mutation == "unknown_source": row["source"] = "missing"
    if mutation == "empty": data["files"] = []
    with pytest.raises(ValueError): m.load_manifest(save_manifest(tmp_path, data))


@pytest.mark.parametrize("field,value", [("license", []), ("license", {}), ("source", []), ("sha256", None)])
def test_malformed_file_types_fail_cleanly(tmp_path, field, value):
    m = module()
    data = manifest()
    data["files"][0][field] = value
    with pytest.raises(ValueError): m.load_manifest(save_manifest(tmp_path, data))


def test_check_is_read_only_and_offline_even_when_missing(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("check attempted network"))
    data = m.load_manifest(save_manifest(tmp_path, manifest()))
    dest = tmp_path / "absent"
    result = m.check(data, dest)
    assert not result["ok"] and result["checked"] == 1
    assert not dest.exists()
    dest.mkdir()
    file = dest / data["files"][0]["path"]
    file.parent.mkdir()
    file.write_bytes(b"<xml/>")
    before = file.stat().st_mtime_ns
    assert m.check(data, dest)["ok"]
    assert file.stat().st_mtime_ns == before
    file.write_bytes(b"BROKEN")
    assert not m.check(data, dest)["ok"]
    assert file.read_bytes() == b"BROKEN"


def test_fetch_requires_license_and_verifies_bytes_before_publish(tmp_path, monkeypatch):
    m = module()
    data = m.load_manifest(save_manifest(tmp_path, manifest()))
    dest = tmp_path / "install"
    requests = []
    def get(url, **kwargs):
        requests.append(url.full_url if isinstance(url, urllib.request.Request) else url)
        return io.BytesIO(b"<xml/>")
    monkeypatch.setattr(urllib.request, "urlopen", get)
    with pytest.raises(ValueError): m.fetch(data, dest)
    assert not requests and not dest.exists()
    result = m.fetch(data, dest, accept_model_license=True)
    assert result["ok"] and m.check(data, dest)["ok"]
    assert requests == ["https://raw.githubusercontent.com/owner/robot/" + "a"*40 + "/robot.xml"]
    # Idempotent fetch doesn't redownload an already verified object.
    assert m.fetch(data, dest, accept_model_license=True)["ok"]
    assert len(requests) == 1
    file = dest / data["files"][0]["path"]
    file.write_bytes(b"WRONG!")
    with pytest.raises(ValueError): m.fetch(data, dest, accept_model_license=True)
    assert file.read_bytes() == b"WRONG!" and len(requests) == 1


@pytest.mark.parametrize("payload", [b"WRONG!", b"short", b"<xml/>EXCESS"])
def test_bad_download_is_not_published(tmp_path, monkeypatch, payload):
    m = module()
    data = m.load_manifest(save_manifest(tmp_path, manifest()))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    dest = tmp_path / "install"
    with pytest.raises(ValueError): m.fetch(data, dest, accept_model_license=True)
    assert not (dest / data["files"][0]["path"]).exists()


@pytest.mark.parametrize("location", ["destination", "parent", "file"])
def test_symlink_cannot_escape_destination(tmp_path, location):
    m = module()
    data = m.load_manifest(save_manifest(tmp_path, manifest()))
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "install"
    if location == "destination": dest.symlink_to(outside, target_is_directory=True)
    else:
        dest.mkdir()
        parent = dest / "model-source"
        if location == "parent": parent.symlink_to(outside, target_is_directory=True)
        else:
            parent.mkdir()
            (parent / "robot.xml").symlink_to(outside / "other.xml")
    with pytest.raises(ValueError): m.check(data, dest)
    with pytest.raises(ValueError): m.fetch(data, dest, accept_model_license=True)
    assert not list(outside.iterdir())


def test_local_fetch_has_same_admission_and_no_network(tmp_path, monkeypatch):
    m = module()
    data = m.load_manifest(save_manifest(tmp_path, manifest()))
    source = tmp_path / "local"
    file = source / data["files"][0]["path"]
    file.parent.mkdir(parents=True)
    file.write_bytes(b"<xml/>")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("local fetch attempted network"))
    dest = tmp_path / "install"
    assert m.fetch(data, dest, source_directory=source, accept_model_license=True)["ok"]
    assert (dest / data["files"][0]["path"]).read_bytes() == b"<xml/>"


def test_cli_requires_operation_and_destination_and_returns_check_exit_code(tmp_path):
    module()
    mf = save_manifest(tmp_path, manifest())
    for args in [[], ["--destination", str(tmp_path / "install")], ["--check"],
                 ["--check", "--fetch", "--destination", str(tmp_path / "install")]]:
        result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
        assert result.returncode != 0
    result = subprocess.run([sys.executable, str(SCRIPT), "--check", "--destination", str(tmp_path / "absent"),
                             "--manifest", str(mf)], capture_output=True, text=True)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["ok"]
    assert not (tmp_path / "absent").exists()


def test_import_never_fetches(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("import attempted network"))
    module()


def test_default_manifest_separates_licenses_pins_all_ten_weights_and_complete_model_dependencies():
    m = module()
    data = m.load_manifest(REPO / "assets/microduck/manifest.json")
    assert data["licenses"]["models"]["license"] == "Creative Commons BY-SA-NC"
    assert data["licenses"]["models"]["version"] is None
    assert data["licenses"]["weights"]["license"] == "Apache-2.0"
    assert data["licenses"]["code"]["license"] == "Apache-2.0"
    weights = {Path(f["path"]).name for f in data["files"] if f["path"].endswith(".onnx")}
    assert weights == {"velstand.onnx", "alpha_stand.onnx", "alpha_walking.onnx", "alpha_sitstand.onnx",
                       "alpha_ground_pick.onnx", "roller.onnx", "roller_crouch.onnx", "roulade.onnx",
                       "ball_kick_left.onnx", "ball_kick_right.onnx"}
    value = os.environ.get("MICRODUCK_MODEL_ROOT")
    if not value:
        pytest.skip("set MICRODUCK_MODEL_ROOT to pinned RL source for dependency closure check")
    root = Path(value)
    prefix = "src/mjlab_microduck/robot/microduck"
    declared = {f["source_path"] for f in data["files"] if f["source"] == "microduck_rl"}
    robots = sorted((root / prefix).glob("robot*.xml"))
    assert len(robots) == 8
    for robot in robots:
        xml = ET.parse(robot).getroot()
        compiler = xml.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        for elem in xml.iter():
            if "file" not in elem.attrib: continue
            directory = meshdir if elem.tag == "mesh" else ""
            dependency = (robot.parent / directory / elem.attrib["file"]).relative_to(root).as_posix()
            assert dependency in declared, dependency
    for f in data["files"]:
        if f["source"] == "microduck_rl":
            raw = (root / f["source_path"]).read_bytes()
            assert len(raw) == f["size"]
            assert hashlib.sha256(raw).hexdigest() == f["sha256"]
