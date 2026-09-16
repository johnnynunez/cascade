"""Offline distribution import and fresh controller startup behavior."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import prepare_bundle as bundle

spec = importlib.util.spec_from_file_location("bundle_deploy", HERE / "deploy.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def expected(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


@pytest.fixture
def distribution(tmp_path, monkeypatch):
    here = tmp_path / "controller"
    source = tmp_path / "separate-source"
    files = {name: b"source fixture\n" for name in bundle.RUNTIME_FILES}
    files.update({"LICENSE": b"source notice", "src/cascade/app.py": b"pass\n",
                  "demo/scene/assets/background.usda": b"licensed background fixture"})
    files["demo/scene/kitchen_config.json"] = json.dumps({
        "background": "assets/background.usda", "props": []}).encode()
    for name, data in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    assets = tmp_path / "asset-manifest.json"
    assets.write_text(json.dumps({"files": {"demo/scene/assets/background.usda": {
        **expected(files["demo/scene/assets/background.usda"]), "provenance": "Licensed fixture"}}}))
    monkeypatch.setattr(bundle, "ASSET_MANIFEST", assets)
    inputs = {"openclaw": b"native archive fixture", "clip": b"tokenizer archive fixture"}
    monkeypatch.setattr(bundle, "BUILD_INPUTS", {k: expected(v) for k, v in inputs.items()})
    archive = tmp_path / "openclaw.tar.zst"
    clip = tmp_path / "clip.tar.gz"
    archive.write_bytes(inputs["openclaw"])
    clip.write_bytes(inputs["clip"])
    profile = {"source": {"origin": "https://example.invalid/reviewed.git", "revision": "1" * 40}}
    record = {"schema": 1, "layout": bundle.LAYOUT, "upstream_url": profile["source"]["origin"],
              "upstream_revision": profile["source"]["revision"], "file_count": len(files),
              "source_bytes": sum(map(len, files.values())), "build_inputs": bundle.BUILD_INPUTS,
              "files": {n: {**expected(v), "provenance": "reviewed fixture"} for n, v in files.items()}}
    manifest = tmp_path / "bundle.json"
    manifest.write_text(json.dumps(record))
    return here, profile, source, manifest, archive, clip


def prepare(args, boundary=lambda: None):
    return bundle.prepare(*args, boundary=boundary)


def test_relocated_distribution_builds_local_receipts_without_importing_gates(distribution):
    result = prepare(distribution)
    here, _, source, _, archive, _ = distribution
    staged = json.loads((here / "publication/STAGED.json").read_text())
    assert staged["staged_root"] == str(source)
    assert result["source_files_hashed"] == len(staged["deploy_files"])
    assert result["source_files_copied"] == result["downloads"] == 0
    assert not (here / "STATE.json").exists()
    assert bundle.admitted_openclaw(here) == archive
    assert (here / "container/clip-source.tar.gz").read_bytes() == distribution[-1].read_bytes()


def test_repeat_preparation_reuses_unchanged_files_without_reading_them(distribution, monkeypatch):
    prepare(distribution)
    here, profile, source, manifest, archive, _ = distribution
    clip = here / "container/clip-source.tar.gz"
    old_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path in [archive, clip, source / "LICENSE", source / "src/cascade/app.py"]:
            pytest.fail("An unchanged certified input was read again")
        return old_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = prepare((here, profile, source, manifest, archive, clip))
    assert result["source_files_hashed"] == 0
    assert result["source_receipts_reused"] == json.loads(manifest.read_text())["file_count"]
    assert result["build_inputs_reused"] == 2


def test_changed_source_cannot_reuse_receipt_even_with_restored_mtime(distribution):
    prepare(distribution)
    here, _, source, *_ = distribution
    prior = (here / "publication/STAGED.json").read_bytes()
    path = source / "src/cascade/app.py"
    old = path.stat()
    path.write_bytes(b"fail\n")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum differs"):
        prepare(distribution)
    assert (here / "publication/STAGED.json").read_bytes() == prior


@pytest.mark.parametrize("name", ["../private", "/absolute", "src//app.py", "escape"])
def test_invalid_member_is_rejected_before_admission(distribution, name):
    here, _, source, manifest, *_ = distribution
    outside = source.parent / "private"
    outside.write_bytes(b"source notice")
    (source / "escape").symlink_to(outside)
    record = json.loads(manifest.read_text())
    record["files"][name] = record["files"].pop("src/cascade/app.py")
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="relative paths|leaves its source"):
        prepare(distribution)
    assert not (here / "publication/STAGED.json").exists()


def test_wrong_upstream_refuses_all_input_admission(distribution):
    here, profile, *_ = distribution
    profile["source"]["revision"] = "2" * 40
    with pytest.raises(ValueError, match="upstream profile differ"):
        prepare(distribution)
    assert not here.exists()


def test_stop_during_new_content_verification_publishes_no_receipt(distribution):
    calls = 0
    def boundary():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("STOP fixture")
    with pytest.raises(InterruptedError):
        prepare(distribution, boundary)
    assert not (distribution[0] / "publication/STAGED.json").exists()


def test_existing_different_clip_is_preserved(distribution):
    here = distribution[0]
    path = here / "container/clip-source.tar.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"previous input")
    with pytest.raises(ValueError, match="unexpected size"):
        prepare(distribution)
    assert path.read_bytes() == b"previous input"
    assert not (here / "publication/LOCAL_INPUTS.json").exists()


def test_archive_change_after_preparation_requires_new_verification(distribution):
    prepare(distribution)
    distribution[-2].write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="changed; run prepare"):
        bundle.admitted_openclaw(distribution[0])


@pytest.mark.parametrize("status", ["PASS", "WAITING_FOR_ACCESS"])
def test_fresh_access_records_only_current_gate_and_keeps_ready_false(tmp_path, monkeypatch, status):
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(deploy, "boundary", lambda: None)
    monkeypatch.setattr(deploy.access_gate, "probe", lambda: {
        "status": status, "evidence": "access/new-probe.json", "reason": "fixture"})
    if status == "PASS":
        deploy.require_access()
    else:
        with pytest.raises(ConnectionError, match="fixture"):
            deploy.require_access()
    state = json.loads((tmp_path / "STATE.json").read_text())
    assert state["gates"]["controller_access"]["status"] == status
    assert all(v["status"] == "pending" for k, v in state["gates"].items() if k != "controller_access")
    assert json.loads((tmp_path / "READY.json").read_text())["status"] == "NOT_READY"


def test_existing_malformed_state_is_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    path = tmp_path / "STATE.json"
    path.write_text("interrupted evidence")
    with pytest.raises(json.JSONDecodeError):
        deploy.update_gate("controller_access", "PASS", "fixture")
    assert path.read_text() == "interrupted evidence"


def test_preparation_refreshes_canonical_fields_while_preserving_provenance(distribution):
    prepare(distribution)
    path = distribution[0] / "publication/STAGED.json"
    record = json.loads(path.read_text())
    record.update(status="INCOMPLETE", deploy_files=["stale.py"], deploy_file_count=1,
                  upstream_url="https://example.invalid/stale", reviewed_notes="retained provenance")
    path.write_text(json.dumps(record))
    prepare(distribution)
    actual = json.loads(path.read_text())
    assert actual["status"] == "COMPLETE"
    manifest = json.loads(distribution[3].read_text())
    assert actual["deploy_file_count"] == manifest["file_count"]
    assert actual["upstream_url"] == distribution[1]["source"]["origin"]
    assert set(actual["deploy_files"]) == set(manifest["files"])
    assert actual["reviewed_notes"] == "retained provenance"


def test_stop_during_final_admission_sweep_preserves_previous_receipts(distribution, monkeypatch):
    prepare(distribution)
    here, _, source, *_ = distribution
    before = {name: (here / "publication" / name).read_bytes()
              for name in ("STAGED.json", "LOCAL_INPUTS.json")}
    stopped = False
    old_matches = bundle.matches
    def matches(path, expected, saved):
        nonlocal stopped
        if path == source / "LICENSE" and expected is saved:
            stopped = True
        return old_matches(path, expected, saved)
    def boundary():
        if stopped:
            raise InterruptedError("STOP fixture")
    monkeypatch.setattr(bundle, "matches", matches)
    with pytest.raises(InterruptedError):
        prepare(distribution, boundary)
    assert all((here / "publication" / name).read_bytes() == data for name, data in before.items())


def test_offline_prepare_stop_never_invokes_remote_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["deploy.py", "prepare"])
    def stopped():
        raise InterruptedError("STOP fixture")
    monkeypatch.setattr(deploy, "boundary", stopped)
    monkeypatch.setattr(deploy, "cancel_preparation", lambda: pytest.fail("Offline preparation contacted a host"))
    assert deploy.main() == 0
    record = json.loads((tmp_path / "proof/LAST_OPERATION.json").read_text())
    assert record["status"] == "STOPPED"
    assert record["preparation_cleanup"]["attempted"] is False


@pytest.mark.parametrize("layout", [None, "legacy-kitchen"])
def test_missing_or_legacy_layout_cannot_bypass_runtime_admission(distribution, layout):
    here, _, _, manifest, *_ = distribution
    record = json.loads(manifest.read_text())
    record.pop("layout")
    if layout is not None:
        record["layout"] = layout
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="unsupported runtime layout"):
        prepare(distribution)
    assert not here.exists()


def test_omitted_asset_cannot_receive_a_complete_source_receipt(distribution):
    here, _, _, manifest, *_ = distribution
    record = json.loads(manifest.read_text())
    omitted = record["files"].pop("demo/scene/assets/background.usda")
    record["file_count"] -= 1
    record["source_bytes"] -= omitted["size_bytes"]
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="runtime bundle is incomplete"):
        prepare(distribution)
    assert not here.exists()
