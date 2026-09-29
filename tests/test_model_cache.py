"""Scope and ordering of advisory model-cache release; no GPU/model loads."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux file-cache advice")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("model_cache", ROOT / "scripts/model_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from cascade.apps import process_owner
    monkeypatch.setattr(process_owner, "is_live", lambda *args: True)
    monkeypatch.setattr(module, "model_is_mapped", lambda *args: False)
    monkeypatch.setattr(module, "cached_pages", lambda fd: {"cached": 8, "dirty": 0, "writeback": 0})
    path = tmp_path / "models/model.gguf"
    path.parent.mkdir()
    path.write_bytes(b"GGUF" + b"owned fixture" * 20)
    def record(path=path):
        info = path.stat()
        return {"repo": str(tmp_path), "role": "qwen", "pid": os.getpid(), "model_binding": {"files": {"model": {
            "path": str(path), "device": info.st_dev, "inode": info.st_ino, "size_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}}}}
    return module, path, record


def test_advice_preserves_file_bytes_and_metadata(cache, tmp_path, monkeypatch):
    module, path, record = cache
    before = path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_size
    calls = []
    real = os.posix_fadvise
    def advise(fd, offset, length, advice):
        assert os.fstat(fd).st_ino == path.stat().st_ino
        calls.append((offset, length, advice))
        real(fd, offset, length, advice)
    monkeypatch.setattr(module.os, "posix_fadvise", advise)
    assert module.release_clean_model_cache(tmp_path, record(), {})["status"] == "advised"
    assert calls == [(0, 0, os.POSIX_FADV_DONTNEED)]
    assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_size) == before


@pytest.mark.parametrize("kind", ["outside", "symlink", "parent_symlink", "hardlink", "mapped", "wrong_uid", "dirty", "writeback", "changed", "unavailable", "foreign"])
def test_cache_hint_refuses_unproven_or_shared_scope(cache, tmp_path, monkeypatch, kind):
    module, path, record = cache
    current = record()
    if kind == "outside":
        outside = tmp_path / "unrelated.gguf"
        outside.write_bytes(b"private unrelated fixture")
        current = record(outside)
    elif kind == "symlink":
        link = path.parent / "link.gguf"
        link.symlink_to(path)
        current = record(link)
    elif kind == "parent_symlink":
        link = path.parent / "linked-directory"
        link.symlink_to(path.parent, target_is_directory=True)
        current = record(link / path.name)
    elif kind == "hardlink":
        os.link(path, path.with_suffix(".shared"))
        current = record()
    elif kind == "mapped":
        monkeypatch.setattr(module, "model_is_mapped", lambda *args: True)
    elif kind == "wrong_uid":
        monkeypatch.setattr(module.os, "getuid", lambda: path.stat().st_uid + 1)
    elif kind in ("dirty", "writeback"):
        monkeypatch.setattr(module, "cached_pages", lambda fd: {"cached": 8, "dirty": int(kind == "dirty"), "writeback": int(kind == "writeback")})
    elif kind == "changed":
        path.write_bytes(b"changed after model loading")
    elif kind == "unavailable":
        def unavailable(fd):
            raise OSError("cachestat unavailable")
        monkeypatch.setattr(module, "cached_pages", unavailable)
    elif kind == "foreign":
        from cascade.apps import process_owner
        monkeypatch.setattr(process_owner, "is_live", lambda *args: False)
    monkeypatch.setattr(module.os, "posix_fadvise", lambda *args: pytest.fail("advised an unproven file"))
    assert module.release_clean_model_cache(tmp_path, current, {})["status"] == "skipped"


def test_model_mapping_uses_inode_and_device(cache):
    import mmap
    module, path, _ = cache
    # Use the actual implementation despite the cache fixture's boundary double.
    spec = importlib.util.spec_from_file_location("actual_model_cache", ROOT / "scripts/model_cache.py")
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ):
        assert actual.model_is_mapped(os.getpid(), path.stat())


def test_hint_runs_after_healthy_owned_qwen_and_before_isaac_start(tmp_path, monkeypatch):
    from test_spark_install import launch_fixture
    from cascade.apps.process_owner import is_live
    support, repo = launch_fixture(tmp_path, monkeypatch)
    calls = []
    def advice(selected, record, owner):
        assert selected == repo and support.model_health()
        assert is_live(record, owner) and record["role"] == "qwen"
        assert not (repo / "launch-record.json").exists()
        calls.append(record["pid"])
        return {"status": "skipped", "reason": "fixture boundary"}
    monkeypatch.setattr(support, "release_clean_model_cache", advice)
    try:
        assert support.launch(repo, "spark", "qwen", no_open=True) == 0
        assert calls == [int((repo / "service.pid").read_text())]
        assert (repo / "launch-record.json").exists()
    finally:
        for child in support._fixture_qwen:
            support.stop_group(child)
