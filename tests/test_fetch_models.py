"""Exercise download integrity and reuse against a local HTTP server."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "deploy/brev/fetch_models.py"
SPEC = importlib.util.spec_from_file_location("brev_fetch_models", SCRIPT)
fetch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fetch
SPEC.loader.exec_module(fetch)
BODY = b"Qwen local download fixture.\n" * 128
MODEL_BYTES = 28595763552 + 931145888
LICENSE_BYTES = 11544


@pytest.fixture(autouse=True)
def local_downloads_only(monkeypatch):
    original = fetch.urllib.request.urlopen

    def open_local(request, *args, **kwargs):
        url = request.full_url if hasattr(request, "full_url") else request
        assert urlsplit(url).hostname in {"127.0.0.1", "localhost", "::1"}, "external model download in a unit test"
        return original(request, *args, **kwargs)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_local)


@contextmanager
def local_server(respond):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.headers.get("Range"))
            respond(self, len(requests))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fixture.gguf", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def artifact(url, body=BODY):
    return fetch.Artifact("fixture.gguf", len(body), hashlib.sha256(body).hexdigest(), url)


def limits(attempts=2):
    return fetch.Limits(time.monotonic() + 10, timeout=1, attempts=attempts, reserve_bytes=0)


def quiet(*args, **kwargs):
    pass


def full_response(handler, body=BODY):
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def test_truncated_transfer_resumes_exact_range_and_certifies_once(tmp_path, monkeypatch):
    cut = len(BODY) // 3

    def respond(handler, request_number):
        if request_number == 1:
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(BODY)))
            handler.end_headers()
            handler.wfile.write(BODY[:cut])
            handler.wfile.flush()
            handler.connection.shutdown(socket.SHUT_RDWR)
            return
        handler.send_response(206)
        handler.send_header("Content-Range", f"bytes {cut}-{len(BODY) - 1}/{len(BODY)}")
        handler.send_header("Content-Length", str(len(BODY) - cut))
        handler.end_headers()
        handler.wfile.write(BODY[cut:])

    verifications = []
    original = fetch._sha256_file

    def verify(*args):
        verifications.append(args[0])
        return original(*args)

    monkeypatch.setattr(fetch, "_sha256_file", verify)
    with local_server(respond) as (url, requests):
        assert fetch.fetch_file(artifact(url), tmp_path, limits(), quiet) == "completed"
    assert requests == [None, f"bytes={cut}-"]
    assert len(verifications) == 1
    destination = tmp_path / "fixture.gguf"
    assert destination.read_bytes() == BODY
    receipt = json.loads((tmp_path / "fixture.gguf.verified.json").read_text())
    assert receipt["sha256"] == hashlib.sha256(BODY).hexdigest()
    assert receipt["mtime_ns"] == destination.stat().st_mtime_ns
    assert not (tmp_path / "fixture.gguf.part").exists()


def test_ignored_range_preserves_partial_without_appending(tmp_path):
    partial = tmp_path / "fixture.gguf.part"
    prefix = BODY[:37]
    partial.write_bytes(prefix)
    with local_server(lambda handler, number: full_response(handler)) as (url, requests):
        with pytest.raises(fetch.DownloadError, match="ignored the resume range"):
            fetch.fetch_file(artifact(url), tmp_path, limits(), quiet)
    assert requests == ["bytes=37-"]
    assert partial.read_bytes() == prefix
    assert not (tmp_path / "fixture.gguf").exists()


@pytest.mark.parametrize("content_range", [
    f"bytes 0-{len(BODY) - 1}/{len(BODY)}",
    f"bytes 37-{len(BODY) - 1}/{len(BODY) + 1}",
    "bytes nonsense",
])
def test_invalid_range_never_modifies_partial(tmp_path, content_range):
    partial = tmp_path / "fixture.gguf.part"
    partial.write_bytes(BODY[:37])

    def respond(handler, number):
        handler.send_response(206)
        handler.send_header("Content-Range", content_range)
        handler.send_header("Content-Length", str(len(BODY) - 37))
        handler.end_headers()

    with local_server(respond) as (url, requests):
        with pytest.raises(fetch.DownloadError, match="range|Content-Range"):
            fetch.fetch_file(artifact(url), tmp_path, limits(), quiet)
    assert requests == ["bytes=37-"]
    assert partial.read_bytes() == BODY[:37]


def test_checksum_mismatch_preserves_uncertified_partial(tmp_path):
    corrupt = b"X" + BODY[1:]
    with local_server(lambda handler, number: full_response(handler, corrupt)) as (url, requests):
        with pytest.raises(fetch.DownloadError, match="SHA256 mismatch"):
            fetch.fetch_file(artifact(url), tmp_path, limits(), quiet)
    assert len(requests) == 1
    assert (tmp_path / "fixture.gguf.part").read_bytes() == corrupt
    assert not (tmp_path / "fixture.gguf").exists()
    assert not (tmp_path / "fixture.gguf.verified.json").exists()


def test_valid_receipt_skips_network_and_checksum(tmp_path, monkeypatch):
    with local_server(lambda handler, number: full_response(handler)) as (url, requests):
        model = artifact(url)
        fetch.fetch_file(model, tmp_path, limits(), quiet)

        def forbidden(*args, **kwargs):
            pytest.fail("Certified files must skip both the network and another full checksum.")

        monkeypatch.setattr(fetch, "_sha256_file", forbidden)
        monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden)
        assert fetch.fetch_file(model, tmp_path, limits(), quiet) == "skipped"
    assert len(requests) == 1


def test_changed_mtime_requires_checksum_before_reuse(tmp_path):
    with local_server(lambda handler, number: full_response(handler)) as (url, requests):
        model = artifact(url)
        fetch.fetch_file(model, tmp_path, limits(), quiet)
        destination = tmp_path / model.filename
        previous = destination.stat()
        destination.write_bytes(b"X" + BODY[1:])
        os.utime(destination, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1))
        with pytest.raises(fetch.DownloadError, match="SHA256 mismatch"):
            fetch.fetch_file(model, tmp_path, limits(), quiet)
    assert len(requests) == 1


def test_metadata_only_does_not_write_or_fetch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fetch, "fetch_file", lambda *a, **kw: pytest.fail("metadata-only fetched a model"))
    root = tmp_path / "heavy-root"
    assert fetch.main(["--root", str(root), "--metadata-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["network_requests"] == 0
    assert output["total_bytes"] == MODEL_BYTES + LICENSE_BYTES
    assert not root.exists()


def test_spark_manifest_pins_q4_and_matching_vision_projector():
    manifest = SCRIPT.parent / "profiles/qwen3.8-27b-q4.json"
    files = fetch.load_manifest(manifest, 32 * 1024**3)
    assert [a.filename for a in files] == ["Qwen3.8-27B-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf", "LICENSE"]
    assert sum(a.size_bytes for a in files) == 18_854_552_600
    assert all("/resolve/f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/" in a.url for a in files[:2])


def test_empty_destination_downloads_from_mirror_and_retains_public_identity(tmp_path, monkeypatch):
    model = fetch.Artifact("Qwen-fixture.gguf", len(BODY), hashlib.sha256(BODY).hexdigest(),
                           "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/" + "a" * 40 + "/Qwen-fixture.gguf")
    monkeypatch.setattr(fetch, "load_manifest", lambda *args: [model])
    root = tmp_path / "fresh"
    with local_server(lambda handler, number: full_response(handler)) as (url, requests):
        assert fetch.main(["--root", str(root), "--model-mirror-url", url, "--reserve-bytes", "0"]) == 0
    assert requests == [None]
    destination = root / fetch.MODEL_DIRECTORY / model.filename
    assert destination.read_bytes() == BODY
    receipt = json.loads(destination.with_name(model.filename + ".verified.json").read_text())
    assert receipt["url"] == model.url
    assert receipt["download_url"] == url
    assert receipt["sha256"] == model.sha256
    before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(fetch, "_sha256_file", lambda *args: pytest.fail("unchanged receipt rehashed"))
    assert fetch.main(["--root", str(root), "--check"]) == 0
    after = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("url", ["https://127.0.0.1/model", "http://example.com/model", "http://user@localhost/model", "http://localhost/model?token=x"])
def test_model_mirror_rejects_nonlocal_or_credentialled_urls(tmp_path, capsys, url):
    root = tmp_path / "absent"
    assert fetch.main(["--root", str(root), "--model-mirror-url", url]) == 1
    assert "loopback HTTP URL" in capsys.readouterr().out
    assert not root.exists()


def test_check_missing_models_does_not_create_destination(tmp_path):
    root = tmp_path / "absent"
    assert fetch.main(["--root", str(root), "--check"]) == 1
    assert not root.exists()


def test_download_requires_an_explicit_root(capsys):
    assert fetch.main([]) == 1
    assert "--root is required" in capsys.readouterr().out


def test_cli_rejects_byte_cap_before_creating_download_directory(tmp_path, capsys):
    root = tmp_path / "heavy-root"
    assert fetch.main(["--root", str(root), "--max-bytes", str(MODEL_BYTES)]) == 1
    assert "byte cap" in capsys.readouterr().out
    assert not root.exists()


def test_existing_stop_file_prevents_any_transfer(tmp_path):
    stop = tmp_path / "STOP"
    stop.touch()
    stopped = fetch.Limits(time.monotonic() + 10, stop_file=stop)
    with pytest.raises(fetch.DownloadError, match="Stop file"):
        fetch.fetch_file(artifact("http://127.0.0.1:1/never"), tmp_path, stopped, quiet)
    assert not (tmp_path / "fixture.gguf.part").exists()


def test_process_time_cap_interrupts_a_stalled_body(tmp_path):
    def respond(handler, number):
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(BODY)))
        handler.end_headers()
        time.sleep(0.5)

    with local_server(respond) as (url, requests):
        start = time.monotonic()
        with pytest.raises(fetch.DeadlineExceeded):
            with fetch._process_time_cap(0.1):
                fetch.fetch_file(artifact(url), tmp_path, limits(), quiet)
        elapsed = time.monotonic() - start
    assert elapsed < 1
    assert len(requests) == 1
    assert not (tmp_path / "fixture.gguf.verified.json").exists()


def test_official_license_is_planned_alongside_both_ggufs():
    artifacts = fetch.load_manifest(fetch.DEFAULT_MANIFEST, 32 * 1024**3)
    assert [item.filename for item in artifacts] == [
        "Qwen3.8-27B-Q8_0.gguf", "mmproj-Qwen3.8-27B-BF16.gguf", "LICENSE",
    ]
    assert artifacts[-1] == fetch.Artifact(
        "LICENSE", 11544, "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
        "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/"
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE",
    )


@pytest.mark.parametrize("path,value", [
    (("revision",), "main"),
    (("revision",), None),
    (("license",), None),
    (("license",), "Apache-2.0"),
    (("license", "url"), "https://example.invalid/LICENSE"),
    (("license", "url"), "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/LICENSE"),
    (("license", "url"), "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/"
                            "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE"),
    (("license", "size_bytes"), 0),
    (("license", "size_bytes"), True),
    (("license", "sha256"), "a" * 63),
    (("license", "sha256"), "g" * 64),
])
def test_license_requires_pinned_official_source(tmp_path, path, value):
    data = json.loads(fetch.DEFAULT_MANIFEST.read_text())
    item = data["source_model"]
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps(data))
    with pytest.raises(fetch.DownloadError, match="source model revision|license|License URL"):
        fetch.load_manifest(manifest, 32 * 1024**3)


def test_license_bytes_count_towards_total_cap():
    total = MODEL_BYTES + LICENSE_BYTES
    with pytest.raises(fetch.DownloadError, match="byte cap"):
        fetch.load_manifest(fetch.DEFAULT_MANIFEST, total - 1)
    assert sum(item.size_bytes for item in fetch.load_manifest(fetch.DEFAULT_MANIFEST, total)) == total


def test_main_fetches_license_once_and_reuses_cached_models(tmp_path, monkeypatch, capsys):
    data = json.loads(fetch.DEFAULT_MANIFEST.read_text())
    for item in [*data["quantization"]["files"], data["source_model"]["license"]]:
        item.update(size_bytes=len(BODY), sha256=hashlib.sha256(BODY).hexdigest())
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps(data))
    artifacts = fetch.load_manifest(manifest, 32 * 1024**3)
    directory = tmp_path / "root" / fetch.MODEL_DIRECTORY
    directory.mkdir(parents=True)
    for item in artifacts[:-1]:
        (directory / item.filename).write_bytes(BODY)
        assert fetch.fetch_file(item, directory, limits(), quiet) == "verified_existing"

    monkeypatch.setattr(fetch.socket, "gethostname", lambda: "brev-license-fixture")
    original_open = fetch.urllib.request.urlopen
    original_hash = fetch._sha256_file
    verifications = []

    def verify(*args):
        verifications.append(args[0].name)
        return original_hash(*args)

    def forbidden(*args, **kwargs):
        pytest.fail("Certified models and license must not be fetched or hashed again.")

    monkeypatch.setattr(fetch, "_sha256_file", verify)
    with local_server(lambda handler, number: full_response(handler)) as (url, requests):
        def open_fixture(request, *, timeout):
            assert request.full_url == artifacts[-1].url
            return original_open(url, timeout=timeout)

        monkeypatch.setattr(fetch.urllib.request, "urlopen", open_fixture)
        args = ["--root", str(tmp_path / "root"), "--manifest", str(manifest),
                "--stop-file", str(tmp_path / "STOP"), "--reserve-bytes", "0"]
        assert fetch.main(args) == 0
        assert requests == [None]
        assert verifications == ["LICENSE.part"]
        assert (directory / "LICENSE").read_bytes() == BODY
        receipt = json.loads((directory / "LICENSE.verified.json").read_text())
        assert receipt["url"] == artifacts[-1].url
        assert receipt["sha256"] == artifacts[-1].sha256
        assert receipt["mtime_ns"] == (directory / "LICENSE").stat().st_mtime_ns
        capsys.readouterr()
        monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden)
        monkeypatch.setattr(fetch, "_sha256_file", forbidden)
        assert fetch.main(args) == 0
        assert [json.loads(line)["event"] for line in capsys.readouterr().out.splitlines()] == [
            "skipped", "skipped", "skipped",
        ]
    assert requests == [None]


def test_existing_license_without_receipt_is_verified_then_skipped(tmp_path, monkeypatch):
    source = artifact("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/"
                      "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE")
    license = fetch.Artifact("LICENSE", source.size_bytes, source.sha256, source.url)
    destination = tmp_path / "LICENSE"
    destination.write_bytes(BODY)

    def forbidden(*args, **kwargs):
        pytest.fail("An existing verified license must not be downloaded or hashed again.")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden)
    assert fetch.fetch_file(license, tmp_path, limits(), quiet) == "verified_existing"
    before = (tmp_path / "LICENSE.verified.json").read_bytes()
    monkeypatch.setattr(fetch, "_sha256_file", forbidden)
    assert fetch.fetch_file(license, tmp_path, limits(), quiet) == "skipped"
    assert (tmp_path / "LICENSE.verified.json").read_bytes() == before
