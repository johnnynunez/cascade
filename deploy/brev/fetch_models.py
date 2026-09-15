#!/usr/bin/env python3
"""Fetch the pinned Qwen model on Brev, preserving verified download receipts."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import tempfile
import time
from typing import Callable
import urllib.error
import urllib.request


DEFAULT_MANIFEST = Path(__file__).parent / "profiles" / "qwen3.8-27b-q8.json"
DEFAULT_STOP = Path(__file__).resolve().parents[2] / "STOP"
MODEL_DIRECTORY = Path("models/qwen3.8-27b")
CHUNK_BYTES = 1024 * 1024


class DownloadError(Exception):
    pass


class TransferInterrupted(DownloadError):
    """A partial transfer may be resumed within the remaining attempt budget."""


class DeadlineExceeded(DownloadError):
    pass


@dataclass(frozen=True)
class Artifact:
    filename: str
    size_bytes: int
    sha256: str
    url: str


@dataclass(frozen=True)
class Limits:
    deadline: float
    timeout: float = 30
    attempts: int = 4
    reserve_bytes: int = 1024**3
    stop_file: Path | None = None

    def check(self) -> float:
        if self.stop_file is not None and self.stop_file.exists():
            raise DownloadError("Stop file is present; partial transfers are preserved.")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise DeadlineExceeded("Download time cap reached; partial transfers are preserved.")
        return remaining


def emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def load_manifest(path: Path, max_bytes: int) -> list[Artifact]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source = data.get("source_model", {})
    if source.get("repo_id") != "Qwen/Qwen3.8-27B":
        raise DownloadError("Manifest must select the official Qwen/Qwen3.8-27B model.")
    quant = data.get("quantization", {})
    if quant.get("repo_id") != "ggml-org/Qwen3.8-27B-GGUF":
        raise DownloadError("Manifest must select the reviewed ggml-org quantization.")
    revision = quant.get("revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise DownloadError("Manifest must pin a complete quantization revision.")
    artifacts = []
    for item in quant.get("files", []):
        filename = item.get("filename", "")
        size = item.get("size_bytes")
        digest = item.get("sha256", "")
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.gguf", filename):
            raise DownloadError("Manifest contains an unsafe artifact filename.")
        if type(size) is not int or size <= 0:
            raise DownloadError("Manifest contains an invalid artifact size.")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DownloadError("Manifest contains an invalid SHA256 digest.")
        url = f"https://huggingface.co/{quant['repo_id']}/resolve/{revision}/{filename}"
        if item.get("url") != url:
            raise DownloadError("Artifact URL does not match its public pinned repository.")
        artifacts.append(Artifact(filename, size, digest, url))
    if not artifacts or len({a.filename for a in artifacts}) != len(artifacts):
        raise DownloadError("Manifest must contain unique artifacts.")
    source_revision = source.get("revision", "")
    if not isinstance(source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise DownloadError("Manifest must pin a complete source model revision.")
    license = source.get("license")
    if not isinstance(license, dict):
        raise DownloadError("Manifest must contain the pinned official model license.")
    size = license.get("size_bytes")
    digest = license.get("sha256", "")
    if type(size) is not int or size <= 0:
        raise DownloadError("Manifest contains an invalid license size.")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DownloadError("Manifest contains an invalid license SHA256 digest.")
    url = f"https://huggingface.co/{source['repo_id']}/resolve/{source_revision}/LICENSE"
    if license.get("url") != url:
        raise DownloadError("License URL does not match the official pinned model repository.")
    artifacts.append(Artifact("LICENSE", size, digest, url))
    if sum(a.size_bytes for a in artifacts) > max_bytes:
        raise DownloadError("Manifest exceeds the configured download byte cap.")
    return artifacts


def _regular_stat(path: Path) -> os.stat_result | None:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(result.st_mode):
        raise DownloadError(f"Expected a regular file: {path.name}")
    return result


def _receipt_path(destination: Path) -> Path:
    return destination.with_name(destination.name + ".verified.json")


def _receipt_matches(destination: Path, artifact: Artifact) -> bool:
    current = _regular_stat(destination)
    receipt = _receipt_path(destination)
    receipt_stat = _regular_stat(receipt)
    if current is None or receipt_stat is None or receipt_stat.st_size > 16384:
        return False
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and all((
        data.get("version") == 1,
        data.get("filename") == artifact.filename,
        data.get("url") == artifact.url,
        data.get("sha256") == artifact.sha256,
        data.get("size_bytes") == current.st_size == artifact.size_bytes,
        data.get("mtime_ns") == current.st_mtime_ns,
    ))


def _sha256_file(path: Path, artifact: Artifact, limits: Limits) -> os.stat_result:
    before = _regular_stat(path)
    if before is None or before.st_size != artifact.size_bytes:
        raise DownloadError(f"Unexpected file size: {artifact.filename}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            limits.check()
            chunk = source.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    after = _regular_stat(path)
    if after is None or (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise DownloadError(f"File changed during checksum verification: {artifact.filename}")
    if digest.hexdigest() != artifact.sha256:
        raise DownloadError(f"SHA256 mismatch; file preserved for inspection: {artifact.filename}")
    return after


def _write_receipt(destination: Path, artifact: Artifact, verified: os.stat_result) -> None:
    receipt = _receipt_path(destination)
    _regular_stat(receipt)
    data = {
        "version": 1,
        "filename": artifact.filename,
        "url": artifact.url,
        "size_bytes": verified.st_size,
        "mtime_ns": verified.st_mtime_ns,
        "sha256": artifact.sha256,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=".receipt-",
                                         encoding="utf-8", delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, receipt)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _response_size(response: object, artifact: Artifact, offset: int) -> int:
    status = response.status
    headers = response.headers
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise DownloadError("Server returned an unexpected content encoding.")
    if status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
        if not match:
            raise DownloadError("Server omitted a valid Content-Range for partial content.")
        start, end, total = map(int, match.groups())
        if start != offset or total != artifact.size_bytes or not start <= end < total:
            raise DownloadError("Server returned a range inconsistent with the pinned artifact.")
        expected = end - start + 1
    elif status == 200 and offset == 0:
        expected = artifact.size_bytes
        if headers.get("Content-Range"):
            raise DownloadError("Server returned Content-Range without partial-content status.")
    elif status == 200:
        raise DownloadError("Server ignored the resume range; existing partial file was preserved.")
    else:
        raise DownloadError(f"Unexpected HTTP status: {status}")
    length = headers.get("Content-Length")
    if length is not None and (not length.isdigit() or int(length) != expected):
        raise DownloadError("Content-Length does not match the expected transfer size.")
    return expected


def _transfer(artifact: Artifact, partial: Path, offset: int, limits: Limits) -> None:
    headers = {"Accept-Encoding": "identity", "User-Agent": "paai-brev-model-fetch/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(artifact.url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=min(limits.timeout, limits.check())) as response:
            expected = _response_size(response, artifact, offset)
            # Validate the response before opening the partial file for append.
            fd = os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
            with os.fdopen(fd, "ab") as output:
                if os.fstat(output.fileno()).st_size != offset:
                    raise DownloadError("Partial file changed before transfer started.")
                received = 0
                while received < expected:
                    limits.check()
                    chunk = response.read1(min(CHUNK_BYTES, expected - received))
                    if not chunk:
                        raise TransferInterrupted("Transfer ended before the declared byte count.")
                    output.write(chunk)
                    received += len(chunk)
                limits.check()
                if response.read1(1):
                    raise DownloadError("Server sent bytes beyond the pinned artifact boundary.")
                output.flush()
                os.fsync(output.fileno())
    except urllib.error.HTTPError as error:
        message = f"HTTP request failed with status {error.code}."
        if error.code in (408, 429) or 500 <= error.code <= 599:
            raise TransferInterrupted(message) from None
        raise DownloadError(message) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
        # Redirect errors may include signed CDN URLs, so never print their text.
        raise TransferInterrupted("Network transfer interrupted.") from None


def fetch_file(artifact: Artifact, directory: Path, limits: Limits,
               report: Callable[..., None] = emit) -> str:
    limits.check()
    destination = directory / artifact.filename
    partial = directory / (artifact.filename + ".part")
    existing = _regular_stat(destination)
    if existing is not None:
        if _receipt_matches(destination, artifact):
            report("skipped", filename=artifact.filename, size_bytes=existing.st_size)
            return "skipped"
        verified = _sha256_file(destination, artifact, limits)
        _write_receipt(destination, artifact, verified)
        report("verified_existing", filename=artifact.filename, size_bytes=verified.st_size)
        return "verified_existing"

    for attempt in range(1, limits.attempts + 1):
        limits.check()
        current = _regular_stat(partial)
        offset = current.st_size if current is not None else 0
        if offset > artifact.size_bytes:
            raise DownloadError(f"Partial file exceeds the pinned size: {artifact.filename}")
        if offset == artifact.size_bytes:
            break
        required = artifact.size_bytes - offset + limits.reserve_bytes
        if shutil.disk_usage(directory).free < required:
            raise DownloadError(f"Insufficient free disk space for {artifact.filename} and reserve.")
        report("resuming" if offset else "downloading", filename=artifact.filename,
               offset_bytes=offset, total_bytes=artifact.size_bytes, attempt=attempt)
        try:
            _transfer(artifact, partial, offset, limits)
        except TransferInterrupted as error:
            report("interrupted", filename=artifact.filename, attempt=attempt, reason=str(error))
            if attempt == limits.attempts:
                raise DownloadError("Transfer attempt limit reached; partial file preserved.") from None
            time.sleep(min(float(attempt), limits.check()))
        current = _regular_stat(partial)
        if current is not None and current.st_size == artifact.size_bytes:
            break
    verified = _sha256_file(partial, artifact, limits)
    limits.check()
    os.replace(partial, destination)
    _write_receipt(destination, artifact, verified)
    report("completed", filename=artifact.filename, size_bytes=verified.st_size, sha256=artifact.sha256)
    return "completed"


@contextlib.contextmanager
def _process_time_cap(seconds: float):
    def expired(signum: int, frame: object) -> None:
        raise DeadlineExceeded("Download time cap reached; partial transfers are preserved.")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Heavy deployment root; files use models/qwen3.8-27b.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata-only", action="store_true", help="Print the plan without writes or network.")
    parser.add_argument("--timeout", type=float, default=30, help="Network operation timeout in seconds.")
    parser.add_argument("--max-seconds", type=float, default=7200, help="Total transfer and verification time cap.")
    parser.add_argument("--attempts", type=int, default=4, help="Maximum HTTP attempts per artifact.")
    parser.add_argument("--max-bytes", type=int, default=32 * 1024**3, help="Maximum total pinned artifact size.")
    parser.add_argument("--reserve-bytes", type=int, default=1024**3, help="Free disk reserve after download.")
    parser.add_argument("--stop-file", type=Path, default=DEFAULT_STOP)
    args = parser.parse_args(argv)
    try:
        if not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.max_seconds)):
            raise DownloadError("Timeout and time cap must be finite positive numbers.")
        if not 1 <= args.attempts <= 10 or args.max_bytes <= 0 or args.reserve_bytes < 0:
            raise DownloadError("Invalid attempt count, byte cap, or disk reserve.")
        artifacts = load_manifest(args.manifest, args.max_bytes)
        if args.metadata_only:
            emit("plan", files=[a.__dict__ for a in artifacts],
                 total_bytes=sum(a.size_bytes for a in artifacts),
                 relative_directory=str(MODEL_DIRECTORY), network_requests=0)
            return 0
        if args.root is None:
            raise DownloadError("--root is required for a download.")
        limits = Limits(time.monotonic() + args.max_seconds, args.timeout, args.attempts,
                        args.reserve_bytes, args.stop_file)
        limits.check()
        directory = args.root.resolve() / MODEL_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(directory / ".fetch-models.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DownloadError("Another model downloader already holds the directory lock.") from None
            with _process_time_cap(limits.check()):
                for artifact in artifacts:
                    fetch_file(artifact, directory, limits)
        return 0
    except (DownloadError, OSError, ValueError, KeyError, TypeError) as error:
        # OSError can contain paths; neither it nor remote exceptions should reveal credentials.
        message = str(error) if isinstance(error, DownloadError) else f"Local operation failed ({type(error).__name__})."
        emit("error", message=message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
