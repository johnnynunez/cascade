#!/usr/bin/env python3
"""Fetch and verify the separately licensed Build a Claw kitchen assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


KITCHEN_URL = (
    "https://github.com/johnnynunez/cascade/releases/download/"
    "kitchen-v1/cascade-kitchen-v1.tar.gz"
)
KITCHEN_ARCHIVE_BYTES = 705_250_534
KITCHEN_ARCHIVE_SHA256 = "c94c2826180e295e4df16d799b5b3f581c9a2e60b08f0c34a99831ca1f80e35d"
PREFIXES = ("demo/scene/assets/", "demo/vendor-kitchen/")
CHUNK_SIZE = 1024 * 1024


def kitchen_manifest(repo: Path) -> dict[str, dict]:
    """Use the same pinned member checksums as the portable runtime bundle."""
    data = json.loads((repo / "deploy/brev/bundle_assets.json").read_text())
    files = {name: entry for name, entry in data["files"].items()
             if name.startswith(PREFIXES)}
    if not files:
        raise ValueError("Kitchen asset manifest is empty")
    for name, entry in files.items():
        parts = name.split("/")
        digest = entry["sha256"]
        if (PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in parts)
                or "\\" in name or not isinstance(entry["size_bytes"], int)
                or entry["size_bytes"] < 0 or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid kitchen manifest entry: {name}")
    return files


def _destination(root: Path, name: str) -> Path:
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Kitchen asset path must not be a symlink: {name}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def kitchen_problems(repo: Path, *, checksums: bool = True) -> list[str]:
    """Report missing/corrupt kitchen files without importing Isaac or USD."""
    problems = []
    for name, expected in kitchen_manifest(repo).items():
        try:
            path = _destination(repo, name)
            if not path.is_file():
                problems.append(f"Missing kitchen asset: {name}")
            elif path.stat().st_size != expected["size_bytes"]:
                problems.append(f"Kitchen asset size mismatch: {name}")
            elif checksums and _sha256(path) != expected["sha256"]:
                problems.append(f"Kitchen asset checksum mismatch: {name}")
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
    return problems


def _download(destination: Path, max_bytes: int) -> None:
    for attempt in range(3):
        try:
            request = urllib.request.Request(KITCHEN_URL, headers={"User-Agent": "CASCADE-installer"})
            with urllib.request.urlopen(request, timeout=60) as incoming, destination.open("wb") as outgoing:
                total = 0
                while chunk := incoming.read(CHUNK_SIZE):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Kitchen archive exceeds the manifest size limit")
                    outgoing.write(chunk)
            return
        except (OSError, urllib.error.URLError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def _unpack_verified(archive: Path, stage: Path, expected: dict[str, dict]) -> None:
    """Write only pinned regular files to a private staging directory."""
    seen = set()
    with tarfile.open(archive, "r|*") as bundle:
        for member in bundle:
            name = member.name
            if name not in expected or not member.isfile() or name in seen:
                raise ValueError(f"Unexpected, duplicate, or unsafe kitchen archive member: {name}")
            if member.size != expected[name]["size_bytes"]:
                raise ValueError(f"Kitchen archive size mismatch: {name}")
            seen.add(name)
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            incoming = bundle.extractfile(member)
            if incoming is None:
                raise ValueError(f"Cannot read kitchen archive member: {name}")
            with incoming, path.open("xb") as outgoing:
                while chunk := incoming.read(CHUNK_SIZE):
                    digest.update(chunk)
                    outgoing.write(chunk)
            if digest.hexdigest() != expected[name]["sha256"]:
                raise ValueError(f"Kitchen archive checksum mismatch: {name}")
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Kitchen archive is incomplete; missing {len(missing)} files: {min(missing)}")


def _verify_archive(archive: Path) -> None:
    actual_bytes = archive.stat().st_size
    if actual_bytes != KITCHEN_ARCHIVE_BYTES:
        raise ValueError(f"Kitchen archive size mismatch: expected {KITCHEN_ARCHIVE_BYTES}, received {actual_bytes}")
    actual_sha256 = _sha256(archive)
    if actual_sha256 != KITCHEN_ARCHIVE_SHA256:
        raise ValueError(f"Kitchen archive SHA-256 mismatch: expected {KITCHEN_ARCHIVE_SHA256}, received {actual_sha256}")


def prepare_kitchen(repo: Path, *, archive: str | Path | None = None) -> None:
    """Install a complete verified archive; use CASCADE_KITCHEN_ARCHIVE offline."""
    repo = repo.resolve()
    expected = kitchen_manifest(repo)
    if not kitchen_problems(repo):
        print(f"[cascade-install] Kitchen: {len(expected)} files verified.")
        return
    local = archive or os.environ.get("CASCADE_KITCHEN_ARCHIVE")
    try:
        # Stage on the destination filesystem. No installed file changes until
        # every archive member (including the license notices) passes its hash.
        with tempfile.TemporaryDirectory(prefix=".kitchen-install-", dir=repo) as temporary:
            stage = Path(temporary)
            source = Path(local).expanduser().resolve() if local else stage / "kitchen.tar.gz"
            if not local:
                print(f"[cascade-install] Fetching separately licensed kitchen: {KITCHEN_URL}", flush=True)
                _download(source, KITCHEN_ARCHIVE_BYTES)
            _verify_archive(source)
            _unpack_verified(source, stage, expected)
            destinations = {name: _destination(repo, name) for name in expected}
            for name, destination in destinations.items():
                if destination.exists() and not destination.is_file():
                    raise ValueError(f"Kitchen asset destination is not a file: {name}")
            for name, destination in destinations.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, destination)
    except (OSError, ValueError, tarfile.TarError, EOFError) as exc:
        raise RuntimeError(
            f"Kitchen installation failed: {exc}. Retry the installer, or set "
            "CASCADE_KITCHEN_ARCHIVE to a local cascade-kitchen-v1.tar.gz."
        ) from exc
    print(f"[cascade-install] Kitchen: installed {len(expected)} files with verified SHA-256 checksums.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, help="Install a local copy of the versioned archive")
    parser.add_argument("--check", action="store_true", help="Verify every kitchen file without downloading")
    args = parser.parse_args()
    try:
        if args.check:
            problems = kitchen_problems(args.repo)
            if problems:
                print("\n".join(problems[:8]), file=sys.stderr)
                if len(problems) > 8:
                    print(f"... and {len(problems) - 8} more kitchen asset problems.", file=sys.stderr)
                print("Run the Spark installer or scripts/kitchen_assets.py to fetch the kitchen.", file=sys.stderr)
                return 1
            print("Kitchen assets verified.")
        else:
            prepare_kitchen(args.repo, archive=args.archive)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
