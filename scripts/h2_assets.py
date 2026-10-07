#!/usr/bin/env python3
"""Fetch and verify the pinned Unitree H2 policy bundle (never committed to git).

    .venv/bin/python scripts/h2_assets.py --repo .          # fetch policy.pt into runs/.install-cache/h2/
    .venv/bin/python scripts/h2_assets.py --repo . --check  # verify the cache and the vendored YAML pins

Every byte is checked against ``configs/h2/bundle.json`` before it is kept; a
mismatch leaves no partial file behind. The USD stays on the asset root (the
owner opens it from there and binds the composed stage identity at open).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

CACHE = Path("runs/.install-cache/h2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, expected_sha256: str, destination: Path, timeout_s: float) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=".fetch-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout_s) as response:
            if response.status != 200:
                raise RuntimeError(f"{url}: HTTP {response.status}")
            for chunk in iter(lambda: response.read(1 << 20), b""):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"{url}: SHA-256 {digest.hexdigest()[:12]}… differs from the pin {expected_sha256[:12]}…")
        os.replace(tmp_name, destination)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return {"url": url, "bytes": size, "sha256": digest.hexdigest(), "path": str(destination)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."), help="CASCADE repository root")
    parser.add_argument("--check", action="store_true", help="verify only; never download")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    manifest = json.loads((repo / "configs/h2/bundle.json").read_text())
    files = manifest["files"]
    report = {"asset_root": manifest["asset_root"], "files": {}}
    status = 0
    for name, entry in files.items():
        if isinstance(entry.get("vendored"), str):
            path = repo / entry["vendored"]
            ok = path.exists() and _sha256(path) == entry["sha256"]
            report["files"][name] = {"path": str(path), "vendored": True, "verified": ok}
            status |= 0 if ok else 2
            continue
        if name == "H2.usda":
            report["files"][name] = {"on_asset_root": f"{manifest['asset_root']}/{entry['path']}", "fetched": False}
            continue
        destination = repo / CACHE / name
        if destination.exists() and _sha256(destination) == entry["sha256"]:
            report["files"][name] = {"path": str(destination), "verified": True, "bytes": destination.stat().st_size}
            continue
        if args.check:
            report["files"][name] = {"path": str(destination), "verified": False,
                                     "reason": "missing or mismatching; run without --check to fetch"}
            status |= 4
            continue
        try:
            fetched = _fetch(f"{manifest['asset_root']}/{entry['path']}", entry["sha256"], destination, args.timeout_s)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed into a partial cache
            report["files"][name] = {"path": str(destination), "verified": False, "error": str(exc)}
            status |= 8
            continue
        report["files"][name] = {**fetched, "verified": True}
    report["ok"] = status == 0
    print(json.dumps(report, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
