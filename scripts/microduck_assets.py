#!/usr/bin/env python3
"""Explicit, pinned MicroDuck admission; no imports initiate network or writes.

--check is read-only/offline. --fetch downloads individual hashed files (NO
archives/extraction), or admits an existing mirror via --source-directory.
Model/mesh/weight bytes must live OUTSIDE this checkout. Existing files are
verified and never overwritten, including corrupt/partial ones. Models have
separate, version-unspecified NC/SA terms: --accept-model-license acknowledges
the notice, not commercial-use permission. Conversion is NOT performed here.

API: load_manifest(path), check(manifest,destination),
fetch(manifest,destination,*,accept_model_license=False,source_directory=None).
Destination layout is <source id>/<source_path>, preserving upstream paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "assets/microduck/manifest.json"


def _path(value):
    if (not isinstance(value, str) or not value or any(ord(c) < 32 for c in value)
            or any(c in value for c in "\\:%?#") or value.startswith("/")
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_folder_source(source):
    """A downloaded folder has no commit; pin the exact file listing instead.

    ``listing_sha256`` is the SHA-256 of the canonical listing (one line per
    file: ``<name> <size> <sha256>``, sorted by name) and identifies the
    folder's content. ``reference`` is a public URL describing the folder.
    Nothing is ever downloaded for this kind: it is admitted only from an
    operator-supplied local directory.
    """
    if set(source) != {"kind", "relative_path", "reference", "listing_sha256"}:
        raise ValueError("folder source requires exactly kind, relative_path, reference and listing_sha256")
    path = source["relative_path"]
    if (not isinstance(path, str) or not path or path.startswith("/") or path.endswith("/") or "//" in path
            or any(part in ("", ".", "..") for part in path.split("/"))
            or any(c in path for c in "\\:%?#") or any(ord(c) < 32 for c in path)):
        raise ValueError("folder relative_path must be a canonical relative path")
    reference = source["reference"]
    if not isinstance(reference, str) or not reference.startswith("https://"):
        raise ValueError("folder reference must be a public https URL")
    digest = source["listing_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("folder listing_sha256 must be exact lowercase hexadecimal")


def listing_digest(rows):
    """Canonical listing digest for a folder source: ``<name> <size> <sha256>`` lines sorted by name."""
    lines = sorted(f"{row['source_path']} {row['size']} {row['sha256']}" for row in rows)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def _validate(data):
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported manifest schema_version")
    sources, licenses, files = data.get("sources"), data.get("licenses"), data.get("files")
    if not isinstance(sources, dict) or not sources or not isinstance(licenses, dict) or not licenses:
        raise ValueError("manifest requires source and separate license records")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest requires a nonempty files list")
    for key, source in sources.items():
        _path(key)
        if "/" in key or not isinstance(source, dict):
            raise ValueError("invalid source id/record")
        if source.get("kind") == "folder":
            _validate_folder_source(source)
            continue
        repo = source.get("repository")
        if (source.get("kind") not in ("github", "huggingface") or not isinstance(repo, str)
                or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None
                or any(p in (".", "..") for p in repo.split("/"))):
            raise ValueError("source must specify a GitHub or Hugging Face owner/repository")
        revision = source.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("source revision must be a full lowercase immutable commit SHA")
    for record in licenses.values():
        if (not isinstance(record, dict) or not isinstance(record.get("license"), str)
                or not record["license"] or not isinstance(record.get("evidence"), str)
                or not record["evidence"].startswith("https://")):
            raise ValueError("license records require a declaration and evidence URL")
    seen = set()
    for file in files:
        if not isinstance(file, dict):
            raise ValueError("invalid file record")
        path, origin = _path(file.get("path")), _path(file.get("source_path"))
        source = file.get("source")
        if not isinstance(source, str) or source not in sources or path != f"{source}/{origin}":
            raise ValueError("file must preserve <source>/<source_path> layout")
        if not isinstance(file.get("license"), str) or file["license"] not in licenses:
            raise ValueError("unknown file license")
        digest = file.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("file SHA-256 must be exact lowercase hexadecimal")
        if type(file.get("size")) is not int or file["size"] < 0:
            raise ValueError("file size must be a nonnegative integer")
        if path in seen:
            raise ValueError("duplicate artifact path")
        seen.add(path)
    for path in seen:
        if any(parent.as_posix() in seen for parent in Path(path).parents):
            raise ValueError("artifact path collides with parent artifact")
    for key, source in sources.items():
        if source.get("kind") == "folder":
            rows = [row for row in files if row["source"] == key]
            if not rows or listing_digest(rows) != source["listing_sha256"]:
                raise ValueError(f"folder listing_sha256 does not match the listed files of {key}")
    return data


def load_manifest(path=DEFAULT_MANIFEST):
    """Parse strictly and reject mutable revisions, ambiguous paths and hashes."""
    with Path(path).open(encoding="utf-8") as stream:
        return _validate(json.load(stream, object_pairs_hook=_unique_object))


def _safe(path):
    # Check every existing ancestor as well as the file, before resolving it.
    # resolve() alone would hide symlinks, including dangling leaf symlinks.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f"symlink refused: {part}")
    return path


def _root(destination):
    root = _safe(Path(os.path.abspath(destination)))
    if root == REPO or REPO in root.parents:
        raise ValueError("external models/weights must be outside the CASCADE checkout")
    if root.exists() and not root.is_dir():
        raise ValueError("destination must be a directory")
    return root


def _target(root, relative):
    target = _safe(root / _path(relative))
    for parent in target.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"non-directory artifact parent: {parent}")
    return target


def _verify(path, row):
    if not path.exists():
        return "missing"
    # Never read devices/FIFOs. O_NOFOLLOW also rejects a swapped leaf symlink.
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"artifact is not a regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"artifact is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > row["size"]:
                return "size mismatch"
            digest.update(block)
    if size != row["size"]:
        return "size mismatch"
    if digest.hexdigest() != row["sha256"]:
        return "SHA-256 mismatch"
    return None


def check(manifest, destination):
    """Verify every declared file without creating anything or using network."""
    data, root = _validate(manifest), _root(destination)
    errors = []
    for row in data["files"]:
        reason = _verify(_target(root, row["path"]), row)
        if reason:
            errors.append({"path": row["path"], "error": reason})
    return {"ok": not errors, "checked": len(data["files"]), "errors": errors}


def _url(source, row):
    if source["kind"] == "folder":
        raise ValueError("folder sources are admitted only from a local directory (--source-directory); "
                         "this tool never downloads them")
    if source["kind"] == "github":
        base = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}"
    else:
        base = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}"
    return f"{base}/{row['source_path']}"


def fetch(manifest, destination, *, accept_model_license=False, source_directory=None):
    """Admit verified bytes, never replace an existing artifact.

    A failed batch may leave earlier VERIFIED files for resume, but never a
    corrupt downloaded target. The destination must be operator-owned and not
    concurrently edited; ancestor symlinks are rejected, not a substitute for
    isolation against a hostile process racing directory renames.
    """
    data, root = _validate(manifest), _root(destination)
    if any(row["license"] == "models" for row in data["files"]) and accept_model_license is not True:
        raise ValueError("read the separate NC/SA model notice and pass --accept-model-license")
    source_root = _safe(Path(os.path.abspath(source_directory))) if source_directory is not None else None
    # Preflight the WHOLE destination before any network or writes, so a bad
    # existing file/symlink cannot be silently repaired or hidden by a retry.
    missing = []
    for row in data["files"]:
        target = _target(root, row["path"])
        reason = _verify(target, row)
        if reason == "missing":
            missing.append(row)
        elif reason:
            raise ValueError(f"refusing to overwrite {target}: {reason}")
        if source_root is not None:
            source_path = _target(source_root, row["path"])
            source_error = _verify(source_path, row)
            if source_error:
                raise ValueError(f"local source {source_path}: {source_error}")
    for row in missing:
        target = _target(root, row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        _safe(target.parent)
        if source_root is None:
            from urllib.request import urlopen
            source = urlopen(_url(data["sources"][row["source"]], row), timeout=60)
        else:
            source = _target(source_root, row["path"]).open("rb")
        stage = None
        try:
            with source, tempfile.NamedTemporaryFile(dir=target.parent, prefix=".microduck-", delete=False) as out:
                stage = Path(out.name)
                remaining = row["size"]
                digest = hashlib.sha256()
                while True:
                    block = source.read(min(1024 * 1024, remaining + 1))
                    if not block:
                        break
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError(f"download exceeds pinned size: {row['path']}")
                    out.write(block)
                    digest.update(block)
                if remaining != 0 or digest.hexdigest() != row["sha256"]:
                    raise ValueError(f"download size/SHA-256 mismatch: {row['path']}")
                out.flush()
                os.fsync(out.fileno())
            _safe(target)
            # Atomic, no-clobber publication. Unlike replace(), link() refuses
            # an existing target even if another fetch created it meanwhile.
            os.link(stage, target)
        finally:
            if stage is not None:
                stage.unlink(missing_ok=True)
    return check(data, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-directory", type=Path, help="offline mirror with the manifest's source/path layout")
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args(argv)
    if args.check and (args.source_directory is not None or args.accept_model_license):
        parser.error("--source-directory/--accept-model-license are fetch-only")
    try:
        data = load_manifest(args.manifest)
        result = (fetch(data, args.destination, source_directory=args.source_directory,
                        accept_model_license=args.accept_model_license)
                  if args.fetch else check(data, args.destination))
    except (ValueError, OSError) as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
