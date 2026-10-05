#!/usr/bin/env python3
"""Explicit, offline admission of the downloaded Isaac Lab MicroDuck USD folder as a CASCADE bundle.

Input: a local directory laid out as ``<root>/isaaclab-microduck-usd/<file>`` (the
downloaded folder) that passes ``scripts/microduck_assets.py --check`` against
``assets/microduck/isaaclab-microduck-usd-manifest.json``. Output: a NEW bundle
directory (never overwritten) holding ``usd/<variant>.usd``, the folder's
``LICENSE`` and ``ATTRIBUTION.txt``, provenance records, ``receipt.json`` and
``receipt.sha256``.

Nothing here opens a network connection, imports USD or runs an engine. The
receipt binds bytes and provenance only; engine, actuator-fidelity, locomotion
and physical statuses start as ``unverified`` and only a later receipt may
change them. The USD files embed Pollen Robotics meshes that upstream declares
Creative Commons BY-SA-NC (version unspecified): ``--accept-model-license``
acknowledges that notice, it does not grant commercial/event use.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat

REPO = Path(__file__).resolve().parents[1]
FOLDER_MANIFEST = REPO / "assets/microduck/isaaclab-microduck-usd-manifest.json"
BASE_MANIFEST = REPO / "assets/microduck/manifest.json"
SOURCE_ID = "isaaclab-microduck-usd"
# Variant -> published file. Only ``allcollisions`` is an admitted CASCADE
# candidate (the converted bundle it replaces was also the all-collisions robot).
VARIANTS = {"allcollisions": "microduck_allcollisions.usd"}
REFERENCE_FILES = ("LICENSE", "ATTRIBUTION.txt")
BUNDLE_KIND = "external-usd"
SCHEMA_VERSION = 2


def _assets_module():
    spec = importlib.util.spec_from_file_location("cascade_microduck_assets", REPO / "scripts/microduck_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"not a regular file: {path}")
    return path


def _write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def _copy_verified(source, target, row):
    source = _regular(source)
    if source.stat().st_size != row["size"] or _sha256(source) != row["sha256"]:
        raise ValueError(f"folder file does not match the manifest pin: {row['path']}")
    _write_new(target, source.read_bytes())
    if target.stat().st_size != row["size"] or _sha256(target) != row["sha256"]:
        raise ValueError(f"copied bytes differ from the manifest pin: {row['path']}")


def build(*, source_directory, destination, variant, manifest=FOLDER_MANIFEST, base_manifest=BASE_MANIFEST,
          accept_model_license=False):
    """Create ``destination`` from a verified local folder; refuse to touch an existing directory."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; admitted variants: {sorted(VARIANTS)}")
    assets = _assets_module()
    manifest_bytes = Path(manifest).read_bytes()
    base_bytes = Path(base_manifest).read_bytes()
    data = assets.load_manifest(Path(manifest))
    assets.load_manifest(Path(base_manifest))
    if set(data["sources"]) != {SOURCE_ID} or data["sources"][SOURCE_ID]["kind"] != "folder":
        raise ValueError("manifest must pin exactly the Isaac Lab MicroDuck USD folder")
    if any(row["license"] == "models" for row in data["files"]) and accept_model_license is not True:
        raise ValueError("read the separate NC/SA model notice and pass --accept-model-license")
    source_root = Path(source_directory).expanduser().resolve(strict=True)
    checked = assets.check(data, source_root)
    if not checked["ok"]:
        raise ValueError("folder does not pass the manifest check: " + "; ".join(map(str, checked["errors"][:3])))
    destination = Path(destination).expanduser().absolute()
    if destination.exists():
        raise ValueError("destination already exists; bundles are never overwritten")
    rows = {row["source_path"]: row for row in data["files"]}
    wanted = (VARIANTS[variant], *REFERENCE_FILES)
    missing = [name for name in wanted if name not in rows]
    if missing:
        raise ValueError(f"manifest lacks the admitted files: {missing}")
    destination.mkdir(parents=True)
    for name in wanted:
        _copy_verified(source_root / rows[name]["path"], destination / "usd" / name, rows[name])
    _write_new(destination / "provenance" / "manifest.json", manifest_bytes)
    _write_new(destination / "provenance" / "base-manifest.json", base_bytes)
    source = data["sources"][SOURCE_ID]
    folder = {
        "relative_path": source["relative_path"], "reference": source["reference"],
        "listing_sha256": source["listing_sha256"],
        "files": [{"path": row["source_path"], "size": row["size"], "sha256": row["sha256"], "license": row["license"]}
                  for row in sorted(data["files"], key=lambda r: r["source_path"])],
        "admitted": {"variant": variant, "file": VARIANTS[variant]},
        "transport": "operator-supplied local folder verified by scripts/microduck_assets.py; no download by this tool",
    }
    _write_new(destination / "provenance" / "folder.json", (json.dumps(folder, indent=2, sort_keys=True) + "\n").encode())
    outputs = []
    for path in sorted(p for p in destination.rglob("*") if not p.is_dir()):
        _regular(path)
        outputs.append({"path": path.relative_to(destination).as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)})
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "variant": variant,
        "origin": {"kind": "folder", "source": SOURCE_ID, "relative_path": source["relative_path"],
                   "reference": source["reference"], "file": VARIANTS[variant],
                   "listing_sha256": source["listing_sha256"]},
        "usd_path": f"usd/{VARIANTS[variant]}",
        "outputs": outputs,
        "source_files": [{"path": row["source_path"], "size": row["size"], "sha256": row["sha256"]}
                         for row in sorted(data["files"], key=lambda r: r["source_path"])],
        "provenance": {"sources": data["sources"], "licenses": data["licenses"],
                       "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                       "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest()},
        "versions": {"admission_tool_sha256": _sha256(Path(__file__).resolve())},
        "statuses": {"actuator_fidelity_status": "unverified", "live_engine_status": "unverified",
                     "locomotion_status": "unverified", "historical_checkpoint_training_match": "unverified",
                     "physical_validation": "none"},
        "authoring": {"actuators": "fourteen NewtonActuator prims with NewtonBamDriveAPI are authored in the USD; "
                                   "the CASCADE owner deactivates them and binds its pinned BAM adapter, recording "
                                   "their authored coefficients as provenance",
                      "drives": "zero-gain PhysicsDriveAPI:angular on the servo joints is removed in the runtime layer"},
    }
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    _write_new(destination / "receipt.json", raw)
    _write_new(destination / "receipt.sha256", (digest + "\n").encode())
    return {"ok": True, "bundle": str(destination), "receipt_sha256": digest, "variant": variant,
            "usd_path": receipt["usd_path"], "asset_sha256": rows[VARIANTS[variant]]["sha256"], "outputs": len(outputs)}


def check(bundle, expected_sha256=None):
    """Offline verification through the runtime's own bundle verifier."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from cascade.sim.microduck_newton import verify_bundle
    root = Path(bundle).expanduser().absolute()
    digest = expected_sha256 or (root / "receipt.sha256").read_text().strip()
    result = verify_bundle(root, expected_sha256=digest)
    return {"ok": True, "bundle": result["bundle"], "kind": result["kind"], "variant": result.get("variant"),
            "asset": result["asset"], "asset_sha256": result["asset_sha256"],
            "asset_receipt_sha256": result["asset_receipt_sha256"], "outputs": len(result["receipt"]["outputs"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="create a new bundle from a verified local folder")
    mode.add_argument("--check", action="store_true", help="verify an existing bundle offline")
    parser.add_argument("--source-directory", type=Path, help="local root with <source>/<file> layout (the downloaded folder)")
    parser.add_argument("--destination", type=Path, help="new bundle directory")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="allcollisions")
    parser.add_argument("--manifest", type=Path, default=FOLDER_MANIFEST)
    parser.add_argument("--base-manifest", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--accept-model-license", action="store_true")
    parser.add_argument("--bundle", type=Path, help="bundle to check")
    parser.add_argument("--expected-sha256", help="independently recorded receipt digest")
    args = parser.parse_args(argv)
    try:
        if args.build:
            if args.source_directory is None or args.destination is None:
                parser.error("--build requires --source-directory and --destination")
            result = build(source_directory=args.source_directory, destination=args.destination, variant=args.variant,
                           manifest=args.manifest, base_manifest=args.base_manifest,
                           accept_model_license=args.accept_model_license)
        else:
            if args.bundle is None:
                parser.error("--check requires --bundle")
            result = check(args.bundle, args.expected_sha256)
    except (ValueError, OSError, KeyError) as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
