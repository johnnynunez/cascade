"""Admit a portable, separately provisioned source and data distribution offline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time

BUILD_INPUTS = {
    "openclaw": {"sha256": "73c595199cd2a4cc46e503e09101318f09c21ca878596f662a540d32cd7243bb",
                 "size_bytes": 90404527},
    "clip": {"sha256": "a6612386ce34f6e08591c603e67855e93abf3b7b418dc81e68b28d891ac5bcd8",
             "size_bytes": 4339014},
}
LAYOUT = "cascade-kitchen-v1"
ASSET_MANIFEST = Path(__file__).with_name("bundle_assets.json")
RUNTIME_FILES = (
    "LICENSE", "pyproject.toml", "src/cascade/__init__.py",
    "scripts/isaac_bridge.py", "scripts/isaac_runtime.py",
    "demo/isaac_scene.py", "demo/serve_isaac_view.py", "demo/scene/kitchen_config.json",
    "demo/scene/props/LICENSE.txt", "deploy/runtime/runtime.py", "deploy/runtime/brain.py",
    "deploy/runtime/brain_qwen.json", "deploy/runtime/web_runtime.py",
    "web/guide/index.html", "web/openclaw-ui/paai-light.js",
    "web/openclaw-ui/paai-light.css", "web/openclaw-ui/OPENCLAW-LICENSE.txt",
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as output:
        os.fchmod(output.fileno(), 0o600)
        json.dump(value, output, indent=2)
        output.write("\n")
    temporary.replace(path)


def signature(path):
    s = path.stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def member(root, name):
    if (not isinstance(name, str) or not name or any(c in name for c in "\r\n\0")
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name):
        raise ValueError("Distribution members must use canonical relative paths")
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Distribution member is missing or leaves its source: {name}")
    return path


def matches(path, expected, saved):
    return (saved.get("sha256") == expected["sha256"]
            and expected["size_bytes"] == saved.get("size_bytes")
            and all(saved.get(k) == v for k, v in signature(path).items()))


def verify(path, expected, saved, boundary):
    boundary()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular distribution file: {path.name}")
    before = signature(path)
    if before["size_bytes"] != expected["size_bytes"]:
        raise ValueError(f"Distribution file has an unexpected size: {path.name}")
    reused = matches(path, expected, saved)
    if not reused:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                boundary()
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Distribution checksum differs: {path.name}")
    if signature(path) != before:
        raise ValueError(f"Distribution file changed during verification: {path.name}")
    return {**before, "sha256": expected["sha256"]}, reused


def manifest_record(path, profile):
    raw = path.read_bytes()
    record = json.loads(raw)
    if (record.get("schema") != 1 or record.get("upstream_url") != profile["source"]["origin"]
            or record.get("upstream_revision") != profile["source"]["revision"]):
        raise ValueError("Distribution and selected upstream profile differ")
    files = record.get("files")
    if not isinstance(files, dict) or not files or record.get("file_count") != len(files):
        raise ValueError("Distribution needs a complete source inventory")
    for entry in files.values():
        if (not isinstance(entry, dict) or not isinstance(entry.get("size_bytes"), int)
                or isinstance(entry["size_bytes"], bool) or entry["size_bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
                or not isinstance(entry.get("provenance"), str) or not entry["provenance"]):
            raise ValueError("Distribution file receipt is invalid")
    if (record.get("source_bytes") != sum(e["size_bytes"] for e in files.values())
            or record.get("build_inputs") != BUILD_INPUTS):
        raise ValueError("Distribution byte count or pinned build inputs differ")
    return record, hashlib.sha256(raw).hexdigest()


def asset_inventory():
    return json.loads(ASSET_MANIFEST.read_text())


def runtime_inventory(source, record):
    """Require the relocated runtime and its separately licensed data inputs."""
    if record.get("layout") != LAYOUT:
        raise ValueError("The bundle uses an unsupported runtime layout")
    files = record["files"]
    missing = set(RUNTIME_FILES) - files.keys()
    assets = asset_inventory()
    missing.update(assets["files"].keys() - files.keys())
    if missing:
        raise ValueError("The runtime bundle is incomplete: " + ", ".join(sorted(missing)))
    for name, expected in assets["files"].items():
        if any(files[name].get(key) != expected[key] for key in ("sha256", "size_bytes")):
            raise ValueError(f"The bundle differs from the pinned asset inventory: {name}")
    config = json.loads(member(source, "demo/scene/kitchen_config.json").read_text())
    for name in [config["background"], *(p["asset"] for p in config["props"] if p.get("asset"))]:
        asset = (source / "demo/scene" / name).resolve()
        if not asset.is_relative_to(source / "demo") or str(asset.relative_to(source)) not in files:
            raise ValueError("Scene configuration references an asset outside the admitted bundle")


def admitted_openclaw(here):
    record = json.loads((here / "publication/LOCAL_INPUTS.json").read_text())
    entry = record["openclaw"]
    archive = Path(entry["path"])
    if (archive.name != "openclaw.tar.zst" or archive.is_symlink() or not archive.is_file()
            or not matches(archive, BUILD_INPUTS["openclaw"], entry)):
        raise ValueError("Local OpenClaw input changed; run prepare to verify it again")
    return archive


def prepare(here, profile, source, manifest, openclaw, clip, *, boundary):
    caller_boundary = boundary
    deadline = time.monotonic() + 900
    def boundary():
        caller_boundary()
        if time.monotonic() >= deadline:
            raise TimeoutError("Offline distribution admission exceeded 900 seconds")
    boundary()
    source = source.resolve()
    record, manifest_sha = manifest_record(manifest, profile)
    runtime_inventory(source, record)
    if not (source / "src/cascade").is_dir():
        raise ValueError("The supplied source needs the complete CASCADE distribution")
    previous = read_optional(here / "publication/STAGED.json")
    saved = (previous.get("deploy_receipts", {})
             if previous.get("staged_root") == str(source) else {})
    receipts, reused = {}, 0
    for name, entry in sorted(record["files"].items()):
        receipt, was_reused = verify(member(source, name), entry, saved.get(name, {}), boundary)
        receipts[name] = {**receipt, "provenance": entry["provenance"]}
        reused += was_reused
    previous_inputs = read_optional(here / "publication/LOCAL_INPUTS.json")
    inputs = {}
    input_reused = 0
    for name, path in (("openclaw", openclaw), ("clip", clip)):
        if path.is_symlink():
            raise ValueError("Build inputs must be regular files, not symlinks")
        path = path.resolve()
        if name == "openclaw" and path.name != "openclaw.tar.zst":
            raise ValueError("The supplied OpenClaw archive must be named openclaw.tar.zst")
        old = previous_inputs.get(name, {})
        receipt, was_reused = verify(path, BUILD_INPUTS[name],
                                    old if old.get("path") == str(path) else {}, boundary)
        inputs[name] = {**receipt, "path": str(path)}
        input_reused += was_reused
    destination = here / "container/clip-source.tar.gz"
    if destination.is_symlink():
        raise ValueError("The local CLIP build input must not be a symlink")
    if destination.resolve() != clip.resolve():
        if destination.exists():
            receipt, _ = verify(destination, BUILD_INPUTS["clip"],
                                previous_inputs.get("clip", {}), boundary)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
            boundary()
            shutil.copyfile(clip, temporary)
            receipt, _ = verify(temporary, BUILD_INPUTS["clip"], {}, boundary)
            boundary()
            temporary.replace(destination)
            receipt.update(signature(destination))
        inputs["clip"] = {**receipt, "path": str(destination.resolve())}
    boundary()
    for name, receipt in receipts.items():
        if not matches(member(source, name), receipt, receipt):
            raise ValueError("Distribution changed before local admission completed")
    for name, receipt in inputs.items():
        if not matches(Path(receipt["path"]), BUILD_INPUTS[name], receipt):
            raise ValueError("Build input changed before local admission completed")
    staged = {"schema": 1, "status": "COMPLETE", "prepared_at": stamp(),
              "staged_root": str(source), "upstream_revision": record["upstream_revision"],
              "upstream_url": record["upstream_url"], "deploy_files": sorted(receipts),
              "deploy_file_count": len(receipts), "deploy_bytes": record["source_bytes"],
              "deploy_receipts": receipts,
              "source_sha256": {name: r["sha256"] for name, r in receipts.items()},
              "portable_manifest_sha256": manifest_sha, "live_brev_verified": False}
    # Keep additional provenance when adopting the same unchanged source.
    if previous.get("staged_root") == str(source) and previous.get("deploy_receipts") == receipts:
        staged = {**previous, **staged, "source_sha256": {
            **previous.get("source_sha256", {}), **staged["source_sha256"]}}
    boundary()
    atomic(here / "publication/LOCAL_INPUTS.json", inputs)
    atomic(here / "publication/STAGED.json", staged)
    return {"status": "PREPARED_LOCAL", "source_files": len(receipts),
            "source_bytes": record["source_bytes"], "source_receipts_reused": reused,
            "source_files_hashed": len(receipts) - reused, "build_inputs_reused": input_reused,
            "portable_manifest_sha256": manifest_sha, "source_files_copied": 0,
            "downloads": 0, "live_brev_verified": False}
