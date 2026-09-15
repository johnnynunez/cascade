#!/usr/bin/env python3
"""Assemble an offline runtime distribution from source and separately supplied assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import prepare_bundle as bundle

HERE = Path(__file__).resolve().parent
SOURCE_PATTERNS = (
    "LICENSE", "pyproject.toml", "uv.lock", "src/cascade/**/*.py", "configs/**/*.yaml",
    "scripts/isaac_bridge.py", "scripts/isaac_runtime.py", "scripts/isaac_camera_readback.py",
    "scripts/isaac_materials.py", "scripts/isaac_self_mask.py", "scripts/isaac_launch.py",
    "scripts/newton_mesh_compat.py",
    "scripts/demo_proof.py", "scripts/judge_run.py", "scripts/install_support.py",
    "demo/**/*.py", "demo/scene/kitchen_config.json", "demo/scene/props/LICENSE.txt",
    "demo/kitchen/visitor-instructions.md", "demo/kitchen/dashboard/*.js",
    "deploy/runtime/*.py", "deploy/runtime/*.mjs", "deploy/runtime/brain_qwen.json",
    "deploy/runtime/web/**/*", "deploy/brev/streaming/*.py", "deploy/brev/public.py",
    "deploy/brev/visitor.*", "deploy/brev/visitor_video.py", "deploy/brev/visitor-player.js",
    "deploy/brev/staff.html", "deploy/brev/staff.css", "deploy/brev/staff-*.png",
    "web/guide/*.html", "web/guide/*.js", "web/guide/*.css",
    "web/guide/assets/*LICENSE*", "web/openclaw-ui/paai-light.*", "web/openclaw-ui/*LICENSE*",
    "extensions/chrome/*.js", "extensions/chrome/*.mjs", "extensions/chrome/*.json", "extensions/chrome/*.html",
    "extensions/chrome/*.css",
)


def source_files(checkout):
    names = set()
    for pattern in SOURCE_PATTERNS:
        for path in checkout.glob(pattern):
            relative = path.relative_to(checkout)
            if path.is_dir() or "__pycache__" in relative.parts or path.name.startswith("test_"):
                continue
            name = relative.as_posix()
            bundle.member(checkout, name)
            names.add(name)
    missing = set(bundle.RUNTIME_FILES) - names
    if missing:
        raise ValueError("The source checkout is incomplete: " + ", ".join(sorted(missing)))
    return sorted(names)


def fingerprint(path, boundary):
    before = bundle.signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            boundary()
            digest.update(chunk)
    if before != bundle.signature(path):
        raise ValueError(f"Source changed while creating the bundle: {path.name}")
    return {"sha256": digest.hexdigest(), "size_bytes": before["size_bytes"]}


def copy_member(source, destination, expected, boundary):
    before = bundle.signature(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(1024 * 1024):
            boundary()
            outgoing.write(chunk)
            digest.update(chunk)
    destination.chmod(source.stat().st_mode & 0o777)
    if (before != bundle.signature(source) or digest.hexdigest() != expected["sha256"]
            or destination.stat().st_size != expected["size_bytes"]):
        raise ValueError("A bundle copy changed; preserve this output and use a new output directory")
    return {**bundle.signature(destination), "sha256": expected["sha256"]}


def assemble(checkout, assets, output, profile, *, boundary):
    checkout, assets, output = (p.resolve() for p in (checkout, assets, output))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (checkout, assets)):
        raise ValueError("Bundle output must be separate from the checkout and supplied assets")
    boundary()
    pinned = bundle.asset_inventory()
    entries = {name: {**fingerprint(bundle.member(checkout, name), boundary),
                      "provenance": "CASCADE source checkout"} for name in source_files(checkout)}
    if entries.keys() & pinned["files"].keys():
        raise ValueError("Source files must not override separately supplied assets")
    entries.update(pinned["files"])
    previous = bundle.read_optional(output / "BUILD_RECEIPTS.json")
    saved_inputs = previous.get("inputs", {}) if previous.get("assets_root") == str(assets) else {}
    saved_outputs = previous.get("outputs", {})
    receipts, inputs, copied, reused = {}, {}, 0, 0
    for name, expected in sorted(entries.items()):
        boundary()
        external = name in pinned["files"]
        path = bundle.member(assets if external else checkout, name)
        receipt, _ = bundle.verify(path, expected, saved_inputs.get(name, {}) if external else {}, boundary)
        inputs[name] = receipt
        destination = output / "source" / name
        if not destination.resolve().is_relative_to(output / "source"):
            raise ValueError("A bundle destination leaves its output directory")
        if destination.exists() or destination.is_symlink():
            member = bundle.member(output / "source", name)
            receipts[name], _ = bundle.verify(member, expected, saved_outputs.get(name, {}), boundary)
            reused += 1
        else:
            receipts[name] = copy_member(path, destination, expected, boundary)
            copied += 1
    record = {"schema": 1, "layout": bundle.LAYOUT,
              "upstream_url": profile["source"]["origin"],
              "upstream_revision": profile["source"]["revision"],
              "file_count": len(entries), "source_bytes": sum(p["size_bytes"] for p in entries.values()),
              "build_inputs": bundle.BUILD_INPUTS, "licenses": pinned["licenses"], "files": entries}
    bundle.runtime_inventory(output / "source", record)
    for name, receipt in receipts.items():
        if not bundle.matches(bundle.member(output / "source", name), entries[name], receipt):
            raise ValueError("The assembled bundle changed before publication")
    boundary()
    bundle.atomic(output / "BUILD_RECEIPTS.json", {"assets_root": str(assets),
                  "inputs": inputs, "outputs": receipts})
    bundle.atomic(output / "PORTABLE_BUNDLE.json", record)
    return {"status": "BUNDLE_ASSEMBLED", "files": len(entries), "copied": copied,
            "reused": reused, "downloads": 0, "live_brev_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=HERE.parents[1])
    parser.add_argument("--assets", type=Path, required=True, help="External files at the paths in bundle_assets.json")
    parser.add_argument("--output", type=Path, required=True, help="Separate immutable distribution directory")
    parser.add_argument("--profile", type=Path, default=HERE / "profiles/brev-rtx6000.json")
    args = parser.parse_args()
    deadline = time.monotonic() + 900

    def boundary():
        if (args.checkout / "STOP").exists():
            raise InterruptedError("Repository STOP exists")
        if time.monotonic() >= deadline:
            raise TimeoutError("Offline bundle assembly exceeded 900 seconds")

    result = assemble(args.checkout, args.assets, args.output, json.loads(args.profile.read_text()),
                      boundary=boundary)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
