"""Bind reviewed kitchen sources and release bytes to the live scene identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


SCENE_NAME = "paai-cocina-asier-v1"
MANIFEST = "demo/scene/own_assets.json"
BUNDLE_MANIFEST = "demo/scene/cocina_asier_bundle.json"
BUNDLE_ROOT = "demo/scene/cocina_asier"
BUNDLE_URL = "https://github.com/johnnynunez/cascade/releases/download/kitchen-asier-v1/cocina-asier-v1.zip"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024 * 1024
BUNDLE_REQUIRED_FILES = frozenset({"cocina_asier.usdc", "geometry-audit.json", "NOTICE.md", "sources.json"})
SOURCE_FILES = (
    "scripts/isaac_bridge.py",
    "demo/isaac_scene.py",
    "demo/own_kitchen.py",
    "demo/own_kitchen_props.py",
    "demo/cocina_asier.py",
    "scripts/build_cocina_asier.py",
    "demo/scene_identity.py",
    "demo/scene/cocina_asier_sources.json",
    "demo/scene/cocina_asier_audit.json",
    BUNDLE_MANIFEST,
    "demo/scene/NOTICE.md",
    "demo/scene/props/LICENSE.txt",
    "demo/scene/props/lemon.usda",
    "demo/scene/props/tomato_can.usda",
)


def validate_recipe(config: dict) -> None:
    """Only reviewed visuals may participate in an original-kitchen identity."""
    if (not isinstance(config, dict) or config.get("version") != 1 or config.get("scene_name") != SCENE_NAME
            or "background" in config):
        raise ValueError("Expected the reviewed Cocina Asier kitchen identity")
    props = config.get("props")
    if (not isinstance(props, list) or len(props) != 3
            or any(not isinstance(prop, dict) for prop in props)
            or {prop.get("name") for prop in props} != {"tomato_can", "lemon", "orange"}):
        raise ValueError("Kitchen recipe must name exactly the three authored prop visuals")
    for prop in props:
        if prop["name"] == "orange":
            if "asset" in prop or prop.get("visual") != "procedural_orange":
                raise ValueError("Orange must use the original procedural visual")
        elif prop.get("asset") != f"props/{prop['name']}.usda":
            raise ValueError("Kitchen prop references an unreviewed visual asset")


def relative_name(name: str) -> PurePosixPath:
    """Reject alternate spellings, traversal, and platform-dependent paths."""
    if not isinstance(name, str) or not name or ":" in name:
        raise ValueError(f"Invalid kitchen source path: {name}")
    relative = PurePosixPath(name)
    if (relative.is_absolute() or str(relative) != name or ".." in relative.parts
            or "\\" in name or any(c in name for c in "\r\n\0")):
        raise ValueError(f"Invalid kitchen source path: {name}")
    return relative


def local_file(repo: Path, name: str) -> Path:
    """Admit only canonical regular files inside this source checkout."""
    relative = relative_name(name)
    path = repo
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Kitchen source must not be a symlink: {name}")
    if not path.resolve().is_relative_to(repo.resolve()) or not path.is_file():
        raise ValueError(f"Missing kitchen source: {name}")
    return path


def file_receipt(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def validate_receipt(entry: dict, name: str) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid kitchen file receipt: {name}")
    size, digest = entry.get("size_bytes"), entry.get("sha256")
    if (type(size) is not int or size < 1 or not isinstance(digest, str)
            or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError(f"Invalid kitchen file receipt: {name}")


def validate_bundle_manifest(data: dict) -> dict:
    """Validate prospective release data without changing a tracked receipt."""
    if (not isinstance(data, dict) or data.get("schema") != 1
            or data.get("scene_name") != SCENE_NAME or data.get("root") != BUNDLE_ROOT
            or not isinstance(data.get("files"), dict)
            or not len(BUNDLE_REQUIRED_FILES) <= len(data["files"]) <= 4096):
        raise ValueError("Invalid Cocina Asier release inventory")
    archive = data.get("archive")
    validate_receipt(archive, "release archive")
    if archive.get("url") != BUNDLE_URL or archive["size_bytes"] > MAX_ARCHIVE_BYTES:
        raise ValueError("Expected the pinned, bounded Cocina Asier release URL")
    if not BUNDLE_REQUIRED_FILES <= data["files"].keys():
        raise ValueError("Kitchen release requires its room layer, geometry audit, NOTICE.md, and sources.json")
    for name, entry in data["files"].items():
        relative = relative_name(name)
        if name not in BUNDLE_REQUIRED_FILES and (
                len(relative.parts) != 2 or relative.parts[0] != "textures"):
            raise ValueError(f"Unexpected kitchen release member: {name}")
        validate_receipt(entry, name)
    if sum(entry["size_bytes"] for entry in data["files"].values()) > MAX_BUNDLE_BYTES:
        raise ValueError("Kitchen release exceeds its unpacked byte budget")
    return data


def bundle_manifest(repo: Path) -> dict:
    return validate_bundle_manifest(json.loads(local_file(repo, BUNDLE_MANIFEST).read_text()))


def source_manifest(repo: Path) -> dict:
    """Maintainer operation: describe original source bytes, without downloads."""
    files = {}
    for name in SOURCE_FILES:
        files[name] = file_receipt(local_file(repo, name))
    return {"schema": 1, "scene_name": SCENE_NAME, "files": files}


def kitchen_manifest(repo: Path) -> dict:
    data = json.loads(local_file(repo, MANIFEST).read_text())
    if (not isinstance(data, dict) or data.get("schema") != 1 or data.get("scene_name") != SCENE_NAME
            or not isinstance(data.get("files"), dict)
            or set(data["files"]) != set(SOURCE_FILES)):
        raise ValueError("Kitchen manifest must contain exactly the authored source inventory")
    for name, entry in data["files"].items():
        validate_receipt(entry, name)
    return data["files"]


def source_problems(repo: Path, *, checksums: bool = True) -> list[str]:
    try:
        files = kitchen_manifest(repo)
        validate_recipe(json.loads(local_file(repo, "demo/scene/kitchen_config.json").read_text()))
    except (OSError, ValueError, TypeError) as exc:
        return [str(exc)]
    problems = []
    for name, expected in files.items():
        try:
            path = local_file(repo, name)
            if path.stat().st_size != expected["size_bytes"]:
                problems.append(f"Kitchen source size mismatch: {name}")
            elif checksums and file_receipt(path)["sha256"] != expected["sha256"]:
                problems.append(f"Kitchen source checksum mismatch: {name}")
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
    return problems


def bundle_problems(repo: Path, *, checksums: bool = True, manifest: dict | None = None) -> list[str]:
    try:
        data = bundle_manifest(repo) if manifest is None else validate_bundle_manifest(manifest)
        root = repo / BUNDLE_ROOT
        if root.is_symlink():
            raise ValueError("Kitchen release directory must not be a symlink")
        members = list(root.rglob("*"))
        if any(path.is_symlink() for path in members):
            raise ValueError("Kitchen release members must not be symlinks")
        actual = {p.relative_to(root).as_posix() for p in members if not p.is_dir()}
        extra = actual - data["files"].keys()
        if extra:
            raise ValueError("Unmanifested kitchen release members: " + ", ".join(sorted(extra)))
        problems = []
        for name, expected in data["files"].items():
            try:
                path = local_file(repo, f"{BUNDLE_ROOT}/{name}")
                if path.stat().st_size != expected["size_bytes"]:
                    problems.append(f"Kitchen release size mismatch: {name}")
                elif checksums and file_receipt(path)["sha256"] != expected["sha256"]:
                    problems.append(f"Kitchen release checksum mismatch: {name}")
            except (OSError, ValueError) as exc:
                problems.append(str(exc))
        return problems
    except (OSError, ValueError, TypeError) as exc:
        return [str(exc)]


def kitchen_problems(repo: Path, *, checksums: bool = True) -> list[str]:
    problems = source_problems(repo, checksums=checksums)
    return problems or bundle_problems(repo, checksums=checksums)


def scene_identity(config_path: Path) -> dict:
    """Bind configuration AND verified generator/material bytes to a live scene."""
    config_path = Path(config_path).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not config_path.is_relative_to(repo):
        raise ValueError("Kitchen configuration must belong to this checkout")
    raw = config_path.read_bytes()
    config = json.loads(raw)
    validate_recipe(config)
    problems = kitchen_problems(repo)
    if problems:
        raise ValueError("Kitchen source verification failed: " + "; ".join(problems))
    inventory = json.dumps({"sources": kitchen_manifest(repo), "release": bundle_manifest(repo)},
                           sort_keys=True, separators=(",", ":")).encode()
    assets_digest = hashlib.sha256(inventory).hexdigest()
    return {
        "scene_name": SCENE_NAME,
        "scene_config": str(config_path),
        "scene_config_sha256": hashlib.sha256(raw).hexdigest(),
        "scene_assets_sha256": assets_digest,
        "scene_content_sha256": hashlib.sha256(raw + b"\0" + inventory).hexdigest(),
    }
