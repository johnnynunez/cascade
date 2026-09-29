"""Content identity for the repository-authored kitchen; standard library only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


SCENE_NAME = "paai-own-kitchen-v1"
MANIFEST = "demo/scene/own_assets.json"
SOURCE_FILES = (
    "scripts/isaac_bridge.py",
    "demo/isaac_scene.py",
    "demo/own_kitchen.py",
    "demo/own_kitchen_props.py",
    "demo/scene_identity.py",
    "demo/scene/NOTICE.md",
    "demo/scene/props/LICENSE.txt",
    "demo/scene/props/lemon.usda",
    "demo/scene/props/tomato_can.usda",
)


def validate_recipe(config: dict) -> None:
    """Only reviewed visuals may participate in an original-kitchen identity."""
    if (not isinstance(config, dict) or config.get("version") != 1 or config.get("scene_name") != SCENE_NAME
            or "background" in config):
        raise ValueError("Expected the repository-authored kitchen identity")
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


def local_file(repo: Path, name: str) -> Path:
    """Admit only canonical regular files inside this source checkout."""
    relative = PurePosixPath(name)
    if (relative.is_absolute() or str(relative) != name or ".." in relative.parts
            or "\\" in name or any(c in name for c in "\r\n\0")):
        raise ValueError(f"Invalid kitchen source path: {name}")
    path = repo
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Kitchen source must not be a symlink: {name}")
    if not path.resolve().is_relative_to(repo.resolve()) or not path.is_file():
        raise ValueError(f"Missing kitchen source: {name}")
    return path


def source_manifest(repo: Path) -> dict:
    """Maintainer operation: describe original source bytes, without downloads."""
    files = {}
    for name in SOURCE_FILES:
        raw = local_file(repo, name).read_bytes()
        files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    return {"schema": 1, "scene_name": SCENE_NAME, "files": files}


def kitchen_manifest(repo: Path) -> dict:
    data = json.loads(local_file(repo, MANIFEST).read_text())
    if (not isinstance(data, dict) or data.get("schema") != 1 or data.get("scene_name") != SCENE_NAME
            or not isinstance(data.get("files"), dict)
            or set(data["files"]) != set(SOURCE_FILES)):
        raise ValueError("Kitchen manifest must contain exactly the authored source inventory")
    for name, entry in data["files"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid kitchen source receipt: {name}")
        size, digest = entry.get("size_bytes"), entry.get("sha256")
        if (type(size) is not int or size < 1 or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"Invalid kitchen source receipt: {name}")
    return data["files"]


def kitchen_problems(repo: Path, *, checksums: bool = True) -> list[str]:
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
            elif checksums and hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
                problems.append(f"Kitchen source checksum mismatch: {name}")
        except (OSError, ValueError) as exc:
            problems.append(str(exc))
    return problems


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
    inventory = json.dumps(kitchen_manifest(repo), sort_keys=True, separators=(",", ":")).encode()
    assets_digest = hashlib.sha256(inventory).hexdigest()
    return {
        "scene_name": SCENE_NAME,
        "scene_config": str(config_path),
        "scene_config_sha256": hashlib.sha256(raw).hexdigest(),
        "scene_assets_sha256": assets_digest,
        "scene_content_sha256": hashlib.sha256(raw + b"\0" + inventory).hexdigest(),
    }
