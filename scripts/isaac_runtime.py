#!/usr/bin/env python3
"""Discover Isaac assets without importing Kit or assuming a source build."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import tomllib


def _check_kit(path: Path) -> Path:
    try:
        with path.open("rb") as stream:
            version = tomllib.load(stream).get("package", {}).get("version")
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"invalid Isaac experience TOML: {path}") from exc
    if version != "6.1.0":
        raise RuntimeError(f"Isaac experience package.version=6.1.0 required; {path} declares {version!r}")
    # Keep the release/apps spelling: source builds symlink .kit files (or
    # apps itself) into source/apps. Kit anchors ${app}/../extsDeprecated
    # and extscache to the supplied path; resolve() loses the built tree.
    return path.absolute()


def find_experience(engine: str, *, release=None, package_roots=None) -> Path:
    if engine not in ("newton", "physx"):
        raise ValueError(f"unknown physics engine: {engine}")
    name = "isaacsim.exp.full.newton.kit" if engine == "newton" else "isaacsim.exp.full.kit"
    release = release or os.environ.get("ISAACSIM_PATH")
    if release:
        # Explicit source roots are authoritative. A missing/stale app must
        # not silently select a wheel belonging to another installation.
        return _check_kit(Path(release).expanduser() / "apps" / name)
    roots = []
    if package_roots is None:
        try:
            distribution = importlib.metadata.distribution("isaacsim")
            roots.append(Path(str(distribution.locate_file("isaacsim"))))
        except importlib.metadata.PackageNotFoundError:
            pass
    else:
        roots.extend(Path(root) for root in package_roots)
    for root in roots:
        candidate = root / "apps" / name
        if candidate.is_file():
            return _check_kit(candidate)
    raise FileNotFoundError(
        f"Isaac {engine} experience {name} not found; install isaacsim[all,extscache]==6.1.0.0 "
        "in the Isaac Python environment, or set ISAACSIM_PATH to a complete release"
    )


def installation_info() -> dict:
    import sys

    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version is not None and version != "6.1.0.0":
        raise RuntimeError(f"Isaac Sim 6.1.0.0 required; selected Python has {version}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Isaac Sim 6.1.0.0 requires Python 3.12")
    release = os.environ.get("ISAACSIM_PATH")
    if release:
        root, selected = Path(release).expanduser().resolve(), Path(sys.executable).resolve()
        embedded = {p.resolve() for p in (root / "kit/python/bin").glob("python*") if p.is_file()}
        if not selected.is_relative_to(root) and selected not in embedded:
            raise RuntimeError("selected Python does not belong to ISAACSIM_PATH; use that release's python.sh")
        layout, version = "source", "6.1.0"
    else:
        layout = "wheel"
        if version is None:
            raise RuntimeError("Isaac Sim 6.1.0.0 is not installed in the selected Python")
    return {"layout": layout, "version": version, "python": sys.executable, "newton_experience": str(find_experience("newton"))}


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    try:
        print(json.dumps(installation_info()))
    except (RuntimeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(3)
