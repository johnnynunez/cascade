"""YAML config loading.

The demo is wired from three profile files plus one main config:

    configs/demo.yaml          - task defaults, safety limits, memory horizon
    configs/cameras/<name>.yaml
    configs/arms/<name>.yaml
    configs/llm/<name>.yaml

Profiles are merged into one `DemoConfig` namespace. Everything is plain
dict-backed so new keys never require code changes here.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the cascade repo root
CONFIG_DIR = PACKAGE_ROOT / "configs"
ASSET_DIR = PACKAGE_ROOT / "assets"


class Cfg:
    """Read-only attribute/dict hybrid over nested config dicts."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError as e:
            raise AttributeError(f"config key missing: {key!r}") from e
        return Cfg(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Cfg(val) if isinstance(val, dict) else val

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Cfg({self._data!r})"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _resolve_paths(data: Any, base: Path) -> Any:
    """Expand `${assets}` and `${repo}` placeholders in string values."""
    if isinstance(data, dict):
        return {k: _resolve_paths(v, base) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_paths(v, base) for v in data]
    if isinstance(data, str):
        return (
            data.replace("${assets}", str(ASSET_DIR))
            .replace("${repo}", str(PACKAGE_ROOT))
        )
    return data


def load_profile(kind: str, name: str, config_dir: Path | None = None) -> Cfg:
    """Load one profile, e.g. load_profile('cameras', 'l515')."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    path = cdir / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (cdir / kind).glob("*.yaml"))
        raise FileNotFoundError(f"no {kind} profile {name!r}; available: {available}")
    return Cfg(_resolve_paths(_load_yaml(path), cdir))


def _deep_merge(base: dict, overlay: dict) -> None:
    """Merge overlay into base in place: dicts recurse, scalars replace."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def booth_mode_enabled() -> bool:
    """CASCADE_BOOTH=1 selects the booth tuning overlay (configs/booth.yaml).
    Whitespace-stripped; the usual negatives all disable it."""
    import os

    return os.environ.get("CASCADE_BOOTH", "").strip().lower() not in (
        "", "0", "false", "no", "off",
    )


def load_demo_config(
    camera: str = "mock",
    arm: str = "mock",
    llm: str = "mock",
    config_dir: Path | None = None,
    cameras: list[str] | None = None,
) -> Cfg:
    """`cameras` (ordered, first = manipulation camera) supersedes `camera`;
    both populate cfg.camera (primary) and cfg.cameras (all).

    When CASCADE_BOOTH is set, configs/booth.yaml is deep-merged on top of
    demo.yaml (bounded worst cases for timed attendee sessions -- see
    docs/BOOTH_RUNBOOK.md §1); every entry point (demo CLI, MCP server,
    dashboard runner) goes through here, so the switch is one env var."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    main = _resolve_paths(_load_yaml(cdir / "demo.yaml"), cdir)
    if booth_mode_enabled():
        booth_path = cdir / "booth.yaml"
        if not booth_path.exists():
            # an explicitly requested overlay must never no-op silently:
            # booth.yaml ships with the repo, so absence = broken checkout
            raise FileNotFoundError(
                f"CASCADE_BOOTH is set but {booth_path} is missing"
            )
        _deep_merge(main, _resolve_paths(_load_yaml(booth_path), cdir))
        main["booth_mode"] = True
    names = [n.strip() for n in (cameras or [camera]) if n and n.strip()]
    cams = []
    for i, name in enumerate(names):
        prof = load_profile("cameras", name, cdir).as_dict()
        prof.setdefault("name", name if names.count(name) == 1 else f"{name}{i}")
        cams.append(prof)
    main["camera"] = cams[0]
    main["cameras"] = cams
    main["arm"] = load_profile("arms", arm, cdir).as_dict()
    main["llm"] = load_profile("llm", llm, cdir).as_dict()
    return Cfg(main)
