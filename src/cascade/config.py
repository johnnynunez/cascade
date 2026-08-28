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


def _load_profile_raw(kind: str, name: str, cdir: Path,
                      chain: tuple[str, ...] = ()) -> dict[str, Any]:
    """Profile dict with any `extends:` parent merged underneath it.

    One robot usually needs several profiles that differ in the TRANSPORT only
    -- the SO-101 ships as mock/MuJoCo/serial, all describing the same physical
    arm. Without inheritance every kinematic constant, gripper measurement and
    workspace override has to be copied into each, and the copies drift: the
    one that gets fixed is the one you happened to be running.
    """
    path = cdir / kind / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (cdir / kind).glob("*.yaml"))
        raise FileNotFoundError(f"no {kind} profile {name!r}; available: {available}")
    if name in chain:
        raise ValueError(
            f"{kind} profile `extends:` cycle: {' -> '.join((*chain, name))}"
        )
    data = _load_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _load_profile_raw(kind, str(parent), cdir, (*chain, name))
    _deep_merge(base, data)
    return base


def load_profile(kind: str, name: str, config_dir: Path | None = None) -> Cfg:
    """Load one profile, e.g. load_profile('cameras', 'l515')."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    return Cfg(_resolve_paths(_load_profile_raw(kind, name, cdir), cdir))


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
    arms: list[str] | None = None,
) -> Cfg:
    """`cameras` (ordered, first = manipulation camera) supersedes `camera`;
    both populate cfg.camera (primary) and cfg.cameras (all). `arms` does the
    same for cfg.arm (primary) and cfg.arms (all).

    When CASCADE_BOOTH is set, configs/booth.yaml is deep-merged on top of
    demo.yaml (bounded worst cases for timed attendee sessions -- see
    docs/BOOTH_RUNBOOK.md §1); every entry point (demo CLI, MCP server,
    dashboard runner) goes through here, so the switch is one env var.

    An arm profile may carry an `overrides:` block that is deep-merged into the
    main config LAST, so it also beats the booth overlay. This is what makes
    the framework robot-agnostic in practice: demo.yaml's `safety.workspace`,
    `safety.table_z`, `grasp.topdown_z_max`, `grasp.drop_zone` and friends are
    properties of a PARTICULAR arm on a particular table, and a second robot
    with a 40 cm reach must not silently inherit the first one's 50 cm box.

    The alternative -- each harness patching the constants it remembers at
    runtime -- was tried in benchmark/libero/run_wrc.py and cost two whole
    benchmark runs to the ones it forgot (topdown_z_max, settle timeout).
    Declaring them in the profile means forgetting is impossible.

    WITH SEVERAL ARMS the same reasoning forbids merging every arm's
    `overrides:` into one global blob: two robots' workspace boxes would
    overwrite each other and the last one loaded would silently define the
    safety envelope for BOTH. So only the PRIMARY arm's overrides reach the
    top level (single-arm behaviour, byte for byte), and every arm keeps its
    own resolved view under `cfg.arms[i].resolved` -- which is what
    build_runtime hands to that arm's SafetyHarness."""
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

    arm_names = [n.strip() for n in (arms or [arm]) if n and n.strip()]
    if not arm_names:
        raise ValueError("no arm profile named")
    arm_profiles, arm_overrides = [], []
    for i, name in enumerate(arm_names):
        prof = load_profile("arms", name, cdir).as_dict()
        arm_overrides.append(prof.pop("overrides", None))
        # Names address arms in the rig (skills take an optional `arm` arg).
        # Two of the SAME profile is a legitimate dual-arm setup, so
        # de-duplicate positionally exactly as the camera rig does.
        prof.setdefault(
            "name", name if arm_names.count(name) == 1 else f"{name}{i}"
        )
        arm_profiles.append(prof)

    main["arm"] = arm_profiles[0]
    main["llm"] = load_profile("llm", llm, cdir).as_dict()

    # Snapshot the config BEFORE any arm's overrides land. Every arm's view is
    # built from THIS, so no arm ever inherits another's retuning.
    #
    # Taking the snapshot after the primary merge (the obvious order) is a
    # real defect, caught by tests/test_arm_rig.py: a second arm with no
    # `overrides:` of its own would silently receive the PRIMARY's workspace
    # box and grasp ceiling -- e.g. a Panda running with an SO-101's 0.05 m
    # top-down ceiling. That is the exact class of bug that cost two LIBERO
    # benchmark runs, reintroduced one layer down.
    pre_override = copy.deepcopy(main)

    # Primary arm's overrides define the top-level (single-arm behaviour).
    if arm_overrides[0]:
        _deep_merge(main, arm_overrides[0])

    # Per-arm resolved view: the global config as THIS arm sees it -- the
    # pre-override base plus only its own overrides.
    #
    # `resolved` is stripped from the snapshot before it is stored: without
    # that, arm 1's view would contain a full copy of arm 0's view nested
    # under `arm`, which grows quadratically and makes `cfg.arms[1].resolved
    # .arm` mean the WRONG robot. Each arm's own profile is planted instead,
    # so `resolved.arm` always describes the arm that owns the view.
    for prof, ov in zip(arm_profiles, arm_overrides):
        base = copy.deepcopy(pre_override)
        if ov:
            _deep_merge(base, ov)
        base["arm"] = {k: v for k, v in prof.items() if k != "resolved"}
        prof["resolved"] = base
    main["arms"] = arm_profiles
    return Cfg(main)
