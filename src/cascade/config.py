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
import os
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
    data = _resolve_paths(_load_profile_raw(kind, name, cdir), cdir)
    if kind == "bases" and data.get("type") == "isaac":
        _mobile_environment(data)
    return Cfg(data)


def _mobile_environment(profile):
    """Only explicit mobile overrides; no arm sidecar ports or hash wildcards."""
    for key in ("asset_sha256", "policy_sha256", "model_identity_sha256", "bridge_port", "engine", "device"):
        env = "CASCADE_MICRODUCK_" + key.upper()
        if env in os.environ:
            value = os.environ[env]
            if key == "bridge_port":
                if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
                    raise ValueError(f"{env} must be an explicit port in 1..65535")
                value = int(value)
            profile[key] = value
    for key in ("asset_sha256", "policy_sha256", "model_identity_sha256"):
        value = profile.get(key)
        if value is not None and (not isinstance(value, str) or len(value) != 64
                                  or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(f"{key} must be an exact lowercase SHA256 digest")
    if profile.get("engine") not in {"physx", "newton"}:
        raise ValueError("mobile engine must be physx or newton")


def _deep_merge(base: dict, overlay: dict) -> None:
    """Merge overlay into base in place: dicts recurse, scalars replace."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _mobile_config(cdir, main, base, bases, llm) -> Cfg:
    """Opt-in morphology: only shared memory/brain settings cross this boundary."""
    names = bases if bases is not None else [base]
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n or n.strip() != n
                   or Path(n).name != n for n in names)):
        raise ValueError("nonempty exact base profile names required")
    if len(set(names)) != len(names):
        raise ValueError("duplicate base profiles; use separately named profiles")
    profiles = []
    for name in names:
        prof = load_profile("bases", name, cdir).as_dict()
        prof.setdefault("name", name)
        # Never merge arm/table limits into a base. Each profile owns its view.
        from .safety.base_harness import BaseSafetyHarness

        BaseSafetyHarness(prof.get("safety", {}))
        prof["resolved"] = {"safety": copy.deepcopy(prof["safety"])}
        profiles.append(prof)
    identifiers = [p["name"] for p in profiles]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate base names")
    return Cfg({"robot_mode": "mobile", "base": profiles[0], "bases": profiles,
                "memory": copy.deepcopy(main.get("memory", {})),
                "llm": load_profile("llm", llm, cdir).as_dict()})


def load_robot_config(robot: str, *, llm: str = "mock", config_dir: Path | None = None) -> Cfg:
    """Resolve an explicit composition without constructing devices or opening IO."""
    import re

    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    if not isinstance(robot, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", robot):
        raise ValueError("robot must be an exact profile slug")
    data = load_profile("robots", robot, cdir).as_dict()
    if (set(data) - {"version", "robot_id", "domains", "embodiment", "whole_body"}
            or type(data.get("version")) is not int or data.get("version") != 1):
        raise ValueError("robot profile requires version 1, robot_id, domains and optional embodiment/whole_body")
    # A robot identity is not a profile filename. Preserve the exact identifier
    # used by its backend (for example microduck-mock) without relabeling it.
    from .robotics.contracts import identifier
    identifier(data.get("robot_id"), "robot_id")
    domains = data.get("domains")
    if not isinstance(domains, dict) or not domains or len(domains) > 16:
        raise ValueError("robot domains must be a nonempty mapping of at most 16 entries")
    resolved = {}
    for name, profile in domains.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name):
            raise ValueError("domain id must be an ASCII slug of at most 24 characters")
        if not isinstance(profile, dict):
            raise ValueError("domain profile must be an object")
        profile = copy.deepcopy(profile)
        kind = profile.get("kind")
        if "mounted_on" in profile and "whole_body" not in data:
            # A mount is only meaningful inside the explicit whole-body contract
            # (disjoint endpoints, dynamic frame chain, coordination policy).
            raise ValueError("mounted_on requires an explicit whole_body contract; mobile-mounted domains are "
                             "otherwise unsupported")
        if kind == "manipulation":
            if set(profile) - {"kind", "arms", "cameras", "robot_id", "offline", "mounted_on"}:
                raise ValueError("unknown manipulation domain fields")
            if type(profile.get("offline", False)) is not bool:
                raise ValueError("offline must be boolean")
            for key in ("arms", "cameras"):
                if not isinstance(profile.get(key), list) or not profile[key] or any(
                        not isinstance(v, str) or not v or Path(v).name != v for v in profile[key]):
                    raise ValueError(f"manipulation domain requires explicit {key} profile names")
            cfg = load_demo_config(arms=profile["arms"], cameras=profile["cameras"], llm=llm,
                                   config_dir=cdir, _ignore_robot_environment=True)
            if profile.get("offline", False):
                if any(p["type"] != "mock" for p in (*cfg.arms, *cfg.cameras)):
                    raise ValueError("offline manipulation requires only explicit mock arms and cameras")
                for view in (cfg._data, *(p["resolved"] for p in cfg.arms)):
                    view.setdefault("grasp", {})["backend"] = "obb"
                    view.setdefault("occupancy", {})["enabled"] = False
            profile["resolved"] = cfg.as_dict()
        elif kind == "locomotion":
            if set(profile) - {"kind", "bases", "robot_id"} or "bases" not in profile:
                raise ValueError("locomotion domain requires kind and bases only")
            cfg = load_demo_config(bases=profile["bases"], llm=llm, config_dir=cdir,
                                   _ignore_robot_environment=True)
            profile["resolved"] = cfg.as_dict()
        elif kind == "fastening":
            from .apps.factory_runtime import validate_factory_profile
            validate_factory_profile(profile)
        elif kind == "hand":
            from .apps.hand_runtime import validate_hand_profile
            validate_hand_profile(profile)
        elif kind not in {"sensors", "spatial"}:
            raise ValueError(f"unknown robot domain kind: {kind!r}")
        profile.setdefault("robot_id", data["robot_id"])
        resolved[name] = profile
    result = {"robot_mode": "composed", "robot_id": data["robot_id"], "domains": resolved,
              "llm": load_profile("llm", llm, cdir).as_dict(), "memory": {}}
    if "embodiment" in data:
        from .robotics.embodiment import EmbodimentDescriptor
        body = EmbodimentDescriptor.from_dict(data["embodiment"])
        if body.robot_id != data["robot_id"]:
            raise ValueError("embodiment robot_id must match robot profile")
        result["embodiment"] = body.as_dict()
    if "whole_body" in data:
        # Opt-in only: profiles without the key keep their exact surface.
        from .robotics.whole_body import validate_contract
        result["whole_body"] = validate_contract(data["whole_body"], resolved)
    return Cfg(result)


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
    base: str | None = None,
    bases: list[str] | None = None,
    robot: str | None = None,
    _ignore_robot_environment: bool = False,
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
    selected_robot = robot if robot is not None else (
        None if _ignore_robot_environment else os.environ.get("CASCADE_ROBOT"))
    if selected_robot is not None:
        if base is not None or bases is not None or arms is not None or arm != "mock" or cameras is not None or camera != "mock":
            raise ValueError("robot composition cannot also select legacy arms, bases or cameras")
        return load_robot_config(selected_robot, llm=llm, config_dir=cdir)
    main = _resolve_paths(_load_yaml(cdir / "demo.yaml"), cdir)
    if base is None and bases is None and not _ignore_robot_environment:
        base = os.environ.get("CASCADE_BASE")
    if base is not None or bases is not None:
        if arms is not None or arm != "mock":
            raise ValueError("base-only configuration cannot also select arms")
        return _mobile_config(cdir, main, base, bases, llm)
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
        # A sim arm that asks for a camera-matched prop needs the camera
        # profile at construction time, but the arm is built before the
        # camera. Plant THIS arm's view of the camera on the profile so the
        # scene generator cannot read a different camera than the one the
        # demo will actually open. Opt-in: absent unless the profile sets
        # `mj_prop_from_camera: true`. See sim/demo_scene.py.
        if prof.get("mj_prop_from_camera") is True:
            view_cams = base.get("cameras")
            prof["mj_prop_from_camera"] = (view_cams[0] if view_cams else base.get("camera"))
            # ...and the FULL camera list, so every rendered (`type: mujoco`)
            # profile gets a <camera> declared in the generated scene, not
            # just the manipulation camera.
            prof["mj_cameras"] = list(view_cams) if view_cams else [base.get("camera")]
    main["arms"] = arm_profiles

    # Several `type: mujoco` arms are a RIG: one generated MJCF containing
    # every arm (sim/demo_scene.write_rig_robot), one shared world. Each arm
    # must be placeable in it -- a `base_pose` (where it is bolted, the same
    # number the inter-arm gate uses), a unique non-empty `mj_prefix` (its
    # joints become `<prefix>shoulder_pan`) and the same robot file as the
    # others -- or the loader refuses, naming the arm. The alternative is two
    # private worlds that look like one rig, with each arm frozen in the
    # other's physics. Every rig arm receives the SAME `mj_rig` list so each
    # derives the same file.
    mujoco_rig = _mujoco_rig(arm_names, arm_profiles, cams)
    for prof in arm_profiles:
        if mujoco_rig is not None and str(prof.get("type", "")) == "mujoco":
            prof["mj_rig"] = copy.deepcopy(mujoco_rig)

    # Explicit launcher selection reaches every arm, including rig runtimes.
    # "--graspgenx none" is an intentional analytic mode, never an implicit
    # downgrade after a failed learned-model request.
    backend = os.environ.get("CASCADE_GRASP_BACKEND")
    if backend:
        if backend not in {"obb", "graspgenx", "hug"}:
            raise ValueError(f"invalid CASCADE_GRASP_BACKEND: {backend}")
        for view in [main, *(prof["resolved"] for prof in arm_profiles)]:
            view.setdefault("grasp", {})["backend"] = backend

    # Rendered sim cameras look INTO a MuJoCo arm's world; tell each which.
    # The primary arm owns the scene (in a rig, the generated file holds every
    # arm and the primary's profile decides the props), so a rendered camera
    # without a MuJoCo primary is a config error worth naming: it would
    # otherwise open, fail to find a world and stream nothing, and the demo
    # would report "no frames" far from the cause.
    # `cams` here is the LIVE list under main["cameras"] (main["camera"] is
    # its first element by reference), so planting on it reaches the config.
    primary = arm_profiles[0]
    for c in cams:
        if str(c.get("type", "")) != "mujoco":
            continue
        if str(primary.get("type", "")) != "mujoco":
            raise ValueError(
                f"camera profile {c.get('name')!r} is type: mujoco (renders a "
                f"MuJoCo world) but the primary arm {arm_names[0]!r} is "
                f"type: {primary.get('type')!r} -- pair it with a mujoco arm "
                "(so101_mujoco, piper_mujoco) or use the mock camera"
            )
        c.setdefault("mj_arm", arm_names[0])
        # The camera attaches to the arm's world BY PATH (sim/mujoco_world.py
        # registry). The path is decided by the arm's profile (its `mjcf`,
        # whether it is part of a rig, and whether scene generation rewrites
        # it), so resolve it here with the SAME functions the arm uses -- two
        # derivations would drift.
        from .sim.demo_scene import resolved_scene_path, rig_robot_path

        prop_cam = primary.get("mj_prop_from_camera")
        robot_file = primary.get("mjcf")
        if mujoco_rig is not None:
            robot_file = str(rig_robot_path(robot_file, mujoco_rig))
        c.setdefault("mj_scene", str(resolved_scene_path(robot_file, prop_cam)))
        # Under the MCP server the arm is a LazyArm built on the FIRST MOTION,
        # so the camera (opened at prewarm) is the first to need the generated
        # scene -- which only the arm's constructor used to write. On a fresh
        # clone that is "file not found"; after a profile edit it is a STALE
        # scene. Hand the camera the same generation inputs so it (re)writes
        # the scene itself; the content is deterministic, so the arm's later
        # write of the same file is a no-op.
        if hasattr(prop_cam, "get"):
            c.setdefault("mj_scene_source", {
                "mjcf": primary.get("mjcf"),
                "prop_cam": prop_cam,
                "cameras": primary.get("mj_cameras"),
                "rig": copy.deepcopy(mujoco_rig),
            })
    return Cfg(main)


def _mujoco_rig(arm_names, arm_profiles, cams):
    """The `mj_rig` list for a run with two or more MuJoCo arms, or None.

    Fails closed (ValueError naming the arm) when a MuJoCo arm lacks a
    `base_pose` or a unique `mj_prefix`, when the arms name different robot
    files (one `<attach>`ed model per rig for now), or when a rendered
    camera asks for a wrist attachment (`mj_attach` reopens a body of the
    robot's include chain; an attached model has no include chain).
    """
    mujoco_arms = [(n, p) for n, p in zip(arm_names, arm_profiles) if str(p.get("type", "")) == "mujoco"]
    if len(mujoco_arms) < 2:
        return None
    rig, seen = [], set()
    first_robot = str(mujoco_arms[0][1].get("mjcf"))
    for name, prof in mujoco_arms:
        pose = prof.get("base_pose")
        if pose is None or len(list(pose)) != 6:
            raise ValueError(
                f"arm {name!r} is one of {len(mujoco_arms)} MuJoCo arms but declares no "
                "`base_pose: [x, y, z, roll, pitch, yaw]` -- a rig cannot place it in the shared world"
            )
        prefix = str(prof.get("mj_prefix") or "")
        if not prefix or prefix in seen:
            raise ValueError(
                f"arm {name!r} needs a unique non-empty `mj_prefix` to be part of a MuJoCo rig "
                f"(got {prefix!r}); its joints are addressed as `<prefix><joint>` in the shared world"
            )
        seen.add(prefix)
        if str(prof.get("mjcf")) != first_robot:
            raise ValueError(
                f"arm {name!r} names a different robot file ({prof.get('mjcf')}) than "
                f"{mujoco_arms[0][0]!r} ({first_robot}); a MuJoCo rig attaches copies of ONE robot file"
            )
        rig.append({"prefix": prefix, "base_pose": [float(v) for v in pose]})
    for c in cams:
        if str(c.get("type", "")) == "mujoco" and c.get("mj_attach") is not None:
            raise ValueError(
                f"camera profile {c.get('name')!r} attaches to a robot body (`mj_attach`), which a "
                "MuJoCo rig does not support yet; use a fixed rendered camera (mujoco_scene)"
            )
    return rig
