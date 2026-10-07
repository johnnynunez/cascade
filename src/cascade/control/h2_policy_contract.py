"""Contract of NVIDIA's Velocity-H2-History-v0 bundle for the Unitree H2 humanoid.

No simulator import. The vendored ``assets/h2/bundle/IO_descriptors.yaml`` and
``assets/h2/bundle/env.yaml`` (pinned by SHA-256 in ``configs/h2/bundle.json``) are
parsed into the exact joint order, observation layout, action decode, PD gains
and fall criteria an owner process must honour when it drives the H2 with this
policy. Everything that differs from the pins, or from the model's 255 -> 14
shape, is refused here instead of being discovered as a fall in physics.

The observation layout mirrors Isaac Sim's ``ObservationHistory``: terms in
descriptor order, each term's history frames from oldest to newest, the first
sample backfilling every frame. ``TermHistory`` reproduces that exactly so the
owner can be tested against it on the CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import yaml

from cascade.config import CONFIG_DIR

BUNDLE_MANIFEST = "h2/bundle.json"
# Descriptor term order and semantics the policy was exported with. Any other
# order, width or history is a different model and is refused.
EXPECTED_OBSERVATION_TERMS = (
    ("base_ang_vel", 3), ("projected_gravity", 3), ("generated_commands", 3),
    ("joint_pos_rel", 14), ("joint_vel_rel", 14), ("last_action", 14),
)
_SLICE_TAG = "tag:yaml.org,2002:python/object/apply:builtins.slice"
_TUPLE_TAG = "tag:yaml.org,2002:python/tuple"


class _BundleLoader(yaml.SafeLoader):
    """SafeLoader plus the two Isaac Lab export tags, recorded and never executed."""


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_slice(loader, node):
    return {"__slice__": tuple(loader.construct_sequence(node, deep=True))}


def _refuse_python_tag(loader, suffix, node):
    raise ValueError(f"refusing to construct python tag {suffix!r} from the bundle YAML")


_BundleLoader.add_constructor(_TUPLE_TAG, _construct_tuple)
_BundleLoader.add_constructor(_SLICE_TAG, _construct_slice)
_BundleLoader.add_multi_constructor("tag:yaml.org,2002:python/", _refuse_python_tag)


def load_bundle_yaml(path: Path):
    with Path(path).open("rb") as handle:
        return yaml.load(handle, Loader=_BundleLoader)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle_manifest(config_dir: Path | None = None) -> dict:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    manifest = json.loads((cdir / BUNDLE_MANIFEST).read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("bundle manifest has no files")
    for name, entry in files.items():
        digest = entry.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"bundle manifest pin for {name} must be a lowercase SHA-256")
    model = manifest.get("model") or {}
    if model.get("input_width") != 255 or model.get("output_width") != 14:
        raise ValueError("bundle manifest must pin the 255 -> 14 model shape")
    return manifest


def verify_pinned_file(path: Path, expected_sha256: str, name: str) -> str:
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ValueError(f"{name} does not match its pinned SHA-256 (got {actual[:12]}…, pinned {expected_sha256[:12]}…)")
    return actual


def vendored_path(manifest: dict, name: str, config_dir: Path | None = None) -> Path:
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    entry = manifest["files"][name]
    vendored = entry.get("vendored")
    if not isinstance(vendored, str):
        raise ValueError(f"{name} is not vendored; fetch it with scripts/h2_assets.py")
    path = cdir.parent / vendored  # repo-relative; the config dir is <repo>/configs
    verify_pinned_file(path, entry["sha256"], name)
    return path


@dataclass(frozen=True)
class ObservationTerm:
    name: str
    width: int
    history_length: int
    scale: float | None


@dataclass(frozen=True)
class JointGains:
    stiffness: float
    damping: float
    effort_limit: float
    velocity_limit: float
    armature: float
    group: str


@dataclass(frozen=True)
class H2PolicyContract:
    """Everything the owner needs, validated once, immutable afterwards."""

    model_sha256: str
    joint_names: tuple          # the 31 articulation joints, descriptor order
    policy_joint_names: tuple   # the 14 policy-controlled joints, action order
    held_joint_names: tuple     # the 17 joints the policy never commands (held at default)
    observation_terms: tuple
    observation_width: int
    action_scale: float
    action_offset: tuple
    action_clip: tuple
    default_joint_pos: tuple    # per articulation joint, descriptor order
    physics_dt: float
    control_dt: float
    decimation: int
    command_ranges: dict
    gains: dict                 # joint name -> JointGains
    init_root_pos: tuple
    init_root_rot_wxyz: tuple
    fall_tilt_rad: float        # torso tilt termination used in training
    fall_pelvis_height_m: float  # pelvis height termination used in training
    root_body: str              # body whose height/orientation the fall criteria read
    illegal_contact_bodies: tuple  # ground contact on these terminated a training episode
    foot_bodies: tuple          # the two links the training rewards treat as feet
    enabled_self_collisions: bool
    solver_iterations: tuple    # (position, velocity) counts the policy was trained with

    @classmethod
    def from_bundle(cls, config_dir: Path | None = None) -> "H2PolicyContract":
        manifest = load_bundle_manifest(config_dir)
        descriptor = load_bundle_yaml(vendored_path(manifest, "IO_descriptors.yaml", config_dir))
        env = load_bundle_yaml(vendored_path(manifest, "env.yaml", config_dir))
        return cls.from_documents(descriptor, env, manifest)

    @classmethod
    def from_documents(cls, descriptor: dict, env: dict, manifest: dict) -> "H2PolicyContract":
        robot = descriptor["articulations"]["robot"]
        joint_names = tuple(str(j) for j in robot["joint_names"])
        if len(joint_names) != len(set(joint_names)) or len(joint_names) != 31:
            raise ValueError("H2 descriptor must name 31 unique articulation joints")
        default_pos = tuple(float(x) for x in robot["default_joint_pos"])
        if len(default_pos) != 31:
            raise ValueError("default_joint_pos must cover the 31 joints")

        actions = [a for a in descriptor["actions"] if a.get("name") == "joint_position_action"]
        if len(actions) != 1:
            raise ValueError("exactly one joint_position_action term expected")
        action = actions[0]
        policy_joints = tuple(str(j) for j in action["joint_names"])
        if len(policy_joints) != 14 or set(policy_joints) - set(joint_names) or len(set(policy_joints)) != 14:
            raise ValueError("the policy must command 14 distinct articulation joints")
        if action.get("shape") != [14]:
            raise ValueError("joint_position_action must have shape [14]")
        scale = action["scale"]
        if isinstance(scale, (list, tuple)):
            if len(set(float(s) for s in scale)) != 1:
                raise ValueError("per-joint action scales are not supported by this contract")
            scale = scale[0]
        scale = float(scale)
        offset = tuple(float(x) for x in action["offset"])
        if len(offset) != 14 or not all(math.isfinite(x) for x in offset) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("action offset/scale must be 14 finite values and a positive scale")
        index = {name: i for i, name in enumerate(joint_names)}
        if any(not math.isclose(offset[k], default_pos[index[j]], abs_tol=1e-6) for k, j in enumerate(policy_joints)):
            raise ValueError("action offsets must equal the default positions of the commanded joints")
        clip = action.get("clip")
        if clip is None:
            clip = tuple((-math.inf, math.inf) for _ in policy_joints)
        else:
            clip = tuple((float(lo), float(hi)) for lo, hi in clip)
            if len(clip) != 14 or any(not lo < hi for lo, hi in clip):
                raise ValueError("action clip must give 14 ordered (low, high) pairs")
        for other in descriptor["actions"]:
            if other is not action and other.get("shape") not in ([0], [], None):
                raise ValueError(f"unexpected non-empty action term {other.get('name')!r}")

        terms = []
        for spec, expected in zip(descriptor["observations"]["policy"], EXPECTED_OBSERVATION_TERMS):
            overloads = spec.get("overloads") or {}
            width = int(np.prod(spec["shape"]))
            history = int(overloads.get("history_length") or 1)
            raw_scale = overloads.get("scale")
            if isinstance(raw_scale, (list, tuple)):
                if len(set(float(s) for s in raw_scale)) != 1:
                    raise ValueError(f"per-element observation scale on {spec['name']} is not supported")
                raw_scale = raw_scale[0]
            term = ObservationTerm(str(spec["name"]), width, history, None if raw_scale is None else float(raw_scale))
            if (term.name, term.width) != expected:
                raise ValueError(f"observation term {term.name!r}/{term.width} differs from the exported {expected}")
            if overloads.get("clip") is not None:
                raise ValueError(f"observation clipping on {term.name} is not part of this contract")
            if spec.get("joint_names") is not None and tuple(spec["joint_names"]) != policy_joints:
                raise ValueError(f"{term.name} joint order differs from the action joint order")
            terms.append(term)
        if len(terms) != len(EXPECTED_OBSERVATION_TERMS) or len(descriptor["observations"]["policy"]) != len(terms):
            raise ValueError("the descriptor must export exactly the six policy observation terms")
        histories = {t.history_length for t in terms}
        if len(histories) != 1:
            raise ValueError("all observation terms must share one history length")
        width = sum(t.width * t.history_length for t in terms)
        if width != manifest["model"]["input_width"]:
            raise ValueError(f"observation width {width} differs from the pinned model input {manifest['model']['input_width']}")
        if manifest["model"]["output_width"] != 14:
            raise ValueError("pinned model output must be 14")

        scene = descriptor["scene"]
        physics_dt, control_dt, decimation = float(scene["physics_dt"]), float(scene["dt"]), int(scene["decimation"])
        if not math.isclose(physics_dt * decimation, control_dt, rel_tol=1e-9):
            raise ValueError("control_dt must equal physics_dt * decimation")
        sim = env["sim"]
        if not math.isclose(float(sim["dt"]), physics_dt, rel_tol=1e-9) or int(env["decimation"]) != decimation:
            raise ValueError("env.yaml timing differs from the IO descriptor")

        ranges = env["commands"]["base_velocity"]["ranges"]
        command_ranges = {}
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            lo, hi = (float(v) for v in ranges[key])
            if not lo < hi:
                raise ValueError(f"command range {key} must be ordered")
            command_ranges[key] = (lo, hi)

        robot_cfg = env["scene"]["robot"]
        gains = _resolve_gains(robot_cfg["actuators"], joint_names)
        init = robot_cfg["init_state"]
        root_pos = tuple(float(x) for x in init["pos"])
        root_rot = tuple(float(x) for x in init["rot"])
        if len(root_rot) != 4 or not math.isclose(sum(x * x for x in root_rot), 1.0, abs_tol=1e-6):
            raise ValueError("init_state.rot must be a unit quaternion")
        terminations = env["terminations"]
        tilt = float(terminations["base_orientation"]["params"]["limit_angle"])
        height_term = terminations["illegal_base_height"]["params"]
        height = float(height_term["height_threshold"])
        root_body = str(height_term["asset_cfg"]["body_names"])
        if not (0 < tilt < math.pi / 2) or not (0 < height < root_pos[2]):
            raise ValueError("training fall criteria are implausible")
        illegal = terminations["illegal_contacts"]["params"]["sensor_cfg"]["body_names"]
        illegal = tuple(str(b) for b in ([illegal] if isinstance(illegal, str) else illegal))
        if not illegal or root_body not in illegal:
            raise ValueError("illegal ground contact must at least name the root body")
        feet = tuple(sorted({str(b) for term in env["rewards"].values()
                             for b in _body_names(term) if re.search(r"ankle_(pitch|roll)_link", b)
                             and not any(c in b for c in ".*[]()")}))
        if len(feet) != 2:
            raise ValueError("the training rewards must name exactly two foot links")
        props = robot_cfg["spawn"]["articulation_props"]
        self_collisions = props["enabled_self_collisions"]
        iterations = (int(props["solver_position_iteration_count"]), int(props["solver_velocity_iteration_count"]))
        if type(self_collisions) is not bool or min(iterations) <= 0:
            raise ValueError("articulation props must state self-collision and positive solver iterations")

        held = tuple(j for j in joint_names if j not in set(policy_joints))
        return cls(
            model_sha256=manifest["files"]["policy.pt"]["sha256"], joint_names=joint_names,
            policy_joint_names=policy_joints, held_joint_names=held, observation_terms=tuple(terms),
            observation_width=width, action_scale=scale, action_offset=offset, action_clip=clip,
            default_joint_pos=default_pos, physics_dt=physics_dt, control_dt=control_dt,
            decimation=decimation, command_ranges=command_ranges, gains=gains, init_root_pos=root_pos,
            init_root_rot_wxyz=root_rot, fall_tilt_rad=tilt, fall_pelvis_height_m=height, root_body=root_body,
            illegal_contact_bodies=illegal, foot_bodies=feet, enabled_self_collisions=self_collisions,
            solver_iterations=iterations,
        )

    # ---- what the owner uses every control step -------------------------------------------

    def observation_layout(self) -> tuple:
        """``(term, frame, start, stop)`` for every slot of the flat observation, oldest frame first."""
        layout, start = [], 0
        for term in self.observation_terms:
            for frame in range(term.history_length):
                layout.append((term.name, frame, start, start + term.width))
                start += term.width
        assert start == self.observation_width
        return tuple(layout)

    def scaled_sample(self, *, base_ang_vel, projected_gravity, command, joint_pos, joint_vel, last_action) -> np.ndarray:
        """One current-frame sample in term order, with the descriptor scales applied.

        ``joint_pos``/``joint_vel`` are the 14 policy joints in action order; the
        relative joint position is taken against the action offsets (the default
        positions of those joints), exactly like Isaac Lab's ``joint_pos_rel``.
        """
        parts = {
            "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float64).reshape(3),
            "projected_gravity": np.asarray(projected_gravity, dtype=np.float64).reshape(3),
            "generated_commands": np.asarray(command, dtype=np.float64).reshape(3),
            "joint_pos_rel": np.asarray(joint_pos, dtype=np.float64).reshape(14) - np.asarray(self.action_offset),
            "joint_vel_rel": np.asarray(joint_vel, dtype=np.float64).reshape(14),
            "last_action": np.asarray(last_action, dtype=np.float64).reshape(14),
        }
        sample = []
        for term in self.observation_terms:
            value = parts[term.name]
            if not np.all(np.isfinite(value)):
                raise ValueError(f"nonfinite {term.name} observation; no policy inference")
            sample.append(value if term.scale is None else value * term.scale)
        return np.concatenate(sample).astype(np.float32)

    def command_within_training_ranges(self, vx: float, vy: float, wz: float) -> bool:
        (x0, x1), (y0, y1), (z0, z1) = (self.command_ranges[k] for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
        return x0 <= vx <= x1 and y0 <= vy <= y1 and z0 <= wz <= z1

    def decode_action(self, raw) -> dict:
        """Raw policy output -> joint position targets (rad) for the 14 policy joints.

        Isaac Lab's JointPositionAction: clip the raw output, then
        ``offset + scale * raw``. Non-finite or mis-sized outputs are refused.
        """
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        if raw.shape != (14,) or not np.all(np.isfinite(raw)):
            raise ValueError("policy output must be 14 finite values")
        clipped = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(raw, self.action_clip)])
        targets = np.asarray(self.action_offset) + self.action_scale * clipped
        return {name: float(t) for name, t in zip(self.policy_joint_names, targets)}

    def held_targets(self) -> dict:
        """Default positions for the joints the policy never commands."""
        index = {name: i for i, name in enumerate(self.joint_names)}
        return {name: self.default_joint_pos[index[name]] for name in self.held_joint_names}


def _body_names(term) -> list:
    """Body names referenced by one exported manager term (reward/termination), flattened."""
    names = []
    params = term.get("params") if isinstance(term, dict) else None
    for cfg in (params or {}).values():
        if isinstance(cfg, dict) and cfg.get("body_names") is not None:
            value = cfg["body_names"]
            names.extend([value] if isinstance(value, str) else list(value))
    return names


def _resolve_gains(actuators: dict, joint_names: tuple) -> dict:
    """First actuator group whose ``joint_names_expr`` matches a joint owns its gains."""
    def value_for(group: dict, key: str, joint: str) -> float:
        value = group.get(key)
        if isinstance(value, dict):
            for pattern, v in value.items():
                if re.fullmatch(pattern, joint):
                    return float(v)
            raise ValueError(f"{key} of actuator group has no entry for {joint}")
        if value is None:
            raise ValueError(f"{key} is undefined for {joint}")
        return float(value)

    gains = {}
    for joint in joint_names:
        owner = None
        for name, group in actuators.items():
            if any(re.fullmatch(p, joint) for p in group["joint_names_expr"]):
                owner = (name, group)
                break
        if owner is None:
            raise ValueError(f"no actuator group drives {joint}")
        name, group = owner
        gains[joint] = JointGains(
            stiffness=value_for(group, "stiffness", joint), damping=value_for(group, "damping", joint),
            effort_limit=value_for(group, "effort_limit_sim", joint),
            velocity_limit=value_for(group, "velocity_limit_sim", joint),
            armature=value_for(group, "armature", joint), group=name,
        )
        g = gains[joint]
        if not (g.stiffness > 0 and g.damping >= 0 and g.effort_limit > 0 and g.velocity_limit > 0 and g.armature >= 0):
            raise ValueError(f"implausible gains for {joint}: {g}")
    return gains


class TermHistory:
    """Isaac Sim's ``ObservationHistory`` semantics, reproduced for CPU tests and the owner.

    Independent circular buffer per term; the first sample backfills every
    frame; ``append`` returns all terms flattened, each from oldest to newest.
    """

    def __init__(self, contract: H2PolicyContract) -> None:
        self._terms = contract.observation_terms
        self._width = sum(t.width for t in self._terms)
        self._buffers = [np.zeros((t.history_length, t.width), np.float32) for t in self._terms]
        self._next = [0] * len(self._terms)
        self._primed = [False] * len(self._terms)

    def reset(self) -> None:
        self._next = [0] * len(self._terms)
        self._primed = [False] * len(self._terms)

    def append(self, sample) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32).reshape(-1)
        if sample.shape != (self._width,):
            raise ValueError(f"observation sample must have {self._width} values")
        out, start = [], 0
        for i, (term, buffer) in enumerate(zip(self._terms, self._buffers)):
            current = sample[start:start + term.width]
            start += term.width
            if not self._primed[i]:
                buffer[:] = current
                self._primed[i] = True
            else:
                buffer[self._next[i]] = current
                self._next[i] = (self._next[i] + 1) % len(buffer)
            n = self._next[i]
            out.append(np.concatenate((buffer[n:], buffer[:n])).reshape(-1))
        return np.concatenate(out)
