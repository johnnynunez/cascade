"""VLA executor: `grasp_object` served by a language-conditioned policy.

ROADMAP mid term "VLA policy backend" (B49). OPT-IN: `grasp.executor: vla`
(or CASCADE_GRASP_EXECUTOR=vla); the default `analytic` is the deterministic
pipeline in `skills/runtime.py`, untouched. The agent layer is unchanged:
the tool is still `grasp_object(label)`, and every composite that grasps by
label (pick_and_place, sort_by_color, handover, ...) reaches the policy
through it. Pixel-addressed grasps (`grasp_at_pixel`) stay analytic -- a
language-conditioned policy needs a name to be conditioned on.

The policy speaks the openpi / LingBot-VLA-v2 websocket protocol
(`grasping/vla_client.py`). One EPISODE is:

  1. route preconditions: rigs whose gates the route does not implement
     (observed-finger gate, native motion planner, payload-tracking map) are
     refused before anything moves -- never bypassed;
  2. connect (a server that does not answer refuses the grasp before any
     motion; there is no analytic fallback: the operator chose the policy);
  3. the runtime opens the jaws and re-homes over its vetted route, exactly
     like the analytic pipeline;
  4. per chunk: the stop latch and the halt generation are checked BETWEEN
     chunks, when the reply arrives -- a stop or halt that landed while the
     policy was computing never reaches the arm (within a chunk SafeArm
     checks both before every waypoint and gripper command, with the halt
     generation the skill started under); a fresh camera frame plus the
     MEASURED joints/jaw form the observation; the reply must arrive inside
     the chunk deadline and the whole episode inside its own deadline (and
     the task budget); the chunk is converted to joint targets (`joint`:
     absolute local joint angles; `tcp`: base-frame x y z roll pitch yaw
     through IK) and malformed chunks are refused;
  5. the SafetyHarness admits the WHOLE chunk (`vet_pose` on every target,
     no grasp exemption) before any of it moves, then every waypoint streams
     through `SafeArm.move_joints`, which calls `harness.approve()` on every
     sample (velocity cap, joint limits, workspace, table, keep-outs,
     neighbours, occupancy). The executor itself never refuses motion on
     geometric grounds: the harness is the sole authority;
  6. the episode ends `chunks_after_close` chunks after the first close
     command (the lift), or after `max_chunks`.

Only the configured action key is read from a reply. Whatever else the
server sends -- `success`, `done`, `is_success`, a reward -- is never
evidence and never ends the episode early: the runtime's jaw-travel check
and the unchanged three-state postcondition verifier (`agent/effects.py`,
kind `holding`) judge the grasp, exactly as for the analytic pipeline.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from ..types import SafetyViolation, SkillError, pose_to_transform

EXECUTORS = ("analytic", "vla")
ACTION_SPACES = ("joint", "tcp")
GRIPPER_CONVENTIONS = ("open_frac", "close_frac")
#: a gripper value up to this far outside [0, 1] is numerical overshoot
#: (flow/diffusion heads produce 1.02) and is clipped; beyond it the chunk
#: does not follow the declared convention and is refused
GRIP_TOLERANCE = 0.05
#: a commanded opening at or below this fraction is a CLOSE command
CLOSE_BELOW = 0.5
#: a gripper command closer than this to the previous one is not re-sent
GRIP_DEADBAND = 0.02


def executor_name(gcfg) -> str:
    """`grasp.executor` validated: `analytic` (default) or `vla`."""
    value = gcfg.get("executor", "analytic") if gcfg is not None else "analytic"
    name = "analytic" if value is None else str(value)
    if name not in EXECUTORS:
        raise ValueError(f"grasp.executor must be one of {list(EXECUTORS)}, got {value!r}")
    return name


def _positive(name: str, value, *, integer: bool = False, minimum: float = 0.0, strict: bool = True):
    try:
        number = int(value) if integer else float(value)
        if integer and float(value) != number:
            raise ValueError
    except (TypeError, ValueError):
        kind = "an integer" if integer else "a number"
        raise ValueError(f"grasp.vla.{name} must be {kind}, got {value!r}") from None
    if not math.isfinite(number) or (number <= minimum if strict else number < minimum):
        bound = f"> {minimum:g}" if strict else f">= {minimum:g}"
        raise ValueError(f"grasp.vla.{name} must be finite and {bound}, got {value!r}")
    return number


@dataclass(frozen=True)
class VLAConfig:
    """`grasp.vla` (configs/demo.yaml documents every key)."""

    host: str = "127.0.0.1"
    port: int = 8000
    connect_timeout_s: float = 1.0
    chunk_timeout_s: float = 2.0
    episode_timeout_s: float = 60.0
    max_chunks: int = 20
    chunks_after_close: int = 1
    max_chunk_len: int = 64
    action_key: str = "actions"
    action_space: str = "joint"
    gripper: str = "open_frac"
    gripper_column: int | None = None
    waypoint_duration_s: float = 0.2
    prompt: str = "pick up the {label}"
    image_key: str = "observation/image"
    state_key: str = "observation/state"
    prompt_key: str = "prompt"
    image_size: int | None = 224
    reset: dict | None = None
    extra_obs: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, section) -> "VLAConfig":
        raw = section.as_dict() if hasattr(section, "as_dict") else dict(section or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"grasp.vla has unknown keys {unknown}")
        merged = {**{f.name: f.default if f.default is not dataclasses.MISSING else f.default_factory()
                     for f in dataclasses.fields(cls)}, **raw}
        port = _positive("port", merged["port"], integer=True)
        if port > 65535:
            raise ValueError(f"grasp.vla.port must be a TCP port in 1..65535, got {merged['port']!r}")
        action_space = str(merged["action_space"])
        if action_space not in ACTION_SPACES:
            raise ValueError(f"grasp.vla.action_space must be one of {list(ACTION_SPACES)}, "
                             f"got {action_space!r}")
        gripper = str(merged["gripper"])
        if gripper not in GRIPPER_CONVENTIONS:
            raise ValueError(f"grasp.vla.gripper must be one of {list(GRIPPER_CONVENTIONS)}, "
                             f"got {gripper!r}")
        column = merged["gripper_column"]
        if column is not None:
            column = _positive("gripper_column", column, integer=True, strict=False)
        size = merged["image_size"]
        if size is not None:
            size = _positive("image_size", size, integer=True, minimum=8, strict=False)
        prompt = str(merged["prompt"])
        try:
            prompt.format(label="x")
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"grasp.vla.prompt may only use the {{label}} field: {exc}") from None
        for key in ("reset", "extra_obs"):
            value = merged[key]
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"grasp.vla.{key} must be a mapping, got {type(value).__name__}")
        for key in ("action_key", "image_key", "state_key", "prompt_key", "host"):
            if not str(merged[key]).strip():
                raise ValueError(f"grasp.vla.{key} must be a non-empty string")
        return cls(
            host=str(merged["host"]), port=port,
            connect_timeout_s=_positive("connect_timeout_s", merged["connect_timeout_s"]),
            chunk_timeout_s=_positive("chunk_timeout_s", merged["chunk_timeout_s"]),
            episode_timeout_s=_positive("episode_timeout_s", merged["episode_timeout_s"]),
            max_chunks=_positive("max_chunks", merged["max_chunks"], integer=True),
            chunks_after_close=_positive("chunks_after_close", merged["chunks_after_close"],
                                         integer=True, strict=False),
            max_chunk_len=_positive("max_chunk_len", merged["max_chunk_len"], integer=True),
            action_key=str(merged["action_key"]), action_space=action_space, gripper=gripper,
            gripper_column=column,
            waypoint_duration_s=_positive("waypoint_duration_s", merged["waypoint_duration_s"]),
            prompt=prompt, image_key=str(merged["image_key"]), state_key=str(merged["state_key"]),
            prompt_key=str(merged["prompt_key"]), image_size=size,
            reset=None if merged["reset"] is None else dict(merged["reset"]),
            extra_obs=dict(merged["extra_obs"] or {}),
        )

    def replace(self, **changes) -> "VLAConfig":
        """A validated copy with `changes` applied."""
        return VLAConfig.from_cfg({**dataclasses.asdict(self), **changes})


def chunk_targets(reply, cfg: VLAConfig, *, n_joints: int, kin, seed_q) -> list[tuple[np.ndarray, float]]:
    """One policy reply -> [(joint target, jaw OPEN fraction 0..1)].

    Refuses (SkillError) a reply that does not follow the configured action
    contract: missing key, not a finite 2-D array, too few columns, too many
    or no rows, a gripper value outside [0, 1] beyond numerical overshoot, a
    TCP row without an IK solution or whose solution leaves the previous
    target's IK branch. Nothing here judges SAFETY: the harness does that.
    """
    key = cfg.action_key
    if not isinstance(reply, dict) or reply.get(key) is None:
        keys = sorted(map(str, reply)) if isinstance(reply, dict) else type(reply).__name__
        raise SkillError(f"policy reply carries no {key!r} actions (got {keys})")
    try:
        chunk = np.asarray(reply[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise SkillError(f"policy {key!r} actions are not numeric: {exc}") from None
    if chunk.ndim != 2:
        raise SkillError(f"action chunk must be 2-D (rows x columns), got shape {chunk.shape}")
    rows, cols = chunk.shape
    if not 1 <= rows <= cfg.max_chunk_len:
        raise SkillError(f"action chunk has {rows} rows; 1..{cfg.max_chunk_len} accepted")
    arm_cols = int(n_joints) if cfg.action_space == "joint" else 6
    grip_col = arm_cols if cfg.gripper_column is None else int(cfg.gripper_column)
    need = max(arm_cols, grip_col + 1)
    if cols < need:
        raise SkillError(f"action chunk has {cols} columns; {need} needed "
                         f"({cfg.action_space} arm columns + gripper column {grip_col})")
    if not np.isfinite(chunk).all():
        raise SkillError("action chunk is not finite (NaN or inf)")
    grip = chunk[:, grip_col].copy()
    if cfg.gripper == "close_frac":
        grip = 1.0 - grip
    if np.any(grip < -GRIP_TOLERANCE) or np.any(grip > 1.0 + GRIP_TOLERANCE):
        raise SkillError(f"gripper column {grip_col} outside [0, 1] ({cfg.gripper}): "
                         f"{np.round(chunk[:, grip_col], 3).tolist()}")
    grip = np.clip(grip, 0.0, 1.0)
    seed = np.asarray(seed_q, dtype=float).reshape(-1)[:n_joints]
    targets = []
    for i, row in enumerate(chunk):
        if cfg.action_space == "joint":
            q = row[:n_joints].copy()
        else:
            if kin is None:
                raise SkillError("tcp actions need the arm's kinematics")
            ik = kin.ik(pose_to_transform(row[:6]), seed)
            if not ik.success:
                raise SkillError(f"IK failed for TCP action row {i + 1}: "
                                 f"{np.round(row[:6], 3).tolist()}")
            q = np.asarray(ik.q, dtype=float).reshape(-1)[:n_joints]
            if np.max(np.abs(q - seed)) > np.pi:
                raise SkillError(f"IK solution for TCP action row {i + 1} leaves the previous "
                                 "target's branch (> pi on a joint)")
        seed = q
        targets.append((q, float(grip[i])))
    return targets


def route_refusal(runtime) -> str | None:
    """Why this rig cannot run the VLA route, or None. These gates own
    geometry contracts the waypoint stream does not implement; the route is
    refused rather than run around them."""
    if os.environ.get("CASCADE_OBSERVED_FINGER_GATE") == "1":
        return ("the observed-finger gate (CASCADE_OBSERVED_FINGER_GATE=1) vets analytic candidates "
                "against observed finger geometry; the VLA route has no such contract")
    if getattr(runtime.arm, "motion_planner", None) is not None:
        return ("this arm is bound to a native motion planner whose curves and evidence a "
                "VLA waypoint stream would bypass")
    if getattr(getattr(runtime.arm.harness, "occupancy", None), "tracks_payload", False):
        return "payload-tracking occupancy requires the analytic contact episode"
    return None


class VLAExecutor:
    """Holds the validated config and the last server status; runs episodes."""

    def __init__(self, config: VLAConfig, *, client_factory=None, clock=time.monotonic):
        self.config = config
        self._client_factory = client_factory
        self._clock = clock
        self.status: dict = {"answered": None, "metadata": None,
                             "detail": f"{config.host}:{config.port} (not probed yet)"}

    @classmethod
    def from_cfg(cls, gcfg) -> "VLAExecutor":
        return cls(VLAConfig.from_cfg(gcfg.get("vla") if gcfg is not None else None))

    def describe(self) -> str:
        return f"vla ({self.status.get('detail')})"

    def probe(self) -> dict:
        """Connect + metadata frame now; never raises. Sets `status`."""
        from .vla_client import probe

        self.status = probe(self.config.host, self.config.port, timeout_s=self.config.connect_timeout_s)
        return self.status

    def connect(self):
        """An open PolicyClient, or VLAUnavailable before any motion."""
        from .vla_client import PolicyClient, VLAUnavailable, describe_metadata

        cfg = self.config
        client = (self._client_factory or PolicyClient)(cfg.host, cfg.port,
                                                        connect_timeout_s=cfg.connect_timeout_s)
        try:
            meta = client.connect()
        except VLAUnavailable as exc:
            self.status = {"answered": False, "detail": str(exc), "metadata": None}
            raise
        self.status = {"answered": True, "detail": describe_metadata(cfg.host, cfg.port, meta),
                       "metadata": meta}
        return client

    # ── one episode ─────────────────────────────────────────────────────

    @staticmethod
    def _live(harness, halt_generation) -> None:
        """The stop latch and halts, honoured between chunks."""
        if harness.estopped:
            raise SafetyViolation("e-stop latched: the VLA episode stopped between chunks")
        harness._check_halt_generation(halt_generation)

    def _remaining(self, deadline: float, runtime) -> float:
        cfg = self.config
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise SkillError(f"VLA episode deadline ({cfg.episode_timeout_s:g}s) passed; "
                             "episode ended, no further motion")
        task = getattr(runtime, "_task_deadline", None)
        if task is not None:
            left = float(task) - time.monotonic()
            if left <= 0:
                raise SkillError("task budget exhausted during the VLA episode; no further motion")
            remaining = min(remaining, left)
        return remaining

    def _observation(self, runtime, prompt: str) -> dict:
        import cv2

        cfg = self.config
        frame = runtime.observe_fresh()
        rgb = np.ascontiguousarray(np.asarray(frame.rgb)[:, :, ::-1])  # Frame.rgb is BGR
        if cfg.image_size is not None:
            rgb = cv2.resize(rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
        state = runtime.arm.get_state()
        jaw = runtime._gripper_width_frac()
        if jaw is None:
            raise SkillError("gripper feedback unavailable: the VLA observation would invent the "
                             "jaw state; episode ended, no further motion")
        return {**cfg.extra_obs,
                cfg.image_key: np.ascontiguousarray(rgb, dtype=np.uint8),
                cfg.state_key: np.r_[np.asarray(state.q, float), float(jaw)].astype(np.float32),
                cfg.prompt_key: prompt}

    @staticmethod
    def _admit(harness, targets, chunk_no: int) -> None:
        """The harness vets every target of the chunk before any of it moves."""
        for i, (q, _) in enumerate(targets):
            reason = harness.vet_pose(q)
            if reason:
                raise SafetyViolation(f"harness refused VLA chunk {chunk_no} at waypoint {i + 1} "
                                      f"before any of it moved: {reason}")

    def run(self, runtime, client, *, label: str, effort: float, halt_generation,
            provisional) -> dict:
        """Run one episode on an open `client`. Returns the report; raises
        SkillError/SafetyViolation when the episode is cut short (no further
        motion after the raise). Sets `runtime._held_provisional` right
        before the first close command, like the analytic close."""
        from .vla_client import VLATimeout, describe_metadata

        cfg = self.config
        arm, harness = runtime.arm, runtime.arm.harness
        n = int(arm.n_joints)
        deadline = self._clock() + cfg.episode_timeout_s
        # last_open_commanded starts at 1.0: the route opened the jaws before the episode
        report = {"chunks": 0, "waypoints": 0, "closed": False, "last_open_commanded": 1.0,
                  "q_at_close": None, "infer_ms": [], "stop": "max_chunks",
                  "server": describe_metadata(cfg.host, cfg.port, client.metadata)}
        prompt = cfg.prompt.format(label=label)
        close_chunk = None
        if cfg.reset is not None:
            # LingBot's per-episode reset: its reply ({"action": None}) is not read
            client.infer(dict(cfg.reset), timeout_s=min(cfg.chunk_timeout_s,
                                                        self._remaining(deadline, runtime)))
        for k in range(cfg.max_chunks):
            if close_chunk is not None and k - close_chunk > cfg.chunks_after_close:
                report["stop"] = "closed"
                break
            budget = self._remaining(deadline, runtime)
            obs = self._observation(runtime, prompt)
            t0 = time.monotonic()
            try:
                reply = client.infer(obs, timeout_s=min(cfg.chunk_timeout_s, budget))
            except VLATimeout as exc:
                if budget < cfg.chunk_timeout_s:
                    raise SkillError(f"VLA episode deadline ({cfg.episode_timeout_s:g}s) passed while "
                                     f"waiting for chunk {k + 1}; no further motion") from exc
                raise SkillError(f"VLA chunk deadline: chunk {k + 1} did not arrive within "
                                 f"{cfg.chunk_timeout_s:g}s; episode ended, no further motion") from exc
            report["infer_ms"].append(round((time.monotonic() - t0) * 1e3, 1))
            # a stop or halt that arrived during inference never reaches the arm
            self._live(harness, halt_generation)
            targets = chunk_targets(reply, cfg, n_joints=n, kin=runtime.kin,
                                    seed_q=arm.get_state().q)
            self._admit(harness, targets, k + 1)
            report["chunks"] += 1
            for i, (q, jaw_open) in enumerate(targets):
                # the episode deadline also cuts a chunk short; the stop latch
                # and halts are SafeArm's per-waypoint checks
                self._remaining(deadline, runtime)
                if not arm.move_joints(q, duration_s=cfg.waypoint_duration_s,
                                       _halt_generation=halt_generation):
                    raise SkillError(f"did not settle at VLA chunk {k + 1} waypoint {i + 1}")
                report["waypoints"] += 1
                if abs(jaw_open - report["last_open_commanded"]) <= GRIP_DEADBAND:
                    continue
                if jaw_open <= CLOSE_BELOW and close_chunk is None:
                    # From the first close command the jaws may hold the
                    # object while `held_object` is still None: record the
                    # provisional marker first (`_reconcile_held` decides).
                    runtime._held_provisional = provisional
                    close_chunk = k
                    report["q_at_close"] = np.asarray(arm.get_state().q, float).copy()
                span = runtime._grip_closed - runtime._grip_open
                arm.set_gripper(runtime._grip_open + span * (1.0 - jaw_open), effort=effort,
                                _halt_generation=halt_generation)
                report["last_open_commanded"] = jaw_open
        report["closed"] = close_chunk is not None
        return report
