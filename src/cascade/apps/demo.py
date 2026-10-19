"""Main demo entry point.

    cascade --task "pick and place pink object" \
             --cameras l515,uvc --arm rebot_rs --llm anthropic

Defaults are the fully-offline stack (mock camera/arm/llm) so the wiring can
always be exercised without hardware or network. `--interactive` keeps the
session open for multiple tasks with persistent memory and beliefs.

Every run now livestreams: N cameras pump continuously (CameraRig), the
WorldWatcher keeps the belief store warm, and an MJPEG dashboard serves the
whole rig to any browser. Routine commands run on the reflex/experience fast
path without an LLM call; only novel tasks reach the model.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple
import numpy as np

from ..agent.advisor import Advisor
from ..agent.llm import make_llm, resolve_llm_profile
from ..agent.orchestrator import AgentOrchestrator, TaskBusyError
from ..agent.reflex import ExperienceMemory, FastPlanner
from ..agent.trace import TraceLogger
from ..config import Cfg, PACKAGE_ROOT, load_demo_config
from ..control.arm_base import make_arm
from ..control.arm_rig import ArmRig
from ..control.kinematics import Kinematics
from ..control.lazy_arm import LazyArm
from ..memory import BeliefStore, EpisodicMemory
from ..memory.beliefs import NEIGHBOUR_COLOUR_IOU
from ..perception.camera_base import make_camera
from ..perception.depth_provider import DepthProvider
from ..perception.grounding import Extrinsics
from ..perception.stream import CameraRig, CameraStream
from ..perception.world import LockedDetector, WatchedCamera, WorldWatcher
from ..perception.workspace import WorkspaceBase, WorkspaceFilter, config_float
from ..safety.harness import SafeArm, SafetyHarness, SafetyLimits
from ..safety.tool_volume import build_tool_volume
from ..skills.runtime import SkillRuntime
from ..types import Frame

logger = logging.getLogger(__name__)


def _camera_cfgs(cfg) -> list[Cfg]:
    """All camera profiles (cfg.cameras when present, else [cfg.camera])."""
    raw = cfg.get("cameras")
    cameras = [Cfg(c) if isinstance(c, dict) else c for c in raw] if raw else [cfg.camera]
    limit = cfg.get("perception_loop", {}).get("max_camera_poll_hz")
    if limit is None:
        return cameras
    import math

    limit = float(limit)
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("max_camera_poll_hz must be finite and at least 1 Hz")
    # Some remote cameras return their last capture on every request. Avoid
    # repeatedly transferring and decoding it faster than the scene renders.
    # Copy the profiles: tuning this runtime must not rewrite calibration or
    # the caller's configured camera rates.
    return [Cfg({**camera.as_dict(), "fps": min(float(camera.get("fps", 30.0)), limit)})
            for camera in cameras]


def _truth_pose_fn(safe_arm, *, camera_rig=None, arm_cfg=None):
    """Ground-truth prop poses when running against a simulator, else None.

    Pigey's postcondition checker prefers a channel the actuator does not
    own. In sim that is the physics state -- the Isaac bridge's RigidPrims or
    the MuJoCo world's free bodies; on real hardware there is none, so this
    returns None and the checker falls back to perception -- identical code
    path in both worlds.

    The callable binds LAZILY (`sim/truth.py::LazyTruthPoseFn`): under the
    MCP server the arm is a LazyArm that materializes on the first motion
    command, so at this point there is nothing to bind to yet. The old
    eager binding therefore returned None in exactly the mode the demo is
    shown in, and every chat-driven pick verified against the belief store
    only. Mock and real arms still yield None on every call, and a real arm
    is never materialized by the probe.

    A warmed, identity-matched Isaac camera supplies its existing transport
    so the first pre-motion snapshot can use physics while the arm is lazy.
    """
    try:
        from ..sim.truth import LazyTruthPoseFn

        # skills only ever hold a SafeArm; the backend is behind .raw
        raw = getattr(safe_arm, "raw", safe_arm)
        fn = LazyTruthPoseFn(raw, camera_rig=camera_rig, arm_cfg=arm_cfg)
        # A MOCK or REAL arm can never grow a truth channel: hand the checker
        # None so its reports say "belief", not a lazy stub that stays empty.
        # Sim arms (Isaac bridge, MuJoCo) may not be bound yet -- keep the
        # lazy callable for them. Judged on the arm's TYPE, without touching
        # a LazyArm (its `_factory` is a private attribute, never __getattr__).
        if not _may_have_truth_channel(raw):
            return None
        return fn
    except Exception:
        return None


def _may_have_truth_channel(raw) -> bool:
    """True for arm objects that are, or will materialize into, a sim arm."""
    name = type(raw).__name__
    if name in ("IsaacArm", "MujocoArm"):
        return True
    if name == "LazyArm":
        real = raw.__dict__.get("_arm")
        if real is not None:
            return type(real).__name__ in ("IsaacArm", "MujocoArm")
        # Not materialized: the factory closes over the profile; ask it for
        # the type without calling it (calling it would power the robot).
        cfg_type = getattr(raw, "_profile_type", None)
        return cfg_type in ("isaac", "mujoco")
    return False


def _arm_cfgs(cfg) -> list[Cfg]:
    """All arm profiles (cfg.arms when present, else [cfg.arm])."""
    raw = cfg.get("arms")
    if raw:
        return [Cfg(a) if isinstance(a, dict) else a for a in raw]
    return [cfg.arm]


def _drift_monitors(cam_cfgs, watched, *, watcher, surface_fn, q_fn, motion_fn, run_dir,
                    memory, arm_name: str = "", ee_frame: str = "") -> list:
    """One ExtrinsicDriftMonitor per camera whose profile enables
    ``extrinsics.drift_monitor``. Enabled where it can never work (a wrist
    camera, a camera type without sensor depth) is a configuration error; an
    uncalibrated camera has nothing to monitor (said on stderr)."""
    from ..calibration.cli import _RGB_ONLY_TYPES
    from ..perception.drift_monitor import DriftMonitorConfig, ExtrinsicDriftMonitor

    out = []
    for ccfg, cam in zip(cam_cfgs, watched):
        ext = ccfg.get("extrinsics") or {}
        dcfg = DriftMonitorConfig.from_config(ext.get("drift_monitor") if hasattr(ext, "get")
                                              else None)
        if not dcfg.enabled:
            continue
        name = cam.stream.name
        if cam.extrinsics.mode != "eye_to_hand":
            raise ValueError(f"camera {name!r}: extrinsics.drift_monitor needs an eye_to_hand "
                             "camera (a wrist camera does not see the arm it rides on)")
        if str(ccfg.get("type", "")) in _RGB_ONLY_TYPES or ccfg.get("rgb_only"):
            raise ValueError(f"camera {name!r}: extrinsics.drift_monitor needs sensor depth; "
                             f"type {ccfg.get('type')!r} has none")
        if not cam.extrinsics.calibrated:
            print(f"[cascade] camera {name}: extrinsic drift monitor not started (camera is "
                  "not calibrated; nothing to monitor)", file=sys.stderr)
            continue
        out.append(ExtrinsicDriftMonitor(
            name, cam, watcher=watcher, surface_fn=surface_fn, q_fn=q_fn, motion_fn=motion_fn,
            config=dcfg, run_dir=run_dir, memory=memory,
            camera_serial=str(ccfg.get("serial", "") or ""), arm=arm_name, ee_frame=ee_frame))
    return out


def _camera_fusion(ccfg, extrinsics) -> tuple[bool, bool | None]:
    """(fuse, map_depth) for one camera stream.

    A camera fuses 3D beliefs when its profile has an `extrinsics:` block
    (or says `fuse_beliefs: true`) -- and, since hand-eye records are gated,
    only while those extrinsics are trusted. A configured calibration that
    is missing/rejected/for another serial turns BOTH off, overriding the
    profile: back-projecting through it would place beliefs and occupancy a
    few centimetres off, the failure the record's gate exists to prevent.
    """
    if not getattr(extrinsics, "calibrated", True):
        return False, False
    return bool(ccfg.get("fuse_beliefs", "extrinsics" in ccfg)), ccfg.get("map_depth")


#: `occupancy.track_payload` replays depth history with the capture's render
#: masks and the frame's capture-time joint state: an Isaac arm only.
_PAYLOAD_NEEDS_ISAAC = "payload segmentation currently requires the Isaac capture contract"


class _MeshArmState:
    """Joint state for an arm's mesh body mask (`body_mask: mesh`).

    The mask is pure config (the URDF meshes), so `_arm_safety` builds it
    before the arm exists; `_build_arm` binds `arm` once the backend is made.
    Unbound, or a standby LazyArm: None, so the mask has no state and the
    frame is refused (fail closed). Never materializes a standby LazyArm:
    powering the motors is not a side effect of looking.
    """

    def __init__(self):
        self.arm = None

    def __call__(self):
        arm = self.arm
        if arm is None or not getattr(arm, "connected", True):
            return None
        return _body_mask_state(arm)


class _RigHeartbeat:
    """Fresh perception keeps EVERY arm's watchdog alive on a rig: the cameras
    watch the shared table, not one arm. The watcher heartbeats this; with one
    harness per arm (#400 follow-up, the MuJoCo pair acceptance): only the
    primary's was kept alive, so a motion on another arm longer than
    `watchdog_s` between its skill's own observes (a MuJoCo pick, the park at
    shutdown) stopped with "perception watchdog: last observation 7.7s old"."""

    def __init__(self, harnesses):
        self._harnesses = tuple(harnesses)

    def heartbeat(self) -> None:
        for harness in self._harnesses:
            harness.heartbeat()


def _watcher_heartbeat(arm_rig, primary_harness):
    """What the watcher heartbeats: the primary's harness itself on one arm
    (unchanged), every arm's harness on a rig of two or more. Harnesses are
    read from the SafeArm's instance dict (never a LazyArm attribute probe)."""
    arms = list(getattr(arm_rig, "arms", {}).values()) if arm_rig is not None else []
    if len(arms) < 2:
        return primary_harness
    return _RigHeartbeat([a.__dict__["harness"] for a in arms])


class _ArmPrepass(NamedTuple):
    """One arm's pure-config build (`_arm_safety`), made before its backend."""

    kin: Kinematics
    harness: SafetyHarness
    #: the MeshBodyMask (`body_mask: mesh`, with a map) or None
    mesh: object | None
    #: its joint-state source; `_build_arm` binds the arm (None: no mesh)
    mesh_state: _MeshArmState | None


def _wants_payload(cfg) -> bool:
    return bool(cfg.get("occupancy", _empty_cfg()).get("track_payload", False))


def _payload_tracked(occupancy) -> bool:
    return occupancy is not None and bool(getattr(occupancy, "tracks_payload", False))


def _body_mask_radius(acfg) -> float:
    """The capsule body mask's radius (`body_mask_radius_m`, metres)."""
    return float(acfg.get("body_mask_radius_m", 0.06))


def _check_body_mask(acfg, arm_name: str) -> None:
    """A `body_mask` block that parses (robot_mask.body_mask_settings refuses
    a bad one), and no mesh mask on an isaac arm."""
    from ..perception.robot_mask import body_mask_settings

    if body_mask_settings(acfg) is not None and acfg.get("type") == "isaac":
        raise ValueError(
            f"arm {arm_name!r}: body_mask: mesh is for real arms, not type isaac. Isaac frames "
            "carry the render self-mask and their capture-time joint state; a mesh mask posed "
            "at the live state would mask an older image. Use body_mask: capsule.")


def _arm_safety(acfg, occupancy, fallback_cfg, *, payload_tracked: bool = False) -> _ArmPrepass:
    """One arm's pure-config build, before any backend exists: kinematics,
    SafetyHarness, park pose and, with a map, its occupancy body mask.

    Every config error an arm build can raise without hardware, a map
    connection or a GPU refuses here: the harness (a malformed
    `safety.tool_volume` block, a tool volume on a tilted `base_pose`,
    `link_radii_m` that do not fit the chain), a `park_q` that is not a
    list and, with a map, the body mask (a bad `body_mask` block, `body_mask:
    mesh` on an isaac arm or with geometry that cannot load, a bad
    `body_mask_radius_m`) and payload tracking (`occupancy.track_payload` on
    a non-isaac arm, beside a mesh mask or on a second arm:
    `payload_tracked` says an earlier arm already tracks it). So the refusal
    comes before `make_arm` and, on an eager build, before `connect()`
    (torque on a real arm), never after it: a failed build leaves arms as
    they are (`_abandon_build`). `_build_runtime` runs it for every arm in a
    pre-pass, so a second arm that refuses never finds the first one
    already connected. `_build_arm` keeps its own checks (defence in depth).
    """
    view = acfg.get("resolved") or fallback_cfg
    arm_name = str(acfg.get("name", acfg.get("type", "arm")))
    # Every arm's grasp section is live under execute(arm=) (spec Q4 a): its
    # jaw-open bound must be valid before any arm exists (R8), never first
    # after a grasp's close on the second arm.
    from ..skills.runtime import close_stall_band, open_wait_timeout_s
    from ..types import SkillError

    try:
        open_wait_timeout_s(view.get("grasp") or _empty_cfg())
        # The close under execute(arm=) reads the bound profile's arm section
        # (the resolved view): its stall band must be valid here too, not only
        # the arm entry's own `gripper` (checked in _build_runtime).
        close_stall_band((view.get("arm") or _empty_cfg()).get("gripper"))
    except SkillError as exc:
        raise ValueError(f"arm {arm_name!r}: {exc}") from None
    kin = Kinematics(
        model_path=acfg.model,
        ee_frame=acfg.get("ee_frame", "gripper_end"),
        n_controlled=int(acfg.get("n_joints", 6)),
        joint_signs=acfg.get("joint_signs"),
        ik_task_weights=acfg.get("ik_task_weights"),
    )
    harness = SafetyHarness(
        SafetyLimits.from_config(view.safety), kinematics=kin, occupancy=occupancy,
        base_pose=_base_transform(acfg),
        # the wrist camera box + open jaws as clearance samples (opt-in per
        # profile: `safety.tool_volume`; a bad block fails the build here)
        tool_volume=build_tool_volume(view, acfg),
        arm_name=arm_name,
    )
    # Optional profile-defined rest pose for the shutdown park (see
    # _park_pose). Defaults to the clamped all-zero fallback when absent.
    park_q = acfg.get("park_q")
    if park_q is not None:
        try:
            harness.park_q = list(park_q)
        except TypeError:
            raise ValueError(f"arm {arm_name!r}: park_q must be a list of joint positions, "
                             f"got {park_q!r}") from None
    mesh = mesh_state = None
    if occupancy is not None:
        from ..perception.occupancy import _PAYLOAD_VS_MESH, OccupancyError
        from ..perception.robot_mask import build_mesh_body_mask

        _check_body_mask(acfg, arm_name)
        mesh_state = _MeshArmState()
        mesh = build_mesh_body_mask(acfg, kin, mesh_state, name=arm_name)
        if mesh is None:
            mesh_state = None
            _body_mask_radius(acfg)                    # a capsule radius that parses
        elif payload_tracked:
            raise OccupancyError(_PAYLOAD_VS_MESH)
        if _wants_payload(fallback_cfg):
            if acfg.get("type") != "isaac":
                raise ValueError(_PAYLOAD_NEEDS_ISAAC)
            if payload_tracked:
                raise ValueError("payload tracking currently requires a single arm")
    return _ArmPrepass(kin, harness, mesh, mesh_state)


def _build_arm(acfg, lazy_arm: bool, occupancy, fallback_cfg, safety=None):
    """One arm: kinematics -> own harness -> backend (maybe lazy) -> SafeArm.

    Each arm resolves its safety envelope from ITS OWN view of the config
    (`acfg.resolved`, see load_demo_config): workspace box, table height,
    velocity cap and joint margins are properties of a particular robot on a
    particular table, and sharing one harness between two arms would apply the
    looser velocity cap to the weaker one.

    `fallback_cfg` covers arm profiles that carry no `resolved` view -- tests
    and harnesses that build a Cfg by hand rather than through
    load_demo_config. Those are single-arm by construction, so the global
    config IS that arm's view.

    `occupancy` is shared, not per-arm: the obstacle cloud describes the
    world, not the robot. It is refreshed once per perception tick by the
    WorldWatcher, so a per-arm copy would leave every non-primary map to age
    past `max_age_s` and be treated as "no data, skip the check" -- an
    obstacle gate that silently stops gating. The per-arm half of that check
    (`min_clearance_m`) already lives in each arm's own SafetyLimits.

    `safety`: this arm's `_arm_safety(...)` when the caller validated it
    already (build_runtime's pre-pass); None builds it here, still before
    the backend (an arm already on `occupancy` that tracks the payload
    counts as the earlier arm).
    """
    n_joints = int(acfg.get("n_joints", 6))
    if safety is None:
        safety = _arm_safety(acfg, occupancy, fallback_cfg, payload_tracked=_payload_tracked(occupancy))
    kin, harness, mesh, mesh_state = safety
    motion_planner = None
    if acfg.get("motion_planner") is not None:
        from ..planning.runtime import RuntimeMotionPlanner
        motion_planner = RuntimeMotionPlanner(acfg.motion_planner, acfg, kin)
        # An explicitly selected SDK must be ready before an actuator exists.
        # CUDA/model startup is not charged to a later live route's deadline.
        motion_planner.prepare()
    if lazy_arm:
        # Perception pre-warms at startup; motors stay untouched until the
        # first motion command materializes the arm (see LazyArm).
        # n_joints must come from the profile: until the arm materializes,
        # LazyArm's hint is the only DOF answer anything can get, and the
        # class default (6) is wrong for a 5-DoF SO-101 or a 7-DoF Panda.
        arm = LazyArm(lambda: make_arm(acfg, kinematics=kin), n_joints=n_joints,
                      profile_type=str(acfg.get("type", "")),
                      on_materialize=(None if occupancy is None
                                      else lambda: _await_occupancy_after_wake(occupancy, acfg)))
    else:
        arm = make_arm(acfg, kinematics=kin)
        arm.connect()
    if occupancy is not None:
        # The depth camera sees THIS arm: register it for body masking so its
        # own links are not integrated as obstacles. Never materialize a
        # LazyArm: real motor bring-up must remain an explicit user action.
        from ..perception.robot_mask import arm_link_points

        base_T = _base_transform(acfg)

        def _body_points(_arm=arm, _kin=kin, _T=base_T, _cfg=acfg, *, frame=None):
            if frame is not None and _cfg.get("type") == "ros2":
                snapshots = getattr(frame, "robot_joint_positions", None) or {}
                q = np.asarray(snapshots.get(str(_cfg.get("name", "arm"))), dtype=float)
                if q.shape != (int(_cfg.get("n_joints", 6)),) or not np.isfinite(q).all():
                    raise ValueError("ROS joint snapshot missing from camera frame")
            elif frame is not None and _cfg.get("type") == "isaac":
                from ..control.isaac_arm import IsaacArm

                q = IsaacArm(_cfg).state_from_frame(frame).q
            elif not getattr(_arm, "connected", True):
                if _cfg.get("type") != "isaac":
                    return None  # occupancy refuses an unmaskable frame
                # The simulator already exists independently of LazyArm.
                # Use a short-lived READ-ONLY connection, reusing IsaacArm's
                # asset->local joint normalization. No actuator factory,
                # targets, gripper, reset or stop; no socket left at teardown.
                from ..control.isaac_arm import IsaacArm

                observer = IsaacArm(_cfg)
                try:
                    observer.connect()
                    q = observer.get_state().q
                finally:
                    observer.disconnect()
            else:
                q = _body_mask_q(_arm)
            pts = arm_link_points(_kin, q)
            if _T is not None:
                pts = pts @ _T[:3, :3].T + _T[:3, 3]
            return pts

        frame_body = None
        if acfg.get("type") in ("isaac", "ros2"):
            # Explicit backend binding. A legacy/malformed Isaac frame must
            # fail closed, not silently read today's q for yesterday's image.
            def frame_body(frame):
                if frame is None:
                    raise ValueError("camera capture snapshot required")
                return _body_points(frame=frame)

        # `body_mask: mesh` (real arms, opt in per profile): the URDF
        # geometry rasterized per depth frame instead of capsules around the
        # joint chain, which left the reBot's jaws and wrist camera in the
        # map (perception/robot_mask.py MeshBodyMask, docs/NVBLOX.md). Built
        # and checked in `_arm_safety`, before the arm; the checks below
        # repeat its config checks (defence in depth).
        _check_body_mask(acfg, str(acfg.get("name", acfg.get("type", "arm"))))
        if mesh_state is not None:
            mesh_state.arm = arm
        if mesh is not None:
            occupancy.add_robot_body_mask(mesh)
        else:
            occupancy.add_robot_body(_body_points, radius_m=_body_mask_radius(acfg),
                                     frame_link_points_fn=frame_body)
        if _wants_payload(fallback_cfg):
            if acfg.get("type") != "isaac":
                raise ValueError(_PAYLOAD_NEEDS_ISAAC)
            def frame_tcp_pose(frame, _cfg=acfg, _kin=kin, _T=base_T):
                from ..control.isaac_arm import IsaacArm
                pose = _kin.fk(IsaacArm(_cfg).state_from_frame(frame).q)
                return pose if _T is None else _T @ pose
            occupancy.track_payload(frame_tcp_pose)
    safe = SafeArm(arm, harness, motion_planner=motion_planner)
    # the mesh body mask (or None): build_runtime tells it when the runtime
    # holds something (`_wire_held_payload`)
    safe.body_mask = mesh
    return arm, safe, kin


def _wire_held_payload(runtime) -> None:
    """Let the primary arm's mesh body mask draw its `held_payload` envelope
    while the runtime holds an object (or may hold one: a provisional hold
    after a close). The capsule mask hid a held object as a side effect; the
    mesh mask is tight, so without this the object would be mapped at the
    TCP and the next motion refused by its own payload. Only the primary
    arm: `held_object` is the runtime's single holding state."""
    mesh = getattr(getattr(runtime, "arm", None), "body_mask", None)
    if mesh is None or not getattr(mesh, "has_held_payload", False):
        return

    def holding() -> bool:
        return (getattr(runtime, "held_object", None) is not None
                or getattr(runtime, "_held_provisional", None) is not None)

    mesh.holding_fn = holding


def _await_occupancy_after_wake(occupancy, acfg) -> None:
    """LazyArm wake hook: wait (bounded by `occupancy.ready_timeout_s`) for
    the map's first masked refresh with the arm powered.

    In standby there is no measured pose to mask the robot out of the depth,
    so the map integrated nothing and latched "robot body pose unavailable"
    (an unmasked arm would map itself as an obstacle). The watcher's next
    tick, a fraction of a second after the arm connects, clears that latch.
    Without this wait the first motion met the latch and was refused; it
    must neither be refused for that nor proceed unchecked. On timeout the
    latch stays and every harness check refuses with its cause."""
    if occupancy.wait_until_usable():
        return
    name = str(acfg.get("name", acfg.get("type", "arm")))
    print(f"[cascade] occupancy: no masked depth refresh within {occupancy.ready_timeout_s:g} s "
          f"after {name} powered on; motion stays refused until one succeeds "
          f"({occupancy.last_error or 'no refresh yet'})", file=sys.stderr)


def _body_mask_q(arm):
    """Joint pose for the occupancy body mask of a CONNECTED arm.

    A backend may serve it without a bus read while it streams
    (`RebotRSArm.body_mask_q`); every other arm keeps its measured pose.
    """
    fn = getattr(arm, "body_mask_q", None)
    return fn() if callable(fn) else arm.get_state().q


def _body_mask_state(arm):
    """Joint state (q AND gripper opening) for the occupancy MESH body mask
    of a CONNECTED arm (`body_mask: mesh`). A backend may serve it without a
    bus read while it streams (`RebotRSArm.body_mask_state`: the tracked
    target with the gripper unknown, so the fingers are swept); every other
    arm keeps its measured state."""
    fn = getattr(arm, "body_mask_state", None)
    return fn() if callable(fn) else arm.get_state()


def _base_transform(acfg):
    """4x4 base->table transform from a profile's `base_pose`, or None.

    None means the base frame IS the table frame -- correct for a single-arm
    rig, and the reason nothing changes for one arm.
    """
    pose = acfg.get("base_pose")
    if pose is None:
        return None
    from ..types import pose_to_transform

    return pose_to_transform(pose)


def _moves_the_base(T) -> bool:
    """A `_base_transform` that moves the arm off the table origin: not None
    and not the exact identity. The identity counts as no base_pose (spec §4,
    D2), so an identity rig keeps today's paths (D5, D6)."""
    return T is not None and not np.array_equal(T, np.eye(4))


def _refuse_eye_in_hand_beside_a_base_pose(cfg, arm_names, arm_cfgs) -> None:
    """Spec D5: a rig with an arm mounted at a (non-identity) `base_pose`
    refuses an eye-in-hand camera, before any arm is made.

    `Extrinsics.cam_to_base()` composes the PRIMARY arm's FK for such a
    camera: a base-frame pose, not the table frame every other camera, the
    map and the beliefs share. No shipped profile combines the two (the
    MuJoCo loader already refuses wrist cameras on a rig). Rigs without a
    base_pose (or with the identity) are not inspected here at all."""
    mounted = [name for name, acfg in zip(arm_names, arm_cfgs, strict=True)
               if _moves_the_base(_base_transform(acfg))]
    if not mounted:
        return
    for i, ccfg in enumerate(_camera_cfgs(cfg)):
        if (ccfg.get("extrinsics") or _empty_cfg()).get("mode") == "eye_in_hand":
            raise ValueError(
                f"camera {str(ccfg.get('name', f'cam{i}'))!r}: an eye_in_hand camera cannot run beside an arm "
                f"mounted with a base_pose ({', '.join(mounted)}): its pose composes the primary arm's "
                "FK, a base frame, not the table frame the rig shares")


def _workspace_bases(cfg, arm_cfgs) -> tuple[WorkspaceBase, ...] | None:
    """Spec D6: every arm's base in the TABLE frame, for the one detection
    filter the watcher and the runtime share.

    None when no arm declares a non-identity `base_pose` (the identity counts
    as none, spec §4): the filter keeps its origin checks, exactly as before.
    Otherwise one entry per arm: at its base_pose translation (an arm without
    one sits at the origin), with that arm's own resolved
    `workspace_filter.base_radius_m` / `base_height_m` / `reach_m` (the
    top-level section for an arm without a resolved view).
    """
    transforms = [_base_transform(a) for a in arm_cfgs]
    if not any(_moves_the_base(T) for T in transforms):
        return None
    defaults = WorkspaceFilter()
    bases = []
    for i, (acfg, T) in enumerate(zip(arm_cfgs, transforms, strict=True)):
        view = acfg.get("resolved")
        own = (view.get("workspace_filter") if view else cfg.get("workspace_filter")) or _empty_cfg()
        x, y, z = (0.0, 0.0, 0.0) if T is None else (float(v) for v in T[:3, 3])
        try:
            bases.append(WorkspaceBase(
                x=x, y=y, z=z,
                radius_m=config_float(own, "base_radius_m", defaults.base_radius_m),
                height_m=config_float(own, "base_height_m", defaults.base_height_m),
                reach_m=config_float(own, "reach_m", defaults.reach_m),
            ))
        except ValueError as e:    # a non-finite base_pose: refuse the build, before any arm is made
            raise ValueError(f"arm {str(acfg.get('name', f'arm{i}'))!r}: base_pose {acfg.get('base_pose')!r}: "
                             f"{e}") from None
    return tuple(bases)


def _shared_map_region(cfg, arm_cfgs):
    """Default region of the ONE occupancy map every arm shares: the union of
    the arms' workspaces, in the map's frame (the TABLE frame).

    A workspace is stated in its arm's base frame, so an arm with a
    `base_pose` contributes the bounding box of its 8 workspace corners
    lifted by that pose; an arm without one contributes its workspace as
    given (the base frame IS the table frame), which keeps a single-arm rig's
    region exactly the old one. Returns (min [3], max [3]) as lists.
    """
    import itertools

    lows, highs = [], []
    for a in arm_cfgs:
        ws = a.get("resolved").safety.workspace if a.get("resolved") else cfg.safety.workspace
        lo, hi = ws.min, ws.max
        T = _base_transform(a)
        if T is not None:
            corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=float)
            lifted = corners @ T[:3, :3].T + T[:3, 3]
            lo, hi = [float(v) for v in lifted.min(axis=0)], [float(v) for v in lifted.max(axis=0)]
        lows.append(lo)
        highs.append(hi)
    return ([min(v[i] for v in lows) for i in range(3)],
            [max(v[i] for v in highs) for i in range(3)])


def _wire_neighbors(arm_rig, raw_arms) -> None:
    """Let every arm's harness see the others' links in the table frame.

    Each arm is given a callable per neighbour rather than the arm object, so
    the harness never holds a robot -- and so this can refuse to read a
    standby LazyArm. Reading joint state off an unmaterialized LazyArm powers
    the motors, and a 50 Hz safety check is the last place that should happen;
    an unreadable neighbour returns None, which the harness treats as unknown
    (skip), falling back to the static workspace partition.

    Only wired when a profile declares `base_pose`: without it, two arms'
    coordinates are not comparable and a distance between them would be a
    meaningless number that silently gates real motion.
    """
    if len(arm_rig) < 2:
        return
    names = arm_rig.names
    posed = {
        n for n, safe in zip(names, arm_rig)
        if getattr(safe.harness, "base_pose", None) is not None
    }
    if len(posed) < 2:
        return

    def reader(other_safe, other_raw):
        def _read():
            # LazyArm's own surface: never materialize the arm from here.
            if not getattr(other_raw, "connected", True):
                return None
            h = other_safe.harness
            if h.base_pose is None:
                return None
            return h.link_points_table_frame(other_safe.get_state().q)
        return _read

    for name, safe in zip(names, arm_rig):
        if name not in posed:
            continue
        for other_name, other_safe, other_raw in zip(names, arm_rig, raw_arms):
            if other_name == name or other_name not in posed:
                continue
            # The neighbour's own thickness travels with its points: the gate
            # subtracts both arms' radii, so each harness needs the other's.
            safe.harness.add_neighbor(
                other_name, reader(other_safe, other_raw),
                link_radii_m=getattr(other_safe.harness.limits, "link_radii_m", None),
            )


def build_runtime(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
) -> tuple[SkillRuntime, object]:
    if cfg.get("robot_mode") == "composed":
        from .robot_runtime import build_robot_runtime
        return build_robot_runtime(cfg, run_dir)
    if cfg.get("robot_mode") == "mobile":
        from .mobile_runtime import build_mobile_runtime

        return build_mobile_runtime(cfg, run_dir)

    # Every owner the build starts is recorded here AS it starts, so a step
    # that raises later (a misspelled `stream.nvblox_map`, a required
    # GraspGen-X that is down) stops exactly those before its error goes up:
    # a failed start must not keep a camera device or a thread running.
    from types import SimpleNamespace

    started = SimpleNamespace(rig=None, detector=None, watcher=None, drift_monitors=[],
                              live_view=None, viewer=None, feedback_arms=[], occupancy=None)
    try:
        return _build_runtime(cfg, run_dir, view=view, lazy_arm=lazy_arm, serve=serve, started=started)
    except BaseException:
        _abandon_build(started)  # never raises: the build's own error is the one re-raised
        raise


#: Per-owner join bound of a FAILED build's teardown (_abandon_build). A
#: watcher stuck in a cold first inference or a hung grab then delays the
#: build's error by this much, instead of hiding it; the thread is named, and
#: the watcher's atexit join (perception/thread_join.py) still catches it.
#: Per owner, not in total: worst case 15 s each for the watcher, the viewer,
#: every drift monitor and every camera stream, plus up to 5 s for the live
#: view's server, up to 3 s per ROS feedback arm detach and the brain
#: detector's own close() bounds (refresh timeout + margin, up to 60 s of SAM
#: warm-up). A reaper that bounds the whole teardown is a follow-up.
_FAILED_BUILD_JOIN_S = 15.0


def _build_runtime(cfg, run_dir: Path, *, view: bool, lazy_arm: bool, serve: bool,
                   started) -> tuple[SkillRuntime, object]:
    from ..planning.backend_evidence import selection_from_environment
    kitchen_renderer = selection_from_environment(cfg, os.environ)

    # ROADMAP #7: the optional memory embedder is resolved FIRST, before any
    # arm, camera or thread exists. A backend that was asked for but cannot
    # run (missing `memory-embed` extra, torch or cached weights) fails the
    # build here with EmbedderUnavailable instead of quietly running without
    # it or with another backend. `backend: none` (shipped) -> None.
    from ..memory.embedder import embedder_config, make_embedder

    memory_embedder = make_embedder(embedder_config(cfg))
    # B43 (ROADMAP #7 v2), both opt-in: crops of every committed watcher
    # detection, and the visual index persisted across restarts. Resolved
    # and validated HERE too, so a bad limit fails the build before any
    # hardware exists, like a bad embedder does.
    visual_recall = _visual_recall_settings(cfg.get("memory", _empty_cfg()), memory_embedder)
    # The bound on waiting for empty jaws to open (open_gripper, the opening
    # of closed jaws before a grasp, an air grasp's reopen): a bad value
    # fails the build here, before any arm exists, never first after a
    # grasp's close (PR-A jaws round 2, M6).
    from ..skills.runtime import close_stall_band, open_wait_timeout_s
    from ..types import SkillError

    try:
        open_wait_timeout_s(cfg.get("grasp") or _empty_cfg())
        # The close-feedback stall band (`gripper.close_stall_band`) of the
        # primary arm, which the close reads, and of every arm in the rig.
        for acfg in (cfg.get("arm"), *(cfg.get("arms") or ())):
            if acfg is not None:
                close_stall_band(acfg.get("gripper"))
    except SkillError as exc:
        raise ValueError(str(exc)) from None

    from ..perception.occupancy import OccupancyMap

    # ── the arm rig: N arms, first = manipulation arm ───────────────────
    arm_cfgs = _arm_cfgs(cfg)

    # One shared obstacle map, spanning every arm's workspace. With a single
    # arm this is exactly the old expression; with several, the union is the
    # honest default -- a region covering only the primary would leave the
    # second arm's half of the table unmapped, and unmapped reads as clear.
    # The map lives in the TABLE frame, so each workspace is lifted by its
    # arm's `base_pose` first (_shared_map_region).
    # An explicit `occupancy.region_min/max` still wins (see from_config).
    region_min, region_max = _shared_map_region(cfg, arm_cfgs)
    occupancy = OccupancyMap.from_config(
        cfg.get("occupancy"),
        workspace_min=region_min,
        workspace_max=region_max,
    )
    started.occupancy = occupancy

    # Every pure-config check of the arms and of the rig runs BEFORE any arm
    # is made (each arm's harness, park pose, body mask and payload tracking
    # in `_arm_safety`; unique arm names): an eager build connects each arm
    # in `_build_arm`, a failed build leaves arms as they are, and an arm
    # that refuses must not find another one already connected.
    arm_names = [str(acfg.get("name", f"arm{i}")) for i, acfg in enumerate(arm_cfgs)]
    if len(set(arm_names)) != len(arm_names):
        raise ValueError(f"duplicate arm names in rig: {arm_names}")    # ArmRig's rule
    _refuse_eye_in_hand_beside_a_base_pose(cfg, arm_names, arm_cfgs)
    # ONE detection filter for the runtime's own fusion and the watcher (R18).
    # On a base_pose rig it excludes every arm's base where it stands and
    # counts reach from any base (spec D6); elsewhere `bases` is None and it
    # equals the filter SkillRuntime builds from the same section.
    workspace_filter = WorkspaceFilter.from_config(cfg.get("workspace_filter"),
                                                   bases=_workspace_bases(cfg, arm_cfgs))
    arm_safety, payload_tracked = [], _payload_tracked(occupancy)
    for acfg in arm_cfgs:
        arm_safety.append(_arm_safety(acfg, occupancy, cfg, payload_tracked=payload_tracked))
        # `_build_arm` registers every arm's payload with the map
        payload_tracked = payload_tracked or (occupancy is not None and _wants_payload(cfg))
    raw_arms, safe_arms, kins = [], [], []
    for i, (acfg, safety) in enumerate(zip(arm_cfgs, arm_safety, strict=True)):
        raw, safe, k = _build_arm(acfg, lazy_arm, occupancy, cfg, safety=safety)
        raw_arms.append(raw)
        safe_arms.append(safe)
        kins.append(k)
        if i == 0:
            kin = k
    arm_rig = ArmRig(safe_arms, arm_names)
    # Inter-arm proximity gating (no-op for a single arm, or when profiles
    # declare no base_pose -- see _wire_neighbors).
    _wire_neighbors(arm_rig, raw_arms)
    # B39: the robot's own pixels from its link geometry, for camera frames
    # that carry no render self-mask (the real rig). Opt-in
    # (`workspace_filter.link_self_mask.enabled`); off, this is None and
    # nothing is loaded or read. Built before any camera opens, so a bad
    # setting fails the build without leaking a stream.
    from ..perception.link_mask import build_link_self_mask

    link_self_mask = build_link_self_mask(
        cfg, list(zip(arm_names, arm_cfgs, kins, raw_arms, strict=True)))
    # The primary arm stays bound to the same names the single-arm code used,
    # so every existing call site (56 `self.arm` uses in the skill runtime,
    # shutdown_runtime, the truth-pose hook) is untouched by the rig.
    arm = raw_arms[0]
    safe_arm = arm_rig.primary
    harness = safe_arm.harness

    # ── the camera rig: N continuous streams, first = manipulation ──────
    cam_cfgs = _camera_cfgs(cfg)
    streams, watched = [], []
    # The detector is built AFTER rig.open() (below): Seeed's WRC rig reset
    # the depth cameras' USB link when the detector's CUDA model loaded before
    # they were streaming (WRC 37285b0; the SAM warm-up loads CUDA the same
    # way, tests/test_detector_after_camera_open.py).

    def fk():
        # Eye-in-hand extrinsics need live FK, but camera streams and the
        # watcher must never materialize a LazyArm (that would power the
        # motors as a side effect of starting the gateway). Until the arm
        # is up, RGB keeps streaming without a camera pose and fusion skips it.
        if not getattr(arm, "connected", True):
            from ..types import SkillError

            raise SkillError("eye-in-hand extrinsics need the arm (still in standby)")
        return kin.fk(arm.get_state().q)

    for i, ccfg in enumerate(cam_cfgs):
        extrinsics = Extrinsics.from_config(
            ccfg.get("extrinsics", _empty_cfg()), fk_tcp2base=fk,
            camera_serial=ccfg.get("serial"),
        )
        capture_geometry = None
        if any(a.get("type") == "ros2" for a in arm_cfgs):
            def capture_geometry(_ext=extrinsics):
                snapshots = {}
                primary_q = None
                for idx, (raw, acfg) in enumerate(zip(raw_arms, arm_cfgs)):
                    if acfg.get("type") != "ros2":
                        continue
                    if not getattr(raw, "connected", True):
                        raise RuntimeError("camera geometry needs connected ROS joint feedback")
                    q = np.array(raw.get_state().q, dtype=float, copy=True)
                    snapshots[str(acfg.get("name", "arm"))] = q
                    if idx == 0:
                        primary_q = q
                T = (_ext.T_compensation @ kin.fk(primary_q) @ _ext.T
                     if _ext.mode == "eye_in_hand" and primary_q is not None
                     else _ext.cam_to_base())
                return T, snapshots
        stream = CameraStream(
            make_camera(ccfg),
            name=str(ccfg.get("name", f"cam{i}")),
            rate_hz=float(ccfg.get("fps", 30.0)),
            capture_transform=(extrinsics.cam_to_base if "extrinsics" in ccfg else None),
            capture_geometry=capture_geometry if "extrinsics" in ccfg else None,
        )
        streams.append(stream)
        if not extrinsics.calibrated:
            print(f"[cascade] camera {stream.name}: {extrinsics.calibration_error} "
                  "-- streaming only, no 3D fusion", file=sys.stderr)
        fuse, map_depth = _camera_fusion(ccfg, extrinsics)
        watched.append(
            WatchedCamera(
                stream=stream,
                depth=DepthProvider(ccfg),
                extrinsics=extrinsics,
                # A camera without calibrated extrinsics must not fuse 3D
                # beliefs (garbage base-frame positions); it still streams
                # video + overlays + heartbeats.
                fuse=fuse,
                map_depth=map_depth,
            )
        )
    rig = CameraRig(streams)
    # From here a failure closes the rig (build_runtime -> _abandon_build):
    # a partial build must not leak open camera streams, even when the ROS
    # attach or rig.open() itself is what failed.
    started.rig = rig
    feedback_arms = started.feedback_arms
    if lazy_arm and any(c.get("extrinsics", _empty_cfg()).get("mode") == "eye_in_hand"
                        for c in cam_cfgs):
        # Native ROS controllers already own the drives. Ros2Arm.connect
        # only subscribes to measured feedback and creates command handles;
        # it sends no target, gripper, enable, or resume command. Attach it
        # before the wrist camera so every frame can bind FK and body-mask
        # joints from one measured snapshot, even at startup under STOP.
        # Preserve LazyArm itself and leave every non-ROS backend untouched.
        # A failed build detaches them again (passive: no active goals).
        for raw, acfg in zip(raw_arms, arm_cfgs):
            if acfg.get("type") == "ros2":
                feedback_arms.append(raw)
                raw.get_state()
    rig.open()
    # The detector's threads (the brain detector's SAM warm-up and refreshes)
    # are joined, bounded, by its close(): a failed build closes it last.
    detector = started.detector = LockedDetector(_make_detector(cfg))
    rig.primary.warm_up(int(cam_cfgs[0].get("warmup_frames", 5)))
    # Compared: the cameras the watcher asks the detector about (the primary
    # and every fusing camera; world.WorldWatcher B4 skips the rest).
    _warn_shared_camera_keys(detector, [cam.stream for cam in watched],
                             skip={str(cam.stream.name) for cam in watched[1:] if not cam.fuse})

    memory = EpisodicMemory(
        horizon_s=float(cfg.memory.get("horizon_s", 15.0)),
        frame_horizon_s=float(cfg.memory.get("frames_horizon_s", 600.0)),
        **({"embedder": memory_embedder} if memory_embedder is not None else {}),
        **({"max_detections": visual_recall["detections"]["max_detections"]}
           if visual_recall["detections"] is not None else {}),
    )
    if memory_embedder is not None:
        print(f"[cascade] memory embedder: {memory_embedder.name} "
              f"(joint image-text: {memory_embedder.joint_space})")
    mcfg = cfg.get("memory", _empty_cfg())
    # Instance-level association (2026-10-08): a camera frame's detections
    # are matched to beliefs one-to-one, so two identical props inside the
    # 8 cm gate stay two beliefs. `memory.instance_association: false` is the
    # per-detection baseline for a live A/B (memory/beliefs.py update_frame).
    # Colour identity per camera (B32b): `memory.per_camera_colour: false` is
    # the one-name baseline for a live A/B (memory/beliefs.py _identity_ok).
    # Size consistency (B40): `memory.size_gate: false` is the no-size-check
    # baseline for a live A/B (memory/beliefs.py _size_ok).
    beliefs = _belief_store(mcfg)
    # Persistent spatial memory (ROADMAP item): the world model survives a
    # restart, so the robot does not re-discover a table it already mapped and
    # can answer "where was the mug" on a cold boot. Everything loaded is aged
    # past the visible horizon, so it reads as `remembered` -- the agent is
    # never told it can SEE something it has not looked at this session.
    # Disable with `memory.persist_beliefs: false` (or CASCADE_BELIEFS=0).
    beliefs_path = None
    if _beliefs_persist_enabled(mcfg):
        beliefs_path = Path(
            os.environ.get("CASCADE_BELIEFS_PATH")
            or str(mcfg.get("beliefs_path") or (PACKAGE_ROOT / "runs" / "beliefs.json"))
        )
        try:
            n = beliefs.load(
                beliefs_path,
                max_age_s=float(mcfg.get("beliefs_max_age_s",
                                         BeliefStore.DEFAULT_MAX_AGE_S)),
            )
            if n:
                print(f"[cascade] recalled {n} object(s) from {beliefs_path}")
        except Exception as e:  # noqa: BLE001 - memory must never block startup
            print(f"[cascade] belief memory not loaded ({e})", file=sys.stderr)
    # B43 (opt-in `memory.persist_episodic`): the visual index survives a
    # restart the same way. Restored entries are aged by wall clock and
    # floored at LOADED_MIN_AGE_S, so every restored hit reads `remembered`;
    # a file from another embedder is refused, a corrupt one ignored.
    episodic_path = None
    if visual_recall["persist"] is not None:
        episodic_path = visual_recall["persist"]["path"]
        try:
            n = memory.load_visual(episodic_path, max_age_s=visual_recall["persist"]["max_age_s"])
            if n:
                print(f"[cascade] recalled {n} remembered appearance(s) from {episodic_path}")
        except Exception as e:  # noqa: BLE001 - memory must never block startup
            print(f"[cascade] visual memory not loaded ({e})", file=sys.stderr)
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        rig.primary, watched[0].depth, detector, watched[0].extrinsics,
        kin, safe_arm, memory, beliefs, trace, cfg,
    )
    runtime.rig = rig
    # ASPIRE (spec §4.5): the sim-only retry ledger. None on every real or unknown
    # profile and whenever a switch is off or its config is bad (`ledger_for`
    # fails closed). Config only: no arm is touched.
    from ..agent.sim_retry import ledger_for

    runtime.sim_retry = ledger_for(cfg)
    # ASPIRE (spec §4.2): the note library is read ONCE, here; run.json records
    # it and every trace row carries its sha. Config only: no arm is touched.
    _stamp_run(runtime, trace, cfg)
    # ASPIRE (spec §4.8): the chat line and learning.json, from that snapshot. None on
    # every real profile and whenever `agent.learning_surface` is off (`learning_state_for`
    # fails closed); written once here, after each learning event, and by the MCP server
    # to its launch state (preflight R31). Config only: no arm is touched.
    from ..agent.learning_surface import learning_state_for

    runtime.learning_state = learning_state_for(cfg, runtime.learned, runtime.sim_retry, trace.run_dir)
    _wire_held_payload(runtime)
    runtime._kitchen_camera_renderer = kitchen_renderer
    # Where to persist the world model on shutdown (None = disabled).
    runtime.beliefs_path = beliefs_path
    # Where to persist the visual index on shutdown (B43; None = disabled).
    runtime.episodic_path = episodic_path
    # The arm rig hangs off the runtime the same way the camera rig does.
    # `runtime.arm` stays the primary SafeArm, so nothing that predates the
    # rig has to learn about it; a skill called with `arm="<name>"` is
    # rebound for that one call by SkillRuntime.execute().
    runtime.arm_rig = arm_rig
    # Each other arm's own Kinematics (the one its harness uses) and resolved
    # profile, bound by execute(arm=<name>) for that call (spec Q4 a, D3). The
    # primary keeps the runtime's own `kin` / `cfg`.
    from ..skills.arm_frame import arm_binding

    runtime.register_arm_bindings({
        name: arm_binding(cfg, acfg, k)
        for name, acfg, k in zip(arm_names[1:], arm_cfgs[1:], kins[1:], strict=True)})
    # Both fusion paths consult the same link self-mask (None = off) and the
    # same workspace filter: the runtime's get_observation / _reobserve here,
    # the watcher below.
    runtime._link_self_mask = link_self_mask
    runtime._workspace = workspace_filter

    # Pigey (arXiv:2607.21725) closed loop: verify each primitive's physical
    # effect against a channel the actuator does not own. In sim the bridge
    # can report ground-truth prim poses, which beats perception; on the real
    # rig the checker falls back to the belief store automatically.
    if bool(cfg.get("verify_effects", True)):
        runtime.attach_verifier(object_pose=_truth_pose_fn(safe_arm, camera_rig=rig, arm_cfg=arm_cfgs[0]))

    pcfg = cfg.get("perception_loop", _empty_cfg())
    if bool(pcfg.get("enabled", True)):
        watcher = WorldWatcher(
            watched, beliefs=beliefs, detector=detector,
            # Empty config means open-world: the watcher reports whatever the
            # detector sees. Never substitute a hard-coded vocabulary here --
            # that silently turns the always-on world model into a closed set.
            classes=list(cfg.get("detect_classes") or []) or None,
            rate_hz=float(pcfg.get("rate_hz", 3.0)),
            harness=_watcher_heartbeat(arm_rig, harness),
            workspace=workspace_filter,
            occupancy=occupancy,
            link_mask=link_self_mask,
            # B43: only with memory.visual_recall_detections (+ an embedder);
            # otherwise the watcher is constructed exactly as before.
            **({"visual_recall": _detection_recorder(memory, visual_recall["detections"])}
               if visual_recall["detections"] is not None else {}),
        )
        started.watcher = watcher
        watcher.start()
        runtime.watcher = watcher

    # Extrinsic drift monitors (perception/drift_monitor.py): per eye-to-hand
    # camera, opt-in. Read-only on the arm: joints via a reader that never
    # materialises a LazyArm, motion from the harness / a paused watcher.
    def _q_now():
        if not getattr(arm, "connected", True):
            return None
        return arm.get_state().q

    def _moving():
        w = runtime.watcher
        return bool(harness.motion_active or (w is not None and w.is_paused))

    def _surface():
        from ..calibration.robot_surface import RobotSurface

        return RobotSurface.from_kinematics(kin, str(arm_cfgs[0].model))

    runtime.drift_monitors = _drift_monitors(
        cam_cfgs, watched, watcher=runtime.watcher, surface_fn=_surface, q_fn=_q_now,
        motion_fn=_moving, run_dir=trace.run_dir, memory=memory,
        arm_name=str(arm_cfgs[0].get("name", "") or ""),
        ee_frame=str(arm_cfgs[0].get("ee_frame", "") or ""))
    started.drift_monitors = runtime.drift_monitors
    for mon in runtime.drift_monitors:
        mon.start()
        c = mon.config
        print(f"[cascade] camera {mon.name}: extrinsic drift monitor on (every {c.period_s:g} s "
              f"while the arm is static; flags > {1000 * c.max_offset_m:.0f} mm / "
              f"{c.max_rot_deg:g} deg x{c.consecutive}; auto_apply={c.auto_apply})")

    # ── live view: headless by default, opened on demand ─────────────────
    # Chat (Hermes / OpenClaw / any MCP host) is the interface; the browser
    # dashboard is a diagnostic surface you attach. `serve=False` from the CLI
    # still forces "off", but the default is now LAZY: nothing binds a port
    # until someone asks to look. Perception keeps running either way, so the
    # agent can answer "what do you see?" with no dashboard at all.
    runtime.stream_server = None
    scfg = cfg.get("stream", _empty_cfg())
    from .live_control import LiveViewController, resolve_mode

    mode, idle_timeout = resolve_mode(scfg, os.environ.get)
    if not serve:
        mode = "off"
    # B47 (opt-in): `stream.wrist_narration: true` + a wrist stream -> the
    # dashboard highlights that tile during motion skills with one line built
    # from verifiable state only. Off, or no wrist stream: nothing attached,
    # nothing served differently (apps/wrist_narration.py).
    from .wrist_narration import build_narrator

    runtime.wrist_narrator = build_narrator(runtime, scfg)
    wrist_kw = ({"wrist_view": runtime.wrist_narrator.camera}
                if runtime.wrist_narrator is not None else {})

    # The nvblox map as one more live-view tile (`nvblox_map`,
    # apps/map_tile.py): resolved HERE so a misspelled switch fails the
    # build. `auto` = only when this runtime has an occupancy map. The tile
    # reads the map's lock-free snapshot and renders only for a connected
    # viewer; standby is read from LazyArm's own flag (never materialized).
    from .map_tile import MapTile, map_tile_settings

    map_settings = map_tile_settings(scfg, occupancy=occupancy, occupancy_cfg=cfg.get("occupancy"))
    map_workspace = (list(region_min), list(region_max))

    def _map_tile():
        if not map_settings.enabled:
            return {}
        return {"map_tile": MapTile(
            occupancy, map_settings, workspace=map_workspace,
            arm_up_fn=lambda: getattr(runtime.arm.raw, "connected", True),
            objects_fn=runtime.beliefs.summary,
        )}

    def _make_stream_server():
        from .stream_server import StreamServer

        return StreamServer(
            rig,
            state_fn=lambda: _runtime_state(runtime),
            host=str(scfg.get("host", "0.0.0.0")),
            port=int(scfg.get("port", 8090)),
            fps=float(scfg.get("fps", 15.0)),
            quality=int(scfg.get("quality", 80)),
            keyframes_dir=trace.run_dir / "keyframes",
            runtime_fn=lambda: runtime,
            depth_max_m=float(scfg.get("depth_max_m", 2.0)),
            on_poll=lambda: runtime.live_view.note_poll(),
            lan_commands=bool(scfg.get("lan_commands", False)),
            **wrist_kw,
            **_map_tile(),
        )

    external_view = os.environ.get("CASCADE_EXTERNAL_VIEW_URL")
    if external_view:
        from .external_view import ExternalLiveViewController
        runtime.live_view = ExternalLiveViewController(external_view)
        mode = "supervised"
    else:
        runtime.live_view = LiveViewController(
            _make_stream_server, mode=mode, idle_timeout_s=idle_timeout
        )
        # Ours, recorded BEFORE open() binds: an error after the bind (a
        # Ctrl+C, a reaper thread that cannot start) still closes it and
        # frees the port. An external view is a supervised service: never
        # recorded, never closed by a failed build.
        started.live_view = runtime.live_view
    if mode == "eager":
        opened = runtime.live_view.open(reason="stream.mode: eager")
        if not opened.get("ok"):
            print(f"[cascade] livestream disabled ({opened.get('error')})",
                  file=sys.stderr)
    # Back-compat: existing code (and tests) read runtime.stream_server.
    # It tracks the controller, so it is None while the view is closed.
    runtime.stream_server = runtime.live_view.server

    if view:
        from .live_view import RigViewer

        vcfg = cfg.get("viewer", {}) or {}
        runtime.viewer = RigViewer(
            rig,
            show_depth=bool(vcfg.get("show_depth", True)),
            # Default to the dashboard's depth range so both views agree.
            depth_max_m=float(vcfg.get("depth_max_m", scfg.get("depth_max_m", 2.0))),
        )
        started.viewer = runtime.viewer
        runtime.viewer.start()

    # Verified-backend banner. GraspGen-X is probed HERE (300 ms) rather
    # than on the first grasp, so the operator sees "grasp_planner=obb
    # (graspgenx down)" before anything moves instead of an 8 s stall and a
    # buried memory note. Occupancy was probed when its map was built.
    _probe_grasp_backend(runtime)
    _probe_vla_executor(runtime)
    runtime.trace.backends_fn = runtime.backends
    b = runtime.backends()
    print(f"[cascade] backends: grasp_planner={b['grasp_planner']} | occupancy={b['occupancy']}")
    if not b["occupancy_live"] and occupancy is not None:
        print("[cascade] WARNING: occupancy is enabled in config but no bridge answered -- "
              "the clearance gate is OFF (start scripts/serve_occupancy.sh)", file=sys.stderr)
    elif (occupancy is not None and occupancy.status
          and occupancy.status.get("backend") == "voxel"):
        # The numpy fallback can exceed the per-frame timeout on a full depth
        # frame (measured ~0.6 s vs 0.5 s on a Jetson Thor at 1280x720); every
        # refresh then fails and the body-mask latch refuses all motion.
        print("[cascade] NOTE: occupancy runs the CPU voxel fallback, which can be too slow "
              f"for occupancy.timeout_ms={int((cfg.get('occupancy') or {}).get('timeout_ms', 500))}; "
              "install the 'occupancy' "
              "extra (warp-lang) and restart scripts/serve_occupancy.sh for the GPU backend",
              file=sys.stderr)
    # Capability matrix (ROADMAP #12, Waddle): what this rig can do, read
    # from the probed state above -- camera depth chain, sidecars, arms,
    # verifier, memory. The MCP server trims its tool surface by it; the
    # line here is what makes a withheld tool visible to the operator.
    from .capabilities import capability_matrix, format_matrix

    print(f"[cascade] capabilities: {format_matrix(capability_matrix(runtime))}")
    return runtime, arm


#: run.json `purpose` values (preflight R18), from CASCADE_RUN_PURPOSE: the
#: verifier and acceptance stacks set theirs; harvest and grading count only
#: "live" runs, so a stack's own runs never teach the library.
RUN_PURPOSES = ("live", "verify", "acceptance")


def _run_purpose() -> str:
    """CASCADE_RUN_PURPOSE, case and surrounding space aside: unset or empty is
    "live"; any value outside RUN_PURPOSES is "unknown" (never "live")."""
    raw = os.environ.get("CASCADE_RUN_PURPOSE", "")
    value = raw.strip().lower()
    if not value:
        return "live"
    if value in RUN_PURPOSES:
        return value
    logger.warning("CASCADE_RUN_PURPOSE=%r is none of %s: this run records purpose \"unknown\"",
                   raw, "/".join(RUN_PURPOSES))
    return "unknown"


def _stamp_run(runtime, trace, cfg) -> None:
    """Take the runtime's one note-library snapshot (`runtime.learned`) and
    record the run in `run.json` (spec §4.2, preflight R18/R23); later trace rows
    carry its `library_sha`. `sim_retry` says whether the runtime built a retry
    ledger (`runtime.sim_retry`, set before this call). `library_root` is the
    resolved path (symlinks and `..` resolved); `skipped` counts the skipped
    note files and lists the first
    `notes.SKIPPED_RECORDED` by id and reason. Never fails the build: a library
    that cannot be read is served as empty (`notes.config_snapshot`;
    `library_error` says why), and a run.json that cannot be written leaves the
    rows unstamped and serves no note, since a served note must be on record."""
    import dataclasses
    from datetime import datetime, timezone

    from .. import sim_profile
    from ..skills import notes

    snapshot = notes.config_snapshot(cfg, sim_profile.profile_class(cfg))
    try:
        meta = {"schema": 1, **snapshot.manifest(),
                "library_root": None if snapshot.root is None else str(snapshot.root.resolve()),
                "library_error": snapshot.error,
                "skipped": {"count": len(snapshot.skipped),
                            "first": [dict(entry) for entry in snapshot.skipped[:notes.SKIPPED_RECORDED]]},
                "arm_types": sim_profile.arm_types(cfg),
                # the ledger this runtime HAS (A7): never true while none exists
                "sim_retry": getattr(runtime, "sim_retry", None) is not None, "purpose": _run_purpose(),
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        trace.stamp(meta)
    except Exception as e:  # noqa: BLE001 -- the run record never stops a robot
        logger.warning("ASPIRE notes: run manifest %s not written (%s: %s); serving no note this session",
                       trace.run_dir / "run.json", type(e).__name__, e)
        snapshot = dataclasses.replace(snapshot, served=())
    runtime.learned = snapshot


def _probe_grasp_backend(runtime) -> None:
    """Resolve `grasp.backend: graspgenx` to a live server / stub / down NOW."""
    gcfg = runtime.cfg.grasp
    if str(gcfg.get("backend", "obb")) == "hug":
        _probe_hug_backend(runtime, gcfg)
        return
    if str(gcfg.get("backend", "obb")) != "graspgenx":
        runtime.grasp_planner_used = str(gcfg.get("backend", "obb"))
        return
    try:
        from ..grasping.graspgenx_backend import GraspGenXPlanner

        planner = GraspGenXPlanner(gcfg)
        planner.probe()
        runtime._graspgenx = planner
        runtime.grasp_planner_used = planner.describe()
    except Exception as e:  # noqa: BLE001 -- optional profiles may fall back, visibly
        if bool(gcfg.graspgenx.get("required", False)):
            raise RuntimeError(f"GraspGen-X required at startup: {e}") from e
        runtime._graspgenx_down = True
        runtime._graspgenx_retry_after = time.monotonic() + 5.0
        runtime.grasp_planner_used = "obb (graspgenx down)"
        print(f"[cascade] WARNING: grasp.backend=graspgenx but no server answered "
              f"({str(e)[:100]}); analytic OBB fallback; will retry the server", file=sys.stderr)


def _probe_vla_executor(runtime) -> None:
    """`grasp.executor: vla` (B49, opt-in): build the executor -- a bad
    `grasp.vla` block or an unknown executor name fails the build -- and probe
    its policy server NOW, so the banner and the capability matrix say whether
    the route can run before anything moves. The analytic executor (the
    default) attaches nothing."""
    from ..grasping.vla_executor import VLAExecutor, executor_name

    gcfg = runtime.cfg.grasp
    if executor_name(gcfg) != "vla":
        return
    executor = VLAExecutor.from_cfg(gcfg)
    runtime.vla_executor = executor
    status = executor.probe()
    if status["answered"]:
        print(f"[cascade] grasp executor: vla -> {status['detail']}")
    else:
        print(f"[cascade] WARNING: grasp.executor=vla but {status['detail']}; label grasps are "
              "refused (no analytic fallback) until a policy server answers and the runtime "
              "restarts", file=sys.stderr)


def _probe_hug_backend(runtime, gcfg) -> None:
    """`grasp.backend: hug` (opt-in): same startup contract as GraspGen-X.
    A required profile refuses to start without a real HUG server (the
    protocol stub included); an optional one says so and falls back to OBB."""
    try:
        from ..grasping.hug_backend import HugPlanner

        planner = HugPlanner(gcfg)
        planner.probe()
        runtime._hug = planner
        runtime.grasp_planner_used = planner.describe()
    except Exception as e:  # noqa: BLE001 -- optional profiles may fall back, visibly
        if bool((gcfg.get("hug") or {}).get("required", True)):
            raise RuntimeError(f"HUG required at startup: {e}") from e
        runtime._hug_down = True
        runtime._hug_retry_after = time.monotonic() + 5.0
        runtime.grasp_planner_used = "obb (hug down)"
        print(f"[cascade] WARNING: grasp.backend=hug but no HUG server answered "
              f"({str(e)[:100]}); analytic OBB fallback; will retry the server", file=sys.stderr)


def _runtime_state(runtime) -> dict:
    """Live world state for the dashboard/MCP (never touches the lazy arm)."""
    _, status = runtime.camera.overlay() if hasattr(runtime.camera, "overlay") else ([], "n/a")
    connected = getattr(runtime.arm.raw, "connected", True)
    out = {
        "agent_status": status,
        "task": runtime.current_task,
        "holding": runtime.held_object,
        # dispatch tier of the last command (reflex/experience/llm/mcp-host)
        "last_path": getattr(runtime, "last_path", None),
        "objects": runtime.beliefs.summary(),
        "arm_connected": connected,
        "stopped": bool(runtime.arm.harness.estopped),
        "connection_error": None,
        # human-readable narration: newest events last (observations, skill
        # calls, outcomes) -- the dashboard renders this as the activity feed
        "events": runtime.memory.digest(max_lines=14).splitlines(),
        # learned grasp priors, one line per object profile (booth panel)
        "grasp_memory": runtime.grasp_memory.summary().splitlines(),
        # verified sidecars (grasp planner / occupancy) -- what is REALLY on
        "backends": runtime.backends(),
    }
    # what this rig can do, from probed state (apps/capabilities.py): the
    # dashboard shows it next to `backends`, so a tool the MCP server
    # withholds is explained where the operator is already looking
    from .capabilities import capability_matrix

    out["capabilities"] = capability_matrix(runtime)
    # Probe the class so LazyArm.__getattr__ cannot connect hardware merely
    # to render the dashboard. Joint feedback alone does not imply that a
    # Franka in Reflex can accept motion.
    raw = runtime.arm.raw
    if isinstance(raw, LazyArm):
        raw = raw._arm  # existing backend only; never materialize for a dashboard
    if getattr(type(raw), "motion_status", None) is not None:
        out["arm_motion"] = raw.motion_status
    if runtime.watcher is not None:
        out["perception"] = runtime.watcher.stats()
    # B47: only with a wrist narrator (opt-in + a wrist stream); absent, the
    # state carries exactly the keys it always had.
    narrator = getattr(runtime, "wrist_narrator", None)
    if narrator is not None:
        out["wrist_view"] = narrator.snapshot(runtime)
    monitors = getattr(runtime, "drift_monitors", None) or []
    if monitors:
        # per camera: ok / drift, fusing, last check, passive candidate
        out["extrinsics_drift"] = {m.name: m.status() for m in monitors}
    return out


def _park_pose(arm) -> list[float]:
    """The rest pose for the shutdown park.

    Prefers the profile's `park_q` (an exact, arm-author-chosen pose, e.g. the
    reBot's mechanical zero). When absent -- or mis-sized -- falls back to the
    all-zero vector clamped into the harness joint margin, because some arms
    (the reBot RS) have a joint whose lower limit is 0 rad and a literal zero
    would sit on the mechanical stop. Returns [0.0]*n when no kinematics or
    limits are available."""
    n = int(arm.n_joints)
    park_q = getattr(arm.harness, "park_q", None)
    if park_q is not None and len(park_q) == n:
        return [float(v) for v in park_q]
    kin = getattr(arm.harness, "kin", None)
    limits = getattr(kin, "joint_limits", None) if kin is not None else None
    q = [0.0] * n
    if limits is not None:
        lo, hi = limits
        m = float(arm.harness.limits.joint_margin)
        for j in range(n):
            low, high = float(lo[j]) + m, float(hi[j]) - m
            q[j] = min(max(0.0, low), high)
    return q


def _park_gripper(arm) -> None:
    """Close the gripper to its zero before torque is cut, so a held object is
    not dropped (or the jaws left open) at disconnect. The reBot RS closes
    under torque/stall detection (`close_gripper_torque`); other backends have
    no such method and are left as they are -- their disconnect is not a real
    drop. Runs after the arm is already materialized by `move_joints`, so
    reaching through `.raw` for the backend method does not power the bus as a
    side effect of a probe.
    """
    closer = getattr(arm.raw, "close_gripper_torque", None)
    if callable(closer):
        print("[cascade] closing gripper (torque-detected) before disconnect")
        closer()


def _park_move(arm, duration_s: float, stage: dict) -> bool:
    """The park motion. If ONLY the occupancy map refuses it, park once more
    without the map.

    Teardown releases torque right after this. A refused park therefore
    drops a raised arm, which is worse than the obstacle the map reported:
    on a real rig that is often the arm's own unmasked parts or the object
    still in its jaws. Joint limits, workspace, table and velocity gates all
    still run; an e-stop latch is never bypassed.
    """
    from ..types import SafetyViolation

    try:
        return arm.move_joints(_park_pose(arm), duration_s=duration_s, joint_margin=0.0)
    except SafetyViolation as exc:
        harness = arm.harness
        occupancy = getattr(harness, "occupancy", None)
        if occupancy is None or harness.estopped or "occupancy" not in str(exc):
            raise
        print(f"[cascade] park refused by the occupancy map ({exc}); parking without the "
              "map: releasing torque with the arm raised would drop it")
        stage["occupancy_bypassed"] = str(exc)
        harness.occupancy = None
        try:
            return arm.move_joints(_park_pose(arm), duration_s=duration_s, joint_margin=0.0)
        finally:
            harness.occupancy = occupancy


def _shutdown_park_required(raw) -> bool:
    """Inspect only an existing backend; never connect an arm for teardown."""
    if isinstance(raw, LazyArm):
        raw = raw._arm
    return getattr(raw, "shutdown_park_required", True) is not False


def _park_arm(runtime, duration_s: float = 2.0) -> dict:
    """Slowly drive every arm to its zero pose before torque is cut.

    `disconnect()` disables torque, so an arm left at the working height drops
    under gravity when the program exits. Parking first streams the joints
    back to zero (near the table) over `duration_s`, so any residual drop is a
    few centimetres onto the table instead of a free fall, then closes the
    gripper so it is not left open (or dropping a held object). It must run
    FIRST in teardown: `move_joints` -> `begin_motion` checks the perception
    watchdog, which is only fresh while the watcher is still running.

    Never raises and never blocks teardown: a standby arm (a LazyArm that was
    never materialized) is not touched (parking it would energize the motors
    for nothing), an e-stopped arm is left exactly where it is (the latch means
    "do not move"), and any arm that cannot move simply has its torque cut by
    the disconnect that follows. Externally controlled arms can opt out:
    native ROS controllers retain the hold and must never receive a new park
    or gripper command merely because this client is exiting.
    """
    # A transport-only disconnect can retain the existing simulation drives
    # without attempting task recovery. Other backends keep their existing
    # park-before-disconnect policy: cutting hardware torque aloft can drop
    # the arm and payload. Decide per arm, including in mixed backend rigs.
    from ..lifecycle import teardown_receipt

    stages = []
    retained = ("held_object", "_held_provisional", "_contact_episode",
                "_carry_attachment", "_release_episode")
    possible_load = any(getattr(runtime, name, None) is not None for name in retained)
    rig = getattr(runtime, "arm_rig", None)
    arms = list(rig) if rig is not None and len(rig) > 1 else [runtime.arm]
    for index, arm in enumerate(arms):
        stage = {"stage": str(index), "ok": True, "complete": True}
        stages.append(stage)
        if arm is None:
            stage["skipped"] = "absent"
            continue
        try:
            if arm.harness.estopped:
                stage["skipped"] = "estopped"
                continue
            if not getattr(arm.raw, "connected", True):
                stage["skipped"] = "not_connected"
                continue
            if not _shutdown_park_required(arm.raw):
                stage["skipped"] = "external_controller_retains_hold"
                continue
            if getattr(arm.raw, "disconnect_preserves_drive_state", False) and (
                    possible_load
                    or getattr(arm.harness, "_pending_contact_episode", None) is not None
                    or getattr(arm.harness, "_pending_release_episode", None) is not None):
                print("[cascade] park skipped: retained or possible payload/contact/release")
                stage["skipped"] = "retained_or_possible_load"
                continue
            print("[cascade] parking arm to safe rest pose before disconnect")
            # joint_margin=0 lets the park reach the mechanical stop (an exact
            # zero on the reBot's joint 2/3, whose lower limit IS 0); every
            # other safety gate (workspace, table, velocity) still runs.
            completed = _park_move(arm, duration_s, stage)
            _park_gripper(arm)
            if completed is False:
                stage.update(ok=False, complete=False,
                             errors=[{"type": "ParkIncomplete", "message": "move_joints returned false"}])
        except Exception as e:  # a failed park must not block teardown
            print(f"[cascade] park skipped ({type(e).__name__}: {e})")
            stage.update(ok=False, complete=False, errors=[{"type": type(e).__name__, "message": str(e)}])
    return teardown_receipt(stages)


def owned_threads(runtime) -> list:
    """Every worker thread a built runtime owns, as (label, Thread) pairs.

    shutdown_runtime stops each of these and must not return while one is
    still alive: a daemon thread still inside native code (torch/CUDA, cv2,
    a bridge socket) at interpreter exit aborts the process. That was the
    launcher's intermittent "terminate called without an active exception".
    """
    import threading

    out = []

    def add(label, owner):
        thread = getattr(owner, "_thread", None) if owner is not None else None
        if isinstance(thread, threading.Thread):
            out.append((label, thread))

    add("watcher", getattr(runtime, "watcher", None))
    for mon in getattr(runtime, "drift_monitors", None) or []:
        add(f"drift-monitor:{mon.name}", mon)
    add("stream-server", getattr(runtime, "stream_server", None))
    add("viewer", getattr(runtime, "viewer", None))
    rig = getattr(runtime, "rig", None)
    for stream in (list(rig) if rig is not None else []):
        add(f"stream-{getattr(stream, 'name', '?')}", stream)
    # The brain detector's SAM warm-up and background refreshes (LockedDetector
    # wraps the detector): joined by the "detector" stage, listed here in case
    # a bound expires.
    detector = getattr(runtime, "detector", None)
    inner = getattr(detector, "_detector", detector)
    warm_up = getattr(inner, "warm_up_thread", None)
    if isinstance(warm_up, threading.Thread):
        out.append(("sam-warm-up", warm_up))
    if callable(getattr(type(inner), "refresh_threads", None)):  # class only: a Mock answers anything
        out.extend(("brain-refresh", t) for t in inner.refresh_threads() if isinstance(t, threading.Thread))
    return out


def shutdown_runtime(runtime, arm) -> dict:
    """Close in dependency order and return every observed software result.

    `arm` is the primary raw backend, kept as a positional for the many call
    sites that predate the arm rig. When a rig is present EVERY arm is
    disconnected through it -- disconnecting only the primary would leave a
    second arm powered (torque on, unsupervised) after teardown reported
    success.

    Order: park (needs the watcher's heartbeat) -> save beliefs -> drift
    monitors (they read the watcher's streams and the arm's joints) ->
    watcher (the consumer of the camera streams) -> dashboard -> viewer ->
    camera streams (producers) -> arms -> detector (only one that defines
    close(), i.e. the brain detector: it starts no further refresh and joins
    those in flight and its SAM warm-up, each with a bound; a warm-up past it
    is listed in pending_threads). The detector goes last: its threads touch
    neither cameras nor arms, and disconnecting the arms must never wait
    behind a brain request. Each thread owner WAITS for its thread's
    in-flight work; a bounded join here once let the watcher's first cold
    CUDA detector inference (> 5 s) outlive shutdown and abort the launcher's
    runtime check at interpreter exit.
    """
    import copy
    from ..lifecycle import retain_teardown_attempt, teardown_receipt, teardown_step

    previous = getattr(runtime, "_shutdown_receipt", None)
    delegated = getattr(runtime, "robot_mode", None) in {"mobile", "composed"}
    if isinstance(previous, dict) and (not delegated or previous.get("complete") is True):
        return copy.deepcopy(previous)
    if delegated:
        # Delegated owners track unfinished IO and already-closed domains. Let
        # them finish cleanup after a pending call; never repeat legacy parking.
        receipt = retain_teardown_attempt(previous,
            teardown_receipt([teardown_step("runtime", runtime.close)]))
        runtime._shutdown_receipt = copy.deepcopy(receipt)
        return receipt

    threads = owned_threads(runtime)

    def _save_beliefs():
        # Persist the world model FIRST: it is the only step whose input the
        # later steps destroy, and a failure here is retained without skipping
        # the remaining teardown stages.
        path = getattr(runtime, "beliefs_path", None)
        if path is None:
            return
        n = runtime.beliefs.save(path)
        print(f"[cascade] remembered {n} object(s) -> {path}")

    def _save_episodic():
        # B43 (opt-in): the visual index, right after the world model and
        # before the watcher (its only other writer) stops; atomic either way.
        path = runtime.episodic_path
        n = runtime.memory.save_visual(path)
        print(f"[cascade] remembered {n} appearance(s) -> {path}")

    def _disconnect_arms():
        rig = getattr(runtime, "arm_rig", None)
        if rig is not None:
            return rig.disconnect()  # includes the primary and per-arm results
        return arm.disconnect()

    stages = []
    for name, step in (
        ("park", lambda: _park_arm(runtime)),
        ("beliefs", _save_beliefs),
        # present only when the visual index is persisted, so the default
        # receipt keeps exactly its stages
        *((("episodic", _save_episodic),) if getattr(runtime, "episodic_path", None) is not None
          else ()),
        *_owner_stages(runtime, arms=_disconnect_arms),
    ):
        stages.append(teardown_step(name, step))

    receipt = teardown_receipt(stages, pending_threads=_pending_threads(threads, after="shutdown"))
    runtime._shutdown_receipt = copy.deepcopy(receipt)
    return receipt


def _still_running(owner) -> bool:
    thread = getattr(owner, "_thread", None) if owner is not None else None
    return isinstance(thread, threading.Thread) and thread.is_alive()


def _owner_stages(runtime, *, arms=None, live_view=None, timeout_s: float | None = None) -> list:
    """The thread/device owners' stages of a teardown, as (name, step) pairs.

    Shared by shutdown_runtime (after park/beliefs/episodic, `arms` = every
    arm's disconnect, no bound) and a failed build's _abandon_build (`arms` =
    only passive ROS feedback attachments, `live_view` = the build's own
    LiveViewController, closed instead of `runtime.stream_server`, and the
    owners' joins bounded by `timeout_s`).
    Order: drift monitors (they read the watcher's streams and the arm's
    joints) -> watcher (the consumer of the camera streams) -> dashboard ->
    viewer -> camera streams (producers) -> arms -> detector. Stages for the
    opt-in drift monitors and a closable detector are present only when they
    exist, so the default receipt keeps exactly its stages.
    """
    def bounded(stop):
        return stop() if timeout_s is None else stop(timeout_s=timeout_s)

    def _stop_drift_monitors():
        for mon in getattr(runtime, "drift_monitors", None) or []:
            bounded(mon.stop)

    def _stop_stream_server():
        if live_view is not None:
            # close() stops the server the controller holds, including one
            # that bound before open() raised (its own join: up to 5 s).
            closed = live_view.close(reason="failed build")
            if not closed.get("ok"):
                raise RuntimeError(closed.get("error"))
            return None
        server = getattr(runtime, "stream_server", None)
        return server.stop() if server else None

    def _close_cameras():
        rig = getattr(runtime, "rig", None)
        if rig:
            return bounded(rig.close)
        camera = getattr(runtime, "camera", None)
        return camera.close() if camera is not None else None

    detector = getattr(runtime, "detector", None)

    def _close_detector():
        # A failed build's BOUNDED teardown can get here with the watcher
        # still running (its bound expired inside a detector call), and
        # LockedDetector.close() would wait for that call under the detector
        # lock: leave it open. shutdown_runtime has no bound and closes the
        # detector whatever the watcher stage reported, as it always did.
        if timeout_s is not None and _still_running(getattr(runtime, "watcher", None)):
            raise RuntimeError("the watcher is still inside the detector; detector left open")
        # Logged here, recorded in the receipt by teardown_step; never raised.
        try:
            detector.close()
        except Exception as e:
            logger.warning("closing the detector failed (%s: %s); shutdown goes on",
                           type(e).__name__, e)
            raise

    return [
        # likewise only with a drift monitor (opt-in, extrinsics.drift_monitor):
        # they read the watcher's streams and the arm's joints, so stop first
        *((("drift_monitors", _stop_drift_monitors),)
          if getattr(runtime, "drift_monitors", None) else ()),
        ("watcher", lambda: bounded(runtime.watcher.stop) if runtime.watcher is not None else None),
        ("stream_server", _stop_stream_server),
        ("viewer", lambda: bounded(runtime.viewer.stop) if getattr(runtime, "viewer", None) else None),
        ("cameras", _close_cameras),
        *((("arms", arms),) if arms is not None else ()),
        # likewise only for a detector with a close() (the brain detector):
        # after the watcher (no new background refresh starts) and LAST, so
        # the arms' disconnect never waits behind a brain request
        *((("detector", _close_detector),) if getattr(detector, "closable", False) is True else ()),
    ]


def _pending_threads(threads, *, after: str) -> list:
    """Owned threads still alive once their owners' stages ran, each named."""
    pending = []
    for label, thread in threads:
        if thread.is_alive():
            pending.append({"owner": label, "thread": thread.name})
            # The owners above wait without a bound, except the brain
            # detector (bounded joins of its refreshes and SAM warm-up) and a
            # failed build's teardown: this fires when such a bound expired,
            # or an owner regressed to one. Name it: the alternative is an
            # unexplained abort at interpreter exit.
            print(f"[cascade] WARNING: {label} thread {thread.name!r} is still running "
                  f"after {after}; native code may abort the process at exit",
                  file=sys.stderr)
    return pending


def _abandon_build(started) -> None:
    """Stop what a failed build_runtime had started. Never raises.

    `started` holds only owners this build started, recorded as each one
    started. They go through the same owner stages as shutdown_runtime, then
    this build's occupancy bridge client is closed (only the socket: the
    bridge and its map are not ours). The arms are left as they are, except
    the passive ROS feedback attachments, which are detached: no park can run
    here, and cutting an arm's torque without one is not a failed start's
    decision. A thread still alive past its bound is named; a stage that
    failed is logged.

    The bound is per owner, not for the whole teardown: worst case, the
    build's error waits _FAILED_BUILD_JOIN_S (15 s) for EACH of the watcher,
    the viewer, every drift monitor and every camera stream, up to 5 s for
    the live view's own server join, up to 3 s per ROS feedback arm detach,
    plus the brain detector's own close() bounds (refresh timeout + margin,
    and up to 60 s of SAM warm-up).
    """
    from ..lifecycle import teardown_receipt, teardown_step

    try:
        live_view = started.live_view
        # For owned_threads' naming only; live_view.close() stops it.
        started.stream_server = live_view.server if live_view is not None else None
        threads = owned_threads(started)
        feedback = list(started.feedback_arms)

        def _detach_feedback_arms():
            errors = []
            for raw in feedback:  # passive ROS attachment: no active goals
                try:
                    raw.disconnect()
                except Exception as e:  # noqa: BLE001 - the next arm still detaches
                    errors.append(e)
            if errors:
                raise errors[0]

        def _close_occupancy_client():
            # The watcher's refresh is its only other user and would reconnect.
            if _still_running(started.watcher):
                raise RuntimeError("the watcher is still running; occupancy client left open")
            started.occupancy.close_client()

        steps = _owner_stages(started, arms=_detach_feedback_arms if feedback else None,
                              live_view=live_view, timeout_s=_FAILED_BUILD_JOIN_S)
        if started.occupancy is not None:
            steps.append(("occupancy_client", _close_occupancy_client))
        stages = [teardown_step(name, step) for name, step in steps]
        receipt = teardown_receipt(
            stages, pending_threads=_pending_threads(threads, after="the failed build's teardown"))
    except Exception as e:  # noqa: BLE001 - the build's own error is the one re-raised
        logger.warning("tearing down a failed build failed (%s: %s)", type(e).__name__, e)
        return
    if not receipt["ok"]:
        failed = [s["stage"] for s in receipt["stages"] if not s.get("ok")]
        logger.warning("a failed build could not stop everything it started: stages %s, "
                       "threads still running %s", failed, receipt["pending_threads"])


def _empty_cfg():
    from ..config import Cfg

    return Cfg({})


def _beliefs_persist_enabled(mcfg) -> bool:
    """`memory.persist_beliefs`, with CASCADE_BELIEFS as the override.

    Same shape as the other kill switches in this file (CASCADE_STREAM): an
    env var wins over the config so a booth machine can be pinned from the
    launcher without editing YAML.
    """
    env = os.environ.get("CASCADE_BELIEFS", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return bool(mcfg.get("persist_beliefs", True))


def _flag(value) -> bool:
    """A YAML/env boolean: the string "false" is false, not a truthy string."""
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _episodic_persist_enabled(mcfg) -> bool:
    """`memory.persist_episodic` (B43, default false), with CASCADE_EPISODIC
    as the override -- the same shape as `_beliefs_persist_enabled`."""
    env = os.environ.get("CASCADE_EPISODIC", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    return _flag(mcfg.get("persist_episodic", False))


def _episodic_path(mcfg) -> Path:
    """Where the visual index persists: CASCADE_EPISODIC_PATH, else
    `memory.episodic_path`, else <repo>/runs/episodic.json."""
    return Path(os.environ.get("CASCADE_EPISODIC_PATH")
                or str(mcfg.get("episodic_path") or (PACKAGE_ROOT / "runs" / "episodic.json")))


def _memory_number(mcfg, key: str, default, *, minimum: float, integer: bool = False):
    """A validated `memory.<key>`: finite, >= `minimum`, an integer when asked.
    Anything else is a ValueError naming the key (fail closed at build)."""
    raw = mcfg.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if (isinstance(raw, bool) or not math.isfinite(value) or value < minimum
            or (integer and value != int(value))):
        kind = "an integer" if integer else "a finite number"
        raise ValueError(f"memory.{key} must be {kind} >= {minimum:g}, got {raw!r}")
    return int(value) if integer else value


def _visual_recall_settings(mcfg, embedder) -> dict:
    """B43's two opt-in switches as validated settings.

    ``detections`` (memory.visual_recall_detections): the recorder's limits.
    ``persist`` (memory.persist_episodic / CASCADE_EPISODIC): path + max age.
    Each is None when off. A switch without a memory embedder has no index
    to feed or persist: it is reported and ignored, never a crash -- memory
    is advisory and must not block startup. An invalid limit raises."""
    out: dict = {"detections": None, "persist": None}
    if _flag(mcfg.get("visual_recall_detections", False)):
        limits = {
            "interval_s": _memory_number(mcfg, "visual_recall_interval_s", 30.0, minimum=0.0),
            "max_per_tick": _memory_number(mcfg, "visual_recall_max_per_tick", 2,
                                           minimum=1, integer=True),
            "max_detections": _memory_number(mcfg, "visual_recall_max_detections", 128,
                                             minimum=1, integer=True),
        }
        if embedder is None:
            print("[cascade] memory.visual_recall_detections needs memory.embedder "
                  "(backend: hash | siglip | clip); watcher crops stay off", file=sys.stderr)
        else:
            out["detections"] = limits
    if _episodic_persist_enabled(mcfg):
        max_age = _memory_number(mcfg, "episodic_max_age_s", EpisodicMemory.DEFAULT_MAX_AGE_S,
                                 minimum=0.0)
        if embedder is None:
            print("[cascade] memory.persist_episodic needs memory.embedder "
                  "(backend: hash | siglip | clip); the visual index is not persisted",
                  file=sys.stderr)
        else:
            out["persist"] = {"path": _episodic_path(mcfg), "max_age_s": max_age}
    return out


def _detection_recorder(memory, limits: dict):
    from ..memory.episodic import DetectionCropRecorder

    return DetectionCropRecorder(memory, interval_s=limits["interval_s"],
                                 max_per_tick=limits["max_per_tick"])


def _instance_association_enabled(mcfg) -> bool:
    """`memory.instance_association` (default true). A YAML string such as
    "false" is honoured as false rather than read as a truthy string."""
    value = mcfg.get("instance_association", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _belief_store(mcfg) -> BeliefStore:
    """The world model as the `memory:` block configures it.

    `per_camera_colour` (default true, B32b): a belief keeps each camera's
    colour name, so one object that two cameras name across a band boundary
    (the Isaac bin: "orange" top, "yellow" side) is one belief; false = the
    one-name rule. `neighbour_colour_iou` (default 0.75) is the 3D box IoU a
    camera that never named a belief needs to fuse a neighbouring name into
    it; outside (0, 1] raises. `size_gate` (default true, B40): a view much
    larger than any view a belief has had is not that object (a camera that
    names a container and the prop inside it alike must not fuse the
    container into the prop); false = the store before it, byte for byte. A
    YAML string such as "false" is honoured.
    """
    value = mcfg.get("per_camera_colour", True)
    if isinstance(value, str):
        value = value.strip().lower() not in ("0", "false", "no", "off")
    gate = mcfg.get("size_gate", True)
    if isinstance(gate, str):
        gate = gate.strip().lower() not in ("0", "false", "no", "off")
    iou = mcfg.get("neighbour_colour_iou")
    return BeliefStore(
        instance_association=_instance_association_enabled(mcfg),
        per_camera_colour=bool(value),
        neighbour_colour_iou=(NEIGHBOUR_COLOUR_IOU if iou is None else float(iou)),
        size_gate=bool(gate),
    )


def _premotion_critic(cfg, llm, runtime, is_mock: bool):
    """ROADMAP follow-up #6: the Human-CLAW pre-motion plausibility critic.

    ADVISORY ONLY (agent/milestones.py): the orchestrator asks it before a
    motion skill is dispatched and shows its answer to the planner; it never
    refuses or rewrites a call -- the safety harness is the sole authority
    that refuses motion. `agent.premotion_check: false` (or
    CASCADE_PREMOTION_CHECK=0, same kill-switch shape as CASCADE_BELIEFS)
    returns None, which is the pre-critic orchestrator path exactly.

    Without a vision-capable brain the critic is still constructed so every
    motion result records *why* it was not judged (`skipped`), rather than
    silently looking identical to a judged one. The mock brain is a labelled
    script, not a judge: asking it would consume the script.
    """
    agent_cfg = cfg.get("agent", {}) or {}
    env = os.environ.get("CASCADE_PREMOTION_CHECK", "").strip().lower()
    if env:
        enabled = env not in ("0", "false", "no", "off")
    else:
        raw = agent_cfg.get("premotion_check", True)
        enabled = (raw.strip().lower() in ("1", "true", "yes", "on")
                   if isinstance(raw, str) else bool(raw))
    if not enabled:
        return None
    from ..agent.milestones import PlausibilityChecker, make_plausibility_verifier

    verifier, skip_reason = None, None
    if is_mock:
        skip_reason = "mock brain is a labelled script, not a vision model"
    elif not getattr(llm, "supports_vision", False):
        skip_reason = "no vision-capable model configured"
    else:
        verifier = make_plausibility_verifier(llm)
    # `Cfg` and dict both answer .get("min"/"max"), which is all the digest reads.
    workspace = cfg.safety.get("workspace") if "safety" in cfg else None
    template = None
    from ..robot_context import declares_base_pose

    if declares_base_pose(cfg):
        # base_pose spec §3.5 (R11): the beliefs are table-frame; the
        # workspace box is one arm's, in its own base frame, so it is not
        # quoted beside them.
        from ..agent.prompts import PLAUSIBILITY_USER_TABLE_FRAME

        template, workspace = PLAUSIBILITY_USER_TABLE_FRAME, None
    return PlausibilityChecker(
        verifier,
        beliefs=getattr(runtime, "beliefs", None),
        held_getter=lambda: getattr(runtime, "held_object", None),
        max_checks=int(agent_cfg.get("premotion_max_checks", 3)),
        workspace=workspace,
        skip_reason=skip_reason,
        prompt_template=template,
    )


def _programs_enabled(cfg) -> bool:
    """The programs-tier switch, ONE for every front-end (the CLI's
    orchestrator and, since B42, the MCP server): `CASCADE_PROGRAMS` (1/0)
    beats `agent.programs` (default false)."""
    env = os.environ.get("CASCADE_PROGRAMS", "").strip().lower()
    if env:
        return env not in ("0", "false", "no", "off")
    agent_cfg = (cfg.get("agent", {}) if cfg is not None else {}) or {}
    raw = agent_cfg.get("programs", False)
    return (raw.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(raw, str) else bool(raw))


def _program_library(cfg):
    """The program store: `runs/programs.jsonl` (`memory.programs_path`,
    CASCADE_PROGRAMS_PATH); reuse after `agent.program_min_tasks` (default 2)
    distinct verified tasks. Raises when the store cannot be opened."""
    from ..memory.programs import ProgramLibrary
    from ..skills.library import PROMOTION_MIN_TASKS

    agent_cfg = cfg.get("agent", {}) or {}
    mem_cfg = cfg.get("memory", {}) or {}
    path = (os.environ.get("CASCADE_PROGRAMS_PATH") or mem_cfg.get("programs_path")
            or PACKAGE_ROOT / "runs" / "programs.jsonl")
    return ProgramLibrary(Path(str(path)).expanduser(),
                          min_tasks=int(agent_cfg.get("program_min_tasks", PROMOTION_MIN_TASKS)))


def _agent_harness(cfg):
    """The learned agent harness named by `agent.harness_path` or CASCADE_AGENT_HARNESS, or None.

    A named file that is missing or fails validation stops the start: an operator who asked for a
    harness must not silently run without it.
    """
    agent_cfg = (cfg.get("agent", {}) if cfg is not None else {}) or {}
    path = os.environ.get("CASCADE_AGENT_HARNESS", "").strip() or agent_cfg.get("harness_path")
    if not path:
        return None
    from ..harnessing import AgentHarness

    harness = AgentHarness.read(Path(str(path)).expanduser())
    print(f"[cascade] agent harness {harness.sha()[:12]} from {path}", flush=True)
    return harness


def _program_tier(cfg, is_mock: bool):
    """ROADMAP follow-up #8: the opt-in programs tier (docs/PROGRAMS_TIER.md).

    OFF by default: `agent.programs: false` (or CASCADE_PROGRAMS=0, the same
    kill-switch shape as CASCADE_PREMOTION_CHECK) returns None, which is the
    pre-change orchestrator path exactly. When on, a task no reflex/habit plan
    covers gets one authoring turn and the program runs step by step through
    SkillRuntime.execute(); the harness stays the sole authority that refuses
    motion. The mock brain is a labelled script, not an author: asking it
    would consume the script, so the tier is never enabled for it.

    Store: `runs/programs.jsonl` (`memory.programs_path`,
    CASCADE_PROGRAMS_PATH); a program is offered for reuse only after it was
    verified in >= `agent.program_min_tasks` (default 2) distinct tasks.
    """
    if not _programs_enabled(cfg):
        return None
    if is_mock:
        print("[cascade] programs tier requested, but the mock brain is a labelled script, "
              "not an author: tier left off")
        return None
    from ..agent.programs import ProgramTier

    return ProgramTier(_program_library(cfg))


#: `detector.type: brain` defaults, also used for keys the block omits:
#: the local vision LLM on :8080 (the PORT every scripts/serve_*_vllm.sh /
#: _llamacpp.sh brain starts on, = configs/demo.yaml's default), the default
#: brain's served id (Nemotron 3.5 Bolt) and the key file the launcher writes.
BRAIN_BASE_URL = "http://127.0.0.1:8080/v1"
BRAIN_MODEL = "nemotron-3.5-bolt"
BRAIN_KEY_FILE = "~/.cascade/nemoclaw/llama.key"

#: Blank frame of the SAM warm-up (shape only; the result is discarded).
_WARM_UP_SHAPE = (240, 320)


#: build_runtime waits at most this long, in total, for the first frame of
#: each watched camera that has none yet before comparing their cache keys.
_CAMERA_KEY_WAIT_S = 2.0


def _warn_shared_camera_keys(detector, streams, *, wait_s: float = _CAMERA_KEY_WAIT_S,
                             skip=()) -> list[list[str]]:
    """One WARNING when two watched cameras' first frames give a detector
    that keys its cache per camera (the brain detector's CLASS defines
    `camera_key`) the same key: they would share one cache entry, and each
    camera's tick would invalidate the other's inventory. Frames without a
    capture identity key by image size + K only (two identical RealSense
    models). Returns the groups of camera names that share a key.

    Only cameras the watcher asks the detector about are compared: `skip`
    names the others (non-fusing and not the primary, world.WorldWatcher B4).
    Every compared camera's key, and every camera left out (skipped, no frame
    within `wait_s` in total, or a frame or key that fails, e.g. a NaN K), is
    logged at INFO. A diagnostic: it never raises into build_runtime."""
    try:
        inner = getattr(detector, "_detector", detector)
        camera_key = getattr(type(inner), "camera_key", None)
        if not callable(camera_key) or len(streams) < 2:
            return []
        deadline = time.monotonic() + wait_s
        names_by_key: dict = {}
        for stream in streams:
            name = str(getattr(stream, "name", stream))
            if name in skip:
                logger.info("camera %s: left out of the brain-detector cache key check (no background "
                            "inventory: it does not fuse and is not the primary camera)", name)
                continue
            try:
                frame = stream.latest()
                if frame is None:
                    frame, _ = stream.wait_newer(0, timeout_s=max(0.0, deadline - time.monotonic()))
                if frame is None:
                    logger.info("camera %s: left out of the brain-detector cache key check (no frame within "
                                "%.1f s)", name, wait_s)
                    continue
                key = camera_key(frame)
                names_by_key.setdefault(key, []).append(name)
            except Exception as e:  # noqa: BLE001 - a diagnostic never fails the build
                logger.info("camera %s: left out of the brain-detector cache key check (%s: %s)",
                            name, type(e).__name__, e)
                continue
            logger.info("camera %s: brain-detector cache key %s", name, key)
        shared = [names for names in names_by_key.values() if len(names) > 1]
        if shared:
            logger.warning(
                "cameras %s share one brain-detector cache key (same image size and intrinsics, no capture "
                "identity): their inventories invalidate each other on every watcher tick",
                "; ".join(" and ".join(names) for names in shared))
        return shared
    except Exception as e:  # noqa: BLE001 - a diagnostic never fails the build
        logger.warning("brain-detector cache key check skipped (%s: %s)", type(e).__name__, e)
        return []


def _make_detector(cfg, *, warm_up: bool = True):
    """The detector the config names: `type: brain` (what configs/demo.yaml
    ships), `mock`, or the deprecated `vlm`. A block without a `type` is
    refused. `type: brain` with a segmenter also starts the SAM warm-up
    thread unless `warm_up` is False (tests); the detector owns that thread
    (its close() joins it)."""
    # A camera profile may pin its own detector (the mock camera uses the
    # mock detector so offline runs never load model weights).
    dcfg = cfg.camera.get("detector") or cfg.detector
    dtype = str(dcfg.get("type", ""))
    if dtype == "mock":
        from ..perception.detector import MockDetector

        # extra_props on the camera profile (multi-prop scenes for the
        # memory tasks) become extra colour-keyed labels, so the mock
        # detector finds exactly the props the scene writer painted.
        extra = [
            str(e.get("label") or f"{e.get('color', 'blue')} cube")
            for e in (cfg.camera.get("extra_props") or [])
        ]
        return MockDetector(label=dcfg.get("label", "red cube"), extra_labels=extra,
                            instances=bool(dcfg.get("instances", False)))
    if dtype == "brain":
        return _make_brain_detector(cfg, dcfg, warm_up=warm_up)
    if dtype == "vlm":
        # Deprecated alias: the BrainDetector shim with the old cosmos3-edge
        # defaults and no segmenter (box-only detections).
        from ..perception.vlm_detector import VLMDetector

        return VLMDetector(
            base_url=str(dcfg.get("base_url", "http://127.0.0.1:8082/v1")),
            model=str(dcfg.get("model", "cosmos3-edge")),
            api_key=str(dcfg.get("api_key", "EMPTY")),
            timeout_s=float(dcfg.get("timeout_s", 20.0)),
            conf=float(dcfg.get("conf", 0.25)),
        )
    if dtype == "yoloe":
        raise ValueError(
            "detector `type: yoloe` was removed: use `type: brain` (the vision LLM names and boxes "
            "the objects, SAM masks them; see configs/demo.yaml), or `type: mock` offline")
    raise ValueError(f"unknown detector type {dtype!r}: use `type: brain`, `type: mock` or `type: vlm`")


def _make_brain_detector(cfg, dcfg, *, warm_up: bool):
    """BrainDetector from the detector block, with a BoxSegmenter built from
    the same `segmenter:` strings as the click PointSegmenter (skills/runtime),
    so both share one loaded SAM. No segmenter when `segmenter.enabled` is
    false or absent: box-only detections."""
    from ..perception.brain_detector import BrainDetector

    segmenter = None
    scfg = cfg.get("segmenter") or {}
    if scfg.get("enabled"):
        from ..perception.segmenter import BoxSegmenter, configured_segmenter

        model_path, device = configured_segmenter(scfg)
        segmenter = BoxSegmenter(model_path=model_path, device=device)
    key_file = dcfg.get("api_key_file", BRAIN_KEY_FILE)
    timeout_s = float(dcfg.get("timeout_s", 20.0))
    # Absent: the detector's default, idle_refresh_s + 5 s.
    max_cache_age_s = dcfg.get("max_cache_age_s")
    detector = BrainDetector(
        base_url=str(dcfg.get("base_url", BRAIN_BASE_URL)),
        model=str(dcfg.get("model", BRAIN_MODEL)),
        # The key is read from this file, never from the config text.
        api_key_file=os.path.expanduser(str(key_file)) if key_file else None,
        timeout_s=timeout_s,
        # Background (watcher) requests: an open-world inventory decodes far
        # more tokens than a refer. Absent: the foreground timeout_s.
        inventory_timeout_s=float(dcfg.get("inventory_timeout_s", timeout_s)),
        max_tokens=int(dcfg.get("max_tokens", 1500)),
        segmenter=segmenter,
        refresh_s=float(dcfg.get("refresh_s", 4.0)),
        max_cache_age_s=None if max_cache_age_s is None else float(max_cache_age_s),
        scene_change=float(dcfg.get("scene_change", 6.0)),
        idle_refresh_s=float(dcfg.get("idle_refresh_s", 30.0)),
        reanchor_local_diff=float(dcfg.get("reanchor_local_diff", 8.0)),
        reanchor_mask_agreement=float(dcfg.get("reanchor_mask_agreement", 0.95)),
        reply_guards=_reply_guards(dcfg.get("reply_guards", "auto")),
    )
    if warm_up and segmenter is not None:
        # Handed to the detector: its close() (a shutdown stage) joins it and
        # owned_threads names it, so SAM's native load/predict is never still
        # running at interpreter exit.
        detector.warm_up_thread = threading.Thread(target=_warm_box_segmenter, args=(segmenter,),
                                                   name="sam-warm-up", daemon=True)
        detector.warm_up_thread.start()
    return detector


def _reply_guards(value) -> bool | None:
    """`detector.reply_guards`: auto (absent) = by the brain's served id
    (brain_detector.GUARDED_REPLY_MODELS: Nemotron 3.5 Bolt), or true / false."""
    if value is None or value == "auto":
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"detector.reply_guards must be auto, true or false (got {value!r})")


def _warm_box_segmenter(segmenter) -> None:
    """Load SAM and run one box predict on a blank frame, so the watcher's
    first track step does not pay the cold start (weights + first CUDA
    predict) under the shared model lock. Best effort: one warning on
    failure, never raises; the first real detection then loads it."""
    try:
        h, w = _WARM_UP_SHAPE
        frame = Frame(rgb=np.zeros((h, w, 3), np.uint8), depth_m=None,
                      K=np.array([[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]]))
        segmenter.masks(frame, np.array([[w / 4, h / 4, 3 * w / 4, 3 * h / 4]], np.float32))
    except Exception as e:  # noqa: BLE001 - a warm-up must never fail startup
        logger.warning("SAM warm-up failed; the first detection loads it instead (%s: %s)",
                       type(e).__name__, e)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CASCADE agentic grasping demo")
    p.add_argument("--task", default=None, help="natural-language task")
    p.add_argument("--camera", default="mock", help="camera profile (mock|l515|d435i|uvc)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera profiles; first = manipulation "
                        "camera (e.g. l515,uvc). Overrides --camera.")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs|so101)")
    p.add_argument("--arms", default=None,
                   help="comma-separated arm profiles; first = manipulation "
                        "arm (e.g. so101_mock,so101_mock). Overrides --arm.")
    p.add_argument("--base", default=None, help="opt-in base-only profile (or CASCADE_BASE)")
    p.add_argument("--robot", default=None, help="explicit robot composition profile (or CASCADE_ROBOT)")
    p.add_argument("--bases", default=None, help="ordered comma-separated distinct base profiles")
    p.add_argument("--llm", default="auto",
                   help="llm profile, or `auto` (default): Hermes/Nous Portal, "
                        "then Anthropic, then OpenAI, whichever has its key in "
                        "the environment, else mock. Name one explicitly to "
                        "pin it (mock|hermes|anthropic|openai|local_nemotron_bolt|"
                        "local_qwen|local_cosmos)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None, help="trace output dir")
    p.add_argument("--interactive", action="store_true", help="multi-task REPL")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--no-serve", action="store_true",
                   help="do not start the MJPEG livestream dashboard")
    args = p.parse_args(argv)

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    llm = resolve_llm_profile(args.llm)
    bases = args.bases.split(",") if args.bases is not None else None
    cfg = load_demo_config(camera=args.camera, cameras=cameras, arm=args.arm,
                           arms=arms, llm=llm, base=args.base, bases=bases, robot=args.robot)
    composed = cfg.get("robot_mode") == "composed"
    mobile = cfg.get("robot_mode") in {"mobile", "composed"}
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    if composed:
        print(f"[cascade] composed robot={cfg.robot_id} llm={llm}; declared capabilities are not physical admission")
    elif mobile:
        print(f"[cascade] base-only bases={[b['name'] for b in cfg.bases]} llm={llm}; "
              "physical admission pending (mock is kinematic only)")
    else:
        print(f"[cascade] cameras={cameras or [args.camera]} "
              f"arms={arms or [args.arm]} "
              f"llm={llm}{' (auto)' if llm != args.llm else ''} view={view}")
    print(f"[cascade] traces -> {run_dir}")

    from .signal_stop import SignalRequest, StopSignals
    import signal

    runtime = arm = None
    status = 0
    with StopSignals() as signals:
        try:
            # Construction is passive (no command dispatch). Publish ownership
            # before acting on a first signal so cleanup cannot lose resources.
            with signals.defer():
                runtime, arm = build_runtime(cfg, run_dir, view=view, serve=not args.no_serve)
            if os.environ.get("CASCADE_START_STOPPED") == "1" and not mobile and not composed:
                runtime.arm.stop()
            signals.checkpoint()
            status = _run_demo(args, cfg, runtime, mobile)
        except SignalRequest:
            # Interrupted with-blocks have released their locks. Invalidate
            # before any logging/teardown. Torque-off backends retain their
            # halt -> park policy. Controller-owned arms stop/hold immediately
            # and disconnect without introducing a new parking motion.
            if runtime is not None:
                if composed:
                    runtime.request_shutdown()
                elif mobile:
                    runtime.stop()
                else:
                    rig = getattr(runtime, "arm_rig", None)
                    for safe_arm in (list(rig) if rig is not None else [runtime.arm]):
                        try:
                            if _shutdown_park_required(safe_arm.raw):
                                safe_arm.harness.halt("signal: graceful shutdown")
                            else:
                                safe_arm.stop()
                        except Exception:
                            pass
        finally:
            with signals.defer():
                if runtime is not None:
                    shutdown_runtime(runtime, arm)
        if signals.signum is not None:
            return 0 if not mobile and signals.signum == signal.SIGINT else 128 + signals.signum
    return status


def _run_demo(args, cfg, runtime, mobile):
    """Run only after signal handling and runtime ownership are established."""
    if runtime.stream_server is not None:
        print(f"[cascade] LIVESTREAM dashboard: {runtime.stream_server.url}")
    from ..agent.llm import MockLLM, LLMResponse, ToolCall

    llm = make_llm(cfg.llm)
    is_mock = isinstance(llm, MockLLM)
    if is_mock and cfg.get("robot_mode") == "composed":
        # The legacy mock script names an arm-only get_observation. The new
        # mode's offline check is explicitly a passive catalog check, with
        # no implied sensor capture or physical task completion.
        llm = MockLLM([
            LLMResponse(tool_calls=[ToolCall("list_resources", {})]),
            LLMResponse(tool_calls=[ToolCall("task_done", {"success": True,
                "summary": "mock wiring check: inspected declared resources; no physical task executed"})]),
        ])
    advisor = Advisor(llm, cfg=cfg) if (llm.supports_vision and not is_mock) else None
    experience = None if mobile else ExperienceMemory(Path(cfg.memory.get("experience_path", PACKAGE_ROOT / "runs" / "experience.json")).expanduser())
    # ROADMAP #7: action<->object consolidation of tier-2 outcomes, opt-in via
    # `memory.action_objects: true` (shipped off: the planner then carries
    # nothing new). Persisted beside experience.json; advisory digest only.
    action_objects = None
    _ao = cfg.memory.get("action_objects", False)
    if not mobile and (_ao is True or str(_ao).strip().lower() in ("1", "true", "yes", "on")):
        from ..memory.consolidation import ActionObjectMemory

        action_objects = ActionObjectMemory(PACKAGE_ROOT / "runs" / "action_objects.json")
    # ASPIRE: validated repairs distilled from earlier runs, retrieved into
    # context at task start. This is the loop the ROADMAP listed as open --
    # `scripts/learn_from_runs.py` writes the entries, the agent reads them.
    # The CLI reads them through the run's snapshot (spec §4.2): exactly the notes
    # `runtime.learned` serves (lint-clean, never refuted, the cross-task gate of
    # `memory.skill_min_tasks`), from the bytes it hashed, and only as the prompt
    # block renders them (title, advice, tag; never a note's body); the directory
    # is never created here.
    from ..skills import notes
    from ..skills.library import PROMOTION_MIN_TASKS

    library = None if mobile else notes.served_library(
        getattr(runtime, "learned", None), cfg.memory.get("skill_min_tasks", PROMOTION_MIN_TASKS))
    agent = AgentOrchestrator(
        llm, runtime, advisor=advisor, max_steps=args.max_steps,
        decompose=not is_mock,
        fast_planner=None if mobile else (
            FastPlanner(experience) if action_objects is None
            else FastPlanner(experience, action_objects=action_objects)),
        skill_library=library, verify_milestones=not is_mock,
        memory_frames_k=int(cfg.memory.get("frames_k", 4)),
        # ROADMAP #6 pre-motion critic: advisory only; arm runtimes only (the
        # orchestrator drops it for mobile/composed modes like the tracker).
        plausibility=None if mobile else _premotion_critic(cfg, llm, runtime, is_mock),
        # ROADMAP #8 programs tier: opt-in (agent.programs / CASCADE_PROGRAMS).
        programs=None if mobile else _program_tier(cfg, is_mock),
        # Harness² agent harness (cascade.harnessing): opt-in advice text learned by
        # scripts/harness2_improve.py (agent.harness_path / CASCADE_AGENT_HARNESS).
        harness=_agent_harness(cfg),
    )

    def _run(task: str, cancel_event=None):
        # Mobile/composed runtimes have no arm camera overlay. Rendering a
        # completion status is optional; it must not turn a valid task into
        # an AttributeError or mask its original failure.
        show_status = getattr(runtime, "_show_status", None)
        try:
            report = agent.run_task(task, cancel_event=cancel_event)
            runtime.memory.add("note", report.summary)
            if callable(show_status):
                show_status("cancelled" if report.cancelled else
                            ("idle" if report.success else f"failed: {report.summary}"))
            return report
        except TaskBusyError:
            raise
        except Exception as exc:
            runtime.memory.add("note", f"command failed: {exc}")
            if callable(show_status):
                show_status(f"failed: {exc}")
            raise

    # Wire the dashboard "send"/"stop" buttons to the live agent so you can
    # drive the robot from the browser (http://<ip>:8090) as well as the REPL.
    # The controller retains callbacks for a lazy initial open and every
    # reopen; wiring only its current HTTP server loses STOP and Resume.
    _srv = getattr(runtime, "live_view", None) or getattr(runtime, "stream_server", None)
    if _srv is not None:
        def _dashboard_task(task, cancel_event):
            report = _run(task, cancel_event)
            _print_report(report)
            return report

        _srv.set_task_fn(_dashboard_task, cancel_aware=True)
        _srv.set_cancel_fn(agent.cancel)
        _srv.set_resume_fn(agent.resume)

    if args.interactive and mobile:
        from .mobile_runtime import run_mobile_interactive

        print("Type a task; stop/emergency_stop are immediate; EOF or empty line cancels and exits.")
        run_mobile_interactive(runtime, _run, _print_report)
    elif args.interactive:
        print("Type a task (empty line to quit).")
        while True:
            try:
                task = input("task> ").strip()
            except EOFError:
                break
            if not task:
                break
            _print_report(_run(task))
    else:
        task = args.task or ("observe the mobile base" if mobile else
                             "look at the table and report what objects you see")
        report = _run(task)
        _print_report(report)
        return 0 if report.success else 1
    return 0


def _print_report(report) -> None:
    print("\n=== task report ===")
    print(f"task:    {report.task}")
    print(f"success: {report.success}")
    print(f"path:    {report.path} ({report.duration_s}s)")
    print(f"steps:   {report.steps}")
    print(f"summary: {report.summary}")


if __name__ == "__main__":
    sys.exit(main())
