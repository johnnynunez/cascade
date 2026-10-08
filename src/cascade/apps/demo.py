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
import os
import sys
import time
from pathlib import Path

from ..agent.advisor import Advisor
from ..agent.llm import make_llm, resolve_llm_profile
from ..agent.orchestrator import AgentOrchestrator
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
from ..perception.workspace import WorkspaceFilter
from ..safety.harness import SafeArm, SafetyHarness, SafetyLimits
from ..skills.runtime import SkillRuntime


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


def _build_arm(acfg, lazy_arm: bool, occupancy, fallback_cfg):
    """One arm: kinematics -> backend (maybe lazy) -> own harness -> SafeArm.

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
    """
    view = acfg.get("resolved") or fallback_cfg
    n_joints = int(acfg.get("n_joints", 6))
    kin = Kinematics(
        model_path=acfg.model,
        ee_frame=acfg.get("ee_frame", "gripper_end"),
        n_controlled=n_joints,
        joint_signs=acfg.get("joint_signs"),
        ik_task_weights=acfg.get("ik_task_weights"),
    )
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
                      profile_type=str(acfg.get("type", "")))
    else:
        arm = make_arm(acfg, kinematics=kin)
        arm.connect()
    harness = SafetyHarness(
        SafetyLimits.from_config(view.safety), kinematics=kin, occupancy=occupancy,
        base_pose=_base_transform(acfg),
    )
    # Optional profile-defined rest pose for the shutdown park (see
    # _park_pose). Defaults to the clamped all-zero fallback when absent.
    park_q = acfg.get("park_q")
    if park_q is not None:
        harness.park_q = list(park_q)
    if occupancy is not None:
        # The depth camera sees THIS arm: register it for body masking so its
        # own links are not integrated as obstacles. Never materialize a
        # LazyArm: real motor bring-up must remain an explicit user action.
        from ..perception.robot_mask import arm_link_points

        base_T = _base_transform(acfg)

        def _body_points(_arm=arm, _kin=kin, _T=base_T, _cfg=acfg, *, frame=None):
            if frame is not None and _cfg.get("type") == "isaac":
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
                q = _arm.get_state().q
            pts = arm_link_points(_kin, q)
            if _T is not None:
                pts = pts @ _T[:3, :3].T + _T[:3, 3]
            return pts

        frame_body = None
        if acfg.get("type") == "isaac":
            # Explicit backend binding. A legacy/malformed Isaac frame must
            # fail closed, not silently read today's q for yesterday's image.
            def frame_body(frame):
                if frame is None:
                    raise ValueError("Isaac capture snapshot required")
                return _body_points(frame=frame)

        occupancy.add_robot_body(_body_points, radius_m=float(acfg.get("body_mask_radius_m", 0.06)),
                                 frame_link_points_fn=frame_body)
        if fallback_cfg.get("occupancy", _empty_cfg()).get("track_payload", False):
            if acfg.get("type") != "isaac":
                raise ValueError("payload segmentation currently requires the Isaac capture contract")
            def frame_tcp_pose(frame, _cfg=acfg, _kin=kin, _T=base_T):
                from ..control.isaac_arm import IsaacArm
                pose = _kin.fk(IsaacArm(_cfg).state_from_frame(frame).q)
                return pose if _T is None else _T @ pose
            occupancy.track_payload(frame_tcp_pose)
    return arm, SafeArm(arm, harness, motion_planner=motion_planner), kin


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

    from ..planning.backend_evidence import selection_from_environment
    kitchen_renderer = selection_from_environment(cfg, os.environ)

    # ROADMAP #7: the optional memory embedder is resolved FIRST, before any
    # arm, camera or thread exists. A backend that was asked for but cannot
    # run (missing `memory-embed` extra, torch or cached weights) fails the
    # build here with EmbedderUnavailable instead of quietly running without
    # it or with another backend. `backend: none` (shipped) -> None.
    from ..memory.embedder import embedder_config, make_embedder

    memory_embedder = make_embedder(embedder_config(cfg))

    from ..perception.occupancy import OccupancyMap

    # ── the arm rig: N arms, first = manipulation arm ───────────────────
    arm_cfgs = _arm_cfgs(cfg)

    # One shared obstacle map, spanning every arm's workspace. With a single
    # arm this is exactly the old expression; with several, the union is the
    # honest default -- a region covering only the primary would leave the
    # second arm's half of the table unmapped, and unmapped reads as clear.
    # An explicit `occupancy.region_min/max` still wins (see from_config).
    ws_min = [a.get("resolved").safety.workspace.min if a.get("resolved")
              else cfg.safety.workspace.min for a in arm_cfgs]
    ws_max = [a.get("resolved").safety.workspace.max if a.get("resolved")
              else cfg.safety.workspace.max for a in arm_cfgs]
    occupancy = OccupancyMap.from_config(
        cfg.get("occupancy"),
        workspace_min=[min(v[i] for v in ws_min) for i in range(3)],
        workspace_max=[max(v[i] for v in ws_max) for i in range(3)],
    )

    raw_arms, safe_arms, arm_names = [], [], []
    for i, acfg in enumerate(arm_cfgs):
        raw, safe, k = _build_arm(acfg, lazy_arm, occupancy, cfg)
        raw_arms.append(raw)
        safe_arms.append(safe)
        arm_names.append(str(acfg.get("name", f"arm{i}")))
        if i == 0:
            kin = k
    arm_rig = ArmRig(safe_arms, arm_names)
    # Inter-arm proximity gating (no-op for a single arm, or when profiles
    # declare no base_pose -- see _wire_neighbors).
    _wire_neighbors(arm_rig, raw_arms)
    # The primary arm stays bound to the same names the single-arm code used,
    # so every existing call site (56 `self.arm` uses in the skill runtime,
    # shutdown_runtime, the truth-pose hook) is untouched by the rig.
    arm = raw_arms[0]
    safe_arm = arm_rig.primary
    harness = safe_arm.harness

    # ── the camera rig: N continuous streams, first = manipulation ──────
    cam_cfgs = _camera_cfgs(cfg)
    streams, watched = [], []
    detector = LockedDetector(_make_detector(cfg))

    def fk():
        # Eye-in-hand extrinsics need live FK -- but the 3 Hz watcher must
        # NEVER be the thing that materializes a LazyArm (that would power
        # the motors as a side effect of starting the gateway). The watcher
        # skips this camera until the arm is up.
        if not getattr(arm, "connected", True):
            from ..types import SkillError

            raise SkillError("eye-in-hand extrinsics need the arm (still in standby)")
        return kin.fk(arm.get_state().q)

    for i, ccfg in enumerate(cam_cfgs):
        stream = CameraStream(
            make_camera(ccfg),
            name=str(ccfg.get("name", f"cam{i}")),
            rate_hz=float(ccfg.get("fps", 30.0)),
        )
        streams.append(stream)
        extrinsics = Extrinsics.from_config(
            ccfg.get("extrinsics", _empty_cfg()), fk_tcp2base=fk,
            camera_serial=ccfg.get("serial"),
        )
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
    rig.open()
    try:
        rig.primary.warm_up(int(cam_cfgs[0].get("warmup_frames", 5)))
    except Exception:
        rig.close()  # a partial build must not leak open camera streams
        raise

    memory = EpisodicMemory(
        horizon_s=float(cfg.memory.get("horizon_s", 15.0)),
        frame_horizon_s=float(cfg.memory.get("frames_horizon_s", 600.0)),
        **({"embedder": memory_embedder} if memory_embedder is not None else {}),
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
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        rig.primary, watched[0].depth, detector, watched[0].extrinsics,
        kin, safe_arm, memory, beliefs, trace, cfg,
    )
    runtime.rig = rig
    runtime._kitchen_camera_renderer = kitchen_renderer
    # Where to persist the world model on shutdown (None = disabled).
    runtime.beliefs_path = beliefs_path
    # The arm rig hangs off the runtime the same way the camera rig does.
    # `runtime.arm` stays the primary SafeArm, so nothing that predates the
    # rig has to learn about it; a skill called with `arm="<name>"` is
    # rebound for that one call by SkillRuntime.execute().
    runtime.arm_rig = arm_rig

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
            harness=harness,
            workspace=WorkspaceFilter.from_config(cfg.get("workspace_filter")),
            occupancy=occupancy,
        )
        watcher.start()
        runtime.watcher = watcher

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
        runtime.viewer.start()

    # Verified-backend banner. GraspGen-X is probed HERE (300 ms) rather
    # than on the first grasp, so the operator sees "grasp_planner=obb
    # (graspgenx down)" before anything moves instead of an 8 s stall and a
    # buried memory note. Occupancy was probed when its map was built.
    _probe_grasp_backend(runtime)
    runtime.trace.backends_fn = runtime.backends
    b = runtime.backends()
    print(f"[cascade] backends: grasp_planner={b['grasp_planner']} | occupancy={b['occupancy']}")
    if not b["occupancy_live"] and occupancy is not None:
        print("[cascade] WARNING: occupancy is enabled in config but no bridge answered -- "
              "the clearance gate is OFF (start scripts/serve_occupancy.sh)", file=sys.stderr)
    # Capability matrix (ROADMAP #12, Waddle): what this rig can do, read
    # from the probed state above -- camera depth chain, sidecars, arms,
    # verifier, memory. The MCP server trims its tool surface by it; the
    # line here is what makes a withheld tool visible to the operator.
    from .capabilities import capability_matrix, format_matrix

    print(f"[cascade] capabilities: {format_matrix(capability_matrix(runtime))}")
    return runtime, arm


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
    out = {
        "agent_status": status,
        "task": runtime.current_task,
        "holding": runtime.held_object,
        # dispatch tier of the last command (reflex/experience/llm/mcp-host)
        "last_path": getattr(runtime, "last_path", None),
        "objects": runtime.beliefs.summary(),
        "arm_connected": getattr(runtime.arm.raw, "connected", True),
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
    if runtime.watcher is not None:
        out["perception"] = runtime.watcher.stats()
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
    the disconnect that follows.
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
            completed = arm.move_joints(_park_pose(arm), duration_s=duration_s, joint_margin=0.0)
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
    add("stream-server", getattr(runtime, "stream_server", None))
    add("viewer", getattr(runtime, "viewer", None))
    rig = getattr(runtime, "rig", None)
    for stream in (list(rig) if rig is not None else []):
        add(f"stream-{getattr(stream, 'name', '?')}", stream)
    return out


def shutdown_runtime(runtime, arm) -> dict:
    """Close in dependency order and return every observed software result.

    `arm` is the primary raw backend, kept as a positional for the many call
    sites that predate the arm rig. When a rig is present EVERY arm is
    disconnected through it -- disconnecting only the primary would leave a
    second arm powered (torque on, unsupervised) after teardown reported
    success.

    Order: park (needs the watcher's heartbeat) -> save beliefs -> watcher
    (the consumer of the camera streams) -> dashboard -> viewer -> camera
    streams (producers) -> arms. Each thread owner WAITS for its thread's
    in-flight work; a bounded join here once let the watcher's first cold
    YOLOE inference (> 5 s) outlive shutdown and abort the launcher's
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

    def _disconnect_arms():
        rig = getattr(runtime, "arm_rig", None)
        if rig is not None:
            return rig.disconnect()  # includes the primary and per-arm results
        return arm.disconnect()

    stages = []
    for name, step in (
        ("park", lambda: _park_arm(runtime)),
        ("beliefs", _save_beliefs),
        ("watcher", lambda: runtime.watcher.stop() if runtime.watcher is not None else None),
        ("stream_server", lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None),
        ("viewer", lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None),
        ("cameras", lambda: runtime.rig.close() if getattr(runtime, "rig", None) else runtime.camera.close()),
        ("arms", _disconnect_arms),
    ):
        stages.append(teardown_step(name, step))

    pending = []
    for label, thread in threads:
        if thread.is_alive():
            pending.append({"owner": label, "thread": thread.name})
            # Every owner above waits without a bound, so this only fires if
            # an owner regresses to a bounded join. Name it: the alternative
            # is an unexplained abort at interpreter exit.
            print(f"[cascade] WARNING: {label} thread {thread.name!r} is still running "
                  "after shutdown; native code may abort the process at exit",
                  file=sys.stderr)
    receipt = teardown_receipt(stages, pending_threads=pending)
    runtime._shutdown_receipt = copy.deepcopy(receipt)
    return receipt


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
    it; outside (0, 1] raises. A YAML string such as "false" is honoured.
    """
    value = mcfg.get("per_camera_colour", True)
    if isinstance(value, str):
        value = value.strip().lower() not in ("0", "false", "no", "off")
    iou = mcfg.get("neighbour_colour_iou")
    return BeliefStore(
        instance_association=_instance_association_enabled(mcfg),
        per_camera_colour=bool(value),
        neighbour_colour_iou=(NEIGHBOUR_COLOUR_IOU if iou is None else float(iou)),
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
    return PlausibilityChecker(
        verifier,
        beliefs=getattr(runtime, "beliefs", None),
        held_getter=lambda: getattr(runtime, "held_object", None),
        max_checks=int(agent_cfg.get("premotion_max_checks", 3)),
        workspace=workspace,
        skip_reason=skip_reason,
    )


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
    agent_cfg = cfg.get("agent", {}) or {}
    env = os.environ.get("CASCADE_PROGRAMS", "").strip().lower()
    if env:
        enabled = env not in ("0", "false", "no", "off")
    else:
        raw = agent_cfg.get("programs", False)
        enabled = (raw.strip().lower() in ("1", "true", "yes", "on")
                   if isinstance(raw, str) else bool(raw))
    if not enabled:
        return None
    if is_mock:
        print("[cascade] programs tier requested, but the mock brain is a labelled script, "
              "not an author: tier left off")
        return None
    from ..agent.programs import ProgramTier
    from ..memory.programs import ProgramLibrary
    from ..skills.library import PROMOTION_MIN_TASKS

    mem_cfg = cfg.get("memory", {}) or {}
    path = (os.environ.get("CASCADE_PROGRAMS_PATH") or mem_cfg.get("programs_path")
            or PACKAGE_ROOT / "runs" / "programs.jsonl")
    library = ProgramLibrary(Path(str(path)).expanduser(),
                             min_tasks=int(agent_cfg.get("program_min_tasks", PROMOTION_MIN_TASKS)))
    return ProgramTier(library)


def _make_detector(cfg):
    # A camera profile may pin its own detector (the mock camera uses the
    # mock detector so offline runs never load model weights).
    dcfg = cfg.camera.get("detector") or cfg.detector
    if dcfg.type == "mock":
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
    if dcfg.type == "vlm":
        # Full-VLM perception (cosmos3-edge or any OpenAI-compatible vision
        # server): no YOLOE, no ultralytics import. ~1-3 s per pass on a
        # local vLLM -- see perception/vlm_detector.py for the trade.
        from ..perception.vlm_detector import VLMDetector

        return VLMDetector(
            base_url=str(dcfg.get("base_url", "http://127.0.0.1:8082/v1")),
            model=str(dcfg.get("model", "cosmos3-edge")),
            api_key=str(dcfg.get("api_key", "EMPTY")),
            timeout_s=float(dcfg.get("timeout_s", 20.0)),
            conf=float(dcfg.get("conf", 0.25)),
        )
    from ..perception.detector import OpenVocabDetector

    return OpenVocabDetector(
        model_path=dcfg.model,
        device=dcfg.get("device", "auto"),
        conf=float(dcfg.get("conf", 0.25)),
        prompt_free=bool(dcfg.get("prompt_free", True)),
    )


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
                        "pin it (mock|hermes|anthropic|openai|local_qwen|"
                        "local_cosmos)")
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
            signals.checkpoint()
            status = _run_demo(args, cfg, runtime, mobile)
        except SignalRequest:
            # Interrupted with-blocks have released their locks. Invalidate
            # before any logging/teardown. Arm CLI retains halt -> park, not
            # e-stop -> torque-off; apply halt to every configured arm.
            if runtime is not None:
                if composed:
                    runtime.request_shutdown()
                elif mobile:
                    runtime.stop()
                else:
                    rig = getattr(runtime, "arm_rig", None)
                    for safe_arm in (list(rig) if rig is not None else [runtime.arm]):
                        try:
                            safe_arm.harness.halt("signal: graceful shutdown")
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
    advisor = Advisor(llm) if (llm.supports_vision and not is_mock) else None
    experience = None if mobile else ExperienceMemory(PACKAGE_ROOT / "runs" / "experience.json")
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
    from ..skills.library import PROMOTION_MIN_TASKS, SkillLibrary

    # Cross-task gate (upstream ASPIRE): a distilled note reaches the agent
    # only after it recurred in >= `memory.skill_min_tasks` distinct tasks
    # (default 2). 1 is the explicit single-observation mode; nothing lowers
    # it silently.
    library = None if mobile else SkillLibrary(
        PACKAGE_ROOT / "skills_library",
        min_tasks=int(cfg.memory.get("skill_min_tasks", PROMOTION_MIN_TASKS)),
    )
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
    )

    def _run(task: str):
        runtime.current_task = task
        try:
            return agent.run_task(task)
        finally:
            runtime.current_task = None

    # Wire the dashboard "send"/"stop" buttons to the live agent so you can
    # drive the robot from the browser (http://<ip>:8090) as well as the REPL.
    _srv = getattr(runtime, "stream_server", None)
    if _srv is not None:
        _srv.set_task_fn(lambda t: _print_report(_run(t)))
        _srv.set_cancel_fn(runtime.arm.stop)

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
