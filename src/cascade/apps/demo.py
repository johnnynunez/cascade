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
    if raw:
        return [Cfg(c) if isinstance(c, dict) else c for c in raw]
    return [cfg.camera]


def _truth_pose_fn(safe_arm):
    """Ground-truth prop poses when running against Isaac Sim, else None.

    Pigey's postcondition checker prefers a channel the actuator does not
    own. In sim that is the physics state; on real hardware there is none, so
    this returns None and the checker falls back to perception -- identical
    code path in both worlds.
    """
    try:
        from ..sim.truth import make_truth_pose_fn

        # skills only ever hold a SafeArm; the backend is behind .raw
        raw = getattr(safe_arm, "raw", safe_arm)
        return make_truth_pose_fn(raw)
    except Exception:
        return None


def _arm_cfgs(cfg) -> list[Cfg]:
    """All arm profiles (cfg.arms when present, else [cfg.arm])."""
    raw = cfg.get("arms")
    if raw:
        return [Cfg(a) if isinstance(a, dict) else a for a in raw]
    return [cfg.arm]


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
    if lazy_arm:
        # Perception pre-warms at startup; motors stay untouched until the
        # first motion command materializes the arm (see LazyArm).
        # n_joints must come from the profile: until the arm materializes,
        # LazyArm's hint is the only DOF answer anything can get, and the
        # class default (6) is wrong for a 5-DoF SO-101 or a 7-DoF Panda.
        arm = LazyArm(lambda: make_arm(acfg, kinematics=kin), n_joints=n_joints)
    else:
        arm = make_arm(acfg, kinematics=kin)
        arm.connect()
    harness = SafetyHarness(
        SafetyLimits.from_config(view.safety), kinematics=kin, occupancy=occupancy,
        base_pose=_base_transform(acfg),
    )
    return arm, SafeArm(arm, harness), kin


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
            safe.harness.add_neighbor(other_name, reader(other_safe, other_raw))


def build_runtime(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
) -> tuple[SkillRuntime, object]:
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
        watched.append(
            WatchedCamera(
                stream=stream,
                depth=DepthProvider(ccfg),
                extrinsics=Extrinsics.from_config(
                    ccfg.get("extrinsics", _empty_cfg()), fk_tcp2base=fk
                ),
                # A camera without calibrated extrinsics must not fuse 3D
                # beliefs (garbage base-frame positions); it still streams
                # video + overlays + heartbeats.
                fuse=bool(ccfg.get("fuse_beliefs", "extrinsics" in ccfg)),
            )
        )
    rig = CameraRig(streams)
    rig.open()
    try:
        rig.primary.warm_up(int(cam_cfgs[0].get("warmup_frames", 5)))
    except Exception:
        rig.close()  # a partial build must not leak open camera streams
        raise

    memory = EpisodicMemory(horizon_s=float(cfg.memory.get("horizon_s", 15.0)))
    beliefs = BeliefStore()
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        rig.primary, watched[0].depth, detector, watched[0].extrinsics,
        kin, safe_arm, memory, beliefs, trace, cfg,
    )
    runtime.rig = rig
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
        runtime.attach_verifier(object_pose=_truth_pose_fn(safe_arm))

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

    return runtime, arm


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
    }
    if runtime.watcher is not None:
        out["perception"] = runtime.watcher.stats()
    return out


def shutdown_runtime(runtime, arm) -> None:
    """Stop threads and hardware in dependency order; never raises.

    `arm` is the primary raw backend, kept as a positional for the many call
    sites that predate the arm rig. When a rig is present EVERY arm is
    disconnected through it -- disconnecting only the primary would leave a
    second arm powered (torque on, unsupervised) after teardown reported
    success.
    """
    import contextlib

    def _disconnect_arms():
        rig = getattr(runtime, "arm_rig", None)
        if rig is not None and len(rig) > 1:
            rig.disconnect()   # includes the primary; never raises
        else:
            arm.disconnect()

    for step in (
        lambda: runtime.watcher.stop() if runtime.watcher is not None else None,
        lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None,
        lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None,
        lambda: runtime.rig.close() if getattr(runtime, "rig", None) else runtime.camera.close(),
        _disconnect_arms,
    ):
        with contextlib.suppress(Exception):
            step()


def _empty_cfg():
    from ..config import Cfg

    return Cfg({})


def _make_detector(cfg):
    # A camera profile may pin its own detector (the mock camera uses the
    # mock detector so offline runs never load model weights).
    dcfg = cfg.camera.get("detector") or cfg.detector
    if dcfg.type == "mock":
        from ..perception.detector import MockDetector

        return MockDetector(label=dcfg.get("label", "red cube"))
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
    cfg = load_demo_config(camera=args.camera, cameras=cameras, arm=args.arm,
                           arms=arms, llm=llm)
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    print(f"[cascade] cameras={cameras or [args.camera]} "
          f"arms={arms or [args.arm]} "
          f"llm={llm}{' (auto)' if llm != args.llm else ''} view={view}")
    print(f"[cascade] traces -> {run_dir}")

    runtime, arm = build_runtime(cfg, run_dir, view=view, serve=not args.no_serve)
    if runtime.stream_server is not None:
        print(f"[cascade] LIVESTREAM dashboard: {runtime.stream_server.url}")

    # Ctrl+C = soft stop (freeze + latch e-stop, no free-fall); a second
    # Ctrl+C raises KeyboardInterrupt and tears the process down.
    import signal

    def _sigint(_sig, _frm):
        print("\n[cascade] SIGINT: soft-stopping the arm (Ctrl+C again to exit)")
        runtime.arm.stop()
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, _sigint)
    from ..agent.llm import MockLLM

    llm = make_llm(cfg.llm)
    is_mock = isinstance(llm, MockLLM)
    advisor = Advisor(llm) if (llm.supports_vision and not is_mock) else None
    experience = ExperienceMemory(PACKAGE_ROOT / "runs" / "experience.json")
    # ASPIRE: validated repairs distilled from earlier runs, retrieved into
    # context at task start. This is the loop the ROADMAP listed as open --
    # `scripts/learn_from_runs.py` writes the entries, the agent reads them.
    from ..skills.library import SkillLibrary

    library = SkillLibrary(PACKAGE_ROOT / "skills_library")
    agent = AgentOrchestrator(
        llm, runtime, advisor=advisor, max_steps=args.max_steps,
        decompose=not is_mock, fast_planner=FastPlanner(experience),
        skill_library=library, verify_milestones=not is_mock,
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

    try:
        if args.interactive:
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
            task = args.task or "look at the table and report what objects you see"
            report = _run(task)
            _print_report(report)
            return 0 if report.success else 1
    finally:
        shutdown_runtime(runtime, arm)
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
