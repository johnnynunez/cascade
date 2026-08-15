"""Main demo entry point.

    wrc-demo --task "pick and place pink object" \
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
from ..agent.llm import make_llm
from ..agent.orchestrator import AgentOrchestrator
from ..agent.reflex import ExperienceMemory, FastPlanner
from ..agent.trace import TraceLogger
from ..config import Cfg, PACKAGE_ROOT, load_demo_config
from ..control.arm_base import make_arm
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


def build_runtime(
    cfg,
    run_dir: Path,
    view: bool = False,
    lazy_arm: bool = False,
    serve: bool = False,
) -> tuple[SkillRuntime, object]:
    kin = Kinematics(
        model_path=cfg.arm.model,
        ee_frame=cfg.arm.get("ee_frame", "gripper_end"),
        n_controlled=int(cfg.arm.get("n_joints", 6)),
        joint_signs=cfg.arm.get("joint_signs"),
    )
    if lazy_arm:
        # Perception pre-warms at startup; motors stay untouched until the
        # first motion command materializes the arm (see LazyArm).
        arm = LazyArm(lambda: make_arm(cfg.arm, kinematics=kin))
    else:
        arm = make_arm(cfg.arm, kinematics=kin)
        arm.connect()
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    safe_arm = SafeArm(arm, harness)

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
            print(f"[wrc-demo] livestream disabled ({opened.get('error')})",
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
    """Stop threads and hardware in dependency order; never raises."""
    import contextlib

    for step in (
        lambda: runtime.watcher.stop() if runtime.watcher is not None else None,
        lambda: runtime.stream_server.stop() if getattr(runtime, "stream_server", None) else None,
        lambda: runtime.viewer.stop() if getattr(runtime, "viewer", None) else None,
        lambda: runtime.rig.close() if getattr(runtime, "rig", None) else runtime.camera.close(),
        arm.disconnect,
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
        device=dcfg.get("device", "cuda:0"),
        conf=float(dcfg.get("conf", 0.25)),
        prompt_free=bool(dcfg.get("prompt_free", True)),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WRC agentic grasping demo")
    p.add_argument("--task", default=None, help="natural-language task")
    p.add_argument("--camera", default="mock", help="camera profile (mock|l515|d435i|uvc)")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera profiles; first = manipulation "
                        "camera (e.g. l515,uvc). Overrides --camera.")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs)")
    p.add_argument("--llm", default="mock", help="llm profile (mock|anthropic|local_qwen|openai)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None, help="trace output dir")
    p.add_argument("--interactive", action="store_true", help="multi-task REPL")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    p.add_argument("--no-serve", action="store_true",
                   help="do not start the MJPEG livestream dashboard")
    args = p.parse_args(argv)

    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    cfg = load_demo_config(camera=args.camera, cameras=cameras, arm=args.arm, llm=args.llm)
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    print(f"[wrc-demo] cameras={cameras or [args.camera]} arm={args.arm} "
          f"llm={args.llm} view={view}")
    print(f"[wrc-demo] traces -> {run_dir}")

    runtime, arm = build_runtime(cfg, run_dir, view=view, serve=not args.no_serve)
    if runtime.stream_server is not None:
        print(f"[wrc-demo] LIVESTREAM dashboard: {runtime.stream_server.url}")

    # Ctrl+C = soft stop (freeze + latch e-stop, no free-fall); a second
    # Ctrl+C raises KeyboardInterrupt and tears the process down.
    import signal

    def _sigint(_sig, _frm):
        print("\n[wrc-demo] SIGINT: soft-stopping the arm (Ctrl+C again to exit)")
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
