"""Main demo entry point.

    wrc-demo --task "put the red cube in the bowl" \
             --camera l515 --arm rebot_rs --llm anthropic

Defaults are the fully-offline stack (mock camera/arm/llm) so the wiring can
always be exercised without hardware or network. `--interactive` keeps the
session open for multiple tasks with persistent memory and beliefs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..agent.advisor import Advisor
from ..agent.llm import make_llm
from ..agent.orchestrator import AgentOrchestrator
from ..agent.trace import TraceLogger
from ..config import PACKAGE_ROOT, load_demo_config
from ..control.arm_base import make_arm
from ..control.kinematics import Kinematics
from ..memory import BeliefStore, EpisodicMemory
from ..perception.camera_base import make_camera
from ..perception.depth_provider import DepthProvider
from ..perception.grounding import Extrinsics
from ..safety.harness import SafeArm, SafetyHarness, SafetyLimits
from ..skills.runtime import SkillRuntime


def build_runtime(cfg, run_dir: Path, view: bool = False) -> tuple[SkillRuntime, object]:
    kin = Kinematics(
        urdf_path=cfg.arm.urdf,
        ee_frame=cfg.arm.get("ee_frame", "gripper_end"),
        n_controlled=int(cfg.arm.get("n_joints", 6)),
    )
    arm = make_arm(cfg.arm, kinematics=kin)
    arm.connect()
    harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin)
    safe_arm = SafeArm(arm, harness)

    camera = make_camera(cfg.camera)
    if view:
        # Always-on visualization: wrap the camera in a FrameHub so a live
        # window shows what the camera sees (plus agent overlays) while the
        # runtime keeps getting fresh frames from the same stream.
        from .live_view import FrameHub

        camera = FrameHub(camera, title="wrc-demo :: live", show=True)
    camera.open()
    camera.warm_up(int(cfg.camera.get("warmup_frames", 5)))
    depth = DepthProvider(cfg.camera)

    extr = Extrinsics.from_config(
        cfg.camera.get("extrinsics", {}) if "extrinsics" in cfg.camera else _empty_cfg(),
        fk_tcp2base=lambda: kin.fk(arm.get_state().q),
    )

    detector = _make_detector(cfg)
    memory = EpisodicMemory(horizon_s=float(cfg.memory.get("horizon_s", 15.0)))
    beliefs = BeliefStore()
    trace = TraceLogger(run_dir)
    runtime = SkillRuntime(
        camera, depth, detector, extr, kin, safe_arm, memory, beliefs, trace, cfg
    )
    return runtime, arm


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
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WRC agentic grasping demo")
    p.add_argument("--task", default=None, help="natural-language task")
    p.add_argument("--camera", default="mock", help="camera profile (mock|l515|d435i|uvc)")
    p.add_argument("--arm", default="mock", help="arm profile (mock|rebot_rs)")
    p.add_argument("--llm", default="mock", help="llm profile (mock|anthropic|local_qwen|openai)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--run-dir", default=None, help="trace output dir")
    p.add_argument("--interactive", action="store_true", help="multi-task REPL")
    p.add_argument("--no-view", action="store_true",
                   help="do not open the live camera window")
    args = p.parse_args(argv)

    cfg = load_demo_config(camera=args.camera, arm=args.arm, llm=args.llm)
    run_dir = Path(args.run_dir) if args.run_dir else (
        PACKAGE_ROOT / "runs" / time.strftime("%Y%m%d_%H%M%S")
    )
    import os

    view = not args.no_view and bool(os.environ.get("DISPLAY"))
    print(f"[wrc-demo] camera={args.camera} arm={args.arm} llm={args.llm} view={view}")
    print(f"[wrc-demo] traces -> {run_dir}")

    runtime, arm = build_runtime(cfg, run_dir, view=view)

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
    agent = AgentOrchestrator(
        llm, runtime, advisor=advisor, max_steps=args.max_steps, decompose=not is_mock
    )

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
                report = agent.run_task(task)
                _print_report(report)
        else:
            task = args.task or "look at the table and report what objects you see"
            report = agent.run_task(task)
            _print_report(report)
            return 0 if report.success else 1
    finally:
        runtime.camera.close()
        arm.disconnect()
    return 0


def _print_report(report) -> None:
    print("\n=== task report ===")
    print(f"task:    {report.task}")
    print(f"success: {report.success}")
    print(f"steps:   {report.steps}")
    print(f"summary: {report.summary}")


if __name__ == "__main__":
    sys.exit(main())
