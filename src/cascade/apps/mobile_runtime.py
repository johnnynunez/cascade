"""Base-only composition. No arm, FK, grasp memory, detector or table frame."""
from __future__ import annotations

from ..agent.trace import TraceLogger
from ..control.mobile_rig import MobileRig
from ..memory.episodic import EpisodicMemory
from ..safety.base_harness import SafeBase
from ..skills.mobile_runtime import MobileSkillRuntime


def make_base(profile: dict):
    if profile["type"] == "mock":
        from ..control.mock_base import MockMobileBase

        return MockMobileBase(wall_lease_s=profile["wall_lease_s"],
                              robot_id=profile["robot_id"], source=profile["source"],
                              dt_s=profile["dt_s"])
    if profile["type"] == "isaac":
        for key in ("asset_sha256", "policy_sha256", "model_identity_sha256"):
            if not profile.get(key):
                raise ValueError(f"explicit {key} required; physical admission is pending")
        from ..control.isaac_base import IsaacBase

        return IsaacBase(profile)
    raise ValueError(f"unsupported mobile backend: {profile['type']!r}")


def build_mobile_runtime(cfg, run_dir, *, checkers=None):
    profiles = cfg.bases
    from ..sim.mobile_frames import MobileFrameReader, camera_profiles

    frame_readers = {p["name"]: MobileFrameReader(p) for p in profiles if camera_profiles(p)}
    bases = [SafeBase(make_base(p), p["resolved"]["safety"],
                      distance_control=p.get("distance_control"),
                      support_contract=p.get("support_contract")) for p in profiles]
    rig = MobileRig(bases, [p["name"] for p in profiles])
    memory_cfg = cfg.memory
    memory = EpisodicMemory(horizon_s=memory_cfg.get("horizon_s", 15.0),
                            frame_horizon_s=memory_cfg.get("frames_horizon_s", 600.0))
    # Construction and metadata are passive; the first state/motion opens IO.
    rt = MobileSkillRuntime(rig, memory, TraceLogger(run_dir), cfg, checkers=checkers)
    rt.frame_readers = frame_readers
    for profile in profiles:
        name = profile["name"]
        rt.configure_stop_observer(profile)
        if profile["type"] == "isaac":
            # Observation must not claim control or cause a stop on teardown.
            rt.observation_readers[name] = None
            try:
                from ..sim.base_truth import BaseTruthReader

                rt.observation_readers[name] = BaseTruthReader(profile)
            except Exception as exc:
                rt.observation_errors[name] = str(exc)
        if name in rt.checkers:
            continue
        reader = None
        try:
            if not isinstance(profile.get("verifier"), dict):
                raise ValueError("independent verifier limits not configured; physical admission pending")
            from ..agent.base_effects import BasePostconditionChecker

            if profile["type"] == "isaac":
                from ..sim.base_truth import BaseTruthReader

                reader = BaseTruthReader(profile)
            else:
                # Never substitute the actor's kinematic feedback for truth.
                reader = lambda: None
            rt.checkers[name] = BasePostconditionChecker(
                reader, limits=profile["verifier"], support_contract=profile.get("support_contract"))
        except Exception as exc:
            rt.verifier_errors[name] = f"independent verifier unavailable: {exc}"
            if reader is not None and hasattr(reader, "close"):
                reader.close()
    return rt, rig


def run_mobile_interactive(runtime, run_task, print_report, *, input_stream=None):
    """POSIX CLI reader: stop and EOF bypass a running task, just like MCP.

    Own a duplicate fd and poll it so cancellation can join the reader even
    when the human never types another line. No arm-oriented input/reflex path.
    """
    import os
    import queue
    import select
    import sys
    import threading

    stream = input_stream if input_stream is not None else sys.stdin
    fd = os.dup(stream.fileno())
    inbox = queue.Queue(maxsize=64)
    halt, eof = threading.Event(), threading.Event()
    gate = threading.Lock()
    generation = 0
    end = object()

    def invalidate():
        nonlocal generation
        with gate:
            generation += 1
        runtime.stop()

    def read_input():
        buffer = b""
        try:
            while not halt.is_set():
                ready, _, _ = select.select([fd], [], [], .05)
                if not ready:
                    continue
                data = os.read(fd, 4096)
                if not data:
                    break
                buffer += data
                if len(buffer) > 65536:
                    raise ValueError("CLI input line exceeds 64 KiB")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8").strip()
                    if not text:
                        return
                    if text in {"stop", "emergency_stop", "stop_navigation"}:
                        invalidate()
                    elif text == "reset_stop":
                        runtime.execute("reset_stop", {})
                    else:
                        with gate:
                            serial = generation
                        inbox.put_nowait((serial, text))
        finally:
            eof.set()
            invalidate()
            os.close(fd)
            # A full input queue is already a fail-closed overload. The main
            # loop observes eof; don't block the safety reader on queue space.
            try:
                inbox.put_nowait(end)
            except queue.Full:
                pass

    reader = threading.Thread(target=read_input, name="mobile-cli-input", daemon=True)
    reader.start()
    try:
        while not eof.is_set():
            item = inbox.get()
            if item is end:
                break
            serial, text = item
            with gate:
                valid = serial == generation and not eof.is_set()
            if valid:
                print_report(run_task(text))
    finally:
        halt.set()
        runtime.stop()
        reader.join()
