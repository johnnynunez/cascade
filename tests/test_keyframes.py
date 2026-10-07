"""Keyframes are the outcome judge's only evidence (eval/progress_judge.py),
so the runtime must record a BEFORE that exists and an AFTER that was
captured once the motion ENDED.

Measured failure this pins: two chat-driven pick_and_place runs logged
byte-identical before/after JPEGs (md5 8cb509e7...) while physics confirmed
a 19.8 cm move -- `last_frame` was whatever the skill's last observe() saw,
which predates the place motion; and the FIRST skill of every run logged
`keyframe_before: null` because nothing had observed yet.
"""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np
import pytest

from cascade.apps.demo import build_runtime


class _TickingCamera:
    """Wraps the runtime's camera so every grab paints a different frame:
    a camera that can WITNESS motion. Counts grabs."""

    def __init__(self, inner):
        self._inner = inner
        self.grabs = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_frame(self, *a, **k):
        frame = self._inner.get_frame(*a, **k)
        with self._lock:
            self.grabs += 1
            n = self.grabs
        rgb = np.ascontiguousarray(frame.rgb).copy()
        rgb[:8, :, :] = (n * 37) % 256   # distinct stripe per grab
        return type(frame)(**{**frame.__dict__, "rgb": rgb})


@pytest.fixture
def runtime_and_arm(demo_cfg, tmp_path):
    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    runtime.camera = _TickingCamera(runtime.camera)
    yield runtime, arm
    runtime.camera._inner.close()
    arm.disconnect()


def _md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _rows(runtime):
    return [json.loads(l) for l in (runtime.trace.run_dir / "trace.jsonl").read_text().splitlines()]


def test_first_skill_of_a_run_has_a_before_keyframe(runtime_and_arm):
    runtime, _ = runtime_and_arm
    assert runtime.last_frame is None                     # nothing observed yet
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    assert row["keyframe_before"], "first skill logged keyframe_before: null"
    assert (runtime.trace.run_dir / row["keyframe_before"]).exists()


def test_motion_skill_after_keyframe_is_a_fresh_frame(runtime_and_arm):
    runtime, _ = runtime_and_arm
    runtime.observe()
    grabs_before = runtime.camera.grabs
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    kb, ka = runtime.trace.run_dir / row["keyframe_before"], runtime.trace.run_dir / row["keyframe_after"]
    assert kb.exists() and ka.exists()
    # the AFTER frame was grabbed AFTER the motion (a new grab happened) ...
    assert runtime.camera.grabs > grabs_before
    # ... and it is not the BEFORE frame re-saved
    assert _md5(kb) != _md5(ka), "before/after keyframes are byte-identical"


def test_non_motion_skill_does_not_grab_an_extra_frame(runtime_and_arm):
    """Only motion skills pay for a fresh AFTER frame; a query skill's AFTER
    is the frame it just observed (no duplicate grab per tool call)."""
    runtime, _ = runtime_and_arm
    runtime.execute("get_observation", {})
    grabs = runtime.camera.grabs
    runtime.execute("robot_status", {}) if hasattr(runtime, "skill_robot_status") else runtime.execute("describe_scene", {})
    # describe_scene / robot_status observe at most once themselves; the
    # execute() wrapper must not add a second grab for a non-motion skill
    assert runtime.camera.grabs - grabs <= 1


def test_trace_rows_carry_the_dispatch_tier(runtime_and_arm):
    """per_tier() in the judge keys on it; 'unknown' for every row was the
    measured state before the dispatcher set runtime.current_tier."""
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.orchestrator import AgentOrchestrator

    runtime, _ = runtime_and_arm
    llm = MockLLM([LLMResponse(text="", tool_calls=[ToolCall("move_home", {})]),
                   LLMResponse(text="", tool_calls=[ToolCall("task_done", {"success": True, "summary": "ok"})])])
    agent = AgentOrchestrator(llm, runtime, advisor=None, decompose=False, max_steps=4)
    agent.run_task("go home please, via the llm tier")   # no reflex plan for this phrasing
    tiers = {r["skill"]: r.get("tier") for r in _rows(runtime)}
    assert tiers["move_home"] in ("llm", "reflex", "experience"), tiers
    assert runtime.current_tier is None                    # always reset after the call


# ── wrist keyframes (ROADMAP #15: the judge's two wrist slots) ──────────────


class _TickingStream:
    """Same trick as _TickingCamera for a rig STREAM: every get_frame() paints
    a distinct stripe and counts, so a wrist AFTER frame taken after the
    motion is distinguishable from the BEFORE one."""

    def __init__(self, inner):
        self._inner = inner
        self.grabs = 0
        self.latest_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _paint(self, frame, n):
        rgb = np.ascontiguousarray(frame.rgb).copy()
        rgb[-8:, :, :] = (n * 53) % 256
        return type(frame)(**{**frame.__dict__, "rgb": rgb})

    def latest(self):
        self.latest_calls += 1
        f = self._inner.latest()
        return None if f is None else self._paint(f, 1000 + self.latest_calls)

    def get_frame(self, *a, **k):
        self.grabs += 1
        return self._paint(self._inner.get_frame(*a, **k), self.grabs)


@pytest.fixture
def wrist_runtime(tmp_path):
    """Mock rig with a second camera declared as a WRIST view (`role: wrist`,
    no belief fusion) -- the generic flag the MuJoCo wrist profile and the
    eye-in-hand profiles share."""
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    cfg = load_demo_config(cameras=["mock_small", "mock"], arm="so101_mock", llm="mock")
    cfg.cameras[1]["role"] = "wrist"
    cfg.cameras[1]["fuse_beliefs"] = False
    runtime, arm = build_runtime(cfg, tmp_path / "run")
    runtime.rig.streams["mock"] = _TickingStream(runtime.rig.streams["mock"])
    yield runtime, arm
    shutdown_runtime(runtime, arm)


def test_motion_skill_records_wrist_keyframes_alongside_the_front_ones(wrist_runtime):
    runtime, _ = wrist_runtime
    wrist = runtime.rig.streams["mock"]
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    assert row["keyframe_before"] and row["keyframe_after"]
    kb, ka = row["keyframe_before_wrists"], row["keyframe_after_wrists"]
    assert list(kb) == ["mock"] and list(ka) == ["mock"], (kb, ka)
    pb, pa = runtime.trace.run_dir / kb["mock"], runtime.trace.run_dir / ka["mock"]
    assert pb.exists() and pa.exists()
    assert pb.name.startswith("0000_move_home_before") and "wrist" in pb.name and "mock" in pb.name
    # the AFTER wrist frame is a fresh grab taken once the motion ended ...
    assert wrist.grabs >= 1
    # ... and not the BEFORE wrist frame re-saved, nor the front keyframe
    assert _md5(pb) != _md5(pa), "before/after wrist keyframes are byte-identical"
    assert _md5(pa) != _md5(runtime.trace.run_dir / row["keyframe_after"])


def test_non_motion_skill_records_no_wrist_keyframes(wrist_runtime):
    """Wrist evidence is for the judge's BEFORE/AFTER pairs of MOTION skills;
    a query skill neither grabs from the wrist stream nor writes wrist files."""
    runtime, _ = wrist_runtime
    wrist = runtime.rig.streams["mock"]
    runtime.execute("get_observation", {})
    row = _rows(runtime)[0]
    assert row["keyframe_before_wrists"] is None and row["keyframe_after_wrists"] is None
    assert wrist.grabs == 0
    assert not any("wrist" in p.name for p in (runtime.trace.run_dir / "keyframes").iterdir())


def test_rig_without_a_wrist_stream_records_none(runtime_and_arm):
    """Single front camera: the keys exist (schema is uniform) and are null,
    which is what tells the judge to fall back to repeating the front view."""
    runtime, _ = runtime_and_arm
    runtime.execute("move_home", {})
    row = _rows(runtime)[0]
    assert "keyframe_before_wrists" in row and row["keyframe_before_wrists"] is None
    assert row["keyframe_after_wrists"] is None


def test_a_failing_wrist_stream_never_blocks_the_skill(wrist_runtime):
    runtime, _ = wrist_runtime

    class _Dead:
        name = "mock"

        def latest(self):
            raise RuntimeError("wrist camera unplugged")

        def get_frame(self, *a, **k):
            raise RuntimeError("wrist camera unplugged")

    real = runtime.rig.streams["mock"]
    runtime.rig.streams["mock"] = _Dead()
    try:
        res = runtime.execute("move_home", {})
    finally:
        runtime.rig.streams["mock"] = real              # let the fixture close the real stream
    assert res["ok"] is True
    row = _rows(runtime)[0]
    assert row["keyframe_before"] and row["keyframe_after"]      # front evidence intact
    assert row["keyframe_before_wrists"] is None and row["keyframe_after_wrists"] is None
