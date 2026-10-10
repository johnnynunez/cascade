"""scripts/home_rebot_rs.py: the first powered move of a reBot RS through cascade.

On the rig (2026-10-11) this sequence moved the real arm rest -> home -> rest
with every segment within 0.013 rad. These tests pin the contract that made it
safe to run, offline (MockArm behind the REAL SafeArm/SafetyHarness of the
`rebot_rs` profile):

* a real run needs --yes; a camera that delivers nothing never connects the
  arm; the camera must beat the harness AFTER it is attached before the first
  command (no synthetic heartbeat, ever);
* the speed cap can only be tightened; the run refuses to start away from the
  folded rest (nothing commanded);
* segmented: a segment whose MEASURED pose misses its target soft-stops the
  arm (hold, torque ON) and nothing else is commanded; the same for a harness
  refusal mid-run;
* torque is released only when the arm is measured back at rest (or when
  nothing was ever commanded);
* only the final down segment relaxes the joint margin (onto the 0-rad stops
  of joints 2/3), exactly like the demo's park.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from conftest import needs_pin

from cascade.config import Cfg
from cascade.types import SafetyViolation

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

#: what the real arm read at rest on 2026-10-11
RIG_REST = [0.003, 0.008, 0.001, 0.006, 0.067, 0.001]
RATE_HZ = 50.0


class FakeCamera:
    """get_frame() delivers `frames` frames (None = forever), then raises."""

    def __init__(self, frames=None):
        self.left = frames
        self.closed = False

    def open(self):
        pass

    def get_frame(self):
        from cascade.perception.camera_base import CameraError

        time.sleep(0.002)
        if self.left is not None:
            if self.left <= 0:
                raise CameraError("unplugged")
            self.left -= 1
        return object()

    def close(self):
        self.closed = True


def _recording_arm_cls():
    from cascade.control.mock_arm import MockArm

    class RecordingArm(MockArm):
        """MockArm that logs every command; optionally stops TRACKING after
        `freeze_after` targets (measured pose stays where it was)."""

        def __init__(self, cfg, kinematics, q0, events, freeze_after=None):
            super().__init__(cfg, kinematics)
            self._q = np.asarray(q0, dtype=float).copy()
            self.events = events
            self.freeze_after = freeze_after
            self._frozen = None

        def send_joint_target(self, q):
            self.events.append(("target", np.asarray(q, dtype=float).copy()))
            n = sum(1 for e in self.events if e[0] == "target")
            if self.freeze_after is not None and n == self.freeze_after:
                self._frozen = self._q.copy()
            super().send_joint_target(q)

        def get_state(self):
            st = super().get_state()
            if self._frozen is not None:
                st.q = self._frozen.copy()
            return st

        def stop(self):
            self.events.append(("stop",))
            super().stop()

        def disconnect(self):
            self.events.append(("disconnect",))
            super().disconnect()

    return RecordingArm


def _builder(q0, events, freeze_after=None, spy_margins=None, refuse_on_move=None):
    """build(cfg, dry_run) -> (raw, safe, kin) through demo._build_arm with
    the real harness; only the backend is a recording MockArm."""
    from cascade.apps import demo

    calls = {"n": 0}

    def build(cfg, dry_run):
        calls["n"] += 1
        Rec = _recording_arm_cls()
        orig = demo.make_arm
        demo.make_arm = lambda a, kinematics=None: Rec(
            Cfg({"home_q": list(q0), "n_joints": int(a.n_joints)}), kinematics, q0, events,
            freeze_after)
        try:
            raw, safe, kin = demo._build_arm(cfg.arm, lazy_arm=False, occupancy=None,
                                             fallback_cfg=cfg)
        finally:
            demo.make_arm = orig
        if spy_margins is not None or refuse_on_move is not None:
            inner = safe.move_joints
            moves = {"n": 0}

            def move_joints(q, duration_s=2.0, joint_margin=None, **kw):
                moves["n"] += 1
                if spy_margins is not None:
                    spy_margins.append(joint_margin)
                if refuse_on_move is not None and moves["n"] == refuse_on_move:
                    raise SafetyViolation("refused for the test")
                return inner(q, duration_s=duration_s, joint_margin=joint_margin, **kw)
            safe.move_joints = move_joints
        return raw, safe, kin

    build.calls = calls
    return build


def _main(argv, build, camera):
    import home_rebot_rs as hr

    return hr.main(argv, build=build, make_camera=lambda cfg: camera)


def _targets(events):
    return [e[1] for e in events if e[0] == "target"]


def _cfg():
    from cascade.config import load_demo_config

    return load_demo_config(arm="rebot_rs", cameras=["mock"], llm="mock")


# ── gates before any command ─────────────────────────────────────────────


@needs_pin
def test_a_real_run_needs_yes():
    events = []
    build = _builder(RIG_REST, events)
    assert _main(["--hold-s", "0"], build, FakeCamera()) == 2
    assert build.calls["n"] == 0 and events == []


@needs_pin
def test_a_dead_camera_never_connects_the_arm():
    events = []
    build = _builder(RIG_REST, events)
    assert _main(["--yes", "--hold-s", "0", "--camera-wait-s", "0.3"], build,
                 FakeCamera(frames=0)) == 1
    assert build.calls["n"] == 0 and events == []


@needs_pin
def test_no_beat_after_attach_commands_nothing():
    """Warm-up frames arrived, then the camera died: the harness never got a
    REAL beat, so nothing may be commanded (no synthetic heartbeat)."""
    import home_rebot_rs as hr

    events = []
    build = _builder(RIG_REST, events)
    cam = FakeCamera(frames=hr.WARMUP_FRAMES)
    assert _main(["--yes", "--hold-s", "0", "--beat-wait-s", "0.3"], build, cam) == 1
    assert build.calls["n"] == 1
    assert _targets(events) == []


@needs_pin
def test_the_speed_cap_can_only_be_tightened():
    events = []
    build = _builder(RIG_REST, events)
    assert _main(["--yes", "--hold-s", "0", "--max-vel", "1.0"], build, FakeCamera()) == 2
    assert _targets(events) == []


@needs_pin
def test_refuses_to_start_away_from_the_folded_rest():
    events = []
    home = list(_cfg().arm.home_q)
    build = _builder(home, events)
    assert _main(["--yes", "--hold-s", "0"], build, FakeCamera()) == 1
    assert _targets(events) == []


# ── the run ──────────────────────────────────────────────────────────────


@needs_pin
def test_rest_home_rest_through_the_harness():
    events, margins = [], []
    build = _builder(RIG_REST, events, spy_margins=margins)
    assert _main(["--yes", "--hold-s", "0"], build, FakeCamera()) == 0
    targets = np.array(_targets(events))
    cfg = _cfg()
    home = np.asarray(cfg.arm.home_q, dtype=float)
    park = np.asarray(cfg.arm.park_q, dtype=float)
    # it reached home, then came back to rest
    assert np.min(np.max(np.abs(targets - home), axis=1)) < 1e-9
    np.testing.assert_allclose(targets[-1], park, atol=1e-9)
    # torque released once, after the last command
    assert events[-1] == ("disconnect",)
    assert sum(1 for e in events if e[0] == "disconnect") == 1
    # the tightened cap held on every streamed step (50 Hz)
    step_vel = np.max(np.abs(np.diff(targets, axis=0)), axis=1) * RATE_HZ
    assert step_vel.max() <= 0.4 + 1e-9
    # only the final down segment relaxes the joint margin
    assert margins[-1] == 0.0 and all(m is None for m in margins[:-1])
    assert len(margins) == 8


@needs_pin
def test_a_segment_that_does_not_track_soft_stops_and_keeps_torque(capsys):
    import home_rebot_rs as hr

    events = []
    # measured pose freezes after the first few targets of segment 1
    build = _builder(RIG_REST, events, freeze_after=5)
    assert _main(["--yes", "--hold-s", "0"], build, FakeCamera()) == 1
    # caught by the per-step MEASURED check, right after the first step
    assert "up step 1: the arm did not reach its target" in capsys.readouterr().out
    assert len(_targets(events)) <= int(hr.SEG_S * RATE_HZ) * 2
    i_stop = next(i for i, e in enumerate(events) if e[0] == "stop")
    assert all(e[0] != "target" for e in events[i_stop:])      # nothing after the stop
    assert ("disconnect",) not in events                       # torque stays ON


@needs_pin
def test_a_harness_refusal_mid_run_soft_stops_and_keeps_torque():
    events = []
    build = _builder(RIG_REST, events, refuse_on_move=3)
    assert _main(["--yes", "--hold-s", "0"], build, FakeCamera()) == 1
    i_stop = next(i for i, e in enumerate(events) if e[0] == "stop")
    assert all(e[0] != "target" for e in events[i_stop:])
    assert ("disconnect",) not in events


# ── the heartbeat feed ───────────────────────────────────────────────────


def test_only_frames_that_arrive_beat_the_harness():
    import home_rebot_rs as hr

    class H:
        beats = 0

        def heartbeat(self):
            H.beats += 1

    hb = hr.CameraHeartbeat(FakeCamera(frames=0))
    hb.start()
    hb.attach(H())
    assert hb.wait_beat(0.2) is False and H.beats == 0
    hb.stop()

    hb = hr.CameraHeartbeat(FakeCamera())
    hb.start()
    assert hb.wait_frames(5, 2.0) is True
    assert H.beats == 0                     # nothing beats before attach
    hb.attach(H())
    assert hb.wait_beat(2.0) is True and H.beats > 0
    hb.stop()
