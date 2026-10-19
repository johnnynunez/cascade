"""The perception watchdog follows the NEWEST observation's age: a stale frame never re-arms it.

`CameraStream.get_frame` waits for the next delivery and, when the camera has died, returns
its last frame of ANY age; a driver can also deliver its last buffer again. On main every
runtime frame grab then marked the watchdog fresh "now" (`SafetyHarness.heartbeat()`), so
an arm whose camera was dead kept moving:
- execute()'s BEFORE frame on a run's first skill: `rotate_joint` sent 100 commands on a
  frame 8.6 s (and 1 hour) old;
- execute()'s AFTER frame after every motion skill, refused ones too: the retry moved;
- in-skill observe(): `grasp_at_pixel` grasped (411 commands) on an 8.7 s-old frame.

Rule: the heartbeat carries the frame's arrival time (`heartbeat(at=...)`,
`_last_heartbeat = max(_last_heartbeat, at)`, non-finite / future stamps ignored), and the
runtime feeds the watchdogs of the camera's per-arm feed (#412), so the watchdog age is the
age of the newest observation that arm may move on.

Every test kills the camera through its driver, then pretends it died `DIED_S` ago: the
stream's last frame and every harness heartbeat move back that far (that is what waiting
would do). Each stream's `get_frame` keeps its exact semantics with a 0.3 s wait instead
of 3 s (a dead camera still returns its last frame once the wait ends).
"""

from __future__ import annotations

import functools
import math
import time

import numpy as np
import pytest
from base_pose_oracle import make_runtime, raw_arm  # noqa: F401  (make_runtime: a fixture, by name)

from cascade.perception.stream import CameraStream
from cascade.safety.harness import SafetyHarness, SafetyLimits

DIED_S = 10.0
MOVE = ("rotate_joint", {"joint": 1, "degrees": 4.0})
MOVE_BACK = ("rotate_joint", {"joint": 1, "degrees": -4.0})


# ── SafetyHarness.heartbeat(at=...) ───────────────────────────────────────


def _harness():
    return SafetyHarness(SafetyLimits(workspace_min=np.full(3, -1.0), workspace_max=np.ones(3)))


def test_a_heartbeat_at_a_frames_arrival_time_sets_the_watchdog_age_to_that_frames_age():
    h = _harness()
    t = time.monotonic()
    h._last_heartbeat = t - 30.0
    h.heartbeat(at=t - 7.0)
    assert h._last_heartbeat == t - 7.0


def test_an_older_observation_never_moves_the_watchdog_back():
    h = _harness()
    t = time.monotonic()
    h._last_heartbeat = t - 1.0
    h.heartbeat(at=t - 20.0)
    assert h._last_heartbeat == t - 1.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "soon", object(), 1e9])
def test_a_non_finite_or_future_or_unreadable_stamp_is_ignored(bad):
    h = _harness()
    t = time.monotonic() - 30.0
    h._last_heartbeat = t
    h.heartbeat(at=bad if bad != 1e9 else time.monotonic() + 60.0)
    assert h._last_heartbeat == t


def test_a_heartbeat_without_a_stamp_still_means_now():
    h = _harness()
    h._last_heartbeat = time.monotonic() - 30.0
    before = time.monotonic()
    h.heartbeat()
    assert before <= h._last_heartbeat <= time.monotonic()


def test_a_stale_heartbeat_leaves_the_motion_refused():
    h = _harness()
    t = time.monotonic()
    h._last_heartbeat = t - 3600.0
    h.heartbeat(at=t - 3600.0 + 1.0)
    with pytest.raises(Exception, match="perception watchdog"):
        h.check_stream_start()


def test_the_rig_heartbeat_forwards_the_stamp_to_every_arm():
    from cascade.apps.demo import _RigHeartbeat

    a, b = _harness(), _harness()
    t = time.monotonic()
    a._last_heartbeat = b._last_heartbeat = t - 30.0
    _RigHeartbeat([a, b]).heartbeat(at=t - 20.0)
    assert a._last_heartbeat == b._last_heartbeat == t - 20.0
    _RigHeartbeat([a, b]).heartbeat(at=t - 40.0)
    assert a._last_heartbeat == b._last_heartbeat == t - 20.0


# ── the mock rig with a dead camera ───────────────────────────────────────


def _short_waits(rt, monkeypatch, wait_s=0.3):
    for stream in rt.rig:
        monkeypatch.setattr(stream, "get_frame", functools.partial(CameraStream.get_frame, stream, timeout_s=wait_s))


def _harnesses(rt):
    return {name: safe.__dict__["harness"] for name, safe in rt.arm_rig.arms.items()}


def _cmds(rt):
    return {name: len(raw_arm(safe).commands) for name, safe in rt.arm_rig.arms.items()}


def _call(rt, name, args, arm=None):
    n0 = _cmds(rt)
    args = dict(args, **({"arm": arm} if arm is not None else {}))
    out = rt.execute(name, args)
    n1 = _cmds(rt)
    return out, {k: n1[k] - n0[k] for k in n1}


def _kill(rt, names=None, mode="dead", died_s=DIED_S):
    """The cameras `names` (all by default) stop delivering, `died_s` ago.

    mode `dead`: the driver raises (the stream keeps its last frame and `get_frame` returns it
    after its wait); `same`: the driver hands its last Frame back again (the stream counts a
    new delivery at once); `grab`: the camera's grab repeats its last buffer, so the camera
    stamps the same Frame with a NEW frame id (the watcher counts it as a new frame too)."""
    cams = [c for c in rt.watcher._cams if names is None or c.stream.name in names]
    assert cams
    lasts = []
    for c in cams:
        last = c.stream.latest()
        assert last is not None
        lasts.append(last)
        if mode == "dead":
            def dead(*a, **k):
                raise RuntimeError("camera unplugged")
            c.stream._camera.get_frame = dead
        elif mode == "same":
            c.stream._camera.get_frame = lambda last=last: (time.sleep(0.02), last)[1]
        else:
            c.stream._camera._grab = lambda last=last: (time.sleep(0.02), last)[1]
    deadline = time.monotonic() + 10.0
    for c, last in zip(cams, lasts):
        while time.monotonic() < deadline:
            f = c.stream.latest()
            settled = c.stream.last_error is not None if mode == "dead" else f is last
            if settled and c.last_seq == f.frame_id:
                break
            time.sleep(0.02)
        assert c.stream.latest() is last or mode == "dead"
    stale = []
    for c in cams:
        f = c.stream.latest()
        f.t -= died_s
        stale.append(f)
    newest = max(f.t for f in stale)
    for h in _harnesses(rt).values():
        h._last_heartbeat = min(h._last_heartbeat, newest)
    return stale


def _mock(make_runtime):  # noqa: F811
    from cascade.config import load_demo_config

    return make_runtime(load_demo_config(arm="mock", camera="mock", llm="mock"))


def test_first_skill_on_a_dead_camera_is_refused_with_no_command(make_runtime, monkeypatch):  # noqa: F811
    """A run's first skill (no frame yet): execute()'s BEFORE frame is the dead camera's last
    frame. Main: the watchdog was re-armed by it and rotate_joint sent 100 commands."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.last_frame = None
    _kill(rt)
    out, sent = _call(rt, *MOVE)
    assert out["ok"] is False, out
    assert "perception watchdog" in str(out.get("error")), out
    assert sent == {name: 0 for name in sent}


def test_a_one_hour_old_frame_never_re_arms_the_watchdog(make_runtime, monkeypatch):  # noqa: F811
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.last_frame = None
    _kill(rt, died_s=3600.0)
    out, sent = _call(rt, *MOVE)
    assert out["ok"] is False and sent == {name: 0 for name in sent}, out
    h = next(iter(_harnesses(rt).values()))
    assert time.monotonic() - h._last_heartbeat > 3600.0


def test_immediate_retries_after_a_refusal_stay_refused(make_runtime, monkeypatch):  # noqa: F811
    """The refused call's AFTER frame (the dead camera's last frame) re-armed the watchdog on
    main: the two retries sent 100 commands each."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.observe()  # a session that has observed already
    _kill(rt)
    rows = [_call(rt, *MOVE), _call(rt, *MOVE_BACK), _call(rt, *MOVE)]
    for out, sent in rows:
        assert out["ok"] is False, out
        assert "perception watchdog" in str(out.get("error")), out
        assert sent == {name: 0 for name in sent}


@pytest.mark.parametrize("mode", ["same", "grab"])
def test_a_driver_repeating_its_last_buffer_never_re_arms_the_watchdog(make_runtime, monkeypatch, mode):  # noqa: F811
    """`same`: the driver returns its last Frame (no 3 s wait at all; main moved at once);
    `grab`: the grab repeats its buffer under a new frame id, so the WATCHER sees a new frame
    and, on main, marked the watchdog fresh from it."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.last_frame = None
    _kill(rt, mode=mode)
    time.sleep(0.5)  # the watcher ticks over the repeated deliveries
    out, sent = _call(rt, *MOVE)
    assert out["ok"] is False, out
    assert sent == {name: 0 for name in sent}
    out, sent = _call(rt, *MOVE_BACK)
    assert out["ok"] is False and sent == {name: 0 for name in sent}, out


def test_grasp_at_pixel_on_a_dead_camera_sends_nothing(make_runtime, monkeypatch):  # noqa: F811
    """In-skill observe() (grasp_at_pixel): main grasped on the 8.7 s-old frame (411 commands)."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.observe()
    _kill(rt)
    out, sent = _call(rt, "grasp_at_pixel", {"u": 320, "v": 230})
    assert out["ok"] is False, out
    assert sent == {name: 0 for name in sent}
    assert "perception watchdog" in str(out.get("error")), out


def test_live_camera_control_still_moves(make_runtime, monkeypatch):  # noqa: F811
    """Positive control: the same calls with the camera live move as before."""
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.last_frame = None
    for args in (MOVE, MOVE_BACK):
        out, sent = _call(rt, *args)
        assert out["ok"] is True, out
        assert all(n > 0 for n in sent.values()), sent


def test_live_camera_grasp_at_pixel_still_grasps(make_runtime, monkeypatch):  # noqa: F811
    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.observe()
    out, sent = _call(rt, "grasp_at_pixel", {"u": 320, "v": 230})
    assert out["ok"] is True, out
    assert all(n > 0 for n in sent.values()), sent


def test_a_live_frame_beats_the_watchdog_at_its_own_arrival_time(make_runtime):  # noqa: F811
    """The watchdog's age equals the newest frame's age (not the time of the grab)."""
    rt = _mock(make_runtime)
    h = next(iter(_harnesses(rt).values()))
    h._last_heartbeat = time.monotonic() - 100.0
    frame = rt.observe()
    assert h._last_heartbeat >= frame.t - 1e-9
    assert h._last_heartbeat <= time.monotonic()


# ── the shipped so101 pair (one fixed scene camera) ───────────────────────


def _so101_pair(make_runtime):  # noqa: F811
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["so101_left", "so101_right"], cameras=["mock_small"], llm="mock")
    cfg._data["relative_motion"]["wrist_camera"] = None
    return make_runtime(cfg)


def test_shipped_pair_first_call_on_a_dead_scene_camera_moves_neither_arm(make_runtime, monkeypatch):  # noqa: F811
    """Main: the run's first skill on so101_right re-armed it from the dead camera's last frame
    (100 + 100 commands), and every call's AFTER frame re-armed so101_left (the primary)."""
    rt = _so101_pair(make_runtime)
    _short_waits(rt, monkeypatch)
    left, right = rt.arm_rig.names
    assert (left, right) == ("so101_left", "so101_right")
    rt.last_frame = None
    _kill(rt)
    hs = _harnesses(rt)
    left_beat = hs[left]._last_heartbeat
    ages = []
    for args in (MOVE, MOVE_BACK):
        out, sent = _call(rt, *args, arm=right)
        assert out["ok"] is False, out
        assert "perception watchdog" in str(out.get("error")), out
        assert sent == {left: 0, right: 0}
        ages.append(time.monotonic() - hs[left]._last_heartbeat)
    assert hs[left]._last_heartbeat == left_beat
    assert ages[1] > ages[0] > hs[left].limits.watchdog_s


def test_shipped_pair_live_scene_camera_still_moves_both_arms(make_runtime, monkeypatch):  # noqa: F811
    rt = _so101_pair(make_runtime)
    _short_waits(rt, monkeypatch)
    rt.last_frame = None
    for arm in rt.arm_rig.names:
        out, sent = _call(rt, *MOVE, arm=arm)
        assert out["ok"] is True, out
        assert sent[arm] > 0 and sum(sent.values()) == sent[arm], sent


# ── #412's eye-in-hand rig: in-skill grabs use the camera's own feed ──────


def _eih_pair(make_runtime, wrist_first):  # noqa: F811
    """Two mock arms; a fixed scene camera and an eye-in-hand camera on the
    primary arm (`wrist_first`: the wrist camera is the rig's primary view)."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["mock", "mock"], cameras=["mock_small", "mock_small"], llm="mock")
    cams = cfg._data["cameras"]
    cams[0]["name"], cams[1]["name"] = "scene", "wrist"
    cams[1]["extrinsics"]["mode"] = "eye_in_hand"
    cams[1]["extrinsics"]["T"] = np.eye(4).tolist()
    if wrist_first:
        cams.insert(0, cams.pop())
    cfg._data["relative_motion"]["wrist_camera"] = None
    return make_runtime(cfg)


@pytest.mark.parametrize("wrist_first", [False, True])
def test_the_other_arm_never_moves_on_a_frame_its_cameras_do_not_watch(make_runtime, monkeypatch, wrist_first):  # noqa: F811
    """The scene camera is dead; the eye-in-hand camera on arm A stays live. Arm B's run starts
    with no frame: its BEFORE frame (the primary camera) must not feed B. Main: B moved
    100 + 100 (the dead scene's last frame, or A's live wrist view, fed B's harness)."""
    rt = _eih_pair(make_runtime, wrist_first)
    _short_waits(rt, monkeypatch)
    a, b = rt.arm_rig.names
    rt.last_frame = None
    _kill(rt, names=["scene"])
    for args in (MOVE, MOVE_BACK):
        out, sent = _call(rt, *args, arm=b)
        assert out["ok"] is False, out
        assert "perception watchdog" in str(out.get("error")), out
        assert sent == {a: 0, b: 0}


def test_the_wrist_cameras_own_arm_still_moves_on_its_live_frames(make_runtime, monkeypatch):  # noqa: F811
    rt = _eih_pair(make_runtime, True)
    _short_waits(rt, monkeypatch)
    a, b = rt.arm_rig.names
    rt.last_frame = None
    _kill(rt, names=["scene"])
    out, sent = _call(rt, *MOVE, arm=a)
    assert out["ok"] is True, out
    assert sent[a] > 0 and sent[b] == 0


# ── localization on a dead camera says so ─────────────────────────────────


def test_localize_on_a_dead_camera_says_the_camera_stopped_not_that_it_is_live(make_runtime, monkeypatch):  # noqa: F811
    """Main: "Perception took 9.0s on a frame that was fresh at capture ... the camera is live --
    retry the observation" for a frame that was already stale when it was grabbed."""
    from cascade.skills.runtime import StaleFrameError

    rt = _mock(make_runtime)
    _short_waits(rt, monkeypatch)
    deadline = time.monotonic() + 10.0
    while rt.beliefs.find("red cube") is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert rt.beliefs.find("red cube") is not None
    rt.observe()
    _kill(rt)
    for belief in rt.beliefs.all():
        belief.last_seen_t = time.monotonic()  # recent enough for the memory fallback
    with pytest.raises(StaleFrameError) as raised:
        rt._localize("red cube")
    text = str(raised.value)
    assert text.startswith("Camera frame is ") and "s old; wait for the camera to reconnect" in text, text
    assert "camera is live" not in text
    out, sent = _call(rt, "grasp_object", {"label": "red cube"})
    assert out["ok"] is False and sent == {name: 0 for name in sent}, out
    assert "wait for the camera to reconnect" in str(out.get("error")), out
    assert "camera is live" not in str(out.get("error")), out


def test_slow_analysis_of_a_fresh_frame_keeps_its_own_text():
    """A frame fresh when analysis began keeps the slow-perception text (the camera is live)."""
    from cascade.skills.runtime import SlowPerceptionError, StaleFrameError, _analyzed_frame_age_s
    from cascade.types import Frame

    frame = Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=None, K=np.eye(3), t=time.monotonic() - 7.0)
    with pytest.raises(SlowPerceptionError) as slow:
        _analyzed_frame_age_s(frame, 6.0)
    assert not isinstance(slow.value, StaleFrameError)
    assert "the camera is live" in str(slow.value)
    with pytest.raises(StaleFrameError, match="Camera frame is 7.0s old; wait for the camera to reconnect"):
        _analyzed_frame_age_s(frame, 1.0)
    with pytest.raises(StaleFrameError):
        _analyzed_frame_age_s(frame, 6.0, age_at_start=5.5)
    with pytest.raises(SlowPerceptionError) as slow:
        _analyzed_frame_age_s(frame, 1.0, age_at_start=0.1)
    assert not isinstance(slow.value, StaleFrameError)
    assert _analyzed_frame_age_s(frame, 1.0, max_age_s=10.0) == pytest.approx(7.0, abs=0.5)


# ── a producer that re-delivers its capture ───────────────────────────────


def _isaac_frame(stamp, epoch="e1"):
    from cascade.types import Frame

    return Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=None, K=np.eye(3),
                 capture={"backend": "isaac", "source": [epoch, 1], "camera": "front",
                          "t": stamp, "proprioception": {"robot_id": "r", "backend": "isaac",
                                                         "time_source": "physics_loop_monotonic", "t": stamp}})


def test_a_re_delivered_producer_capture_keeps_its_first_arrival_time():
    """An Isaac bridge delivers its cached capture again under a NEW Frame and receipt time:
    the watchdog sees the first delivery's arrival, never the copy's."""
    from cascade.perception.freshness import observed_at

    stream = CameraStream(camera=None, name="isaac")
    first = _isaac_frame(10.0)
    first.t -= 8.0
    stream._note_first_receipt(first)
    assert observed_at(first) == first.t
    copy = _isaac_frame(10.0)
    stream._note_first_receipt(copy)
    assert copy.observed_t == first.t and observed_at(copy) == first.t
    newer = _isaac_frame(10.5)
    stream._note_first_receipt(newer)
    assert newer.observed_t is None and observed_at(newer) == newer.t
    other = _isaac_frame(10.5, epoch="e2")  # another producer identity: its own first receipt
    stream._note_first_receipt(other)
    assert other.observed_t is None
    h = _harness()
    h._last_heartbeat = first.t - 1.0
    h.heartbeat(at=observed_at(copy))
    assert h._last_heartbeat == first.t


def test_local_frames_are_never_restamped():
    from cascade.perception.freshness import observed_at
    from cascade.types import Frame

    stream = CameraStream(camera=None, name="local")
    f = Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=None, K=np.eye(3))
    stream._note_first_receipt(f)
    stream._note_first_receipt(f)
    assert f.observed_t is None and observed_at(f) == f.t
    assert math.isnan(observed_at(object()))


# ── calibration grabs ──────────────────────────────────────────────────────


def test_a_calibration_grab_beats_at_the_frames_arrival_time():
    from types import SimpleNamespace

    from cascade.calibration.session import CollectionSession

    old = SimpleNamespace(t=time.monotonic() - 30.0)
    h = _harness()
    h._last_heartbeat = time.monotonic() - 60.0
    session = CollectionSession.__new__(CollectionSession)
    session.camera = SimpleNamespace(get_frame=lambda: old)
    session.harness = h
    session.log = lambda *_a, **_k: None
    assert session.grab() is old
    assert h._last_heartbeat == old.t
