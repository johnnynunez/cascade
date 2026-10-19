"""Per-arm perception watchdog feeds on a multi-arm rig (#407 review follow-up 5).

The WorldWatcher heartbeats a harness for every NEW camera frame. On a rig of
two or more arms #407 fed every arm's watchdog from every camera ("the cameras
watch the shared table"). That holds for a FIXED camera, but a wrist
(eye-in-hand) camera looks out of ONE arm's hand: on main its frames kept every
other arm alive too, so on a multi-arm rig without base_pose (where eye-in-hand
cameras are allowed) an arm whose only cameras were dead still moved.

Rule (controller ruling, revisitable by the owner):
- a fixed camera feeds every arm's watchdog;
- an eye-in-hand camera feeds only the arm it is mounted on: the PRIMARY arm,
  whose FK its extrinsics compose (`build_runtime`'s `fk()`), the only wrist
  mount the build wires;
- a `role: wrist` view without hand-eye extrinsics names no arm, so on a rig
  it feeds none (fail closed);
- one arm: the watcher keeps heartbeating that harness itself (unchanged).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from base_pose_oracle import make_runtime, raw_arm  # noqa: F401  (make_runtime: a fixture, by name)

WRIST = "wrist"
SCENE = "mock_small"


def _cfg(arms, wrist):
    """`arms` on the mock stack with two cameras: the fixed scene camera
    (primary) and a second camera `wrist`, an eye-in-hand camera
    (`wrist="eye_in_hand"`) or a `role: wrist` view without extrinsics
    (`wrist="role"`). No arm declares a base_pose."""
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=list(arms), cameras=[SCENE, SCENE], llm="mock")
    cams = cfg._data["cameras"]
    cams[0]["name"] = SCENE
    cam = cams[1]
    cam["name"] = WRIST
    if wrist == "eye_in_hand":
        cam["extrinsics"]["mode"] = "eye_in_hand"
        cam["extrinsics"]["T"] = np.eye(4).tolist()
    else:
        cam.pop("extrinsics")
        cam["role"] = "wrist"
    for entry in [cfg._data["arm"], *(cfg._data.get("arms") or [])]:
        assert entry.get("base_pose") is None
    cfg._data["relative_motion"]["wrist_camera"] = None
    return cfg


def _harnesses(rt):
    return {name: safe.__dict__["harness"] for name, safe in rt.arm_rig.arms.items()}


def _kill(rt, *names):
    """These cameras die (the device stops delivering): their streams keep
    the last frame, which the watcher has processed once the call returns (no
    new frame, no heartbeat). The session has observed before
    (`rt.observe()`): a run's first skill takes its BEFORE frame through
    observe(), which heartbeats the arm under the call itself (the runtime's
    own observation, not the watcher's)."""
    rt.observe()

    def dead(*a, **k):
        raise RuntimeError("camera unplugged")

    killed = [cam for cam in rt.watcher._cams if cam.stream.name in names]
    assert len(killed) == len(names)
    for cam in killed:
        cam.stream._camera.get_frame = dead
    deadline = time.monotonic() + 10.0
    for cam in killed:
        # a grab failed after the swap (no frame in flight), and the watcher
        # has seen the stream's last frame
        while time.monotonic() < deadline and not (
                cam.stream.last_error and cam.last_seq == cam.stream.latest().frame_id):
            time.sleep(0.02)
        assert cam.stream.last_error and cam.last_seq == cam.stream.latest().frame_id


def _fed_after_ticks(rt, ticks=3):
    """Mark every arm's watchdog 100 s stale, let the watcher process `ticks`
    new frames (only live cameras count), and return which arms it fed."""
    hs = _harnesses(rt)
    stale = time.monotonic() - 100.0
    for h in hs.values():
        h._last_heartbeat = stale
    t0, deadline = rt.watcher.ticks, time.monotonic() + 10.0
    while rt.watcher.ticks < t0 + ticks and time.monotonic() < deadline:
        time.sleep(0.02)
    assert rt.watcher.ticks >= t0 + ticks, "the watcher processed the live camera's frames"
    return {name: h._last_heartbeat > stale for name, h in hs.items()}


def _move_up(rt, arm):
    raws = {name: raw_arm(safe) for name, safe in rt.arm_rig.arms.items()}
    n0 = {name: len(r.commands) for name, r in raws.items()}
    out = rt.execute("move_relative", {"direction": "up", "frame": "base", "distance_m": 0.02, "arm": arm})
    return out, {name: len(r.commands) - n0[name] for name, r in raws.items()}


def test_a_wrist_camera_does_not_keep_the_other_arm_alive(make_runtime):  # noqa: F811
    """Arm A (mock0, primary) carries the eye-in-hand camera, which stays
    live; the fixed scene camera is dead. Arm B's watchdog is never fed, its
    age grows past `watchdog_s`, and its motion is refused with 0 commands
    (on main A's wrist kept B alive: B moved, 50 commands). A stays fed."""
    rt = make_runtime(_cfg(["mock", "mock"], "eye_in_hand"))
    a, b = rt.arm_rig.names
    _kill(rt, SCENE)
    assert _fed_after_ticks(rt) == {a: True, b: False}
    h_b = _harnesses(rt)[b]
    assert time.monotonic() - h_b._last_heartbeat > h_b.limits.watchdog_s
    out, sent = _move_up(rt, b)
    assert out["ok"] is False, out
    assert "perception watchdog" in str(out.get("error")), out
    assert sent == {a: 0, b: 0}


def test_the_wrist_cameras_own_arm_stays_alive_and_moves(make_runtime):  # noqa: F811
    """Positive control: the arm the eye-in-hand camera rides on (the
    primary) is fed by it alone and still moves."""
    rt = make_runtime(_cfg(["mock", "mock"], "eye_in_hand"))
    a, b = rt.arm_rig.names
    _kill(rt, SCENE)
    assert _fed_after_ticks(rt)[a] is True
    out, sent = _move_up(rt, a)
    assert out["ok"] is True, out
    assert sent[a] > 0 and sent[b] == 0


@pytest.mark.parametrize("wrist", ["eye_in_hand", "role"])
def test_a_fixed_camera_keeps_every_arm_alive(make_runtime, wrist):  # noqa: F811
    """The fixed scene camera alone (the wrist view dead) feeds every arm:
    #407's rig-wide rule, kept for fixed cameras."""
    rt = make_runtime(_cfg(["mock", "mock"], wrist))
    _kill(rt, WRIST)
    assert _fed_after_ticks(rt) == {name: True for name in rt.arm_rig.names}


def test_a_wrist_view_without_hand_eye_feeds_no_arm_on_a_rig(make_runtime):  # noqa: F811
    """A `role: wrist` view without eye-in-hand extrinsics does not say which
    arm it rides on: on a rig it feeds no arm (fail closed). On main it fed
    both."""
    rt = make_runtime(_cfg(["mock", "mock"], "role"))
    _kill(rt, SCENE)
    assert _fed_after_ticks(rt) == {name: False for name in rt.arm_rig.names}


@pytest.mark.parametrize("wrist", ["eye_in_hand", "role"])
def test_one_arm_keeps_every_camera_feeding_its_watchdog(make_runtime, wrist):  # noqa: F811
    """One arm: unchanged. The watcher heartbeats that harness object itself,
    from any camera, the wrist view included."""
    rt = make_runtime(_cfg(["mock"], wrist))
    (name,) = rt.arm_rig.names
    assert rt.watcher._harness is _harnesses(rt)[name]
    _kill(rt, SCENE)
    assert _fed_after_ticks(rt) == {name: True}
