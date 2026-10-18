"""`point_at` keeps the closed jaws `safety.min_clearance_m` above the object.

The gesture hovers the closed gripper over the object. Its hover was
`min(top + pregrasp_offset_m, topdown_z_max - 0.01)`, with a fallback rung at
`topdown_z_max - 0.03`, and nothing checked either against the object. The
SO-101 profiles cap top-down poses at `topdown_z_max: 0.05`, so the hover was
0.04 m (fallback 0.02 m): over the 5 cm mock cube the TCP stopped 1 cm BELOW
the cube's top, the closed jaws ~2.5 cm below it. MuJoCo (so101_mujoco)
shoved the prop 7.7 mm and reported "did not settle"; on the mock stack the
call reported ok.

Rule now: every pose point_at commands keeps the TCP AND the jaws (the
profile's open-jaw sweep box, which bounds the closed jaws) at least
`safety.min_clearance_m` (0.03) above the object's observed top, checked on
the SOLVED pose. A hover below that is raised when the top-down ceiling
allows it, a fallback rung below it is never tried, and when the ceiling is
below the object top + margin the call refuses before any arm or jaw command.
Profiles whose hover already cleared command exactly what they did before.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every arm and jaw command; the MuJoCo
case needs `mujoco` and the fetched SO-101 assets (CPU, MUJOCO_GL=disable).
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import REPO, needs_pin, needs_pin_so101

from cascade.control.kinematics import IKResult
from cascade.control.mock_arm import MockArm

#: safety.min_clearance_m in configs/demo.yaml (no arm profile overrides it).
#: A literal on purpose: the test pins the shipped margin.
MARGIN = 0.03
#: mock / mock_small synthetic box: 5 cm tall, on the table
MOCK_TOP = 0.05
CEILING_TEXT = "cannot point at 'red cube' from above on this arm: its top-down ceiling is"


class _RecArm(MockArm):
    """MockArm that records every jaw command in the rig log."""

    def __init__(self, cfg, kinematics, log):
        super().__init__(cfg, kinematics)
        self.log = log

    def set_gripper(self, pos, effort=1.0):
        self.log.jaws.append(float(pos))
        super().set_gripper(pos, effort)


@pytest.fixture
def rig(monkeypatch, tmp_path):
    from cascade.apps import demo
    from cascade.apps.demo import shutdown_runtime
    from cascade.config import load_demo_config

    built = []

    def build(profile, camera="mock"):
        cfg = load_demo_config(camera=camera, arm=profile, llm="mock")
        # rebot_rs requires its live camera map and conftest pins
        # CASCADE_OCCUPANCY=0: that combination refuses startup by design.
        occ = cfg._data.get("occupancy")
        if isinstance(occ, dict) and occ.get("required"):
            occ["required"] = False
        r = SimpleNamespace(cfg=cfg, jaws=[], arms=[])

        def make_arm(acfg, kinematics=None):
            arm = _RecArm(acfg, kinematics, r)
            r.arms.append(arm)
            return arm

        monkeypatch.setattr(demo, "make_arm", make_arm)
        r.runtime, arm = demo.build_runtime(cfg, tmp_path / f"{profile}-{camera}", lazy_arm=True)
        built.append((r.runtime, arm))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and r.runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        return r

    yield build
    for runtime, arm in built:
        shutdown_runtime(runtime, arm)


def _commands(r) -> list:
    return [q for arm in r.arms for q in arm.commands]


def _jaw_points(cfg) -> np.ndarray:
    """Oracle: the TCP and the corners of the profile's open-jaw sweep box
    (grasp.graspgenx.sweep: origin at the TCP, +X closing, +Y third, +Z
    approach) in the TCP frame."""
    sweep = cfg.grasp.graspgenx.sweep
    ext = np.asarray(sweep.get("extents_open"), float)
    off = np.asarray(sweep.get("offset_open", [0.0, 0.0, 0.0]), float)
    a, j, t = {"down_open": (0, 1, 2), "open_down": (2, 0, 1),
               "third_open_down": (2, 1, 0)}[str(cfg.arm.get("tool_axis_order", "down_open"))]
    axes = np.eye(3)[:, [j, t, a]]
    corners = np.array(list(itertools.product(*[(-h, h) for h in ext / 2.0]))) + off
    return np.vstack([np.zeros(3), corners @ axes.T])


def _lowest_tool_z(r, q) -> float:
    T = r.runtime.kin.fk(q)
    return float((_jaw_points(r.cfg) @ T[:3, :3].T + T[:3, 3])[:, 2].min())


def _with_top(monkeypatch, runtime, top):
    """Localize normally, then move the fix so its observed top is `top`."""
    orig = runtime._localize

    def localize(label, *a, **k):
        frame, fix = orig(label, *a, **k)
        dz = top - float(fix.points[:, 2].max())
        return frame, dataclasses.replace(fix, points=fix.points + [0.0, 0.0, dz],
                                          position=fix.position + [0.0, 0.0, dz])

    monkeypatch.setattr(runtime, "_localize", localize)


def _record_ik(monkeypatch, runtime, solve=None):
    """Record the target z of every IK call; `solve(T, q0, real)` replaces it."""
    kin = runtime.kin
    real = kin.ik
    asked = []

    def ik(T, q0, *a, **k):
        asked.append(round(float(T[2, 3]), 6))
        return real(T, q0, *a, **k) if solve is None else solve(T, q0, real)

    monkeypatch.setattr(kin, "ik", ik)
    return asked


def _refused_untouched(r, out):
    assert out["ok"] is False, out
    assert _commands(r) == [] and r.jaws == [], "point_at sent a command before refusing"


# ── the shipped SO-101 ceiling: refuse, never dip into the object ─────────


@needs_pin_so101
@pytest.mark.parametrize("camera", ["mock_small", "mock"])
def test_so101_refuses_to_point_below_the_margin_before_any_command(rig, camera):
    r = rig("so101_mock", camera)
    assert float(r.cfg.safety.get("min_clearance_m")) == MARGIN

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert CEILING_TEXT + " 0.040 m, below the object top + margin" in out["error"], out
    assert "nothing was moved" in out["error"], out


@needs_pin_so101
@pytest.mark.parametrize("camera", ["mock_small", "mock"])
def test_so101_never_commands_a_pose_whose_jaws_reach_into_the_margin(rig, camera):
    """The physical property, whatever the outcome: no commanded pose puts the
    TCP or the jaws within the margin of the object's top."""
    r = rig("so101_mock", camera)
    r.runtime.execute("point_at", {"label": "red cube"})
    for q in _commands(r):
        assert _lowest_tool_z(r, q) >= MOCK_TOP + MARGIN - 1e-9


# ── profiles whose hover already clears: unchanged ─────────────────────────


@needs_pin
@pytest.mark.parametrize("profile", ["mock", "rebot_rs", "isaac_kitchen"])
def test_a_hover_that_clears_points_exactly_as_before(rig, monkeypatch, profile):
    r = rig(profile)
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    # today's hover: the cube's top + pregrasp_offset_m (0.04), first IK call
    assert asked[0] == pytest.approx(MOCK_TOP + 0.04, abs=1e-6), asked
    end = r.runtime.kin.fk(_commands(r)[-1])[2, 3]
    assert end == pytest.approx(MOCK_TOP + 0.04, abs=1e-3)
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


# ── the rule, rung by rung (mock arm, controlled object top) ───────────────


@needs_pin
def test_a_tall_object_above_the_ceiling_is_refused_before_any_command(rig, monkeypatch):
    """mock ceiling = topdown_z_max 0.15 - 0.01 = 0.14 m; a 12 cm top needs the
    TCP at 0.15 m. Before: the TCP hovered at 0.14, 2 cm over the top."""
    r = rig("mock")
    _with_top(monkeypatch, r.runtime, 0.12)
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert asked == [], "refused before any IK solve"
    assert (CEILING_TEXT + " 0.140 m, below the object top + margin (0.150 m: top 0.120 m + clearance "
            "0.030 m + 0.000 m of jaw below the TCP); nothing was moved") in out["error"], out


@needs_pin
def test_a_fallback_rung_inside_the_margin_is_never_tried(rig, monkeypatch):
    """Top 0.10 m: the hover rung (0.14) clears, the fallback rung (0.12) does
    not. With the hover unreachable the call refuses; before, it pointed from
    0.12 m, 2 cm over the top."""
    r = rig("mock")
    _with_top(monkeypatch, r.runtime, 0.10)

    def ceiling(T, q0, real):     # nothing solves above 0.135 m
        if T[2, 3] > 0.135:
            return IKResult(q=np.asarray(q0, float).copy(), success=False, error=1.0, iterations=0)
        return real(T, q0)

    asked = _record_ik(monkeypatch, r.runtime, ceiling)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert asked and min(asked) >= 0.10 + MARGIN, asked
    assert "cannot reach a pointing pose above 'red cube'" in out["error"], out


@needs_pin
def test_a_short_pregrasp_offset_is_raised_to_the_margin(rig, monkeypatch):
    """pregrasp_offset_m 0.01 put the TCP 1 cm over the top; the hover rises
    to the margin (+1 mm for the IK tolerance) while the ceiling allows."""
    r = rig("mock")
    r.runtime.cfg.grasp._data["pregrasp_offset_m"] = 0.01
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert asked[0] == pytest.approx(MOCK_TOP + MARGIN + 0.001, abs=1e-6), asked
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


@needs_pin
def test_jaws_that_reach_past_the_tcp_raise_the_hover_by_their_reach(rig, monkeypatch):
    """The SO-101 sweep box (offset 0, 30 mm deep along the approach) reaches
    15 mm past the TCP: the TCP must clear top + margin + 15 mm."""
    r = rig("mock")
    sweep = r.runtime.cfg.grasp.graspgenx.sweep._data
    sweep["extents_open"], sweep["offset_open"] = [0.055, 0.016, 0.030], [0.0, 0.0, 0.0]
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert asked[0] == pytest.approx(MOCK_TOP + MARGIN + 0.015 + 0.001, abs=1e-6), asked
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


@needs_pin
def test_the_solved_pose_is_checked_not_the_target(rig, monkeypatch):
    """IK answers that land lower than asked (TCP 0.04 m for the 0.09 m rung,
    0.03 m for the 0.12 m rung, over a 5 cm top) are rejected: the pose that
    would execute decides, before any command. The refusal names the highest
    of them."""
    r = rig("mock")

    def low(T, q0, real):
        T = np.array(T, dtype=float)
        T[2, 3] -= 0.05 if T[2, 3] < 0.1 else 0.09
        return real(T, q0)

    asked = _record_ik(monkeypatch, r.runtime, low)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert sorted(set(asked)) == [0.09, 0.12], asked      # both rungs tried, every yaw
    assert ("cannot reach a pointing pose above 'red cube' that keeps the closed jaws 0.030 m above "
            "its top (0.050 m): the best solved pose put them at 0.040 m; nothing was moved") in out["error"], out


@needs_pin
def test_a_solution_inside_the_margin_falls_through_to_the_next_rung(rig, monkeypatch):
    """The hover rung's answer lands 2 cm low (0.07 m, inside the margin);
    the fallback rung's (0.10 m) clears and is the one that executes."""
    r = rig("mock")

    def low(T, q0, real):
        T = np.array(T, dtype=float)
        T[2, 3] -= 0.02
        return real(T, q0)

    _record_ik(monkeypatch, r.runtime, low)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert r.runtime.kin.fk(_commands(r)[-1])[2, 3] == pytest.approx(0.10, abs=1e-3)
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


# ── halts and e-stops: zero further commands ───────────────────────────────


def _stop(runtime, kind):
    if kind == "halt":
        runtime.arm.harness.halt("operator: stop")
    else:
        runtime.arm.harness.estop("operator e-stop")


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_during_localization_sends_nothing(rig, monkeypatch, kind):
    r = rig("mock")
    orig = r.runtime._localize

    def localize_then_stop(label, *a, **k):
        res = orig(label, *a, **k)
        _stop(r.runtime, kind)
        return res

    monkeypatch.setattr(r.runtime, "_localize", localize_then_stop)
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is False, out
    assert _commands(r) == [] and r.jaws == []


@needs_pin
@pytest.mark.parametrize("kind", ["halt", "estop"])
def test_a_stop_mid_gesture_sends_no_further_command(rig, kind):
    r = rig("mock")
    r.runtime.arm.get_state()              # materialize the lazy arm
    arm = r.arms[0]
    send = arm.send_joint_target
    at_stop = {}

    def send_then_stop(q):
        send(q)
        if not at_stop and len(arm.commands) >= 5:
            _stop(r.runtime, kind)
            at_stop.update(arm=len(arm.commands), jaws=len(r.jaws))

    arm.send_joint_target = send_then_stop
    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert at_stop, "the gesture never streamed"
    assert len(arm.commands) == at_stop["arm"] and len(r.jaws) == at_stop["jaws"]
    assert out["ok"] is False, out


# ── MuJoCo physics: the prop stays where it was ────────────────────────────

MJCF = REPO / "assets" / "mjcf" / "so101" / "scene.xml"


def _has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


@needs_pin_so101
@pytest.mark.skipif(not (_has_mujoco() and MJCF.exists()),
                    reason="needs mujoco + `python scripts/fetch_robot_assets.py so101`")
def test_so101_mujoco_point_at_leaves_the_prop_where_it_was(monkeypatch, tmp_path):
    """Before: the closed jaws came down to 0.04 m on the 5 cm prop and pushed
    it 7.7 mm ("did not settle at the pointing pose")."""
    monkeypatch.setenv("MUJOCO_GL", "disable")
    from cascade.apps import demo
    from cascade.config import load_demo_config
    from cascade.sim.truth import make_truth_pose_fn

    cfg = load_demo_config(camera="mock_small", arm="so101_mujoco", llm="mock")
    runtime, raw = demo.build_runtime(cfg, tmp_path / "run")
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        truth = make_truth_pose_fn(raw)
        before = {k: np.asarray(v, float)[:3].copy() for k, v in truth.all_poses().items()}
        assert before, "the scene has no prop"
        sent, jaws = [], []
        send, grip = raw.send_joint_target, raw.set_gripper
        monkeypatch.setattr(raw, "send_joint_target", lambda q: (sent.append(q), send(q))[1])
        monkeypatch.setattr(raw, "set_gripper", lambda *a, **k: (jaws.append(a), grip(*a, **k))[1])

        out = runtime.execute("point_at", {"label": "red cube"})
        time.sleep(0.3)

        after = {k: np.asarray(v, float)[:3] for k, v in truth.all_poses().items()}
        moved = max(float(np.linalg.norm(after[k] - before[k])) for k in before)
        assert moved < 0.001, f"the prop moved {moved * 1000:.1f} mm: {out}"
        assert out["ok"] is False and CEILING_TEXT in out["error"], out
        assert sent == [] and jaws == []
    finally:
        demo.shutdown_runtime(runtime, raw)
