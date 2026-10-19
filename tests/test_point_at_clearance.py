"""`point_at` keeps the closed jaws `safety.min_clearance_m` above the object.

The gesture hovers the closed gripper over the object. Its hover was
`min(top + pregrasp_offset_m, topdown_z_max - 0.01)`, with a fallback rung at
`topdown_z_max - 0.03`, and nothing checked either against the object. The
SO-101 profiles cap top-down poses at `topdown_z_max: 0.05`, so the hover was
0.04 m (fallback 0.02 m): over the 5 cm mock cube the TCP stopped 1 cm BELOW
the cube's top, the closed jaws ~2.5 cm below it. MuJoCo (so101_mujoco)
shoved the prop 7.7 mm and reported "did not settle"; on the mock stack the
call reported ok.

Rule now: the TOP-DOWN pointing pose keeps the TCP AND the jaws (the
profile's open-jaw sweep box, which bounds the closed jaws) at least
`safety.min_clearance_m` (0.03) above the object's top, checked on the SOLVED
pose. A hover below that is raised when the top-down ceiling allows it and a
fallback rung below it is never tried. When no top-down pose clears (the
ceiling is below the object top + margin, or every solve lands inside it),
point_at leans in instead or refuses before any arm or jaw command: the
SO-101 profiles and tall objects (tests/test_point_at_lean_in.py). Profiles
whose hover already cleared command exactly what they did before.

Real `build_runtime` wiring (LazyArm -> SafeArm -> SafetyHarness of each
profile) over a MockArm that records every arm and jaw command.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin

from cascade.control.kinematics import IKResult
from cascade.control.mock_arm import MockArm

#: safety.min_clearance_m in configs/demo.yaml (no arm profile overrides it).
#: A literal on purpose: the test pins the shipped margin.
MARGIN = 0.03
#: mock / mock_small synthetic box: 5 cm tall, on the table
MOCK_TOP = 0.05
LEAN_TEXT = "cannot point at 'red cube' on this arm: "


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
    assert "nothing was moved" in out["error"], out


def _no_lean_in(monkeypatch, runtime):
    """Block the lean-in family (every solve fails) to test the top-down rule alone."""
    def fail(position, approach, q0, *a, **k):
        return IKResult(q=np.asarray(q0, float).copy(), success=False, error=1.0, iterations=0)

    monkeypatch.setattr(runtime.kin, "ik_approach", fail)


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
def test_a_tall_object_above_the_ceiling_leans_in_without_a_top_down_solve(rig, monkeypatch):
    """mock ceiling = topdown_z_max 0.15 - 0.01 = 0.14 m; a 12 cm top needs the
    TCP at 0.15 m. Before #369: the TCP hovered at 0.14, 2 cm over the top;
    #369 refused; now no top-down rung is tried and the arm leans in."""
    r = rig("mock")
    _with_top(monkeypatch, r.runtime, 0.12)
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True and out["lean_deg"] == 15, out
    assert asked == [], "no top-down solve: the ceiling is below the top + margin"


@needs_pin
def test_an_object_out_of_reach_either_way_is_refused_before_any_command(rig, monkeypatch):
    """A 60 cm top: above the top-down ceiling, and no lean-in pose reaches
    it. The refusal names both."""
    r = rig("mock")
    _with_top(monkeypatch, r.runtime, 0.60)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert (LEAN_TEXT + "from above, its top-down ceiling is 0.140 m, below the object top + margin "
            "(0.630 m: top 0.600 m + clearance 0.030 m + 0.000 m of jaw below the TCP); leaning in 15-45 deg, "
            "no pose keeps the TCP, the jaws and the arm 0.030 m from it, every known object and the table") \
        in out["error"], out
    assert out["error"].count("nothing was moved") == 1, out


@needs_pin
def test_a_fallback_rung_inside_the_margin_is_never_tried(rig, monkeypatch):
    """Top 0.10 m: the hover rung (0.14) clears, the fallback rung (0.12) does
    not. With the hover unreachable (and the lean-in blocked) the call
    refuses; before #369 it pointed from 0.12 m, 2 cm over the top."""
    r = rig("mock")
    _with_top(monkeypatch, r.runtime, 0.10)

    def ceiling(T, q0, real):     # nothing solves above 0.135 m
        if T[2, 3] > 0.135:
            return IKResult(q=np.asarray(q0, float).copy(), success=False, error=1.0, iterations=0)
        return real(T, q0)

    asked = _record_ik(monkeypatch, r.runtime, ceiling)
    _no_lean_in(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert asked and min(asked) >= 0.10 + MARGIN, asked
    assert LEAN_TEXT + "from above, no pose solved; leaning in" in out["error"], out


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
    would execute decides, before any command. With the lean-in blocked the
    refusal names the highest of them."""
    r = rig("mock")

    def low(T, q0, real):
        T = np.array(T, dtype=float)
        T[2, 3] -= 0.05 if T[2, 3] < 0.1 else 0.09
        return real(T, q0)

    asked = _record_ik(monkeypatch, r.runtime, low)
    _no_lean_in(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    _refused_untouched(r, out)
    assert sorted(set(asked)) == [0.09, 0.12], asked      # both rungs tried, every yaw
    assert (LEAN_TEXT + "from above, the best solved pose put the closed jaws at 0.040 m, less than "
            "0.030 m above its top (0.050 m); leaning in") in out["error"], out


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


# ── strict CUDA perception: the fix's cloud is a device tensor ─────────────


class _DeviceCloud:
    """A CUDA tensor's host contract, without needing a GPU: numpy refuses it
    with torch's own message, `.detach().cpu().numpy()` copies it."""

    def __init__(self, values):
        self._host = np.asarray(values, dtype=float)

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the "
                        "tensor to host memory first.")

    def detach(self):
        return self

    def cpu(self):
        return SimpleNamespace(numpy=lambda: self._host.copy())

    def __len__(self):
        return len(self._host)

    @property
    def shape(self):
        return self._host.shape


def _cuda_cloud(values):
    """A real CUDA tensor when this host has one, else the stand-in."""
    try:
        import torch
    except ImportError:
        return _DeviceCloud(values)
    if not torch.cuda.is_available():
        return _DeviceCloud(values)
    return torch.as_tensor(np.asarray(values, dtype=float), device="cuda")


def test_host_fix_copies_device_geometry_and_keeps_a_host_fix():
    from cascade.skills.runtime import _host_fix
    from cascade.types import Detection, ObjectFix

    pts = np.array([[0.3, 0.0, 0.05], [0.31, 0.01, 0.04]])
    host = ObjectFix(label="lemon", position=np.array([0.3, 0.0, 0.03]), points=pts,
                     detection=Detection(label="lemon", conf=1.0, bbox=np.zeros(4)),
                     extent=np.array([0.05, 0.04, 0.04]), axes=np.eye(3))
    assert _host_fix(host) is host

    moved = _host_fix(dataclasses.replace(host, points=_DeviceCloud(pts)))
    assert isinstance(moved.points, np.ndarray)
    np.testing.assert_array_equal(moved.points, pts)
    assert moved.position is host.position and moved.label == "lemon"


@needs_pin
def test_point_at_works_when_strict_cuda_perception_returns_a_device_cloud(rig, monkeypatch):
    """Live kitchen (isaac_kitchen_gpu, CASCADE_REQUIRE_CUDA=1): `_localize`
    returns the cloud as a CUDA tensor and point_at failed before any command
    with "bad arguments for point_at: can't convert cuda:0 device type tensor
    to numpy". It now points exactly as with a host cloud."""
    r = rig("isaac_kitchen")
    orig = r.runtime._localize

    def localize_on_cuda(label, *a, **k):
        frame, fix = orig(label, *a, **k)
        return frame, dataclasses.replace(fix, points=_cuda_cloud(fix.points))

    monkeypatch.setattr(r.runtime, "_localize", localize_on_cuda)
    asked = _record_ik(monkeypatch, r.runtime)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert asked[0] == pytest.approx(MOCK_TOP + 0.04, abs=1e-6), asked
    assert r.runtime.kin.fk(_commands(r)[-1])[2, 3] == pytest.approx(MOCK_TOP + 0.04, abs=1e-3)
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


@needs_pin
def test_point_at_works_when_a_remembered_neighbour_cloud_is_on_cuda(rig, monkeypatch):
    """Strict CUDA mode also remembers every object's cloud on the GPU
    (cuda_math.remember_cloud). Live, with the other kitchen objects
    remembered, point_at failed the same way in the neighbours' footprints
    after the target's own fix had been copied. A CUDA-cloud neighbour now
    keeps its margin like a host one."""
    r = rig("isaac_kitchen")
    cube = r.runtime.beliefs.find("red cube")
    xy = np.asarray(cube.position, dtype=float)[:2] + [0.0, 0.15]
    rng = np.random.default_rng(0)
    pts = np.column_stack([xy[0] + rng.uniform(-0.02, 0.02, 200), xy[1] + rng.uniform(-0.02, 0.02, 200),
                           rng.uniform(0.0, 0.06, 200)])
    r.runtime.beliefs.update("bottle", np.array([xy[0], xy[1], 0.03]), 0.9, extent=np.array([0.06, 0.04, 0.04]),
                             top_z=0.06, color="blue", points=pts)
    bottle = r.runtime.beliefs.find("bottle")
    bottle.points = _cuda_cloud(bottle.points)
    orig = r.runtime._localize

    def localize_on_cuda(label, *a, **k):
        frame, fix = orig(label, *a, **k)
        return frame, dataclasses.replace(fix, points=_cuda_cloud(fix.points))

    monkeypatch.setattr(r.runtime, "_localize", localize_on_cuda)

    out = r.runtime.execute("point_at", {"label": "red cube"})

    assert out["ok"] is True, out
    assert min(_lowest_tool_z(r, q) for q in _commands(r)) >= MOCK_TOP + MARGIN


def test_belief_footprint_reads_a_device_cloud_like_a_host_one():
    from cascade.skills.runtime import _belief_footprint

    pts = np.array([[0.30, 0.10, 0.0], [0.34, 0.12, 0.06], [0.32, 0.08, 0.03]])
    host = SimpleNamespace(label="bottle", color="blue", position=np.array([0.32, 0.10, 0.03]),
                           extent=np.array([0.06, 0.04, 0.04]), points=pts, top_z=0.06)
    device = SimpleNamespace(**{**vars(host), "points": _DeviceCloud(pts)})
    poly, top = _belief_footprint(host)
    poly_d, top_d = _belief_footprint(device)
    np.testing.assert_array_equal(poly, poly_d)
    assert top == top_d == 0.06
