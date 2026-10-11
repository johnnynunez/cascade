"""Even with a computed trajectory, every step is checked against the CURRENT map.

The user's requirement after the 2026-10-11 rig incident: "Aunque tenga la
trayectoria calculada, siempre debe estar comprobando con las camaras!" --
even with the trajectory already computed, it must always keep checking with
the cameras. Pinned here, offline (MockArm on the reBot URDF, a synthetic
distance-grid bridge, no hardware):

(a) the WorldWatcher keeps refreshing the map with robot-masked depth WHILE
    the arm moves (belief fusion is paused, the map is not), and the mask is
    posed at the moving arm's state;
(b) every streamed step of every motion path (move_joints, move_planned,
    move_joint_path, move_cartesian) is approved against the map as it is at
    that step, not as it was when the route was planned: an obstacle that
    appears mid-motion stops the arm at the next step;
(c) a required map that goes stale or fails mid-motion stops the arm with a
    SafetyViolation that says why; it holds the last commanded pose.
"""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, depth_frame, make_map

from cascade.control.mock_arm import MockArm
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import build_tool_volume
from cascade.types import SafetyViolation

T_CAM = np.array([[0.0, 0.0, -1.0, 1.2], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.3], [0, 0, 0, 1.0]])
TURN = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])        # base yaw: the TCP sweeps sideways


class PacedArm(MockArm):
    """MockArm whose stream takes real time (`period_s` per command) and that
    calls `on_step(i)` after the i-th command (1-based)."""

    def __init__(self, cfg, kin, period_s=0.0, on_step=None):
        super().__init__(cfg, kin)
        self.period_s, self.on_step = period_s, on_step
        self.connect()

    def send_joint_target(self, q):
        super().send_joint_target(q)
        if self.period_s:
            time.sleep(self.period_s)
        if self.on_step is not None:
            self.on_step(len(self.commands))


@pytest.fixture
def rig():
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics

    cfg = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    a = cfg.arm
    kin = Kinematics(a.model, a.get("ee_frame", "gripper_end"), 6, joint_signs=a.get("joint_signs"))

    def build(*, period_s=0.0, on_step=None, **map_kw):
        bridge = GridBridge([TABLE], voxel=0.01)
        occ = make_map(bridge, unobserved="free", **map_kw)
        occ.refresh(depth_frame(), T_CAM)
        assert occ.usable()
        arm = PacedArm(a, kin, period_s=period_s, on_step=on_step)
        harness = SafetyHarness(SafetyLimits.from_config(cfg.safety), kinematics=kin, occupancy=occ,
                                tool_volume=build_tool_volume(cfg, a))
        return SimpleNamespace(bridge=bridge, occ=occ, arm=arm, harness=harness,
                               safe=SafeArm(arm, harness))

    return SimpleNamespace(cfg=cfg, kin=kin, home=np.asarray(a.home_q, dtype=float), build=build)


def obstacle_ahead(rig, h, q, gap=0.02):
    """A box 2 cm beyond the volume's leading edge (in y) at pose q: already
    inside the 3 cm margin, and the base turn closes in on it."""
    ahead = np.sign(rig.kin.fk(q + 0.01 * TURN)[1, 3] - rig.kin.fk(q)[1, 3])
    pts = h.tool_volume.points(rig.kin.fk(q))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if ahead > 0:
        y0, y1 = hi[1] + gap, hi[1] + gap + 0.05
    else:
        y0, y1 = lo[1] - gap - 0.05, lo[1] - gap
    return ((lo[0] - 0.02, y0, lo[2] - 0.02), (hi[0] + 0.02, y1, hi[2] + 0.02))


def _move(path, safe, start, goal, kin):
    if path == "move_joints":
        return safe.move_joints(goal, duration_s=1.0)
    if path == "move_planned":
        return safe.move_planned(goal, duration_s=1.0)
    if path == "move_joint_path":
        return safe.move_joint_path([start + 0.5 * (goal - start), goal], duration_s=1.0)
    if path == "move_cartesian":
        return safe.move_cartesian(kin.fk(goal), duration_s=1.0)
    raise AssertionError(path)


PATHS = ["move_joints", "move_planned", "move_joint_path", "move_cartesian"]


@needs_pin
@pytest.mark.parametrize("path", PATHS)
def test_premise_the_motion_completes_on_an_unchanged_map(rig, path):
    r = rig.build()
    assert _move(path, r.safe, rig.home, rig.home + TURN, rig.kin)
    np.testing.assert_allclose(r.arm.get_state().q, rig.home + TURN, atol=1e-3)


# ── (b) an obstacle that appears mid-motion stops the arm at the next step ──


@needs_pin
@pytest.mark.parametrize("path", PATHS)
def test_an_obstacle_that_appears_mid_motion_stops_the_arm_at_the_next_step(rig, path):
    k = 10
    state = {}

    def on_step(i):
        if i == k:
            # the camera sees it: the watcher's next refresh puts it in the map
            q = r.arm.get_state().q
            r.bridge.boxes.append(obstacle_ahead(rig, r.harness, q))
            r.occ.refresh(depth_frame(), T_CAM)
            state["q"] = q.copy()

    r = rig.build(on_step=on_step)
    with pytest.raises(SafetyViolation, match=r"(wrist camera|jaws) clearance .*\(occupancy map\)"):
        _move(path, r.safe, rig.home, rig.home + TURN, rig.kin)
    assert len(r.arm.commands) == k, "a command went out after the map showed the obstacle"
    np.testing.assert_allclose(r.arm.get_state().q, state["q"])       # holding the last pose


@needs_pin
def test_the_route_vetted_before_the_obstacle_does_not_authorize_the_rest(rig):
    """The whole route passed its preflight on the map of its start; the map
    that counts for each step is the one in force at that step."""
    from cascade.safety.trajectory import vet_segment

    k = 10
    r = rig.build(on_step=lambda i: i == k and (
        r.bridge.boxes.append(obstacle_ahead(rig, r.harness, r.arm.get_state().q)),
        r.occ.refresh(depth_frame(), T_CAM)))
    assert vet_segment(r.harness, rig.home, rig.home + TURN, 1.0) is None     # vetted, all clear
    with pytest.raises(SafetyViolation):
        r.safe.move_planned(rig.home + TURN, duration_s=1.0)
    assert len(r.arm.commands) == k


# ── (a) the map keeps refreshing, robot-masked, while the arm moves ─────────


class FakeStream:
    """A camera that delivers a new depth frame on every read."""

    name = "scene"

    def __init__(self):
        self.n = 0

    def latest(self):
        self.n += 1
        f = depth_frame()
        f.frame_id = self.n
        return f

    def set_overlay(self, **_):
        pass


@needs_pin
def test_the_watcher_refreshes_the_masked_map_while_the_arm_moves(rig):
    from cascade.perception.grounding import Extrinsics
    from cascade.perception.world import WatchedCamera, WorldWatcher

    r = rig.build(period_s=0.01)
    seen = []

    def mask(depth, K, T_base_cam):
        # the mesh mask's contract: posed at the arm's CURRENT state
        seen.append((time.monotonic(), r.arm.get_state().q.copy(), watcher.is_paused))
        return np.zeros(np.shape(depth), dtype=bool)

    r.occ.add_robot_body_mask(mask)
    cam = WatchedCamera(FakeStream(), SimpleNamespace(ensure_depth=lambda f: f),
                        Extrinsics(T=T_CAM), fuse=False, map_depth=True)
    detector = SimpleNamespace(detect=lambda frame, classes=None: [])
    watcher = WorldWatcher([cam], detector, SimpleNamespace(), rate_hz=20.0, harness=r.harness,
                           occupancy=r.occ)
    watcher.start()
    try:
        assert r.occ.wait_until_usable(2.0), r.occ.last_error   # first masked refresh
        t0 = time.monotonic()
        with watcher.paused():                       # what every motion skill does
            assert r.safe.move_joints(rig.home + TURN, duration_s=1.0)
        t1 = time.monotonic()
    finally:
        watcher.stop()
    during = [(q, paused) for t, q, paused in seen if t0 <= t <= t1]
    assert len(during) >= 8, f"only {len(during)} refreshes in {t1 - t0:.2f} s of motion"
    assert all(paused for _, paused in during)
    yaws = sorted({round(float(q[0]), 3) for q, _ in during})
    assert len(yaws) >= 5, "the body mask did not follow the moving arm"
    assert r.occ.last_error is None and r.occ.usable()


# ── (c) a stale or failed map mid-motion stops the arm ──────────────────────


@needs_pin
def test_a_required_map_that_goes_stale_mid_motion_stops_the_arm(rig):
    r = rig.build(period_s=0.02, max_age_s=0.3)       # nobody refreshes from here on
    with pytest.raises(SafetyViolation, match=r"required occupancy unavailable: distance grid is stale "
                                              r"\(0\.[3-9]\d s old, max_age_s 0\.3\)"):
        r.safe.move_joints(rig.home + TURN, duration_s=1.0)
    n = len(r.arm.commands)
    assert 0 < n < 50, n                               # it moved, then held
    np.testing.assert_allclose(r.arm.get_state().q, r.arm.commands[-1])


@needs_pin
def test_a_failed_refresh_mid_motion_stops_the_arm_at_the_next_step(rig):
    k = 10

    def on_step(i):
        if i == k:
            r.bridge.fail = "occupancy bridge at 127.0.0.1:5557 timed out (500 ms)"
            r.occ.refresh(depth_frame(), T_CAM)

    r = rig.build(on_step=on_step)
    with pytest.raises(SafetyViolation, match="refresh failed: .*timed out"):
        r.safe.move_joints(rig.home + TURN, duration_s=1.0)
    assert len(r.arm.commands) == k


@needs_pin
def test_an_optional_map_that_goes_stale_is_skipped_which_is_why_the_rebot_requires_it(rig):
    r = rig.build(period_s=0.02, required=False, max_age_s=0.3)
    assert r.safe.move_joints(rig.home + TURN, duration_s=1.0)


# ── the same through a skill on the real runtime wiring ─────────────────────


@needs_pin
def test_a_spatial_nudge_stops_when_an_obstacle_appears_beside_the_camera(demo_cfg, tmp_path, rig):
    """move_relative (vetted route, streamed through SafeArm by the skill
    runtime): the map changes after the 10th command, the 11th is refused."""
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        r = rig.build()
        harness = runtime.arm.harness
        harness.occupancy, harness.tool_volume = r.occ, r.harness.tool_volume
        n0, k = len(arm.commands), 10
        send = arm.send_joint_target

        def spy(q):
            send(q)
            if len(arm.commands) == n0 + k:
                pts = harness.tool_volume.points(harness.kin.fk(arm.get_state().q))
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                r.bridge.boxes.append(((lo[0] - 0.02, hi[1] + 0.02, lo[2] - 0.02),
                                       (hi[0] + 0.02, hi[1] + 0.07, hi[2] + 0.02)))
                r.occ.refresh(depth_frame(), T_CAM)

        arm.send_joint_target = spy
        out = runtime.execute("move_relative", {"direction": "left", "distance_m": 0.10})
        assert out["ok"] is False, out
        assert "occupancy map" in str(out.get("error", "")), out
        assert len(arm.commands) == n0 + k
    finally:
        shutdown_runtime(runtime, arm)
