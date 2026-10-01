"""Only successful post-close multi-camera geometry authorizes a held lift."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.config import Cfg
from cascade.perception.occupancy import OccupancyError
from cascade.safety.harness import SafeArm, SafetyHarness, SafetyLimits
from cascade.skills import contact_episode
from cascade.types import SafetyViolation, SkillError
from test_occupancy_payload import PROP, frame, mapping


def capture(camera, stamp, *, attached=True, occluded=False):
    f = frame(attached, stamp)
    f.capture["camera"] = camera
    if occluded:
        f.payload_mask[:] = False
        f.robot_mask[:] = False
        f.prop_masks[PROP][:] = False
    return f


def wait(m, floors, *, timeout=.03, guard=lambda: None, expected=(PROP,)):
    return m.wait_payload_ready(floors, deadline=time.monotonic() + timeout,
                                guard=guard, expected_paths=expected)


def integrate(m, camera, stamp, **kwargs):
    m.refresh(capture(camera, stamp, **kwargs), np.eye(4))


def test_every_map_camera_must_commit_after_all_post_close_floors():
    m = mapping()
    floors = [capture("main", 10.), capture("side", 11.), capture("wrist", 12.)]
    for camera, stamp in (("main", 11.), ("side", 13.), ("wrist", 14.)):
        integrate(m, camera, stamp)
    with pytest.raises(OccupancyError, match="deadline"):
        wait(m, floors)
    integrate(m, "main", 15.)
    result = wait(m, floors)
    assert {r["camera"] for r in result["integrated"]} == {"main", "side", "wrist"}
    assert min(r["t"] for r in result["integrated"]) > result["shared_floor"] == 12.


@pytest.mark.parametrize("action,nth", [("clear", 1), ("integrate_depth", 1),
                                      ("integrate_depth", 2), ("query", 1)])
def test_transition_failure_publishes_no_capture_and_old_state_cannot_clear_fault(action, nth):
    m = mapping()
    integrate(m, "main", 1., attached=False)
    old_floors = m._prop_history_floor.copy()
    request, seen = m._client.request, []
    def failing(packet):
        if packet["action"] == action:
            seen.append(packet)
            if len(seen) == nth:
                raise OccupancyError("injected RPC deadline")
        return request(packet)
    m._client.request = failing
    integrate(m, "main", 3.)
    assert m._contact_paths == () and m._prop_history_floor == old_floors
    assert not m._integrated_captures and m._transition_pending == ((PROP,), 3.)
    count = len(m._client.requests)
    integrate(m, "side", 2., attached=False)
    assert len(m._client.requests) == count
    with pytest.raises(SafetyViolation, match="payload tracking"):
        m.clearance(np.zeros((1, 3)))
    m._client.request = request
    integrate(m, "main", 4.)
    assert m._contact_paths == (PROP,) and m._transition_pending is None
    assert m._body_error is None
    assert wait(m, [capture("main", 3.)])["contact_paths"] == [PROP]


@pytest.mark.parametrize("response", [{}, {"points": [[0, 0, np.nan]]},
    {"points": [], "grid": np.ones((2, 2)), "origin": [0, 0, 0], "voxel": .01},
    {"points": [], "grid": np.ones((2, 2, 2)), "origin": [0, 0, 0], "voxel": -1}])
def test_malformed_query_does_not_commit_depth_capture(response):
    m = mapping()
    request = m._client.request
    m._client.request = lambda packet: response if packet["action"] == "query" else request(packet)
    integrate(m, "main", 3.)
    assert m._body_error and m.last_error and not m._integrated_captures
    with pytest.raises(OccupancyError, match="deadline"):
        wait(m, [capture("main", 1.)])


@pytest.mark.parametrize("change", ["source", "robot", "clock"])
def test_changed_capture_identity_cannot_borrow_previous_valid_camera_commit(change):
    m = mapping()
    for cam in ("main", "side", "wrist"):
        integrate(m, cam, 3.)
    floors = [capture(cam, 1.) for cam in ("main", "side", "wrist")]
    bad = capture("side", 4.)
    if change == "source":
        bad.capture["source"] = ["different-producer", 2]
    elif change == "robot":
        bad.capture["proprioception"]["robot_id"] = "/Other"
    else:
        bad.capture["proprioception"]["time_source"] = "different_clock"
    m.refresh(bad, np.eye(4))
    integrate(m, "main", 5.)  # success elsewhere cannot repair the missing source
    with pytest.raises(OccupancyError, match="deadline"):
        wait(m, floors)
    integrate(m, "side", 6.)
    assert wait(m, floors)["contact_paths"] == [PROP]


def test_contact_transition_between_cameras_invalidates_prior_capture_set():
    m = mapping()
    integrate(m, "main", 3.)
    integrate(m, "side", 4., attached=False)
    integrate(m, "wrist", 5.)
    floors = [capture(cam, 1.) for cam in ("main", "side", "wrist")]
    with pytest.raises(OccupancyError, match="deadline"):
        wait(m, floors)
    integrate(m, "main", 6.)
    integrate(m, "side", 7.)
    assert wait(m, floors)["contact_paths"] == [PROP]


def test_occluded_camera_advances_map_evidence_without_erasing_measured_surface():
    m = mapping()
    integrate(m, "main", 1.)
    points = m._payload_samples["main"].copy()
    integrate(m, "main", 3., occluded=True)
    integrate(m, "side", 4., occluded=True)
    result = wait(m, [capture("main", 2.), capture("side", 2.)])
    assert {r["camera"] for r in result["integrated"]} == {"main", "side"}
    np.testing.assert_array_equal(m._payload_samples["main"], points)
    assert result["surfaces_by_camera"] == {"main": len(points)}


def test_completely_unobserved_attached_surface_never_passes():
    m = mapping()
    integrate(m, "main", 3., occluded=True)
    integrate(m, "side", 4., occluded=True)
    with pytest.raises(OccupancyError, match="no measured depth surface"):
        wait(m, [capture("main", 1.), capture("side", 1.)])


def test_wait_releases_refresh_lock_and_cannot_return_after_deadline():
    m = mapping()
    acquired = threading.Event()
    def producer():
        time.sleep(.01)
        integrate(m, "main", 3.)
        acquired.set()
    worker = threading.Thread(target=producer)
    worker.start()
    assert wait(m, [capture("main", 1.)], timeout=.3)["contact_paths"] == [PROP]
    worker.join()
    assert acquired.is_set()
    calls = []
    def slow_guard():
        calls.append(1)
        if len(calls) == 2:
            time.sleep(.03)
    with pytest.raises(OccupancyError, match="deadline"):
        wait(m, [capture("main", 1.)], timeout=.01, guard=slow_guard)


def episode_case():
    class Kin:
        joint_limits = (np.full(3, -3.), np.full(3, 3.))
        def fk(self, q):
            T = np.eye(4); T[:3, 3] = q
            return T
        def link_positions(self, q):
            return np.array([[0., 0., .3]])
    events = []
    kin = Kin()
    harness = SafetyHarness(SafetyLimits(workspace_min=np.full(3, -1.),
                           workspace_max=np.ones(3), watchdog_s=10.), kin)
    class Raw:
        q = np.array([.2, .1, .02])
        def get_state(self):
            return SimpleNamespace(q=self.q.copy())
        def set_gripper(self, *args):
            events.append(("gripper", args))
    raw = Raw()
    arm = SafeArm(raw, harness)
    m = mapping()
    harness.occupancy = m
    harness.allow_grasp_descent(raw.q[:2], radius_m=.07, z_min=-.06)
    cfg = Cfg({"arm": {"type": "isaac", "bridge_host": "test", "bridge_port": 1,
                       "bridge_robot_id": "/Robot"}, "grasp": {}})
    streams = [SimpleNamespace(name=name, get_fresh_frame=lambda name=name, **kw:
                              capture(name, 5.)) for name in ("main", "side", "wrist")]
    rt = SimpleNamespace(arm=arm, kin=kin, cfg=cfg,
                         watcher=SimpleNamespace(_cams=[SimpleNamespace(stream=s, maps_depth=True)
                                                        for s in streams]), camera=streams[0])
    ep = contact_episode.begin(rt, np.array([.2, .1, .12]))
    ep["expected_paths"] = (PROP,)
    ep["source_identity"] = (("test", 1), "/Robot", "physics_loop_monotonic")
    ep["camera_streams"] = tuple(streams)
    contact_episode.finish(rt, ep, completed=False)
    harness.clear_grasp_exemption()
    original = m.wait_payload_ready
    def committed(floors, **kwargs):
        assert harness._grasp_exempt is None
        for f in floors:
            integrate(m, f.capture["camera"], 6.)
        return original(floors, **kwargs)
    m.wait_payload_ready = committed
    def move(q, duration_s):
        harness.check_contact_episode()
        events.append(("retreat", q.copy(), harness._grasp_exempt))
        raw.q = q.copy()
        return True
    arm.move_planned = move
    return rt, ep, events


def test_recovery_uses_only_original_cylinder_and_pregrasp_without_jaw_commands():
    rt, ep, events = episode_case()
    result = contact_episode.recover(rt)
    assert result["retreated_to_original_pregrasp"]
    assert len(events) == 1 and events[0][0] == "retreat"
    np.testing.assert_array_equal(events[0][1], ep["q_pre"])
    np.testing.assert_array_equal(events[0][2][0], ep["cylinder"][0])
    assert events[0][2][1:] == ep["cylinder"][1:]
    assert rt.arm.harness._grasp_exempt is None
    assert rt._contact_episode is None and rt.arm.harness._pending_contact_episode is None


@pytest.mark.parametrize("failure", ["feedback", "identity", "halt", "estop", "camera", "unknown_contact"])
def test_recovery_failure_keeps_episode_and_jaws_without_motion(failure):
    rt, ep, events = episode_case()
    before = ep["q_failure"].copy()
    if failure == "feedback":
        rt.arm.raw.q[0] += .002
    elif failure == "identity":
        rt.cfg.arm._data["bridge_robot_id"] = "/Other"
    elif failure == "halt":
        rt.arm.harness.halt("new stop")
    elif failure == "estop":
        rt.arm.harness.estop("new stop")
    elif failure == "camera":
        def stalled(**kwargs):
            raise TimeoutError("frozen camera")
        rt.camera.get_fresh_frame = stalled
    else:
        ep["expected_paths"] = None
    with pytest.raises((SkillError, SafetyViolation)):
        contact_episode.recover(rt)
    assert not events and rt._contact_episode is ep
    np.testing.assert_array_equal(ep["q_failure"], before)
    assert rt.arm.harness._grasp_exempt is None
    with pytest.raises(SafetyViolation, match="unfinished contact"):
        rt.arm.set_gripper(1.)


def test_recovery_authority_is_thread_local_and_never_authorizes_gripper():
    rt, ep, events = episode_case()
    refusals = []
    real_move = rt.arm.move_planned
    def other_thread():
        for action in (lambda: rt.arm.harness.begin_motion(), lambda: rt.arm.set_gripper(1.)):
            try:
                action()
            except SafetyViolation as exc:
                refusals.append(str(exc))
    def move(q, duration_s):
        worker = threading.Thread(target=other_thread)
        worker.start(); worker.join(timeout=1.)
        assert not worker.is_alive()
        with pytest.raises(SafetyViolation, match="unfinished contact"):
            rt.arm.set_gripper(1.)
        return real_move(q, duration_s)
    rt.arm.move_planned = move
    contact_episode.recover(rt)
    assert len(refusals) == 2 and len(events) == 1


def test_halt_between_geometry_and_begin_motion_cannot_be_cleared():
    rt, ep, events = episode_case()
    rt.arm.harness._contact_scope.value = (ep, False)
    rt.arm.harness.halt("after geometry, before lift")
    with pytest.raises(SafetyViolation, match="halt received"):
        rt.arm.harness.begin_motion()
    assert not events and rt.arm.harness.halted == "after geometry, before lift"


def test_halt_during_last_recovery_feedback_read_blocks_retreat():
    rt, ep, events = episode_case()
    original = rt.arm.raw.get_state
    reads = []
    def state():
        reads.append(1)
        if len(reads) == 2:
            rt.arm.harness.halt("between recovery guard and motion")
        return original()
    rt.arm.raw.get_state = state
    with pytest.raises(SafetyViolation, match="halt received"):
        contact_episode.recover(rt)
    assert not events and rt._contact_episode is ep


def test_failed_post_close_barrier_retains_provisional_and_never_requests_lift(monkeypatch):
    from test_grasp_contact_recovery import fixture
    from cascade.skills.runtime import SkillRuntime
    rt, run, moves, jaws, messages = fixture(monkeypatch, "post_close")
    m = mapping()
    integrate(m, "main", 1., attached=False)
    rt.arm.harness.occupancy = m
    def frozen(**kwargs):
        raise TimeoutError("no post-close capture")
    rt.camera = SimpleNamespace(get_fresh_frame=frozen)
    rt.watcher = None
    with pytest.raises(SkillError, match="post-close payload geometry unavailable"):
        run()
    assert len(moves) == 3 and len(jaws) == 2  # home, approach, descent; open, close
    ep = rt._contact_episode
    assert ep is rt.arm.harness._pending_contact_episode
    assert ep["q_failure"] is not None and rt._held_provisional[0] == "orange"
    SkillRuntime._reconcile_held(rt)
    assert rt._held_provisional[0] == "orange" and rt.held_object is None
    assert "explicit reset_scene" in SkillRuntime._grasp_retry_verdict(rt, "orange", 1, "failure")
    with pytest.raises(SafetyViolation, match="unfinished contact"):
        rt.arm.set_gripper(1.)
    with pytest.raises(SafetyViolation, match="unfinished contact"):
        rt.arm.move_joints(ep["q_pre"])
    assert len(moves) == 3 and len(jaws) == 2 and rt.arm.harness._grasp_exempt is None


def test_map_reset_cannot_rebind_retained_contact_to_another_producer():
    rt, ep, events = episode_case()
    rt.arm.harness.occupancy.begin_scene_reset()
    def changed(**kwargs):
        f = capture("main", 5.)
        f.capture["source"] = ["new-producer", 99]
        return f
    rt.camera.get_fresh_frame = changed
    with pytest.raises(SkillError, match="retained contact producer/robot/clock identity changed"):
        contact_episode.recover(rt)
    assert not events and ep["source_identity"][0] == ("test", 1)


def test_recreated_lazy_backend_is_not_the_retained_arm():
    from cascade.control.lazy_arm import LazyArm
    rt, ep, events = episode_case()
    lazy = LazyArm(lambda: None)
    lazy._arm = rt.arm.raw
    rt.arm._arm = lazy
    ep["raw"] = lazy
    lazy._arm = SimpleNamespace(get_state=lambda: SimpleNamespace(q=ep["q_failure"]))
    with pytest.raises(SafetyViolation, match="arm identity changed"):
        contact_episode.recover(rt)
    assert not events


def test_recovery_cannot_reduce_the_retained_map_camera_set():
    rt, ep, events = episode_case()
    rt.watcher._cams.pop()
    with pytest.raises(SkillError, match="map-depth camera set changed"):
        contact_episode.recover(rt)
    assert not events and rt._contact_episode is ep


@pytest.mark.parametrize("geometry", ["unknown", "19mm"])
def test_recovery_preserves_unknown_rejection_and_30mm_clearance(geometry):
    rt, ep, events = episode_case()
    occupancy = rt.arm.harness.occupancy
    ready = occupancy.wait_payload_ready
    def altered(floors, **kwargs):
        result = ready(floors, **kwargs)
        # Keep some finite support so the cache is a real ESDF; corrupt only
        # the candidate's local voxels for the independent motion gate.
        occupancy._grid[:] = np.inf if geometry == "unknown" else .019
        occupancy._grid[0, 0, 0] = .5
        # A wide held surface extends outside the original 70 mm cylinder;
        # its collision gate must still run during this contact retreat.
        occupancy._payload_samples["side"] = np.array([[.15, 0., .10]])
        return result
    occupancy.wait_payload_ready = altered
    rt.arm.move_planned = SafeArm.move_planned.__get__(rt.arm)
    rt.arm.raw.stream_to = lambda *a, **kw: events.append(("unexpected target",))
    with pytest.raises(SkillError, match="unobserved|clearance"):
        contact_episode.recover(rt)
    assert not events and rt._contact_episode is ep
    assert rt.arm.harness._grasp_exempt is None
