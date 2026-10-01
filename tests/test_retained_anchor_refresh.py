"""Measured anchor preservation, with synthetic mapping I/O and real barriers."""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError
from cascade.types import SafetyViolation
from test_occupancy_payload import PROP, frame, mapping
from test_home_routes import case as home_case
from test_reset_capture_freshness import runtime as bare_runtime
from cascade.perception.world import WatchedCamera, WorldWatcher
from cascade.types import SkillError

NAMES = ('cam0', 'side', 'proof')
CLOCK = {'source': ['test', 1], 'robot_id': '/Robot', 'epoch': 'epoch-one'}


def capture(name, stamp, *, attached=True, epoch='epoch-one'):
    f = frame(attached, stamp)
    f.capture['camera'] = name
    f.capture['proprioception']['producer_epoch'] = epoch
    return f


def ready():
    m = mapping()
    for i, name in enumerate(NAMES):
        m.refresh(capture(name, i+1., attached=False), np.eye(4))
    for name in NAMES:
        m.refresh(capture(name, 10.), np.eye(4))
    assert not m._body_error
    return m


def start(m):
    assert m.begin_capture_refresh(CLOCK)
    floors = [capture(name, 11.+i) for i, name in enumerate(NAMES)]
    m.finish_scene_reset(floors)
    return floors


def test_refresh_replays_measured_anchors_and_requires_every_new_commit():
    m = ready()
    anchors = {name: copy.deepcopy(rows[0]) for name, rows in m._depth_history.items()}
    floors = start(m)
    assert m.scene_reset_pending and m._grid is None and not m._integrated_captures
    for name in NAMES[:-1]:
        m.refresh(capture(name, 14.), np.eye(4))
        assert m.scene_reset_pending
        with pytest.raises(SafetyViolation):
            m.payload_clearance(np.zeros((1, 3)))
    m.refresh(capture('proof', 14.), np.eye(4))
    assert not m.scene_reset_pending and not m._body_error
    report = m.wait_payload_ready(floors, deadline=time.monotonic()+.1, guard=lambda: None, expected_paths=[PROP])
    assert len(report['integrated']) == 3 and m.last_replayed_frames == 6
    from test_nvblox_camera_recovery_probe import probe
    assert all(probe.retained_anchor_checks({'retained_anchor_refresh': m.last_capture_refresh}, set(NAMES)).values())
    for fault in ('anchor', 'epoch', 'missing_commit', 'old_commit'):
        broken = copy.deepcopy(m.last_capture_refresh)
        if fault == 'anchor':
            broken['anchors_after']['side']['t'] += 1.
        elif fault == 'epoch':
            broken['integrated'][0]['producer_epoch'] = 'replacement-process'
        elif fault == 'missing_commit':
            broken['integrated'].pop()
        else:
            broken['integrated'][0]['t'] = broken['shared_floor']
        assert not all(probe.retained_anchor_checks({'retained_anchor_refresh': broken}, set(NAMES)).values())
    for name in NAMES:
        assert m._depth_history[name][0]['t'] == anchors[name]['t']
        np.testing.assert_array_equal(m._depth_history[name][0]['depth'], anchors[name]['depth'])
        assert m._depth_history[name][0]['marker'] == anchors[name]['marker']


def test_true_scene_reset_discards_history_and_cannot_downgrade_to_refresh():
    m = ready()
    assert m.begin_scene_reset()
    assert not m._depth_history and not m._payload_samples
    with pytest.raises(OccupancyError, match='cannot be downgraded'):
        m.begin_capture_refresh(CLOCK)
    assert m.scene_reset_pending and not m._depth_history


@pytest.mark.parametrize('fault', ['contact', 'camera', 'epoch'])
def test_changed_fresh_capture_cannot_be_repaired_by_matching_later_frame(fault):
    m = ready()
    start(m)
    changed = capture('side', 14., attached=fault != 'contact', epoch='new' if fault == 'epoch' else 'epoch-one')
    if fault == 'camera':
        changed.capture['camera'] = 'replacement'
    if fault == 'camera':
        with pytest.raises(OccupancyError):
            m.refresh(changed, np.eye(4))
    else:
        m.refresh(changed, np.eye(4))
    for name in NAMES:
        m.refresh(capture(name, 15.), np.eye(4))
    assert m.scene_reset_pending and m._body_error
    with pytest.raises(SafetyViolation):
        m.payload_clearance(np.zeros((1, 3)))


def test_fresh_occluded_camera_keeps_its_previously_measured_payload_surface():
    m = ready()
    old = m._payload_samples['side'].copy()
    start(m)
    for name in NAMES:
        f = capture(name, 14.)
        if name == 'side':
            f.depth_m[f.payload_mask] = 0.
        m.refresh(f, np.eye(4))
    assert not m.scene_reset_pending
    np.testing.assert_array_equal(m._payload_samples['side'], old)


def test_refresh_keeps_prop_history_floor_that_excludes_discarded_old_poses():
    m = ready()
    m._prop_history_floor['/World_Props/discarded'] = 99.
    start(m)
    for name in NAMES:
        m.refresh(capture(name, 14.), np.eye(4))
    assert m._prop_history_floor['/World_Props/discarded'] == 99.


@pytest.mark.parametrize('fault', ['missing_marker', 'missing_epoch', 'wrong_epoch', 'wrong_source'])
def test_invalid_retained_anchor_never_replays(fault):
    m = ready()
    old = m._depth_history['side'][0]
    if fault == 'missing_marker':
        del old['marker']
    elif fault == 'missing_epoch':
        old['producer_epoch'] = None
    elif fault == 'wrong_epoch':
        old['producer_epoch'] = 'another-epoch'
    else:
        old['marker']['source'] = ['other', 1]
    count = len(m._client.requests)
    with pytest.raises(OccupancyError):
        m.begin_capture_refresh(CLOCK)
    assert len(m._client.requests) == count and m.scene_reset_pending
    assert m._grid is None


@pytest.mark.parametrize('fault', ['epoch', 'contact', 'camera_subset', 'camera_replace', 'source'])
def test_changed_floor_identity_latches_rejection(fault):
    m = ready()
    m.begin_capture_refresh(CLOCK)
    floors = [capture(name, 12.) for name in NAMES]
    if fault == 'epoch':
        floors[1].capture['proprioception']['producer_epoch'] = 'new'
    elif fault == 'contact':
        floors[1].capture['contact_paths'] = []
    elif fault == 'camera_subset':
        floors.pop()
    elif fault == 'camera_replace':
        floors[1].capture['camera'] = 'new'
    else:
        floors[1].capture['source'] = ['new', 1]
    with pytest.raises(OccupancyError):
        m.finish_scene_reset(floors)
    with pytest.raises(OccupancyError):
        m.begin_capture_refresh(CLOCK)
    assert m.scene_reset_pending and not m._integrated_captures


@pytest.mark.parametrize('action', ['clear', 'integrate_depth', 'query'])
def test_partial_mapper_failure_old_frame_cannot_publish_recovery(monkeypatch, action):
    m = ready()
    start(m)
    original = m._client.request
    failed = [False]
    def request(packet):
        if packet['action'] == action and not failed[0]:
            failed[0] = True
            raise OccupancyError('injected mapper failure')
        return original(packet)
    monkeypatch.setattr(m._client, 'request', request)
    m.refresh(capture('cam0', 14.), np.eye(4))
    assert failed[0] and m._body_error and m.scene_reset_pending
    count = len(m._client.requests)
    m.refresh(capture('side', 13.5), np.eye(4))
    assert len(m._client.requests) == count and m._body_error
    for name in NAMES:
        m.refresh(capture(name, 15.), np.eye(4))
    assert not m.scene_reset_pending and not m._body_error
    assert all(rows[0]['t'] < 10. for rows in m._depth_history.values())


def test_epoch_change_after_success_still_blocks_payload_clearance():
    m = ready()
    start(m)
    for name in NAMES:
        m.refresh(capture(name, 14.), np.eye(4))
    assert not m.scene_reset_pending
    m.refresh(capture('cam0', 15., epoch='new'), np.eye(4))
    with pytest.raises(SafetyViolation, match='epoch changed'):
        m.payload_clearance(np.zeros((1, 3)))
    m.refresh(capture('cam0', 16.), np.eye(4))
    with pytest.raises(SafetyViolation, match='epoch changed'):
        m.payload_clearance(np.zeros((1, 3)))


@pytest.mark.parametrize('when', ['before', 'feedback'])
@pytest.mark.parametrize('rate_hz', [30., 50.])
def test_recovery_home_retains_earlier_halt_generation(when, rate_hz):
    safe, raw, harness = home_case()
    generation = harness._halt_generation
    if when == 'before':
        harness.halt('after camera barrier')
    else:
        original = safe.get_state
        def state():
            result = original()
            harness.halt('during final feedback')
            return result
        safe.get_state = state
    with pytest.raises(SafetyViolation, match='halt'):
        safe.move_planned([.4, 0, .4], _halt_generation=generation, rate_hz=rate_hz)
    assert not raw.commands


def runtime_case():
    """Real runtime/watcher/map plumbing, only synthetic camera and mapper I/O."""
    m = ready()
    safe, raw, harness = home_case(occupancy=m)
    raw.validate_simulation_clock = lambda: copy.deepcopy(CLOCK)
    streams = []
    for name in NAMES:
        def fresh(*, after=None, timeout_s=5, name=name):
            return capture(name, 12. if after is None else 14.)
        streams.append(SimpleNamespace(name=name, get_fresh_frame=fresh))
    rt = bare_runtime(streams[0])
    rt._arm = safe
    rt.held_object = 'tomato can'
    extrinsics = SimpleNamespace(cam_to_base=lambda: np.eye(4))
    rt.watcher = WorldWatcher([WatchedCamera(stream, rt.depth, extrinsics) for stream in streams],
                              rt.detector, rt.beliefs, occupancy=m)
    return rt, m, raw, harness, streams


@pytest.mark.parametrize('fault', ['missing', 'raises', 'wrong_epoch'])
def test_initial_clock_failure_fences_previously_fresh_map_before_any_action(fault):
    rt, m, raw, _, _ = runtime_case()
    assert m._grid is not None and not m.is_stale()
    if fault == 'missing':
        raw.validate_simulation_clock = None
    elif fault == 'raises':
        def fail():
            raise SafetyViolation('clock unavailable')
        raw.validate_simulation_clock = fail
    else:
        raw.validate_simulation_clock = lambda: {**CLOCK, 'epoch': 'new-process'}
    with rt.watcher.paused(), pytest.raises((SkillError, SafetyViolation, OccupancyError)):
        rt._reset_camera_frames(require_geometry=True, scene_changed=False)
    assert rt._reset_observation_pending and m.scene_reset_pending
    assert m._grid is None and not raw.commands
    with pytest.raises(SafetyViolation):
        m.payload_clearance(np.zeros((1, 3)))


def test_runtime_frozen_retry_preserves_anchors_then_true_reset_discards_them():
    rt, m, raw, _, streams = runtime_case()
    original = streams[1].get_fresh_frame
    def frozen(**kwargs):
        raise TimeoutError('synthetic frozen side camera')
    streams[1].get_fresh_frame = frozen
    with rt.watcher.paused():
        with pytest.raises(TimeoutError):
            rt._reset_camera_frames(require_geometry=True, scene_changed=False)
        blocked = rt.skill_reset_scene()
        assert not blocked['ok'] and blocked['home_skipped'] and not raw.commands
        assert [rows[0]['t'] for rows in m._depth_history.values()] == [1., 2., 3.]
        streams[1].get_fresh_frame = original
        rt._reset_camera_frames(require_geometry=True, scene_changed=False)
        assert not m.scene_reset_pending and not rt._reset_observation_pending
        assert [rows[0]['t'] for rows in m._depth_history.values()] == [1., 2., 3.]
        rt._reset_camera_frames(scene_changed=True)
        assert all(rows[0]['t'] == 14. for rows in m._depth_history.values())


def test_runtime_true_reset_failure_cannot_be_retried_as_unchanged_scene():
    rt, m, _, _, streams = runtime_case()
    def frozen(**kwargs):
        raise TimeoutError('synthetic camera failure')
    streams[1].get_fresh_frame = frozen
    with rt.watcher.paused():
        for changed in (True, False):
            with pytest.raises(TimeoutError):
                rt._reset_camera_frames(scene_changed=changed)
            assert rt._reset_scene_changed and m._scene_reset_invalidated
            assert not m._depth_history


@pytest.mark.parametrize('when', ['last_clock', 'home', 'jaw'])
def test_late_halt_blocks_every_following_recovery_action(monkeypatch, when):
    rt, m, raw, harness, _ = runtime_case()
    calls = []
    def home(**kwargs):
        calls.append('home')
        assert kwargs['_halt_generation'] == 0
        if when == 'home':
            harness.halt('after home')
    def jaw(value, effort=1.):
        calls.append('jaw')
        if when == 'jaw':
            harness.halt('during jaw')
    def props():
        calls.append('reset_props')
        raise AssertionError('reset must not follow a halt')
    rt.skill_move_home = home
    raw.set_gripper, raw.reset_props = jaw, props
    if when == 'last_clock':
        readings = [0]
        def clock():
            readings[0] += 1
            if readings[0] == 2:
                harness.halt('during final clock validation')
            return copy.deepcopy(CLOCK)
        raw.validate_simulation_clock = clock
    # Avoid importing any simulator while testing the post-jaw cancellation.
    from types import ModuleType
    import cascade.sim
    fake_world = ModuleType('cascade.sim.mujoco_world')
    fake_world.live_worlds = lambda: []
    monkeypatch.setitem(__import__('sys').modules, fake_world.__name__, fake_world)
    monkeypatch.setattr(cascade.sim, 'mujoco_world', fake_world, raising=False)
    rt._reset_observation_pending, rt._reset_scene_changed = True, False
    rt._refresh_halt_generation = harness._halt_generation
    with rt.watcher.paused():
        result = rt.skill_reset_scene()
    assert not result['ok'] and 'halt' in result['recovery_error']
    assert calls == ([] if when == 'last_clock' else ['home'] if when == 'home' else ['home', 'jaw'])
    assert not raw.commands
