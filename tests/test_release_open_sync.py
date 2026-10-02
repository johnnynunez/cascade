"""Opening feedback and RGBD can arrive at different physical steps.

The NV11 intermediate jaw values below are recorded measurements; the fake
transport supplies controlled clocks without SDK, actuator or mapper RPCs.
"""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from cascade.skills import release_episode as release
from cascade.skills import runtime as runtime_module
from cascade.types import SafetyViolation, SkillError
from test_release_episode import atomic_attachment, case, capture, retained


class Clock:
    def __init__(self):
        self.now = time.monotonic()

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def opening_case(monkeypatch):
    rt, ep, events = case()
    release.open_hand(rt, ep)
    events.clear()
    clock = Clock()
    monkeypatch.setattr(release, 'time', clock)
    return rt, ep, events, clock


def test_individual_feedback_waits_even_when_average_already_passes(monkeypatch):
    rt, ep, events, clock = opening_case(monkeypatch)
    values = iter(([.0485, .05], [.0488967478, .049060113], [.05, .04999724]))
    original = rt.arm.raw.get_state
    reads = []

    def state(**kwargs):
        result = original(**kwargs)
        result.gripper_joints['position_m'] = next(values, [.05, .04999724])
        reads.append(kwargs['timeout_s'])
        return atomic_attachment(result)

    rt.arm.raw.get_state = state
    start = clock.now
    release.wait_open(rt, ep, timeout_s=8.)
    assert reads == [1.] * 15 and clock.now-start == pytest.approx(.7)
    assert not ep['released'] and ep['barrier'] is None and not events


@pytest.mark.parametrize('late_open', [False, True])
def test_open_deadline_never_accepts_late_or_incomplete_feedback(monkeypatch, late_open):
    rt, ep, events, clock = opening_case(monkeypatch)
    original = rt.arm.raw.get_state
    reads = []

    def state(**kwargs):
        reads.append(kwargs['timeout_s'])
        result = original(**kwargs)
        result.gripper_joints['position_m'] = [.05, .05 if late_open else .0485]
        clock.sleep(8. if late_open else .2)
        return atomic_attachment(result)

    rt.arm.raw.get_state = state
    start = clock.now
    with pytest.raises(SkillError, match='opening|deadline'):
        release.wait_open(rt, ep, timeout_s=8.)
    assert clock.now-start <= 8.2 and all(0 < t <= 1 for t in reads)
    assert not ep['released'] and not events


@pytest.mark.parametrize('fault', ['epoch', 'limits', 'malformed', 'halt', 'timeout', 'same_step'])
def test_invalid_opening_feedback_is_terminal_without_poll_retry(monkeypatch, fault):
    rt, ep, events, clock = opening_case(monkeypatch)
    original = rt.arm.raw.get_state
    previous_clock = dict(ep['clock'])
    # begin() observed a later state than validate_simulation_clock().
    previous_clock.update(physics_step=rt.arm.raw.step, sim_time=rt.arm.raw.step/120.)
    calls = []

    def state(**kwargs):
        calls.append(kwargs)
        if fault == 'timeout':
            raise TimeoutError('native state request timed out')
        result = original(**kwargs)
        result.gripper_joints['position_m'] = [.04, .049]
        if fault == 'epoch': result.physics_clock['epoch'] = 'replacement'
        if fault == 'limits': result.gripper_joints['upper_m'] = [.06, .06]
        if fault == 'malformed': result.gripper_joints['position_m'][0] = float('nan')
        if fault == 'halt': rt.arm.harness.halt('while opening')
        if fault == 'same_step': result.physics_clock = previous_clock
        return atomic_attachment(result)

    rt.arm.raw.get_state = state
    with pytest.raises(SafetyViolation):
        release.wait_open(rt, ep, timeout_s=8.)
    assert len(calls) == 1 and not events and not ep['released']


@pytest.mark.parametrize('attached', [False, True])
def test_intermediate_capture_is_pending_and_never_a_map_floor(monkeypatch, attached):
    rt, ep, events, clock = opening_case(monkeypatch)
    first = capture('cam0', 10., attached=attached)
    first.capture['proprioception']['gripper_joints']['position_m'] = [
        .036803651601076126, .03710227459669113]
    ready = capture('cam0', 10.1)
    packets = iter((first, ready))
    calls = []

    def camera(*, after, timeout_s):
        assert not ep['released'] and ep['barrier'] is None
        calls.append((after, timeout_s))
        clock.sleep(.01)
        return next(packets)

    rt.camera.get_fresh_frame = camera
    evidence = release.wait_geometry(rt, ep)
    assert calls[0][0] is None and calls[1][0] is first
    assert 0 < calls[1][1] < calls[0][1]
    # Subtracting an absolute float deadline can round a few ULPs above 5 s.
    assert calls[0][1] == pytest.approx(5., rel=0., abs=1e-12)
    assert [f['t'] for f in evidence['floors']] == [10.1, 10., 10.]
    assert all(f['t'] > 10.1 for f in evidence['integrated'])
    assert evidence['contact_paths'] == [] and ep['released'] and not events


@pytest.mark.parametrize('fault', ['epoch', 'source', 'camera', 'contact', 'malformed_contact',
                                  'jaws', 'stamp', 'halt'])
def test_pending_capture_never_rebinds_identity_or_swallows_hard_error(monkeypatch, fault):
    rt, ep, events, clock = opening_case(monkeypatch)
    calls = []

    def camera(*, after, timeout_s):
        calls.append(after)
        packet = capture('cam0', 10. if after is None else 10.1)
        if after is None:
            packet.capture['proprioception']['gripper_joints']['position_m'] = [.04, .049]
            return packet
        assert len(ep['stream_capture_keys']) == 1
        if fault == 'epoch': packet.capture['proprioception']['producer_epoch'] = 'replacement'
        if fault == 'source': packet.capture['source'] = ['other-host', 1]
        if fault == 'camera': packet.capture['camera'] = 'side'
        if fault == 'contact': packet.capture['contact_paths'] = ['/World/other_prop']
        if fault == 'malformed_contact': packet.capture['contact_paths'] = None
        if fault == 'jaws': packet.capture['proprioception']['gripper_joints']['version'] = True
        if fault == 'stamp': return after  # New delivery is not a new producer capture.
        if fault == 'halt': rt.arm.harness.halt('during camera wait')
        return packet

    rt.camera.get_fresh_frame = camera
    rt.arm.harness.occupancy.wait_released_ready = lambda *a, **kw: pytest.fail('published a floor')
    with pytest.raises(SafetyViolation):
        release.wait_geometry(rt, ep)
    assert len(calls) == 2 and not ep['released'] and not events


@pytest.mark.parametrize('late_ready', [False, True])
def test_capture_deadline_never_publishes_partial_or_late_floors(monkeypatch, late_ready):
    rt, ep, events, clock = opening_case(monkeypatch)
    def camera(*, after, timeout_s):
        clock.sleep(timeout_s)
        packet = capture('cam0', 10.)
        if not late_ready:
            packet.capture['proprioception']['gripper_joints']['position_m'] = [.04, .05]
        return packet
    rt.camera.get_fresh_frame = camera
    rt.arm.harness.occupancy.wait_released_ready = lambda *a, **kw: pytest.fail('published a floor')
    with pytest.raises(SkillError, match='deadline'):
        release.wait_geometry(rt, ep)
    assert not ep['released'] and not events


@pytest.mark.parametrize('regression', ['jaws', 'attachment'])
def test_recovery_cannot_wait_through_opening_or_contact_regression(regression):
    rt, ep, events = retained()
    calls = []
    def camera(**kwargs):
        calls.append(kwargs)
        packet = capture('cam0', 12., attached=regression == 'attachment')
        if regression == 'jaws':
            packet.capture['proprioception']['gripper_joints']['position_m'] = [.04, .05]
        return packet
    rt.camera.get_fresh_frame = camera
    with pytest.raises(SafetyViolation, match='regressed'):
        release.recover(rt)
    assert len(calls) == 1 and not events and ep['barrier'] is None


@pytest.mark.parametrize('stability_timeout', [False, True])
def test_runtime_place_uses_individual_opening_before_capture_barrier(monkeypatch, stability_timeout):
    rt, ep, events = case()
    release.finish(rt, ep, completed=True)
    rt.cfg._data['safety'] = {'table_z': 0.}
    rt.cfg._data['grasp'].update(pregrasp_offset_m=.1, topdown_z_max=.8,
        post_place_retreat_offset_m=.1, close_settle_s=0., move_duration_s=2.,
        release_open_timeout_s=8.)
    rt.kin.ik = lambda pose, q: SimpleNamespace(q=pose[:3, 3].copy(), success=True)
    rt._held_det_label = 'green cube'
    rt._held_color = rt._held_support_offset_m = None
    rt._held_object_offset = lambda: np.zeros(3)
    rt.memory = SimpleNamespace(add=lambda *a: None)
    rt.beliefs = SimpleNamespace(update=lambda *a, **kw: None)
    rt._reconcile_held = lambda: None
    rt.skill_place_on_object = lambda destination: rt.skill_place_at(.2, .1, .3)
    flow = []
    rt.skill_move_home = lambda **kw: flow.append('home')
    rt.skill_reset_scene = lambda **kw: pytest.fail('release must not reset the scene')
    rt._gripper_width_frac = lambda: pytest.fail('average opening is not release evidence')
    clock = Clock()
    monkeypatch.setattr(release, 'time', clock)
    monkeypatch.setattr(runtime_module, 'time', clock)
    original = rt.arm.raw.get_state
    opened_reads = []
    pending_reads = []
    def state(**kwargs):
        result = original(**kwargs)
        if events and events[-1][0] == 'jaw':
            pending_reads.append(result.physics_clock['physics_step'])
            flow.append('observation')
            if len(opened_reads) < 2:
                opened_reads.append(True)
                result.gripper_joints['position_m'] = [.0485, .05]
            if stability_timeout:
                result.q[-1] += .0006*len(pending_reads)
        return atomic_attachment(result)
    rt.arm.raw.get_state = state
    for name in ('open_hand', 'wait_geometry', 'withdraw'):
        original_call = getattr(release, name)
        def recorded(*args, _call=original_call, _name=name, **kwargs):
            flow.append(_name)
            return _call(*args, **kwargs)
        monkeypatch.setattr(release, name, recorded)
    result = rt.skill_pick_and_place('green cube', destination='shelf')
    assert len(opened_reads) == 2 and len(set(pending_reads)) >= 3
    assert flow.count('open_hand') == 1 and result['place_attempts'] == 1
    if stability_timeout:
        assert not result['ok'] and result['home_skipped']
        assert result['return_home']['attempted'] is False
        assert 'deadline' in result['error']
        assert not any(x in flow for x in ('wait_geometry', 'withdraw', 'home'))
        assert [e[0] for e in events] == ['move', 'move', 'jaw']
        assert rt._release_episode is not None and not rt._release_episode['released']
    else:
        assert result['post_place_retreat']['ok'] and result['return_home']['ok']
        assert flow.count('wait_geometry') == flow.count('withdraw') == flow.count('home') == 1
        assert flow.index('open_hand') < flow.index('observation') < flow.index('wait_geometry')
        assert flow.index('wait_geometry') < flow.index('withdraw') < flow.index('home')
        assert [e[0] for e in events] == ['move', 'move', 'jaw', 'move']
        assert rt._release_episode is None
    assert rt.arm.harness._grasp_exempt is None
