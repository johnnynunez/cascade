"""Complete solved support is shared internally, never through mutable replies."""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest
from test_microduck_shared_scene import contact, layout_fixture, shared
from test_microduck_stepper import SoftwareBackend
from test_mobile_bridge import control, publish

from cascade.control.mobile_support import SolvedContact, SupportObservation
from cascade.sim.microduck_shared import bind_scene
from cascade.sim.microduck_shared_native import SharedKitNewtonBackend, SharedRobotView


def raw_support(layout, *, step=2, time=SoftwareBackend.dt * 2):
    # Native decoder order/types, including contacts on every other robot.
    rows = []
    for binding in layout.robots:
        for point in range(16):
            c = contact(layout, 0, layout.shape_labels.index(binding.foot_shapes[point % 2]))
            rows.append(dict(shape_a=c['shape_a'], shape_b=c['shape_b'],
                shape_a_id=c['shape_a_id'], shape_b_id=c['shape_b_id'],
                force_on_b_world_n=c['force_on_b_world_n'], normal_force_n=c['normal_force_n'],
                point_world_m=[point * .001, 0., 0.], normal_a_to_b_world=c['normal_a_to_b_world']))
    return dict(version=1, status='known', reason='', step=step, sim_time_s=time, contacts=rows)


def owner_fixture(monkeypatch, count):
    import cascade.sim.microduck_contact_support as support_module
    import cascade.sim.microduck_shared_native as native

    owner = SharedKitNewtonBackend.__new__(SharedKitNewtonBackend)
    owner.layout = owner._support_layout = bind_scene(**layout_fixture(count))
    owner._bound_identity, owner._shared_read = True, None
    owner.ns = NS(simulation_step_count=2, sim_time=SoftwareBackend.dt * 2)
    owner._last_support_solve = owner.physics_clock
    owner._guard = lambda: None
    owner.receipt = {'support_extraction': {'source_admitted': True}}
    owner.admission = {'limits': {'max_contacts': 512, 'max_constraints': 2400}}
    owner._solver_graph = NS(telemetry=lambda: {'synthetic': True})
    calls = []

    def decode(ns, *, last_solved_clock, source_admitted):
        calls.append((last_solved_clock, source_admitted))
        raw = raw_support(owner.layout, step=ns.simulation_step_count, time=float(ns.sim_time))
        if not source_admitted or last_solved_clock != owner.physics_clock:
            raw.update(status='unavailable', reason='unadmitted or unsolved', contacts=[])
        return raw

    def capture(ns, *, robots, **_):
        sample = SoftwareBackend().read()
        sample.update(step=ns.simulation_step_count, sim_time=float(ns.sim_time))
        return {robot: copy.deepcopy(sample) for robot in robots}

    monkeypatch.setattr(support_module, 'read_support', decode)
    monkeypatch.setattr(native, '_read_native_states', capture)
    return owner, calls


def encoded(value):
    def default(item):
        if type(item) is SupportObservation:
            return item.as_observation_dict()
        return item.tolist()
    return json.dumps(value, default=default, allow_nan=False)


@pytest.mark.parametrize('count', [1, 2, 12])
def test_once_per_solve_full_contacts_and_exact_public_replies(monkeypatch, count):
    owner, reads = owner_fixture(monkeypatch, count)
    _, _, steppers = shared(count)
    parsed = []
    original = SolvedContact.__post_init__

    def checked(row):
        parsed.append(row)
        original(row)

    monkeypatch.setattr(SolvedContact, '__post_init__', checked)
    views = [SharedRobotView(owner, b, s.controller.hello()['epoch'])
             for b, s in zip(owner.layout.robots, steppers)]
    samples = [view.read() for view in views]
    assert len(parsed) == 16 * count and len(reads) == 1
    for view, sample, stepper in zip(views, samples, steppers):
        assert type(sample['support']) is SupportObservation
        assert all(a is b for a, b in zip(sample['support'].contacts, samples[0]['support'].contacts))
        assert len(sample['support'].contacts) == count * 16
        stepper.controller.publish(sample | {'q': sample['q'].tolist(), 'dq': sample['dq'].tolist(),
                                            'fallen': False, 'balance_active': True})
        expected = stepper.controller.state()
        raw = owner.read_robot(view.binding.robot_id)
        raw.update(robot_id=view.binding.robot_id, epoch=view.epoch,
                   model_identity_sha256=view.binding.model_identity_sha256)
        raw['support'].update(epoch=view.epoch, model_identity_sha256=view.binding.model_identity_sha256)
        assert encoded(sample) == encoded(raw)
        assert expected['support'] == raw['support']
        assert isinstance(expected['support']['contacts'][0]['point_world_m'], list)
        assert isinstance(expected['state']['support']['contacts'][0]['point_world_m'], tuple)
        expected['support']['contacts'][0]['point_world_m'][0] = 999.
        expected['state']['support']['contacts'][0]['force_on_b_world_n'] = (0., 0., 999.)
        raw['q'][0] = 999.
        assert stepper.controller.state()['support'] == sample['support'].as_observation_dict()
        assert view.read()['q'][0] != 999.
    fleet_reply = owner.read_robots()
    fleet_reply[views[0].binding.robot_id]['support']['contacts'][0]['point_world_m'][0] = 999.
    assert views[-1].read()['support'].contacts[0].point_world_m[0] == 0.
    assert len(parsed) == 16 * count and len(reads) == 1
    owner.ns.simulation_step_count += 1
    owner.ns.sim_time += SoftwareBackend.dt
    owner._last_support_solve = owner.physics_clock
    new = views[0].read()
    assert new['support'].step == 3 and new['support'].contacts[0] is not samples[0]['support'].contacts[0]
    assert len(parsed) == 32 * count and len(reads) == 2


def test_captured_input_detached_and_complete_shape_map_checked():
    layout = bind_scene(**layout_fixture(12))
    raw = raw_support(layout)
    captured = layout.capture_support(raw, clock=(raw['step'], raw['sim_time_s']))
    raw['contacts'][-1]['force_on_b_world_n'][2] = 999.
    assert captured.observation.contacts[-1].force_on_b_world_n == (0., 0., 3.)
    assert copy.deepcopy(captured) is captured
    bad_contact = replace(captured.observation.contacts[-1], shape_b='/World/Unbound')
    with pytest.raises(ValueError, match='shape'):
        replace(captured, observation=replace(captured.observation, contacts=(bad_contact,)))


@pytest.mark.parametrize('bad', ['shape_id', 'shape_name', 'normal', 'force', 'clock',
                                 'epoch', 'model', 'schema', 'unavailable_partial', 'boolean_step'])
def test_bad_global_observation_rejected_before_any_robot_can_use_it(bad):
    layout = bind_scene(**layout_fixture(12))
    raw = raw_support(layout)
    clock = raw['step'], raw['sim_time_s']
    if bad == 'shape_id': raw['contacts'][-1]['shape_b_id'] = len(layout.shape_labels)
    elif bad == 'shape_name': raw['contacts'][-1]['shape_b'] = '/World/Foreign'
    elif bad == 'normal': raw['contacts'][-1]['normal_a_to_b_world'] = [0., 0., 2.]
    elif bad == 'force': raw['contacts'][-1]['force_on_b_world_n'][2] = float('nan')
    elif bad == 'clock': raw['sim_time_s'] += .005
    elif bad == 'epoch': raw['epoch'] = 'foreign'
    elif bad == 'model': raw['model_identity_sha256'] = 'f' * 64
    elif bad == 'schema': raw['unexpected'] = True
    elif bad == 'unavailable_partial': raw.update(status='unavailable', reason='missing contacts')
    else: raw['step'] = True
    with pytest.raises(ValueError): layout.capture_support(raw, clock=clock)


@pytest.mark.parametrize('bad', ['layout', 'robot', 'clock', 'model'])
def test_bound_snapshot_cannot_cross_layout_robot_model_or_clock(bad):
    layout = bind_scene(**layout_fixture(2))
    raw = raw_support(layout)
    clock = raw['step'], raw['sim_time_s']
    captured = layout.capture_support(raw, clock=clock)
    binding = layout.robots[0]
    if bad == 'layout': layout = replace(layout)
    elif bad == 'robot': binding = replace(binding, robot_id='foreign')
    elif bad == 'clock': clock = (clock[0] + 1, clock[1])
    else:
        with pytest.raises(ValueError):
            replace(captured, observation=replace(captured.observation, model_identity_sha256='f' * 64))
        return
    with pytest.raises(ValueError): layout.robot_support(captured, binding, epoch='one', clock=clock)


def test_owner_cache_does_not_hide_layout_change_or_new_unsolved_clock(monkeypatch):
    owner, reads = owner_fixture(monkeypatch, 2)
    owner.read_robots()
    original = owner.layout
    owner.layout = replace(original)
    with pytest.raises(RuntimeError, match='layout changed'): owner.read_robot('duck0')
    owner.layout = original
    owner.ns.simulation_step_count += 1
    owner.ns.sim_time += SoftwareBackend.dt
    sample = owner.read_robot('duck0')
    assert sample['support']['status'] == 'unavailable' and not sample['support']['contacts']
    assert len(reads) == 2
    owner._last_support_solve = owner.physics_clock
    assert owner.read_robot('duck1')['support']['status'] == 'known'
    assert len(reads) == 3
    owner.receipt['support_extraction']['source_admitted'] = False
    assert owner.read_robot('duck0')['support']['status'] == 'unavailable'
    assert len(reads) == 4
    owner.receipt['support_extraction']['source_admitted'] = 1
    with pytest.raises(ValueError, match='boolean'): owner.read_robots()


@pytest.mark.parametrize('field,value', [('epoch', 'foreign'), ('model_identity_sha256', 'f' * 64),
                                       ('step', 1), ('sim_time_s', .1)])
def test_typed_support_still_checks_controller_provenance_and_completed_clock(monkeypatch, field, value):
    owner, _ = owner_fixture(monkeypatch, 1)
    _, _, steppers = shared(1)
    controller = steppers[0].controller
    view = SharedRobotView(owner, owner.layout.robots[0], controller.hello()['epoch'])
    sample = view.read()
    sample.update(q=sample['q'].tolist(), dq=sample['dq'].tolist(), fallen=False)
    controller.publish(sample)
    prior = controller.state()['state']
    owner.ns.simulation_step_count += 1
    owner.ns.sim_time += SoftwareBackend.dt
    owner._last_support_solve = owner.physics_clock
    sample = view.read()
    sample.update(q=sample['q'].tolist(), dq=sample['dq'].tolist(), fallen=False)
    with pytest.raises(ValueError):
        controller.publish(sample | {'support': replace(sample['support'], **{field: value})})
    failed = controller.state()['state']
    assert failed['support'] == prior['support'] and failed['latched']
    assert failed['controller_status'] == 'fault'


@pytest.mark.parametrize('field', ['q_indices', 'dof_indices', 'free_q_indices', 'free_dof_indices',
    'body_indices', 'shape_indices', 'robot_shapes', 'foot_shapes', 'ground_shapes'])
def test_mutable_binding_cannot_enter_shared_snapshot(field):
    layout = bind_scene(**layout_fixture(1))
    binding = replace(layout.robots[0], **{field: list(getattr(layout.robots[0], field))})
    malformed = replace(layout, robots=(binding,))
    raw = raw_support(layout)
    with pytest.raises(ValueError, match='immutable'):
        malformed.capture_support(raw, clock=(raw['step'], raw['sim_time_s']))


def test_mutable_contact_subclass_cannot_enter_frozen_schema():
    class MutableContact(SolvedContact):
        pass
    layout = bind_scene(**layout_fixture(1))
    raw = raw_support(layout)
    raw['contacts'][0] = MutableContact(**raw['contacts'][0])
    with pytest.raises(ValueError, match='exact'):
        layout.capture_support(raw, clock=(raw['step'], raw['sim_time_s']))


@pytest.mark.parametrize('field', ['reason', 'status', 'epoch', 'model_identity_sha256', 'shape_a', 'shape_b'])
def test_string_subclass_attributes_are_detached_in_typed_and_legacy_paths(control, field):
    from mobile_support_fixture import support
    from cascade.control.mobile_support import immutable_support
    from cascade.sim.mobile_bridge import _copy_observation

    class MutableString(str):
        pass

    controller, _ = control
    raw = support(1, .005) | {'epoch': controller.hello()['epoch'], 'model_identity_sha256': 'e' * 64}
    container = raw['contacts'][0] if field.startswith('shape_') else raw
    value = MutableString(container[field])
    value.mutable = []
    container[field] = value
    typed = SupportObservation.from_dict(raw)
    assert not immutable_support(typed)
    copied = _copy_observation({'support': typed})['support']
    assert copied is not typed
    wire = typed.as_observation_dict()
    published = []
    # The raw dictionary route must retain its existing scalar types/copies too.
    for observed in (raw, typed):
        controller.begin_epoch()
        controller._epoch = raw['epoch']
        publish(controller, support=observed)
        reply = controller.state()
        field_value = reply['support']['contacts'][0][field] if field.startswith('shape_') else reply['support'][field]
        assert type(field_value) is MutableString and field_value.mutable == []
        field_value.mutable.append('reply')
        fresh = controller.state()['support']
        fresh_value = fresh['contacts'][0][field] if field.startswith('shape_') else fresh[field]
        assert fresh_value.mutable == []
        published.append(controller._state['support'])
    value.mutable.append('input')
    for observed in (copied, wire, *published):
        result = observed.as_observation_dict() if type(observed) is SupportObservation else observed
        field_value = result['contacts'][0][field] if field.startswith('shape_') else result[field]
        assert field_value.mutable == []


def test_nonplain_observation_cannot_be_shared_even_when_schema_is_valid():
    class MutableString(str):
        pass
    layout = bind_scene(**layout_fixture(1))
    raw = raw_support(layout)
    raw['reason'] = MutableString('ok')
    raw['reason'].mutable = []
    with pytest.raises(ValueError, match='immutable'):
        layout.capture_support(raw, clock=(raw['step'], raw['sim_time_s']))


def test_shared_support_is_rebound_without_revalidating_every_contact(monkeypatch):
    """Twelve robots bind one decoded contact set per step: metadata only.

    ``dataclasses.replace`` re-ran ``SupportObservation.__post_init__`` (and its
    walk over every solved contact) 24 times per step. The decode was validated
    when it was captured and is immutable, so binding checks only the two
    identifiers; every reader still gets its own record with the shared tuple.
    """
    layout = bind_scene(**layout_fixture(12))
    raw = raw_support(layout)
    clock = (raw['step'], raw['sim_time_s'])
    captured = layout.capture_support(raw, clock=clock)
    revalidated = []
    original = SupportObservation.__post_init__

    def counted(record):
        revalidated.append(record)
        original(record)

    monkeypatch.setattr(SupportObservation, '__post_init__', counted)
    bound = [layout.robot_support(captured, b, epoch='fixture', clock=clock) for b in layout.robots]
    assert revalidated == []
    for binding, observation in zip(layout.robots, bound):
        assert type(observation) is SupportObservation and observation is not captured.observation
        assert observation.contacts is captured.observation.contacts
        assert observation == replace(captured.observation, epoch='fixture',
                                      model_identity_sha256=binding.model_identity_sha256)
    assert len({o.model_identity_sha256 for o in bound}) == 12
    assert len(revalidated) == 12  # the twelve ``replace`` reference records above
    # The identifiers themselves stay validated on the fast path.
    for epoch, model in (('', 'a' * 64), (' padded', 'a' * 64), ('ok', 'A' * 64), ('ok', 'abc')):
        with pytest.raises(ValueError):
            SupportObservation.rebound(captured.observation, epoch=epoch, model_identity_sha256=model)


def test_rebinding_a_nonplain_observation_keeps_full_revalidation(monkeypatch):
    class MutableString(str):
        pass
    layout = bind_scene(**layout_fixture(1))
    raw = raw_support(layout)
    reason = MutableString('ok')
    reason.mutable = []
    legacy = SupportObservation.from_dict(dict(raw, reason=reason, epoch='legacy',
                                               model_identity_sha256=layout.robots[0].scene_model_sha256))
    revalidated = []
    original = SupportObservation.__post_init__

    def counted(record):
        revalidated.append(record)
        original(record)

    monkeypatch.setattr(SupportObservation, '__post_init__', counted)
    rebound = SupportObservation.rebound(legacy, epoch='fixture', model_identity_sha256='b' * 64)
    assert len(revalidated) == 1 and rebound.epoch == 'fixture' and rebound.reason == 'ok'
    assert type(rebound.reason) is MutableString  # replace semantics, not a laundering copy


def test_binding_model_identity_digest_is_computed_once_and_follows_the_fields(monkeypatch):
    import cascade.sim.microduck_shared as shared_module
    layout = bind_scene(**layout_fixture(2))
    binding = layout.robots[0]
    from dataclasses import asdict
    import hashlib
    expected = hashlib.sha256(json.dumps(asdict(binding), sort_keys=True, separators=(',', ':'),
                                         allow_nan=False).encode()).hexdigest()
    dumps = []
    real = json.dumps

    def counted(value, *args, **kwargs):
        dumps.append(value)
        return real(value, *args, **kwargs)

    monkeypatch.setattr(shared_module.json, 'dumps', counted)
    fresh = replace(binding)  # no cache carried over from ``binding``
    assert fresh.model_identity_sha256 == expected
    assert fresh.model_identity_sha256 == expected and fresh.support_contract()['model_identity_sha256'] == expected
    assert len(dumps) == 1
    other = replace(fresh, robot_id=fresh.robot_id + '-b')
    assert other.model_identity_sha256 != expected and len(dumps) == 2
    assert copy.deepcopy(fresh).model_identity_sha256 == expected and len(dumps) == 2
    assert fresh == replace(binding) and asdict(fresh) == asdict(binding)  # the cache is not a field


def test_reader_polls_walk_the_shared_contacts_once_per_step_with_fresh_containers(monkeypatch):
    """Each robot is polled about once per step: the plain copy of the ~90 solved
    contacts is computed once per shared contact tuple, not per robot per poll,
    and every reply still gets its own containers."""
    import cascade.control.mobile_base as base_module
    count = 4
    owner, _ = owner_fixture(monkeypatch, count)
    _, _, steppers = shared(count)
    views = [SharedRobotView(owner, b, s.controller.hello()['epoch'])
             for b, s in zip(owner.layout.robots, steppers)]
    for view, stepper in zip(views, steppers):
        sample = view.read()
        stepper.controller.publish(sample | {'q': sample['q'].tolist(), 'dq': sample['dq'].tolist(),
                                            'fallen': False, 'balance_active': True})
    walked = []
    original = base_module._plain_record

    def counted(record, cls, nested=()):
        if cls is SolvedContact:
            walked.append(record)
        return original(record, cls, nested)

    monkeypatch.setattr(base_module, '_plain_record', counted)
    replies = [s.controller.state() for s in steppers] + [s.controller.state() for s in steppers]
    assert len(walked) == 16 * count  # one walk for 2 polls x count robots
    first, again = replies[0]['state']['support'], replies[count]['state']['support']
    assert first == again and first is not again
    assert first['contacts'] == again['contacts'] and first['contacts'][0] is not again['contacts'][0]
    assert isinstance(first['contacts'], tuple) and isinstance(first['contacts'][0]['point_world_m'], tuple)
    first['contacts'][0]['normal_force_n'] = 999.
    first['contacts'][0]['extra'] = True
    assert steppers[0].controller.state()['state']['support']['contacts'][0] == again['contacts'][0]
    assert replies[1]['state']['support']['contacts'][0]['normal_force_n'] != 999.
    assert len(walked) == 16 * count
    # A legacy (non-plain) record keeps asdict's full copy path and is never memoized.
    class MutableString(str):
        pass
    legacy = replace(views[0].read()['support'].contacts[0], shape_a=MutableString('/World/Ground'))
    assert base_module._plain_record(legacy, SolvedContact) is None
