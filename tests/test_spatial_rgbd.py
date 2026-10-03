"""CPU retained captures and real TCP/MCP consumers; no native camera proof."""
from dataclasses import replace
import json

import numpy as np
import pytest

from cascade.sensing import BufferedSensorProvider, SensorHub, SensorError
from cascade.sensing.hub import SensorDescriptor
from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload
from cascade.spatial.rgbd import RgbdSpatialDomain
from test_sensing_rgbd import endpoint as endpoint, calibration
from cascade.sim.mobile_rgbd import calibration_record


def capture(**changes):
    t = np.eye(4)
    t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    t[:3, 3] = [1, 2, 3]
    depth = np.full((3, 4), 2, dtype='<f4')
    depth[0, 0] = 0
    payload = RgbdPayload(MeasurementMetadata('camera', 'c'*64), 4, 3, b'\x10'*36,
        depth.tobytes(), (2, .25, 1, 0, 4, 1, 0, 0, 1), tuple(t.flatten()), 'world')
    return ObservationEnvelope(**(dict(source='camera-service', sensor_id='overview', epoch='one',
        sequence=7, clock_domain='simulation', capture_time_s=.35, received_monotonic_s=10.,
        producer_age_s=.1, model_identity_sha256='e'*64, measurement_kind='physics', payload=payload) | changes))


@pytest.fixture
def rig():
    now = [10.1]
    descriptor = SensorDescriptor('overview', 'duck', 'camera-service', 'rgbd', 'camera',
        'simulation', 'physics', model_identity_sha256='e'*64, calibration_id='c'*64, max_age_s=.5,
        read_timeout_s=1.)
    hub = SensorHub(clock=lambda: now[0], max_history=2)
    provider = BufferedSensorProvider(descriptor)
    hub.register(provider)
    domain = RgbdSpatialDomain('space', 'duck', hub, sensor_domain='sensing', sensor_id='overview',
        map_id='room', world_frame_id='world', capacity=2, clock=lambda: now[0])
    value = capture()
    provider.publish(value)
    assert hub.read('overview') is value
    args = dict(epoch='one', sequence=7, capture_sha256=value.sha256, pixel=[3, 2],
                observation_id='pixel-a', label='cup?')
    try:
        yield now, hub, provider, domain, value, args
    finally:
        domain.close()
        assert hub.close(2)['ok']


def test_same_capture_fans_out_to_two_consumers_without_read_or_rejuvenation(rig):
    now, hub, provider, domain, obs, args = rig
    before = hub._slots['overview'].watermark
    def forbidden():
        pytest.fail('projection must not acquire another capture')
    provider.read = forbidden
    first = domain.execute('annotate_pixel', args)
    assert first['ok'], first
    second = RgbdSpatialDomain('other', 'duck', hub, sensor_domain='sensing', sensor_id='overview',
        map_id='other-map', world_frame_id='world', clock=lambda: now[0])
    now[0] = 10.2
    try:
        result = second.execute('annotate_pixel', args)
        assert result['ok']
        assert first['result']['point_map_m'] == pytest.approx([.5, 3.9375, 5.])
        assert result['result']['point_map_m'] == first['result']['point_map_m']
        assert first['capture_age_s'] == pytest.approx(.2)
        assert result['capture_age_s'] == pytest.approx(.3)
        assert hub.history() == (obs,) and hub._slots['overview'].watermark == before
        assert hub.retained('overview', **{k: args[k] for k in ('epoch', 'sequence', 'capture_sha256')}) is obs
        entry = result['result']
        assert entry['stamp']['time_s'] == obs.capture_time_s
        assert entry['confidence'] is None and entry['provenance']['uncertainty'] == 'not_estimated'
        assert entry['provenance']['label_kind'] == 'caller_annotation'
        assert entry['provenance']['geometry'] == 'observed_surface_point'
        edge = entry['transform']['samples'][0]
        assert edge['position_error_m'] is None and edge['angular_error_rad'] is None
        assert entry['provenance']['model_identity_sha256'] == obs.model_identity_sha256
    finally:
        second.close()


@pytest.mark.parametrize('change,reason', [
    ({'pixel': [True, 0]}, 'pixel coordinate'), ({'pixel': [1., 1]}, 'pixel coordinate'),
    ({'pixel': [-1, 0]}, 'pixel coordinate'), ({'pixel': [4, 1]}, 'outside'),
    ({'pixel': [0, 0]}, 'zero depth'), ({'pixel': [1]}, 'integer'),
    ({'epoch': 'foreign'}, 'not retained'), ({'sequence': 8}, 'not retained'),
    ({'capture_sha256': 'f'*64}, 'SHA256'), ({'label': ''}, 'label'),
    ({'observation_id': '../bad'}, 'observation_id'), ({'time_s': .35}, 'exact capture'),
])
def test_bad_pixels_or_capture_cannot_add_geometry(rig, change, reason):
    _, _, _, domain, _, args = rig
    result = domain.execute('annotate_pixel', args | change)
    assert not result['ok'] and reason in result['error']
    assert domain.memory is None


def test_retained_stale_future_evicted_closed_captures_fail_and_watermark_stays(rig):
    now, hub, provider, domain, obs, args = rig
    watermark = hub._slots['overview'].watermark
    now[0] = 10.6
    assert 'stale' in domain.execute('annotate_pixel', args)['error']
    assert hub._slots['overview'].watermark == watermark
    provider.publish(replace(obs, received_monotonic_s=10.6, producer_age_s=0))
    with pytest.raises(SensorError, match='replay'):
        hub.read('overview')
    now[0] = 9.9
    assert 'future receipt' in domain.execute('annotate_pixel', args)['error']
    now[0] = 10.2
    for seq in (8, 9):
        provider.publish(replace(obs, sequence=seq, capture_time_s=seq*.05))
        hub.read('overview')
    assert 'not retained' in domain.execute('annotate_pixel', args)['error']
    hub.close()
    assert 'closed' in domain.execute('annotate_pixel', args)['error']


@pytest.mark.parametrize('payload_change,reason', [
    ({'world_from_camera': None, 'world_frame_id': None}, 'calibrated'),
    ({'world_frame_id': 'foreign'}, 'world frame'),
    ({'metadata': MeasurementMetadata('camera', 'c'*64, True)}, 'calibrated'),
    ({'intrinsics': (1, 1, 1, 1, 1, 1, 0, 0, 1)}, 'Singular'),
])
def test_missing_or_invalid_capture_geometry_is_not_substituted(rig, payload_change, reason):
    _, hub, provider, domain, obs, args = rig
    value = replace(obs, sequence=8, capture_time_s=.4, payload=replace(obs.payload, **payload_change))
    provider.publish(value); hub.read('overview')
    result = domain.execute('annotate_pixel', args | dict(sequence=8, capture_sha256=value.sha256))
    assert not result['ok'] and reason in result['error']
    assert domain.memory is None


def test_mismatched_calibration_epoch_or_model_is_rejected_by_original_hub(rig):
    _, hub, provider, domain, obs, args = rig
    for change in ({'epoch': 'two'}, {'model_identity_sha256': 'f'*64},
                   {'payload': replace(obs.payload, metadata=MeasurementMetadata('camera', 'd'*64))}):
        provider.publish(replace(obs, sequence=8, capture_time_s=.4, **change))
        with pytest.raises(SensorError):
            hub.read('overview')
    assert hub.history() == (obs,)
    assert domain.execute('annotate_pixel', args)['ok']


def test_changed_geometry_under_same_calibration_is_not_used(rig):
    _, hub, provider, domain, obs, args = rig
    assert domain.execute('annotate_pixel', args)['ok']
    k = list(obs.payload.intrinsics); k[0] = 5
    value = replace(obs, sequence=8, capture_time_s=.4, payload=replace(obs.payload, intrinsics=k))
    provider.publish(value); hub.read('overview')
    result = domain.execute('annotate_pixel', args | dict(sequence=8, capture_sha256=value.sha256,
                                                       observation_id='next'))
    assert not result['ok'] and 'calibration' in result['error']


def test_two_retained_captures_can_be_consumed_in_reverse_order(rig):
    _, hub, provider, domain, obs, args = rig
    second = replace(obs, sequence=8, capture_time_s=.4)
    provider.publish(second); hub.read('overview')
    result = domain.execute('annotate_pixel', args | dict(sequence=8,
        capture_sha256=second.sha256, observation_id='newer'))
    assert result['ok'], result
    older = domain.execute('annotate_pixel', args)
    assert older['ok'], older
    assert older['result']['transform']['samples'][0]['stamp']['source_sha256'] == obs.sha256
    assert result['result']['transform']['samples'][0]['stamp']['source_sha256'] == second.sha256
    assert older['result']['stamp']['time_s'] == .35
    assert hub._slots['overview'].watermark[1:3] == (8, .4)


def test_bounded_memory_mutation_historical_age_and_stop_do_not_reauthorize(rig):
    now, hub, provider, domain, obs, args = rig
    first = domain.execute('annotate_pixel', args)
    assert first['ok']
    first['result']['provenance']['world_from_camera'][0] = 100
    first['result']['point_map_m'][0] = 100
    assert 'replay' in domain.execute('annotate_pixel', args)['error']
    assert domain.execute('annotate_pixel', args | {'observation_id': 'second'})['ok']
    assert 'full' in domain.execute('annotate_pixel', args | {'observation_id': 'third'})['error']
    domain.stop(); domain.reset_stop()
    assert 'replay' in domain.execute('annotate_pixel', args)['error']
    now[0] = 100.
    result = domain.execute('recall', {'epoch': 'one', 'label': 'cup?'})
    assert result['ok'] and result['historical']
    assert len(result['result']) == 2
    entry = result['result'][0]
    assert entry['point_map_m'] == pytest.approx([.5, 3.9375, 5.])
    assert entry['provenance']['world_from_camera'][0] == 0
    age = result['capture_ages'][entry['observation_id']]
    assert age['capture_age_s'] == pytest.approx(90.1)
    assert not age['within_sensor_age_bound'] and not entry['physical_admission']
    assert entry['stamp']['time_s'] == .35
    assert not domain.execute('recall', {'epoch': 'two', 'label': 'cup?'})['ok']
    provider.publish(obs)
    with pytest.raises(SensorError, match='replay'):
        hub.read('overview')


def test_recall_preserves_annotation_digest_and_bytes_at_every_query_time(rig):
    from cascade.spatial.frames import sha256
    now, _, _, domain, _, args = rig
    annotation = domain.execute('annotate_pixel', args)['result']
    original_bytes = json.dumps(annotation, sort_keys=True, separators=(',', ':'))
    for when, expected_age in ((10.2, .3), (100., 90.1)):
        now[0] = when
        result = domain.execute('recall', {'epoch': 'one', 'label': 'cup?'})
        assert result['ok']
        entry = result['result'][0]
        assert json.dumps(entry, sort_keys=True, separators=(',', ':')) == original_bytes
        assert sha256({k: v for k, v in entry.items() if k != 'sha256'}) == entry['sha256']
        assert result['capture_ages'][entry['observation_id']]['capture_age_s'] == pytest.approx(expected_age)
        assert entry['provenance']['received_monotonic_s'] == 10.
        assert entry['provenance']['producer_age_s'] == .1


def test_final_freshness_gate_before_memory_insertion(rig, monkeypatch):
    now, _, _, domain, _, args = rig
    solve = np.linalg.solve
    def aged(*a, **kw):
        result = solve(*a, **kw)
        now[0] = 11.
        return result
    monkeypatch.setattr(np.linalg, 'solve', aged)
    result = domain.execute('annotate_pixel', args)
    assert not result['ok'] and 'stale' in result['error']
    assert domain.memory is None


def config(p):
    from cascade.config import Cfg
    return Cfg(dict(robot_mode='composed', robot_id='duck', memory={}, domains={
        # Reverse declaration order deliberately: binding is explicit, not order-dependent.
        'spatial': dict(kind='spatial', robot_id='duck', rgbd=dict(sensor_domain='sensing',
            sensor_id='overview', map_id='room', world_frame_id='world', capacity=2)),
        'sensing': dict(kind='sensors', robot_id='duck', read_timeout_s=2., max_age_s=5., providers=[
            dict(id='overview', kind='mobile_rgbd', profile=p, camera='overview',
                 calibration_sha256=calibration_record(calibration())[1])])}))


def test_real_mcp_sensor_to_spatial_shares_actual_retained_tcp_capture(endpoint, monkeypatch, tmp_path):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.control.isaac_base import IsaacBase
    from cascade.control.lazy_arm import LazyArm
    c, bridge, _, p, _, depth, publish, operations = endpoint
    def forbidden(*a, **kw):
        pytest.fail('passive spatial catalog/consumer constructed an actuator')
    monkeypatch.setattr(IsaacBase, '__init__', forbidden)
    monkeypatch.setattr(LazyArm, '__init__', forbidden)
    monkeypatch.delenv('CASCADE_BASE', raising=False)
    monkeypatch.setenv('CASCADE_ROBOT', 'spatial_rgbd_fixture')
    monkeypatch.setenv('CASCADE_RUN_DIR', str(tmp_path))
    srv = McpSkillServer(); srv._mobile_config = config(p)
    def call(name, args):
        return json.loads(srv.call_tool(name, args)['content'][-1]['text'])
    try:
        names = {s['name'] for s in srv.list_tools()}
        assert {'sensing.read_sensor', 'spatial.annotate_pixel', 'spatial.recall'} <= names
        assert not any(n.startswith(('locomotion.', 'manipulation.')) for n in names)
        assert 'spatial.plan_route' not in names and not operations and srv._runtime is None
        catalog = call('list_resources', {})
        assert all(v['controller_id'] is None and v['writer_id'] is None for v in catalog['resources'])
        assert all(v['admission'] == 'unvalidated' for v in catalog['resources'])
        publish()
        captured = call('sensing.read_sensor', {'sensor_id': 'overview'})
        assert captured['ok'], captured
        count = len(operations)
        obs = captured['observation']
        args = dict(epoch=obs['epoch'], sequence=obs['sequence'], capture_sha256=captured['capture_sha256'],
                    pixel=[3, 2], observation_id='selected-pixel', label='requested-cup')
        result = call('spatial.annotate_pixel', args)
        assert result['ok'], result
        assert result['result']['point_map_m'] == pytest.approx([0, 0, float(depth[2, 3])])
        assert result['result']['provenance']['model_identity_sha256'] == p['model_identity_sha256']
        assert result['result']['provenance']['capture_sha256'] == captured['capture_sha256']
        assert result['result']['confidence'] is None
        assert len(operations) == count  # no socket request during spatial tools
        recalled = call('spatial.recall', {'epoch': obs['epoch'], 'label': 'requested-cup'})
        assert recalled['ok'] and recalled['historical'] and len(operations) == count
        assert 'replay' in call('sensing.read_sensor', {'sensor_id': 'overview'})['error']
        assert not bridge._owners and c.hello()['generation'] == 0
        assert {r['op'] for r in operations} == {'hello', 'frame'}
        rt = srv._runtime
        assert 'sensing/overview' in rt.tool_descriptors['spatial.annotate_pixel'].requires
    finally:
        srv.shutdown()


@pytest.mark.parametrize('change', [
    {'sensor_domain': 'missing'}, {'sensor_domain': 'spatial'}, {'sensor_id': 'missing'},
    {'capacity': 0}, {'capacity': 1025}, {'unknown': 'ignored'},
])
def test_profile_bad_reference_or_bound_fails_without_connections(monkeypatch, change):
    import socket
    from cascade.apps.robot_runtime import describe_robot
    from test_sensing_providers import profile
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **kw: pytest.fail('discovery connected'))
    cfg = config(profile())
    cfg._data['domains']['spatial']['rgbd'].update(change)
    with pytest.raises(ValueError):
        describe_robot(cfg)
