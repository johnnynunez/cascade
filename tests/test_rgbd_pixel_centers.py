"""CPU projection of retained native depth samples, not a new rendering run."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cascade.sensing import BufferedSensorProvider, SensorHub
from cascade.sensing.hub import SensorDescriptor
from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload, wire
from cascade.sim.mobile_rgbd import calibration_record
from cascade.spatial.rgbd import RgbdSpatialDomain
from test_sensing_rgbd import calibration
from test_spatial_rgbd import rig as rig


def test_retained_rtx_ground_projects_at_explicit_raster_center():
    fixture = json.loads((Path(__file__).parent/'fixtures/rgbd-native-ground-20261003.json').read_text())
    c = fixture['calibration_v1_as_captured']
    _, derived_digest = calibration_record(c | {'version': 2, 'pixel_center_offset_uv': [.5, .5]})
    depth = np.zeros((c['height'], c['width']), dtype='<f4')
    for sample in fixture['samples']:
        u, v = sample['pixel']
        depth[v, u] = sample['depth_m']
    # Derived offline fixture, explicitly not fresh native physics/old-model admission.
    payload = RgbdPayload(MeasurementMetadata(c['frame_id'], derived_digest),
        c['width'], c['height'], b'\0'*(3*c['width']*c['height']), depth.tobytes(),
        c['intrinsics'], c['world_from_camera'], c['world_frame_id'], (.5, .5))
    obs = ObservationEnvelope('retained-fixture', 'overview', 'offline', 3,
        'fixture', .015, 10., 0., None, 'synthetic', payload)
    provider = BufferedSensorProvider(SensorDescriptor('overview', 'duck', obs.source,
        'rgbd', c['frame_id'], 'fixture', 'synthetic', calibration_id=derived_digest))
    hub = SensorHub(clock=lambda: 10.)
    hub.register(provider)
    domain = RgbdSpatialDomain('space', 'duck', hub, sensor_domain='sensing', sensor_id='overview',
        map_id='retained-ground', world_frame_id='world', clock=lambda: 10.)
    try:
        provider.publish(obs)
        hub.read('overview')
        for index, sample in enumerate(fixture['samples']):
            result = domain.execute('annotate_pixel', dict(epoch=obs.epoch, sequence=obs.sequence,
                capture_sha256=obs.sha256, pixel=sample['pixel'], observation_id=f'sample-{index}', label='ground?'))
            assert result['ok'], result
            assert abs(result['result']['point_map_m'][2]) < 1e-6
            assert result['result']['provenance']['pixel_center_offset_uv'] == [.5, .5]
        assert hub.history() == (obs,)
    finally:
        domain.close()
        assert hub.close(1)['ok']


@pytest.mark.parametrize('offset', [(0., 0.), (.5, .5)])
def test_payload_offset_is_immutable_wire_and_capture_identity(offset):
    from test_spatial_rgbd import capture
    old = capture()
    mutable = list(offset)
    new = replace(old, payload=replace(old.payload, pixel_center_offset_uv=mutable))
    mutable[0] = 100
    assert new.payload.pixel_center_offset_uv == offset
    assert wire(new.payload)['pixel_center_offset_uv'] == list(offset)
    assert new.sha256 != old.sha256
    assert 'pixel_center_offset_uv' not in wire(old.payload)


@pytest.mark.parametrize('bad', [(True, .5), (0, .5), (.1, .1), (.5,), [float('nan'), .5], 'center'])
def test_invalid_pixel_convention_is_rejected(bad):
    from test_spatial_rgbd import capture
    with pytest.raises(ValueError):
        replace(capture().payload, pixel_center_offset_uv=bad)


@pytest.mark.parametrize('change', [{'version': 1}, {'pixel_center_offset_uv': [0., 0.]},
                                  {'pixel_center_offset_uv': [True, .5]}, {'pixel_center_offset_uv': None}])
def test_native_calibration_refuses_old_or_mismatched_convention(change):
    with pytest.raises(ValueError):
        calibration_record(calibration() | change)


def test_v1_native_calibration_is_not_silently_upgraded():
    old = calibration()
    old['version'] = 1
    old.pop('pixel_center_offset_uv')
    with pytest.raises(ValueError, match='schema'):
        calibration_record(old)


def test_same_pin_cannot_switch_coordinate_convention(rig):
    now, hub, provider, domain, obs, args = rig
    assert domain.execute('annotate_pixel', args)['ok']
    changed = replace(obs, sequence=8, capture_time_s=.4,
                      payload=replace(obs.payload, pixel_center_offset_uv=(.5, .5)))
    provider.publish(changed)
    hub.read('overview')
    result = domain.execute('annotate_pixel', args | {'sequence': 8, 'capture_sha256': changed.sha256,
                                                     'observation_id': 'changed'})
    assert not result['ok'] and 'calibration' in result['error']


@pytest.mark.parametrize('offset,expected', [((0., 0.), [.5, 3.9375, 5.]),
                                           ((.5, .5), [.25, 4.40625, 5.])])
def test_generic_camera_keeps_its_declared_coordinate_convention(rig, offset, expected):
    now, hub, provider, domain, obs, args = rig
    changed = replace(obs, sequence=8, capture_time_s=.4,
                      payload=replace(obs.payload, pixel_center_offset_uv=offset))
    provider.publish(changed)
    hub.read('overview')
    result = domain.execute('annotate_pixel', args | {'sequence': 8, 'capture_sha256': changed.sha256})
    assert result['ok'], result
    assert result['result']['point_map_m'] == pytest.approx(expected)
