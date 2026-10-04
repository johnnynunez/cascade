"""Moving capture geometry through the real TCP/Hub path; no native camera proof."""
import copy
from dataclasses import replace
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.sensing import MobileRgbdSensorProvider, SensorError, SensorHub
from cascade.sim.mobile_rgbd import MobileRgbdReader, RgbdFrameCache, calibration_record
from cascade.spatial.rgbd import RgbdSpatialDomain
from test_sensing_rgbd import calibration, endpoint as endpoint
from test_microduck_stepper import render_times
from test_mobile_identity import recipe_inputs as recipe_inputs


def mount_calibration():
    value = calibration()
    value.pop('world_from_camera')
    mount = np.eye(4); mount[:3, 3] = [.2, 0., .1]
    return value | dict(version=3, rig_frame_id='base', rig_from_camera=list(mount.flat),
                        mount_position_error_m=.01, mount_angular_error_rad=.02)


@pytest.fixture
def moving(endpoint):
    controller, server, _, profile, rgb, depth, _, operations = endpoint
    cal = mount_calibration()
    cache = RgbdFrameCache(controller.hello(), calibration=cal, max_pixels=1000, max_jpeg_bytes=10000)
    server._frame = cache
    depth[:] = 2
    def pose(step=10, matrix=None, **changes):
        return dict(epoch=profile['epoch'], step=step, sim_time_s=step*.005,
            model_identity_sha256=profile['model_identity_sha256'], world_frame_id='world',
            world_from_rig=list(np.eye(4).flat) if matrix is None else list(matrix.flat),
            position_error_m=None, angular_error_rad=None, render_reference=render_times(step*.005)) | changes
    def publish(value=None, *, step=10, age=0., calibration=None):
        refs = render_times(step*.005)
        cache.publish(rgb, depth_m=depth, calibration=cal if calibration is None else calibration,
            step=step, sim_time_s=step*.005, captured_at=time.monotonic()-age, render_times=refs,
            rgbd_render_times={'rgb': refs, 'depth': refs}, capture_pose=pose(step) if value is None else value)
    provider = MobileRgbdSensorProvider('overview', profile, 'overview',
        calibration_sha256=calibration_record(cal)[1], max_age_s=5., read_timeout_s=2.)
    hub = SensorHub(max_history=2); hub.register(provider)
    domain = RgbdSpatialDomain('space', 'duck', hub, sensor_domain='sensors', sensor_id='overview',
                              map_id='room', world_frame_id='world')
    try:
        yield SimpleNamespace(**locals())
    finally:
        domain.close(); hub.close(2); cache.close()


def annotate(rig, obs, name):
    return rig.domain.execute('annotate_pixel', dict(epoch=obs.epoch, sequence=obs.sequence,
        capture_sha256=obs.sha256, pixel=[3, 2], observation_id=name, label='surface'))


def test_two_camera_poses_keep_one_mount_and_capture_specific_geometry(moving):
    r = moving
    r.publish(); first = r.hub.read('overview')
    matrix = np.eye(4); matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    matrix[:3, 3] = [1, 2, 0]
    r.publish(r.pose(11, matrix), step=11); second = r.hub.read('overview')
    assert first.payload.metadata.calibration_id == second.payload.metadata.calibration_id
    assert first.sha256 != second.sha256 and first.payload.world_from_camera != second.payload.world_from_camera
    operations = copy.deepcopy(r.operations)
    # Reverse consumption uses each retained capture, never the latest rig pose.
    newer, older = annotate(r, second, 'new'), annotate(r, first, 'old')
    assert newer['ok'] and older['ok'], (newer, older)
    assert older['result']['point_map_m'] == pytest.approx([.21, .01, 2.1])
    assert newer['result']['point_map_m'] == pytest.approx([.99, 2.21, 2.1])
    for result, obs in ((older, first), (newer, second)):
        sample = result['result']['transform']['samples'][0]
        assert not sample['static'] and sample['stamp']['source_sha256'] == obs.sha256
        assert sample['stamp']['time_s'] == obs.capture_time_s
        assert sample['position_error_m'] is sample['angular_error_rad'] is None
        assert not result['physical_admission']
    newer['result']['provenance']['capture_pose']['world_from_rig'][3] = 99
    assert second.payload.capture_pose.world_from_rig[3] == 1
    assert r.operations == operations and not r.server._owners
    assert r.controller.hello()['generation'] == 0


@pytest.mark.parametrize('change', [
    {'epoch': 'foreign'}, {'step': 11}, {'sim_time_s': .06}, {'model_identity_sha256': 'f'*64},
    {'world_frame_id': 'base'}, {'world_frame_id': 'camera:overview'}, {'world_frame_id': 'foreign-world'},
    {'world_from_rig': [0]*16}, {'position_error_m': -1}, {'angular_error_rad': math.pi+.01},
    {'render_reference': render_times(.06)}, {'world_frame_id': 'foreign-world'},
])
def test_producer_refuses_pose_from_another_capture_or_invalid_geometry(moving, change):
    with pytest.raises(ValueError):
        moving.publish(moving.pose(**change))
    assert moving.cache._frame is None


@pytest.mark.parametrize('change', [
    {'epoch': 'foreign'}, {'step': 11}, {'sim_time_s': .06}, {'model_identity_sha256': 'f'*64},
    {'render_reference': render_times(.06)},
])
def test_reader_rechecks_pose_binding_even_if_packet_bypasses_producer(moving, change):
    r = moving; r.publish()
    r.cache._frame['capture_pose'].update(change)
    with pytest.raises(SensorError):
        r.hub.read('overview')
    assert not r.hub.history()


def test_unknown_and_finite_uncertainty_are_distinct_and_include_mount_lever_arm(moving):
    r = moving; r.publish(r.pose(position_error_m=.03, angular_error_rad=.1))
    obs = r.hub.read('overview'); result = annotate(r, obs, 'bounded')
    assert result['ok'], result
    sample = result['result']['transform']['samples'][0]
    assert sample['position_error_m'] == pytest.approx(.04 + 2*math.sqrt(.05)*math.sin(.05))
    assert sample['angular_error_rad'] == pytest.approx(.12)
    # Depth/semantic uncertainty is still unestimated, despite supplied pose bounds.
    assert result['result']['confidence'] is None
    assert result['result']['provenance']['uncertainty'] == 'not_estimated'
    for change in ({'epoch': 'other'}, {'sequence': 12}, {'capture_time_s': .06},
                   {'clock_domain': 'other'}, {'model_identity_sha256': 'f'*64}):
        with pytest.raises(ValueError, match='pose identity/clock'):
            replace(obs, **change)
    with pytest.raises(ValueError, match='composition'):
        replace(obs.payload, world_from_camera=tuple(np.eye(4).flat))


def test_mount_changes_and_cross_schema_pose_injection_are_not_admitted(moving):
    r = moving; r.publish()
    original = copy.deepcopy(r.cache._frame)
    for cal in (mount_calibration() | {'rig_frame_id': 'new'},
                mount_calibration() | {'mount_position_error_m': None}, calibration()):
        with pytest.raises(ValueError, match='calibration changed'):
            r.publish(step=11, calibration=cal)
    assert r.cache._frame == original
    # The payload version cannot relabel a mounted capture as the static schema.
    r.cache._frame['version'] = 1
    with pytest.raises(SensorError):
        r.hub.read('overview')


def test_dynamic_payload_cannot_hide_a_changed_mount_under_same_calibration_pin(moving):
    from cascade.sensing import BufferedSensorProvider
    r = moving; r.publish(); first = r.hub.read('overview')
    assert annotate(r, first, 'first')['ok']
    matrix = list(first.payload.capture_pose.rig_from_camera); matrix[3] += .01
    pose = replace(first.payload.capture_pose, sequence=11, capture_time_s=.055, rig_from_camera=matrix)
    payload = replace(first.payload, capture_pose=pose, world_from_camera=pose.world_from_camera)
    newer = replace(first, sequence=11, capture_time_s=.055, payload=payload)
    provider = BufferedSensorProvider(r.provider.descriptor)
    hub = SensorHub(); hub.register(provider)
    domain = RgbdSpatialDomain('other', 'duck', hub, sensor_domain='sensors', sensor_id='overview',
                              map_id='room', world_frame_id='world')
    try:
        provider.publish(first); hub.read('overview')
        context = SimpleNamespace(domain=domain)
        assert annotate(context, first, 'first')['ok']
        provider.publish(newer); hub.read('overview')
        result = annotate(context, newer, 'changed')
        assert not result['ok'] and 'calibration' in result['error']
    finally:
        domain.close(); hub.close(2)


def test_static_cache_and_payload_keep_original_schema_and_refuse_dynamic_pose(endpoint):
    _, _, cache, _, _, _, publish, _ = endpoint
    publish()
    before = copy.deepcopy(cache._frame)
    # A legacy capture cannot gain the opt-in extension by adding a pose field.
    with pytest.raises(ValueError, match='static RGB-D'):
        refs = render_times(.055)
        cache.publish(np.zeros((4, 6, 3), np.uint8), depth_m=np.ones((4, 6), np.float32),
            calibration=calibration(), step=11, sim_time_s=.055, captured_at=time.monotonic(),
            render_times=refs, rgbd_render_times={'rgb': refs, 'depth': refs}, capture_pose={})
    assert cache._frame == before and before['version'] == 1 and 'capture_pose' not in before


def test_moving_pose_does_not_rejuvenate_stale_replayed_or_foreign_epoch_pixels(moving):
    r = moving; r.publish(age=6)
    reader = MobileRgbdReader(r.profile, 'overview', calibration_sha256=calibration_record(r.cal)[1], max_age_s=.5)
    try:
        assert reader() is None and 'stale' in reader.last_error
        r.cache._captured_at = time.monotonic()
        r.cache._frame['capture_pose']['world_from_rig'][3] = 5
        assert reader() is None and 'replay' in reader.last_error
        r.publish(step=11); assert reader() is not None, reader.last_error
        r.cache._frame['epoch'] = 'new'
        assert reader() is None and 'epoch' in reader.last_error
    finally:
        reader.close()


def test_rigid_mount_enters_model_identity_and_cannot_change_under_old_digest(recipe_inputs):
    from cascade.sim.mobile_identity import build_model_identity
    admission, native, paths = recipe_inputs
    cal, digest = calibration_record(mount_calibration())
    native['rgbd_camera'] = dict(calibration=cal, calibration_sha256=digest)
    first = build_model_identity(admission, native, **paths)['model_identity_sha256']
    cal['rig_from_camera'][3] += .1
    with pytest.raises(ValueError, match='calibration digest'):
        build_model_identity(admission, native, **paths)
    native['rgbd_camera']['calibration_sha256'] = calibration_record(cal)[1]
    assert build_model_identity(admission, native, **paths)['model_identity_sha256'] != first
