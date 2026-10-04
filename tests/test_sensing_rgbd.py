"""CPU captures and real loopback TCP/MCP routing; no native camera proof."""
import base64
import copy
import json
import socket
import time
import zlib

import numpy as np
import pytest

from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
from cascade.sim.mobile_rgbd import (MobileRgbdReader, RgbdFrameCache, calibration_record,
                                     read_static_calibration)
from cascade.sensing import MobileRgbdSensorProvider, SensorHub, SensorError
from mobile_support_fixture import support_contract
from test_microduck_stepper import render_times
from test_sensing_providers import profile
from test_mobile_identity import recipe_inputs as recipe_inputs


def calibration(width=6, height=4):
    return dict(version=2, camera='overview', frame_id='camera:overview', world_frame_id='world',
        width=width, height=height, intrinsics=[100., 0., width/2, 0., 100., height/2, 0., 0., 1.],
        world_from_camera=np.eye(4).flatten().tolist(), depth_convention='optical_z_m_zero_invalid',
        pixel_center_offset_uv=[.5, .5])


@pytest.fixture
def endpoint():
    c = MobileBridgeController(robot_id='duck', source='isolated-bridge', engine='newton', device='cuda:0',
        asset_sha256='a'*64, policy_sha256='b'*64, model_identity_sha256='e'*64,
        support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8,
        max_duration_s=5., lease_s=.3, max_state_age_s=5., max_action_wall_s=2.)
    cache = RgbdFrameCache(c.hello(), calibration=calibration(), max_pixels=1000, max_jpeg_bytes=10000)
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[:, :, 0] = 200  # unmistakably red in RGB, never a BGR round trip
    depth = np.arange(24, dtype=np.float32).reshape(4, 6)/100
    def publish(step=10, age=0):
        refs = render_times(step*.005)
        cache.publish(rgb, depth_m=depth, calibration=calibration(), step=step, sim_time_s=step*.005,
            captured_at=time.monotonic()-age, render_times=refs, rgbd_render_times={'rgb': refs, 'depth': refs})
    operations = []
    server = MobileBridgeServer(c, port=0, frame_callback=cache)
    dispatch = server.dispatch
    def recorded(request):
        operations.append({k: v for k, v in request.items() if not k.startswith('_')})
        return dispatch(request)
    server.dispatch = recorded
    server.start()
    p = profile(server.address[1])
    p['timeout_s'] = 1.
    p['epoch'] = c.hello()['epoch']
    try:
        yield c, server, cache, p, rgb, depth, publish, operations
    finally:
        server.close()
        cache.close()


def reader(profile, **kw):
    return MobileRgbdReader(profile, 'overview', calibration_sha256=calibration_record(calibration())[1], **kw)


def test_camera_only_mcp_retains_capture_and_never_constructs_arm(endpoint, monkeypatch, tmp_path):
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.config import Cfg
    from cascade.control.isaac_base import IsaacBase
    from cascade.control.lazy_arm import LazyArm
    c, bridge, cache, p, rgb, depth, publish, operations = endpoint
    def forbidden(*a, **kw):
        pytest.fail('sensor-only MCP constructed an actuator')
    monkeypatch.setattr(IsaacBase, '__init__', forbidden)
    monkeypatch.setattr(LazyArm, '__init__', forbidden)
    monkeypatch.delenv('CASCADE_BASE', raising=False)
    monkeypatch.setenv('CASCADE_ROBOT', 'camera_only_fixture')
    monkeypatch.setenv('CASCADE_RUN_DIR', str(tmp_path))
    srv = McpSkillServer()
    digest = calibration_record(calibration())[1]
    srv._mobile_config = Cfg(dict(robot_mode='composed', robot_id='duck', memory={},
        domains={'sensing': dict(kind='sensors', robot_id='duck', read_timeout_s=2., max_age_s=5., providers=[
            dict(id='overview', kind='mobile_rgbd', profile=p, camera='overview', calibration_sha256=digest)])}))
    def call(name, args):
        return json.loads(srv.call_tool(name, args)['content'][-1]['text'])
    try:
        names = {row['name'] for row in srv.list_tools()}
        assert 'sensing.read_sensor' in names
        assert not any(n.startswith(('manipulation.', 'locomotion.')) for n in names)
        catalog = call('list_resources', {})
        sensor = catalog['resources'][0]
        assert sensor['capabilities'] == ['rgbd'] and sensor['admission'] == 'unvalidated'
        assert sensor['controller_id'] is None and sensor['writer_id'] is None
        assert sensor['metadata']['calibration_id'] == digest
        assert not operations and srv._runtime is None
        publish()
        result = call('sensing.read_sensor', {'sensor_id': 'overview'})
        assert result['ok'], result
        obs = result['observation']
        assert (obs['sequence'], obs['capture_time_s'], obs['epoch']) == (10, .05, p['epoch'])
        assert obs['model_identity_sha256'] == p['model_identity_sha256']
        payload = obs['payload']
        assert base64.b64decode(payload['rgb8']['data']) == rgb.tobytes()
        assert base64.b64decode(payload['depth_m_f32le']['data']) == depth.astype('<f4').tobytes()
        assert payload['intrinsics'] == calibration()['intrinsics']
        assert payload['world_from_camera'] == calibration()['world_from_camera']
        assert payload['world_frame_id'] == 'world'
        assert payload['pixel_center_offset_uv'] == [.5, .5]
        assert 'capture_pose' not in payload
        assert 'replay' in call('sensing.read_sensor', {'sensor_id': 'overview'})['error']
        assert not bridge._owners and c.hello()['generation'] == 0
        assert {op['op'] for op in operations} == {'hello', 'frame'}
        assert all(op.get('role', 'reader') == 'reader' for op in operations)
    finally:
        srv.shutdown()


def test_legacy_rgb_reply_and_depth_copy_are_independent(endpoint):
    from cascade.sim.mobile_frames import FRAME_KEYS, MobileFrameReader
    _, _, cache, p, rgb, depth, publish, _ = endpoint
    publish()
    raw = cache({'camera': 'overview'})['frame']
    assert set(raw) == FRAME_KEYS
    native = reader(p, max_age_s=5)
    old = MobileFrameReader(p)
    try:
        value = native()
        assert value is not None, native.last_error
        exposed = value.metadata
        exposed['step'] = -1
        exposed['sim_time_s'] = -1
        exposed['calibration']['intrinsics'][0] = -1
        assert value.metadata['step'] == 10 and value.metadata['calibration']['intrinsics'][0] == 100
        with pytest.raises(TypeError):
            value._metadata['calibration']['intrinsics'][0] = -1
        assert native() is None and 'replay' in native.last_error
        rgb[:] = 0; depth[:] = 0
        assert value.rgb8 != rgb.tobytes() and value.depth_m_f32le != depth.tobytes()
        assert old('overview') is not None, old.last_error
    finally:
        native.close(); old.close()


@pytest.mark.parametrize('key,bad', [
    ('epoch', 'foreign'), ('model_identity_sha256', 'f'*64), ('robot_id', 'other'),
    ('camera', 'other'), ('source', 'other'), ('step', True), ('width', 100000),
    ('producer_age_s', 6.), ('calibration_sha256', 'f'*64), ('version', True),
    ('depth_m_f32le_z_b64', base64.b64encode(zlib.compress(b'x' * 100000)).decode()),
    ('depth_m_f32le_z_b64', base64.b64encode(zlib.compress(np.full(24, np.nan, '<f4').tobytes())).decode()),
])
def test_malformed_retained_capture_refused_before_sensor_admission(endpoint, key, bad):
    _, server, cache, p, _, _, publish, _ = endpoint
    publish()
    original = cache({'camera': 'overview', 'modality': 'rgbd'})
    original['rgbd'][key] = bad
    class BadCache:
        rgbd_enabled = True
        def __call__(self, req):
            return copy.deepcopy(original)
    server._frame = BadCache()
    provider = MobileRgbdSensorProvider('overview', p, 'overview',
        calibration_sha256=calibration_record(calibration())[1], max_age_s=5, read_timeout_s=2)
    hub = SensorHub()
    hub.register(provider)
    try:
        with pytest.raises(SensorError):
            hub.read('overview')
        assert not hub.history()
    finally:
        assert hub.close(2)['ok']


def test_stale_capture_cannot_be_restamped_and_new_epoch_requires_new_reader(endpoint):
    _, server, cache, p, _, _, publish, _ = endpoint
    publish(age=6)
    r = reader(p, max_age_s=5)
    try:
        assert r() is None and 'stale' in r.last_error
        cache._captured_at = time.monotonic()  # hostile software producer; no new pixels
        assert r() is None and 'replay' in r.last_error
        publish(11)
        assert r() is not None, r.last_error
        server.controller.epoch = 'changed'
        cache._frame['epoch'] = 'changed'
        assert r() is None and 'epoch' in r.last_error
    finally:
        r.close()


def test_catalog_construction_is_passive_and_calibration_pin_is_required(monkeypatch):
    from cascade.sensing import build_sensor_domain
    def forbidden(*a, **k):
        pytest.fail('passive metadata connected a socket')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    declaration = dict(kind='sensors', robot_id='duck', read_timeout_s=1., providers=[
        dict(id='overview', kind='mobile_rgbd', profile=profile(), camera='overview',
             calibration_sha256=calibration_record(calibration())[1])])
    domain = build_sensor_domain('sensing', declaration)
    assert domain.execute('list_sensors', {})['sensors'][0]['modality'] == 'rgbd'
    assert domain.close()['ok']
    declaration['providers'][0]['calibration_sha256'] = 'unbound'
    with pytest.raises(ValueError, match='SHA'):
        build_sensor_domain('sensing', declaration)


def test_read_actual_usd_optics_and_pose_with_no_arm():
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import UsdGeom, Gf
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    camera = UsdGeom.Camera.Define(stage, '/World/Overview')
    camera.CreateFocalLengthAttr(20.)
    camera.CreateHorizontalApertureAttr(20.)
    camera.CreateVerticalApertureAttr(10.)
    camera.AddTranslateOp().Set(Gf.Vec3d(1., 2., 3.))
    record = read_static_calibration(stage, '/World/Overview', width=6, height=4)
    assert record['intrinsics'] == [6., 0., 3., 0., 8., 2., 0., 0., 1.]
    assert record['version'] == 2 and record['pixel_center_offset_uv'] == [.5, .5]
    t = np.array(record['world_from_camera']).reshape(4, 4)
    np.testing.assert_equal(t[:3, 3], [1., 2., 3.])
    np.testing.assert_equal(t[:3, :3], np.diag([1., -1., -1.]))
    first = calibration_record(record)[1]
    camera.GetFocalLengthAttr().Set(21.)
    assert calibration_record(read_static_calibration(stage, '/World/Overview', width=6, height=4))[1] != first
    camera.GetFocalLengthAttr().Set(22., 1.)
    with pytest.raises(ValueError, match='static'):
        read_static_calibration(stage, '/World/Overview')


def test_rgbd_calibration_changes_effective_model_and_cannot_be_forged(recipe_inputs):
    from cascade.sim.mobile_identity import build_model_identity
    admission, native, paths = recipe_inputs
    old = build_model_identity(admission, native, **paths)['model_identity_sha256']
    cal, digest = calibration_record(calibration())
    native['rgbd_camera'] = dict(calibration=cal, calibration_sha256=digest,
                                 render_product='/World/OverviewRenderProduct', annotator='distance_to_image_plane')
    current = build_model_identity(admission, native, **paths)['model_identity_sha256']
    assert current != old
    native['rgbd_camera']['calibration']['world_from_camera'][3] = .01
    with pytest.raises(ValueError, match='calibration digest'):
        build_model_identity(admission, native, **paths)
    native['rgbd_camera']['calibration_sha256'] = calibration_record(native['rgbd_camera']['calibration'])[1]
    assert build_model_identity(admission, native, **paths)['model_identity_sha256'] != current


@pytest.mark.parametrize('bad', [None, 'old_depth', 'during_depth', 'physics', 'calibration', 'missing_depth'])
def test_native_capture_binds_each_aov_and_refuses_async_or_physics_change(bad):
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import capture_bound_rgb
    from test_microduck_stepper import native_fixture
    ns = native_fixture()
    ns.update_fabric = lambda: None
    events = []
    class Readback:
        def get_render_times(self):
            return render_times(ns.sim_time)
        def get_data_bound(self, name, *, checkpoint):
            events.append(name)
            if name == 'rgb':
                return np.zeros((480, 640, 3), np.uint8), {}, render_times(ns.sim_time)
            if bad == 'during_depth':
                raise RuntimeError('render product changed during AOV readback')
            if bad == 'physics':
                ns.simulation_step_count += 1
            result = np.ones((480, 640, 1), np.float32)
            result[0, 0] = np.inf
            return (None if bad == 'missing_depth' else result), {}, render_times(ns.sim_time-(.005 if bad == 'old_depth' else 0))
    cal = calibration(640, 480)
    def observed_calibration():
        result = copy.deepcopy(cal)
        if bad == 'calibration' and events:
            result['world_from_camera'][3] += .1
        return result
    before = (ns.simulation_step_count, ns.sim_time)
    if bad:
        with pytest.raises((ValueError, RuntimeError)):
            capture_bound_rgb(ns, NS(update=lambda: None), Readback(), updates=3, calibration=observed_calibration)
    else:
        capture = capture_bound_rgb(ns, NS(update=lambda: None), Readback(), updates=3, calibration=observed_calibration)
        assert capture['rgbd_render_times']['rgb'] == capture['rgbd_render_times']['depth'] == capture['render_times']
        assert capture['depth_m'][0, 0] == 0 and capture['depth_m'][1, 1] == 1
        assert capture['calibration'] == cal
        assert (ns.simulation_step_count, ns.sim_time) == before
    assert events == ['rgb', 'distance_to_image_plane']


def test_aov_readback_requires_unchanged_reference_around_each_channel():
    import runpy
    from pathlib import Path
    cls = runpy.run_path(str(Path(__file__).parents[1]/'scripts/isaac_camera_readback.py'))['CpuCameraReadback']
    obj = cls(None)
    token = render_times(.01)
    obj.get_render_times = lambda: copy.deepcopy(token)
    def get_data(name):
        token['IsaacReadSimulationTime']['simulationTime'] += .005
        return np.zeros((2, 2, 3), np.uint8), {}
    obj.get_data = get_data
    with pytest.raises(RuntimeError, match='during AOV'):
        obj.get_data_bound('rgb')


def test_absorbed_signal_in_depth_read_is_fenced_before_any_further_read_or_copy():
    import runpy
    from pathlib import Path
    from types import SimpleNamespace as NS
    from cascade.sim.microduck_newton import capture_bound_rgb
    from test_microduck_stepper import native_fixture
    cls = runpy.run_path(str(Path(__file__).parents[1]/'scripts/isaac_camera_readback.py'))['CpuCameraReadback']
    obj = cls(None)
    ns = native_fixture()
    ns.update_fabric = lambda: None
    pending = False
    events = []
    def checkpoint():
        if pending:
            raise KeyboardInterrupt('retained signal notification')
    def data(name):
        nonlocal pending
        events.append(name)
        if name == 'rgb':
            return np.zeros((480, 640, 3), np.uint8), {}
        pending = True  # SDK swallowed the exception but the notification survives
        return object(), {}  # must not be converted/copied after this return
    def times():
        assert not pending, 'read another annotator after signal'
        events.append('reference')
        return render_times(ns.sim_time)
    def cal():
        assert not pending, 'read calibration after signal'
        return calibration(640, 480)
    obj.get_data, obj.get_render_times = data, times
    with pytest.raises(KeyboardInterrupt, match='retained'):
        capture_bound_rgb(ns, NS(update=lambda: None), obj, updates=1, calibration=cal, checkpoint=checkpoint)
    assert events[-1] == 'distance_to_image_plane'


def test_rgbd_requires_real_capability_and_transport_deadline(endpoint):
    import threading
    _, server, cache, p, _, _, publish, operations = endpoint
    publish()
    server._frame = lambda req: cache(req)  # callable has RGB-D data but no advertised capability
    r = reader(p, max_age_s=5)
    try:
        assert r() is None and 'advertise' in r.last_error
        assert all(op['op'] == 'hello' for op in operations)
    finally:
        r.close()
    entered, release = threading.Event(), threading.Event()
    class Held:
        rgbd_enabled = True
        def __call__(self, req):
            entered.set()
            assert release.wait(3)
            return cache(req)
    server._frame = Held()
    p['timeout_s'] = .05
    r = reader(p, max_age_s=5)
    try:
        assert r() is None
        assert entered.is_set() and ('timed out' in r.last_error or 'deadline' in r.last_error)
    finally:
        release.set()
        r.close()


def test_full_native_resolution_fits_wire_limit_without_loss(endpoint):
    c, server, _, p, _, _, _, _ = endpoint
    cal = calibration(640, 480)
    cache = RgbdFrameCache(c.hello(), calibration=cal, max_pixels=640*480, max_jpeg_bytes=2*1024**2)
    server._frame = cache
    random = np.random.default_rng(123)
    rgb = random.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    depth = random.uniform(.01, 20., (480, 640)).astype(np.float32)
    refs = render_times(.05)
    for key in refs['rpFabricTime']:
        refs['rpFabricTime'][key] = np.int64(refs['rpFabricTime'][key])
    refs['IsaacReadSimulationTime'].update(simulationTime=np.float64(.05), execOut=np.int64(1))
    cache.publish(rgb, depth_m=depth, calibration=cal, step=10, sim_time_s=.05, captured_at=time.monotonic(),
                  render_times=refs, rgbd_render_times={'rgb': refs, 'depth': refs})
    p['timeout_s'] = 2.
    r = MobileRgbdReader(p, 'overview', calibration_sha256=calibration_record(cal)[1], max_age_s=5.)
    try:
        result = r()
        assert result is not None, r.last_error
        assert result.rgb8 == rgb.tobytes() and result.depth_m_f32le == depth.astype('<f4').tobytes()
        assert result.metadata['render_reference']['rgb']['IsaacReadSimulationTime']['execOut'] == 1
        packet = {'ok': True, **cache({'camera': 'overview', 'modality': 'rgbd'})}
        assert len(json.dumps(packet).encode()) < r._client.max_reply
    finally:
        r.close()
        cache.close()


@pytest.mark.parametrize('bad', ['float_rational', 'extra_field', 'boolean_rational'])
def test_render_reference_normalization_never_truncates_or_discards(bad):
    from cascade.sim.mobile_rgbd import render_reference_record
    refs = render_times(.05)
    if bad == 'float_rational':
        refs['rpFabricTime']['fabricFrameTimeNumerator'] = float(refs['rpFabricTime']['fabricFrameTimeNumerator'])
    elif bad == 'extra_field':
        refs['IsaacReadSimulationTime']['otherCaptureTime'] = .01
    else:
        refs['rpFabricTime']['fabricFrameTimeDenominator'] = True
    with pytest.raises(ValueError):
        render_reference_record(refs, .05)
