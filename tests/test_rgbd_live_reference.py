"""Real composed MCP dispatch over labelled CPU-raster data, no native claim."""
import json

import cv2
import pytest

from benchmark.rgbd.live_reference import annotate_capture
from cascade.apps.mcp_server import McpSkillServer
from cascade.apps.robot_runtime import DomainAdapter
from cascade.robotics.runtime import RobotRuntime
from cascade.sensing import BufferedSensorProvider, SensorHub
from cascade.sensing.domain import SensorDomain
from cascade.sensing.hub import SensorDescriptor
from cascade.spatial.rgbd import RgbdSpatialDomain
from test_rgbd_planar_reference import projection_episode


@pytest.mark.parametrize('mutation', [None, 'fx', 'fy', 'stale', 'extra_rpc'])
def test_live_reference_uses_ordinary_mcp_and_exact_retained_capture(monkeypatch, tmp_path, mutation):
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        _, _, capture = projection_episode(mutation if mutation in {'fx', 'fy'} else None,
                                            with_capture=True)
    finally:
        cv2.setNumThreads(previous)
    now = [10.1]
    hub = SensorHub(clock=lambda: now[0])
    provider = BufferedSensorProvider(SensorDescriptor(
        'overview', 'fixture', capture.source, 'rgbd', 'camera', 'synthetic', 'synthetic',
        calibration_id=capture.payload.metadata.calibration_id, max_age_s=2., read_timeout_s=2.))
    hub.register(provider)
    sensor = SensorDomain('sensing', hub)
    spatial = RgbdSpatialDomain('spatial', 'fixture', hub, sensor_domain='sensing',
        sensor_id='overview', map_id='cpu-board', world_frame_id='world', clock=lambda: now[0])
    domains = {name: DomainAdapter(name, {'kind': kind}, domain.resources,
        domain.tool_specs, frozenset(), runtime=domain,
        required_resources=getattr(domain, 'required_resources', ()))
        for name, kind, domain in [('sensing', 'sensors', sensor), ('spatial', 'spatial', spatial)]}
    runtime = RobotRuntime(domains)
    monkeypatch.setenv('CASCADE_ROBOT', 'synthetic_planar_reference')
    monkeypatch.setenv('CASCADE_RUN_DIR', str(tmp_path))
    server = McpSkillServer()
    server._runtime = runtime
    # Inject the declared CPU providers at the composition seam, not dispatch
    # results: catalog, argument validation and MCP execution remain ordinary.
    from cascade.apps import robot_runtime
    from cascade.config import Cfg
    monkeypatch.setattr(robot_runtime, 'describe_robot', lambda cfg: domains)
    server._mobile_config = Cfg({'robot_id': 'fixture', 'domains': {}})
    provider.publish(capture)
    def call(name, args):
        return json.loads(server.call_tool(name, args)['content'][-1]['text'])
    retained = []
    try:
        read_result = call('sensing.read_sensor', {'sensor_id': 'overview'})
        assert read_result['ok'], read_result
        watermark = hub._slots['overview'].watermark
        monkeypatch.setattr(provider, 'read', lambda: pytest.fail('fan-out reacquired sensor'))
        if mutation == 'stale':
            now[0] = 12.1
        count = [0]
        def save(index, row):
            retained.append(row)
            if mutation == 'extra_rpc':
                count[0] += 1
        previous = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            report = annotate_capture(server, read_result, save_record=save,
                                      operation_count=lambda: count[0])
        finally:
            cv2.setNumThreads(previous)
        assert report['passed'] is (mutation is None), report
        if mutation in {None, 'fx', 'fy'}:
            assert len(retained) == 17 and all(row['result']['ok'] for row in retained)
            assert report['comparison']['capture_binding']['capture_sha256'] == capture.sha256
        elif mutation == 'stale':
            assert not retained and 'stale retained' in report['error']
        else:
            assert len(retained) == 1 and 'another reader RPC' in report['error']
        assert hub.history() == (capture,) and hub._slots['overview'].watermark == watermark
        assert report['physical_admission'] is False
    finally:
        assert runtime.close()['ok']
