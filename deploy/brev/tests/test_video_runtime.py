"""Runtime video routing must match the independently admitted host profile."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE / 'streaming'), str(ROOT / 'deploy/runtime')]
spec = importlib.util.spec_from_file_location('video_runtime', ROOT / 'deploy/runtime/runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_omitted_config_has_no_video_route():
    assert runtime.camera_video_configuration({}, [], '100.64.0.10') is None


def test_disabled_config_has_no_video_route(tmp_path):
    path = tmp_path / 'video.json'
    path.write_text(json.dumps({'schema': 1, 'enabled': False}))
    assert runtime.camera_video_configuration({'PAAI_CAMERA_VIDEO_CONFIG': str(path)}, [], '100.64.0.10') is None


@pytest.mark.parametrize('change', [
    {}, {'tailnet_ipv4': '100.64.0.11', 'visitor_origin': 'http://100.64.0.11:8092'},
    {'visitor_origin': 'https://unrelated.invalid'}, {'http_port': 9000},
    {'media_port': 9001}, {'rtsp_ports': [9002, 9003, 9004]},
])
def test_video_origin_and_ports_must_match_frontend(tmp_path, change):
    path = tmp_path / 'video.json'
    config = {'schema': 1, 'enabled': True, 'tailnet_ipv4': '100.64.0.10',
              'visitor_origin': 'http://100.64.0.10:8092', **change}
    path.write_text(json.dumps(config))
    env = {'PAAI_CAMERA_VIDEO_CONFIG': str(path)}
    if change:
        with pytest.raises(ValueError, match='match the private frontend'):
            runtime.camera_video_configuration(env, ['http://100.64.0.10:8092'], '100.64.0.10')
    else:
        assert runtime.camera_video_configuration(env, ['http://100.64.0.10:8092'], '100.64.0.10') == {
            'transport': 'whep', 'live_verified': False}
