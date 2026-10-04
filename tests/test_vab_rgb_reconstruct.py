import json
import base64
import os

import numpy as np
import pytest

from benchmark.vab.reconstruct_rgb import reconstruct, read_json
from cascade.perception.rgb_triangulation import RgbView
from test_vab_rgb_capture import scene, make, render_pair


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX FIFO required')
def test_nonregular_json_is_refused_before_read(tmp_path):
    path = tmp_path/'pipe.json'
    os.mkfifo(path)
    with pytest.raises(ValueError, match='regular file'):
        read_json(path, 65536)


def archive(tmp_path):
    env, obs = scene()
    env.sim.model.cam_fovy[:] = 45.
    rgb = np.random.default_rng(74).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    obs['images']['agentview'][:] = rgb
    obs['images']['robot0_eye_in_hand'][:] = 0
    obs['images']['robot0_eye_in_hand'][:, :-10] = rgb[:, 10:]
    recorder = make(env, tmp_path)
    recorder.capture(obs, solver_step=20, simulation_time_s=.04)
    env.sim.data.time = .09
    render_pair(env)
    recorder.capture(obs, solver_step=45, simulation_time_s=.09)
    assert recorder.close()['ok']
    return tmp_path/'rgb'


def test_saved_rgb_produces_only_unknown_uncertainty_sparse_estimates(tmp_path):
    pytest.importorskip('cv2')
    directory = archive(tmp_path)
    result = reconstruct(directory)
    assert result['captures'][0]['status'] == 'unverified'
    analysis = result['captures'][1]['reconstruction']
    assert analysis['estimated_points'] >= 5
    assert not result['physical_admission'] and result['position_error_m'] is None
    assert result['models_constructed'] == result['physics_steps_advanced'] == 0
    assert len(result['artifact_sha256']) == 3
    assert set(p.name for p in directory.iterdir()) == {'calibration.json', '0000.json', '0001.json'}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('fault', ['pixels', 'digest', 'calibration', 'missing', 'unexpected_initial'])
def test_changed_or_incomplete_archive_is_refused(tmp_path, fault):
    directory = archive(tmp_path)
    path = directory/'0001.json'
    value = json.loads(path.read_text())
    if fault == 'pixels': value['views'][0]['view']['rgb'] = 'AAAA' + value['views'][0]['view']['rgb'][4:]
    elif fault == 'digest': value['views'][0]['capture_sha256'] = '0'*64
    elif fault == 'calibration':
        path = directory/'calibration.json';value = json.loads(path.read_text());value['recipe']['robot_root'] = 3
    elif fault == 'missing': (directory/'0000.json').unlink()
    else: value.update(status='unverified', views=[], reason='reset images precede render observer')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        reconstruct(directory)


@pytest.mark.parametrize('fault', ['intrinsics', 'offset', 'frame', 'dimensions', 'returned_step', 'returned_time'])
def test_recomputed_hashes_cannot_replace_camera_recipe_or_return_clock(tmp_path, fault):
    directory = archive(tmp_path); path = directory/'0001.json'
    data = json.loads(path.read_text())
    item = data['views'][0]; view = item['view']
    if fault == 'intrinsics': view['intrinsics'][0] += 1.
    elif fault == 'offset': view['pixel_center_offset_uv'] = [0., 0.]
    elif fault == 'frame': view['world_frame_id'] = 'different_world'
    elif fault == 'dimensions':
        view['width'] = 127
        view['rgb'] = base64.b64encode(bytes(127*128*3)).decode()
    elif fault == 'returned_step': data['returned_solver_step'] = 44
    else: data['returned_simulation_time_s'] = .089
    decoded = dict(view); decoded['rgb'] = base64.b64decode(view['rgb'])
    item['capture_sha256'] = RgbView(**decoded).sha256
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='recipe|return'):
        reconstruct(directory)
