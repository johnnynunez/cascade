import base64
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from benchmark.vab.rgb_capture import PublicRgbRecorder
from cascade.perception.rgb_triangulation import RgbView


def scene():
    model = NS(body_parentid=np.array([0, 0, 1, 0]), cam_bodyid=np.array([0, 2]),
        cam_mode=np.array([0, 0]), cam_fovy=np.array([45., 75.]),
        cam_pos=np.zeros((2, 3)), cam_quat=np.tile([1., 0, 0, 0], (2, 1)),
        camera_name2id=lambda name: ('agentview', 'robot0_eye_in_hand').index(name),
        body_name2id=lambda name: 1 if name == 'robot' else 3, opt=NS(timestep=.002))
    # No object state, depth, contacts, render, forward or step is available.
    data = NS(cam_xpos=np.array([[0., 0, 0], [.1, 0, 0]]),
              cam_xmat=np.tile(np.diag([1., -1., -1.]).ravel(), (2, 1)), time=.04)
    env = NS(sim=NS(model=model, data=data), robots=[NS(robot_model=NS(root_body='robot'))])
    rgb = np.arange(128*128*3, dtype=np.uint8).reshape(128, 128, 3)
    observation = {'images': {name: rgb.copy() for name in ('agentview', 'robot0_eye_in_hand')}}
    env.calls = []
    def render(**kwargs):
        env.calls.append(dict(kwargs))
        return observation['images'][kwargs['camera_name']].copy()
    env.sim.render = render
    return env, observation


def make(env, tmp_path, **kwargs):
    return PublicRgbRecorder(env, tmp_path/'rgb', model_identity_sha256='a'*64, epoch='episode', **kwargs)


def render_pair(env):
    for name in ('agentview', 'robot0_eye_in_hand'):
        env.sim.render(camera_name=name, width=128, height=128, depth=False)


def test_public_pair_keeps_rgb_bytes_flips_y_only_and_binds_wrist_pose(tmp_path):
    env, observation = scene()
    original = env.sim.render
    recorder = make(env, tmp_path)
    render_pair(env)
    recorder.capture(observation, solver_step=20, simulation_time_s=.04)
    data = json.loads((tmp_path/'rgb/0000.json').read_text())
    for item, name in zip(data['views'], ('agentview', 'robot0_eye_in_hand')):
        value = item['view']; value['rgb'] = base64.b64decode(value['rgb'], validate=True)
        view = RgbView(**value)
        assert view.sha256 == item['capture_sha256']
        assert view.rgb == observation['images'][name][::-1].tobytes()
        assert view.solver_step == 20 and view.simulation_time_s == .04
    env.sim.data.cam_xpos[1, 0] = .2
    env.sim.data.time = .09
    render_pair(env)
    recorder.capture(observation, solver_step=45, simulation_time_s=.09)
    next_row = json.loads((tmp_path/'rgb/0001.json').read_text())
    assert next_row['views'][1]['view']['world_from_camera'][3] == .2
    assert next_row['views'][0]['view']['calibration_sha256'] == recorder.calibration_sha256
    assert recorder.close() == dict(ok=True, captures=2, error=None, physical_admission=False, position_error_m=None)
    assert env.sim.render is original and len(env.calls) == 4


@pytest.mark.parametrize('kind', ['object_mount', 'track_mode', 'bad_fovy', 'missing_robot'])
def test_object_mounted_or_tracking_cameras_are_not_public_robot_calibration(tmp_path, kind):
    env, _ = scene()
    if kind == 'object_mount': env.sim.model.cam_bodyid[1] = 3
    elif kind == 'track_mode': env.sim.model.cam_mode[1] = 3
    elif kind == 'bad_fovy': env.sim.model.cam_fovy[1] = np.nan
    else: env.sim.model.body_name2id = lambda _: 0
    with pytest.raises(ValueError):
        make(env, tmp_path)
    assert not (tmp_path/'rgb').exists()


@pytest.mark.parametrize('kind', ['clock', 'repeat', 'geometry', 'image', 'pose', 'capacity', 'write'])
def test_failed_capture_is_sticky_and_never_closes_successfully(tmp_path, kind, monkeypatch):
    env, observation = scene()
    recorder = make(env, tmp_path, max_captures=1 if kind == 'capacity' else 256)
    render_pair(env)
    recorder.capture(observation, solver_step=20, simulation_time_s=.04)
    env.sim.data.time = .09
    if kind == 'repeat': env.sim.data.time = .04
    elif kind == 'geometry': env.sim.model.cam_pos[1, 0] = .1
    elif kind == 'image': observation['images']['agentview'] = np.zeros((1, 1, 3), np.uint8)
    elif kind == 'pose': env.sim.data.cam_xpos[1, 0] = np.nan
    elif kind == 'write': (tmp_path/'rgb/0001.json').write_text('existing artifact')
    with pytest.raises((ValueError, OSError)):
        render_pair(env)
        recorder.capture(observation, solver_step=20 if kind == 'repeat' else 45,
                         simulation_time_s=.08 if kind == 'clock' else env.sim.data.time)
    with pytest.raises(RuntimeError):
        recorder.capture(observation, solver_step=45, simulation_time_s=.09)
    result = recorder.close()
    assert not result['ok'] and result['captures'] == 1 and result['error']
    assert len(json.loads((tmp_path/'rgb/0000.json').read_text())['views']) == 2
    if kind == 'write': assert (tmp_path/'rgb/0001.json').read_text() == 'existing artifact'


def test_initial_reset_is_retained_without_invented_render_pose(tmp_path):
    env, observation = scene(); original = env.sim.render
    recorder = make(env, tmp_path)
    recorder.capture(observation, solver_step=20, simulation_time_s=.04)
    row = json.loads((tmp_path/'rgb/0000.json').read_text())
    assert row['status'] == 'unverified' and row['views'] == [] and not env.calls
    assert len(row['public_rgb_sha256']) == 2
    assert recorder.close()['ok'] and env.sim.render is original


def test_control_return_does_not_relabel_earlier_rgb_with_current_camera_pose(tmp_path):
    env, observation = scene()
    recorder = make(env, tmp_path)
    render_pair(env)  # Original image sampled inside the control interval.
    env.sim.data.time = .09
    env.sim.data.cam_xpos[1, 0] = .4
    recorder.capture(observation, solver_step=45, simulation_time_s=.09)
    row = json.loads((tmp_path/'rgb/0000.json').read_text())
    assert row['returned_solver_step'] == 45 and row['returned_simulation_time_s'] == .09
    assert row['views'][1]['view']['world_from_camera'][3] == .1
    assert all(v['view']['solver_step'] == 20 and v['view']['simulation_time_s'] == .04 for v in row['views'])
    assert recorder.close()['ok'] and len(env.calls) == 2


@pytest.mark.parametrize('fault', ['mismatched_pixels', 'asynchronous', 'lost_hook', 'render_mutation'])
def test_render_origin_failure_is_retained_and_hook_cleanup_preserves_other_owner(tmp_path, fault):
    env, observation = scene()
    if fault == 'render_mutation':
        original = env.sim.render
        def moving(**kwargs):
            result = original(**kwargs); env.sim.data.cam_xpos[1, 0] += .1
            return result
        env.sim.render = moving
    recorder = make(env, tmp_path)
    with pytest.raises(ValueError):
        env.sim.render(camera_name='robot0_eye_in_hand', width=128, height=128, depth=False)
        if fault == 'asynchronous': env.sim.data.time = .05
        env.sim.render(camera_name='agentview', width=128, height=128, depth=False)
        if fault == 'mismatched_pixels': observation['images']['agentview'][0, 0, 0] ^= 1
        if fault == 'lost_hook': env.sim.render = lambda **kwargs: None
        recorder.capture(observation, solver_step=round(env.sim.data.time/.002), simulation_time_s=env.sim.data.time)
    foreign = env.sim.render
    assert not recorder.close()['ok']
    if fault == 'lost_hook': assert env.sim.render is foreign
