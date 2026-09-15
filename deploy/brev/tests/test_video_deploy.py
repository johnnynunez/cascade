"""The optional video profile must not affect a basic install or claim live media."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('video_deploy', HERE / 'deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, 'HERE', tmp_path)
    monkeypatch.setattr(deploy, 'STOP', tmp_path / 'STOP')
    return tmp_path


def video_profile(enabled=True):
    return {'architecture': 'x86_64', 'storage': {'root': '/data/paai'},
            'streaming': {'enabled': enabled}}


def test_disabled_video_does_not_install_firewall_or_pull(isolated, monkeypatch):
    monkeypatch.setattr(deploy, 'remote', lambda *a, **kw: pytest.fail('Unexpected host mutation'))
    monkeypatch.setattr(deploy, 'sync', lambda *a, **kw: pytest.fail('Unexpected transfer'))
    deploy.prepare_camera_video(video_profile(False), 'invalid-unused-address')


def test_compose_adds_overlay_only_for_explicit_video(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, 'remote', lambda args, **kwargs: calls.append(args))
    for value in (False, True):
        deploy.compose(video_profile(value), 'ps')
    assert '/data/paai/deployment/compose.streaming.yaml' not in calls[0]
    assert calls[1][-3:] == ['-f', '/data/paai/deployment/compose.streaming.yaml', 'ps']


def test_failed_host_admission_cannot_publish_config_or_download(isolated, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, 'remote', lambda args, **kwargs: calls.append(args) or '{}')
    monkeypatch.setattr(deploy, 'sync', lambda *a, **kw: pytest.fail('Unprotected video config transferred'))
    with pytest.raises(ValueError, match='firewall'):
        deploy.prepare_camera_video(video_profile(), '100.64.0.10')
    assert len(calls) == 1 and calls[0][0] == 'python3'


@pytest.mark.parametrize('cached', [True, False])
def test_relay_reuses_pinned_image_and_records_no_media_claim(isolated, monkeypatch, cached):
    calls, files = [], {}
    image_id = 'sha256:' + 'a' * 64
    def remote(args, **kwargs):
        calls.append(args)
        if args[0] == 'python3':
            return json.dumps({'status': 'PASS_HOST_SETUP', 'active': True})
        if 'inspect' in args:
            if not cached and not any('pull' in a for a in calls):
                raise RuntimeError('Image absent')
            return image_id
        return ''
    def sync(root, target, selected, **kwargs):
        files.update({name: json.loads((root / name).read_text()) for name in selected})
    monkeypatch.setattr(deploy, 'remote', remote)
    monkeypatch.setattr(deploy, 'sync', sync)
    deploy.prepare_camera_video(video_profile(), '100.64.0.10')
    assert sum('pull' in command for command in calls) == int(not cached)
    assert files['camera-video.json']['tailnet_ipv4'] == '100.64.0.10'
    assert files['mediamtx.yml']['webrtcLocalUDPAddress'] == '100.64.0.10:8189'
    assert files['mediamtx.yml']['webrtcAddress'] == '127.0.0.1:8889'
    receipt = json.loads((isolated / 'proof/CAMERA_RELAY_IMAGE.json').read_text())
    assert receipt['reused'] is cached and receipt['live_media_verified'] is False
    assert receipt['image_id'] == image_id


def test_running_relay_does_not_force_healthy_demo_restart(isolated, monkeypatch):
    monkeypatch.setattr(deploy, 'remote_preparation', lambda p: {'active': False})
    monkeypatch.setattr(deploy, 'remote_installation', lambda p: {'status': 'PREPARED'})
    monkeypatch.setattr(deploy, 'runtime_image_matches', lambda *a: True)
    monkeypatch.setattr(deploy, 'require_installed_bundle', lambda *a: None)
    monkeypatch.setattr(deploy, 'running_rows', lambda p: [
        {'Service': 'demo', 'State': 'running', 'Health': 'healthy'},
        {'Service': 'camera-relay', 'State': 'running', 'Health': ''}])
    monkeypatch.setattr(deploy, 'preflight', lambda *a, **kw: {})
    monkeypatch.setattr(deploy, 'compose', lambda *a, **kw: pytest.fail('Healthy demo restarted'))
    monkeypatch.setattr(deploy, 'status', lambda p: {'ready': False})
    assert deploy.start(video_profile(), Path('profile.json')) == {'ready': False}


def test_missing_relay_is_started_without_dropping_demo(isolated, monkeypatch):
    monkeypatch.setattr(deploy, 'remote_preparation', lambda p: {'active': False})
    monkeypatch.setattr(deploy, 'remote_installation', lambda p: {'status': 'PREPARED'})
    monkeypatch.setattr(deploy, 'runtime_image_matches', lambda *a: True)
    monkeypatch.setattr(deploy, 'require_installed_bundle', lambda *a: None)
    monkeypatch.setattr(deploy, 'running_rows', lambda p: [
        {'Service': 'demo', 'State': 'running', 'Health': 'healthy'}])
    monkeypatch.setattr(deploy, 'preflight', lambda *a, **kw: {})
    calls = []
    monkeypatch.setattr(deploy, 'compose', lambda p, *a, **kw: calls.append(a))
    monkeypatch.setattr(deploy, 'status', lambda p: {'ready': False})
    deploy.start(video_profile(), Path('profile.json'))
    assert len(calls) == 1 and calls[0][0] == 'up'
    assert '--no-build' in calls[0]


@pytest.mark.parametrize('matches', [True, False])
def test_start_cannot_toggle_profile_or_change_source_without_install(isolated, monkeypatch, matches):
    monkeypatch.setattr(deploy, 'remote_preparation', lambda p: {'active': False})
    monkeypatch.setattr(deploy, 'remote_installation', lambda p: {
        'status': 'PREPARED', 'bundle_identity': 'current' if matches else 'old'})
    monkeypatch.setattr(deploy, 'runtime_image_matches', lambda *a: True)
    monkeypatch.setattr(deploy, 'staged_bundle', lambda: (isolated, ['runtime.py']))
    monkeypatch.setattr(deploy, 'bundle_identity', lambda *a: 'current')
    admitted = []
    monkeypatch.setattr(deploy, 'running_rows', lambda p: [
        {'Service': 'demo', 'State': 'running', 'Health': 'healthy'}])
    monkeypatch.setattr(deploy, 'preflight', lambda *a, **kw: admitted.append(kw))
    monkeypatch.setattr(deploy, 'compose', lambda *a, **kw: pytest.fail('Unrequested restart'))
    monkeypatch.setattr(deploy, 'status', lambda p: {'ready': False})
    if matches:
        assert deploy.start(video_profile(False), Path('profile.json')) == {'ready': False}
        assert len(admitted) == 1
    else:
        with pytest.raises(RuntimeError, match='resume install'):
            deploy.start(video_profile(False), Path('profile.json'))
        assert not admitted
