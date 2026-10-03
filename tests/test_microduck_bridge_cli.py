"""CPU CLI/admission tests. Synthetic artifacts never certify physics."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))


def cli():
    return importlib.import_module('isaac_microduck_bridge')


def test_help_is_portable_and_does_not_import_optional_runtime():
    script = REPO / 'scripts/isaac_microduck_bridge.py'
    assert script.is_file(), 'production CLI missing'
    code = ("import runpy,sys; sys.argv=['bridge','--help']; "
            f"runpy.run_path({str(script)!r},run_name='__main__')")
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True,
                            text=True, timeout=10, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
    assert result.returncode == 0, result.stderr
    for flag in ('--release', '--bundle', '--asset', '--policy-sha256', '--bam-profile',
                 '--bam-source-root', '--bundle-sha256', '--port', '--max-wall-s', '--max-steps',
                 '--limits', '--python-extra-path', '--check-only'):
        assert flag in result.stdout


def forbid_live_imports(monkeypatch):
    # Other tests may already have legitimately loaded ORT. Assert that this
    # operation never ATTEMPTS a live import, not global process-wide absence.
    import builtins
    original = builtins.__import__
    original_module = importlib.import_module
    forbidden = {'isaacsim', 'onnxruntime', 'newton', 'warp'}
    def imported(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            raise AssertionError(f'offline operation imported {name}')
        return original(name, *args, **kwargs)
    def imported_module(name, *args, **kwargs):
        if name.split('.')[0] in forbidden:
            raise AssertionError(f'offline operation imported {name}')
        return original_module(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imported)
    monkeypatch.setattr(importlib, 'import_module', imported_module)


def test_explicit_flags_and_physx_rejected_without_importing_kit(monkeypatch):
    forbid_live_imports(monkeypatch)
    with pytest.raises(SystemExit):
        cli().parse_args([])
    with pytest.raises(SystemExit):
        cli().parse_args(['--engine', 'physx'])


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def software_bundle(tmp_path, monkeypatch):
    module = importlib.import_module('cascade.sim.microduck_newton')
    root = tmp_path / 'bundle'
    root.mkdir()
    manifest = {'sources': {'fixture': {'revision': 'test-only'}}, 'licenses': {'test': {}},
                'files': [{'path': 'microduck_rl/src/mjlab_microduck/robot/microduck/robot_allcollisions.xml',
                           'sha256': digest(b'software-only XML'), 'size': len(b'software-only XML')}]}
    monkeypatch.setattr(module, 'source_manifest', lambda: (manifest, b'fixture manifest'))
    contents = {'source/robot_allcollisions.xml': b'software-only XML',
                'provenance/manifest.json': b'fixture manifest', 'usd/microduck.usda': b'software-only USD'}
    contents.update({f'fixture/{i}.bin': str(i).encode() for i in range(131)})
    rows = []
    for name, data in sorted(contents.items()):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        rows.append({'path': name, 'size': len(data), 'sha256': digest(data)})
    receipt = {'schema_version': 1, 'outputs': rows, 'usd_path': 'usd/microduck.usda',
               'source_xml': 'robot_allcollisions.xml', 'source_files': [
                   {'path': 'robot_allcollisions.xml', 'sha256': digest(b'software-only XML'), 'size': len(b'software-only XML')}],
               'collision_profile': 'velstand', 'provenance': {'sources': manifest['sources'],
                   'licenses': manifest['licenses'], 'manifest_sha256': digest(b'fixture manifest')},
               'versions': {'adapter_sha256': digest((REPO / 'scripts/convert_microduck.py').read_bytes())}}
    raw = json.dumps(receipt).encode()
    (root / 'receipt.json').write_bytes(raw)
    (root / 'receipt.sha256').write_text(digest(raw)+'\n')
    return module, root, digest(raw), receipt


def test_all_bundle_outputs_and_source_identity_verified(software_bundle):
    module, root, receipt_hash, receipt = software_bundle
    result = module.verify_bundle(root, expected_sha256=receipt_hash)
    assert result['asset_sha256'] == digest((root / 'usd/microduck.usda').read_bytes())
    assert result['asset_receipt_sha256'] == receipt_hash
    assert len(result['receipt']['outputs']) == 134
    (root / receipt['outputs'][-1]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash|size|SHA'):
        module.verify_bundle(root, expected_sha256=receipt_hash)


@pytest.mark.parametrize('bad', ['missing', 'extra', 'symlink', 'traversal', 'duplicates', 'receipt_hash', 'source'])
def test_bad_bundle_fails_before_kit(software_bundle, bad):
    module, root, receipt_hash, receipt = software_bundle
    if bad == 'missing':
        (root / receipt['outputs'][0]['path']).unlink()
    elif bad == 'extra':
        (root / 'unexpected.usda').write_text('extra')
    elif bad == 'symlink':
        p = root / receipt['outputs'][0]['path']
        p.unlink()
        p.symlink_to(root / 'receipt.json')
    elif bad == 'receipt_hash':
        receipt_hash = '0'*64
    else:
        if bad == 'traversal':
            receipt['outputs'][0]['path'] = '../outside'
        elif bad == 'duplicates':
            receipt['outputs'][1] = receipt['outputs'][0]
        else:
            receipt['provenance']['sources'] = {}
        raw = json.dumps(receipt).encode()
        receipt_hash = digest(raw)
        (root / 'receipt.json').write_bytes(raw)
        (root / 'receipt.sha256').write_text(receipt_hash+'\n')
    with pytest.raises((ValueError, FileNotFoundError)):
        module.verify_bundle(root, expected_sha256=receipt_hash)


def test_portable_dedicated_experience_has_no_full_base_importer_inheritance(tmp_path):
    module = importlib.import_module('cascade.sim.microduck_newton')
    release = tmp_path / 'release with spaces'
    for name in ('apps', 'exts', 'extscache', 'extsUser', 'extsDeprecated'):
        (release / name).mkdir(parents=True, exist_ok=True)
    text = module.experience_text(release)
    config = tomllib.loads(text)
    deps = config['dependencies']
    assert 'isaacsim.physics.newton' in deps and 'isaacsim.sensors.experimental.rtx' in deps
    assert not any(x in str(deps) for x in ('exp.full', 'exp.base', 'importer', 'pink'))
    assert config['settings']['persistent']['renderer']['startupMessageDisplayed'] is True
    assert config['settings']['app']['exts']['folders']['++'] == [str(release / n) for n in
        ('apps', 'exts', 'extscache', 'extsUser', 'extsDeprecated')]
    assert '/home/' not in text.replace(str(release), '')


def software_limits():
    return dict(max_linear_speed=.3, max_angular_speed=.5, max_duration_s=5., lease_s=.3,
                max_state_age_s=1., max_action_wall_s=10., min_height_m=.05,
                max_height_m=.3, max_tilt_rad=.7, max_contacts=512, max_constraints=2400)


def software_model_identity(*args, **kwargs):
    """Synthetic CLI lifecycle only. Actual recipe construction has separate tests."""
    from mobile_support_fixture import support_contract
    return {'model_identity_sha256': 'e'*64, 'recipe': {'software_fixture': True},
            'support_contract': support_contract()}


def test_limits_require_complete_explicit_positive_contract(tmp_path):
    p = tmp_path / 'limits.json'
    p.write_text(json.dumps(software_limits()))
    assert cli().load_limits(p) == software_limits()
    for key in software_limits():
        bad = software_limits()
        bad.pop(key)
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            cli().load_limits(p)
    for key in software_limits():
        bad = software_limits()
        bad[key] = True
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            cli().load_limits(p)


def test_python_extra_path_cannot_inject_foreign_numpy_usd_or_venv(tmp_path):
    extras = tmp_path / 'extras'
    extras.mkdir()
    (extras / 'onnxruntime').mkdir()
    assert cli().validate_extra_paths([extras]) == [str(extras)]
    (extras / 'numpy').mkdir()
    with pytest.raises(ValueError, match='onnxruntime'):
        cli().validate_extra_paths([extras])


@pytest.mark.parametrize('mode', ['normal', 'stop_race', 'inference', 'boot', 'capture', 'identity',
                                'probe_fail', 'probe_stale', 'rgbd'])
def test_run_exercises_real_rpc_bounded_loop_trace_and_teardown(tmp_path, mode, capsys, monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from test_microduck_stepper import SoftwareBackend, SoftwareActuator, SoftwarePolicy, render_times
    from cascade.sim.mobile_bridge import MobileBridgeServer
    from cascade.sim.bridge_client import BridgeClient, BridgeError
    from cascade.sim import mobile_identity
    def model_identity(*args, **kwargs):
        if mode == 'identity':
            raise ValueError('effective native recipe changed during bootstrap')
        return software_model_identity(*args, **kwargs)
    monkeypatch.setattr(mobile_identity, 'build_model_identity', model_identity)
    output = tmp_path / 'run'
    args = NS(out=output, device='cuda:0', robot_id='microduck', source='software-only',
              max_wall_s=3., max_steps=9, port=0, camera_every=4, max_jpeg_bytes=100000,
              policy=tmp_path / 'fixture.onnx', policy_sha256='b'*64, python_extra_path=[])
    args.camera_rgbd = mode == 'rgbd'
    admission = dict(asset_sha256='a'*64, asset_receipt_sha256='c'*64,
                     bam_params={}, limits=software_limits(), experience_text='software fixture\n')
    created = []
    class Backend(SoftwareBackend):
        def __init__(self, *unused):
            super().__init__()
            self.bam = SoftwareActuator(self)
            self.receipt = {'software_fixture': True}
            if mode == 'rgbd':
                from test_sensing_rgbd import calibration
                self.receipt['rgbd_camera'] = {'calibration': calibration(32, 24)}
            self.shutdown_code = None
            created.append(self)
        def open(self):
            if mode == 'boot':
                raise RuntimeError('test boot failure')
        def shutdown(self, exit_code):
            assert (output / 'receipt.json').is_file(), 'Kit may terminate inside close; persist first'
            self.shutdown_code = exit_code
        def capture(self):
            self.events.append(('capture', self.step_count))
            if mode == 'capture':
                raise RuntimeError('test capture failure')
            value = dict(rgb=np.zeros((24, 32, 3), np.uint8), step=self.step_count,
                         sim_time_s=self.sim_time, captured_at=0., render_times=render_times(self.sim_time))
            if mode == 'rgbd':
                value.update(depth_m=np.full((24, 32), .5, np.float32),
                    calibration=self.receipt['rgbd_camera']['calibration'],
                    rgbd_render_times={'rgb': render_times(self.sim_time), 'depth': render_times(self.sim_time)})
            return value
        def support_probe(self):
            # Exercise runner discrimination only; this is NOT a solver probe.
            return dict(passed=mode != 'probe_fail', step=self.step_count - (mode == 'probe_stale'),
                        sim_time_s=self.sim_time, max_force_torque_difference=0.)
    controlled = {}
    def policy_factory(*unused):
        p = SoftwarePolicy(created[0])
        if mode == 'inference':
            p.infer = lambda _: (_ for _ in ()).throw(RuntimeError('test failed inference'))
        if mode == 'stop_race':
            def crossed(obs):
                if obs[0,48] != 0:
                    controlled['controller'].stop(latch=True)
                return np.full(14, obs[0,48], np.float32)
            p.infer = crossed
        return p
    servers = []
    class Server(MobileBridgeServer):
        def __init__(self, controller, **kwargs):
            controlled['controller'] = controller
            super().__init__(controller, **kwargs)
        def start(self):
            super().start()
            servers.append(self)
            client = BridgeClient(self.address[0], self.address[1], timeout_s=1.)
            client.connect()
            try:
                hello = client.request({'op': 'hello', 'role': 'reader'})
                state = client.request({'op': 'state'})['state']
                image = client.request({'op': 'frame', 'camera': 'overview'})
                assert hello['robot_id'] == 'microduck'
                assert state['step'] == 3
                assert image['frame']['step'] == 3
                if mode == 'rgbd':
                    assert 'rgbd' in hello['capabilities']
                    rgbd = client.request({'op': 'frame', 'camera': 'overview', 'modality': 'rgbd'})['rgbd']
                    assert rgbd['step'] == state['step'] and rgbd['epoch'] == hello['epoch']
                with pytest.raises(BridgeError):
                    client.request({'op': 'exec', 'code': 'no'})
            finally:
                client.close()
            if mode == 'stop_race':
                c = controlled['controller']
                h = c.hello()
                accepted = c.command_velocity(dict(robot_id=h['robot_id'], source=h['source'],
                    epoch=h['epoch'], generation=h['generation'], owner='fixture-owner',
                    command_id='cross-inference', vx=.1, vy=0., wz=0., duration_s=.1))
                assert accepted['ok']
    result = cli().run(args, admission, backend_factory=Backend, policy_factory=policy_factory, server_factory=Server)
    success = mode in ('normal', 'stop_race', 'rgbd')
    assert result['completed'] is success
    assert created[0].closed == 1
    assert created[0].shutdown_code == (0 if success else 1)
    saved = json.loads((output / 'receipt.json').read_text())
    assert saved['physical_acceptance'] is False
    assert saved['teardown_errors'] == []
    if success:
        # Warm graphics on unscored bootstrap state, DISCARD that frame; the
        # first RPC image above must still come from a NEW controlled solve.
        assert created[0].events[0] == ('capture', 2)
        frames = [json.loads(x) for x in (output / 'frames.jsonl').read_text().splitlines()]
        assert frames[0]['step'] == 3 and all(f['step'] > 2 for f in frames)
        assert saved['discarded_bootstrap_capture']['published'] is False
        rows = [json.loads(x) for x in (output / 'physics.jsonl').read_text().splitlines()]
        inputs = [json.loads(x) for x in (output / 'policy.jsonl').read_text().splitlines()]
        assert len(rows) == 9
        if mode == 'stop_race':
            assert len(inputs) == 4
            assert [r['observation_step'] for r in inputs] == [2, 6, 6, 10]
            assert inputs[1]['status'] == 'discarded' and inputs[1]['committed'] is False
            assert inputs[1]['commands'][0] != 0 and inputs[2]['commands'][0] == 0
            assert saved['policy_attempts'] == 4 and saved['policy_commits'] == 3
            from cascade.control.microduck_policy import HOME_Q
            for target in created[0].bam.targets:
                np.testing.assert_array_equal(target, HOME_Q)
        else:
            assert len(inputs) == 3
            assert [r['observation_step'] for r in inputs] == [2, 6, 10]
        assert all(len(r['observation']) == 61 for r in inputs)
        assert (output / 'BRIDGE_LISTENING.json').exists()
        assert 'BRIDGE_LISTENING' in capsys.readouterr().out
        assert saved['steps'] == 9
        if mode == 'rgbd':
            pairs = sorted((output/'frames').glob('*.rgbd.json'))
            assert len(pairs) == len(frames)
            assert [json.loads(p.read_text())['step'] for p in pairs] == [f['step'] for f in frames]
        probes = [json.loads(x) for x in (output / 'support-probe.jsonl').read_text().splitlines()]
        assert saved['support_probe_count'] == len(probes) == 4
        assert [p['step'] for p in probes] == [3, 6, 10, 11]
        assert all(p['model_identity_sha256'] == 'e'*64 for p in probes)
    else:
        assert saved['error']
        assert not (output / 'BRIDGE_LISTENING.json').exists()
        if mode == 'identity':
            assert not created[0].bam.targets
            assert not [e for e in created[0].events if e[0] == 'solve']
            assert not servers
        if mode.startswith('probe_'):
            assert saved['support_probe_count'] == 1
            assert 'support-force probe' in saved['error']
            assert not servers
        if mode == 'inference':
            inputs = [json.loads(x) for x in (output / 'policy.jsonl').read_text().splitlines()]
            assert len(inputs) == 1 and inputs[0]['status'] == 'failed'
            assert not [e for e in created[0].events if e[0] == 'solve']
    for server in servers:
        assert not server._accept_thread.is_alive() and not server._watchdog_thread.is_alive()
    with pytest.raises(FileExistsError):
        cli().run(args, admission, backend_factory=Backend, policy_factory=policy_factory, server_factory=Server)


def test_check_only_admits_without_kit_network_or_outdir(software_bundle, tmp_path, monkeypatch, capsys):
    module, root, receipt_hash, receipt = software_bundle
    from cascade.control import newton_bam
    forbid_live_imports(monkeypatch)
    release = tmp_path / 'release'
    for folder in ('apps', 'exts', 'extscache', 'extsUser', 'extsDeprecated'):
        (release / folder).mkdir(parents=True, exist_ok=True)
    (release / 'python.sh').write_text('software fixture; never executed')
    bam_root = tmp_path / 'bam'
    bam_root.mkdir()
    (bam_root / 'fixture.py').write_text('not executed')
    monkeypatch.setattr(newton_bam, 'SOURCE_SHA256', {'fixture.py': digest(b'not executed')})
    policy = tmp_path / 'fixture.onnx'
    policy.write_bytes(b'software fixture; never executed')
    limits = tmp_path / 'limits.json'
    limits.write_text(json.dumps(software_limits()))
    manifest, raw_manifest = module.source_manifest()
    manifest['files'].append({'path': 'microduck-policies/velstand.onnx',
                             'sha256': digest(policy.read_bytes()), 'size': policy.stat().st_size})
    output = tmp_path / 'output'
    argv = ['--engine', 'newton', '--release', str(release), '--bundle', str(root),
            '--bundle-sha256', receipt_hash, '--policy', str(policy), '--policy-sha256', digest(policy.read_bytes()),
            '--bam-source-root', str(bam_root), '--bam-profile', 'nominal_no_current_limit_no_delay',
            '--limits', str(limits), '--out', str(output), '--port', '0', '--robot-id', 'microduck',
            '--source', 'software-only', '--device', 'cuda:0', '--max-wall-s', '60', '--max-steps', '10', '--check-only']
    assert cli().main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['physical_acceptance'] is False and result['output_count'] == 134
    assert not output.exists()
    for flag, value in [('--port', '-1'), ('--policy-sha256', 'A'*64), ('--device', 'cpu'),
                        ('--max-wall-s', 'nan'), ('--max-steps', '0'), ('--bam-profile', 'implicit')]:
        bad = list(argv)
        bad[bad.index(flag)+1] = value
        assert cli().main(bad) != 0
    assert not output.exists()
