"""Container configuration and authenticated private visitor boundaries."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urljoin

from multidict import CIMultiDict
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runtime
import web_runtime


@pytest.fixture
def configured(tmp_path):
    original_root, original_here = runtime.ROOT, runtime.HERE
    runtime.select_root(HERE.parents[1])
    executables = {}
    for name in ('node', 'python', 'isaac-python.sh', 'llama-server', 'openclaw.mjs'):
        path = tmp_path / 'bin' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
        path.chmod(0o700)
        executables[name] = str(path)
    model, projector = tmp_path / 'brain.gguf', tmp_path / 'vision.gguf'
    model.write_bytes(b'fixture')
    projector.write_bytes(b'fixture')
    env = {
        'PAAI_DATA_DIR': str(tmp_path / 'data'), 'PAAI_STATE_DIR': str(tmp_path / 'state'),
        'PAAI_NODE': executables['node'], 'PAAI_APP_PYTHON': executables['python'],
        'PAAI_ISAAC_PYTHON': executables['isaac-python.sh'],
        'PAAI_LLAMA_SERVER': executables['llama-server'],
        'PAAI_OPENCLAW_CLI': executables['openclaw.mjs'],
        'PAAI_OPENCLAW_STATE_DIR': str(tmp_path / 'openclaw'),
        'PAAI_OPENCLAW_UI_ROOT': str(tmp_path / 'web/openclaw-ui'),
        'PAAI_MODEL_FILE': str(model), 'PAAI_MMPROJ_FILE': str(projector),
        'PAAI_PERCEPTION_DIR': str(tmp_path / 'perception'),
        'PAAI_HTTP_BIND': '100.101.102.103', 'PAAI_ACCESS_MODE': 'tailscale',
        'PAAI_MODEL_ID': 'Qwen/Qwen3.8-27B', 'PAAI_CONTEXT_WINDOW': '16384',
        'LD_LIBRARY_PATH': '/opt/paai/cuda/lib:/usr/local/nvidia/lib64',
        'UNRELATED_SECRET': 'must-not-enter-child-environment',
    }
    with patch.dict(os.environ, env, clear=True), patch.object(runtime, 'private_run',
            return_value=subprocess.CompletedProcess([], 0, 'v24.16.0\n', '')):
        yield env
    runtime.ROOT, runtime.HERE = original_root, original_here


def test_container_environment_uses_external_data_and_preserves_gpu_runtime(configured):
    env = runtime.runtime_environment()
    assert env['ISAACSIM_PYTHON_EXE'] == configured['PAAI_ISAAC_PYTHON']
    assert env['CASCADE_GPU_MODEL'] == configured['PAAI_MODEL_FILE']
    assert env['CASCADE_GPU_MMPROJ'] == configured['PAAI_MMPROJ_FILE']
    assert env['PAAI_CONTEXT_WINDOW'] == '16384'
    assert env['CASCADE_GPU_MODEL_ID'] == 'Qwen/Qwen3.8-27B'
    assert env['OPENCLAW_STATE_DIR'] == configured['PAAI_OPENCLAW_STATE_DIR']
    assert env['OPENCLAW_SUPERVISOR_MODE'] == 'external'
    assert env['LD_LIBRARY_PATH'].endswith(configured['LD_LIBRARY_PATH'])
    assert 'UNRELATED_SECRET' not in env
    assert 'PYTHONPATH' not in env
    assert Path(env['CASCADE_GRASP_MEMORY_PATH']).is_relative_to(runtime.state_dir())
    assert Path(env['TMPDIR']).is_relative_to(Path(configured['PAAI_DATA_DIR']))


def test_exited_gateway_fails_startup_immediately_and_reaps_owned_children(configured):
    runtime.atomic(runtime.state_dir() / 'deployment.json', {'environment': {}})
    processes = {}
    supervisor = {'pid': os.getpid(), 'argv': ['fixture-supervisor'], 'cwd': str(runtime.ROOT)}

    def launch(command, **kwargs):
        pid = 500 + len(processes)
        record = {'pid': pid, 'argv': command, 'cwd': str(kwargs['cwd'])}
        child = SimpleNamespace(pid=pid, wait=MagicMock(),
            poll=lambda: 1 if command[-3] == 'gateway' else None)
        processes[pid] = (record, child)
        return child

    with patch.object(runtime, 'remote_only'), \
            patch.object(runtime.ctypes, 'CDLL', return_value=SimpleNamespace(prctl=lambda *a: 0)), \
            patch.object(runtime.signal, 'signal'), \
            patch.object(runtime.subprocess, 'Popen', side_effect=launch), \
            patch.object(runtime, 'identity', side_effect=lambda pid: supervisor if pid == os.getpid() else processes[pid][0]), \
            patch.object(runtime, 'brain_probe'), patch.object(runtime, 'native_brain_probe'), \
            patch.object(runtime, 'simulator_probe'), patch.object(runtime, 'camera_probe'), \
            patch.object(runtime, 'http', side_effect=AssertionError('Exited gateway must not be polled')), \
            patch.object(runtime, 'process_tree', side_effect=lambda _: [supervisor, *[p[0] for p in processes.values()]]), \
            patch.object(runtime, 'terminate_tree') as cleanup:
        assert runtime.serve(1200) == 1
    result = runtime.read(runtime.state_dir() / 'supervisor.json')
    assert result['phase'] == 'failed'
    assert result['reason'] == 'OpenClaw exited during startup'
    assert len(processes) == 5
    assert cleanup.call_args.args[0] == [p[0] for p in processes.values()]
    for _, child in processes.values():
        child.wait.assert_called_once_with(timeout=5)


@pytest.mark.parametrize('value', ['0.0.0.0', '192.168.1.2', '8.8.8.8', '::', 'relative'])
def test_public_or_ambiguous_http_bind_is_rejected(value):
    with pytest.raises(ValueError):
        runtime.private_bind(value)


def test_http_visitor_url_must_name_exact_private_listener(configured):
    assert runtime.validate_urls({'tutorial': 'http://100.101.102.103:8092/guide/'})
    for url in ('http://100.101.102.104:8092/', 'http://example.com/',
                'http://127.0.0.1:8092/', 'http://100.101.102.103:8092/?token=x'):
        with pytest.raises(ValueError):
            runtime.validate_urls({'tutorial': url})


@pytest.mark.parametrize('model', [
    'qwen36-35b-a3b-gpu', 'Qwen/Qwen3.6-27B', 'nvidia/Cosmos3-Edge',
    'Qwen/Qwen3.8-8B', 'qwen/Qwen3.8-27B', 'arbitrary-substitute', '',
])
def test_replaced_model_is_rejected_before_configuration(model):
    with patch.object(runtime, 'MODEL_ID', model):
        with pytest.raises(ValueError, match='must be Qwen/Qwen3.8-27B'):
            runtime.model_identity()


def test_runtime_and_brain_plan_agree_without_model_file_overrides(configured):
    os.environ.pop('PAAI_MODEL_FILE')
    os.environ.pop('PAAI_MMPROJ_FILE')
    env = runtime.runtime_environment()
    plan = json.loads((runtime.ROOT / 'deploy/runtime/brain_qwen.json').read_text())
    assert Path(env['CASCADE_GPU_MODEL']).name == 'Qwen3.8-27B-Q8_0.gguf'
    assert Path(plan['model_path']).name == Path(env['CASCADE_GPU_MODEL']).name
    assert Path(env['CASCADE_GPU_MMPROJ']).name == 'mmproj-Qwen3.8-27B-BF16.gguf'
    assert Path(plan['mmproj_path']).name == Path(env['CASCADE_GPU_MMPROJ']).name


def test_setup_installs_current_visitor_policy_and_keeps_existing_auth(configured):
    profile = Path(configured['PAAI_OPENCLAW_STATE_DIR'])
    (profile / 'workspace').mkdir(parents=True)
    runtime.atomic(profile / 'paai-owner.json', {'root': str(runtime.ROOT)})
    instructions = profile / 'workspace/AGENTS.md'
    instructions.write_text('Existing operator note.\n')
    with patch.object(runtime, 'remote_only'), patch.object(runtime, 'openclaw',
            return_value=subprocess.CompletedProcess([], 0, '', '')):
        runtime.setup()
        token = (profile / 'gateway.token').read_bytes()
        runtime.setup()
    assert (profile / 'gateway.token').read_bytes() == token
    config = runtime.read(profile / 'openclaw.json')
    deployment = runtime.read(runtime.state_dir() / 'deployment.json')
    assert config['gateway']['bind'] == 'loopback'
    assert config['gateway']['controlUi']['root'] == configured['PAAI_OPENCLAW_UI_ROOT']
    assert config['models']['providers']['cascade-gpu']['models'][0]['contextWindow'] == 16384
    assert config['models']['providers']['cascade-gpu']['models'][0]['maxTokens'] == 4096
    assert config['models']['providers']['cascade-gpu']['timeoutSeconds'] == 180
    assert config['agents']['entries']['main']['params']['temperature'] == 0
    assert config['agents']['defaults']['compaction']['keepRecentTokens'] == 2048
    assert config['agents']['entries']['main']['params']['extra_body'] == {'max_completion_tokens': 1024}
    assert config['mcp']['sessionIdleTtlMs'] == 60000
    assert config['mcp']['servers']['cascade']['command'] == configured['PAAI_APP_PYTHON']
    assert 'PYTHONPATH' not in config['mcp']['servers']['cascade']['env']
    policy = instructions.read_text()
    template = (runtime.ROOT / 'demo/kitchen/visitor-instructions.md').read_text().strip()
    assert template in policy
    assert 'Existing operator note.' in policy
    assert 'at most once' in policy and 'camera_snapshot' in policy
    assert policy.count('<!-- BEGIN CASCADE KITCHEN VISITOR INSTRUCTIONS -->') == 1
    assert deployment['state_dir'] == configured['PAAI_STATE_DIR']
    assert deployment['brain_plan']['context_window'] == 16384
    argv = deployment['brain_plan']['argv']
    assert argv[argv.index('--ctx-size') + 1] == '16384'
    assert argv[argv.index('--model') + 1] == configured['PAAI_MODEL_FILE']


@pytest.mark.parametrize('tutorial,public_origin', [
    ('http://100.101.102.103:8092/guide/', None),
    ('https://kitchen.example/guide/', 'https://kitchen.example'),
])
def test_public_origin_only_for_https_tutorial(configured, tmp_path, tutorial, public_origin):
    urls = tmp_path / 'urls.json'
    urls.write_text(json.dumps({'tutorial': tutorial}))
    with patch.object(runtime, 'remote_only'), patch.object(runtime, 'openclaw',
            return_value=subprocess.CompletedProcess([], 0, '', '')):
        runtime.setup(urls_file=urls)
    gateway = runtime.read(Path(configured['PAAI_OPENCLAW_STATE_DIR']) / 'openclaw.json')['gateway']
    assert gateway['bind'] == 'loopback'
    assert tutorial.removesuffix('/guide/') in gateway['controlUi']['allowedOrigins']
    if public_origin is None:
        assert 'publicOrigin' not in gateway
    else:
        assert gateway['publicOrigin'] == public_origin


@pytest.mark.parametrize('name,vram,accepted', [
    ('NVIDIA RTX PRO 6000 Blackwell Server Edition', 97887, True),
    ('NVIDIA RTX PRO 6000 Blackwell Server Edition', 88000, False),
    ('NVIDIA L40S', 97887, False),
    ('NVIDIA RTX PRO 6000', 97535, False),
])
def test_gpu_preflight_binds_card_and_memory(name, vram, accepted):
    with patch.dict(os.environ, {}, clear=True), patch.object(runtime, 'private_run',
            return_value=subprocess.CompletedProcess([], 0, f'{name}, {vram}, 30\n', '')):
        if accepted:
            assert runtime.gpu_info()['name'] == name
        else:
            with pytest.raises(ValueError):
                runtime.gpu_info()


def visitor(peer='100.90.80.70', local='100.101.102.103', **extra_headers):
    headers = CIMultiDict({'Origin': 'http://100.101.102.103:8092',
        'X-PAAI-Action': 'openclaw', 'Sec-Fetch-Site': 'same-origin', **extra_headers})
    transport = SimpleNamespace(get_extra_info=lambda field: {
        'peername': (peer, 53000), 'sockname': (local, 8092)}.get(field))
    return SimpleNamespace(method='POST', host='100.101.102.103:8092', headers=headers,
        content_type='application/json', content_length=2, transport=transport,
        json=AsyncMock(return_value={}), path='/api/openclaw-bootstrap')


def web_instance(tmp_path):
    return web_runtime.WebRuntime(HERE.parents[1], {
        'access_mode': 'tailscale', 'web_bind': '100.101.102.103',
        'origins': ['http://100.101.102.103:8092'], 'state_dir': str(tmp_path),
        'environment': {}, 'brain_plan': {'model_id': 'Qwen/Qwen3.8-27B'},
    }, None)


@pytest.mark.parametrize('peer,local', [
    ('8.8.8.8', '100.101.102.103'), ('127.0.0.1', '100.101.102.103'),
    ('100.90.80.70', '127.0.0.1'), ('100.90.80.70', '100.101.102.104'),
])
def test_grant_requires_real_tailnet_peer_and_exact_listener(tmp_path, peer, local):
    app = web_instance(tmp_path)
    request = visitor(peer, local, **{'X-Forwarded-For': '100.90.80.70',
        'Cf-Access-Jwt-Assertion': 'untrusted', 'Tailscale-User-Login': 'untrusted'})
    with patch.object(runtime, 'openclaw') as mint:
        response = asyncio.run(app.bootstrap(request))
    assert response.status == 403
    mint.assert_not_called()


@pytest.mark.parametrize('change', ['origin', 'duplicate_action', 'cross_site', 'large_body', 'unbounded_body'])
def test_authenticated_private_peer_still_needs_exact_csrf_contract(tmp_path, change):
    request = visitor()
    if change == 'origin':
        request.headers['Origin'] = 'https://untrusted.example'
    elif change == 'duplicate_action':
        request.headers.add('X-PAAI-Action', 'openclaw')
    elif change == 'cross_site':
        request.headers['Sec-Fetch-Site'] = 'cross-site'
    elif change == 'large_body':
        request.content_length = 2049
    else:
        request.content_length = None
    with patch.object(runtime, 'openclaw') as mint:
        response = asyncio.run(web_instance(tmp_path).bootstrap(request))
    assert response.status == 403
    mint.assert_not_called()


def test_private_grant_is_native_finite_and_never_a_url(tmp_path):
    payload = {'browserUrl': 'http://127.0.0.1:18790/#bootstrapToken=' + 'fixture' * 8 + '&bootstrapProfile=owner',
               'browserBootstrapExpiresAtMs': time.time() * 1000 + 300000}
    with patch.object(runtime, 'openclaw', return_value=subprocess.CompletedProcess(
            [], 0, json.dumps(payload), '')) as mint:
        response = asyncio.run(web_instance(tmp_path).bootstrap(visitor()))
    result = json.loads(response.body)
    assert response.status == 200
    assert result['bootstrapToken'] == 'fixture' * 8
    assert 'browserUrl' not in result
    assert response.headers['Cache-Control'] == 'no-store'
    assert mint.call_args.kwargs['timeout'] == 20


@pytest.mark.parametrize('failure', ['timeout', 'malformed', 'unexpected_fields'])
def test_unfinished_or_invalid_browser_body_cannot_hold_or_mint_access(tmp_path, failure):
    request = visitor()
    if failure == 'timeout':
        request.json.side_effect = asyncio.TimeoutError
    elif failure == 'malformed':
        request.json.side_effect = ValueError('private request body')
    else:
        request.json.return_value = {'private_request_body': 'unexpected'}
    app = web_instance(tmp_path)
    with patch.object(runtime, 'openclaw') as mint:
        response = asyncio.run(app.bootstrap(request))
    assert response.status == 400
    assert json.loads(response.body) == {'error': 'invalid_request'}
    assert not app.bootstrap_lock.locked()
    mint.assert_not_called()


def test_camera_page_chat_link_follows_visitor_origin():
    from cascade.apps.stream_server import _make_handler
    server = SimpleNamespace(state=lambda: {'read_only': True}, _rig=SimpleNamespace(names=['worktop']))
    handler = object.__new__(_make_handler(server))
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler._index()
    page = handler.wfile.getvalue().decode()
    assert 'href="/openclaw/?connect=1"' in page
    assert '100.101.102.104' not in page
    for origin in ('http://100.101.102.103:8092', 'https://kitchen.example'):
        assert urljoin(origin + '/cameras/', '/openclaw/?connect=1') == origin + '/openclaw/?connect=1'


@pytest.mark.parametrize('external', [None, 'http://100.101.102.103:8092/cameras/'])
def test_standalone_camera_reports_configured_or_local_url(external, capsys):
    module_path = HERE.parents[1] / 'demo/serve_isaac_view.py'
    spec = importlib.util.spec_from_file_location('portable_camera_view', module_path)
    camera = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {}, clear=True):
        if external:
            os.environ['CASCADE_EXTERNAL_VIEW_URL'] = external
        spec.loader.exec_module(camera)
    rig = SimpleNamespace(state=lambda: {}, open=MagicMock(), close=MagicMock())
    server = SimpleNamespace(start=MagicMock(), stop=MagicMock())
    with patch.object(camera, 'RecoveringViewRig', return_value=rig):
        camera.serve(SimpleNamespace(wait=lambda: None), server_factory=MagicMock(return_value=server))
    assert json.loads(capsys.readouterr().out)['url'] == (external or 'http://127.0.0.1:8091/')


def test_current_guide_and_bootstrap_use_same_window_receipt(tmp_path):
    app = web_instance(tmp_path)
    guide = app.static_response(app.guide, 'index.html').body.decode()
    bootstrap = app.static_response(app.here / 'web', 'browser_bootstrap.js').body.decode()
    assert '/learner-flow.js' in guide
    assert 'data-learner-openclaw' in guide
    assert 'window.opener?.postMessage' in bootstrap
    assert "'paai-openclaw-ready'" in bootstrap
    assert "snapshot.hello?.type==='hello-ok'" in bootstrap


def test_web_only_listens_on_loopback_and_selected_tailnet_address(tmp_path):
    runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
    site = MagicMock(return_value=SimpleNamespace(start=AsyncMock()))
    deployment = {'web_bind': '100.101.102.103', 'access_mode': 'tailscale'}
    with patch.object(web_runtime.web, 'AppRunner', return_value=runner), \
            patch.object(web_runtime.web, 'TCPSite', site):
        asyncio.run(web_runtime.run_web(HERE.parents[1], deployment, .001))
    assert {(call.args[1], call.args[2]) for call in site.call_args_list} == {
        (address, port) for address in ('127.0.0.1', '100.101.102.103') for port in (8092, 18791)}
    runner.cleanup.assert_awaited_once()
