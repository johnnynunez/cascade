#!/usr/bin/env python3
"""Brev kitchen runtime with owned CUDA processes and bounded startup.

The deploy orchestrator calls setup/start/health/stop. No service manager is
required. Credentials remain in the instance user's private OpenClaw profile.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REQUIRED_MODEL_ID = 'Qwen/Qwen3.8-27B'
MODEL_ID = os.environ.get('PAAI_MODEL_ID', REQUIRED_MODEL_ID)
REQUIRED_TOOLS = {'cascade__pick_and_place', 'cascade__get_observation',
                  'cascade__analyze_scene', 'cascade__world_state', 'cascade__reset_scene'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + secrets.token_hex(6) + '.tmp')
    with temporary.open('x') as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def instance_identity():
    return os.environ.get('PAAI_INSTANCE') or socket.gethostname().split('.')[0]


def remote_only():
    if platform.machine() != 'x86_64':
        raise ValueError('The Brev container requires x86_64')


def private_run(command, *, timeout=30, env=None, cwd=None, check=True):
    """Never print subprocess output: configuration commands can contain secrets."""
    child = subprocess.Popen([str(x) for x in command], cwd=cwd or ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException:
        # Native CLIs can respawn. Closing only the outer PID is insufficient.
        records = process_tree(child.pid)
        terminate_tree(records, timeout=3)
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.communicate(timeout=3)
        raise
    if check and child.returncode:
        raise RuntimeError(f'{Path(str(command[0])).name} failed (exit {child.returncode}); output withheld')
    return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def select_root(root):
    global ROOT, HERE
    ROOT = Path(root).expanduser().resolve()
    HERE = ROOT / 'deploy/runtime'
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'demo/kitchen'),
                    str(ROOT / 'demo')]


def state_dir():
    return configured_path('PAAI_STATE_DIR', '/data/private/runtime')


def configured_path(name, default):
    value = Path(os.environ.get(name, str(default))).expanduser()
    if not value.is_absolute() or any(ord(c) < 32 for c in str(value)):
        raise ValueError(f'{name} must be an absolute path without control characters')
    return value


def app_python():
    return configured_path('PAAI_APP_PYTHON', '/opt/paai/venv/bin/python')


def private_bind(value):
    address = ipaddress.ip_address(value)
    if address.version != 4 or not (address.is_loopback or address in ipaddress.ip_network('100.64.0.0/10')):
        raise ValueError('HTTP bind must be loopback or the target Tailscale IPv4 address')
    return str(address)


def access_mode():
    mode = os.environ.get('PAAI_ACCESS_MODE', 'tailscale')
    if mode not in ('tailscale', 'cloudflare'):
        raise ValueError('PAAI_ACCESS_MODE must be tailscale or cloudflare')
    return mode


def model_identity():
    if MODEL_ID != REQUIRED_MODEL_ID:
        raise ValueError('The production model must be Qwen/Qwen3.8-27B')
    return MODEL_ID


def identity(pid):
    if type(pid) is not int or pid <= 1:
        return None
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z' or path.stat().st_uid != os.getuid():
            return None
        command = (path / 'cmdline').read_bytes().rstrip(b'\0')
        if not command:
            return None  # Exiting processes can clear cmdline just before becoming zombies.
        return {'pid': pid, 'birth': fields[19],
                'boot': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'argv': command.decode().split('\0'),
                'cwd': str((path / 'cwd').resolve())}
    except (OSError, ValueError):
        return None


def live(record):
    current = identity(record.get('pid')) if isinstance(record, dict) else None
    return current is not None and current == record


def process_tree(pid):
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and entry.stat().st_uid == os.getuid():
                parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError):
            pass
    owned = {pid}
    for _ in range(len(parents) + 1):
        extra = {child for child, parent in parents.items() if parent in owned} - owned
        if not extra:
            break
        owned.update(extra)
    return [record for child in sorted(owned) if (record := identity(child))]


def terminate_tree(records, timeout=35):
    for record in records:
        if live(record):
            with contextlib.suppress(ProcessLookupError):
                os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(live(row) for row in records):
        time.sleep(.1)
    for record in reversed(records):
        if live(record):
            with contextlib.suppress(ProcessLookupError):
                os.kill(record['pid'], signal.SIGKILL)


@contextlib.contextmanager
def lock(name, timeout=10):
    state_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state_dir() / name).open('a') as handle:
        deadline = time.monotonic() + timeout
        acquired = False
        while time.monotonic() < deadline:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(.1)
        if not acquired:
            raise TimeoutError('Another deployment operation holds the runtime lock')
        yield


def http(url, *, timeout=3, body=None, as_json=True):
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        payload = response.read(4 * 1024 * 1024)
        return json.loads(payload) if as_json else response.status


def validate_urls(value):
    source = value.get('urls', value) if isinstance(value, dict) else {}
    urls = {}
    for key in ('tutorial', 'openclaw', 'cameras'):
        url = source.get(key)
        if not url:
            continue
        parsed = urlsplit(url)
        private_http = (parsed.scheme == 'http' and access_mode() == 'tailscale'
                        and parsed.hostname == private_bind(os.environ.get('PAAI_HTTP_BIND', '127.0.0.1'))
                        and not ipaddress.ip_address(parsed.hostname).is_loopback)
        if (not (parsed.scheme == 'https' or private_http) or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or any(ord(c) < 33 for c in url)):
            raise ValueError('Visitor URLs must use HTTPS or the exact configured Tailscale address')
        urls[key] = url.rstrip('/') + '/'
    return urls


def runtime_environment():
    node = configured_path('PAAI_NODE', shutil.which('node') or '/usr/local/bin/node')
    version = private_run([node, '--version'], timeout=5).stdout.strip().lstrip('v')
    parts = tuple(int(x) for x in version.split('.'))
    if not ((24, 16, 0) <= parts < (25, 0, 0) or parts >= (26, 1, 0)):
        raise ValueError('OpenClaw requires Node >=24.16<25 or >=26.1')
    # Start from a narrow non-secret environment; no unrelated caller settings.
    env = {key: os.environ[key] for key in ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL',
        'XDG_RUNTIME_DIR', 'DBUS_SESSION_BUS_ADDRESS', 'NVIDIA_VISIBLE_DEVICES',
        'NVIDIA_DRIVER_CAPABILITIES', 'VK_ICD_FILENAMES', '__GLX_VENDOR_LIBRARY_NAME') if key in os.environ}
    data = configured_path('PAAI_DATA_DIR', '/data')
    model_dir = configured_path('PAAI_MODEL_DIR', data / 'models/qwen3.8-27b')
    perception = configured_path('PAAI_PERCEPTION_DIR', ROOT / 'models')
    llama = configured_path('PAAI_LLAMA_SERVER', '/opt/paai/llama/llama-server')
    profile = os.environ.get('PAAI_OPENCLAW_PROFILE', 'cascade')
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', profile):
        raise ValueError('Invalid private OpenClaw profile name')
    context = int(os.environ.get('PAAI_CONTEXT_WINDOW', '16384'))
    if not 4096 <= context <= 32768:
        raise ValueError('The context window must be between 4096 and 32768 tokens')
    libraries = str(llama.parent)
    if os.environ.get('LD_LIBRARY_PATH'):
        libraries += os.pathsep + os.environ['LD_LIBRARY_PATH']
    env.update(PATH=f'{node.parent}:/usr/local/bin:/usr/bin:/bin',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH=libraries,
        CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8', NUMEXPR_MAX_THREADS='8',
        OPENCLAW_STATE_DIR=str(configured_path('PAAI_OPENCLAW_STATE_DIR', data / 'private/openclaw')),
        OPENCLAW_PROFILE=profile, CASCADE_OPENCLAW_PROFILE=profile,
        OPENCLAW_SUPERVISOR_MODE='external',
        PAAI_OPENCLAW_CLI=str(configured_path('PAAI_OPENCLAW_CLI', '/opt/paai/openclaw/node_modules/openclaw/openclaw.mjs')),
        PAAI_APP_PYTHON=str(app_python()), PAAI_STATE_DIR=str(state_dir()),
        PAAI_DATA_DIR=str(data), PAAI_MODEL_ID=model_identity(), PAAI_CONTEXT_WINDOW=str(context),
        PAAI_INSTANCE=instance_identity(),
        PAAI_HTTP_BIND=private_bind(os.environ.get('PAAI_HTTP_BIND', '127.0.0.1')),
        PAAI_ACCESS_MODE=access_mode(), PAAI_OPENCLAW_PROFILE=profile,
        PAAI_PERCEPTION_DIR=str(perception),
        PAAI_EXPECTED_GPU=os.environ.get('PAAI_EXPECTED_GPU', 'NVIDIA RTX PRO 6000 Blackwell Server Edition'),
        PAAI_MIN_VRAM_MIB=os.environ.get('PAAI_MIN_VRAM_MIB', '90000'),
        PAAI_OPENCLAW_UI_ROOT=str(configured_path('PAAI_OPENCLAW_UI_ROOT', data / 'web/openclaw-ui')),
        CASCADE_LAUNCH_STATE=str(state_dir() / 'launch'),
        XDG_CONFIG_HOME=str(data / 'config'), XDG_CACHE_HOME=str(data / 'cache'),
        CUDA_CACHE_PATH=str(data / 'cache/cuda'), WARP_CACHE_PATH=str(data / 'cache/warp'),
        __GL_SHADER_DISK_CACHE_PATH=str(data / 'cache/gl'), TMPDIR=str(data / 'tmp'),
        OMNI_KIT_ACCEPT_EULA='YES', ACCEPT_EULA='Y',
        ISAACSIM_PYTHON_EXE=str(configured_path('PAAI_ISAAC_PYTHON', '/isaac-sim/python.sh')),
        CASCADE_PHYSICS_DEVICE='cuda:0', CASCADE_REQUIRE_CUDA='1', CASCADE_DEVICE='cuda:0',
        CASCADE_BRIDGE_BIND='127.0.0.1', CASCADE_PROOF_CAMERA='1',
        CASCADE_ISAAC_WIDTH='960', CASCADE_ISAAC_HEIGHT='540', CASCADE_ISAAC_CAM_EVERY='4',
        CASCADE_ISAAC_DT=str(1 / 120), CASCADE_VIEW='0', CASCADE_MJ_VIEW='0',
        CASCADE_BELIEFS='0', CASCADE_OCCUPANCY='0', CASCADE_STREAM='0',
        CASCADE_GRASP_MEMORY_PATH=str(state_dir() / 'memory/grasp.json'),
        CASCADE_ENVELOPE_PATH=str(state_dir() / 'memory/envelope.json'),
        CASCADE_BELIEFS_PATH=str(state_dir() / 'memory/beliefs.json'),
        CASCADE_GPU_PERCEPTION_EVIDENCE_DIR=str(state_dir() / 'perception'),
        CASCADE_QWEN_BASE_URL='http://127.0.0.1:8041/v1',
        CASCADE_JUDGE_CONFIG=str(state_dir() / 'judge.json'), CASCADE_PROOF_TIMEOUT_SCALE='2',
        CASCADE_GPU_PROFILE='qwen', CASCADE_CAMERAS='isaac,isaac_side,isaac_proof',
        CASCADE_ARM='isaac_kitchen_gpu', CASCADE_DETECTOR_MODEL=str(perception / 'yoloe-11s-seg.pt'),
        YOLO_OFFLINE='True', ULTRALYTICS_OFFLINE='True', CASCADE_PREWARM='0',
        CASCADE_GPU_SERVER=str(llama),
        CASCADE_GPU_MODEL=str(configured_path('PAAI_MODEL_FILE', model_dir / 'Qwen3.8-27B-Q8_0.gguf')),
        CASCADE_GPU_MMPROJ=str(configured_path('PAAI_MMPROJ_FILE', model_dir / 'mmproj-Qwen3.8-27B-BF16.gguf')),
        CASCADE_GPU_MODEL_ID=model_identity())
    for name in ('CASCADE_ISAAC_WIDTH', 'CASCADE_ISAAC_HEIGHT', 'CASCADE_ISAAC_CAM_EVERY', 'CASCADE_ISAAC_DT'):
        if name in os.environ:
            env[name] = os.environ[name]
    if os.environ.get('PAAI_CAMERA_VIDEO_CONFIG'):
        env['PAAI_CAMERA_VIDEO_CONFIG'] = str(configured_path('PAAI_CAMERA_VIDEO_CONFIG', '/data/camera-video.json'))
    for name in ('TMPDIR', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH',
                 'WARP_CACHE_PATH', '__GL_SHADER_DISK_CACHE_PATH'):
        Path(env[name]).mkdir(mode=0o700, parents=True, exist_ok=True)
    return env


def openclaw(command, *, timeout=30, env=None):
    env = env or read(state_dir() / 'deployment.json')['environment']
    executable = env['PAAI_OPENCLAW_CLI']
    node = shutil.which('node', path=env['PATH'])
    return private_run([node, executable, '--profile', env['OPENCLAW_PROFILE'], *command], env=env, timeout=timeout)


def camera_video_configuration(env, origins, web_bind):
    if not env.get('PAAI_CAMERA_VIDEO_CONFIG'):
        return None
    sys.path.insert(0, str(ROOT / 'deploy/brev/streaming'))
    from camera_stream_lifecycle import camera_video_from_environment
    video = camera_video_from_environment(env)
    if video is None:
        return None
    if (video.profile.tailnet_ipv4 != web_bind or video.profile.visitor_origin not in origins
            or video.profile.http_port != 8889 or video.profile.media_port != 8189
            or video.profile.rtsp_ports != (8554, 8555, 8556)):
        raise ValueError('Camera video must match the private frontend and admitted relay ports')
    return {'transport': 'whep', 'live_verified': False}


def setup(urls_file=None, sso_confirmed=False, sso_issuer=None, sso_audience=None):
    remote_only()
    env = runtime_environment()
    for value in (env['PAAI_APP_PYTHON'], env['PAAI_OPENCLAW_CLI'], env['ISAACSIM_PYTHON_EXE']):
        if not Path(value).is_file():
            raise ValueError('The configured container runtime executable is missing')
    urls = validate_urls(read(urls_file)) if urls_file else read(state_dir() / 'deployment.json').get('urls', {})
    web_bind = env['PAAI_HTTP_BIND']
    if env['PAAI_ACCESS_MODE'] == 'tailscale' and not ipaddress.ip_address(web_bind).is_loopback and not urls:
        origin = f'http://{web_bind}:8092'
        urls = {'tutorial': origin + '/', 'openclaw': origin + '/openclaw/', 'cameras': origin + '/cameras/'}
    origins = sorted({urlsplit(url).scheme + '://' + urlsplit(url).netloc for url in urls.values()})
    camera_video = camera_video_configuration(env, origins, web_bind)
    if sso_confirmed and not urls.get('tutorial'):
        raise ValueError('SSO bootstrap needs a verified tutorial HTTPS link')
    sso = {}
    if sso_issuer or sso_audience:
        import re
        if not (isinstance(sso_issuer, str) and re.fullmatch(r'https://[a-z0-9-]+\.cloudflareaccess\.com', sso_issuer)
                and isinstance(sso_audience, str) and re.fullmatch(r'[a-f0-9]{64}', sso_audience)):
            raise ValueError('SSO requires the verified Cloudflare Access issuer and exact application audience')
        sso = {'issuer': sso_issuer, 'audience': sso_audience}
    profile = Path(env['OPENCLAW_STATE_DIR'])
    marker = profile / 'paai-owner.json'
    if profile.exists() and not marker.exists() and any(profile.iterdir()):
        raise ValueError('An existing OpenClaw profile is unowned; refusing to replace it')
    if marker.exists() and read(marker).get('root') != str(ROOT):
        raise ValueError('OpenClaw profile belongs to another deployment')
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(profile, 0o700)
    atomic(marker, {'root': str(ROOT), 'instance': instance_identity()})
    token_file = profile / 'gateway.token'
    if not token_file.exists():
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(secrets.token_urlsafe(48) + '\n')
    os.chmod(token_file, 0o600)
    token = token_file.read_text().strip()
    if len(token) < 32:
        raise ValueError('The local gateway credential is invalid')
    workspace = profile / 'workspace'
    workspace.mkdir(mode=0o700, exist_ok=True)
    from visitor_instructions import sync_visitor_instructions
    sync_visitor_instructions(workspace)
    from cascade.apps.process_owner import load_owner
    profile_name = env['OPENCLAW_PROFILE']
    launch = state_dir() / ('launch/profile-' + profile_name)
    owner = load_owner(launch, ROOT, profile_name, create=True)
    mcp_env = {key: value for key, value in env.items() if key.startswith(('CASCADE_', 'PYTHON', 'OMP_',
                                                                        'MKL_', 'OPENBLAS_', 'YOLO_', 'ULTRALYTICS_'))}
    provider = {'baseUrl': 'http://127.0.0.1:8041/v1', 'api': 'openai-completions',
        'models': [{'id': MODEL_ID, 'name': MODEL_ID, 'input': ['text', 'image'],
                    'contextWindow': int(env['PAAI_CONTEXT_WINDOW']), 'maxTokens': 4096, 'reasoning': False}]}
    config = {'gateway': {'mode': 'local', 'port': 18790, 'bind': 'loopback',
                    'auth': {'mode': 'token', 'token': token},
                    'controlUi': {'basePath': '/openclaw',
                        'root': env['PAAI_OPENCLAW_UI_ROOT'],
                        'allowedOrigins': origins + ['http://127.0.0.1:18790', 'http://localhost:18790',
                                                    'http://127.0.0.1:18791', 'http://127.0.0.1:8092',
                                                    'http://127.0.0.1:8888']},
                    'trustedProxies': ['127.0.0.1', '::1']},
        'models': {'mode': 'merge', 'providers': {'cascade-gpu': provider}},
        'agents': {'defaults': {'workspace': str(workspace), 'model': {'primary': 'cascade-gpu/' + MODEL_ID},
                                'skills': [], 'skipBootstrap': True},
                   'entries': {'main': {'name': 'main', 'workspace': str(workspace),
                     'agentDir': str(profile / 'agents/main/agent'), 'identity': {'name': 'Kitchen robot'},
                     'model': 'cascade-gpu/' + MODEL_ID, 'params': {'temperature': 0}}}},
        'tools': {'profile': 'coding', 'allow': ['cascade__*']},
        'mcp': {'sessionIdleTtlMs': 900000, 'servers': {'cascade': {'command': env['PAAI_APP_PYTHON'],
            'args': ['-m', 'cascade.apps.mcp_server', '--launch-owner', owner['owner'], '--launch-state-dir', str(launch)],
            'cwd': env['PAAI_PERCEPTION_DIR'], 'env': mcp_env, 'connectionTimeoutMs': 120000, 'requestTimeoutMs': 300000}}}}
    if urls.get('tutorial') and urlsplit(urls['tutorial']).scheme == 'https':
        config['gateway']['publicOrigin'] = urlsplit(urls['tutorial']).scheme + '://' + urlsplit(urls['tutorial']).netloc
    atomic(profile / 'openclaw.json', config)
    del token, config
    openclaw(['config', 'validate'], timeout=40, env=env)
    os.environ.update(env)
    import brain
    brain_plan = brain.make_plan()
    brain_plan['argv'][brain_plan['argv'].index('--threads-batch') + 1] = '12'
    brain_plan['argv'][brain_plan['argv'].index('--ctx-size') + 1] = env['PAAI_CONTEXT_WINDOW']
    brain_plan['context_window'] = int(env['PAAI_CONTEXT_WINDOW'])
    brain_plan['plan_sha256'] = hashlib.sha256(json.dumps(brain_plan['argv']).encode()).hexdigest()
    brain_plan['state_file'] = str(state_dir() / 'brain-state.json')
    brain_plan['plan_file'] = str(state_dir() / 'brain-plan.json')
    atomic(brain_plan['plan_file'], brain_plan)
    atomic(state_dir() / 'judge.json', {'backend': 'vlm', 'base_url': provider['baseUrl'],
        'api_key': 'local-endpoint-no-auth', 'model': MODEL_ID, 'fresh_session': False,
        'mode': 'incremental', 'temperature': .1, 'top_p': .9, 'max_tokens': 1536,
        'timeout_s': 300, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}})
    deployment = {'root': str(ROOT), 'state_dir': str(state_dir()), 'environment': env, 'urls': urls, 'origins': origins,
                  'web_bind': web_bind, 'access_mode': env['PAAI_ACCESS_MODE'],
                  'openclaw_ui_root': env['PAAI_OPENCLAW_UI_ROOT'],
                  'sso_confirmed': sso_confirmed, 'sso': sso, 'brain_plan': brain_plan, 'launch_state': str(launch),
                  'launch_owner': owner['owner'], 'token_file': str(token_file), 'setup_at': stamp()}
    if camera_video is not None:
        deployment['camera_video'] = camera_video
    atomic(state_dir() / 'deployment.json', deployment)
    return {'status': 'CONFIGURED', 'instance': instance_identity(), 'urls': urls,
            'sso_bootstrap_enabled': bool(sso_confirmed and sso), 'gateway_credential': 'instance-local'}


def gpu_info():
    output = private_run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used',
                          '--format=csv,noheader,nounits'], timeout=10).stdout.strip()
    rows = [line.split(',') for line in output.splitlines()]
    if len(rows) != 1:
        raise ValueError('Expected exactly one GPU')
    name = rows[0][0].strip()
    expected = os.environ.get('PAAI_EXPECTED_GPU', 'NVIDIA RTX PRO 6000 Blackwell Server Edition')
    minimum = float(os.environ.get('PAAI_MIN_VRAM_MIB', '90000'))
    if not expected or not math.isfinite(minimum) or minimum < 1:
        raise ValueError('GPU preflight configuration is invalid')
    if expected != name or float(rows[0][1]) < minimum:
        raise ValueError('The GPU does not satisfy the configured VRAM preflight')
    return {'name': name, 'memory_total_mib': float(rows[0][1]), 'memory_used_mib': float(rows[0][2])}


def memory_info():
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return {'total_mib': int(mem['MemTotal'].split()[0]) / 1024,
            'available_mib': int(mem['MemAvailable'].split()[0]) / 1024}


def simulator_probe():
    from cascade.sim.bridge_client import BridgeClient
    client = BridgeClient(host='127.0.0.1', port=8611, timeout_s=2)
    try:
        client.connect()
        pong = client.request({'op': 'ping'})
        state = client.request({'op': 'state'})
    finally:
        client.close()
    scene = ROOT / 'demo/scene/kitchen_config.json'
    attestation = pong.get('gpu_attestation', {})
    valid = (pong.get('ok') is True and pong.get('engine') == 'physx' and
        pong.get('scene_config') == str(scene) and
        pong.get('scene_config_sha256') == hashlib.sha256(scene.read_bytes()).hexdigest() and
        math.isclose(pong.get('physics_dt_s', 0), 1 / 120, rel_tol=1e-6) and
        pong.get('physics_gpu') is True and attestation.get('cpu_fallback_allowed') is False and
        attestation.get('device') == 'cuda:0' and state.get('q') and all(math.isfinite(x) for x in state['q']))
    if not valid:
        raise ValueError('Isaac did not prove the expected scene, live joints, and GPU PhysX')
    return {'ready': True, 'gpu_attestation': attestation, 'physics_dt_s': pong['physics_dt_s'],
            'scene_sha256': pong['scene_config_sha256'], 'joints_finite': True}


def smoke():
    """Read actual green-cube contact tensors twice; never move or reset a body."""
    from cascade.sim.bridge_client import BridgeClient
    simulator = simulator_probe()
    client = BridgeClient(host='127.0.0.1', port=8611, timeout_s=4)
    snapshots = []
    code = 'import json; print(json.dumps(_gpu_contact_snapshot("green_cube"), allow_nan=False))'
    try:
        client.connect()
        for index in range(2):
            response = client.request({'op': 'exec', 'code': code})
            rows = [line for line in response.get('stdout', '').splitlines() if line.startswith('{')]
            if len(rows) != 1:
                raise ValueError('GPU contact snapshot did not return one structured record')
            snapshot = json.loads(rows[0])
            forces = [*snapshot.get('net_force_n', []),
                      *(value for row in snapshot.get('jaw_forces_n', []) for value in row)]
            counts = snapshot.get('jaw_contact_counts', [])
            if (snapshot.get('channel') != 'physx_gpu_contact_tensor' or snapshot.get('device') != 'cuda:0'
                    or len(forces) != 9 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in forces)
                    or len(counts) != 2 or not all(isinstance(x, int) and x >= 0 for x in counts)
                    or type(snapshot.get('physics_step')) is not int):
                raise ValueError('Live contact tensors are not finite CUDA data')
            snapshots.append(snapshot)
            if index == 0:
                time.sleep(.5)
    finally:
        client.close()
    if snapshots[1]['physics_step'] <= snapshots[0]['physics_step']:
        raise ValueError('The GPU physics clock did not advance between contact snapshots')
    result = {'status': 'PASS', 'instance': instance_identity(), 'kind': 'read-only GPU physics smoke',
              'physics_gpu': True, 'device': 'cuda:0', 'object': 'green_cube',
              'physics_step_increased': True, 'snapshots': snapshots,
              'simulator': simulator, 'pick_place_verified': False, 'checked_at': stamp()}
    atomic(state_dir() / 'physics-smoke.json', result)
    return result


def camera_probe():
    state = http('http://127.0.0.1:8091/state')
    rows = state.get('cameras', [])
    cameras = {row['name']: row for row in rows} if isinstance(rows, list) else rows
    if not all(cameras.get(name, {}).get('online') is True and cameras.get(name, {}).get('frame_id', 0) > 0
               for name in ('kitchen', 'worktop', 'side')):
        raise ValueError('All three cameras need fresh frames')
    return {'ready': True, 'cameras': cameras}


def brain_probe(deployment):
    import brain
    state = read(state_dir() / 'brain-state.json')
    record = state.get('identity', {})
    current = brain.process_identity(record.get('pid'))
    plan = deployment['brain_plan']
    if not brain.same_process(record, current) or current['argv'] != plan['argv'] or state.get('gpu_violation'):
        raise ValueError('GPU brain does not match this deployment process')
    if not brain.owns_listener(current['pid'], 8041) or not brain.endpoint_health(plan):
        raise ValueError('GPU brain listener is not healthy')
    models = http('http://127.0.0.1:8041/v1/models')
    if [row.get('id') for row in models.get('data', [])] != [MODEL_ID]:
        raise ValueError('GPU brain model differs from the selected Qwen model')
    audit = brain.audit_log(Path(state['log_file']).read_text(errors='replace'))
    telemetry = brain.telemetry(current['pid'])
    return {'ready': True, 'model_id': MODEL_ID, 'backend': 'llama.cpp CUDA', 'cuda_verified': True,
            'pid': current['pid'], 'cuda': audit, 'telemetry': telemetry}


def native_brain_probe(deployment):
    import demo_proof
    began = time.monotonic()
    original_request = demo_proof.request_json
    def bounded_request(url, payload=None, timeout=5):
        remaining = 90 - (time.monotonic() - began)
        if remaining <= 0:
            raise TimeoutError('Native tool-call inference deadline expired')
        return http(url, body=payload, timeout=min(timeout, remaining))
    try:
        demo_proof.request_json = bounded_request
        result = demo_proof.probe_native_tools('http://127.0.0.1:8041/v1', MODEL_ID)
        brain_probe(deployment)
        atomic(state_dir() / 'native-brain.json', {'ready': result.get('native_tools') is True,
             'checked_at': stamp(), 'wall_s': round(time.monotonic() - began, 3)})
    finally:
        demo_proof.request_json = original_request


def native_browser_handoff_probe(deployment):
    """Prove native owner-grant issuance on the instance, without browser redemption.

    Only explicit deep checks call this. The native command owns its expiring
    grant store; its credential-bearing JSON remains in this process's memory.
    This receipt does not prove public SSO forwarding or an authenticated browser.
    """
    remote_only()
    response = None
    try:
        response = openclaw(['dashboard', '--json'], timeout=25, env=deployment['environment'])
        payload = json.loads(response.stdout)
        browser = urlsplit(payload.get('browserUrl', ''))
        fragment = parse_qs(browser.fragment, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
        if (payload.get('ok') is not True or browser.scheme != 'http' or browser.hostname != '127.0.0.1'
                or browser.port != 18790 or browser.path != '/openclaw/' or browser.query
                or browser.username or browser.password
                or set(fragment) != {'bootstrapToken', 'bootstrapProfile', 'gatewayUrl'}
                or any(len(values) != 1 for values in fragment.values())
                or fragment['bootstrapProfile'] != ['owner']
                or fragment['gatewayUrl'] != ['ws://127.0.0.1:18790/openclaw']):
            raise ValueError()
        token = fragment['bootstrapToken'][0]
        expiry = payload.get('browserBootstrapExpiresAtMs')
        if (not 16 <= len(token) <= 2048 or any(not 33 <= ord(char) <= 126 for char in token)
                or type(expiry) not in (int, float) or not math.isfinite(expiry)):
            raise ValueError()
        remaining_s = (expiry - time.time() * 1000) / 1000
        if not 1 < remaining_s < 660:
            raise ValueError()
        return {'ready': True, 'remaining_ttl_s': round(remaining_s, 3)}
    except Exception:
        # Do not propagate native output or a JSON parser's credential-bearing document.
        raise ValueError('Native owner browser handoff capability was not verified') from None
    finally:
        if response is not None:
            response.stdout = response.stderr = ''


def deep_probe(deployment):
    authentication = private_run([shutil.which('node', path=deployment['environment']['PATH']),
        HERE / 'verify_gateway.mjs', state_dir() / 'deployment.json'],
        env=deployment['environment'], timeout=35)
    auth = json.loads(authentication.stdout)
    if auth.get('authenticated_websocket') is not True:
        raise ValueError('Authenticated OpenClaw WebSocket health failed')
    catalogue = json.loads(openclaw(['mcp', 'probe', 'cascade', '--json'], timeout=60).stdout)
    names = set(catalogue.get('tools', []))
    if not REQUIRED_TOOLS <= names or len(names) < 41:
        raise ValueError('The native CASCADE MCP catalogue is incomplete')
    result = {'authentication': {'ready': True, 'authenticated_websocket': True},
              'tools': {'ready': True, 'count': len(names), 'names': sorted(names)},
              'browser_handoff': native_browser_handoff_probe(deployment),
              'checked_at': stamp(), 'supervisor': read(state_dir() / 'supervisor.json').get('identity')}
    atomic(state_dir() / 'deep-health.json', result)
    return result


def health(deep=False):
    deployment = read(state_dir() / 'deployment.json')
    owned = read(state_dir() / 'supervisor.json')
    components = {}
    checks = {'brain': lambda: brain_probe(deployment), 'simulator': simulator_probe, 'cameras': camera_probe,
              'gateway': lambda: {'ready': http('http://127.0.0.1:18790/health', as_json=False) == 200},
              'tutorial': lambda: {'ready': http('http://127.0.0.1:8092/', as_json=False) == 200},
              'proxy': lambda: {'ready': http('http://127.0.0.1:18791/openclaw/', as_json=False) == 200}}
    for key, check in checks.items():
        try:
            components[key] = check()
        except Exception as error:
            components[key] = {'ready': False, 'error': type(error).__name__}
    if deep:
        try:
            deep_probe(deployment)
        except Exception as error:
            components['authentication'] = {'ready': False, 'error': type(error).__name__}
            components['tools'] = {'ready': False}
            components['browser_handoff'] = {'ready': False}
    saved = read(state_dir() / 'deep-health.json')
    if saved.get('supervisor') == owned.get('identity') and live(owned.get('identity')):
        for key in ('authentication', 'tools', 'browser_handoff'):
            components.setdefault(key, saved.get(key, {'ready': False}))
    for key in ('authentication', 'tools', 'browser_handoff'):
        components.setdefault(key, {'ready': False})
    supervisor_live = (live(owned.get('identity')) and owned.get('phase') == 'running'
                       and set(owned.get('children', {})) == {'web', 'brain', 'isaac', 'gateway', 'cameras'})
    for role, row in owned.get('children', {}).items():
        if not live(row):
            supervisor_live = False
    ready = supervisor_live and all(row.get('ready') is True for row in components.values())
    result = {'version': 1, 'generated_at': stamp(), 'instance': instance_identity(),
              'interactive_ready': ready, 'phase': 'interactive_ready' if ready else 'starting' if supervisor_live else 'stopped',
              'components': components, 'urls': deployment.get('urls', {}), 'memory': memory_info(),
              'supervisor': owned.get('identity'),
              'browser_bootstrap_available': bool(
                  (deployment.get('access_mode') == 'tailscale' and deployment.get('web_bind') != '127.0.0.1')
                  or (deployment.get('sso_confirmed') and deployment.get('sso'))),
              'light_theme': True, 'certification': {'state': 'pending', 'message': 'Physics smoke checks live GPU state; full pick/place is separate.'}}
    try:
        result['gpu'] = gpu_info()
        apps = private_run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], timeout=10).stdout.splitlines()
        process_memory = {}
        for line in apps:
            values = [value.strip() for value in line.split(',')]
            if len(values) == 2 and values[0].isdigit():
                process_memory[int(values[0])] = values[1]
        result['gpu']['owned_processes'] = {}
        for role in ('brain', 'isaac'):
            row = owned.get('children', {}).get(role)
            children = process_tree(row['pid']) if live(row) else []
            result['gpu']['owned_processes'][role] = [
                {'pid': child['pid'], 'used_gpu_memory_mib': process_memory[child['pid']]}
                for child in children if child['pid'] in process_memory]
            if not result['gpu']['owned_processes'][role]:
                result['interactive_ready'] = False
    except Exception as error:
        result['gpu'] = {'error': type(error).__name__}
        result['interactive_ready'] = False
    atomic(state_dir() / 'health.json', result)
    return result


def stop_owned(saved):
    """Stop only the supervisor represented by this exact ownership receipt."""
    own = saved.get('identity')
    if not live(own):
        # A supervisor killed externally may leave children; only exact receipts authorize cleanup.
        records = [row for row in saved.get('children', {}).values() if live(row)]
        records = [item for row in records for item in process_tree(row['pid'])]
    else:
        records = process_tree(own['pid'])
    terminate_tree(records)


def stop():
    remote_only()
    stop_owned(read(state_dir() / 'supervisor.json'))
    return {'status': 'STOPPED', 'instance': instance_identity()}


def start(timeout=900):
    remote_only()
    deployment = read(state_dir() / 'deployment.json')
    if deployment.get('root') != str(ROOT):
        raise ValueError('Run runtime setup before start')
    gpu_info()
    deadline = time.monotonic() + timeout
    launched, expected, owned = None, None, {}
    try:
        with lock('operation.lock'):
            existing = read(state_dir() / 'supervisor.json')
            reused = (live(existing.get('identity')) and existing.get('phase') in ('starting', 'running'))
            if reused:
                owned, expected = existing, existing['identity']
            else:
                stop()
                for port in (8041, 8611, 8091, 8092, 8888, 18790, 18791):
                    try:
                        with socket.create_connection(('127.0.0.1', port), timeout=.3):
                            raise ValueError(f'Port {port} belongs to an unowned process')
                    except OSError:
                        pass
                (state_dir() / 'health.json').unlink(missing_ok=True)
                (state_dir() / 'deep-health.json').unlink(missing_ok=True)
                logfile = state_dir() / 'supervisor.log'
                fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                command = [str(app_python()), '-B', str(HERE / 'runtime.py'),
                           'serve', '--root', str(ROOT), '--timeout-s', str(timeout)]
                with os.fdopen(fd, 'a') as log:
                    launched = subprocess.Popen(command, cwd=ROOT,
                        env=deployment['environment'], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                        start_new_session=True)
                # Keep the operation lock until the new supervisor publishes its
                # own receipt. A failed receipt from the preceding run must not
                # abort or authorize cleanup of this newly launched process.
                while time.monotonic() < deadline:
                    if launched.poll() is not None:
                        raise RuntimeError('The new supervisor exited before publishing its ownership receipt')
                    candidate = identity(launched.pid)
                    if candidate and candidate['argv'] == command and candidate['cwd'] == str(ROOT):
                        expected = candidate
                        current = read(state_dir() / 'supervisor.json')
                        if current.get('identity') == expected:
                            owned = current
                            break
                    time.sleep(.05)
                else:
                    raise TimeoutError('The new supervisor did not publish its ownership receipt before the startup deadline')
        while time.monotonic() < deadline:
            snapshot = read(state_dir() / 'health.json')
            current = read(state_dir() / 'supervisor.json')
            if current.get('identity') == expected:
                owned = current
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(snapshot.get('generated_at', ''))).total_seconds()
                except (TypeError, ValueError):
                    age = float('inf')
                if (current.get('phase') == 'running' and snapshot.get('interactive_ready') is True
                        and snapshot.get('supervisor') == expected and 0 <= age < 90 and live(expected)):
                    return {**snapshot, 'reused': reused}
                if current.get('phase') in ('failed', 'stopped') or not live(expected):
                    raise RuntimeError('Runtime startup failed; consult the private component health receipt')
            elif expected and not live(expected):
                raise RuntimeError('The owned supervisor exited before startup completed')
            time.sleep(min(2, max(0, deadline - time.monotonic())))
        raise TimeoutError('Runtime startup deadline expired; all owned processes were stopped')
    except BaseException:
        # A caller interruption can arrive before any receipt has been written.
        # The unreaped Popen handle still identifies exactly the child we own.
        if launched is not None and launched.poll() is None:
            terminate_tree(process_tree(launched.pid))
        else:
            current = read(state_dir() / 'supervisor.json')
            if expected is not None and current.get('identity') == expected:
                owned = current
            stop_owned(owned)
        if launched is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                launched.wait(timeout=5)
        raise


def parent_death_signal():
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def serve(timeout):
    remote_only()
    with lock('supervisor.lock', timeout=2):
        # If a native launcher exits unexpectedly, adopt its descendants here
        # instead of losing GPU processes to PID 1 before the next health check.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise RuntimeError('Could not establish descendant cleanup ownership')
        deployment = read(state_dir() / 'deployment.json')
        os.environ.update(deployment['environment'])
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        signal.signal(signal.SIGINT, lambda *_: stopped.set())
        children, logs = {}, []
        own = identity(os.getpid())
        saved = {'identity': own, 'children': {}, 'phase': 'starting', 'started_at': stamp()}
        atomic(state_dir() / 'supervisor.json', saved)
        def launch(role):
            fd = os.open(state_dir() / f'{role}.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            log = os.fdopen(fd, 'a')
            logs.append(log)
            command = [str(app_python()), '-B', str(HERE / 'runtime.py'),
                       'worker', role, '--root', str(ROOT)]
            child = subprocess.Popen(command, cwd=ROOT, env=deployment['environment'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                preexec_fn=parent_death_signal)
            children[role] = child
            for _ in range(100):
                current = identity(child.pid)
                if current and current['argv'] == command and current['cwd'] == str(ROOT):
                    break
                time.sleep(.01)
            else:
                raise RuntimeError('A worker did not establish its exact process identity')
            saved['children'][role] = current
            atomic(state_dir() / 'supervisor.json', saved)
        failed = False
        try:
            for role in ('web', 'cameras', 'brain'):
                launch(role)
            # Finish model allocation before starting Isaac on the same GPU.
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not stopped.is_set():
                if children['brain'].poll() is not None:
                    raise RuntimeError('GPU brain exited during model allocation')
                try:
                    brain_probe(deployment)
                    break
                except Exception:
                    stopped.wait(2)
            else:
                raise TimeoutError('GPU brain did not become ready before the startup deadline')
            native_brain_probe(deployment)
            launch('isaac')
            # Gateway/MCP is brought up only after the sole simulator owns the live bridge.
            while time.monotonic() < deadline and not stopped.is_set():
                if children['isaac'].poll() is not None:
                    raise RuntimeError('Isaac exited during startup')
                try:
                    simulator_probe()
                    camera_probe()
                    break
                except Exception:
                    stopped.wait(2)
            else:
                raise TimeoutError('Isaac did not prove GPU physics before the startup deadline')
            launch('gateway')
            # Workers retain their stable wrapper identity even when native children respawn.
            while time.monotonic() < deadline and not stopped.is_set():
                if children['gateway'].poll() is not None:
                    raise RuntimeError('OpenClaw exited during startup')
                try:
                    if http('http://127.0.0.1:18790/health', as_json=False) == 200:
                        from cascade.apps.process_owner import load_owner, register_process
                        register_process(deployment['launch_state'], load_owner(deployment['launch_state'], ROOT, deployment['environment']['OPENCLAW_PROFILE']),
                                         children['gateway'].pid, 'gateway')
                        deep_probe(deployment)
                        smoke()
                        break
                except Exception:
                    stopped.wait(2)
            else:
                raise TimeoutError('OpenClaw authentication/tools did not become ready')
            saved['phase'] = 'running'
            atomic(state_dir() / 'supervisor.json', saved)
            while not stopped.is_set():
                if any(child.poll() is not None for child in children.values()):
                    raise RuntimeError('A required component exited')
                snapshot = health()
                if snapshot['memory']['available_mib'] < 768:
                    raise RuntimeError('Available RAM fell below the runtime reserve')
                stopped.wait(12)
        except Exception as error:
            failed = True
            # All locally raised lifecycle messages are fixed text; do not include
            # arbitrary native/config exception strings in the receipt.
            known = {'GPU brain exited during model allocation',
                     'GPU brain did not become ready before the startup deadline',
                     'Isaac exited during startup',
                     'Isaac did not prove GPU physics before the startup deadline',
                     'OpenClaw exited during startup',
                     'OpenClaw authentication/tools did not become ready',
                     'A required component exited', 'Available RAM fell below the runtime reserve'}
            saved.update(error=type(error).__name__, reason=str(error) if str(error) in known else 'component check failed', phase='failed')
            atomic(state_dir() / 'supervisor.json', saved)
        finally:
            # Snapshot descendants before signalling the parents, including Kit's private group.
            descendants = [record for record in process_tree(os.getpid()) if record['pid'] != os.getpid()]
            terminate_tree(descendants)
            for child in children.values():
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=5)
            for log in logs:
                log.close()
            saved.update(phase='failed' if failed else 'stopped', stopped_at=stamp())
            atomic(state_dir() / 'supervisor.json', saved)
        return 1 if failed else 0


def worker(role):
    remote_only()
    deployment = read(state_dir() / 'deployment.json')
    os.environ.update(deployment['environment'])
    if role == 'brain':
        import brain
        return brain.serve(deployment['brain_plan'])
    if role == 'cameras':
        from serve_isaac_view import RecoveringViewRig, KitchenStreamServer
        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        signal.signal(signal.SIGINT, lambda *_: stopped.set())
        rig = RecoveringViewRig(stop_event=stopped)
        server = KitchenStreamServer(rig, state_fn=rig.state, host='127.0.0.1', port=8091, fps=8, quality=90)
        try:
            server.start()
            rig.open()
            stopped.wait()
        finally:
            rig.close()
            server.stop()
        return 0
    if role == 'web':
        from web_runtime import run_web
        return asyncio.run(run_web(ROOT, deployment))
    if role == 'isaac':
        command = [deployment['environment']['ISAACSIM_PYTHON_EXE'], str(ROOT / 'scripts/isaac_bridge.py'),
            '--port', '8611', '--engine', 'physx', '--usd', str(ROOT / 'assets/usd/RS-rebot-dev-arm/RS-rebot-dev-arm.usda'),
            '--scene-config', str(ROOT / 'demo/scene/kitchen_config.json')]
    elif role == 'gateway':
        command = [shutil.which('node', path=deployment['environment']['PATH']),
                   deployment['environment']['PAAI_OPENCLAW_CLI'],
                   '--profile', deployment['environment']['OPENCLAW_PROFILE'], 'gateway', 'run', '--bind', 'loopback', '--port', '18790']
    else:
        raise ValueError('Unknown worker role')
    child = subprocess.Popen(command, cwd=ROOT, env=deployment['environment'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL if role == 'gateway' else None,
                             stderr=subprocess.DEVNULL if role == 'gateway' else None, preexec_fn=parent_death_signal)
    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    return child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('setup', 'start', 'health', 'status', 'smoke', 'stop', 'serve', 'worker'))
    parser.add_argument('role', nargs='?')
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--urls-file')
    parser.add_argument('--sso-confirmed', action='store_true')
    parser.add_argument('--sso-issuer')
    parser.add_argument('--sso-audience')
    parser.add_argument('--timeout-s', type=float, default=900)
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    select_root(args.root)
    os.umask(0o077)
    try:
        if (args.operation in ('health', 'status', 'smoke', 'serve', 'worker') and
                Path(sys.prefix).resolve() != app_python().parent.parent.resolve()):
            python = app_python()
            if not python.is_file():
                raise ValueError('The configured application Python environment is missing')
            os.execv(str(python), [str(python), '-B', str(HERE / 'runtime.py'), *sys.argv[1:]])
        if not 1 <= args.timeout_s <= 1800:
            raise ValueError('Startup timeout must be between 1 and 1800 seconds')
        if args.operation == 'setup':
            with lock('operation.lock'):
                result = setup(args.urls_file, args.sso_confirmed, args.sso_issuer, args.sso_audience)
        elif args.operation == 'start':
            result = start(args.timeout_s)
        elif args.operation in ('health', 'status'):
            result = health(args.deep)
        elif args.operation == 'smoke':
            remote_only()
            result = smoke()
        elif args.operation == 'stop':
            with lock('operation.lock'):
                result = stop()
        elif args.operation == 'serve':
            return serve(args.timeout_s)
        else:
            return worker(args.role)
        print(json.dumps(result, allow_nan=False))
        return 0 if args.operation not in ('health', 'status') or result['interactive_ready'] else 1
    except Exception as error:
        # Native/config output and request bodies are never included in this status.
        print(json.dumps({'status': 'FAILED', 'error': type(error).__name__, 'message': str(error)[:240]}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
