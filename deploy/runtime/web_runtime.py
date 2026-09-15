"""Single-origin light tutorial, cameras and native authenticated OpenClaw UI."""
from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
import html
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from urllib.parse import parse_qs, urlsplit

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

CAMERAS = ('kitchen', 'worktop', 'side')
TAILNET = ipaddress.ip_network('100.64.0.0/10')
HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length',
               'content-encoding', 'set-cookie', 'server'}


class WebRuntime:
    def __init__(self, root, deployment, session):
        self.root, self.deployment, self.session = root, deployment, session
        self.here = root / 'deploy/runtime'
        self.state = Path(deployment.get('state_dir', self.here / '.runtime'))
        self.static = root / 'web'
        self.guide = root / 'web/guide'
        self.ui = Path(deployment.get('openclaw_ui_root', self.static / 'openclaw-ui'))
        self.bootstrap_lock = asyncio.Lock()
        self.bootstrap_last = 0

    def origin(self, request):
        # Forwarded headers never establish authentication. Only preconfigured origins
        # can affect discovery links, and a signed edge assertion gates grants below.
        host = request.host.lower()
        for origin in self.deployment.get('origins', []):
            if urlsplit(origin).netloc.lower() == host:
                return origin
        if re.fullmatch(r'(127\.0\.0\.1|localhost)(?::\d{1,5})?', host):
            return 'http://' + host
        return None

    @staticmethod
    def socket_address(request, field):
        try:
            value = request.transport.get_extra_info(field)
            address = ipaddress.ip_address(value[0])
            return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address
        except (AttributeError, TypeError, ValueError, IndexError):
            return None

    def tailnet_client(self, request):
        if self.deployment.get('access_mode') != 'tailscale':
            return False
        peer = self.socket_address(request, 'peername')
        local = self.socket_address(request, 'sockname')
        return bool(peer and peer in TAILNET and local and local in TAILNET
                    and str(local) == self.deployment.get('web_bind'))

    def private_client(self, request):
        peer = self.socket_address(request, 'peername')
        return bool(peer and (peer.is_loopback or self.tailnet_client(request)))

    def proxy_attribution(self, request):
        peer = self.socket_address(request, 'peername')
        if peer is None:
            raise web.HTTPForbidden(headers=self.headers())
        # Native ingress rejects forwarded loopback clients; retain direct-local attribution.
        if peer.is_loopback:
            return {}
        headers = {'X-Forwarded-For': str(peer)}
        origin = self.origin(request)
        if origin:
            headers['X-Forwarded-Host'] = urlsplit(origin).netloc
            headers['X-Forwarded-Proto'] = urlsplit(origin).scheme
        return headers

    def bootstrap_available(self):
        try:
            private = (self.deployment.get('access_mode') == 'tailscale'
                       and ipaddress.ip_address(self.deployment.get('web_bind', '')) in TAILNET)
        except ValueError:
            private = False
        return bool(private or (self.deployment.get('sso_confirmed') and self.deployment.get('sso')))

    def headers(self, *, html_document=False):
        headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                   'Referrer-Policy': 'no-referrer'}
        if html_document:
            headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        return headers

    def status(self, request):
        from runtime import read, stamp, live
        raw = read(self.state / 'health.json')
        supervisor = read(self.state / 'supervisor.json')
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(raw.get('generated_at', ''))).total_seconds()
        except ValueError:
            age = float('inf')
        current = (0 <= age < 90 and supervisor.get('phase') == 'running'
                   and raw.get('supervisor') == supervisor.get('identity') and live(supervisor.get('identity')))
        if not current:
            raw = {**raw, 'interactive_ready': False, 'phase': 'starting' if live(supervisor.get('identity')) else 'stopped',
                   'components': {key: {'ready': False} for key in raw.get('components', {})}}
        origin = self.origin(request)
        if not origin:
            # Interface HTTP is still usable for health checks, without trusting
            # arbitrary Host values as destinations or credential delivery targets.
            origin = self.deployment.get('urls', {}).get('tutorial', '').rstrip('/')
        base = origin + '/cameras'
        components = raw.get('components', {})
        cameras = components.get('cameras', {})
        result = {'version': 1, 'generated_at': stamp(), 'interactive_ready': raw.get('interactive_ready') is True,
            'phase': raw.get('phase', 'starting'), 'components': {key: {'ready': value.get('ready') is True}
                for key, value in components.items() if isinstance(value, dict)},
            'certification': raw.get('certification', {'state': 'pending'}),
            'urls': {'tutorial': origin + '/', 'openclaw': origin + '/openclaw/?connect=1',
                     'openclaw_remote': origin + '/openclaw/?connect=1',
                     'viewer': base + '/', 'camera_view': base + '/',
                     'cameras': {name: base + '/stream/' + name for name in CAMERAS}},
            'camera_discovery': {'version': 1, 'base_url': base, 'state_url': base + '/state',
                'default_camera': 'worktop', 'live': cameras.get('ready') is True,
                'source': 'live_camera_probe' if cameras.get('ready') else 'offline_deployment_defaults',
                'cameras': [{'name': name, 'label': name.capitalize(), 'stream_url': base + '/stream/' + name} for name in CAMERAS]},
            'browser_bootstrap_available': self.bootstrap_available(),
            'model': {'name': self.deployment.get('brain_plan', {}).get('model_id', 'unconfigured'),
                      'backend': 'llama.cpp CUDA', 'device': 'CUDA'}}
        return web.json_response(result, headers=self.headers())

    async def authenticated_edge(self, request):
        if self.deployment.get('access_mode') == 'tailscale':
            # Only the real encrypted-network peer on the selected listener counts.
            # X-Forwarded-For, Cloudflare and Tailscale user headers are not evidence.
            return self.tailnet_client(request)
        settings = self.deployment.get('sso', {})
        assertion = request.headers.get('Cf-Access-Jwt-Assertion', '')
        if not self.deployment.get('sso_confirmed') or not settings or not assertion:
            return False
        node = shutil.which('node', path=self.deployment['environment']['PATH'])
        def verify():
            result = subprocess.run([node, str(self.here / 'verify_sso.mjs')], input=json.dumps({
                'assertion': assertion, 'issuer': settings['issuer'], 'audience': settings['audience']}),
                capture_output=True, text=True, timeout=8, cwd=self.root,
                env=self.deployment['environment'])
            return result.returncode == 0 and json.loads(result.stdout).get('authenticated') is True
        try:
            return await asyncio.to_thread(verify)
        except Exception:
            return False

    async def bootstrap(self, request):
        origin = self.origin(request)
        if (request.method != 'POST' or not origin or not (origin.startswith('https://') or self.tailnet_client(request)) or
                request.headers.getall('Origin', []) != [origin] or request.content_type != 'application/json' or
                request.headers.getall('X-PAAI-Action', []) != ['openclaw'] or
                request.headers.get('Sec-Fetch-Site', 'same-origin') != 'same-origin' or
                request.content_length is None or request.content_length > 2048):
            return web.json_response({'error': 'access_required'}, status=403, headers=self.headers())
        if self.bootstrap_lock.locked() or time.monotonic() - self.bootstrap_last < 2:
            return web.json_response({'error': 'retry_shortly'}, status=429, headers=self.headers())
        from runtime import openclaw
        async with self.bootstrap_lock:
            if not await self.authenticated_edge(request):
                return web.json_response({'error': 'authenticated_access_required'}, status=403, headers=self.headers())
            try:
                try:
                    body = await asyncio.wait_for(request.json(), timeout=5)
                except (asyncio.TimeoutError, ValueError, TypeError):
                    return web.json_response({'error': 'invalid_request'}, status=400, headers=self.headers())
                if body != {}:
                    return web.json_response({'error': 'invalid_request'}, status=400, headers=self.headers())
                result = await asyncio.to_thread(openclaw, ['dashboard', '--json'], timeout=20)
                payload = json.loads(result.stdout)
                fragment = parse_qs(urlsplit(payload.get('browserUrl', '')).fragment)
                token = fragment.get('bootstrapToken', [''])[0]
                expiry = payload.get('browserBootstrapExpiresAtMs', 0)
                if (fragment.get('bootstrapProfile') != ['owner'] or not 16 <= len(token) <= 2048
                        or not time.time() * 1000 + 1000 < expiry < time.time() * 1000 + 660000):
                    raise ValueError('Native handoff was not valid')
                self.bootstrap_last = time.monotonic()
                return web.json_response({'bootstrapToken': token, 'bootstrapProfile': 'owner',
                    'expiresAtMs': expiry}, headers=self.headers())
            except Exception:
                return web.json_response({'error': 'native_handoff_unavailable'}, status=503, headers=self.headers())

    def static_response(self, root, name):
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file() or file.stat().st_size > 40 * 1024 * 1024:
            return web.Response(status=404, headers=self.headers())
        data = file.read_bytes()
        mime = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        return web.Response(body=data, content_type=mime, headers=self.headers(html_document=mime == 'text/html'))

    async def proxy(self, request, *, port, path):
        attribution = self.proxy_attribution(request)
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            if request.headers.get('Origin') and request.headers.get('Origin') not in self.deployment.get('origins', []) + [
                    'http://127.0.0.1:18791', 'http://127.0.0.1:8092', 'http://127.0.0.1:8888']:
                return web.Response(status=403)
            upstream = None
            try:
                upstream = await self.session.ws_connect(f'http://127.0.0.1:{port}{path}',
                    headers={**attribution, **({'Origin': request.headers['Origin']} if request.headers.get('Origin') else {})},
                    heartbeat=30, max_msg_size=32 * 1024 * 1024, autoping=True)
                downstream = web.WebSocketResponse(heartbeat=30, max_msg_size=32 * 1024 * 1024)
                await downstream.prepare(request)
                async def forward(source, destination):
                    async for message in source:
                        if message.type == WSMsgType.TEXT:
                            await destination.send_str(message.data)
                        elif message.type == WSMsgType.BINARY:
                            await destination.send_bytes(message.data)
                        elif message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                            break
                jobs = [asyncio.create_task(forward(upstream, downstream)), asyncio.create_task(forward(downstream, upstream))]
                try:
                    await asyncio.wait(jobs, return_when=asyncio.FIRST_COMPLETED, timeout=18 * 3600)
                finally:
                    for job in jobs:
                        job.cancel()
                    await asyncio.gather(*jobs, return_exceptions=True)
                    await downstream.close()
                return downstream
            except Exception:
                return web.Response(status=502, headers=self.headers())
            finally:
                if upstream:
                    await upstream.close()
        if request.method not in ('GET', 'HEAD', 'POST') or (request.content_length or 0) > 32 * 1024 * 1024:
            return web.Response(status=405, headers=self.headers())
        headers = {key: value for key, value in request.headers.items() if key.lower() in ('accept', 'content-type', 'origin', 'authorization')}
        headers['Accept-Encoding'] = 'identity'
        headers.update(attribution)
        try:
            async with self.session.request(request.method, f'http://127.0.0.1:{port}{path}',
                    headers=headers, data=await request.read() if request.can_read_body else None,
                    allow_redirects=False, timeout=ClientTimeout(total=None, connect=5, sock_read=45)) as upstream:
                response_headers = {key: value for key, value in upstream.headers.items() if key.lower() not in HOP_HEADERS}
                response_headers.update(self.headers())
                if 'text/html' in upstream.headers.get('Content-Type', ''):
                    payload = await upstream.read()
                    if len(payload) > 8 * 1024 * 1024:
                        return web.Response(status=502, headers=self.headers())
                    text = payload.decode()
                    if port == 18790:
                        injected = ('<meta name="openclaw-demo" content="/api/status">'
                            '<script src="/paai-light.js"></script>'
                            '<link rel="stylesheet" href="/paai-light.css">'
                            '<script src="/paai-browser-bootstrap.js"></script>')
                        text = text.replace('<head>', '<head>' + injected, 1)
                    else:
                        text = text.replace('http://127.0.0.1:18790/', '/openclaw/?connect=1')
                        text = text.replace("'/stream/", "'/cameras/stream/").replace('"/stream/', '"/cameras/stream/')
                        text = text.replace("'/state'", "'/cameras/state'").replace('"/state"', '"/cameras/state"')
                    return web.Response(body=text.encode(), status=upstream.status, headers=response_headers)
                response = web.StreamResponse(status=upstream.status, headers=response_headers)
                await response.prepare(request)
                async for chunk in upstream.content.iter_chunked(64 * 1024):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (ConnectionError, asyncio.TimeoutError):
            return web.Response(status=502, headers=self.headers())

    async def handle(self, request):
        if self.deployment.get('access_mode') == 'tailscale' and not self.private_client(request):
            return web.json_response({'error': 'private_access_required'}, status=403, headers=self.headers())
        path = request.path
        if path == '/api/status':
            return self.status(request)
        if path == '/api/openclaw-bootstrap':
            return await self.bootstrap(request)
        if path.startswith('/camera-video/'):
            if self.deployment.get('camera_video', {}).get('transport') != 'whep':
                return web.Response(status=404, headers=self.headers())
            from whep_proxy import WhepProxy
            return await WhepProxy(self.session, authenticate=self.authenticated_edge,
                                   origin=self.origin).handle(request)
        if path.startswith('/camera-player/') or (path == '/cameras/'
                and request.query.get('transport') != 'jpeg'
                and self.deployment.get('camera_video', {}).get('transport') == 'whep'):
            if self.deployment.get('camera_video', {}).get('transport') != 'whep':
                return web.Response(status=404, headers=self.headers())
            if not await self.authenticated_edge(request):
                return web.Response(status=403, headers=self.headers())
            if request.method not in ('GET', 'HEAD'):
                return web.Response(status=405, headers={**self.headers(), 'Allow': 'GET, HEAD'})
            name = 'index.html' if path == '/cameras/' else path.removeprefix('/camera-player/')
            if name not in ('index.html', 'viewer.mjs', 'viewer.css', 'mediamtx-reader.js'):
                return web.Response(status=404, headers=self.headers())
            return self.static_response(self.here / 'web/camera-player', name)
        if path == '/paai-browser-bootstrap.js':
            return self.static_response(self.here / 'web', 'browser_bootstrap.js')
        if path in ('/paai-light.js', '/paai-light.css'):
            return self.static_response(self.ui, path[1:])
        if path == '/learner-flow.js':
            return self.static_response(self.root / 'demo/kitchen/dashboard', 'learner-flow.js')
        if path == '/' and request.transport.get_extra_info('sockname')[1] == 18791:
            raise web.HTTPFound('/openclaw/')
        if path in ('/', '/guide'):
            raise web.HTTPFound('/guide/')
        if path == '/guide/':
            return self.static_response(self.guide, 'index.html')
        if path.startswith('/guide/'):
            return self.static_response(self.guide, path[len('/guide/'):])
        if path == '/cameras':
            raise web.HTTPFound('/cameras/')
        if path.startswith('/cameras/'):
            return await self.proxy(request, port=8091, path=request.rel_url.path_qs[len('/cameras'):])
        if path == '/openclaw':
            raise web.HTTPFound('/openclaw/?connect=1')
        if path.startswith(('/openclaw/', '/__openclaw/')):
            return await self.proxy(request, port=18790, path=request.rel_url.path_qs)
        return web.Response(status=404, headers=self.headers())


async def run_web(root, deployment, timeout_s=None):
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)
    timeout = ClientTimeout(total=30, connect=5, sock_read=20)
    async with ClientSession(timeout=timeout, trust_env=False, auto_decompress=False) as session:
        runtime = WebRuntime(root, deployment, session)
        @web.middleware
        async def private_errors(request, handler):
            try:
                return await handler(request)
            except web.HTTPException:
                raise
            except Exception:
                # aiohttp's default traceback can include private header/request data.
                return web.json_response({'error': 'service_unavailable'}, status=503, headers=runtime.headers())
        app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[private_errors])
        app.router.add_route('*', '/{path:.*}', runtime.handle)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=8)
        await runner.setup()
        try:
            from runtime import private_bind
            bind = private_bind(deployment.get('web_bind', '127.0.0.1'))
            for port in (8092, 18791):
                for address in sorted({'127.0.0.1', bind}):
                    await web.TCPSite(runner, address, port).start()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stopped.wait(), timeout=timeout_s)
        finally:
            await runner.cleanup()
    return 0
