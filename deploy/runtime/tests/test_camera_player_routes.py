"""The optional browser viewer uses the existing visitor access boundary."""
import asyncio
from pathlib import Path
import sys
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import make_mocked_request
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from web_runtime import WebRuntime

ASSETS = ('index.html', 'viewer.mjs', 'viewer.css', 'mediamtx-reader.js')


def runtime(*, enabled=True, mode='tailscale'):
    app = WebRuntime(HERE.parents[1], {
        'access_mode': mode, 'web_bind': '100.64.0.20',
        'origins': ['http://100.64.0.20:8092'],
        'camera_video': {'transport': 'whep'} if enabled else {},
    }, None)
    app.proxy = AsyncMock(return_value=web.Response(text='JPEG fixture'))
    return app


def request(path='/cameras/', method='GET', peer='100.64.0.10', local='100.64.0.20'):
    result = make_mocked_request(method, path, headers={'Host': '100.64.0.20:8092'})
    result.transport.get_extra_info.side_effect = lambda name: {
        'peername': (peer, 32123), 'sockname': (local, 8092)}[name]
    return result


@pytest.mark.parametrize('name', ASSETS)
def test_inactive_profile_does_not_serve_video_assets(name):
    app = runtime(enabled=False)
    result = asyncio.run(app.handle(request('/camera-player/' + name)))
    assert result.status == 404
    app.proxy.assert_not_called()


@pytest.mark.parametrize('method', ['GET', 'HEAD'])
def test_enabled_private_viewer_serves_local_english_ui_and_real_openclaw(method):
    app = runtime()
    result = asyncio.run(app.handle(request(method=method)))
    assert result.status == 200 and result.content_type == 'text/html'
    assert result.headers['Cache-Control'] == 'no-store'
    assert result.headers['Permissions-Policy'] == 'camera=(), microphone=(), geolocation=()'
    assert b'<html lang="en">' in result.body
    assert b'href="/openclaw/?connect=1"' in result.body
    assert b'href="/cameras/?transport=jpeg"' in result.body
    for name in ('kitchen', 'worktop', 'side'):
        assert f'data-camera="{name}"'.encode() in result.body
    app.proxy.assert_not_called()


@pytest.mark.parametrize('name,mime', [('viewer.css', 'text/css'),
    ('viewer.mjs', 'text/javascript'), ('mediamtx-reader.js', 'text/javascript')])
def test_known_assets_are_local_and_have_browser_mime_types(name, mime):
    app = runtime()
    result = asyncio.run(app.handle(request('/camera-player/' + name)))
    assert result.status == 200 and result.content_type == mime
    assert result.headers['X-Content-Type-Options'] == 'nosniff'
    assert result.body == (HERE / 'web/camera-player' / name).read_bytes()


@pytest.mark.parametrize('path', ['/cameras/', '/camera-player/index.html',
    '/camera-player/viewer.mjs', '/camera-player/viewer.css', '/camera-player/mediamtx-reader.js'])
def test_viewer_and_every_asset_require_confirmed_edge_authentication(path):
    app = runtime(mode='sso')
    app.authenticated_edge = AsyncMock(return_value=False)
    result = asyncio.run(app.handle(request(path)))
    assert result.status == 403
    app.authenticated_edge.assert_awaited_once()
    app.proxy.assert_not_called()


@pytest.mark.parametrize('peer,local', [('203.0.113.10', '100.64.0.20'),
    ('127.0.0.1', '100.64.0.20'), ('100.64.0.10', '127.0.0.1'),
    ('100.64.0.10', '100.64.0.21')])
def test_loopback_and_forged_private_peers_cannot_fetch_viewer(peer, local):
    app = runtime()
    result = asyncio.run(app.handle(request(peer=peer, local=local)))
    assert result.status == 403
    app.proxy.assert_not_called()


@pytest.mark.parametrize('path', ['/camera-player/README.md', '/camera-player/../../runtime.py',
    '/camera-player/nested/viewer.css'])
def test_static_asset_allowlist_cannot_serve_arbitrary_files(path):
    assert asyncio.run(runtime().handle(request(path))).status == 404


@pytest.mark.parametrize('method', ['POST', 'PATCH', 'DELETE', 'OPTIONS'])
def test_viewer_and_assets_are_read_only(method):
    app = runtime()
    for path in ('/cameras/', '/camera-player/viewer.mjs'):
        result = asyncio.run(app.handle(request(path, method)))
        assert result.status == 405 and result.headers['Allow'] == 'GET, HEAD'


@pytest.mark.parametrize('enabled,path,upstream', [(False, '/cameras/', '/'),
    (True, '/cameras/?transport=jpeg', '/?transport=jpeg'),
    (True, '/cameras/stream/worktop', '/stream/worktop'),
    (True, '/cameras/state', '/state')])
def test_original_jpeg_page_stream_and_state_remain_available(enabled, path, upstream):
    app = runtime(enabled=enabled)
    req = request(path)
    result = asyncio.run(app.handle(req))
    assert result.text == 'JPEG fixture'
    app.proxy.assert_awaited_once_with(req, port=8091, path=upstream)
