"""WHEP must retain authentication and private session routing across all methods."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

from aiohttp.test_utils import make_mocked_request
from multidict import CIMultiDict
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import whep_proxy as whep
import web_runtime


class Content:
    def __init__(self, data=b''):
        self.data = data

    async def iter_chunked(self, count):
        for offset in range(0, len(self.data), count):
            yield self.data[offset:offset + count]


class Response:
    def __init__(self, status=201, headers=None, body=b'v=0\r\n'):
        self.status = status
        self.headers = CIMultiDict(headers if headers is not None else {
            'Content-Type': 'application/sdp', 'Location': '/kitchen/whep/session-1', 'ETag': 'fixture'})
        self.content = Content(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Session:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or Response()

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


def request(method='POST', path='/camera-video/kitchen/whep', body=b'v=0\r\n', headers=None):
    values = {'Host': '100.64.0.20:8092', 'Origin': 'http://100.64.0.20:8092',
              'Sec-Fetch-Site': 'same-origin'}
    if method == 'POST':
        values['Content-Type'] = 'application/sdp'
    if method == 'PATCH':
        values.update({'Content-Type': 'application/trickle-ice-sdpfrag', 'If-Match': '*'})
    values.update(headers or {})
    return make_mocked_request(method, path, headers=values, payload=Content(body))


def proxy(session, allowed=True):
    async def authenticate(req):
        return allowed
    return whep.WhepProxy(session, authenticate=authenticate,
                          origin=lambda req: 'http://100.64.0.20:8092')


@pytest.mark.parametrize('method,path', [('OPTIONS', '/camera-video/kitchen/whep'),
    ('POST', '/camera-video/kitchen/whep'), ('PATCH', '/camera-video/kitchen/whep/session-1'),
    ('DELETE', '/camera-video/kitchen/whep/session-1')])
def test_every_session_method_requires_authentication_before_relay(method, path):
    session = Session()
    result = asyncio.run(proxy(session, False).handle(request(method, path)))
    assert result.status == 403 and not session.calls


@pytest.mark.parametrize('headers', [{'Origin': 'https://foreign.invalid'},
                                    {'Sec-Fetch-Site': 'cross-site'}])
def test_foreign_browser_cannot_open_camera_session(headers):
    session = Session()
    assert asyncio.run(proxy(session).handle(request(headers=headers))).status == 403
    assert not session.calls


def test_created_session_stays_same_origin_and_never_forwards_browser_credentials():
    session = Session()
    result = asyncio.run(proxy(session).handle(request(headers={
        'Authorization': 'Bearer private-fixture', 'Cookie': 'private-fixture'})))
    assert result.status == 201 and result.body == b'v=0\r\n'
    assert result.headers['Location'] == '/camera-video/kitchen/whep/session-1'
    assert result.headers['ETag'] == 'fixture'
    method, url, args = session.calls[0]
    assert (method, url) == ('POST', 'http://127.0.0.1:8889/kitchen/whep')
    assert args['data'] == b'v=0\r\n' and args['allow_redirects'] is False
    assert 'private-fixture' not in repr(args['headers'])
    assert args['timeout'].total == 15


@pytest.mark.parametrize('method', ['PATCH', 'DELETE', 'OPTIONS'])
def test_session_followups_keep_prefix_mapping_and_patch_condition(method):
    session = Session(Response(204, {}))
    body = b'a=candidate:fixture' if method == 'PATCH' else b''
    result = asyncio.run(proxy(session).handle(request(method,
        '/camera-video/kitchen/whep/session-1', body)))
    assert result.status == 204
    assert session.calls[0][1] == 'http://127.0.0.1:8889/kitchen/whep/session-1'
    if method == 'PATCH':
        assert session.calls[0][2]['headers']['If-Match'] == '*'


@pytest.mark.parametrize('location', ['https://foreign.invalid/kitchen/whep/session-1',
    '//foreign.invalid/kitchen/whep/session-1', '/side/whep/session-1', '/kitchen/whep',
    '/kitchen/whep/../session-1', '/kitchen/whep/session-1?token=x', '/kitchen/whep/session-1#x',
    '/kitchen/whep/%2e%2e', '/kitchen/whep/session-1\r\nInjected: yes'])
def test_relay_location_cannot_escape_camera_or_origin(location):
    session = Session(Response(headers={'Location': location}))
    result = asyncio.run(proxy(session).handle(request()))
    assert result.status == 502 and 'Location' not in result.headers


def test_absolute_loopback_location_is_rewritten():
    assert whep.session_location('http://127.0.0.1:8889/side/whep/id-1', 'side') == '/camera-video/side/whep/id-1'


@pytest.mark.parametrize('path', ['/camera-video/unknown/whep', '/camera-video/kitchen/other',
    '/camera-video/kitchen/whep?url=http://private.invalid', '/camera-video/kitchen/whep/%41'])
def test_only_canonical_named_camera_paths_reach_relay(path):
    session = Session()
    assert asyncio.run(proxy(session).handle(request(path=path))).status == 404
    assert not session.calls


def test_oversized_streamed_offer_is_refused_before_relay():
    session = Session()
    assert asyncio.run(proxy(session).handle(request(body=b'x' * (whep.MAX_BODY + 1)))).status == 413
    assert not session.calls


def test_oversized_relay_answer_and_missing_location_fail_closed():
    for response in (Response(body=b'x' * (whep.MAX_BODY + 1)), Response(headers={})):
        assert asyncio.run(proxy(Session(response)).handle(request())).status == 502


def test_web_runtime_route_requires_activation_and_actual_tailnet_socket(tmp_path):
    session = Session()
    deployment = {'access_mode': 'tailscale', 'web_bind': '100.64.0.20',
                  'origins': ['http://100.64.0.20:8092']}
    runtime = web_runtime.WebRuntime(tmp_path, deployment, session)
    req = request()
    req.transport.get_extra_info.side_effect = lambda name: {
        'peername': ('100.64.0.10', 32123), 'sockname': ('100.64.0.20', 8092)}[name]
    assert asyncio.run(runtime.handle(req)).status == 404
    assert not session.calls
    deployment['camera_video'] = {'transport': 'whep'}
    assert asyncio.run(runtime.handle(req)).status == 201
    req = request()
    req.transport.get_extra_info.side_effect = lambda name: {
        'peername': ('203.0.113.1', 32123), 'sockname': ('100.64.0.20', 8092)}[name]
    assert asyncio.run(runtime.handle(req)).status == 403
    assert len(session.calls) == 1
