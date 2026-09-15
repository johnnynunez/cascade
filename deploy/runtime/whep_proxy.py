"""Authenticated, bounded WHEP session proxy for the local camera relay."""
from __future__ import annotations

import asyncio
import re
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientTimeout, web

PREFIX = '/camera-video'
RELAY = 'http://127.0.0.1:8889'
PATH = re.compile(r'/(kitchen|worktop|side)/whep(?:/([A-Za-z0-9_-]{1,128}))?')
MAX_BODY = 256 * 1024
HEADERS = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
           'Referrer-Policy': 'no-referrer'}


def session_location(value, camera):
    """Keep relay session operations on the same private visitor origin."""
    if not isinstance(value, str) or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError('Invalid camera session location')
    parsed = urlsplit(value)
    if (parsed.scheme or parsed.netloc) and (parsed.scheme, parsed.netloc) != ('http', '127.0.0.1:8889'):
        raise ValueError('Camera session left the local relay')
    match = PATH.fullmatch(parsed.path)
    if (not match or match[1] != camera or not match[2]
            or parsed.query or parsed.fragment or '?' in value or '#' in value):
        raise ValueError('Invalid camera session path')
    return PREFIX + parsed.path


async def bounded_body(content):
    result = bytearray()
    async for chunk in content.iter_chunked(16384):
        result.extend(chunk)
        if len(result) > MAX_BODY:
            raise web.HTTPRequestEntityTooLarge(max_size=MAX_BODY, actual_size=len(result))
    return bytes(result)


class WhepProxy:
    def __init__(self, session, *, authenticate, origin):
        self.session, self.authenticate, self.origin = session, authenticate, origin

    async def handle(self, request):
        # Authorize every session operation, including trickle ICE and deletion.
        origin = self.origin(request)
        if (not origin or request.headers.getall('Origin', []) not in ([], [origin])
                or request.headers.get('Sec-Fetch-Site', 'same-origin') not in ('same-origin', 'none')
                or not await self.authenticate(request)):
            return web.Response(status=403, headers=HEADERS)
        path = request.path.removeprefix(PREFIX)
        match = PATH.fullmatch(path)
        if (not request.path.startswith(PREFIX + '/') or not match
                or request.query_string or '%' in request.raw_path):
            return web.Response(status=404, headers=HEADERS)
        methods = ('OPTIONS', 'PATCH', 'DELETE') if match[2] else ('OPTIONS', 'POST')
        if request.method not in methods:
            return web.Response(status=405, headers={**HEADERS, 'Allow': ', '.join(methods)})
        expected_type = {'POST': 'application/sdp', 'PATCH': 'application/trickle-ice-sdpfrag'}.get(request.method)
        if expected_type and request.content_type != expected_type:
            return web.Response(status=415, headers=HEADERS)
        if request.method == 'PATCH' and request.headers.getall('If-Match', []) != ['*']:
            return web.Response(status=428, headers=HEADERS)
        if request.content_length is not None and request.content_length > MAX_BODY:
            return web.Response(status=413, headers=HEADERS)
        try:
            body = await asyncio.wait_for(bounded_body(request.content), timeout=5)
        except web.HTTPRequestEntityTooLarge:
            return web.Response(status=413, headers=HEADERS)
        except asyncio.TimeoutError:
            return web.Response(status=408, headers=HEADERS)
        if request.method in ('OPTIONS', 'DELETE') and body:
            return web.Response(status=400, headers=HEADERS)
        try:
            headers = {key: value for key, value in request.headers.items()
                       if key.lower() in ('accept', 'content-type', 'if-match')}
            # Relay access uses the trusted loopback connection; no browser token is forwarded.
            headers.update({'Origin': origin, 'Accept-Encoding': 'identity'})
            async with self.session.request(request.method, RELAY + path, headers=headers,
                    data=body, allow_redirects=False,
                    timeout=ClientTimeout(total=15, connect=3, sock_read=10)) as response:
                if not 200 <= response.status < 300:
                    return web.Response(status=response.status if 400 <= response.status <= 599 else 502,
                                        headers=HEADERS)
                result_headers = {key: value for key, value in response.headers.items()
                                  if key.lower() in ('content-type', 'etag', 'accept-patch', 'allow')}
                locations = response.headers.getall('Location', [])
                if response.status == 201 and len(locations) != 1:
                    raise ValueError('Created camera session has no unique location')
                if locations:
                    if len(locations) != 1:
                        raise ValueError('Ambiguous camera session location')
                    result_headers['Location'] = session_location(locations[0], match[1])
                payload = await asyncio.wait_for(bounded_body(response.content), timeout=10)
                return web.Response(status=response.status, body=payload,
                                    headers={**result_headers, **HEADERS})
        except (ClientError, ConnectionError, asyncio.TimeoutError, ValueError, web.HTTPRequestEntityTooLarge):
            return web.Response(status=502, headers=HEADERS)
