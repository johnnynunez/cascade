"""The private launcher serves the same guide as the public visitor."""
import asyncio
from pathlib import Path
import sys

from aiohttp.test_utils import make_mocked_request
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from web_runtime import WebRuntime


def serve(path, *, method='GET', peer='100.64.0.10'):
    app = WebRuntime(ROOT, {'access_mode': 'tailscale', 'web_bind': '100.64.0.20'}, None)
    request = make_mocked_request(method, path, headers={'Host': '100.64.0.20:8092'})
    request.transport.get_extra_info.side_effect = lambda field: {
        'peername': (peer, 32000), 'sockname': ('100.64.0.20', 8092)}[field]
    return asyncio.run(app.handle(request))


@pytest.mark.parametrize('path,name,mime', [
    ('/staff/', 'staff.html', 'text/html'),
    ('/staff/style.css', 'staff.css', 'text/css'),
    ('/staff/media/openclaw-cameras.png', 'staff-openclaw-cameras.png', 'image/png'),
])
def test_canonical_guide_assets_require_private_access(path, name, mime):
    response = serve(path)
    assert response.status == 200 and response.content_type == mime
    assert response.body == (ROOT / 'deploy/brev' / name).read_bytes()
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert serve(path, peer='203.0.113.10').status == 403
    assert serve(path, method='POST').status == 405


@pytest.mark.parametrize('path', [
    '/staff/visitor.py', '/staff/media/auth.json', '/staff/media/unknown.png',
    '/staff/media/../visitor.py', '/staff/media/%2e%2e/visitor.py',
])
def test_staff_asset_routes_do_not_expose_other_files(path):
    assert serve(path).status == 404
