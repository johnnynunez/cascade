"""Read-only camera controls stay within the supported visitor routes."""
from html.parser import HTMLParser
import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cascade.apps.stream_server import _make_handler


def camera_page(read_only):
    server = SimpleNamespace(state=lambda: {"read_only": read_only},
                             _rig=SimpleNamespace(names=["worktop"]))
    handler = object.__new__(_make_handler(server))
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler._index()
    return handler.wfile.getvalue().decode()


class CameraButtons(HTMLParser):
    def __init__(self):
        super().__init__()
        self.visible = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        action = attrs.get("onclick", "")
        if tag == "button" and action.startswith("setView(") and "hidden" not in attrs:
            self.visible.append(action.split("'")[3])


@pytest.mark.parametrize("read_only,expected", [(True, ["rgb"]), (False, ["rgb", "depth", "agent"])])
def test_camera_controls_match_the_page_scope(read_only, expected):
    buttons = CameraButtons()
    buttons.feed(camera_page(read_only))
    assert buttons.visible == expected


def test_connected_images_report_connection_without_claiming_frame_freshness():
    page = camera_page(True)
    source = page[page.index("function statusSummary("):page.index("async function tick(")]
    script = """
        const fs = require('node:fs');
        const vm = require('node:vm');
        const context = {document: {getElementById: () => ({naturalWidth: 640,
            closest: () => ({dataset: {streamState: 'connected'}})})}};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync(0, 'utf8'), context);
        console.log(JSON.stringify([
            context.statusSummary({read_only:true,cameras:{worktop:{online:true}}}),
            context.statusSummary({read_only:true,cameras:{worktop:{online:false}}})]));
    """
    result = subprocess.run(["node", "-e", script], input=source, text=True,
                            capture_output=True, timeout=10, check=True)
    connected, offline = json.loads(result.stdout)
    assert connected.startswith("1 camera feeds connected")
    assert "live" not in connected.lower()
    assert offline.startswith("0 of 1 views connected")
