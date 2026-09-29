"""Only the local Spark demo page may frame the Chrome or Firefox camera extension."""
import base64

import pytest

from test_spark_ui_hints import get, serving
from test_spark_visitor_chat import chat as _chat

chat = _chat
BASE = ("default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; style-src 'self'; "
        "script-src 'self'; frame-ancestors 'none'")


def test_local_demo_page_allows_both_extension_frame_schemes(chat):
    with serving(chat, port=8092) as port:
        for path in ("/", "/api/chat", "/visitor.js"):
            code, headers, _ = get(port, path, Host="127.0.0.1:8092")
            assert code == 200
            assert headers["Content-Security-Policy"] == BASE + "; frame-src chrome-extension: moz-extension:"


@pytest.mark.parametrize("port,with_chat,authorization", [(8093, True, "Basic fixture"), (8092, False, ""),
                                                          (8093, False, "")])
def test_other_surfaces_never_allow_extension_frames(chat, port, with_chat, authorization):
    headers = {"Host": f"127.0.0.1:{port}"}
    if authorization:
        headers["Authorization"] = authorization
    with serving(chat if with_chat else None, port=port, authorization=authorization) as actual:
        code, response, _ = get(actual, "/", **headers)
        assert code == 200
        assert response["Content-Security-Policy"] == BASE
        assert "extension" not in response["Content-Security-Policy"]
