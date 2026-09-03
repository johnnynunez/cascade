"""VLM-only perception (perception/vlm_detector.py): the "quitar YOLOE,
full cosmos3-edge" path.

The parser is pure text -> Detection and is tested exhaustively; the client
construction is tested against a stub `openai` module injected into
sys.modules (CI installs no `openai` -- same convention as
test_hermes_brain.py, per the conftest note "do not gate a test on an
optional extra when the contract can be faked").
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cascade.perception.vlm_detector import parse_detections, _extract_json_array
from cascade.types import Frame


W, H = 640, 480


def test_parse_plain_array():
    text = (
        '[{"label": "red cube", "bbox_2d": [100, 100, 300, 300], "confidence": 0.9},'
        ' {"label": "banana", "bbox_2d": [500, 400, 700, 600], "confidence": 0.7}]'
    )
    dets = parse_detections(text, W, H)
    assert [d.label for d in dets] == ["red cube", "banana"]
    # 0-1000 normalized -> pixels
    np.testing.assert_allclose(dets[0].bbox, [64, 48, 192, 144], atol=0.5)
    assert dets[0].conf == pytest.approx(0.9)
    # a VLM returns boxes, not masks -- grounding must fall back to bbox mask
    assert dets[0].mask is None


def test_parse_tolerates_fences_prose_and_thinking():
    text = (
        "<think>I see two objects, let me quote the format: "
        '[{"label": "example", "bbox_2d": [0, 0, 10, 10]}]</think>\n'
        "Here is the result:\n```json\n"
        '[{"label": "cup", "bbox_2d": [200, 200, 400, 500], "confidence": 0.8}]'
        "\n```"
    )
    dets = parse_detections(text, W, H)
    assert [d.label for d in dets] == ["cup"]


def test_parse_takes_last_array_when_reasoning_quotes_one():
    # No <think> tags, but the reply quotes an example array before the answer.
    text = (
        'Example format: [{"label": "x", "bbox_2d": [0, 0, 100, 100]}]\n'
        'Answer: [{"label": "bottle", "bbox_2d": [100, 100, 200, 200], "confidence": 1.0}]'
    )
    dets = parse_detections(text, W, H)
    assert [d.label for d in dets] == ["bottle"]


def test_parse_empty_scene_and_garbage():
    assert parse_detections("[]", W, H) == []
    assert parse_detections("no objects visible", W, H) == []
    assert parse_detections('{"label": "not-an-array"}', W, H) == []
    assert parse_detections("", W, H) == []


def test_parse_rejects_malformed_entries():
    text = (
        "["
        '{"label": "", "bbox_2d": [0, 0, 500, 500]},'          # empty label
        '{"label": "a", "bbox_2d": [100, 100]},'               # short bbox
        '{"label": "b", "bbox_2d": ["x", 0, 500, 500]},'       # non-numeric
        '{"label": "tiny", "bbox_2d": [500, 500, 501, 501]},'  # degenerate box
        '{"label": "ok", "bbox_2d": [0, 0, 500, 500], "confidence": 0.5}'
        "]"
    )
    dets = parse_detections(text, W, H)
    assert [d.label for d in dets] == ["ok"]


def test_parse_clips_and_orders_coordinates():
    text = '[{"label": "spill", "bbox_2d": [900, 800, 1200, -50], "confidence": 0.6}]'
    (d,) = parse_detections(text, W, H)
    x0, y0, x1, y1 = d.bbox
    assert 0 <= x0 <= x1 <= W
    assert 0 <= y0 <= y1 <= H


def test_parse_applies_min_conf():
    text = (
        '[{"label": "faint", "bbox_2d": [0, 0, 500, 500], "confidence": 0.1},'
        ' {"label": "solid", "bbox_2d": [0, 0, 500, 500], "confidence": 0.9}]'
    )
    dets = parse_detections(text, W, H, min_conf=0.25)
    assert [d.label for d in dets] == ["solid"]


def test_extract_json_array_survives_unbalanced_bracket():
    # A truncated reply must not hang or crash the walker.
    assert _extract_json_array("[ this never closes") is None
    assert _extract_json_array("junk ][ more junk") is None


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.requests.append(kwargs)
        if self._owner.fail:
            raise RuntimeError("server down")
        msg = types.SimpleNamespace(content=self._owner.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeOpenAI:
    def __init__(self, base_url=None, api_key=None, timeout=None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.requests: list[dict] = []
        self.fail = False
        self.reply = "[]"
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(self))
        _FakeOpenAI.last = self


@pytest.fixture
def stub_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


def _frame(w=W, h=H):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[100:200, 100:200] = (0, 0, 255)  # red square (BGR)
    K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=None, K=K)


def test_vlm_detector_detects_via_stub(stub_openai):
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="cosmos3-edge")
    client = _FakeOpenAI.last
    client.reply = '[{"label": "red cube", "bbox_2d": [156, 208, 312, 417], "confidence": 0.9}]'
    dets = det.detect(_frame())
    assert [d.label for d in dets] == ["red cube"]
    req = client.requests[-1]
    assert req["model"] == "cosmos3-edge"
    assert req["temperature"] == 0.0
    # image + text parts present
    parts = req["messages"][0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_detector_class_restriction_is_prompt_text(stub_openai):
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="m")
    client = _FakeOpenAI.last
    det.set_classes(["banana", "cup"])
    det.detect(_frame())
    text = client.requests[-1]["messages"][0]["content"][1]["text"]
    assert "banana, cup" in text
    # None restores open-world: the restriction line disappears
    det.set_classes(None)
    det.detect(_frame())
    text = client.requests[-1]["messages"][0]["content"][1]["text"]
    assert "banana" not in text


def test_vlm_detector_booth_rule_on_server_failure(stub_openai):
    """A down brain server yields [] (perception heartbeat survives), never
    an exception into the watcher thread."""
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="m")
    _FakeOpenAI.last.fail = True
    assert det.detect(_frame()) == []


def test_demo_wiring_builds_vlm_detector(stub_openai, demo_cfg):
    """`detector.type: vlm` in the config reaches VLMDetector with the
    profile's base_url/model -- the switch is config-only, no code."""
    from cascade.apps.demo import _make_detector
    from cascade.perception.vlm_detector import VLMDetector

    demo_cfg._data["detector"] = {
        "type": "vlm", "base_url": "http://127.0.0.1:8082/v1",
        "model": "cosmos3-edge", "conf": 0.3,
    }
    # the mock camera pins detector.type: mock; clear it so the global wins
    demo_cfg._data["camera"].pop("detector", None)
    det = _make_detector(demo_cfg)
    assert isinstance(det, VLMDetector)
    assert _FakeOpenAI.last.base_url == "http://127.0.0.1:8082/v1"
