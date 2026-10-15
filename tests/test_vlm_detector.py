"""VLMDetector (perception/vlm_detector.py): the deprecated cosmos3-edge alias.

It is now a thin BrainDetector subclass with the old defaults (cosmos3-edge on
:8082, no segmenter) and the old constructor kwargs; the reply parser lives in
brain_detector.parse_boxes and is tested in test_brain_detector_parse.py.

The client construction is tested against a stub `openai` module injected into
sys.modules (CI installs no `openai` -- same convention as
test_hermes_brain.py, per the conftest note "do not gate a test on an
optional extra when the contract can be faked").
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from cascade.perception.brain_detector import (
    INVENTORY_PROMPT,
    REFER_PROMPT,
    BrainDetector,
)
from cascade.types import Frame

W, H = 640, 480


class _FakeCompletions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.requests.append(kwargs)
        if _FakeOpenAI.fail:
            raise RuntimeError("server down")
        msg = types.SimpleNamespace(content=_FakeOpenAI.reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeOpenAI:
    fail = False
    reply = "[]"
    last = None

    def __init__(self, base_url=None, api_key=None, timeout=None, max_retries=None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(self))
        _FakeOpenAI.last = self


@pytest.fixture
def stub_openai(monkeypatch):
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    monkeypatch.setattr(_FakeOpenAI, "fail", False)
    monkeypatch.setattr(_FakeOpenAI, "reply", "[]")
    monkeypatch.setattr(_FakeOpenAI, "last", None)
    return mod


def _frame(w=W, h=H):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[100:200, 100:200] = (0, 0, 255)  # red square (BGR)
    K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=None, K=K)


def test_vlm_detector_is_a_brain_detector_with_the_old_defaults(stub_openai):
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector()
    _FakeOpenAI.reply = '[{"label": "red cube", "bbox_2d": [156, 208, 312, 417]}]'
    (d,) = det.detect(_frame())
    det.close()
    assert isinstance(det, BrainDetector)
    client = _FakeOpenAI.last
    assert client.base_url == "http://127.0.0.1:8082/v1"
    assert client.api_key == "EMPTY"
    assert client.max_retries == 0  # the next detect is the retry
    req = client.requests[-1]
    assert req["model"] == "cosmos3-edge"
    assert req["temperature"] == 0
    parts = req["messages"][0]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[1]["text"] == INVENTORY_PROMPT
    assert d.label == "red cube"
    np.testing.assert_allclose(d.bbox, [99.84, 99.84, 199.68, 200.16], rtol=1e-5)
    # no segmenter by default: grounding falls back to the bbox mask
    assert d.mask is None


def test_vlm_detector_maps_the_old_constructor_kwargs(stub_openai):
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="m", api_key="abc", timeout_s=3.0, conf=0.3)
    det.detect(_frame())
    det.close()
    client = _FakeOpenAI.last
    assert (client.base_url, client.api_key, client.timeout, client.max_retries) == ("http://x/v1", "abc", 3.0, 0)
    assert client.requests[-1]["model"] == "m"
    assert "abc" not in repr(det)


def test_vlm_detector_classes_are_refer_prompts(stub_openai):
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="m")
    det.detect(_frame(), ["banana", "cup"])
    texts = [r["messages"][0]["content"][1]["text"] for r in _FakeOpenAI.last.requests]
    assert texts == [REFER_PROMPT.format(q="banana"), REFER_PROMPT.format(q="cup")]
    # None restores open-world: the inventory prompt
    det.detect(_frame(), None)
    det.close()
    assert _FakeOpenAI.last.requests[-1]["messages"][0]["content"][1]["text"] == INVENTORY_PROMPT


def test_vlm_detector_booth_rule_on_server_failure(stub_openai):
    """A down brain server yields [] (perception heartbeat survives), never
    an exception into the watcher thread."""
    from cascade.perception.vlm_detector import VLMDetector

    det = VLMDetector(base_url="http://x/v1", model="m")
    _FakeOpenAI.fail = True
    assert det.detect(_frame()) == []
    det.close()


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
    det.detect(_frame())
    det.close()
    assert _FakeOpenAI.last.base_url == "http://127.0.0.1:8082/v1"
