"""demo/kitchen/physics/perception_probe.py runs on the kitchen's brain detector.

The probe itself needs CUDA, the Isaac bridge and the brain, so it never runs
here. These tests load the module by path and exercise only its detector:
built by `apps.demo._make_detector` from configs/demo.yaml with SAM pinned to
cuda:0, answering the targeted query and the open-world inventory. Fakes only:
a client with `chat.completions.create`, a box segmenter; no CUDA, no network,
no model load.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cascade.config as config
import cascade.perception.segmenter as segmenter_module
from cascade.apps import demo
from cascade.config import load_demo_config
from cascade.perception.brain_detector import INVENTORY_PROMPT, REFER_PROMPT, BrainDetector
from cascade.perception.segmenter import BoxSegmenter
from cascade.types import Frame

PROBE = Path(__file__).resolve().parents[1] / "demo/kitchen/physics/perception_probe.py"
W, H = 320, 180


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """The probe module, under demo.yaml defaults (no perception overrides, a
    private HOME so no real key file is read) and with the SAM warm-up
    forbidden: the probe builds its detector without that thread."""
    for name in ("CASCADE_PERCEPTION_URL", "CASCADE_PERCEPTION_MODEL", "CASCADE_PERCEPTION_KEY_FILE",
                 "CASCADE_BOOTH", "CASCADE_ROBOT", "CASCADE_BASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the module prepends <repo>/src

    def no_warm_up(seg):
        raise AssertionError("the probe must not start the SAM warm-up thread")

    monkeypatch.setattr(demo, "_warm_box_segmenter", no_warm_up)
    spec = importlib.util.spec_from_file_location("kitchen_perception_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_probe_builds_demo_yaml_brain_detector_with_sam_on_cuda0(probe):
    det = probe.make_probe_detector()
    try:
        assert type(det) is BrainDetector
        assert (det._base_url, det._model) == ("http://127.0.0.1:8080/v1", "nemotron-3.5-bolt")
        assert type(det._segmenter) is BoxSegmenter
        model = load_demo_config(camera="isaac").segmenter.get("model")
        assert (det._segmenter._model_path, det._segmenter._device_spec) == (model, "cuda:0")
        assert det._segmenter._model is None  # lazy: nothing loaded, no CUDA touched
        assert det.warm_up_thread is None
    finally:
        det.close()
    # The cuda:0 pin is the probe's own copy of the config, never demo.yaml's.
    assert load_demo_config(camera="isaac").segmenter.get("device") == "auto"


@pytest.mark.parametrize("change,needs", [
    (lambda data: data["segmenter"].update(enabled=False), "segmenter.enabled"),
    (lambda data: data.update(detector={"type": "mock"}), "type: brain"),
])
def test_the_probe_refuses_a_config_without_the_brain_and_sam(probe, monkeypatch, change, needs):
    def edited(**kwargs):
        cfg = load_demo_config(**kwargs)
        change(cfg._data)
        return cfg

    monkeypatch.setattr(config, "load_demo_config", edited)
    with pytest.raises(RuntimeError, match=needs):
        probe.make_probe_detector()


class FakeClient:
    """Canned brain replies: the target for the refer prompt, two objects for the inventory."""

    def __init__(self, target: str):
        self.prompts: list[str] = []
        self.replies = {
            REFER_PROMPT.format(q=target): [{"label": target, "bbox_2d": [100, 200, 300, 600]}],
            INVENTORY_PROMPT: [{"label": target, "bbox_2d": [100, 200, 300, 600]},
                               {"label": "bowl", "bbox_2d": [500, 100, 700, 400]}],
        }
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"][1]["text"]
        self.prompts.append(prompt)
        msg = SimpleNamespace(content=json.dumps(self.replies.get(prompt, [])))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


class FakeSegmenter:
    """BoxSegmenter's constructor and `masks`: one box-shaped mask per box."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path, self.device = model_path, device

    def masks(self, frame, boxes):
        out = []
        for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.float64).reshape(-1, 4).astype(int):
            mask = np.zeros(frame.rgb.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(mask)
        return out


def test_the_targeted_check_and_the_open_world_check_with_fakes(probe, monkeypatch):
    monkeypatch.setattr(segmenter_module, "BoxSegmenter", FakeSegmenter)
    det = probe.make_probe_detector()
    try:
        assert det._segmenter.device == "cuda:0"
        client = det._client = FakeClient(probe.TARGET)
        frame = Frame(rgb=np.full((H, W, 3), 90, np.uint8), depth_m=None, K=np.eye(3))

        targeted = det.detect(frame, classes=[probe.TARGET])
        # What the probe keeps: a learned mask of >= 24 px labelled as the query
        # (localize_object matches the label).
        assert [d.label for d in targeted] == [probe.TARGET]
        assert int(targeted[0].mask.sum()) >= 24

        open_world = det.detect(frame, None)
        assert sorted(d.label for d in open_world) == ["bowl", probe.TARGET]
        assert client.prompts == [REFER_PROMPT.format(q=probe.TARGET), INVENTORY_PROMPT]
    finally:
        det.close()


def test_the_probe_stays_read_only_and_names_no_removed_detector():
    source = PROBE.read_text()
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute)}
    assert not names & {"make_arm", "build_runtime", "_build_arm", "OpenVocabDetector"}
    assert "yoloe" not in source.lower()
