"""Synthetic camera checks for the annotated image delivered to MCP clients."""

import base64
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.agent.trace import TraceLogger
from cascade.apps.mcp_server import McpSkillServer
from cascade.config import Cfg
from cascade.memory import BeliefStore
from cascade.skills.runtime import SkillRuntime, _jpeg
from cascade.types import Frame, SkillError


def annotated_runtime(tmp_path):
    image = np.full((240, 320, 3), 40, dtype=np.uint8)
    image[100:140, 145:185] = [20, 180, 20]
    frame = Frame(
        image, None, np.array([[240., 0, 160], [0, 240., 120], [0, 0, 1]]),
        frame_id=42,
        T_base_cam=np.array([[1., 0, 0, 0], [0, -1., 0, 0],
                             [0, 0, -1., 1.], [0, 0, 0, 1.]]),
    )
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = Cfg({"grasp": {}, "safety": {"table_z": 0.,
        "workspace": {"min": [-.4, -.3, 0], "max": [.4, .3, 1]}}})
    rt.extrinsics, rt.camera = None, SimpleNamespace()
    rt.beliefs, rt.trace = BeliefStore(), TraceLogger(tmp_path / "trace")
    rt.beliefs.update("green cube", [.02, 0, .05], .9, t=time.monotonic())
    rt._tcp = lambda: None
    rt.last_frame = rt.last_annotated_frame = None
    captures = []

    def capture():
        captures.append(frame)
        rt.last_frame = frame
        return frame

    rt.observe_fresh = capture
    rt.observe = lambda: pytest.fail("Annotated image requested an unvalidated cached frame")
    rt.execute = lambda name, args: {**getattr(rt, "skill_" + name)(**args), "ok": True}
    server = McpSkillServer()
    server._runtime = rt
    return server, rt, frame, captures


def text_payload(response):
    return json.loads(next(block["text"] for block in response["content"] if block["type"] == "text"))


def test_annotated_image_matches_its_key_and_capture_without_replacing_raw_pixels(tmp_path):
    server, rt, frame, captures = annotated_runtime(tmp_path)
    raw = frame.rgb.copy()
    result = server.call_tool("annotated_view", {})

    assert not result["isError"] and captures == [frame]
    images = [block for block in result["content"] if block["type"] == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0]["data"]) == _jpeg(rt.last_annotated_frame.rgb)
    assert base64.b64decode(images[0]["data"]) != _jpeg(raw)
    assert np.array_equal(frame.rgb, raw)
    assert rt.last_frame is frame
    data = text_payload(result)
    assert data["objects"][0]["index"] == 1
    assert data["objects"][0]["label"] == "green cube"
    assert "1. green cube" in data["key"]
    assert data["observation_frame"]["frame_id"] == 42
    assert data["observation_frame"]["age_clock"] == "client_monotonic"
    assert "image" not in data
    assert "does not mean the scene is empty" in data["note"]
    assert "does not verify reachability" in data["observation_note"]
    assert list((rt.trace.run_dir / "keyframes").glob("*_annotated_view.jpg"))


def test_delivery_uses_the_rendered_capture_even_if_last_raw_frame_changes(tmp_path):
    server, rt, frame, captures = annotated_runtime(tmp_path)
    saved = rt.skill_annotated_view()
    rendered = _jpeg(rt.last_annotated_frame.rgb)
    rt.last_frame = Frame(np.zeros_like(frame.rgb), None, frame.K, frame_id=43)
    rt.execute = lambda *_: {**saved, "ok": True}

    result = server.call_tool("annotated_view", {})

    assert not result["isError"] and captures == [frame]
    assert base64.b64decode(result["content"][0]["data"]) == rendered
    assert text_payload(result)["observation_frame"]["frame_id"] == 42
    assert "image" in saved  # the runtime and trace keep the saved path


@pytest.mark.parametrize("failure", ["fresh_capture", "render", "aged_image"])
def test_failed_annotation_never_delivers_an_earlier_successful_image(tmp_path, monkeypatch, failure):
    server, rt, frame, _ = annotated_runtime(tmp_path)
    assert not server.call_tool("annotated_view", {})["isError"]
    if failure == "fresh_capture":
        def unavailable():
            raise SkillError("No fresh frame")
        rt.observe_fresh = unavailable
    elif failure == "render":
        monkeypatch.setattr("cascade.perception.visual_interface.annotate_frame", lambda _: (None, []))
    else:
        frame.t = time.monotonic() - 6

    def execute(name, args):
        try:
            return {**getattr(rt, "skill_" + name)(**args), "ok": True}
        except SkillError as exc:
            return {"ok": False, "error": str(exc)}
    rt.execute = execute

    result = server.call_tool("annotated_view", {})

    assert result["isError"] and not text_payload(result)["ok"]
    assert all(block["type"] != "image" for block in result["content"])
    if failure != "aged_image":
        assert rt.last_annotated_frame is None
