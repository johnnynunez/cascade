"""Guarded replies for a brain whose box list can loop (perception/brain_detector.py, GUARDED_REPLY_MODELS).

Nemotron 3.5 Bolt, greedy, sometimes repeats one label with its box sliding a few units per entry until the token
cap (6 of the 47 bake-off frames), and lists categories the inventory prompt excludes. Its requests ask for
BOX_LIST_SCHEMA within GUARDED_MAX_TOKENS, a cut reply keeps its complete boxes, a reply that hit a bound loses the
loop, and inventory entries naming an excluded category are dropped. Every other brain (Qwen3.8) keeps the request
and parse it had. The loop replies here are synthetic, shaped like Bolt's (one entry per line, the same label, the
box moving by a constant step).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cascade.apps import demo
from cascade.perception.brain_detector import (
    GUARDED_MAX_BOXES,
    GUARDED_MAX_TOKENS,
    BOX_LIST_SCHEMA,
    INVENTORY_PROMPT,
    BrainDetector,
    parse_guarded_boxes,
    parse_boxes,
)
from cascade.types import Frame
from test_brain_detector import FakeClient, FakeSegmenter

W, H = 1000, 1000  # pixel box == 0..1000 box


def entry(label, box) -> str:
    return '{"label": "%s", "bbox_2d": [%d, %d, %d, %d]}' % (label, *box)


def reply(entries, *, closed=True) -> str:
    text = "[\n  " + ",\n  ".join(entries)
    return text + "\n]" if closed else text


REAL = [("red plush lobster", (573, 552, 738, 708)), ("black pouch", (390, 462, 560, 620)),
        ("white cup", (405, 320, 435, 380))]
LOOP = [("white cable", (380 + 20 * k, 620, 430 + 20 * k, 650)) for k in range(30)]


def labels(found):
    return [label for label, _ in found]


def test_a_cut_loop_keeps_the_objects_before_it_and_the_loops_first_box():
    text = reply([entry(*e) for e in REAL + LOOP], closed=False) + ',\n  {"label": "white cab'
    assert parse_boxes(text, W, H) == []  # what the detector rejects today
    found = parse_guarded_boxes(text, W, H, cut=True)
    assert labels(found) == ["red plush lobster", "black pouch", "white cup", "white cable"]
    np.testing.assert_allclose(found[3][1], [380, 620, 430, 650])


def test_a_loop_ended_by_the_schemas_bound_loses_its_tail():
    items = REAL + LOOP[:GUARDED_MAX_BOXES - len(REAL)]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["red plush lobster", "black pouch", "white cup", "white cable"]


def test_every_looping_label_in_a_reply_at_the_bound_keeps_its_first_box():
    run = [("black cable", (100, 100 + 10 * k, 140, 140 + 10 * k)) for k in range(6)]
    items = REAL[:1] + run + REAL[1:] + LOOP[:GUARDED_MAX_BOXES - len(REAL) - len(run)]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["red plush lobster", "black cable", "black pouch", "white cup", "white cable"]


def test_three_entries_of_one_label_are_a_row_of_objects():
    cups = [("paper cup", (100 + 120 * k, 400, 200 + 120 * k, 520)) for k in range(3)]
    items = cups + LOOP[:GUARDED_MAX_BOXES - 3]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["paper cup"] * 3 + ["white cable"]


def test_a_reply_that_ended_by_itself_keeps_every_entry():
    # Below the bound and not cut: five cups in a row are five cups, as for any brain.
    cups = [("paper cup", (100 + 120 * k, 400, 200 + 120 * k, 520)) for k in range(5)]
    text = reply([entry(*e) for e in REAL + cups])
    assert labels(parse_guarded_boxes(text, W, H, cut=False)) == labels(parse_boxes(text, W, H))
    assert len(parse_guarded_boxes(text, W, H, cut=False)) == 8


def test_a_loop_alternating_two_labels_keeps_one_box_of_each():
    # s11's shape: "white ..." and "black rectangular object" alternate, then black repeats to the bound.
    alt = [("white rectangular object", (140, 640, 210, 740)), ("black rectangular object", (200, 560, 260, 690)),
           ("white rectangular object", (300, 560, 350, 610)), ("black rectangular object", (250, 550, 300, 625))]
    tail = [("black rectangular object", (150 + 12 * k, 500 - 25 * k, 190 + 12 * k, 560 - 25 * k)) for k in range(30)]
    items = REAL + alt + tail[:GUARDED_MAX_BOXES - len(REAL) - len(alt)]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["red plush lobster", "black pouch", "white cup", "white rectangular object",
                             "black rectangular object", "white rectangular object"]


def test_a_label_looping_between_other_entries_keeps_its_first_box():
    # s10's shape: "black rectangular object" comes back between other entries, never twice in a row.
    others = [(f"{c} rectangular object", (500 + 10 * k, 100, 540 + 10 * k, 140))
              for k, c in enumerate(("white", "yellow", "orange", "silver", "blue", "green", "pink", "gray"))]
    items = REAL[:2] + [x for k, o in enumerate(others) for x in (("black rectangular object",
                                                                  (100 + 30 * k, 900, 140 + 30 * k, 999)), o)]
    assert len(items) == 18
    items += [("black rectangular object", (400, 900, 440, 999)), ("tan rectangular object", (600, 100, 640, 140))]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["red plush lobster", "black pouch", "black rectangular object"] + [
        label for label, _ in others] + ["tan rectangular object"]


@pytest.mark.parametrize("label", ["computer keyboard", "gray keyboard", "black and yellow robotic arm",
                                   "white robot arm", "black computer monitor", "laptop", "Human Hand",
                                   "wooden table", "phone screen"])
def test_an_inventory_entry_the_prompt_excludes_is_dropped(label):
    text = reply([entry(*REAL[0]), entry(label, (100, 100, 300, 300))])
    assert labels(parse_guarded_boxes(text, W, H, cut=False)) == ["red plush lobster"]
    # A refer names what the agent asked for: "find the keyboard" keeps its boxes.
    assert labels(parse_guarded_boxes(text, W, H, cut=False, inventory=False)) == ["red plush lobster", label]


@pytest.mark.parametrize("label", ["table tennis ball", "hand cream", "monitor stand", "arm band", "screwdriver",
                                   "gripper pad", "red plush lobster"])
def test_only_the_head_noun_counts(label):
    text = reply([entry(label, (100, 100, 300, 300))])
    assert labels(parse_guarded_boxes(text, W, H, cut=False)) == [label]


def test_an_exact_repeat_is_dropped_a_second_instance_is_kept():
    items = REAL + [REAL[0], ("red plush lobster", (100, 100, 200, 200))]
    found = parse_guarded_boxes(reply([entry(*e) for e in items]), W, H, cut=False)
    assert labels(found) == ["red plush lobster", "black pouch", "white cup", "red plush lobster"]
    np.testing.assert_allclose(found[3][1], [100, 100, 200, 200])


@pytest.mark.parametrize("text", ['[\n  {"label": "white cab', "", "I see a cup.", "[]\n"])
def test_a_cut_reply_without_a_complete_box_has_none(text):
    assert parse_guarded_boxes(text, W, H, cut=True) == []


def test_the_schema_asks_for_a_bounded_list_of_integer_boxes():
    assert BOX_LIST_SCHEMA["type"] == "array" and BOX_LIST_SCHEMA["maxItems"] == GUARDED_MAX_BOXES
    item = BOX_LIST_SCHEMA["items"]
    assert item["required"] == ["label", "bbox_2d"] and item["additionalProperties"] is False
    assert item["properties"]["bbox_2d"] == {"type": "array", "minItems": 4, "maxItems": 4,
                                             "items": {"type": "integer", "minimum": 0, "maximum": 1000}}
    assert item["properties"]["label"]["maxLength"] == 64
    # A loop of Bolt-sized entries (~36 tokens) ends at the schema's bound well before the token cap.
    assert GUARDED_MAX_BOXES * 40 < GUARDED_MAX_TOKENS < 1500


# -- the detector ---------------------------------------------------------------------------------------------------

def frame() -> Frame:
    rgb = np.zeros((H // 4, W // 4, 3), dtype=np.uint8)
    rgb[40:80, 60:120] = 200
    return Frame(rgb=rgb, depth_m=None, K=np.eye(3), t=0.0)


def detector(model, client, **kw):
    return BrainDetector(base_url="http://127.0.0.1:9/v1", model=model, client=client, segmenter=FakeSegmenter(),
                         **kw)


def test_bolt_asks_for_the_schema_and_keeps_a_cut_loops_objects():
    client = FakeClient()
    client.inventory = reply([entry(*e) for e in REAL + LOOP], closed=False)
    client.finish_reason = "length"
    det = detector("nemotron-3.5-bolt", client)
    found = det.detect(frame())
    assert [d.label for d in found] == ["red plush lobster", "black pouch", "white cup", "white cable"]
    (req,) = client.requests
    assert req["max_tokens"] == GUARDED_MAX_TOKENS
    assert req["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False},
                                 "response_format": {"type": "json_schema",
                                                     "json_schema": {"name": "box_list", "schema": BOX_LIST_SCHEMA}}}


def test_bolt_refers_ask_for_the_schema_too():
    client = FakeClient()
    client.refer = {"white cup": reply([entry("white cup", REAL[2][1])])}
    det = detector("nemotron-3.5-bolt", client)
    assert [d.label for d in det.detect(frame(), ["white cup"])] == ["white cup"]
    assert client.requests[0]["extra_body"]["response_format"]["json_schema"]["schema"] == BOX_LIST_SCHEMA


def test_bolt_finds_an_excluded_category_when_the_agent_asks_for_it():
    client = FakeClient()
    client.refer = {"keyboard": reply([entry("keyboard", (100, 100, 300, 300))])}
    client.inventory = reply([entry(*REAL[0]), entry("keyboard", (100, 100, 300, 300))])
    det = detector("nemotron-3.5-bolt", client)
    assert [d.label for d in det.detect(frame(), ["keyboard"])] == ["keyboard"]
    assert [d.label for d in det.detect(frame())] == ["red plush lobster"]


def test_a_bolt_reply_cut_before_any_complete_box_is_still_a_failed_call(caplog):
    client = FakeClient()
    client.inventory = '[\n  {"label": "white cab'
    client.finish_reason = "length"
    det = detector("nemotron-3.5-bolt", client)
    with caplog.at_level("WARNING"):
        assert det.detect(frame()) == []
    assert "truncated at max_tokens=1024 without a usable box" in caplog.text


def test_bolts_streamed_refresh_keeps_a_cut_loops_objects():
    client = FakeClient()
    client.inventory = reply([entry(*e) for e in REAL + LOOP], closed=False)
    client.finish_reason = "length"
    det = detector("nemotron-3.5-bolt", client)
    f = frame()
    key = det.camera_key(f), None
    from cascade.perception.brain_detector import _thumbnail
    det._start_refresh(key, f, _thumbnail(f), 0.0)
    for t in list(det.refresh_threads()):
        t.join(5)
    assert client.requests[0].get("stream") is True
    assert det._cache[key].labels == ("red plush lobster", "black pouch", "white cup", "white cable")


def _main_request(model, max_tokens=1500):
    """The request main sent for every brain, byte for byte (minus the image)."""
    return {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "<image>"}},
                                                      {"type": "text", "text": INVENTORY_PROMPT}]}],
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _without_image(req):
    req = json.loads(json.dumps(req))
    req["messages"][0]["content"][0]["image_url"]["url"] = "<image>"
    req.pop("timeout", None)
    return req


@pytest.mark.parametrize("model", ["qwen3.8-27b", "cosmos3-edge"])
def test_other_brains_keep_mains_request_and_reject_a_cut_reply(model, caplog):
    client = FakeClient()
    client.inventory = reply([entry(*e) for e in REAL + LOOP], closed=False)
    client.finish_reason = "length"
    det = detector(model, client)
    with caplog.at_level("WARNING"):
        assert det.detect(frame()) == []
    assert "garbled reply, truncated at max_tokens=1500" in caplog.text
    assert _without_image(client.requests[0]) == _main_request(model)


def test_other_brains_parse_a_long_run_as_before():
    items = REAL + LOOP[:GUARDED_MAX_BOXES]
    client = FakeClient()
    client.inventory = reply([entry(*e) for e in items])
    det = detector("qwen3.8-27b", client)
    assert len(det.detect(frame())) == len(items)


def test_reply_guards_can_be_forced_either_way():
    for model, guarded, expect in (("nemotron-3.5-bolt", False, None), ("qwen3.8-27b", True, BOX_LIST_SCHEMA)):
        client = FakeClient()
        client.inventory = reply([entry(*REAL[0])])
        detector(model, client, reply_guards=guarded).detect(frame())
        rf = client.requests[0]["extra_body"].get("response_format")
        assert (rf and rf["json_schema"]["schema"]) == expect
        assert client.requests[0]["max_tokens"] == (GUARDED_MAX_TOKENS if guarded else 1500)


def test_a_lower_configured_cap_wins_over_the_guarded_cap():
    client = FakeClient()
    client.inventory = reply([entry(*REAL[0])])
    detector("nemotron-3.5-bolt", client, max_tokens=512).detect(frame())
    assert client.requests[0]["max_tokens"] == 512


@pytest.mark.parametrize("value, expect", [(None, None), ("auto", None), (True, True), (False, False)])
def test_the_config_key_reaches_the_detector(value, expect):
    assert demo._reply_guards(value) is expect


@pytest.mark.parametrize("value", ["yes", 1, "off"])
def test_a_bad_config_value_is_refused(value):
    with pytest.raises(ValueError, match="reply_guards"):
        demo._reply_guards(value)
