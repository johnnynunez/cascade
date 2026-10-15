"""Brain box replies -> labelled pixel boxes (perception/brain_detector.py).

s05, s16 and w03 are real Qwen3.8 replies recorded by the perception bake-off
on 1280x720 frames; thinking.txt is hand-written. Expected pixel boxes are
derived by hand: x * 1280 / 1000, y * 720 / 1000.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cascade.perception.brain_detector import REFER_PROMPT, parse_boxes

FIXTURES = Path(__file__).parent / "fixtures" / "brain_replies"
W, H = 1280, 720


def _reply(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


def _labels(boxes):
    return [label for label, _ in boxes]


def test_trailing_empty_array_keeps_answer():
    # s16: the 12-object answer is followed by "\n\n\n[]"; the answer must win.
    boxes = parse_boxes(_reply("s16"), W, H)
    assert len(boxes) == 12
    label, box = boxes[-1]
    assert label == "white box with 'AVELINES' text"
    np.testing.assert_allclose(box, [664.32, 577.44, 1113.6, 720.0], rtol=1e-6)


def test_fenced_reply_scaled_to_pixels():
    boxes = parse_boxes(_reply("s05"), W, H)
    assert _labels(boxes) == [
        "black computer tower", "silver computer tower", "red plush lobster",
        "black computer mouse", "paper cup", "white box", "gray plastic basket",
    ]
    _, box = boxes[0]
    assert box.dtype == np.float32
    np.testing.assert_allclose(box, [426.24, 98.64, 570.88, 293.76], rtol=1e-6)


def test_single_object_reply():
    (label, box), = parse_boxes(_reply("w03"), W, H)
    assert label == "pink plush lobster"
    np.testing.assert_allclose(box, [273.92, 392.4, 974.08, 720.0], rtol=1e-6)


def test_thinking_and_fences_stripped():
    # The reasoning quotes an example list with bbox_2d before the real answer.
    boxes = parse_boxes(_reply("thinking"), 1000, 1000)
    assert _labels(boxes) == ["paper cup"]
    np.testing.assert_allclose(boxes[0][1], [450, 400, 550, 600])


def test_unterminated_thinking_yields_nothing():
    text = '<think>Maybe the format is [{"label": "cup", "bbox_2d": [0, 0, 500, 500]}] and'
    assert parse_boxes(text, 1000, 1000) == []


def test_reasoning_before_orphan_think_close_is_ignored():
    # Chat templates that open <think> in the prompt leave only the closing tag.
    text = ('Example: [{"label": "cup", "bbox_2d": [0, 0, 500, 500]}]\n</think>\n'
            '[{"label": "red plush lobster", "bbox_2d": [100, 100, 200, 200]}]')
    assert _labels(parse_boxes(text, 1000, 1000)) == ["red plush lobster"]


def test_first_answer_list_wins():
    text = ('[{"label": "paper cup", "bbox_2d": [0, 0, 100, 100]}]\n'
            '[{"label": "green ball", "bbox_2d": [0, 0, 100, 100]}]')
    assert _labels(parse_boxes(text, 1000, 1000)) == ["paper cup"]


def test_lists_without_boxes_are_skipped():
    text = ('Kinds: ["paper cup"]. Empty: []. Answer: '
            '[{"label": "paper cup", "bbox_2d": [0, 0, 500, 500]}]')
    assert _labels(parse_boxes(text, 1000, 1000)) == ["paper cup"]


def test_scale_clamp_and_order():
    (_, box), = parse_boxes('[{"label":"cup","bbox_2d":[600,500,400,1200]}]', 1280, 720)
    np.testing.assert_allclose(box, [512, 360, 768, 720])
    (_, box), = parse_boxes('[{"label":"cup","bbox_2d":[-100,-20,500,500]}]', 1000, 1000)
    np.testing.assert_allclose(box, [0, 0, 500, 500])
    (_, box), = parse_boxes('[{"label":"cup","bbox_2d":[100,900,500,100]}]', 1000, 1000)
    np.testing.assert_allclose(box, [100, 100, 500, 900])


def test_zero_area_and_bad_entries_dropped():
    assert parse_boxes('[{"label":"a","bbox_2d":[5,5,5,9]},{"bbox_2d":[1,2,3,4]},"x"]', 100, 100) == []
    # Zero width / height only after clamping to the image.
    assert parse_boxes('[{"label":"b","bbox_2d":[1100,10,1200,50]}]', 100, 100) == []
    assert parse_boxes('[{"label":"c","bbox_2d":[10,-50,50,-5]}]', 100, 100) == []


def test_malformed_entries_dropped_and_label_stripped():
    text = ('[{"label": "  cup ", "bbox_2d": [0, 0, 100, 100]},'
            ' {"label": " ", "bbox_2d": [0, 0, 100, 100]},'
            ' {"label": "a", "bbox_2d": [1, 2, "x", 40]},'
            ' {"label": "b", "bbox_2d": [1, 2, 30]},'
            ' {"label": "c", "bbox_2d": [1, 2, 30, 40, 50]},'
            ' {"label": "d", "bbox_2d": "1, 2, 30, 40"},'
            ' {"label": "e", "bbox_2d": [NaN, 2, 30, 40]},'
            ' {"label": "f", "bbox_2d": [true, 2, 30, 40]}]')
    boxes = parse_boxes(text, 100, 100)
    assert _labels(boxes) == ["cup"]
    np.testing.assert_allclose(boxes[0][1], [0, 0, 10, 10])


def test_garbage_is_empty():
    assert parse_boxes("I cannot see anything.", 640, 480) == []
    assert parse_boxes("", 640, 480) == []


def test_refer_prompt_takes_the_query():
    prompt = REFER_PROMPT.format(q="paper cup")
    assert 'Locate every instance of "paper cup"' in prompt
    assert '[{"label": "paper cup", "bbox_2d": [x1, y1, x2, y2]}]' in prompt
