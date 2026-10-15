"""Brain-driven perception: the local vision LLM (the brain) boxes objects, SAM masks them.

This module holds the two prompts measured in the perception bake-off and the parser
that turns the brain's text reply into labelled pixel boxes.

Wire format: the brain answers a JSON list `[{"label": ..., "bbox_2d": [x1, y1, x2, y2]}]`
with coordinates normalized to 0-1000 of the full image.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

_EXCLUDE = ("Exclude the robot arm and its gripper, people and anything in their hands, screens, monitors, "
            "laptops, keyboards, furniture, walls and the table itself.")

# Open-world inventory: every manipulable object on the table, one entry per instance.
INVENTORY_PROMPT = (
    "You are the perception module of a tabletop robot arm. List every manipulable object resting on the "
    "table or inside a container on the table, one entry per instance. " + _EXCLUDE + " Give each a short "
    "specific noun phrase (e.g. 'red plush lobster', 'green filament spool', 'paper cup', 'gray plastic "
    "basket') and a tight bounding box. Answer ONLY with JSON: [{\"label\": \"...\", \"bbox_2d\": [x1, y1, "
    "x2, y2]}] with coordinates normalized to 0-1000. Answer [] if there is none.")

# Referring query for one named object; format with q=<the agent's words>.
REFER_PROMPT = (
    "Locate every instance of \"{q}\" in the image. Answer ONLY with JSON: [{{\"label\": \"{q}\", "
    "\"bbox_2d\": [x1, y1, x2, y2]}}] with coordinates normalized to 0-1000, or [] if it is not visible.")

_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?")


def _answer_list(text: str) -> list:
    """The FIRST JSON list in `text` that holds at least one dict with `bbox_2d`.

    The brain sometimes appends a second `[]` after its answer, and lists without
    boxes (an empty answer, a list of names) are not the answer, so keep scanning.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and any(isinstance(o, dict) and "bbox_2d" in o for o in value):
            return value
    return []


def _is_coord(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_boxes(text: str, width: int, height: int) -> list[tuple[str, np.ndarray]]:
    """Brain reply -> [(label, float32 xyxy pixel box)].

    Thinking blocks (closed, unterminated, or only closed because the template
    opened them) and code fences are stripped first. Coordinates are clamped to
    0..1000, corners ordered, then scaled to pixels; entries without a label or
    with a malformed or zero-area box are dropped.
    """
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = _FENCE.sub("", text)

    out: list[tuple[str, np.ndarray]] = []
    for item in _answer_list(text):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("bbox_2d")
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(box, list) or len(box) != 4 or not all(_is_coord(v) for v in box):
            continue
        x1, y1, x2, y2 = (min(max(float(v), 0.0), 1000.0) for v in box)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 == x1 or y2 == y1:
            continue
        sx, sy = width / 1000.0, height / 1000.0
        out.append((label.strip(), np.array([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dtype=np.float32)))
    return out
