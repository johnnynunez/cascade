"""Color regressions from real segmented RGB-D observations, no model needed."""
from pathlib import Path
import json

import numpy as np
import pytest

from cascade.perception.colors import mask_color


_FIXTURES = Path(__file__).parent / "fixtures"
_CASES = json.loads((_FIXTURES / "segmented_color_samples.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["key"])
def test_recorded_segmentation_colors(case):
    """Pastel surfaces stay chromatic without relabeling neutral controls.

    These are unchanged BGR pixels selected by real YOLOE masks from two
    cameras, using mask_color's deterministic 4000-pixel sampling. Detection
    names are provenance only and are never passed to the color estimator.
    """
    with np.load(_FIXTURES / "segmented_color_samples.npz") as data:
        bgr = data[case["key"]]
    mask = np.ones(bgr.shape[:2], dtype=bool)
    assert mask_color(bgr, mask) == case["expected"]


@pytest.mark.parametrize("hue, expected", [
    (3, "pink"), (17, "orange"), (28, "yellow"), (60, "green"),
    (95, "cyan"), (115, "blue"), (135, "purple"), (160, "pink"),
])
def test_pastel_chroma_is_not_limited_to_one_hue(hue, expected):
    """A saturation fix must apply to arbitrary objects and hue bands."""
    from cascade.perception.colors import classify_hsv
    assert classify_hsv(hue, 35, 230) == expected


@pytest.mark.parametrize("hue", range(0, 180, 15))
@pytest.mark.parametrize("value, expected", [(230, "white"), (100, "gray"), (30, "black")])
def test_weak_tints_and_dark_pixels_remain_achromatic(hue, value, expected):
    from cascade.perception.colors import classify_hsv
    assert classify_hsv(hue, 20, value) == expected


def test_white_object_with_small_colored_logo_stays_white():
    """Do not solve pale colors by discarding the achromatic majority."""
    bgr = np.full((100, 100, 3), 230, dtype=np.uint8)
    bgr[:15] = (203, 105, 255)
    assert mask_color(bgr, np.ones(bgr.shape[:2], dtype=bool)) == "white"


def test_empty_mask_is_unknown_not_requested_color():
    bgr = np.full((20, 20, 3), 230, dtype=np.uint8)
    assert mask_color(bgr, np.zeros(bgr.shape[:2], dtype=bool)) is None


@pytest.mark.parametrize("color, expected_index", [("pink", 2), ("green", 1)])
def test_localize_uses_measured_color_not_label_or_confidence(color, expected_index):
    from cascade.perception.detector import MockDetector
    from cascade.perception.grounding import Extrinsics, localize_object
    frame, dets = _recorded_colors_in_synthetic_layout()
    fix = localize_object(frame, color + " object", MockDetector(detections=dets),
                          Extrinsics(), color=color)
    assert fix.detection is dets[expected_index]


@pytest.mark.parametrize("absent", ["blue", "red", "purple"])
def test_localize_refuses_absent_color_with_real_pixel_samples(absent):
    from cascade.perception.detector import MockDetector
    from cascade.perception.grounding import Extrinsics, localize_object
    from cascade.types import SkillError
    frame, dets = _recorded_colors_in_synthetic_layout()
    with pytest.raises(SkillError, match="not " + absent):
        localize_object(frame, absent + " object", MockDetector(detections=dets),
                        Extrinsics(), color=absent)


def _recorded_colors_in_synthetic_layout():
    """Actual segmented colors; synthetic geometry, not live pose evidence."""
    from cascade.types import Detection, Frame
    bgr = np.zeros((80, 150, 3), dtype=np.uint8)
    dets = []
    # Orange has highest confidence; all share an arbitrary unseen class name.
    with np.load(_FIXTURES / "segmented_color_samples.npz") as samples:
        for index, key in enumerate(("cam0_4", "cam0_0", "cam0_2")):
            x0, x1 = index * 50, (index + 1) * 50
            bgr[:, x0:x1] = samples[key].reshape(80, 50, 3)
            mask = np.zeros(bgr.shape[:2], dtype=bool)
            mask[:, x0:x1] = True
            dets.append(Detection("unseen visitor item", .99 - .1 * index,
                                  np.array([x0, 0, x1, 80]), mask=mask))
    frame = Frame(bgr, np.full(bgr.shape[:2], .5, dtype=np.float32),
                  np.array([[200., 0., 75.], [0., 200., 40.], [0., 0., 1.]]))
    return frame, dets
