"""Malformed archived inputs must not produce measured-coverage claims."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "nvblox_payload_replay", Path(__file__).resolve().parents[1] / "benchmark/diagnostics/nvblox_payload_replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)
PROP = "/World_Props/tomato_can"


def archive(folder, *, stamp=1., held=False, mutate_frame=None, mutate_capture=None):
    folder.mkdir()
    frame = {"depth": np.array([[.5, .6]], np.float32), "K": np.eye(3), "T": np.eye(4),
             "robot_mask": np.array([[False, held]]), "prop_0": np.array([[False, True]])}
    capture = {"backend": "isaac", "camera": "cam0", "source": ["localhost", 8691],
               "t": stamp, "contact_paths": [PROP] if held else [],
               "proprioception": {"backend": "isaac", "robot_id": "/Robot", "t": stamp,
                                  "time_source": "physics_loop_monotonic", "joint_convention": "asset"}}
    if mutate_frame:
        mutate_frame(frame)
    if mutate_capture:
        mutate_capture(capture)
    path = folder / "cam0.npz"
    np.savez_compressed(path, **frame)
    (folder / "metadata.json").write_text(json.dumps({"cameras": {"cam0": {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "capture": capture,
        "props": {"prop_0": PROP}}}}))
    return REPLAY.load_capture(folder, ["cam0"], PROP)


@pytest.mark.parametrize("mutation", [
    lambda frame: frame.update(depth=frame["depth"].astype(np.uint16)),
    lambda frame: frame["K"].__setitem__((0, 0), 0),
    lambda frame: frame["T"].__setitem__((slice(0, 3), slice(0, 3)), 2 * np.eye(3)),
])
def test_checksums_do_not_validate_depth_units_or_camera_geometry(tmp_path, mutation):
    with pytest.raises(ValueError, match="floating-point|Calibration"):
        archive(tmp_path / "bad", mutate_frame=mutation)


@pytest.mark.parametrize("mutation", [
    lambda capture: capture.update(camera="wrong_camera"),
    lambda capture: capture.update(backend="mock"),
    lambda capture: capture["proprioception"].update(t=2.),
])
def test_checksums_do_not_bind_capture_to_requested_camera(tmp_path, mutation):
    with pytest.raises(ValueError, match="Unbound Isaac capture"):
        archive(tmp_path / "bad", mutate_capture=mutation)


def test_pair_requires_same_robot_and_actual_later_attachment(tmp_path):
    anchor, a = archive(tmp_path / "anchor")
    held, h = archive(tmp_path / "held", stamp=2., held=True)
    REPLAY.validate_pair(anchor, held, a, h, PROP)
    np.testing.assert_allclose(REPLAY.measured_surface(held["cam0"]), [[.6, 0, .6]])
    h["cameras"]["cam0"]["capture"]["proprioception"]["robot_id"] = "/OtherRobot"
    with pytest.raises(ValueError, match="source, robot or clock changed"):
        REPLAY.validate_pair(anchor, held, a, h, PROP)
