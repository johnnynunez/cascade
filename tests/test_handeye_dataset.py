"""Schema-v1 hand-eye record: save/load, and the gate the loader enforces.

Ported from WRC tests/test_calib_dataset.py. WRC's loader returned whatever
the file held; cascade's extrinsic loader (`perception.calibration.
load_extrinsic`) refuses a rejected record, and so does this one -- with the
gate RECOMPUTED from the stored metrics, so a hand-edited `"acceptable":
true` or a threshold tightened since the file was written cannot smuggle a
bad calibration in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cascade.calibration.dataset import (
    KIND,
    SCHEMA_VERSION,
    MarkerSpec,
    load_hand_eye,
    read_hand_eye,
    record_from_fit,
    save_hand_eye,
)
from cascade.calibration.frames import se3_inv, so3_exp
from cascade.calibration.handeye import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    HandEyeSample,
    solve_hand_eye,
)


def _T(t, w):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(w, dtype=float))
    T[:3, 3] = t
    return T


X = _T([0.30, -0.05, 0.85], [3.0, 0.05, 0.1])
Y = _T([0.005, 0.01, 0.03], [0.05, -0.02, 0.03])


def _fit(mode=EYE_TO_HAND, *, degenerate=False, n=12, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        w = [0.0, 0.0, 0.1 * i] if degenerate else rng.normal(scale=0.5, size=3)
        G = _T(rng.uniform([0.22, -0.1, 0.2], [0.38, 0.1, 0.36]), w)
        M = (se3_inv(X) @ G @ Y if mode == EYE_TO_HAND
             else se3_inv(X) @ se3_inv(G) @ Y)
        samples.append(HandEyeSample(G, M, label=f"auto_{i:02d}",
                                     q=tuple(float(v) for v in rng.normal(size=6)),
                                     reprojection_px=0.2))
    return solve_hand_eye(samples, mode), samples


def _record(mode=EYE_TO_HAND, **kw):
    fit, samples = _fit(mode, **kw)
    return record_from_fit(
        fit, samples, marker=MarkerSpec("4x4_50", 0, 0.10), camera="d455f_scene",
        camera_serial="261422303968", arm="rebot_rs", ee_frame="gripper_end",
        K=np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]]), D=np.zeros(5),
        image_size=(1280, 720), note="test")


def test_eye_to_hand_round_trip(tmp_path):
    rec = _record(EYE_TO_HAND)
    assert rec.acceptable
    path = save_hand_eye(tmp_path / "scene.json", rec)
    raw = json.loads(path.read_text())
    assert raw["kind"] == KIND and raw["schema_version"] == SCHEMA_VERSION == 1
    assert raw["mode"] == "eye_to_hand"
    # The matrices are stored under names that say what they ARE.
    assert "T_cam2base" in raw and "T_marker2gripper" in raw
    assert "T_cam2gripper" not in raw and "T_marker2base" not in raw
    assert raw["acceptable"] is True and raw["rejection_reasons"] == []
    assert raw["marker"] == {"dictionary": "4x4_50", "id": 0, "size_m": 0.10}
    assert "frame_convention" in raw and "gripper_end" in json.dumps(raw)

    back = load_hand_eye(path)
    assert back is not None
    assert np.allclose(back.T_cam2base, rec.T_cam2base)
    assert np.allclose(back.T_marker2gripper, rec.T_marker2gripper)
    assert back.camera_serial == "261422303968" and back.arm == "rebot_rs"
    assert np.allclose(back.K, rec.K) and back.image_size == (1280, 720)
    assert len(back.samples) == 12
    assert back.samples[3].label == "auto_03" and len(back.samples[3].q) == 6
    assert np.allclose(back.samples[3].T_marker2cam, rec.samples[3].T_marker2cam)
    assert back.metrics["translation_rmse_m"] == pytest.approx(rec.metrics["translation_rmse_m"])


def test_eye_in_hand_round_trip(tmp_path):
    rec = _record(EYE_IN_HAND)
    path = save_hand_eye(tmp_path / "wrist.json", rec)
    raw = json.loads(path.read_text())
    assert raw["mode"] == "eye_in_hand"
    assert "T_cam2gripper" in raw and "T_marker2base" in raw
    assert "T_cam2base" not in raw
    back = load_hand_eye(path)
    assert np.allclose(back.T_cam2gripper, rec.T_cam2gripper)
    assert np.allclose(back.T_hand_eye, X, atol=1e-6)


def test_inlier_flags_survive(tmp_path):
    rec = _record()
    rec2 = rec.with_outliers((1, 4))
    back = read_hand_eye(save_hand_eye(tmp_path / "x.json", rec2))
    assert back.outlier_indices == (1, 4)
    assert [s_in for s_in in back.inlier_flags] == [i not in (1, 4) for i in range(12)]


def test_a_rejected_calibration_does_not_load_by_default(tmp_path):
    rec = _record(degenerate=True)
    assert not rec.acceptable
    path = save_hand_eye(tmp_path / "bad.json", rec)
    raw = json.loads(path.read_text())
    assert raw["acceptable"] is False
    assert any("rotation spread" in r for r in raw["rejection_reasons"])
    assert load_hand_eye(path) is None
    # Explicit override for inspection / re-solving only.
    assert load_hand_eye(path, trust_unacceptable=True) is not None


def test_hand_edited_acceptable_flag_is_not_trusted(tmp_path):
    path = save_hand_eye(tmp_path / "bad.json", _record(degenerate=True))
    raw = json.loads(path.read_text())
    raw["acceptable"] = True
    raw["rejection_reasons"] = []
    path.write_text(json.dumps(raw))
    assert load_hand_eye(path) is None


def test_stored_rejection_is_honoured_even_if_metrics_now_pass(tmp_path):
    path = save_hand_eye(tmp_path / "x.json", _record())
    raw = json.loads(path.read_text())
    raw["acceptable"] = False
    path.write_text(json.dumps(raw))
    assert load_hand_eye(path) is None


def test_missing_file_is_none(tmp_path):
    assert load_hand_eye(tmp_path / "nope.json") is None


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("kind"),
    lambda d: d.update(schema_version=2),          # e.g. WRC's anonymised file
    lambda d: d.update(mode="eye_on_base"),
    lambda d: d.update(T_cam2base=[[1, 0, 0, 0]] * 4),
    lambda d: d["T_cam2base"][0].__setitem__(3, float("nan")),
])
def test_malformed_records_raise(tmp_path, mutate):
    path = save_hand_eye(tmp_path / "x.json", _record())
    raw = json.loads(path.read_text())
    mutate(raw)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        read_hand_eye(path)
    with pytest.raises(ValueError):
        load_hand_eye(path)


def test_save_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "sub" / "x.json"
    save_hand_eye(path, _record(seed=1))
    save_hand_eye(path, _record(seed=2))
    assert read_hand_eye(path).acceptable
    assert sorted(p.name for p in path.parent.iterdir()) == ["x.json"]
