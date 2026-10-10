"""Markerless records in the schema-v1 hand-eye file, with a PER-METHOD gate.

The runtime loader (Extrinsics.from_config -> hand_eye_json) must accept a
markerless record exactly like a marker one -- same file, same kind, same
schema -- while recomputing ITS gate (assess_markerless) from the stored
metrics on every load. Marker records must keep loading and gating exactly
as before (test_handeye_dataset / test_extrinsics_handeye stay untouched).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cascade.calibration.dataset import (
    KIND,
    SCHEMA_VERSION,
    assess_record,
    load_hand_eye,
    read_hand_eye,
    save_hand_eye,
)
from cascade.calibration.frames import so3_exp
from cascade.calibration.markerless import (
    METHOD,
    DepthSample,
    MarkerlessFit,
    PoseStats,
    assess_markerless,
    record_from_markerless,
)
from cascade.config import Cfg
from cascade.perception.grounding import Extrinsics

GOOD = {"n_poses": 13, "n_poses_used": 13, "n_inliers": 12000, "n_visible": 16000,
        "inlier_fraction": 0.75, "rmse_m": 0.003, "pose_offset_max_m": 0.002,
        "pose_offset_max_deg": 0.4, "degeneracy_min_eig": 0.05, "condition_number": 14.0,
        "tcp_spread_m": 0.02, "rotation_spread_deg": 26.0, "fitness": 0.75}

T = np.eye(4)
T[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]) @ so3_exp([0.05, -0.06, 0])
T[:3, 3] = [0.30, -0.03, 0.95]


def _record(metrics=GOOD, n=3):
    K = np.array([[320.0, 0, 319.5], [0, 320.0, 179.5], [0, 0, 1]])
    samples = [DepthSample(q=tuple(float(v) for v in np.linspace(0, 1, 6) + i),
                           T_gripper2base=np.eye(4), depth_m=np.zeros((4, 4), np.float32), K=K,
                           label=f"pose_{i:02d}") for i in range(n)]
    poses = tuple(PoseStats(f"pose_{i:02d}", 1000, 800, 0.003, 0.001, 0.2) for i in range(n))
    fit = MarkerlessFit(T_cam2base=T, metrics=dict(metrics), poses=poses)
    return record_from_markerless(fit, samples, camera="d455f_scene", camera_serial="261422303968",
                                  arm="rebot_rs", ee_frame="gripper_end", K=K,
                                  image_size=(640, 360), note="test",
                                  depth_files=[f"captures/pose_{i:02d}_depth.png" for i in range(n)])


def test_markerless_round_trip_in_the_same_schema(tmp_path):
    rec = _record()
    assert rec.acceptable and rec.method == METHOD == "depth_icp_markerless"
    path = save_hand_eye(tmp_path / "scene.json", rec)
    raw = json.loads(path.read_text())
    assert raw["kind"] == KIND and raw["schema_version"] == SCHEMA_VERSION
    assert raw["method"] == METHOD and raw["mode"] == "eye_to_hand"
    assert "T_cam2base" in raw
    # No marker: no marker transform or spec pretending there was one.
    assert "T_marker2gripper" not in raw and raw["marker"] is None
    assert raw["acceptable"] is True and raw["rejection_reasons"] == []
    s0 = raw["samples"][0]
    assert s0["label"] == "pose_00" and len(s0["q"]) == 6 and s0["n_inliers"] == 800
    assert s0["depth_file"] == "captures/pose_00_depth.png"
    back = load_hand_eye(path)
    assert back is not None and back.method == METHOD
    assert np.allclose(back.T_cam2base, T)
    assert back.T_marker is None and back.marker is None
    assert back.samples[2].label == "pose_02"
    assert "markerless" in back.summary() and "[OK]" in back.summary()


def test_the_markerless_gate_is_recomputed_on_load(tmp_path):
    degenerate = dict(GOOD, degeneracy_min_eig=0.0005)
    path = save_hand_eye(tmp_path / "bad.json", _record(degenerate))
    raw = json.loads(path.read_text())
    assert raw["acceptable"] is False
    assert any("degenerate" in r for r in raw["rejection_reasons"])
    raw["acceptable"], raw["rejection_reasons"] = True, []
    path.write_text(json.dumps(raw))
    assert load_hand_eye(path) is None
    assert load_hand_eye(path, trust_unacceptable=True) is not None


def test_each_method_gets_its_own_gate():
    # A good markerless fit has no marker metrics -- the marker gate would
    # refuse it for "missing translation rmse"; its own gate passes it.
    assert assess_record(METHOD, GOOD) == []
    assert assess_record("joint_se3_lm_huber", GOOD)
    assert assess_record(METHOD, {}) == assess_markerless({})
    assert any("unknown calibration method" in r for r in assess_record("sorcery", GOOD))


def test_unknown_method_never_loads(tmp_path):
    path = save_hand_eye(tmp_path / "x.json", _record())
    raw = json.loads(path.read_text())
    raw["method"] = "sorcery"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown calibration method"):
        read_hand_eye(path)
    e = Extrinsics.from_config(Cfg({"hand_eye_json": str(path)}))
    assert not e.calibrated and "unknown calibration method" in e.calibration_error


def test_markerless_eye_in_hand_is_malformed(tmp_path):
    path = save_hand_eye(tmp_path / "x.json", _record())
    raw = json.loads(path.read_text())
    raw["mode"] = "eye_in_hand"
    raw["T_cam2gripper"] = raw.pop("T_cam2base")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="eye_to_hand"):
        read_hand_eye(path)


def test_runtime_loads_an_acceptable_markerless_record(tmp_path):
    path = save_hand_eye(tmp_path / "scene.json", _record())
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "hand_eye_json": str(path)}),
                               camera_serial="261422303968")
    assert e.calibrated
    assert np.allclose(e.cam_to_base(), T)


def test_runtime_refuses_a_rejected_markerless_record(tmp_path):
    path = save_hand_eye(tmp_path / "scene.json", _record(dict(GOOD, n_poses_used=1)))
    e = Extrinsics.from_config(Cfg({"hand_eye_json": str(path)}))
    assert not e.calibrated
    assert "usable poses" in e.calibration_error
