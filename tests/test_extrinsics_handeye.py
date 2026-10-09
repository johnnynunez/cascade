"""Camera profiles -> Extrinsics: hand-eye JSON records + ADR-0009 compensation.

Ported from WRC tests/test_extrinsics_loader.py and
tests/test_hand_eye_compensation_per_camera.py, adapted to cascade's gate:

* a profile's ``extrinsics.hand_eye_json`` points at a schema-v1 record
  (scripts/calibrate_handeye.py output);
* a record that is missing, malformed, rejected by the quality gate or made
  for ANOTHER camera serial leaves the camera UNCALIBRATED: it streams but
  never fuses, and any back-projection through it raises SkillError naming
  why. WRC warned and used the matrix anyway; with two identical D455Fs a
  swapped serial is exactly the silent centimetres-off failure to prevent;
* ``extrinsics.hand_eye_compensation_m: {x, y, z}`` is a per-camera
  base-frame nudge (identity by default, never inherited from demo.yaml):
      eye_to_hand: T_cam2base = T_comp @ T_hand_eye
      eye_in_hand: T_cam2base = T_comp @ T_tcp2base @ T_hand_eye
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cascade.calibration.dataset import MarkerSpec, record_from_fit, save_hand_eye
from cascade.calibration.frames import se3_inv, so3_exp
from cascade.calibration.handeye import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    HandEyeSample,
    solve_hand_eye,
)
from cascade.config import Cfg, load_demo_config
from cascade.perception.grounding import Extrinsics
from cascade.types import SkillError


def _T(t, w):
    T = np.eye(4)
    T[:3, :3] = so3_exp(np.asarray(w, dtype=float))
    T[:3, 3] = t
    return T


X = _T([0.30, -0.05, 0.85], [3.0, 0.05, 0.1])
Y = _T([0.005, 0.01, 0.03], [0.05, -0.02, 0.03])


def _record_file(tmp_path, mode=EYE_TO_HAND, *, degenerate=False, serial="261422303968",
                 name="cal.json"):
    rng = np.random.default_rng(3)
    samples = []
    for i in range(12):
        w = [0.0, 0.0, 0.1 * i] if degenerate else rng.normal(scale=0.5, size=3)
        G = _T(rng.uniform([0.22, -0.1, 0.2], [0.38, 0.1, 0.36]), w)
        M = (se3_inv(X) @ G @ Y if mode == EYE_TO_HAND else se3_inv(X) @ se3_inv(G) @ Y)
        samples.append(HandEyeSample(G, M, label=f"s{i}"))
    fit = solve_hand_eye(samples, mode)
    rec = record_from_fit(fit, samples, marker=MarkerSpec(), camera="d455f_scene",
                          camera_serial=serial, arm="rebot_rs", ee_frame="gripper_end")
    return save_hand_eye(tmp_path / name, rec)


# ── loading a record ─────────────────────────────────────────────────────


def test_eye_to_hand_record_loads(tmp_path):
    p = _record_file(tmp_path)
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "hand_eye_json": str(p)}),
                               camera_serial="261422303968")
    assert e.calibrated and e.mode == "eye_to_hand"
    assert np.allclose(e.cam_to_base(), X, atol=1e-6)
    assert np.allclose(e.T, X, atol=1e-6)
    assert "cal.json" in e.source


def test_mode_defaults_to_the_record(tmp_path):
    p = _record_file(tmp_path, EYE_IN_HAND)
    e = Extrinsics.from_config(Cfg({"hand_eye_json": str(p)}), fk_tcp2base=lambda: np.eye(4))
    assert e.mode == "eye_in_hand"


def test_eye_in_hand_record_composes_with_live_fk(tmp_path):
    p = _record_file(tmp_path, EYE_IN_HAND)
    tcp = _T([0.3, 0.0, 0.3], [0.0, 0.6, 0.0])
    e = Extrinsics.from_config(Cfg({"mode": "eye_in_hand", "hand_eye_json": str(p)}),
                               fk_tcp2base=lambda: tcp)
    assert e.calibrated
    assert np.allclose(e.T, X, atol=1e-6)               # T_cam2gripper
    assert np.allclose(e.cam_to_base(), tcp @ X, atol=1e-6)


def test_profile_mode_contradicting_the_record_is_a_config_error(tmp_path):
    p = _record_file(tmp_path, EYE_IN_HAND)
    with pytest.raises(ValueError, match="mode"):
        Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "hand_eye_json": str(p)}))


def test_rejected_record_leaves_the_camera_uncalibrated(tmp_path):
    p = _record_file(tmp_path, degenerate=True)
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "hand_eye_json": str(p)}))
    assert not e.calibrated
    assert "rotation spread" in e.calibration_error
    with pytest.raises(SkillError, match="rotation spread"):
        e.cam_to_base()
    # The static matrix is not a usable number either (evidence audits read it).
    assert not np.any(np.isfinite(e.T))


def test_missing_record_leaves_the_camera_uncalibrated(tmp_path):
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand",
                                    "hand_eye_json": str(tmp_path / "nope.json"),
                                    "T": np.eye(4).tolist()}))
    # A placeholder inline T must NOT be used as a fallback for a configured
    # calibration that is absent: that is the silent wrong-extrinsics case.
    assert not e.calibrated and "not found" in e.calibration_error
    with pytest.raises(SkillError):
        e.cam_to_base()


def test_malformed_record_leaves_the_camera_uncalibrated(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema_version": 2, "T_cam2base": np.eye(4).tolist()}))
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "hand_eye_json": str(p)}))
    assert not e.calibrated and "hand-eye record" in e.calibration_error


def test_serial_mismatch_is_refused(tmp_path):
    p = _record_file(tmp_path, serial="261422303968")
    cfg = Cfg({"mode": "eye_to_hand", "hand_eye_json": str(p)})
    e = Extrinsics.from_config(cfg, camera_serial="261522301814")   # the wrist unit
    assert not e.calibrated and "261522301814" in e.calibration_error
    assert Extrinsics.from_config(cfg, camera_serial="261422303968").calibrated
    # No serial pinned on either side: nothing to compare, nothing refused.
    assert Extrinsics.from_config(cfg).calibrated
    p2 = _record_file(tmp_path, serial="", name="noserial.json")
    assert Extrinsics.from_config(Cfg({"hand_eye_json": str(p2)}),
                                  camera_serial="261422303968").calibrated


def test_legacy_inline_and_npz_paths_are_still_calibrated(tmp_path):
    assert Extrinsics.from_config(Cfg({"T": np.eye(4).tolist()})).calibrated
    npz = tmp_path / "he.npz"
    np.savez(npz, T_result=X, mode=np.array(["eye_to_hand"]))
    e = Extrinsics.from_config(Cfg({"hand_eye_npz": str(npz)}))
    assert e.calibrated and np.allclose(e.cam_to_base(), X)


# ── ADR-0009 hand_eye_compensation_m ─────────────────────────────────────


def test_compensation_defaults_to_identity():
    e = Extrinsics.from_config(Cfg({"mode": "eye_to_hand", "T": np.eye(4).tolist()}))
    assert np.allclose(e.T_compensation, np.eye(4))
    assert np.allclose(e.cam_to_base(), np.eye(4))


def test_compensation_eye_to_hand_shifts_the_origin():
    e = Extrinsics.from_config(Cfg({
        "mode": "eye_to_hand", "T": np.eye(4).tolist(),
        "hand_eye_compensation_m": {"x": 0.0, "y": 0.0, "z": -0.01},
    }))
    expected = np.eye(4)
    expected[2, 3] = -0.01
    assert np.allclose(e.cam_to_base(), expected)
    # .T is the EFFECTIVE static transform, so audits record what was used.
    assert np.allclose(e.T, expected)


def test_compensation_eye_in_hand_is_applied_in_the_base_frame():
    tcp = _T([0.3, 0.0, 0.3], [0.0, 0.6, 0.0])
    T_he = _T([0.05, 0.0, 0.02], [0.0, 1.2, 0.0])
    e = Extrinsics.from_config(Cfg({
        "mode": "eye_in_hand", "T": T_he.tolist(),
        "hand_eye_compensation_m": {"x": 0.002, "y": 0.0, "z": -0.01},
    }), fk_tcp2base=lambda: tcp)
    comp = np.eye(4)
    comp[:3, 3] = [0.002, 0.0, -0.01]
    assert np.allclose(e.cam_to_base(), comp @ tcp @ T_he)
    assert np.allclose(e.T, T_he)     # T_cam2gripper itself is not altered


def test_compensation_applies_on_top_of_a_record(tmp_path):
    p = _record_file(tmp_path)
    e = Extrinsics.from_config(Cfg({"hand_eye_json": str(p),
                                    "hand_eye_compensation_m": {"x": 0.0, "y": 0.004, "z": 0.0}}))
    assert np.allclose(e.cam_to_base()[:3, 3], X[:3, 3] + [0.0, 0.004, 0.0], atol=1e-6)


def test_compensation_rejects_non_finite_or_partial_garbage():
    with pytest.raises(ValueError, match="hand_eye_compensation_m"):
        Extrinsics.from_config(Cfg({"T": np.eye(4).tolist(),
                                    "hand_eye_compensation_m": {"x": "a lot"}}))
    with pytest.raises(ValueError, match="hand_eye_compensation_m"):
        Extrinsics.from_config(Cfg({"T": np.eye(4).tolist(),
                                    "hand_eye_compensation_m": {"z": float("nan")}}))
    with pytest.raises(ValueError, match="hand_eye_compensation_m"):
        Extrinsics.from_config(Cfg({"T": np.eye(4).tolist(),
                                    "hand_eye_compensation_m": {"q": 0.01}}))


def _config_tree(root: Path, camera: dict, demo_extra: dict | None = None) -> Path:
    cdir = root / "configs"
    for sub in ("cameras", "arms", "llm"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    demo = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "demo.yaml").read_text())
    demo.update(demo_extra or {})
    (cdir / "demo.yaml").write_text(yaml.safe_dump(demo))
    (cdir / "arms" / "mock.yaml").write_text(
        (Path(__file__).resolve().parents[1] / "configs" / "arms" / "mock.yaml").read_text())
    (cdir / "llm" / "mock.yaml").write_text(yaml.safe_dump({"type": "mock"}))
    (cdir / "cameras" / "cal_cam.yaml").write_text(yaml.safe_dump(camera))
    return cdir


def test_per_camera_compensation_loads_from_the_profile(tmp_path):
    cdir = _config_tree(tmp_path, {"type": "mock", "extrinsics": {
        "mode": "eye_to_hand", "T": np.eye(4).tolist(),
        "hand_eye_compensation_m": {"x": 0.005, "y": 0.0, "z": -0.015}}})
    cfg = load_demo_config(camera="cal_cam", config_dir=cdir)
    comp = cfg.camera.extrinsics.hand_eye_compensation_m
    assert (comp.get("x"), comp.get("y"), comp.get("z")) == (0.005, 0.0, -0.015)
    e = Extrinsics.from_config(cfg.camera.extrinsics)
    assert np.allclose(e.cam_to_base()[:3, 3], [0.005, 0.0, -0.015])


def test_no_demo_yaml_fallback_for_compensation(tmp_path):
    """ADR-0009 v2: a global default would be silently inherited by every
    camera profile added later. Per-camera only."""
    cdir = _config_tree(tmp_path, {"type": "mock", "extrinsics": {
        "mode": "eye_to_hand", "T": np.eye(4).tolist()}},
        demo_extra={"hand_eye_compensation_m": {"x": 0.0, "y": 0.0, "z": -0.01}})
    cfg = load_demo_config(camera="cal_cam", config_dir=cdir)
    assert cfg.camera.extrinsics.get("hand_eye_compensation_m") is None
    assert np.allclose(Extrinsics.from_config(cfg.camera.extrinsics).cam_to_base(), np.eye(4))


# ── the runtime's fusion decision ────────────────────────────────────────


def test_an_uncalibrated_camera_neither_fuses_nor_maps(tmp_path):
    from cascade.apps.demo import _camera_fusion

    bad = Extrinsics.from_config(Cfg({"hand_eye_json": str(tmp_path / "missing.json")}))
    ccfg = Cfg({"type": "mock", "fuse_beliefs": True, "map_depth": True,
                "extrinsics": {"hand_eye_json": "x"}})
    assert _camera_fusion(ccfg, bad) == (False, False)
    good = Extrinsics.from_config(Cfg({"T": np.eye(4).tolist()}))
    assert _camera_fusion(Cfg({"type": "mock", "extrinsics": {}}), good) == (True, None)
    assert _camera_fusion(Cfg({"type": "mock"}), good) == (False, None)
    assert _camera_fusion(ccfg, good) == (True, True)
