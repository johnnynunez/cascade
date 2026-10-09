"""Markerless calibration through the REAL collection path and the CLI.

The sweep is the marker method's: presets IK'd from home_q, vetted at plan
time AND right before each move, moved through SafeArm on the arm profile's
own harness (here over the kinematic MockArm), FK of the MEASURED joints --
only the capture differs (settle, then a temporal median of depth frames
instead of a marker). Shutdown is the same halt -> park -> torque off.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from conftest import needs_pin

from cascade.calibration import cli
from cascade.calibration.dataset import load_hand_eye, read_hand_eye
from cascade.calibration.frames import pose_error, se3_inv, so3_exp

pytestmark = needs_pin


def _err(T_est, T_ref):
    e = pose_error(se3_inv(T_ref) @ T_est)
    return 1000 * float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(e[3:])))


# ── the session ──────────────────────────────────────────────────────────


def _depth_session(rig, tmp_path, **cfg):
    from cascade.calibration.dataset import MarkerSpec
    from cascade.calibration.session import DepthCollectionSession, SessionConfig

    config = SessionConfig(mode="eye_to_hand", marker=MarkerSpec(), settle_s=0.0,
                           marker_timeout_s=1.0, min_move_s=1.0, **cfg)
    return DepthCollectionSession(
        safe_arm=rig.safe_arm, kin=rig.kin, camera=rig.camera, config=config,
        home_q=rig.home_q, trace_path=tmp_path / "trace.jsonl",
        capture_dir=tmp_path / "captures", log=lambda *_: None, sleep=lambda _s: None)


def test_depth_session_captures_settled_depth_at_measured_joints(tmp_path):
    from cascade.calibration.markerless import DepthSample
    from cascade.calibration.session import load_poses

    rig = cli.dry_run_rig("rebot_rs", "eye_to_hand", method="markerless", seed=0)
    vetted = []
    orig = rig.safe_arm.harness.vet_pose
    rig.safe_arm.harness.vet_pose = lambda q, *a, **k: (vetted.append(np.round(q, 6).tolist()),
                                                         orig(q, *a, **k))[1]
    session = _depth_session(rig, tmp_path, depth_frames=3)
    grabs0 = rig.camera.grabs
    samples = session.run_auto(load_poses("rebot_rs", "eye_to_hand")[:4], start_home=True)
    assert len(samples) == 4 and all(isinstance(s, DepthSample) for s in samples)
    assert rig.raw_arm.commands                      # moved, through SafeArm
    for s in samples:
        q = np.asarray(s.q)
        assert np.allclose(s.T_gripper2base, rig.kin.fk(q))
        assert s.depth_m.shape == (360, 640) and s.depth_m.dtype == np.float32
        assert np.mean(s.depth_m > 0) > 0.5
        # Vetted when planned AND again immediately before the move.
        assert vetted.count(np.round(q, 6).tolist()) >= 2
    assert rig.camera.grabs - grabs0 >= 4 * 3        # a temporal median per pose
    # (greedy joint ordering: visit order != preset order)
    assert sorted(p.name for p in (tmp_path / "captures").iterdir()) == \
        sorted(f"{s.label}_depth.png" for s in samples)
    assert session.depth_files == [f"captures/{s.label}_depth.png" for s in samples]
    events = [json.loads(x)["event"] for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert events.count("sample_recorded") == 4 and events[0] == "session_start"


def test_a_frame_without_sensor_depth_skips_the_pose(tmp_path):
    from cascade.calibration.session import load_poses

    rig = cli.dry_run_rig("rebot_rs", "eye_to_hand", method="markerless", seed=0)
    real = rig.camera.get_frame

    def rgb_only():
        f = real()
        f.depth_m, f.depth_source = None, "none"
        return f
    rig.camera.get_frame = rgb_only
    session = _depth_session(rig, tmp_path, depth_frames=2)
    assert session.run_auto(load_poses("rebot_rs", "eye_to_hand")[:2]) == []
    rows = [json.loads(x) for x in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert any(r.get("reason") == "no_depth" for r in rows)


def test_the_depth_session_is_eye_to_hand_only():
    from cascade.calibration.dataset import MarkerSpec
    from cascade.calibration.session import DepthCollectionSession, SessionConfig

    with pytest.raises(ValueError, match="eye_to_hand"):
        DepthCollectionSession(safe_arm=None, kin=None, camera=None,
                               config=SessionConfig(mode="eye_in_hand", marker=MarkerSpec()),
                               home_q=np.zeros(6))


# ── CLI refusals ─────────────────────────────────────────────────────────


def test_markerless_refuses_an_eye_in_hand_camera(capsys):
    rc = cli.main(["--dry-run", "--method", "markerless", "--camera", "d455f_wrist"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "eye_to_hand" in err and "does not see the arm" in err


def test_markerless_refuses_a_camera_without_sensor_depth(capsys):
    rc = cli.main(["--dry-run", "--method", "markerless", "--camera", "uvc"])
    assert rc == 2
    assert "depth" in capsys.readouterr().err


def test_initial_guess_comes_from_the_profile():
    from cascade.config import Cfg

    T = cli.markerless_initial_guess(Cfg({"extrinsics": {"mode": "eye_to_hand", "T": [
        [0.0, -1.0, 0.0, 0.28], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.6], [0, 0, 0, 1]]}}))
    assert T is not None and T[2, 3] == pytest.approx(0.6)
    # No guess at all is refused rather than started from the base origin.
    assert cli.markerless_initial_guess(Cfg({"extrinsics": {"mode": "eye_to_hand"}})) is None


# ── --dry-run --method markerless ────────────────────────────────────────


def test_dry_run_markerless_end_to_end(tmp_path, monkeypatch, capsys):
    from cascade.apps import demo

    built = []
    real_make_arm = demo.make_arm
    monkeypatch.setattr(demo, "make_arm", lambda acfg, **kw: built.append(acfg.type)
                        or real_make_arm(acfg, **kw))
    out = tmp_path / "dry.json"
    rc = cli.main(["--dry-run", "--method", "markerless", "--dry-run-out", str(out),
                   "--run-dir", str(tmp_path / "run"), "--depth-frames", "2"])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert built == ["mock"]                          # never a hardware backend
    rec = load_hand_eye(out)
    assert rec is not None and rec.markerless and rec.camera_serial == "SYNTHETIC"
    assert "DRY RUN" in rec.note
    assert rec.metrics["n_poses"] >= 30
    assert "recovered" in text and "initial guess" in text
    rows = [json.loads(x) for x in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert rows[1]["event"] == "home"
    assert len(list((tmp_path / "run" / "captures").glob("*_depth.png"))) == rec.metrics["n_poses"]


# ── real-run orchestration on the mock stack ─────────────────────────────


@pytest.fixture
def depth_rig(monkeypatch):
    """build_rig -> the profile's harness on a MockArm + a synthetic depth
    camera ~10 cm / 5 deg away from the profile's placeholder T."""
    made = {}

    def build(cfg, args):
        from cascade.calibration.synthetic_depth import SyntheticDepthCamera

        dr = cli.dry_run_rig(args.arm, "eye_to_hand", method="markerless")
        T0 = np.asarray(cfg.camera.extrinsics.T, dtype=float)
        truth = T0.copy()
        truth[:3, :3] = T0[:3, :3] @ so3_exp(np.radians([3.0, -4.0, 2.0]))
        truth[:3, 3] += [0.03, -0.03, 0.09]
        cam = SyntheticDepthCamera(dr.kin, dr.model_path, truth,
                                   lambda: dr.safe_arm.get_state().q, seed=3)
        rig = cli.Rig(safe_arm=dr.safe_arm, raw_arm=dr.raw_arm, kin=dr.kin, camera=cam,
                      home_q=dr.home_q, ee_frame=dr.ee_frame, model_path=dr.model_path,
                      camera_serial=cfg.camera.get("serial", ""))
        made.update(rig=rig, truth=truth)
        return rig
    monkeypatch.setattr(cli, "build_rig", build)
    return made


def _poses_file(tmp_path, every=3):
    from cascade.calibration.session import load_poses

    p = tmp_path / "poses.yaml"
    p.write_text(yaml.safe_dump(load_poses("rebot_rs", "eye_to_hand")[::every]))
    return p


def test_real_run_markerless_saves_an_acceptable_record_then_parks(tmp_path, depth_rig):
    calls = []
    real_build = cli.build_rig

    def build(cfg, args):
        rig = real_build(cfg, args)
        orig_move = rig.safe_arm.move_joints

        def move_joints(q, *a, **kw):
            calls.append(("move_joints", kw.get("joint_margin")))
            return orig_move(q, *a, **kw)
        rig.safe_arm.move_joints = move_joints
        orig_disc = rig.raw_arm.disconnect
        rig.raw_arm.disconnect = lambda: (calls.append(("disconnect",)), orig_disc())[1]
        return rig
    cli.build_rig = build
    try:
        out = tmp_path / "out.json"
        rc = cli.main(["--method", "markerless", "--camera", "d455f_scene", "--arm", "rebot_rs",
                       "--yes", "--settle-time", "0", "--depth-frames", "2",
                       "--poses", str(_poses_file(tmp_path)),
                       "--out", str(out), "--run-dir", str(tmp_path / "run")])
    finally:
        cli.build_rig = real_build
    assert rc == 0
    rec = read_hand_eye(out)
    assert rec.acceptable and rec.markerless and rec.camera_serial == "261422303968"
    mm, deg = _err(rec.T_cam2base, depth_rig["truth"])
    assert mm < 5.0 and deg < 0.5, (mm, deg)
    park = [i for i, c in enumerate(calls) if c == ("move_joints", 0.0)]
    assert park and park[-1] < calls.index(("disconnect",))
    assert (tmp_path / "run" / "d455f_scene.handeye.json").exists()


def test_a_rejected_markerless_fit_is_kept_for_diagnosis_but_not_written(tmp_path, depth_rig):
    poses = tmp_path / "p.yaml"
    poses.write_text(yaml.safe_dump(
        [[0.30, float(y), 0.28, 0.0, 0.0, 0.0] for y in np.linspace(-0.10, 0.05, 8)]))
    out = tmp_path / "out.json"
    rc = cli.main(["--method", "markerless", "--camera", "d455f_scene", "--arm", "rebot_rs",
                   "--yes", "--settle-time", "0", "--depth-frames", "1", "--poses", str(poses),
                   "--out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 1 and not out.exists()
    kept = list((tmp_path / "run").glob("*.REJECTED.json"))
    assert len(kept) == 1 and not read_hand_eye(kept[0]).acceptable


def test_a_camera_without_depth_is_refused_before_anything_moves(tmp_path, depth_rig,
                                                                  monkeypatch, capsys):
    real_build = cli.build_rig

    def build(cfg, args):
        rig = real_build(cfg, args)
        rig.camera.has_depth = False
        return rig
    monkeypatch.setattr(cli, "build_rig", build)
    rc = cli.main(["--method", "markerless", "--camera", "d455f_scene", "--arm", "rebot_rs",
                   "--yes", "--out", str(tmp_path / "o.json"), "--run-dir", str(tmp_path / "r")])
    assert rc == 2
    assert depth_rig["rig"].raw_arm.commands == []
    assert "depth" in capsys.readouterr().err


def test_verify_works_for_a_markerless_record(tmp_path, monkeypatch, capsys):
    """--verify is method-independent: lay a marker (described on the command
    line, since the record has none) at measured base points."""
    from test_handeye_cli import PlacedMarkerCamera
    from test_markerless_record import GOOD

    from cascade.calibration.dataset import save_hand_eye
    from cascade.calibration.markerless import MarkerlessFit, record_from_markerless

    truth = cli._dry_run_truth()["eye_to_hand"][0]
    rec = record_from_markerless(MarkerlessFit(T_cam2base=truth, metrics=dict(GOOD)), [],
                                 camera="d455f_scene", camera_serial="261422303968")
    path = save_hand_eye(tmp_path / "rec.json", rec)
    known = [(0.30, 0.00, 0.0), (0.25, -0.05, 0.0), (0.35, 0.05, 0.0)]
    cam = PlacedMarkerCamera(truth, known)
    monkeypatch.setattr(cli, "open_camera", lambda cfg: cam)

    def prompt(msg=""):
        cam.place_next()
        return ""
    monkeypatch.setattr(cli, "_prompt", prompt)
    pts = ";".join(",".join(str(v) for v in p) for p in known)
    rc = cli.main(["--verify", str(path), "--camera", "d455f_scene", "--known-points", pts,
                   "--settle-time", "0", "--stable-frames", "2", "--marker-size", "0.10"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "markerless" in out and "PASS" in out
