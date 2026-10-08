"""The hand-eye CLI: --list/--bind, --dry-run, --verify, real-run orchestration.

Ported from WRC tests/test_calib_session.py (CLI half). Hardware is never
touched: camera SDKs are stubbed in `sys.modules` (the AGENTS.md rule: fake
the contract instead of skipping), the arm is the kinematic MockArm behind
the real SafeArm/SafetyHarness, and the camera renders the marker.
"""

from __future__ import annotations

import json
import signal
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import needs_pin

from cascade.calibration import cli
from cascade.calibration.dataset import load_hand_eye, read_hand_eye


# ── device enumeration / binding ─────────────────────────────────────────


def _fake_realsense(serials):
    info = types.SimpleNamespace(serial_number="sn", name="name", firmware_version="fw",
                                 usb_type_descriptor="usb")

    class Dev:
        def __init__(self, sn):
            self.d = {"sn": sn, "name": "Intel RealSense D455F", "fw": "5.17.0.10", "usb": "3.2"}

        def get_info(self, key):
            return self.d[key]

    class Context:
        def query_devices(self):
            return [Dev(s) for s in serials]

    return types.SimpleNamespace(context=Context, camera_info=info)


def _fake_orbbec(serials):
    class Info:
        def __init__(self, sn):
            self.sn = sn

        def get_name(self):
            return "Orbbec Gemini 2"

        def get_serial_number(self):
            return self.sn

    class DeviceList:
        def get_count(self):
            return len(serials)

        def get_device_by_index(self, i):
            return types.SimpleNamespace(get_device_info=lambda i=i: Info(serials[i]))

    class Context:
        def query_devices(self):
            return DeviceList()

    return types.SimpleNamespace(Context=Context)


@pytest.fixture
def sdks(monkeypatch):
    def install(realsense=None, orbbec=None):
        for name, mod in (("pyrealsense2", realsense), ("pyorbbecsdk", orbbec)):
            if mod is None:
                monkeypatch.setitem(sys.modules, name, None)   # import -> ImportError
            else:
                monkeypatch.setitem(sys.modules, name, mod)
    return install


def test_list_enumerates_realsense_and_orbbec_serials(sdks, capsys):
    from cascade.calibration.devices import list_devices

    sdks(_fake_realsense(["261422303968", "261522301814"]), _fake_orbbec(["AY3794300W4"]))
    devices, notes = list_devices()
    assert [(d.backend, d.serial) for d in devices] == [
        ("realsense", "261422303968"), ("realsense", "261522301814"), ("orbbec", "AY3794300W4")]
    assert notes == []
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "261522301814" in out and "AY3794300W4" in out and "D455F" in out


def test_list_without_any_sdk_says_so(sdks, capsys):
    sdks(None, None)
    assert cli.main(["--list"]) == 1
    out = capsys.readouterr().out
    assert "pyrealsense2" in out and "pyorbbecsdk" in out


def _profile_dir(tmp_path, text):
    cdir = tmp_path / "configs"
    (cdir / "cameras").mkdir(parents=True)
    (cdir / "cameras" / "scene.yaml").write_text(text)
    return cdir


PROFILE = """# The scene unit.
type: realsense
serial: "000000000000"   # pinned by --bind
width: 1280
"""


def test_bind_pins_the_serial_and_keeps_the_comments(sdks, tmp_path):
    sdks(_fake_realsense(["261422303968", "261522301814"]), None)
    cdir = _profile_dir(tmp_path, PROFILE)
    assert cli.main(["--bind", "scene", "--serial", "261522301814",
                     "--config-dir", str(cdir)]) == 0
    text = (cdir / "cameras" / "scene.yaml").read_text()
    assert yaml.safe_load(text)["serial"] == "261522301814"
    assert text.startswith("# The scene unit.") and "width: 1280" in text


def test_bind_by_index_and_insert_when_absent(sdks, tmp_path):
    sdks(_fake_realsense(["261422303968", "261522301814"]), None)
    cdir = _profile_dir(tmp_path, "type: realsense\nwidth: 1280\n")
    assert cli.main(["--bind", "scene", "--index", "1", "--config-dir", str(cdir)]) == 0
    assert yaml.safe_load((cdir / "cameras" / "scene.yaml").read_text())["serial"] == "261522301814"


def test_bind_refuses_a_serial_that_is_not_connected(sdks, tmp_path):
    """Two identical D455Fs: a typo'd serial binds nothing and silently
    picks whichever unit enumerates first at runtime."""
    sdks(_fake_realsense(["261422303968"]), None)
    cdir = _profile_dir(tmp_path, PROFILE)
    assert cli.main(["--bind", "scene", "--serial", "261422303969",
                     "--config-dir", str(cdir)]) == 2
    assert yaml.safe_load((cdir / "cameras" / "scene.yaml").read_text())["serial"] == "000000000000"


def test_bind_unknown_profile_fails(sdks, tmp_path):
    sdks(None, None)
    cdir = _profile_dir(tmp_path, PROFILE)
    assert cli.main(["--bind", "nope", "--serial", "1", "--config-dir", str(cdir)]) == 2


# ── refusals that need no hardware ───────────────────────────────────────


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_script_and_console_entry_point_reach_the_cli():
    """scripts/calibrate_handeye.py runs from a checkout without an install
    (fresh interpreter, no PYTHONPATH); pyproject exposes the same main."""
    import subprocess
    import tomllib

    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(repo / "scripts" / "calibrate_handeye.py"), "--help"],
                       capture_output=True, text=True, timeout=60, cwd=repo.parent, env=env)
    assert r.returncode == 0, r.stderr
    assert "--dry-run" in r.stdout and "safety" in r.stdout
    scripts = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["cascade-calib-handeye"] == "cascade.calibration.cli:main"


def test_gravity_comp_is_refused_not_faked(capsys):
    assert cli.main(["--gravity-comp"]) == 2
    assert "not ported" in capsys.readouterr().err


@pytest.mark.parametrize("frac", ["0", "1.5"])
def test_speed_fraction_above_the_cap_is_refused(frac, capsys):
    assert cli.main(["--dry-run", "--speed-frac", frac]) == 2


# ── --dry-run: the whole pipeline, synthetic ─────────────────────────────


@needs_pin
@pytest.mark.parametrize("camera,mode", [("d455f_scene", "eye_to_hand"),
                                         ("d455f_wrist", "eye_in_hand")])
def test_dry_run_end_to_end(tmp_path, monkeypatch, camera, mode, capsys):
    import cascade.apps.demo as demo

    built = []
    real_make_arm = demo.make_arm
    monkeypatch.setattr(demo, "make_arm", lambda acfg, **kw: built.append(acfg.type)
                        or real_make_arm(acfg, **kw))
    out = tmp_path / "dry.json"
    rc = cli.main(["--dry-run", "--camera", camera, "--arm", "rebot_rs", "--marker-timeout", "0.5",
                   "--dry-run-out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 0, capsys.readouterr()
    assert built == ["mock"]                      # never a hardware backend
    rec = load_hand_eye(out)
    assert rec is not None and rec.mode == mode and rec.camera_serial == "SYNTHETIC"
    assert "DRY RUN" in rec.note
    assert "recovered" in capsys.readouterr().out
    events = [json.loads(x) for x in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "session_start" and events[-1]["event"] == "session_end"
    assert events[1]["event"] == "home"             # the sweep starts from a vetted home
    # The MockArm does not move in real time: nothing to settle, no 1 s waits.
    assert events[0]["settle_s"] == 0.0


def test_real_runs_keep_the_settle_default():
    args = cli.build_parser().parse_args(["--camera", "d455f_scene", "--arm", "rebot_rs"])
    assert cli._session_config(args, "eye_to_hand", dry_run=False).settle_s == 1.0
    assert cli._session_config(args, "eye_to_hand", dry_run=True).settle_s == 0.0
    args = cli.build_parser().parse_args(["--dry-run", "--settle-time", "0.4"])
    assert cli._session_config(args, "eye_to_hand", dry_run=True).settle_s == 0.4


def test_dry_run_refuses_to_clobber_the_real_output(tmp_path):
    same = tmp_path / "calib.json"
    assert cli.main(["--dry-run", "--out", str(same), "--dry-run-out", str(same)]) == 2
    assert not same.exists()


# ── real-run orchestration, on the mock stack ────────────────────────────


@pytest.fixture
def fake_rig(monkeypatch):
    """Swap the hardware builder for the dry-run rig (MockArm + synthetic
    camera). Everything after build_rig is the real-run code path."""
    made = {}

    def build(cfg, args):
        rig = cli.dry_run_rig(args.arm, cfg.camera.extrinsics.mode, noise_px=0.3, seed=0)
        rig = cli.Rig(safe_arm=rig.safe_arm, raw_arm=rig.raw_arm, kin=rig.kin,
                      camera=rig.camera, home_q=rig.home_q, ee_frame=rig.ee_frame,
                      camera_serial=cfg.camera.get("serial", ""))
        made["rig"] = rig
        return rig
    monkeypatch.setattr(cli, "build_rig", build)
    return made


def _answers(monkeypatch, *answers):
    it = iter(answers)
    monkeypatch.setattr(cli, "_prompt", lambda msg="": next(it))


@needs_pin
def test_real_run_requires_operator_confirmation(tmp_path, monkeypatch, fake_rig):
    _answers(monkeypatch, "no")
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs",
                   "--out", str(tmp_path / "out.json"), "--run-dir", str(tmp_path / "run")])
    assert rc == 1
    assert fake_rig["rig"].raw_arm.commands == []         # nothing moved
    assert not (tmp_path / "out.json").exists()


@needs_pin
def test_real_run_saves_an_acceptable_record_then_parks(tmp_path, monkeypatch, fake_rig):
    calls = []
    real_build = cli.build_rig

    def build(cfg, args):
        rig = real_build(cfg, args)
        orig_move = rig.safe_arm.move_joints

        def move_joints(q, *a, **kw):
            calls.append(("move_joints", list(np.round(q, 6)), kw.get("joint_margin")))
            return orig_move(q, *a, **kw)
        rig.safe_arm.move_joints = move_joints
        orig_disc = rig.raw_arm.disconnect
        rig.raw_arm.disconnect = lambda: (calls.append(("disconnect",)), orig_disc())[1]
        return rig
    monkeypatch.setattr(cli, "build_rig", build)
    out = tmp_path / "out.json"
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes",
                   "--settle-time", "0", "--stable-frames", "2",
                   "--out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 0
    rec = read_hand_eye(out)
    assert rec.acceptable and rec.arm == "rebot_rs" and rec.camera == "d455f_scene"
    assert rec.camera_serial == "261422303968"          # the profile's pinned unit
    rig = fake_rig["rig"]
    # Shutdown = the runtime's: park to the profile's park_q through SafeArm
    # (joint margin relaxed for that one move), THEN torque off.
    park = [c for c in calls if c[0] == "move_joints" and c[2] == 0.0]
    assert park and park[-1][1] == list(np.round(rig.safe_arm.harness.park_q, 6))
    assert calls[-1] == ("disconnect",)
    assert rig.raw_arm._connected is False
    assert (tmp_path / "run" / "captures").is_dir()


@needs_pin
def test_ctrl_c_halts_then_parks_then_disconnects(tmp_path, monkeypatch, fake_rig):
    """WRC parked via an atexit hook + SIGTERM handler; cascade's arm CLI
    convention is halt -> park -> disconnect (apps/demo.py), never e-stop
    -> torque-off with the arm aloft."""
    from cascade.apps.signal_stop import SignalRequest

    calls = []
    real_build = cli.build_rig

    def build(cfg, args):
        rig = real_build(cfg, args)
        h = rig.safe_arm.harness
        orig_halt = h.halt
        h.halt = lambda reason: (calls.append(("halt", reason)), orig_halt(reason))[1]
        orig_move = rig.safe_arm.move_joints

        def move_joints(q, *a, **kw):
            calls.append(("move_joints", kw.get("joint_margin")))
            return orig_move(q, *a, **kw)
        rig.safe_arm.move_joints = move_joints
        orig_disc = rig.raw_arm.disconnect
        rig.raw_arm.disconnect = lambda: (calls.append(("disconnect",)), orig_disc())[1]
        cam = rig.camera
        orig_grab = cam.get_frame
        fired = []

        def grab():
            # One Ctrl+C mid-sweep. (While cleanup runs, StopSignals defers
            # a real signal, so the fake must not re-fire on cleanup's grab.)
            if cam.grabs == 12 and not fired:
                fired.append(True)
                raise SignalRequest(signal.SIGINT)
            return orig_grab()
        cam.get_frame = grab
        return rig
    monkeypatch.setattr(cli, "build_rig", build)
    out = tmp_path / "out.json"
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes",
                   "--settle-time", "0", "--stable-frames", "2",
                   "--out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 130
    assert not out.exists()
    names = [c[0] for c in calls]
    i_halt = names.index("halt")
    i_park = max(i for i, c in enumerate(calls) if c == ("move_joints", 0.0))
    i_disc = names.index("disconnect")
    assert i_halt < i_park < i_disc
    assert "interrupt" in calls[i_halt][1]


@needs_pin
def test_too_few_samples_saves_nothing(tmp_path, monkeypatch, fake_rig):
    poses = tmp_path / "p.yaml"
    poses.write_text(yaml.safe_dump([[0.30, -0.05, 0.28, 0.0, 0.0, 0.0]]))
    out = tmp_path / "out.json"
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes", "--poses", str(poses),
                   "--settle-time", "0", "--out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 1 and not out.exists()


@needs_pin
def test_a_rejected_fit_is_kept_for_diagnosis_but_not_written_to_out(tmp_path, monkeypatch,
                                                                     fake_rig):
    poses = tmp_path / "p.yaml"
    poses.write_text(yaml.safe_dump(
        [[0.30, -0.05, 0.28, 0.0, 0.0, float(y)] for y in np.linspace(-0.6, 0.6, 12)]))
    out = tmp_path / "out.json"
    rc = cli.main(["--camera", "d455f_scene", "--arm", "rebot_rs", "--yes", "--poses", str(poses),
                   "--settle-time", "0", "--stable-frames", "2",
                   "--out", str(out), "--run-dir", str(tmp_path / "run")])
    assert rc == 1 and not out.exists()
    kept = list((tmp_path / "run").glob("*.REJECTED.json"))
    assert len(kept) == 1 and not read_hand_eye(kept[0]).acceptable


# ── --verify against operator-measured points ────────────────────────────


def test_verify_known_points_math():
    rep = cli.verify_known_points([[0.30, 0.0, 0.02], [0.25, -0.05, 0.02]],
                                  [[0.303, 0.004, 0.02], [0.25, -0.05, 0.02]])
    assert rep.rmse_m == pytest.approx(np.sqrt((0.005**2 + 0.0) / 2))
    assert rep.max_m == pytest.approx(0.005) and rep.passed
    bad = cli.verify_known_points([[0.3, 0, 0]], [[0.3, 0.02, 0]])
    assert not bad.passed and bad.rmse_m > cli.VERIFY_MAX_RMSE_M


class PlacedMarkerCamera:
    """Top camera seeing a marker the operator places flat on the table."""

    def __init__(self, T_cam2base, positions, offset=(0.0, 0.0, 0.0)):
        from cascade.calibration.synthetic import SyntheticMarkerCamera

        self.positions, self.k = positions, 0
        self.offset = np.asarray(offset, dtype=float)
        # eye_in_hand with the TCP pinned at identity: cam2gripper == cam2base.
        self._cam = SyntheticMarkerCamera("eye_in_hand", T_cam2base, np.eye(4),
                                          lambda: np.eye(4))
        self.dist_coeffs = np.zeros(5)
        self.serial = "SYNTHETIC"

    def place_next(self):
        p = np.asarray(self.positions[self.k], dtype=float) + self.offset
        self.k += 1
        Z = np.eye(4)
        Z[:3, 3] = p
        self._cam.T_marker = Z

    def get_frame(self):
        return self._cam.get_frame()

    def open(self):
        pass

    def close(self):
        pass


@needs_pin
@pytest.mark.parametrize("offset,rc_expected", [((0.0, 0.0, 0.0), 0), ((0.0, 0.02, 0.0), 1)])
def test_verify_against_known_points(tmp_path, monkeypatch, capsys, offset, rc_expected):
    """ETH: the operator lays the marker at measured base-frame points; the
    calibration must put it there within 10 mm RMSE (Seeed wiki §5.3). A
    2 cm error in where the marker really was fails."""
    rig = cli.dry_run_rig("rebot_rs", "eye_to_hand", noise_px=0.2, seed=0)
    from cascade.calibration.dataset import MarkerSpec, save_hand_eye
    from cascade.calibration.handeye import HandEyeSample
    from cascade.calibration.frames import se3_inv

    # A clean record whose T_cam2base is the synthetic truth.
    rng = np.random.default_rng(0)
    from cascade.calibration.frames import so3_exp
    samples = []
    for _ in range(12):
        G = np.eye(4)
        G[:3, :3] = so3_exp(rng.normal(scale=0.4, size=3))
        G[:3, 3] = rng.uniform([0.25, -0.1, 0.2], [0.35, 0.05, 0.35])
        samples.append(HandEyeSample(G, se3_inv(rig.T_hand_eye) @ G @ rig.T_marker))
    rec = cli.solve_and_record(samples, mode="eye_to_hand", marker=MarkerSpec(),
                               camera="d455f_scene", camera_serial="261422303968")
    path = save_hand_eye(tmp_path / "rec.json", rec)

    known = [(0.30, 0.00, 0.0), (0.25, -0.05, 0.0), (0.35, 0.05, 0.0)]
    cam = PlacedMarkerCamera(rig.T_hand_eye, known, offset=offset)
    monkeypatch.setattr(cli, "open_camera", lambda cfg: cam)

    def prompt(msg=""):
        cam.place_next()
        return ""
    monkeypatch.setattr(cli, "_prompt", prompt)
    pts = ";".join(",".join(str(v) for v in p) for p in known)
    rc = cli.main(["--verify", str(path), "--camera", "d455f_scene", "--known-points", pts,
                   "--settle-time", "0", "--stable-frames", "2"])
    out = capsys.readouterr().out
    assert rc == rc_expected, out
    assert "RMSE" in out


def test_verify_refuses_a_rejected_record(tmp_path, capsys):
    from cascade.calibration.dataset import MarkerSpec, save_hand_eye
    from cascade.calibration.handeye import HandEyeSample
    from cascade.calibration.frames import so3_exp

    samples = []
    for i in range(10):
        G = np.eye(4)
        G[:3, :3] = so3_exp([0, 0, 0.1 * i])
        G[:3, 3] = [0.3, 0.0, 0.3]
        samples.append(HandEyeSample(G, G))
    rec = cli.solve_and_record(samples, mode="eye_to_hand", marker=MarkerSpec())
    path = save_hand_eye(tmp_path / "bad.json", rec)
    rc = cli.main(["--verify", str(path), "--camera", "d455f_scene",
                   "--known-points", "0.3,0,0"])
    assert rc == 1
    assert "rotation spread" in capsys.readouterr().out
