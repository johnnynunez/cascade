"""Control/causality fixtures; synthetic joints and contacts are not native proof."""
import builtins
from dataclasses import replace
import hashlib
import io
import runpy
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from mobile_tick_fixture import healthy_episode_gc as healthy_episode_gc

from cascade.apps.hand_runtime import hand_description, validate_hand_profile
from cascade.apps.robot_runtime import build_robot_runtime, robot_tool_descriptors
from cascade.config import load_robot_config
from cascade.control.hand import HandContact, HandController, HandFault, HandLimits, HandSample, check_hand_sample
from cascade.robotics.runtime import RobotRuntime
from cascade.sim.leap_hand import JOINTS, LeapHandBackend, RECIPE
from cascade.skills.hand_runtime import HandDomain


class SyntheticBackend:
    synthetic = True
    joint_names = JOINTS
    geom_names = ("finger", "object")
    model_sha256, epoch = "a"*64, "synthetic-hand"
    initial_targets = (0.,)*16

    def __init__(self):
        self.limits = HandLimits((-1.,)*16, (1.,)*16)
        self.step, self.targets, self.q = 0, self.initial_targets, self.initial_targets
        self.upload_hook = self.solve_hook = None
        self.uploads = []
        self.fault = None

    def upload(self, targets):
        if self.upload_hook:
            self.upload_hook()
        self.targets = tuple(targets)
        self.uploads.append((threading.get_ident(), self.targets))

    def advance(self, generation, clock):
        if self.solve_hook:
            self.solve_hook()
        old = self.q
        self.q = self.targets
        self.step += 1
        dt = self.limits.dt_s
        sample = HandSample(self.model_sha256, self.epoch, generation, self.step, self.step*dt,
            (self.step-1)*dt, clock(), self.q, tuple((a-b)/dt for a, b in zip(self.q, old)),
            (0.,)*16, self.targets, ())
        if self.fault:
            sample = self.fault(sample)
        return sample


def profile():
    return {"kind": "hand", "recipe": RECIPE, "asset_root": "/unprepared/leap_hand", "model_identity_sha256": None}


def start():
    backend = SyntheticBackend()
    controller = HandController(backend)
    controller.start()
    deadline = time.monotonic()+2
    while not controller.read():
        assert time.monotonic() < deadline
    assert controller.reset_stop()["ok"]
    resources, _ = hand_description("hand", profile())
    resources = tuple(replace(r, synthetic=True, admission="software_only") for r in resources)
    return backend, controller, HandDomain(controller, resources)


def test_native_profile_discovery_and_unprepared_refusal_never_import_sdk(monkeypatch, tmp_path):
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "newton", "warp", "isaacsim"}:
            pytest.fail("passive hand discovery opened SDK")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    cfg = load_robot_config("leap_hand_right_mujoco")
    tools = robot_tool_descriptors(cfg)
    assert tools["hand.move_fingers"].effect == "motion"
    assert tools["hand.move_fingers"].writes == ("hand/fingers",)
    assert tools["hand.get_hand_state"].effect == "read" and not tools["hand.get_hand_state"].writes
    with pytest.raises(HandFault, match="unprepared"):
        build_robot_runtime(cfg, tmp_path)


@pytest.mark.parametrize("change", [{"recipe": "grasp"}, {"kind": "gripper"}, {"robot_id": "other"},
    {"model_identity_sha256": "anything"}, {"asset_root": "relative"}, {"torque_limit": 10.}])
def test_profile_cannot_invent_capabilities_or_override_fixed_recipe(change):
    with pytest.raises(ValueError):
        validate_hand_profile(profile() | change)


def test_bad_asset_is_refused_before_native_import_or_model_construction(tmp_path, monkeypatch):
    (tmp_path/"LICENSE").write_text("wrong source")
    original = builtins.__import__
    def checked(name, *args, **kwargs):
        if name == "mujoco":
            pytest.fail("invalid assets reached model construction")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)
    with pytest.raises(HandFault, match="source asset differs"):
        LeapHandBackend(tmp_path)


def test_ordinary_runtime_requires_retained_rest_and_never_credits_synthetic_task(healthy_episode_gc):
    backend, controller, domain = start()
    runtime = RobotRuntime({"hand": domain})
    try:
        targets = [.03]+[0.]*15
        result = runtime.execute("hand.move_fingers", {"positions_rad": targets})
        assert result["ok"] and result["verified"] and result["synthetic"], result
        assert result["rest"]["window_sim_s"] >= .2-1e-9
        assert not result["physical_stop_verified"] and not result["rest"]["motor_power_off"]
        assert not runtime.execute("task_done", {"success": True, "summary": "synthetic fingers"})["success"]
        observed = runtime.execute("hand.get_hand_state", {})
        assert observed["ok"] and observed["tactile_calibration"] is None
        assert observed["sample"]["position_rad"][0] == .03
        assert len({ident for ident, _ in backend.uploads}) == 1
    finally:
        assert runtime.close()["ok"]
    assert not controller._thread.is_alive()


def test_stop_waits_only_for_existing_upload_and_revokes_next_target(healthy_episode_gc):
    backend, controller, _ = start()
    entered, release = threading.Event(), threading.Event()
    returned, ack = threading.Event(), []
    def held_upload():
        backend.upload_hook = None
        entered.set()
        assert release.wait(2)
    backend.upload_hook = held_upload
    worker = threading.Thread(target=lambda: (ack.append(controller.stop()), returned.set()))
    try:
        assert entered.wait(2)
        worker.start()
        assert not returned.is_set()
        release.set(); worker.join(2)
        assert returned.is_set() and ack[0]["ok"]
        with pytest.raises(HandFault, match="stopped"):
            controller.admit([.03]+[0.]*15)
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        assert controller.close()["ok"]


def test_pre_reset_solve_in_flight_finishes_before_successful_reset_ack(healthy_episode_gc):
    backend, controller, _ = start()
    controller.stop()
    entered, release = threading.Event(), threading.Event()
    result = []
    def held_solve():
        backend.solve_hook = None
        entered.set()
        assert release.wait(2)
    backend.solve_hook = held_solve
    worker = threading.Thread(target=lambda: result.append(controller.reset_stop()))
    try:
        assert entered.wait(2)
        worker.start()
        # The old solve cannot be relabelled as completed reset generation.
        assert not result
        release.set(); worker.join(2)
        assert result and result[0]["ok"], result
        assert controller.read()[-1].generation == result[0]["generation"]
        assert controller.admit([.03]+[0.]*15)["generation"] == result[0]["generation"]+1
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(2)
        assert controller.close()["ok"]


def sample(backend, **changes):
    row = HandSample(backend.model_sha256, backend.epoch, 0, 1, .002, 0., 10.,
        (0.,)*16, (0.,)*16, (0.,)*16, (0.,)*16, ())
    return replace(row, **changes)


@pytest.mark.parametrize("change,reason", [
    ({"epoch": "other"}, "identity"), ({"model_sha256": "b"*64}, "identity"),
    ({"captured_monotonic_s": 9.7}, "stale"), ({"captured_monotonic_s": 10.1}, "future"),
    ({"constraint_time_s": .002}, "clocks"), ({"simulation_time_s": .004}, "clocks"),
    ({"position_rad": (.99,)+(0.,)*15}, "margin"),
    ({"commanded_position_rad": (.99,)+(0.,)*15}, "target margin"),
    ({"velocity_rad_s": (2.1,)+(0.,)*15}, "speed"),
    ({"actuator_effort_nm": (.501,)+(0.,)*15}, "effort"),
])
def test_same_solve_observation_refuses_clock_identity_and_safety_faults(change, reason):
    backend = SyntheticBackend()
    with pytest.raises(HandFault, match=reason):
        check_hand_sample(sample(backend, **change), backend, 10.)


def test_enabled_loaded_contact_veto_is_sticky_and_retained_in_archive(healthy_episode_gc):
    backend, controller, _ = start()
    contact = HandContact(0, "finger", "object", 0, (0., 0., 0.), (0., 0., 1.), (0., 0., 1.), 1.)
    backend.fault = lambda row: replace(row, contacts=(contact,))
    try:
        assert controller._shutdown.wait(2)
        with pytest.raises(HandFault, match="contact"):
            controller.read()
        assert not controller.reset_stop()["ok"]
    finally:
        result = controller.close()
    assert not result["ok"] and result["owner_thread_closed"]
    assert controller.records()[-1].contacts == (contact,)


@pytest.mark.parametrize("fault", ["stop", "owner_fault", "deadline"])
def test_terminal_observation_cannot_override_live_revocation_or_deadline(fault, healthy_episode_gc):
    _, controller, _ = start()
    generation, deadline = controller.generation, time.monotonic()+1.
    if fault == "stop":
        controller.stop()
    elif fault == "owner_fault":
        with controller._condition:
            controller._error = "injected owner fault after batch capture"
    else:
        deadline = time.monotonic()
    try:
        with pytest.raises(HandFault, match="revoked or expired"):
            controller.confirm_generation(generation, deadline)
    finally:
        controller.close()


@pytest.mark.parametrize("fault,reason", [
    ("target_loss", "lost"), ("hold_change", "lost"),
    ("generation", "generation changed"), ("skip", "gap"),
    ("late", "deadline"),
])
def test_rest_cannot_credit_a_good_prefix_after_terminal_failure(fault, reason):
    backend = SyntheticBackend()
    previous = sample(backend, generation=1)
    rows = [replace(previous, generation=2, step=step, simulation_time_s=step*.002,
        constraint_time_s=(step-1)*.002, captured_monotonic_s=10.1) for step in range(2, 103)]
    if fault == "target_loss":
        rows[-1] = replace(rows[-1], position_rad=(.03,)+(0.,)*15)
    elif fault == "hold_change":
        rows[-1] = replace(rows[-1], commanded_position_rad=(.001,)+(0.,)*15)
    elif fault == "generation":
        rows[-1] = replace(rows[-1], generation=3)
    elif fault == "skip":
        rows.pop(-2)
    else:
        rows = [replace(row, captured_monotonic_s=13.) for row in rows]
    confirmations = []
    controller = SimpleNamespace(backend=backend, limits=backend.limits,
        clock=lambda: 10.1, read=lambda cursor: tuple(rows),
        confirm_generation=lambda *args: confirmations.append(args))
    if fault == "late":
        def late_read(cursor):
            controller.clock = lambda: 13.
            return tuple(rows)
        controller.read = late_read
    resources, _ = hand_description("hand", profile())
    domain = HandDomain(controller, resources)
    with pytest.raises(HandFault, match=reason):
        domain._rest({"generation": 1}, {"generation": 2, "accepted_monotonic_s": 10.,
            "targets_rad": (0.,)*16}, previous, (0.,)*16)
    assert not confirmations


def test_broken_exception_formatting_cannot_skip_attempted_stop():
    class Unreadable(Exception):
        def __str__(self):
            raise ValueError("broken error formatter")
    stops = []
    def fail_admit(targets):
        raise Unreadable()
    def fail_stop():
        stops.append(True)
        raise RuntimeError("stop acknowledgement failed")
    resources, _ = hand_description("hand", profile())
    domain = HandDomain(SimpleNamespace(backend=SyntheticBackend(), admit=fail_admit, stop=fail_stop), resources)
    result = domain.execute("move_fingers", {"positions_rad": [0.]*16})
    assert stops == [True]
    assert not result["ok"] and not result["verified"] and not result["stop"]["ok"]
    assert "unreadable exception" in result["error"]


def test_failed_start_preserves_original_error_and_attempts_closure(tmp_path, monkeypatch):
    from cascade.apps import hand_runtime as app
    backend = SyntheticBackend()
    backend.synthetic, backend.document = False, {}
    closed = []
    class FailedOwner:
        def __init__(self, backend):
            pass
        def start(self):
            raise HandFault("original start failure")
        def close(self):
            closed.append(True)
            raise RuntimeError("secondary close failure")
    monkeypatch.setattr(app, "prepare_hand_model", lambda profile: backend)
    monkeypatch.setattr(app, "HandController", FailedOwner)
    with pytest.raises(HandFault, match="original start failure"):
        app.build_hand_runtime(profile() | {"model_identity_sha256": backend.model_sha256},
            tmp_path, domain_id="hand")
    assert closed == [True]
    assert '"error_type": "RuntimeError"' in (tmp_path/"startup-failure.json").read_text()


def test_asset_fetch_checks_existing_and_downloaded_content_before_promotion(tmp_path, monkeypatch):
    fetcher = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/fetch_robot_assets.py"))
    target = tmp_path/"mesh.obj"
    expected = hashlib.sha256(b"pinned mesh").hexdigest()
    target.write_bytes(b"corrupt cache")
    with pytest.raises(SystemExit, match="existing asset differs"):
        fetcher["fetch"]("https://example.invalid/unused", target, expected_sha256=expected)
    response = io.BytesIO(b"wrong download")
    response.headers = {}
    monkeypatch.setattr(fetcher["urllib"].request, "urlopen", lambda *a, **k: response)
    with pytest.raises(SystemExit, match="download differs"):
        fetcher["fetch"]("https://example.invalid/unused", target, force=True, expected_sha256=expected)
    assert target.read_bytes() == b"corrupt cache"
    assert not target.with_suffix(".obj.part").exists()
