"""G1 three-finger driver fastening: configuration route, turn program, grasp and binding (CPU, no SDK).

Physics evidence (Newton, SDF thread contact) is produced by
scripts/validate_humanoid_fastening.py in a Newton environment; nothing here
simulates or claims a physical turn.
"""
import builtins
from dataclasses import fields
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.apps.factory_runtime import validate_factory_profile
from cascade.apps.humanoid_fastening_runtime import (
    CONTROLLER_ID, RECIPE, ROBOT_ID, build_humanoid_fastening_runtime, prepare_humanoid_model,
    validate_humanoid_profile,
)
from cascade.apps.robot_runtime import build_robot_runtime, describe_robot, robot_tool_descriptors
from cascade.config import load_robot_config
from cascade.control.fastening import FasteningFault, FasteningLimits
from cascade.sim.humanoid_fastening import (
    MIN_CONTACT_FRACTION, MIN_FINGER_SHARE, grip_row, humanoid_binding, humanoid_limits, stroke_grip_stats,
)
from cascade.sim.newton_humanoid_screw import (
    CLOSING_SIGN, GRASP, HAND_COLLIDERS, HELD_POSE, IK_JOINTS, RIGHT_FINGERS, ROLL_JOINT, SIMPLIFICATIONS,
    WristDriverProgram, grip_target, hand_rotation, quat_wxyz, rot_xyzw,
)
from cascade.sim.threading_verification import ThreadContract

PROFILE = "g1_wrist_driver_fastening"


def no_sdk(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.split(".")[0] in {"warp", "newton", "mujoco", "torch", "isaacsim", "trimesh"}:
            pytest.fail("passive configuration tried to import native SDK: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", checked)


def shipped():
    return load_robot_config(PROFILE).domains.fastening.as_dict()


def test_shipped_profile_is_discoverable_on_the_ordinary_route_without_sdk(monkeypatch):
    no_sdk(monkeypatch)
    cfg = load_robot_config(PROFILE)
    profile = cfg.domains.fastening.as_dict()
    assert profile["recipe"] == RECIPE and profile["device"] is None and profile["model_identity_sha256"] is None
    assert profile["gc_policy"] == "freeze-startup-heap"
    tools = robot_tool_descriptors(cfg)
    tool = tools["fastening.turn_screw"]
    assert tool.effect == "motion" and tool.writes == ("fastening/mounted_arm_spindle",)
    assert tool.parameters["properties"]["turns"]["const"] == 1.
    resource = describe_robot(cfg)["fastening"].resources[0]
    assert resource.robot_id == ROBOT_ID and resource.controller_id == CONTROLLER_ID
    assert not resource.synthetic and resource.admission == "unvalidated"
    assert resource.metadata["turning_joint"] == "right_wrist_roll_joint"
    assert resource.metadata["pelvis"] == "fixed" and resource.metadata["grasp"] == GRASP == "three_finger_contact"
    assert resource.metadata["grasp_fingers"] == ("thumb", "index", "middle")


def test_unpinned_build_refuses_before_sdk_or_outputs(monkeypatch, tmp_path):
    no_sdk(monkeypatch)
    with pytest.raises(FasteningFault, match="unprepared"):
        build_robot_runtime(load_robot_config(PROFILE), tmp_path)
    assert not list(tmp_path.rglob("model.json")) and not list(tmp_path.rglob("solves.jsonl"))
    with pytest.raises(FasteningFault, match="unprepared"):
        build_humanoid_fastening_runtime(shipped() | {"model_identity_sha256": "a" * 64}, tmp_path / "x",
                                         domain_id="fastening")
    with pytest.raises(FasteningFault, match="unprepared"):
        prepare_humanoid_model(shipped(), tmp_path / "cache")
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("change,reason", [
    ({"seating": "automatic"}, "explicit wrist-driver profile fields"),
    ({"robot_id": "so101_factory_m20"}, "G1 wrist-driver"),
    ({"assets": "assets/factory/nut_bolt"}, "absolute paths"),
    ({"device": "auto"}, "explicit CUDA ordinal"),
    ({"device": "cpu"}, "explicit CUDA ordinal"),
    ({"model_identity_sha256": "abc"}, "exact SHA256"),
    ({"gc_policy": "disable"}, "host GC policy"),
])
def test_profile_validation_fails_closed(change, reason):
    with pytest.raises(ValueError, match=reason):
        validate_humanoid_profile(shipped() | change)
    with pytest.raises(ValueError, match=reason):
        validate_factory_profile(shipped() | change)   # the ordinary dispatch reaches the same rules


def test_factory_profiles_keep_their_own_rules():
    factory = load_robot_config("factory_m20_mounted").domains.fastening.as_dict()
    validate_factory_profile(factory)
    with pytest.raises(ValueError):
        validate_factory_profile(factory | {"recipe": "g1_wrist_driver_m20_vX"})


def _run(program, plant_steps, *, dt=1 / 600, settle_after=0):
    """Ideal wrist plant: the measured angle follows the reference; fingers move instantly."""
    roll, speed, phases, t = program.roll_ref, 0., [], 0.
    for i in range(plant_steps):
        previous = roll
        settling = i < settle_after
        ref, hand = program.advance(dt, roll, 5. if settling else speed, True, t)
        roll, speed = ref, (ref - previous) / dt
        t += dt
        if not phases or phases[-1] != program.phase:
            phases.append(program.phase)
        assert program.roll_low <= ref <= program.roll_high
        if program.phase in ("return", "settle_return"):
            assert hand == "open"
        if program.phase == "stroke":
            assert hand == "closed"
    return phases


def test_wrist_program_strokes_releases_rolls_back_and_regrips():
    program = WristDriverProgram(roll_low=-1.62, roll_high=1.62, stroke_speed=2., return_speed=4.)
    program.start(-1.62, 0.)
    phases = _run(program, 4000)
    # A command first re-grips from the open hand, then strokes; re-grips sit between strokes.
    assert program.events[0][1] == "release"     # fingers already open in this ideal plant
    assert phases[:8] == ["return", "settle_return", "regrip", "stroke", "settle_stroke", "release", "return",
                          "settle_return"]
    assert program.regrips >= 1 and program.strokes >= 1
    events = [phase for _, phase in program.events]
    assert events.index("release") < events.index("return") < events.index("regrip") < events.index("stroke")


def test_wrist_program_never_regrips_on_a_moving_wrist():
    program = WristDriverProgram(roll_low=-1.62, roll_high=1.62, stroke_speed=2., return_speed=4.)
    program.start(-1.0, 0.)
    program.phase, program.hand, program.roll_ref = "settle_return", "open", -1.62
    for i in range(600):   # measured wrist keeps moving: no re-grip, no reference motion
        ref, hand = program.advance(1 / 600, -1.62, .5, True, i / 600)
        assert (program.phase, hand, ref) == ("settle_return", "open", -1.62)


def test_wrist_program_resumes_from_the_measured_angle():
    program = WristDriverProgram(roll_low=-1.62, roll_high=1.62, stroke_speed=2., return_speed=4.)
    program.start(.3, 0.)
    assert (program.phase, program.roll_ref, program.hand) == ("release", .3, "open")
    program.start(1.6, 1.)
    assert (program.phase, program.roll_ref) == ("return", 1.6)
    program.halt(3.)
    ref, _ = program.advance(1 / 600, 0., 0., True, 3.)
    assert program.phase == "stopped" and ref == program.roll_ref
    with pytest.raises(ValueError):
        WristDriverProgram(roll_low=1., roll_high=-1., stroke_speed=1., return_speed=1.)


def test_grip_target_is_force_limited_past_the_measured_finger_angle():
    # Free finger: driven toward the goal by at most the squeeze; blocked finger:
    # target exactly squeeze past where it is (kp * squeeze), never more.
    assert grip_target(1.2, .5, 1, squeeze=.07) == pytest.approx(.57)
    assert grip_target(-.6, -.1, -1, squeeze=.07) == pytest.approx(-.17)
    assert grip_target(1.2, 1.18, 1, squeeze=.07) == pytest.approx(1.2)    # within reach: the goal itself
    assert grip_target(1.2, 1.3, 1) == 1.2                                  # past the goal: the goal
    assert grip_target(0., .4, 0) == 0.                                      # held thumb abduction
    rng = np.random.default_rng(5)
    for _ in range(200):
        goal, measured, sign = rng.uniform(-2, 2), rng.uniform(-2, 2), int(rng.choice((-1, 1)))
        target = grip_target(goal, measured, sign)
        assert sign * (target - measured) <= .07 + 1e-12 or sign * (goal - measured) <= 0


def test_grasp_uses_thumb_index_and_middle_with_asset_collision_and_no_weld():
    assert set(CLOSING_SIGN) == set(RIGHT_FINGERS)
    assert {CLOSING_SIGN[n] for n in RIGHT_FINGERS if "index" in n or "middle" in n} == {1}
    assert {CLOSING_SIGN[n] for n in ("right_hand_thumb_1_joint", "right_hand_thumb_2_joint")} == {-1}
    for link in ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"):
        assert link in HAND_COLLIDERS
    assert ROLL_JOINT in IK_JOINTS and not any(name.startswith("right_wrist") for name in HELD_POSE)
    import inspect
    from cascade.sim import newton_humanoid_screw as module
    source = inspect.getsource(module)
    assert "'weld'" not in source and "equality_constraint_relpose" not in source
    assert "convert_mjc_equality_constraints=False" in source


def test_hand_rotation_is_palm_down_and_turns_clockwise_from_above():
    for phi in np.linspace(-2, 2, 9):
        r = hand_rotation(phi)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.)
        assert np.allclose(r[:, 1], [0., 0., -1.])          # palm normal down the tool axis
    a, b = hand_rotation(0.)[:, 0], hand_rotation(.1)[:, 0]
    assert np.cross(a, b)[2] < 0                              # positive phi = clockwise = tightening


def test_quaternion_helpers_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = rot_xyzw(q)
        w, x, y, z = quat_wxyz(r)
        assert np.allclose(rot_xyzw((x, y, z, w)), r, atol=1e-9)
    assert np.allclose(quat_wxyz(np.diag([1., -1., -1.])), [0., 1., 0., 0.]) or \
        np.allclose(quat_wxyz(np.diag([1., -1., -1.])), [0., -1., 0., 0.])


def _fake_scene():
    names = ("waist_yaw_joint", "right_elbow_joint", *RIGHT_FINGERS)
    labels = ["world/bolt", "world/nut", "g1/driver/socket_floor", "world/bench", "g1/right_wrist_yaw_link/geom_2",
              "g1/driver/driver_handle", "g1/right_hand_thumb_2_link/geom_1", "g1/right_hand_index_1_link/geom_1",
              "g1/right_hand_middle_1_link/geom_1"]
    labels += [f"g1/driver/socket_wall_{i}" for i in range(6)]
    index = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(labels)}
    shapes = ("nut", "bolt", "socket_floor", "bench", "driver_handle", *(f"socket_wall_{i}" for i in range(6)))
    return NS(lower=np.full(len(names), -2.), upper=np.full(len(names), 2.), effort_caps=np.full(len(names), 25.),
              motor_limit_nm=1., joint_names=names, dt=1 / 600, fixture_position=np.array([.24, 0., 0.]),
              fixture_recipe="factory_m20_fixed_axis_v1", pitch_m=.0025,
              model=NS(shape_label=labels), shape={k: index[k] for k in shapes},
              grip_shapes={"palm": (4,), "thumb": (6,), "index": (7,), "middle": (8,)})


def test_binding_reuses_the_factory_fixture_and_unchanged_verifier_contracts():
    scene = _fake_scene()
    limits = humanoid_limits(scene)
    binding = humanoid_binding(scene, limits, {"document": 1})
    assert binding.fixture_recipe == "factory_m20_fixed_axis_v1" and binding.thread_pitch_m == .0025
    assert binding.fixture_origin_m == (.24, 0., 0.)
    assert binding.thread_contact_pair == ("world/bolt", "world/nut")
    # Only the six hex walls witness the tool; the frictionless floor is allowed, never a witness.
    assert len(binding.tool_contact_pairs) == 6
    assert all(any("socket_wall_" in name for name in pair) for pair in binding.tool_contact_pairs)
    floor = tuple(sorted(("world/nut", "g1/driver/socket_floor")))
    assert floor in binding.allowed_contact_pairs and floor not in binding.tool_contact_pairs
    # The grasp (hand links on the handle) is allowed, never a tool witness; a hand
    # or handle contact with the nut, bolt or table is a forbidden pair.
    grasp = {pair for pair in binding.allowed_contact_pairs if "driver_handle" in pair[0] + pair[1]}
    assert {link.split("/")[1] for pair in grasp for link in pair if "driver_handle" not in link} == {
        "right_wrist_yaw_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"}
    assert not grasp & set(binding.tool_contact_pairs)
    for a, b in binding.allowed_contact_pairs:
        if "right_" in a + b or "driver_handle" in a + b:
            assert "nut" not in a + b and "bolt" not in a + b and "bench" not in a + b
    # Shared defaults stay; only the documented task budgets differ.
    defaults = {f.name: f.default for f in fields(FasteningLimits) if f.name not in (
        "joint_lower_rad", "joint_upper_rad", "joint_effort_nm", "workspace_min_m", "workspace_max_m", "table_z_m")}
    changed = {k for k, v in defaults.items() if getattr(limits, k) != v}
    assert changed == {"max_joint_speed_rad_s", "spindle_effort_nm", "max_command_sim_s", "max_command_wall_s"}
    assert limits.rest_window_sim_s == .5 and limits.max_observation_age_s == .2
    assert ThreadContract(pitch_m=.0025).turn_tolerance == .05   # judged by the unchanged contract


def test_simplifications_are_declared():
    text = " ".join(SIMPLIFICATIONS)
    for phrase in ("pelvis fixed", "gravity compensation", "pre-engaged", "re-grips"):
        assert phrase in text
    assert "weld" not in text
    assert math.isfinite(len(SIMPLIFICATIONS))


def _rows(forces, *, n=100, slip=0.):
    """Synthetic gripped stroke: hand turns 1 rad, driver lags by ``slip``."""
    out = []
    for i in range(n):
        f = i / (n - 1)
        record = {"grip_normal_force_n": dict(forces), "grip_contacts": {k: int(v > 0) for k, v in forces.items()},
                  "grip_torque_nm": .02, "hand_driver_angle_rad": -slip * f}
        out.append(grip_row("stroke", i / 600, record, f, f / (2 * math.pi)))
    return [dict(out[0], phase="regrip")] + out + [dict(out[-1], phase="settle_stroke")]


def test_stroke_stats_require_load_bearing_thumb_index_and_middle():
    ok = stroke_grip_stats(_rows({"thumb": 20., "index": 30., "middle": 30., "palm": 0.}, slip=.02))
    assert len(ok) == 1 and ok[0]["grasp_ok"]
    assert ok[0]["fingers"]["middle"]["share"] == pytest.approx(30 / 80)
    assert ok[0]["slip_rad"] == pytest.approx(.02) and ok[0]["hand_rotation_rad"] == pytest.approx(1.)
    # A pinch without the middle finger is a failed grasp, as is a middle finger below 15 %.
    assert not stroke_grip_stats(_rows({"thumb": 20., "index": 30., "middle": 0., "palm": 0.}))[0]["grasp_ok"]
    low = stroke_grip_stats(_rows({"thumb": 40., "index": 40., "middle": 10., "palm": 0.}))[0]
    assert low["fingers"]["middle"]["share"] < MIN_FINGER_SHARE and not low["grasp_ok"]
    assert MIN_FINGER_SHARE == .15 and MIN_CONTACT_FRACTION == .95
    # Two strokes are segmented separately.
    two = _rows({"thumb": 20., "index": 30., "middle": 30., "palm": 0.}) * 2
    assert [s["stroke"] for s in stroke_grip_stats(two)] == [1, 2]
