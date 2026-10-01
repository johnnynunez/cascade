"""Malformed passive evidence must never authorize episode rehydration."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark/diagnostics"))
from nvblox_contact_recovery import validate_live
import nvblox_contact_recovery as diagnostic


def fixture():
    physics = {"engine": "physx", "robot_id": "/Robot", "physics_step": 2,
        "q": [0.] * 6 + [.02, .02],
        "joint_names": ["joint" + str(i) for i in range(8)],
        "gripper": {"indices": [6, 7], "q": [.02, .02]},
        "spawn_positions_m": {"tomato_can": [.2, .1, .04]},
        "scene_geometry": {"scene_config_sha256": "unchanged", "convex_collider": {
            "physx_support": {"physics_step": 2, "source_vertices_f32_sha256": "same"}}},
        "props": {"tomato_can": {"position_m": [.2, .1, .04], "orientation_wxyz": [1., 0, 0, 0]}},
        "contacts": {"channel": "physx_gpu_contact_tensor", "device": "cuda:0",
            "sensor_paths": ["/World_Props/tomato_can"], "physics_step": 2,
            "jaw_forces_n": [[1., 0, 0], [-1., 0, 0]], "jaw_contact_counts": [1, 1],
            "filter_paths": [["/Robot/left", "/Robot/right"]]}}
    historical = {"object": "tomato_can", "stationary_physics": copy.deepcopy(physics)}
    historical["stationary_physics"]["physics_step"] = 1
    runtime = SimpleNamespace(cfg=SimpleNamespace(arm=SimpleNamespace(bridge_robot_id="/Robot",
        joint_signs=[-1.] * 6)), arm=SimpleNamespace(get_state=lambda: SimpleNamespace(q=np.zeros(6))))
    return runtime, physics, historical


def test_live_binding_requires_same_retained_arm_jaws_and_actual_contacts():
    runtime, physics, historical = fixture()
    result = validate_live(runtime, physics, historical)
    assert result["max_arm_delta_rad"] == 0. and result["max_jaw_delta_m"] == 0.
    physics["contacts"]["jaw_contact_counts"][1] = 0
    with pytest.raises(ValueError, match="bilateral"):
        validate_live(runtime, physics, historical)


@pytest.mark.parametrize("field", ["live_q", "old_q", "runtime_q", "prop", "old_prop", "signs"])
@pytest.mark.parametrize("malformed", ["nan", "shape"])
def test_nan_or_malformed_binding_rejected(field, malformed):
    runtime, physics, historical = fixture()
    value = [float("nan")] * (3 if "prop" in field else 6 if field in {"runtime_q", "signs"} else 8)
    if malformed == "shape":
        value = value[:-1]
    if field == "live_q":
        physics["q"] = value
    elif field == "old_q":
        historical["stationary_physics"]["q"] = value
    elif field == "runtime_q":
        runtime.arm.get_state = lambda: SimpleNamespace(q=value)
    elif field == "prop":
        physics["props"]["tomato_can"]["position_m"] = value
    elif field == "old_prop":
        historical["stationary_physics"]["props"]["tomato_can"]["position_m"] = value
    else:
        runtime.cfg.arm.joint_signs = value
    with pytest.raises(ValueError, match="finite measured"):
        validate_live(runtime, physics, historical)


@pytest.mark.parametrize("field", ["joint", "jaw", "prop", "robot", "clock"])
def test_changed_retained_state_rejected(field):
    runtime, physics, historical = fixture()
    if field == "joint":
        physics["q"][0] += .002
    elif field == "jaw":
        physics["q"][6] += .002
    elif field == "prop":
        physics["props"]["tomato_can"]["position_m"][0] += .002
    elif field == "robot":
        physics["robot_id"] = "/AnotherRobot"
    else:
        physics["physics_step"] = 1
    with pytest.raises(ValueError):
        validate_live(runtime, physics, historical)


@pytest.mark.parametrize("field", ["geometry", "jaws", "spawn", "orientation", "nan_orientation"])
def test_changed_scene_or_orientation_is_not_the_retained_contact(field):
    runtime, physics, historical = fixture()
    if field == "geometry":
        physics["scene_geometry"]["scene_config_sha256"] = "different"
    elif field == "jaws":
        physics["gripper"]["indices"] = [0, 1]
    elif field == "spawn":
        physics["spawn_positions_m"]["tomato_can"][0] += .01
    elif field == "orientation":
        physics["props"]["tomato_can"]["orientation_wxyz"] = [np.cos(.01), 0, 0, np.sin(.01)]
    else:
        physics["props"]["tomato_can"]["orientation_wxyz"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_live(runtime, physics, historical)


@pytest.mark.parametrize("failure", [None, "slipped", "no_lift", "wrong_target", "jaw_command"])
def test_independent_retreat_validation_stops_reset_before_following_actions(monkeypatch, tmp_path, failure):
    from cascade.config import Cfg
    from cascade.types import SkillError
    runtime, after, historical = fixture()
    before = copy.deepcopy(after)
    after["physics_step"] += 1
    after["contacts"]["physics_step"] += 1
    after["props"]["tomato_can"]["position_m"][2] += .03
    if failure == "slipped":
        after["contacts"]["jaw_contact_counts"][1] = 0
    if failure == "no_lift":
        after["props"]["tomato_can"]["position_m"][2] -= .03
    runtime.cfg = Cfg({"arm": {"joint_signs": [-1.] * 6, "settle_tol": .02}})
    historical.update(q_pre=np.zeros(6), cylinder=[[.2, .1], .15, -.06])
    frames = iter([{"physics": before}, {"physics": after}])
    monkeypatch.setattr(diagnostic, "fresh_state", lambda *a: next(frames))
    calls = [{"layer": "actuator", "call": "send_joint_target"}]
    if failure == "jaw_command":
        calls.append({"layer": "actuator", "call": "set_gripper"})
    recorder = SimpleNamespace(phase="public_reset", calls=lambda phase: calls)
    observer = SimpleNamespace(mark=lambda name: None)
    def original(rt):
        command = {"monotonic": time.monotonic(), "event": "begin", "call": "move_joints",
            "args": [[.1 if failure == "wrong_target" else 0.] * 6], "exemption": historical["cylinder"]}
        (tmp_path / "skills.jsonl").write_text(json.dumps(command) + "\n")
        return {"retreated_to_original_pregrasp": True}
    if failure is None:
        assert diagnostic.retained_retreat_phase(observer, runtime, recorder, historical,
                                                 tmp_path, original)["retreated_to_original_pregrasp"]
        assert runtime._diagnostic_contact_retreat["pass"]
    else:
        with pytest.raises(SkillError, match="independent physical"):
            diagnostic.retained_retreat_phase(observer, runtime, recorder, historical, tmp_path, original)
        assert not runtime._diagnostic_contact_retreat["pass"]
    assert recorder.phase == "public_reset"
