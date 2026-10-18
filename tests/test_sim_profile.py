"""ASPIRE A1: which profiles count as simulation (spec §4.4, preflight R21/R22).

`profile_class(cfg)` is "sim" only when EVERY arm profile the runtime builds
(`apps.demo._arm_cfgs`) has a `type` in {isaac, mujoco, mock} and the booth is
off. Anything else -- a hardware type, a missing or unknown type, an empty or
malformed arm list, a mobile or composed config, an exception -- is "real":
real profiles get no sim retry and only verifier-passed notes. Config reads only;
no arm object is built.
"""
from __future__ import annotations

import pytest

from cascade import sim_profile
from cascade.config import load_demo_config, load_robot_config
from cascade.sim_profile import (
    SIM_ARM_TYPES,
    arm_types,
    learning_surface_enabled,
    profile_class,
    sim_retry_enabled,
)


@pytest.fixture(autouse=True)
def _no_ambient_kill_switch(monkeypatch):
    # A shell that exported the kill switch must not decide these tests; the
    # observed-finger gate is on (the Spark default) unless a test says otherwise:
    # the retry needs it (preflight R8, A7).
    monkeypatch.delenv("CASCADE_SIM_RETRY", raising=False)
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")


def test_sim_arm_types_are_exactly_the_simulated_backends():
    assert SIM_ARM_TYPES == frozenset({"isaac", "mujoco", "mock"})


# ros2_generic is a template (no model, n_joints, home_q): config-only here (R22).
@pytest.mark.parametrize("arm", ["rebot_rs", "rebot_rs_mb", "so101", "fr3", "piper", "ros2_generic"])
def test_hardware_profiles_are_real(arm):
    assert profile_class(load_demo_config(arm=arm)) == "real"


@pytest.mark.parametrize("arm", ["mock", "isaac", "isaac_kitchen", "isaac_kitchen_gpu", "so101_mujoco"])
def test_sim_profiles_are_sim(arm):
    assert profile_class(load_demo_config(arm=arm)) == "sim"


def test_arm_types_lists_every_arm_of_the_rig():
    assert arm_types(load_demo_config(arm="isaac_kitchen_gpu")) == ["isaac"]
    assert arm_types(load_demo_config(arms=["so101_left_mujoco", "so101_right_mujoco"])) == ["mujoco", "mujoco"]


def test_two_sim_arms_are_sim():
    assert profile_class(load_demo_config(arms=["so101_left_mujoco", "so101_right_mujoco"])) == "sim"


def test_mixed_rig_is_real():  # Review Focus 3: one isaac arm + one rebot_rs arm
    cfg = load_demo_config(arm="isaac")
    cfg._data["arms"] = [dict(cfg.arm._data), {"type": "rebot_rs"}]
    assert arm_types(cfg) == ["isaac", "rebot_rs"]
    assert profile_class(cfg) == "real"
    # ...in either order: the hardware arm is never the one that decides "sim"
    cfg._data["arms"] = [{"type": "rebot_rs"}, dict(cfg.arm._data)]
    assert profile_class(cfg) == "real"


@pytest.mark.parametrize("arm_cfg", [{}, {"type": None}, {"type": "isaac_next"}, {"type": "Isaac"},
                                     {"type": ["isaac"]}, {"type": " mock"}],
                         ids=["no-type", "null-type", "unknown-type", "other-case", "list-type", "padded"])
def test_missing_or_unknown_type_is_real(arm_cfg):
    cfg = load_demo_config(arm="isaac")
    cfg._data["arm"] = dict(arm_cfg)
    cfg._data["arms"] = [dict(arm_cfg)]
    assert profile_class(cfg) == "real"
    cfg._data.pop("arms")  # the single-arm fallback reads cfg.arm
    assert profile_class(cfg) == "real"
    kind = arm_cfg.get("type")
    assert arm_types(cfg) == [kind if isinstance(kind, str) else ""]  # always a list of strings


def test_empty_arm_list_without_an_arm_is_real():  # R21: all([]) is True
    cfg = load_demo_config(arm="isaac")
    cfg._data["arms"] = []
    cfg._data.pop("arm")
    assert arm_types(cfg) == []
    assert profile_class(cfg) == "real"


def test_no_arm_types_at_all_is_real_whatever_arm_types_returns(monkeypatch):
    # R21: the guard does not rely on _arm_cfgs never returning [].
    monkeypatch.setattr(sim_profile, "arm_types", lambda cfg: [])
    assert profile_class(load_demo_config(arm="isaac")) == "real"


def test_malformed_arm_entries_are_real():
    cfg = load_demo_config(arm="isaac")
    cfg._data["arms"] = ["isaac"]  # a bare string, not a profile
    assert profile_class(cfg) == "real"
    cfg._data["arms"] = [dict(cfg.arm._data), None]
    assert profile_class(cfg) == "real"


def test_mobile_and_composed_configs_are_real():  # R21
    assert profile_class(load_demo_config(base="microduck_mock")) == "real"
    # a composition of mock arms is still not a plain sim arm rig
    assert profile_class(load_robot_config("fixed_so101_mock")) == "real"


def test_unreadable_config_is_real():
    assert profile_class(None) == "real"
    assert profile_class(object()) == "real"


def test_booth_makes_a_sim_profile_real(monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "1")
    assert profile_class(load_demo_config(arm="isaac_kitchen_gpu")) == "real"


def test_a_booth_switched_on_after_the_config_was_loaded_makes_it_real(monkeypatch):
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    monkeypatch.setenv("CASCADE_BOOTH", "1")
    assert profile_class(cfg) == "real"


def test_a_config_loaded_under_the_booth_stays_real_after_the_switch_is_gone(monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "1")
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    monkeypatch.delenv("CASCADE_BOOTH")
    assert profile_class(cfg) == "real"


def test_booth_negatives_keep_sim(monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "0")
    assert profile_class(load_demo_config(arm="isaac_kitchen_gpu")) == "sim"


def test_sim_retry_needs_sim_class_flag_and_no_kill_switch(monkeypatch):
    cfg = load_demo_config(arm="isaac")
    assert sim_retry_enabled(cfg) is False  # configs/demo.yaml: agent.sim_retry false
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    assert sim_retry_enabled(cfg) is True
    assert sim_retry_enabled(load_demo_config(arm="isaac_kitchen_gpu")) is True  # the kitchen turns it on (A7)
    monkeypatch.setenv("CASCADE_SIM_RETRY", "0")
    assert sim_retry_enabled(cfg) is False
    real = load_demo_config(arm="rebot_rs")
    real._data.setdefault("agent", {})["sim_retry"] = True
    monkeypatch.delenv("CASCADE_SIM_RETRY")
    assert sim_retry_enabled(real) is False


@pytest.mark.parametrize("value", ["0", "false", " Off ", "no", "disabled", "2"])
def test_kill_switch_values_that_are_not_a_clear_yes_disable_the_retry(monkeypatch, value):
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    monkeypatch.setenv("CASCADE_SIM_RETRY", value)
    assert sim_retry_enabled(cfg) is False


@pytest.mark.parametrize("value", ["", "1", "true", " ON "])
def test_kill_switch_unset_or_yes_keeps_the_retry(monkeypatch, value):
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    monkeypatch.setenv("CASCADE_SIM_RETRY", value)
    assert sim_retry_enabled(cfg) is True


@pytest.mark.parametrize("flag", ["true", 1, "yes", None])
def test_sim_retry_flag_must_be_a_real_true(flag):
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    cfg._data.setdefault("agent", {})["sim_retry"] = flag
    assert sim_retry_enabled(cfg) is False


@pytest.mark.parametrize("arm", ["rebot_rs", "so101", "fr3", "piper", "ros2_generic"])
def test_no_setting_enables_the_retry_on_a_real_profile(arm):
    cfg = load_demo_config(arm=arm)
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    assert sim_retry_enabled(cfg) is False


def test_sim_retry_on_a_booth_config_is_off(monkeypatch):
    monkeypatch.setenv("CASCADE_BOOTH", "1")
    cfg = load_demo_config(arm="isaac_kitchen_gpu")
    cfg._data.setdefault("agent", {})["sim_retry"] = True
    assert sim_retry_enabled(cfg) is False


def test_sim_retry_of_an_unreadable_config_is_off():
    assert sim_retry_enabled(None) is False


@pytest.mark.parametrize("gate", [None, "0", "", "true", " 1", "yes"])
def test_sim_retry_needs_the_observed_finger_gate(monkeypatch, gate):  # preflight R8 (A7)
    if gate is None:
        monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE")
    else:
        monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", gate)
    assert sim_retry_enabled(load_demo_config(arm="isaac_kitchen_gpu")) is False
    monkeypatch.setenv("CASCADE_OBSERVED_FINGER_GATE", "1")
    assert sim_retry_enabled(load_demo_config(arm="isaac_kitchen_gpu")) is True


def test_learning_surface_reads_its_own_switch_on_any_profile():
    for arm in ("isaac_kitchen_gpu", "rebot_rs"):
        cfg = load_demo_config(arm=arm)
        assert learning_surface_enabled(cfg) is False  # default off (A9 sets the kitchen profile)
        cfg._data.setdefault("agent", {})["learning_surface"] = True
        assert learning_surface_enabled(cfg) is True
        cfg._data["agent"]["learning_surface"] = "yes"
        assert learning_surface_enabled(cfg) is False
    assert learning_surface_enabled(None) is False
