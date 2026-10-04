"""MicroDuck selects base capabilities through the same robot-agnostic gateway."""
import pytest

from cascade.apps.robot_runtime import build_robot_runtime
from cascade.config import load_robot_config
from cascade.conversation.domain import ConversationDomain


def test_microduck_conversation_has_no_fictitious_arm_and_read_only_default(tmp_path):
    cfg = load_robot_config("microduck_conversation_mock")
    rt, _ = build_robot_runtime(cfg, tmp_path)
    try:
        assert cfg.robot_id == "microduck-mock"
        assert "locomotion.get_base_state" in rt.tool_descriptors
        assert not any("grasp" in name or "gripper" in name for name in rt.tool_descriptors)
        domain = ConversationDomain(rt, robot_id=cfg.robot_id, allow_tools=("locomotion.get_base_state",))
        assert len(domain.specs()) == 1
        state = rt.execute("locomotion.get_base_state")
        assert state["ok"] and state["state"]["measurement_kind"] == "kinematic_mock"
        assert len(rt.resources) == 1 and next(iter(rt.resources)).synthetic
        with pytest.raises(ValueError, match="explicitly enabled"):
            ConversationDomain(rt, robot_id=cfg.robot_id, allow_tools=("locomotion.walk_velocity",))
    finally:
        rt.close()


def test_native_conversation_profile_keeps_missing_identity_refusal(monkeypatch, tmp_path):
    import os
    for key in tuple(os.environ):
        if key.startswith("CASCADE_MICRODUCK_"):
            monkeypatch.delenv(key)
    from cascade.sim.bridge_client import BridgeClient
    def forbidden(*args, **kwargs):
        raise AssertionError("missing native identity opened a transport")
    monkeypatch.setattr(BridgeClient, "__init__", forbidden)
    cfg = load_robot_config("microduck_conversation_native")
    with pytest.raises(ValueError):
        build_robot_runtime(cfg, tmp_path)


def test_explicit_distance_profile_composes_with_conversation_without_native_io(monkeypatch, tmp_path):
    """Real descriptors/SafeBase composition; dummy identity earns no physics credit."""
    import asyncio
    from cascade.config import load_profile
    from cascade.control.isaac_base import IsaacBase
    from cascade.safety.base_harness import SafeBase
    from cascade.sim.bridge_client import BridgeClient
    from mobile_support_fixture import support_contract

    transport_attempts = []
    def forbidden(*args, **kwargs):
        transport_attempts.append(True)
        raise AssertionError("conversation admission opened a native transport")
    monkeypatch.setattr(BridgeClient, "connect", forbidden)
    cfg = load_robot_config("microduck_conversation_native")
    profile = load_profile("bases", "microduck_distance_native_slow").as_dict()
    profile.update(name="distance_fixture", asset_sha256="a" * 64, policy_sha256="b" * 64,
                   model_identity_sha256="e" * 64, device="cpu", bridge_port=17661,
                   support_contract=support_contract())
    profile["resolved"] = {"safety": profile["safety"]}
    cfg._data["domains"]["locomotion"]["resolved"]["bases"] = [profile]
    runtime, _ = build_robot_runtime(cfg, tmp_path)
    domain = None
    try:
        safe = runtime.domains["locomotion"].runtime.base_rig.primary
        assert isinstance(safe, SafeBase) and isinstance(safe.raw, IsaacBase)
        assert "walk_distance" in safe.capabilities
        descriptor = runtime.tool_descriptors["locomotion.walk_distance"]
        assert descriptor.local_name == "walk_distance" and descriptor.effect == "motion"
        with pytest.raises(ValueError, match="explicitly enabled"):
            ConversationDomain(runtime, robot_id=cfg.robot_id,
                               allow_tools=("locomotion.walk_distance",))
        domain = ConversationDomain(runtime, robot_id=cfg.robot_id, allow_motion=True,
            allow_tools=("locomotion.walk_distance", "emergency_stop"),
            intent_timeout_s=60, execution_timeout_s=30, barge_in="stop_robot")
        assert domain.tools["locomotion.walk_distance"] is descriptor
        assert [tool["name"] for tool in domain.specs()] == ["robot_tool_0", "robot_tool_1"]
        with pytest.raises(ValueError, match="interruption"):
            ConversationDomain(runtime, robot_id=cfg.robot_id, allow_motion=True,
                allow_tools=("locomotion.walk_distance",), barge_in="speech_only")
        with pytest.raises(ValueError, match="permission"):
            ConversationDomain(runtime, robot_id=cfg.robot_id, allow_motion=True,
                               allow_tools=("reset_stop",))
        assert not transport_attempts  # Composition/admission are passive.
    finally:
        if domain is not None:
            closure = asyncio.run(domain.close())
            # No endpoint exists in this composition test: close the worker,
            # without upgrading an undelivered stop into physical success.
            assert not closure["ok"] and closure["stop_worker_closed"]
            assert not closure["action_pending"] and not closure["stop_delivery_pending"]
        assert runtime.close()["ok"]
