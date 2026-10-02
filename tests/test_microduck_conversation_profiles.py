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
