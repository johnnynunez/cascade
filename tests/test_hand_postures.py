"""Named hand routing uses the real gates with synthetic completed solves only."""
import asyncio
import time

import pytest
from mobile_tick_fixture import healthy_episode_gc as healthy_episode_gc

from cascade.apps.robot_runtime import robot_tool_descriptors
from cascade.config import load_robot_config
from cascade.conversation.domain import ConversationDomain, ToolIntent
from cascade.robotics.runtime import RobotRuntime
from cascade.skills.hand_runtime import HAND_POSTURES
from test_hand_runtime import start


def test_named_catalog_matches_live_motion_writes_and_capability():
    backend, controller, hand = start()
    try:
        declared = robot_tool_descriptors(load_robot_config("leap_hand_right_mujoco"))
        live = {t.name: t for t in hand.tool_descriptors}
        tool = live["hand.set_hand_posture"]
        assert declared[tool.name].as_dict() == tool.as_dict()
        assert tool.effect == "motion" and tool.writes == tool.requires == ("hand/fingers",)
        assert tool.local_name in hand.motion_skills
        assert "named_hand_postures" in hand.resources[0].capabilities
        assert tool.parameters["additionalProperties"] is False
        assert tuple(tool.parameters["properties"]["posture"]["enum"]) == tuple(HAND_POSTURES)
    finally:
        assert controller.close()["ok"]


@pytest.mark.parametrize("posture", ["index_flex", "neutral"])
def test_conversation_named_posture_reaches_same_owner_and_retained_rest(posture, healthy_episode_gc):
    async def scenario():
        backend, controller, hand = start()
        runtime = RobotRuntime({"hand": hand})
        domain = ConversationDomain(runtime, robot_id="leap_hand_right",
            allow_tools=["hand.set_hand_posture"], allow_motion=True)
        try:
            domain.claim("voice")
            result = await domain.dispatch(ToolIntent("voice", "leap_hand_right", "response", "posture",
                runtime.cancellation_token, time.monotonic()+2, "hand.set_hand_posture", {"posture": posture}))
            assert result["ok"] and result["verified"], result
            assert result["synthetic"] and not result["physical_stop_verified"]
            assert result["rest"]["window_sim_s"] >= .2-1e-9
            assert result["stop"]["generation"] == result["admission"]["generation"]+1
            assert result["rest"]["generation"] == result["stop"]["generation"]
            assert tuple(result["stop"]["targets_rad"]) == HAND_POSTURES[posture]
            assert backend.q == HAND_POSTURES[posture]
            assert len({ident for ident, _ in backend.uploads}) == 1
            assert controller._latched and controller._permit is None
        finally:
            assert (await domain.close())["ok"]
            assert runtime.close()["ok"]
    asyncio.run(scenario())


@pytest.mark.parametrize("args", [{}, {"posture": "close"}, {"posture": None}, {"posture": True},
    {"posture": []}, {"posture": "neutral", "positions_rad": [1.]*16},
    {"positions_rad": [0.]*16}, {"posture": "index_flex", "duration": 20}])
def test_unknown_or_raw_posture_arguments_never_reach_controller(args):
    backend, controller, hand = start()
    runtime = RobotRuntime({"hand": hand})
    generation = controller._generation
    try:
        result = runtime.execute("hand.set_hand_posture", args)
        assert not result["ok"]
        assert controller._generation == generation and controller._permit is None
        assert backend.targets == (0.,)*16
    finally:
        assert runtime.close()["ok"]


def test_raw_fingers_and_unenabled_named_motion_remain_outside_voice():
    _, controller, hand = start()
    runtime = RobotRuntime({"hand": hand})
    try:
        for tool, allow_motion in (("hand.move_fingers", True), ("hand.set_hand_posture", False)):
            with pytest.raises(ValueError, match="curated semantic motion"):
                ConversationDomain(runtime, robot_id="leap_hand_right", allow_tools=[tool], allow_motion=allow_motion)
        with pytest.raises(TypeError):
            HAND_POSTURES["index_flex"] = (1.,)*16
    finally:
        assert runtime.close()["ok"]


def test_expired_intent_and_priority_stop_prevent_named_motion():
    async def scenario():
        backend, controller, hand = start()
        runtime = RobotRuntime({"hand": hand})
        domain = ConversationDomain(runtime, robot_id="leap_hand_right",
            allow_tools=["hand.set_hand_posture"], allow_motion=True)
        try:
            domain.claim("voice")
            generation = controller._generation
            expired = ToolIntent("voice", "leap_hand_right", "r", "expired",
                runtime.cancellation_token, time.monotonic()-1, "hand.set_hand_posture", {"posture": "index_flex"})
            assert not (await domain.dispatch(expired))["ok"]
            assert controller._generation == generation and controller._permit is None
            token = runtime.cancellation_token
            assert (await domain.stop())["ok"]
            stopped = ToolIntent("voice", "leap_hand_right", "r", "stopped",
                token, time.monotonic()+2, "hand.set_hand_posture", {"posture": "index_flex"})
            assert not (await domain.dispatch(stopped))["ok"]
            assert runtime.stopped and controller._latched and controller._permit is None
            assert backend.targets == (0.,)*16
        finally:
            assert (await domain.close())["ok"]
            assert runtime.close()["ok"]
    asyncio.run(scenario())
