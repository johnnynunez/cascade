""""open the gripper" / "close the gripper" on a real arm.

Found on the physical reBot (mock brain, reflex tier):
- "open gripper" was reported failed ("gripper still closed at 0.07"): the
  skill returned as soon as the target was sent and the postcondition read
  the jaws 0.5 ms later, before they had moved.
- "close gripper" had no reflex rule, so it needed the LLM tier; the offline
  mock brain spent its 30-step budget and never closed the jaws.
"""

from types import SimpleNamespace

import pytest

from cascade.agent.reflex import parse_command
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


@pytest.mark.parametrize("text,tool", [
    ("close the gripper", "close_gripper"),
    ("close gripper", "close_gripper"),
    ("please close the gripper", "close_gripper"),
    ("cierra la pinza", "close_gripper"),
    ("open gripper", "open_gripper"),
    ("open the gripper", "open_gripper"),
])
def test_gripper_commands_run_without_the_llm(text, tool):
    plan = parse_command(text)
    assert plan is not None, text
    assert plan.calls == [(tool, {})]


def test_close_rule_does_not_capture_other_commands():
    assert parse_command("close the gripper and then go home") is None   # sequences go to the LLM
    assert parse_command("close the box") is None


def opener(widths, *, timeout):
    """SkillRuntime stand-in whose jaw feedback replays `widths` (last repeats)."""
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = SimpleNamespace(grasp={"open_wait_timeout_s": timeout})
    rt._grip_open = 4.7212
    sent, reads = [], []
    rt.arm = SimpleNamespace(set_gripper=lambda pos, effort=1.0: sent.append(pos))
    rt.held_object = None
    rt.memory = SimpleNamespace(add=lambda *a: None)

    def feedback():
        reads.append(1)
        return widths[min(len(reads), len(widths)) - 1]

    rt._gripper_width_frac = feedback
    return rt, sent, reads


def test_open_returns_only_after_the_jaws_opened():
    rt, sent, reads = opener([0.07, 0.3, 0.6, 0.9, 0.97], timeout=6.0)
    assert rt.skill_open_gripper() == {"gripper": "open"}
    assert sent == [4.7212]
    assert len(reads) == 5            # kept reading until the jaws were open


def test_default_returns_as_soon_as_the_target_is_sent():
    rt, sent, reads = opener([0.07], timeout=None)
    rt.skill_open_gripper()
    assert sent == [4.7212] and reads == []


def test_a_stall_short_of_open_ends_the_wait_for_the_postcondition_to_judge():
    rt, _, reads = opener([0.07, 0.2, 0.4], timeout=6.0)   # blocked at 0.4
    rt.skill_open_gripper()
    assert 3 < len(reads) < 30        # ~0.5 s of a steady reading, not the timeout


def test_unknown_feedback_does_not_wait():
    rt, _, reads = opener([None], timeout=6.0)
    rt.skill_open_gripper()
    assert len(reads) == 1


def test_jaws_still_moving_at_the_deadline_is_an_error():
    rt, _, _ = opener([0.07 + 0.01 * i for i in range(200)], timeout=0.2)
    with pytest.raises(SkillError, match="still opening"):
        rt.skill_open_gripper()


def test_release_bookkeeping_is_unchanged():
    rt, _, _ = opener([0.97], timeout=6.0)
    rt.held_object, rt._held_det_label = "cup", "cup"
    rt.skill_open_gripper()
    assert rt.held_object is None and rt._held_det_label is None


def test_only_the_rebot_profile_waits():
    from cascade.config import load_demo_config

    rebot = load_demo_config(arm="rebot_rs", camera="mock", llm="mock")
    assert rebot.grasp.get("open_wait_timeout_s") == 6.0
    mock = load_demo_config(arm="mock", camera="mock", llm="mock")
    assert mock.grasp.get("open_wait_timeout_s") is None
