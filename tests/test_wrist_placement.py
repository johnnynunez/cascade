"""A wrist camera must resolve the destination before its view is occluded."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cascade.config import load_demo_config
from cascade.memory import EpisodicMemory
from cascade.skills.runtime import SkillRuntime
from cascade.types import SkillError


def runtime(held=None):
    rt = SkillRuntime.__new__(SkillRuntime)
    rt.cfg = load_demo_config(arm="fr3", camera="d435i", llm="mock")
    rt.held_object = held
    rt._held_det_label = held
    rt._held_offset = np.array([.03, 0., -.12])
    rt._object_pose = None
    rt.memory = EpisodicMemory()
    rt.arm = SimpleNamespace(harness=SimpleNamespace(estopped=False))
    rt._reconcile_held = Mock()
    rt._home_return_skip_reason = Mock(return_value=None)
    rt.skill_move_home = Mock(return_value={"at": "home"})
    rt.observe = Mock()
    rt._destination_fix = Mock(return_value=(np.array([.45, -.15, .03]), .04))
    rt.skill_place_at = Mock(return_value={"at": [.45, -.15, .10]})
    return rt


def test_pick_retains_destination_when_wrist_view_changes():
    rt = runtime()
    destination = rt._destination_fix.return_value[0]

    def grasp(label, **kwargs):
        rt._destination_fix.assert_called_once_with("blue tray")
        destination[:] = 9.  # shared perception storage changes during pickup
        rt.held_object = label
        rt._destination_fix.side_effect = AssertionError("tray now outside wrist view")
        return {"held": label}

    rt.skill_grasp_object = grasp
    result = rt.skill_pick_and_place("ball", "blue tray")
    assert result["destination"] == "blue tray"
    assert result["place_attempts"] == 1
    np.testing.assert_allclose(rt.skill_place_at.call_args.args, [.45, -.15, .10])
    rt.skill_move_home.assert_called_once()


def test_unseen_destination_fails_before_picking():
    rt = runtime()
    rt._destination_fix.return_value = None
    rt.skill_grasp_object = Mock()
    with pytest.raises(SkillError, match="cannot locate destination"):
        rt.skill_pick_and_place("ball", "missing tray")
    rt.skill_grasp_object.assert_not_called()
    rt.skill_place_at.assert_not_called()


def test_standalone_place_gets_fresh_destination_from_overview():
    rt = runtime(held="ball")
    order = []
    rt.skill_move_home.side_effect = lambda: order.append("home")
    rt.observe.side_effect = lambda **kw: order.append("frame")
    rt._destination_fix.side_effect = lambda label: (
        order.append(label) or (np.array([.5, .1, .02]), .025))
    rt.skill_place_on_object("tray")
    assert order == ["home", "frame", "tray"]
    np.testing.assert_allclose(rt.skill_place_at.call_args.args, [.5, .1, .085])
    assert rt.held_object == "ball"  # mocked place has not released it


def test_missing_standalone_destination_does_not_retry_or_release():
    rt = runtime(held="ball")
    rt._destination_fix.return_value = None
    with pytest.raises(SkillError, match="cannot locate destination"):
        rt.skill_place_on_object("tray")
    rt._destination_fix.assert_called_once()
    rt.skill_place_at.assert_not_called()
    assert rt.held_object == "ball"


def test_failed_place_reports_once_and_keeps_held_object():
    rt = runtime(held="ball")
    rt.skill_place_at.side_effect = SkillError("target unreachable")
    result = rt.skill_pick_and_place("ball", "tray")
    assert not result["ok"] and result["stage"] == "place"
    assert "target unreachable" in result["error"]
    assert rt.held_object == "ball"
    rt.skill_place_at.assert_called_once()


def test_failed_home_prevents_destination_lookup_and_placement():
    rt = runtime(held="ball")
    rt.skill_move_home.side_effect = SkillError("STOP requested")
    with pytest.raises(SkillError, match="STOP requested"):
        rt.skill_place_on_object("tray")
    rt._destination_fix.assert_not_called()
    rt.skill_place_at.assert_not_called()
    assert rt.held_object == "ball"


def test_wrist_camera_does_not_use_table_position_as_held_object_offset():
    rt = runtime(held="ball")
    rt.kin = Mock()
    assert rt._held_object_offset() is None
    rt.observe.assert_not_called()
    assert rt.held_object == "ball"


def test_independent_held_pose_still_detects_slip():
    rt = runtime(held="ball")
    rt.arm.get_state = Mock(return_value=SimpleNamespace(q=np.zeros(7)))
    tcp = np.eye(4)
    tcp[:3, 3] = [.4, 0., .3]
    rt.kin = SimpleNamespace(fk=lambda q: tcp)
    rt._object_pose = lambda label: np.array([.4, 0., .02])
    np.testing.assert_allclose(rt._held_object_offset(), [0., 0., -.28])
    rt.observe.assert_not_called()


def test_native_empty_jaws_prevent_standalone_place_before_moving():
    rt = runtime(held="ball")
    rt._reconcile_held = SkillRuntime._reconcile_held.__get__(rt)
    rt._gripper_width_frac = lambda: 0.
    rt._finite_width_frac = lambda: 0.
    rt._gripper_width_m = lambda: 0.
    with pytest.raises(SkillError, match="not holding anything"):
        rt.skill_place_on_object("tray")
    rt.skill_move_home.assert_not_called()
    rt.skill_place_at.assert_not_called()
    assert rt.held_object is None
