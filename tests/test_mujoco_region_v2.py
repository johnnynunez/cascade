"""Version2 static geometry and source-bound refusal controls; no stepping."""
from copy import deepcopy
import json

import numpy as np
import pytest

from cascade.sim.mujoco_placement import Region
from cascade.skills.mujoco_region import select
from cascade.skills.runtime import _PostPlaceRetreatPlanError
from test_mujoco_withdrawal import model_runtime as model_runtime, captured, CAPTURE  # noqa: F401

V2 = {'version': 2, 'name': 'drop zone', 'frame': 'robot_base',
      'bounds_xy_m': [[.13, -.18], [.27, -.06]], 'planning_margin_m': .02,
      'capacity_scope': 'all_free_bodies'}


def test_version2_contract_is_explicit_and_version1_wire_remains_exact():
    assert Region.parse(V2).as_dict() == V2
    v1 = {'version': 1, 'name': 'drop zone', 'frame': 'robot_base', 'bounds_xy_m': [[.15, -.16], [.24, -.08]]}
    assert Region.parse(v1).as_dict() == v1
    with pytest.raises(ValueError, match='whole-inventory'):
        Region.parse(V2).candidates([[-.0175, -.0175], [.0175, .0175]], .02)


@pytest.mark.parametrize('change', [dict(planning_margin_m=0), dict(planning_margin_m=True),
    dict(planning_margin_m=float('nan')), dict(planning_margin_m=.08),
    dict(capacity_scope='held_only'), dict(capacity_scope=None)])
def test_version2_rejects_unknown_or_missing_margin_capacity(change):
    with pytest.raises(ValueError):
        Region.parse(V2 | change)


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_full_candidate_has_oriented_capacity_and_preserves_world(model_runtime, tmp_path):
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    data = rt.arm.raw.world.data
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    result = select(rt)
    (tmp_path/'region2-static-plan.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['region']['version'] == 2
    capacity = result['capacity']
    assert capacity['planning_margin_m'] == .02 and capacity['future_task_admission'] is False
    assert set(capacity['initial_layout']['footprints']) == {9, 10}
    assert set(capacity['after_oriented_release']['footprints']) == {9, 10}
    assert result['physical_task_verdict'] is False
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    np.testing.assert_array_equal(data.ctrl, before[2])
    assert data.time == before[3] == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
def test_too_small_region_refuses_whole_inventory_before_ik(model_runtime, monkeypatch):
    from cascade.sim.mujoco_placement import PlacementHistory
    from cascade.skills import mujoco_region
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    rt.cfg.arm._data['mj_delivery_area'] = deepcopy(V2) | {'bounds_xy_m': [[.15, -.16], [.24, -.08]]}
    world = rt.arm.raw.world
    world.placement_history = PlacementHistory(world, rt.arm.raw)
    captured(rt, CAPTURE[0]['rows'][0])
    before = (world.data.qpos.copy(), world.data.ctrl.copy(), world.data.time)
    monkeypatch.setattr(mujoco_region, 'plan_geometry', lambda *a, **k: pytest.fail('capacity refused before IK'))
    with pytest.raises(_PostPlaceRetreatPlanError, match='cannot fit'):
        select(rt)
    np.testing.assert_array_equal(world.data.qpos, before[0])
    np.testing.assert_array_equal(world.data.ctrl, before[1])
    assert world.data.time == before[2] == 0.


@pytest.mark.parametrize('model_runtime', ['so101_mujoco'], indirect=True)
@pytest.mark.parametrize('case', ['prefix_fault', 'workspace', 'binding'])
def test_region2_refuses_invalid_admission_without_motion(model_runtime, case):
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    world = rt.arm.raw.world
    before = (world.data.qpos.copy(), world.data.ctrl.copy(), world.data.time)
    if case == 'prefix_fault':
        world.placement_history.prefix_faults['blue_cube'] = {'step': 1, 'reason': 'retained old fault'}
    elif case == 'workspace':
        rt.arm.harness.limits.workspace_max[0] = .26
    else:
        rt.cfg.arm._data['mj_delivery_area']['planning_margin_m'] = .021
    with pytest.raises(_PostPlaceRetreatPlanError, match='disturbed|workspace|bound'):
        select(rt)
    np.testing.assert_array_equal(world.data.qpos, before[0])
    np.testing.assert_array_equal(world.data.ctrl, before[1])
    assert world.data.time == before[2] == 0.
