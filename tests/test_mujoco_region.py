"""Region contract and real native-structure static controls; no task admission."""
from pathlib import Path
import json

import numpy as np
import pytest

from cascade.sim.mujoco_placement import Region, audit

REGION = {'version': 1, 'name': 'drop zone', 'frame': 'robot_base',
          'bounds_xy_m': [[.15, -.16], [.24, -.08]]}


def rows():
    return [{'epoch': 'epoch', 'model_sha256': 'model', 'step': index+1,
             'time_s': index*.12, 'final_state_time_s': index*.12+.005,
             'open_since_s': 0., 'gripper_open_fraction': 1., 'final_gripper_open_fraction': 1.,
             'final_footprints': {'cube': {'lower_m': [.18, -.14, 0.], 'upper_m': [.215, -.105, .035],
                                         'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.}},
             'objects': {'cube': {'position_m': [.1975, -.1225, .0175],
                        'lower_m': [.18, -.14, 0.], 'upper_m': [.215, -.105, .035],
                        'support_up_n': .5, 'support_contacts': 4, 'other_loaded_contacts': 0, 'arm_active_contacts': 0,
                        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.}}}
            for index in range(7)]


def check(samples):
    return audit(samples, 'cube', Region.parse(REGION), 'model', 'epoch')


def test_semantic_edges_erode_by_footprint_separation_is_between_objects():
    region = Region.parse(REGION)
    points = np.asarray(region.candidates([[-.0175, -.0175], [.0175, .0175]], .02))
    np.testing.assert_allclose(points.min(axis=0), [.1675, -.1425])
    np.testing.assert_allclose(points.max(axis=0), [.2225, -.0975])
    # 90 mm width - 35 mm footprint = 55 mm of center range, NOT a
    # claim that another 20 mm has also been eroded from each boundary.
    assert np.ptp(points[:, 0]) == pytest.approx(.055)
    assert region.candidates([[-.06, -.06], [.06, .06]], .02) == []


@pytest.mark.parametrize('change', [dict(version=True), dict(frame='camera'), dict(name=''),
    dict(bounds_xy_m=[[1, 1], [0, 0]]), dict(bounds_xy_m=[[0, 0], [float('nan'), 1]])])
def test_invalid_region_refuses(change):
    with pytest.raises(ValueError):
        Region.parse(REGION | change)


def test_sampled_settling_window_confirmed_without_claiming_all_substeps():
    report = check(rows())
    assert report['status'] == 'confirmed'
    assert report['measured']['duration_s'] >= .6
    assert report['measured']['samples'] >= 6
    assert report['measured']['sampling'] == 'last-solve of existing arm batches'


@pytest.mark.parametrize('case,expected', [('short', 'unverified'), ('frozen', 'unverified'),
    ('gap', 'unverified'), ('epoch', 'unverified'), ('identity', 'unverified'),
    ('no_release', 'unverified'), ('release_late', 'unverified'), ('outside', 'refuted'),
    ('final_outside', 'refuted'), ('closed', 'refuted'), ('no_support', 'refuted'),
    ('robot_touch', 'refuted'), ('moving', 'refuted'), ('spinning', 'refuted'),
    ('drifting', 'refuted'), ('unknown_force', 'unverified')])
def test_evidence_adversaries_preserve_verdicts(case, expected):
    samples = rows()
    if case == 'short': samples = samples[-4:]
    elif case == 'frozen': samples[-2]['time_s'] = samples[-1]['time_s']
    elif case == 'gap': samples[2]['time_s'] = .1
    elif case == 'epoch': samples[2]['epoch'] = 'another'
    elif case == 'identity': samples[2]['model_sha256'] = 'another'
    elif case == 'no_release': samples[2]['open_since_s'] = None
    elif case == 'release_late': samples[2]['open_since_s'] = .3
    elif case == 'outside': samples[2]['objects']['cube']['lower_m'][0] = .149
    elif case == 'final_outside': samples[-1]['final_footprints']['cube']['upper_m'][0] = .241
    elif case == 'closed': samples[2]['gripper_open_fraction'] = .97
    elif case == 'no_support': samples[2]['objects']['cube']['support_up_n'] = 0.
    elif case == 'robot_touch': samples[2]['objects']['cube']['other_loaded_contacts'] = 1
    elif case == 'moving': samples[2]['objects']['cube']['linear_speed_m_s'] = .021
    elif case == 'spinning': samples[2]['objects']['cube']['angular_speed_rad_s'] = .201
    elif case == 'drifting': samples[2]['objects']['cube']['position_m'][0] += .006
    elif case == 'unknown_force': samples[2]['objects']['cube']['support_up_n'] = float('nan')
    assert check(samples)['status'] == expected


# Static fixture consumes captured model coordinates, never steps a solver.
from test_mujoco_withdrawal import model_runtime as model_runtime, captured, CAPTURE  # noqa: E402,F401


def test_native_history_model_admitted_without_advancing_or_rewriting_state(model_runtime):
    rt = model_runtime
    history = rt.arm.raw.world.placement_history
    data = rt.arm.raw.world.data
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    lower, upper = history.geometry()
    assert lower.shape == upper.shape == (history.model.ngeom, 3)
    assert history.support_names == ['floor', 'demo_floor']
    np.testing.assert_array_equal(before[0], data.qpos)
    np.testing.assert_array_equal(before[1], data.qvel)
    np.testing.assert_array_equal(before[2], data.ctrl)
    assert data.time == before[3] == 0.
    with pytest.raises(ValueError, match='current native state'):
        history.read('red cube')


def test_model_mutation_or_data_rebind_invalidates_history(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    history.model.geom_size[-1, 0] += .001
    with pytest.raises(ValueError, match='identity changed'):
        history.guard()


def test_history_reset_forgets_release_and_prefix(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    epoch = history.epoch
    history.rows.extend(rows()); history.confirmed_prefix.add('red_cube'); history.closed_seen = True
    history.reset()
    assert history.epoch != epoch and not history.rows and not history.confirmed_prefix
    assert not history.closed_seen and history.open_since is None


def test_unbound_region_reader_never_falls_back_to_point_verdict(tmp_path):
    from cascade.sim.mujoco_placement import read_placement
    report = read_placement(str(tmp_path/'absent.xml'), {'mj_delivery_area': REGION},
                            'cube', 'drop zone')
    assert report['status'] == 'unverified'


def test_geometry_extraction_matches_original_ordinary_skill_ast(model_runtime):
    """Execute the exact former pure block on the captured held state."""
    import ast
    from cascade.grasping.obb_grasp import _yaw_rotation
    from cascade.skills.runtime import _PreCarryLiftError, _PostPlaceRetreatPlanError
    from cascade.skills.place_geometry import plan
    from cascade.types import make_transform, SkillError
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    captured(rt, CAPTURE[0]['rows'][0])
    q = rt.arm.get_state().q
    target = np.array([.20, -.12, .045])
    # Pinned pre-extraction source tracked in the parent commit, not a copy
    # hand-edited to mirror the replacement helper.
    fixture = json.loads((Path(__file__).parent/'fixtures/place_geometry_original.json').read_text())
    tree = ast.parse(fixture['source'])
    scope = dict(self=rt, q_now=q, target=target.copy(), x=.20, y=-.12, release_z=.045,
                 z_cap=.045, gcfg=rt.cfg.grasp, table_z=0., np=np, make_transform=make_transform,
                 _yaw_rotation=_yaw_rotation, _PreCarryLiftError=_PreCarryLiftError,
                 _PostPlaceRetreatPlanError=_PostPlaceRetreatPlanError, SkillError=SkillError)
    exec(compile(tree, '<pinned-ordinary-IK>', 'exec'), scope)
    got = plan(rt, q, target, x=.20, y=-.12, release_z=.045, z_cap=.045)
    for name in ('pre', 'low'):
        np.testing.assert_array_equal(getattr(got, name).q, scope[name].q)
    np.testing.assert_array_equal(got.rotation, scope['R'])
    np.testing.assert_array_equal(got.hover, scope['hover'])
    assert rt.arm.raw.world.data.time == 0.


def test_region_full_path_proposal_preserves_the_live_captured_world(model_runtime, tmp_path):
    from cascade.skills.mujoco_region import select
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    data = rt.arm.raw.world.data
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time)
    result = select(rt)
    (tmp_path/'static-plan.json').write_text(json.dumps(result, indent=2)+'\n')
    assert result['destination_kind'] == 'configured_region'
    assert result['physical_task_verdict'] is False
    assert result['target'] != rt.cfg.grasp.get('drop_zone')
    np.testing.assert_array_equal(before[0], data.qpos)
    np.testing.assert_array_equal(before[1], data.qvel)
    np.testing.assert_array_equal(before[2], data.ctrl)
    assert data.time == before[3] == 0.


def test_missing_independent_runtime_reader_is_explicitly_unverified(model_runtime):
    from cascade.skills.runtime import SkillRuntime
    rt = model_runtime
    rt.arm_rig = None
    rt._object_pose = lambda _: [.2, -.12, .02]
    report = SkillRuntime._verify_configured_placement(rt, 'red cube', None)
    assert report['status'] == 'unverified' and report['measured']['destination'] == 'drop zone'


def test_history_rejects_integrated_geometry_mixed_with_previous_contacts(model_runtime):
    """No solve: deliberate contradictory phase fixture must poison the channel."""
    rt = model_runtime
    history, data = rt.arm.raw.world.placement_history, rt.arm.raw.world.data
    before = {'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'time_s': 0.}
    # Derived geom positions still describe the original qpos. Neither qpos
    # edit nor the made-up clock is presented as a physical measurement.
    before['qpos'][rt.arm.raw._qadr[0]] += .1
    data.time = float(history.model.opt.timestep)
    history.capture(1, before)
    assert not history.rows and 'solved geometry differs' in history.error


def test_native_model_serialization_binds_actuator_and_shape_changes(model_runtime):
    from cascade.sim.mujoco_placement import model_digest
    history = model_runtime.arm.raw.world.placement_history
    assert model_digest(history.model, history.descriptor) == history.identity
    history.model.actuator_gainprm[0, 0] += 1
    with pytest.raises(ValueError, match='identity changed'):
        history.guard()


def test_data_identity_not_just_equal_coordinates_is_required(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    history.world.data = history.mj.MjData(history.model)
    with pytest.raises(ValueError, match='identity changed'):
        history.guard()


def test_read_rejects_old_batch_after_new_state_write(model_runtime):
    from cascade.sim.mujoco_placement import state_digest
    history = model_runtime.arm.raw.world.placement_history
    history.rows.append({'time_s': 0., 'final_state_time_s': 0., 'state_sha256': state_digest(history.data)})
    history.data.qpos[0] += .001
    with pytest.raises(ValueError, match='current native state'):
        history.read('red cube')


def test_region_rejects_cancelled_generation_without_any_motion(model_runtime):
    from cascade.skills.mujoco_region import select
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    rt.arm.harness.halt('operator stop')
    q = rt.arm.raw.world.data.qpos.copy()
    with pytest.raises(Exception, match='halt|cancel|stop'):
        select(rt)
    np.testing.assert_array_equal(q, rt.arm.raw.world.data.qpos)
    assert rt.arm.raw.world.data.time == 0.


def test_region_shared_deadline_cannot_be_renewed_per_candidate(model_runtime, monkeypatch):
    from cascade.skills import mujoco_region
    from types import SimpleNamespace
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    clock = iter([0., 3.1])
    # Module binding only; no production/global clock monkeypatch.
    monkeypatch.setattr(mujoco_region, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(Exception, match='deadline expired'):
        mujoco_region.select(rt)
    assert rt.arm.raw.world.data.time == 0.


def test_region_unreachable_full_area_fails_without_move_or_retry(model_runtime):
    from cascade.skills.mujoco_region import select
    rt = model_runtime
    rt._tool_axis_order = 'open_down'
    rt._supported_release_height = lambda support, fallback: fallback
    captured(rt, CAPTURE[0]['rows'][0])
    # A synthetic unreachable request preserves the actual model and grasp;
    # no safety/geometry is replaced by an always-accepting test double.
    value = dict(REGION, bounds_xy_m=[[1., 1.], [1.09, 1.08]])
    rt.cfg.arm._data['mj_delivery_area'] = value
    from cascade.sim.mujoco_placement import PlacementHistory
    rt.arm.raw.world.placement_history = PlacementHistory(rt.arm.raw.world, rt.arm.raw)
    with pytest.raises(Exception, match='no feasible|deadline'):
        select(rt)
    assert rt.arm.raw.world.data.time == 0.


def test_existing_step_writer_captures_only_last_solve_input_and_no_extra_steps():
    from types import SimpleNamespace
    from cascade.control.mujoco_arm import _MjcEngine
    captured_rows, calls = [], []
    data = SimpleNamespace(qpos=np.array([0.]), qvel=np.array([.2]), time=0.)
    def synthetic_step(model, actual):
        calls.append(actual.time)
        actual.qpos += 1.
        actual.qvel += .1
        actual.time += .005
    history = SimpleNamespace(capture=lambda count, before: captured_rows.append((count, before)))
    engine = SimpleNamespace(data=data, model=object(), _mj=SimpleNamespace(mj_step=synthetic_step),
                             _world=SimpleNamespace(placement_history=history), _viewer=None)
    _MjcEngine.step(engine, 4)
    assert len(calls) == 4 and len(captured_rows) == 1
    count, before = captured_rows[0]
    assert count == 4 and before['time_s'] == .015
    np.testing.assert_array_equal(before['qpos'], [3.])
    np.testing.assert_allclose(before['qvel'], [.5], atol=1e-15)
    assert data.qpos[0] == 4. and data.time == .02
    _MjcEngine.step(engine, 0)
    assert len(calls) == 4 and len(captured_rows) == 1


def test_changed_profile_binding_does_not_relabel_old_history(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    model_runtime.cfg.arm._data['mj_delivery_area'] = dict(REGION, name='another')
    with pytest.raises(ValueError, match='profile binding changed'):
        history.guard()


def test_reader_rechecks_previously_confirmed_prefix_and_retains_intervening_fault(model_runtime):
    from cascade.sim.mujoco_placement import state_digest
    from copy import deepcopy
    history = model_runtime.arm.raw.world.placement_history
    samples = rows()
    for row in samples:
        row['epoch'], row['model_sha256'] = history.epoch, history.identity
        obj, footprint = row['objects'].pop('cube'), row['final_footprints'].pop('cube')
        row['objects'] = {'red_cube': deepcopy(obj), 'blue_cube': deepcopy(obj)}
        row['final_footprints'] = {'red_cube': deepcopy(footprint), 'blue_cube': deepcopy(footprint)}
    # Explicit synthetic history for the independent evaluator; no claim that
    # these colocated fixtures have been produced by a physical solve.
    history.data.time = samples[-1]['final_state_time_s']
    samples[-1]['state_sha256'] = state_digest(history.data)
    history.rows.extend(samples)
    first, _ = history.read('red cube')
    assert first['status'] == 'confirmed' and history.confirmed_prefix == {'red_cube'}
    history.prefix_faults['red_cube'] = {'step': 2, 'measurement': {'other_loaded_contacts': 1}}
    # Repeating the same passive read must not replace its old obligation,
    # even though the most recent synthetic settling window is healthy.
    repeated, _ = history.read('red cube')
    assert repeated['status'] == 'refuted'
    assert repeated['measured']['first_disturbance']['step'] == 2
    assert history.prefix_faults['red_cube']['step'] == 2
    assert history.confirmed_prefix == {'red_cube'}
    second, _ = history.read('blue cube')
    assert second['status'] == 'refuted'
    assert second['measured']['previously_confirmed']['red_cube']['measured']['first_disturbance']['step'] == 2
    assert 'blue_cube' not in history.confirmed_prefix


def test_current_final_pose_cannot_borrow_support_from_previous_solve():
    samples = rows()
    # Even though all solve-phase samples are supported and inside, the
    # separately bound integrated footprint must also remain inside now.
    samples[-1]['final_footprints']['cube']['lower_m'][1] = -.161
    assert check(samples)['status'] == 'refuted'


@pytest.mark.parametrize('field,value', [('linear_speed_m_s', .03), ('angular_speed_rad_s', .3)])
def test_final_integration_impulse_does_not_borrow_rest_from_solve_input(field, value):
    samples = rows()
    samples[-1]['final_footprints']['cube'][field] = value
    assert check(samples)['status'] == 'refuted'


def test_final_integrated_closing_cannot_borrow_previous_opening():
    samples = rows()
    samples[-1]['final_gripper_open_fraction'] = .97
    assert check(samples)['status'] == 'refuted'


@pytest.mark.parametrize('field,value', [('_qadr', [4, 3, 2, 1, 0]), ('_dadr', [4, 3, 2, 1, 0]),
    ('_grip_qadr', 0), ('_grip_open', .59), ('_grip_closed', -.14),
    ('_joint_names', ['a', 'b', 'c', 'd', 'e']), ('_grip_joint', 'different')])
def test_current_driver_coordinate_binding_cannot_drift(model_runtime, field, value):
    history = model_runtime.arm.raw.world.placement_history
    setattr(model_runtime.arm.raw, field, value)
    with pytest.raises(ValueError, match='profile binding changed'):
        history.guard()


def test_active_arm_contact_with_zero_force_still_refutes_release():
    samples = rows()
    samples[2]['objects']['cube']['arm_active_contacts'] = 1
    # No loaded-arm force is required for this distinct contact veto.
    assert samples[2]['objects']['cube']['other_loaded_contacts'] == 0
    assert check(samples)['status'] == 'refuted'


def retirement_fixture(history):
    """Synthetic history only; exercised by the real goal-ledger predicate."""
    from copy import deepcopy
    from cascade.sim.mujoco_placement import state_digest
    samples = rows()
    for row in samples:
        row['epoch'], row['model_sha256'] = history.epoch, history.identity
        obj, footprint = row['objects'].pop('cube'), row['final_footprints'].pop('cube')
        for value in (obj, footprint):
            value['lower_m'][0] += .1; value['upper_m'][0] += .1
        obj['position_m'][0] += .1
        row['objects'], row['final_footprints'] = {'red_cube': obj}, {'red_cube': footprint}
    history.data.time = samples[-1]['final_state_time_s']
    qa = history.bodies['red_cube'][1]
    history.data.qpos[qa:qa+3] = samples[-1]['objects']['red_cube']['position_m']
    samples[-1]['state_sha256'] = state_digest(history.data)
    history.rows.extend(samples)
    history.confirmed_prefix.add('red_cube')
    history.prefix_faults['red_cube'] = {'step': 2, 'reason': 'old goal was disturbed'}
    request = {'object': 'red_cube', 'target_xy_m': [.2975, -.1225], 'epoch': history.epoch,
               'model_sha256': history.identity, 'request_step': 0, 'request_final_time_s': 0., 'tool': 'place_at',
               'generation': 0, 'cancellation_token': 0}
    result = {'ok': True, 'postcondition': {'status': 'confirmed', 'channel': 'physics'}}
    return deepcopy(request), result


def test_explicit_verified_point_retires_old_area_goal_but_retains_its_fault(model_runtime):
    history = model_runtime.arm.raw.world.placement_history
    request, result = retirement_fixture(history)
    retired = history.retire_explicit(request, result, harness=model_runtime.arm.harness)
    assert retired['retired'] and 'red_cube' not in history.confirmed_prefix
    assert retired['previous_disturbance']['reason'] == 'old goal was disturbed'
    assert len(history.retired_obligations) == 1
    assert retired['verification']['measured']['containment_required'] is False


@pytest.mark.parametrize('case', ['failed', 'unknown', 'actor_only', 'epoch', 'stale', 'old_release', 'no_support'])
def test_failed_or_unobserved_point_never_clears_old_region_goal(model_runtime, case):
    history = model_runtime.arm.raw.world.placement_history
    request, result = retirement_fixture(history)
    if case == 'failed': result['ok'] = False
    elif case == 'unknown': result['postcondition']['status'] = 'unverified'
    elif case == 'actor_only': result['postcondition']['channel'] = 'belief'
    elif case == 'epoch': request['epoch'] = 'other'
    elif case == 'stale': history.data.time += .005
    elif case == 'old_release': request['request_final_time_s'] = .001
    elif case == 'no_support': history.rows[-2]['objects']['red_cube']['support_up_n'] = 0.
    result = history.retire_explicit(request, result, harness=model_runtime.arm.harness)
    assert result['retired'] is False
    assert 'red_cube' in history.confirmed_prefix and 'red_cube' in history.prefix_faults
    assert not history.retired_obligations


def test_stop_between_point_result_and_retirement_keeps_pending_goal(model_runtime):
    from cascade.skills.mujoco_region import finish_explicit
    rt = model_runtime
    history = rt.arm.raw.world.placement_history
    request, result = retirement_fixture(history)
    request['generation'] = rt.arm.harness._halt_generation
    rt.arm.harness.halt('after motion result, before ledger retirement')
    finish_explicit((history, request, rt.arm.harness), result)
    assert result['region_obligation']['retired'] is False
    assert 'red_cube' in history.confirmed_prefix and not history.retired_obligations


@pytest.mark.parametrize('stop', ['halt', 'estop', 'estop_then_reset'])
def test_stop_returns_during_slow_audit_and_cancels_ledger_commit(model_runtime, monkeypatch, stop):
    import threading
    from cascade.sim import mujoco_placement
    from cascade.skills.mujoco_region import finish_explicit
    history = model_runtime.arm.raw.world.placement_history
    harness = model_runtime.arm.harness
    request, result = retirement_fixture(history)
    entered, release, stop_returned = threading.Event(), threading.Event(), threading.Event()
    original = mujoco_placement.audit

    def slow_audit(*args, **kwargs):
        checked = original(*args, **kwargs)
        assert checked['status'] == 'confirmed'
        entered.set()
        assert release.wait(2), 'bounded test audit was not released'
        return checked

    def stop_now():
        if stop == 'halt':
            harness.halt('during independent observation')
        else:
            harness.estop('during independent observation')
            if stop == 'estop_then_reset':
                harness.reset_estop()
        stop_returned.set()

    monkeypatch.setattr(mujoco_placement, 'audit', slow_audit)
    worker = threading.Thread(target=finish_explicit, args=((history, request, harness), result))
    stopper = threading.Thread(target=stop_now)
    worker.start()
    try:
        assert entered.wait(2)
        stopper.start()
        # The audit holds world.lock but must not hold the stop-state lock.
        assert stop_returned.wait(1), 'stop waited for the slow observation'
        assert worker.is_alive() and not history.retired_obligations
        assert 'red_cube' in history.confirmed_prefix and 'red_cube' in history.prefix_faults
    finally:
        release.set()
        worker.join(2)
        if stopper.ident is not None:
            stopper.join(2)
    assert not worker.is_alive() and not stopper.is_alive()
    assert result['region_obligation']['retired'] is False
    assert 'cancelled' in result['region_obligation']['reason']
    assert 'red_cube' in history.confirmed_prefix and 'red_cube' in history.prefix_faults
    assert not history.retired_obligations
    if stop == 'estop_then_reset':
        assert not harness.estopped and harness._halt_generation == request['generation']
        assert harness._observation_cancel_generation > request['cancellation_token']


from test_placement_reporting import runtime as runtime  # noqa: E402,F401


def test_only_external_point_dispatch_can_replace_region_goal(request, monkeypatch):
    from cascade.skills import mujoco_region
    rt = request.getfixturevalue('runtime')
    seen = []
    token = object()
    def context(arm, label, args):
        seen.append(('context', label, dict(args)))
        return token
    def finish(value, result):
        assert value is token and result['postcondition']['status'] == 'confirmed'
        seen.append(('finish', result['postcondition']['channel']))
        result['region_obligation'] = {'retired': True, 'synthetic_wiring_control': True}
    monkeypatch.setattr(mujoco_region, 'explicit_context', context)
    monkeypatch.setattr(mujoco_region, 'finish_explicit', finish)
    rt.held_object = 'red cube'
    rt.attach_verifier(object_pose=lambda _: [.3, -.12, .02])
    rt.skill_place_at = lambda **_: {'placed': 'red cube', 'at': [.3, -.12, .02]}
    direct = rt.execute('place_at', {'x': .3, 'y': -.12})
    assert seen == [('context', 'red cube', {'x': .3, 'y': -.12}), ('finish', 'physics')]
    assert direct['region_obligation']['synthetic_wiring_control']
    seen.clear()
    rt.skill_pick_and_place = lambda **_: rt.skill_place_at(x=.3, y=-.12)
    rt.execute('pick_and_place', {'object': 'red cube'})
    assert seen == []


def test_plain_robot_scene_can_connect_but_cannot_invent_region_objects():
    from cascade.config import load_profile
    from cascade.control.mujoco_arm import MujocoArm
    from cascade.control.kinematics import Kinematics
    pytest.importorskip('mujoco')
    pytest.importorskip('pinocchio')
    if not (Path(__file__).resolve().parents[1]/'assets/mjcf/so101/scene.xml').exists():
        pytest.skip('requires scripts/fetch_robot_assets.py so101')
    cfg = load_profile('arms', 'so101_mujoco')
    cfg._data['mj_prop_from_camera'] = False
    kin = Kinematics(cfg.model, cfg.ee_frame, 5, cfg.get('joint_signs'))
    arm = MujocoArm(cfg, kin)
    try:
        arm.connect()
        history = arm.world.placement_history
        assert not history.bodies and not history.objects and arm.world.data.time == 0.
        with pytest.raises(ValueError, match='current native state'):
            history.read('invented cube')
    finally:
        arm.disconnect()
