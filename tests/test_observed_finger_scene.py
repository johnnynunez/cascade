"""Observed surfaces and calibrated fingers; no bridge, truth poses or robot IO."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.control.motion_profile import nominal_profile
from cascade.grasping.observed_scene import FingerGeometry, ObservedScene, ObservedFingerGate, ROOT
from cascade.types import Frame, RobotState, SafetyViolation


@pytest.fixture(scope="module")
def geometry():
    return FingerGeometry()


def snapshot(q=(.05, .05)):
    return dict(version=1, names=['joint_left', 'joint_right'], position_m=list(q),
                lower_m=[0., 0.], upper_m=[float(np.float32(.05))] * 2)


def scene_at(point, *, target=True):
    capture = dict(backend='isaac', source=('fake', 1), camera='cam', t=1., proprioception=dict(
        version=1, backend='isaac', robot_id='/robot', joint_convention='asset', producer_epoch='epoch',
        t=1., time_source='physics_loop_monotonic'))
    frame = Frame(np.zeros((1, 2, 3), np.uint8), np.ones((1, 2)), np.eye(3), depth_source='sensor',
                  capture=capture, robot_mask=np.zeros((1, 2), bool))
    mask = np.array([[target, True]])
    T = np.eye(4); T[:3, 3] = np.array(point) - [0, 0, 1]
    return frame, mask, T


def construct(frame, mask, T):
    return ObservedScene(frame, mask, T, source=('fake', 1), robot_id='/robot', epoch='epoch')


def test_artifact_covers_every_original_vertex_and_triangle(geometry, monkeypatch):
    spec = importlib.util.spec_from_file_location('geometry_builder', ROOT / 'scripts/build_gripper_scene_geometry.py')
    build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)
    source_components = []
    convex_hull = build.ConvexHull
    def record_source_points(points):
        source_components.append(points.copy())
        return convex_hull(points)
    monkeypatch.setattr(build, 'ConvexHull', record_source_points)
    rebuilt = build.build()
    assert [sum(c['triangles'] for c in f['components']) for f in rebuilt['fingers']] == [17246, 17246]
    assert [sum(c['vertices'] for c in f['components']) for f in rebuilt['fingers']] == [8649, 8649]
    assert [len(f['components']) for f in rebuilt['fingers']] == [8, 8]
    assert [len(f['components']) for f in geometry.fingers] == [8, 8]
    assert max(c['coverage_error_m'] for f in rebuilt['fingers'] for c in f['components']) <= 1e-12
    assert len(source_components) == 16
    sources = iter(source_components)
    for finger, rebuilt_finger in zip(geometry.fingers, rebuilt['fingers']):
        for part, rebuilt_part in zip(finger['components'], rebuilt_finger['components']):
            assert part['vertices'] == rebuilt_part['vertices']
            assert part['triangles'] == rebuilt_part['triangles']
            # Qhull facet order and roundoff vary by platform. Check the shipped
            # halfspaces against every transformed original STL vertex instead
            # of comparing two ordered floating-point serialization results.
            points = next(sources)
            planes = np.asarray(part['planes'])
            np.testing.assert_allclose(np.linalg.norm(planes[:, :3], axis=1), 1., rtol=0, atol=1e-12)
            support = np.max(points @ planes[:, :3].T + planes[:, 3], axis=0)
            assert np.max(support) <= 1e-12  # Convexity also covers whole triangles.
            assert np.min(support) >= -3e-12  # Every plane remains tight to its source.
            for key in ('min_m', 'max_m'):
                np.testing.assert_allclose(part[key], rebuilt_part[key], rtol=0, atol=1e-12)


@pytest.mark.parametrize('target', [True, False])
def test_target_and_neighbor_surfaces_both_veto(geometry, target):
    envelopes = geometry.envelopes(np.array([.0499, .0499]), np.array([.0501, .0501]))
    # Obtain an interior mesh point from a small convex component's vertices.
    part = envelopes[1]
    point = part[5]
    scene = construct(*scene_at(point, target=target))
    hit = scene.conflict(np.eye(4), envelopes)
    assert hit is not None and hit['finger'] == 'joint_left'
    assert hit['surface'] == ('target' if target else 'other observed surface')
    close_hit = scene.conflict(np.eye(4), envelopes, allow_target_contact=True)
    assert (close_hit is None) == target
    # Closure-only contact permission cannot mutate later approach checks.
    assert scene.conflict(np.eye(4), envelopes) == hit


def test_robot_mask_excludes_only_exact_self_pixels(geometry):
    env = geometry.envelopes(np.ones(2)*.05, np.ones(2)*.05)
    frame, mask, T = scene_at(env[1][5])
    frame.robot_mask[0, 0] = True
    scene = construct(frame, mask, T)
    assert scene.conflict(np.eye(4), env) is None
    assert scene.receipt['observed_points'] == 1


@pytest.mark.parametrize('change', ['epoch', 'source', 'robot', 'timestamp', 'mask', 'depth', 'transform'])
def test_capture_mismatch_fails_closed(change):
    frame, mask, T = scene_at([0, 0, 0])
    if change == 'epoch': frame.capture['proprioception']['producer_epoch'] = 'other'
    if change == 'source': frame.capture['source'] = ('fake', 2)
    if change == 'robot': frame.capture['proprioception']['robot_id'] = '/other'
    if change == 'timestamp': frame.capture['proprioception']['t'] = 2.
    if change == 'mask': mask = None
    if change == 'depth': frame.depth_source = 'plane'
    if change == 'transform': T[0, 0] = 2.
    with pytest.raises(SafetyViolation): construct(frame, mask, T)


def test_actual_torch_mask_uses_explicit_host_safety_boundary():
    torch = pytest.importorskip('torch')
    frame, mask, T = scene_at([0, 0, 0])
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        scene = construct(frame, torch.as_tensor(mask, device=device), T)
        assert scene.is_target.tolist() == [True, True]


def test_uncertainty_expands_geometry_broadphase_and_feedback_interval(geometry):
    # Empty nearby scene is valid; the target is observed far from the robot.
    scene = construct(*scene_at([10, 10, 10]))
    kin = SimpleNamespace(fk=lambda q: np.eye(4))
    clock = dict(version=1, engine='physx', clock='SimulationManager', source=('fake', 1),
                 robot_id='/robot', epoch='epoch', sim_time=0., physics_step=0, physics_dt_s=.01)
    state = RobotState(np.zeros(1), physics_clock=clock, gripper_joints=snapshot((.05000001, .04999998)))
    gate = ObservedFingerGate(scene, geometry, kin, state, uncertainty_m=.0001)
    np.testing.assert_allclose(gate.lower, [.0499, .04989998])
    np.testing.assert_allclose(gate.upper, [.05010001, .0501])
    exact = geometry.envelopes(geometry.upper, geometry.upper)
    assert all(a[6] > b[6] for a,b in zip(gate.envelopes, exact))
    state.physics_clock = {**clock, 'sim_time': .01, 'physics_step': 1}
    state.gripper_joints = snapshot((.0498, .05))
    with pytest.raises(SafetyViolation, match='opening interval'): gate.feedback(state)
    # No adaptive expansion to accommodate a physical deflection.
    assert gate.lower[0] == pytest.approx(.0499)


def test_profile_checks_exact_legacy_actual_and_lookahead_points():
    gate = ObservedFingerGate.__new__(ObservedFingerGate)
    seen = []
    gate.pose = lambda q: seen.append(tuple(q)) or None
    start, goal = np.array([0.]), np.array([1.])
    assert gate.profile(start, goal, 1., 30.) is None
    expected = [tuple(q) for w in nominal_profile(start, goal, 1., 30.) for edge in w.checks for q in edge[:2]]
    assert seen == expected
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2: raise SafetyViolation('cancelled')
    with pytest.raises(SafetyViolation, match='cancelled'):
        gate.profile(start, goal, 1., 30., check=cancel)


def test_geometry_overshoot_is_not_clipped_and_nonrigid_tcp_is_rejected(geometry):
    np.testing.assert_equal(geometry.strokes(snapshot((.051, .049))), [.051, .049])
    scene = construct(*scene_at([10, 10, 10]))
    T = np.eye(4); T[0, 0] = 2.
    with pytest.raises(SafetyViolation, match='rigid'):
        scene.conflict(T, geometry.envelopes(geometry.lower, geometry.upper))


@pytest.mark.parametrize('changed', ['q', 'finger', 'epoch'])
def test_snapshot_cannot_change_without_new_physics_step(geometry, changed):
    scene = construct(*scene_at([10, 10, 10]))
    kin = SimpleNamespace(fk=lambda q: np.eye(4))
    clock = dict(version=1, engine='physx', clock='SimulationManager', source=('fake', 1),
                 robot_id='/robot', epoch='epoch', sim_time=0., physics_step=0, physics_dt_s=.01)
    state = RobotState(np.zeros(1), physics_clock=clock, gripper_joints=snapshot())
    gate = ObservedFingerGate(scene, geometry, kin, state, uncertainty_m=.0001)
    if changed == 'q': state.q[0] += .00001
    if changed == 'finger': state.gripper_joints['position_m'][0] += .00001
    if changed == 'epoch': state.physics_clock = {**clock, 'epoch': 'restarted'}
    with pytest.raises(SafetyViolation, match='changed|epoch'):
        gate.feedback(state)


def closing_gate(geometry):
    scene = construct(*scene_at([10, 10, 10]))
    clock = dict(version=1, engine='physx', clock='SimulationManager', source=('fake', 1),
                 robot_id='/robot', epoch='epoch', sim_time=0., physics_step=0, physics_dt_s=.01)
    state = RobotState(np.zeros(1), physics_clock=clock, gripper_joints=snapshot())
    gate = ObservedFingerGate(scene, geometry, SimpleNamespace(fk=lambda q: np.eye(4)), state, uncertainty_m=.0001)
    return gate, state


def test_closing_envelope_contains_independent_finger_strokes_and_padding(geometry):
    gate, _ = closing_gate(geometry)
    for strokes in ([0., .05], [.05, 0.], [.013, .042], [-.0001, .0501]):
        exact = geometry.envelopes(np.asarray(strokes), np.asarray(strokes))
        for outer, inner in zip(gate.closing_envelopes, exact):
            assert np.all(outer[3] <= inner[3] + 1e-12)
            assert np.all(outer[4] >= inner[4] - 1e-12)
            np.testing.assert_allclose(outer[2][:,:3], inner[2][:,:3])
            assert np.all(outer[2][:,3] <= inner[2][:,3] + 1e-12)


def test_second_close_accepts_new_partly_closed_snapshot_but_not_open_motion(geometry):
    gate, state = closing_gate(geometry)
    state.physics_clock = {**state.physics_clock, 'sim_time': .01, 'physics_step': 1}
    state.gripper_joints = snapshot((.025, .019))
    gate.require_closing(state)
    gate.require_closing(state)  # identical same-step snapshot is legitimate
    with pytest.raises(SafetyViolation, match='opening interval'): gate.feedback(state)


@pytest.mark.parametrize('changed', ['q', 'finger', 'epoch', 'nan', 'outside'])
def test_close_keeps_atomic_epoch_finite_and_fixed_bounds_contract(geometry, changed):
    gate, state = closing_gate(geometry)
    if changed == 'q': state.q[0] += .00001
    if changed == 'finger': state.gripper_joints['position_m'][0] -= .00001
    if changed == 'epoch': state.physics_clock = {**state.physics_clock, 'epoch': 'new'}
    if changed in ('nan', 'outside'):
        state.physics_clock = {**state.physics_clock, 'sim_time': .01, 'physics_step': 1}
        state.gripper_joints = snapshot((np.nan if changed=='nan' else -.001, .02))
    before = gate.closing_lower.copy()
    with pytest.raises(SafetyViolation): gate.require_closing(state)
    np.testing.assert_array_equal(gate.closing_lower, before)


def test_recorded_observed_neighbor_point_is_clear_open_but_vetoes_closure(geometry):
    from cascade.config import load_demo_config
    from cascade.control.kinematics import Kinematics
    cfg = load_demo_config(arm='isaac_kitchen_gpu', camera='mock', llm='mock').arm
    kin = Kinematics(cfg.model, cfg.ee_frame, 6, cfg.joint_signs)
    # One retained RGBD point, pixel (223,751), outside the target mask. Its
    # semantic label/ground-truth pose is not supplied to the predicate.
    scene = construct(*scene_at([.25285849249653447, .12765121104416072, .060234708493281386], target=False))
    _, state = closing_gate(geometry)
    state.q = np.asarray(cfg.home_q)
    gate = ObservedFingerGate(scene, geometry, kin, state, uncertainty_m=.0001)
    q = np.array([-.4454485906686117, 1.7624410876478933, 1.1331546533194867,
                  -1.1452525526669397, .15528819138473887, -1.5643258051539406])
    assert gate.pose(q) is None
    assert gate.closing_pose(q)['surface'] == 'other observed surface'
    state.physics_clock = {**state.physics_clock, 'sim_time': .01, 'physics_step': 1}
    state.q = q
    with pytest.raises(SafetyViolation, match='closing fingers'): gate.require_closing(state)


def test_runtime_factory_binds_source_without_transport_reads():
    from cascade.config import load_demo_config
    from cascade.control.isaac_arm import IsaacArm
    from cascade.grasping.observed_scene import for_runtime
    cfg = load_demo_config(arm='isaac_kitchen_gpu', camera='isaac', llm='mock')
    raw = IsaacArm(cfg.arm)
    raw._client = SimpleNamespace(_addr=('fake', 1))  # no state method: IO would fail
    frame, mask, T = scene_at([10, 10, 10])
    frame.T_base_cam = T
    robot = cfg.arm.bridge_robot_id
    frame.capture['proprioception'].update(q=[0.] * 6, robot_id=robot, gripper_joints=snapshot())
    state = RobotState(np.zeros(6), physics_clock=dict(version=1, engine='physx', clock='SimulationManager',
                      source=('fake', 1), robot_id=robot, epoch='epoch', sim_time=0., physics_step=0,
                      physics_dt_s=.01), gripper_joints=snapshot())
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=raw), cfg=cfg,
                              kin=SimpleNamespace(ee_frame='gripper_end', fk=lambda q: np.eye(4)))
    fix = SimpleNamespace(detection=SimpleNamespace(mask=mask))
    gate = for_runtime(runtime, frame, fix, state)
    assert gate.scene.identity == (('fake', 1), robot, 'epoch')
    frame.capture['proprioception']['producer_epoch'] = 'other'
    with pytest.raises(SafetyViolation, match='epoch'):
        for_runtime(runtime, frame, fix, state)
