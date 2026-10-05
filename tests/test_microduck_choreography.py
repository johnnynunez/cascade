"""Showcase choreography: presenter path, follow formation, scripted controller twists, spawn layout."""
import json
import math

import pytest

from cascade.sim.microduck_choreography import (Choreography, Formation, PresenterPath, WalkCycle,
                                                DEFAULT_LIGHTING)
from cascade.sim.microduck_shared_native import CHOREOGRAPHY_MIN_SEPARATION_M, placements


def _state(x, y, yaw):
    return {'position': [x, y, .12], 'orientation_wxyz': [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
            'balance_active': True}


def _choreo_dict(tmp_path, **over):
    data = {'presenter': {'waypoints': [[-4., 2.], [0., 2.], [4., 2.5]], 'speed_m_s': .18, 'start_delay_s': 3.},
            'formation': {'slots': [[-.8, -.9], [-.8, -.3], [-.8, .3], [-.8, .9], [-1.4, -.9], [-1.4, -.3],
                                    [-1.4, .3], [-1.4, .9], [-2., -.9], [-2., -.3], [-2., .3], [-2., .9]]}}
    data.update(over)
    path = tmp_path / 'choreo.json'
    path.write_text(json.dumps(data))
    return path


# --- presenter path / walk cycle -------------------------------------------------------------

def test_presenter_path_holds_then_walks_at_constant_speed_and_stops_at_the_end():
    p = PresenterPath(((0., 0.), (4., 0.), (4., 3.)), speed_m_s=.2, start_delay_s=2.)
    assert p.length() == 7. and p.duration_s() == 2. + 35.
    x, y, h, d, walking = p.pose_at(1.)
    assert (x, y, d, walking) == (0., 0., 0., False) and h == 0.
    x, y, h, d, walking = p.pose_at(12.)      # 10 s walking -> 2 m along +x
    assert (round(x, 9), y, round(d, 9), walking) == (2., 0., 2., True) and h == 0.
    x, y, h, d, walking = p.pose_at(2. + 25.)  # 5 m: 1 m into the second leg (+y)
    assert (round(x, 9), round(y, 9), round(d, 9), walking) == (4., 1., 5., True)
    assert math.isclose(h, math.pi / 2)
    x, y, h, d, walking = p.pose_at(100.)
    assert (x, y, d, walking) == (4., 3., 7., False) and math.isclose(h, math.pi / 2)


def test_presenter_pauses_at_intermediate_waypoints_and_stands_while_paused():
    p = PresenterPath(((0., 0.), (2., 0.), (4., 0.)), speed_m_s=.2, start_delay_s=1., dwell_s=(5.,))
    assert p.duration_s() == 1. + 20. + 5.
    assert p.pose_at(1. + 10.)[3:] == (2., False)       # arrived at the pause
    assert p.pose_at(1. + 12.)[3:] == (2., False)       # still paused
    x, _, _, d, walking = p.pose_at(1. + 15. + 5.)      # 5 s after the pause: 1 m into leg two
    assert math.isclose(x, 3.) and math.isclose(d, 3.) and walking
    assert p.pose_at(1. + 9.)[4] is True
    with pytest.raises(ValueError, match='at most one pause'):
        PresenterPath.from_dict({'waypoints': [[0., 0.], [1., 0.]], 'speed_m_s': .2, 'dwell_s': [1.]})


def test_walk_cycle_is_symmetric_and_still_when_standing():
    w = WalkCycle()
    still = w.pose(1.23, False)
    assert all(v == 0. for v in still.values())
    quarter = w.pose(w.stride_m / 2, True)  # phase pi/2: left hip forward, right back and flexed
    assert math.isclose(quarter['HipL'], w.hip_swing_rad) and math.isclose(quarter['HipR'], -w.hip_swing_rad)
    assert quarter['KneeL'] == 0. and math.isclose(quarter['KneeR'], w.knee_flex_rad)
    assert math.isclose(quarter['ShoulderL'], -w.arm_swing_rad) and math.isclose(quarter['bob'], w.bob_m)


# --- formation follow law ------------------------------------------------------------------

def test_formation_slots_follow_the_presenter_frame_and_twist_steers_toward_the_slot():
    f = Formation(((-1., 0.), (-1., .5)))
    presenter = (2., 1., math.pi / 2, 0., True)  # facing +y: "behind" is -y, "left" is -x
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(f.slot_world(0, presenter), (2., 0.)))
    assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(f.slot_world(1, presenter), (1.5, 0.)))
    # robot 1 m short of its slot, facing it: pure forward speed, bounded by gain*distance
    vx, vy, wz = f.twist(0, _state(2., -1., math.pi / 2), presenter)
    assert math.isclose(vx, min(f.max_speed_m_s, f.gain_per_s * 1.)) and vy == 0. and abs(wz) < 1e-9
    # slot off to the left (90 deg): turn in place first
    vx, vy, wz = f.twist(0, _state(3., 0., math.pi / 2), presenter)
    assert vx == 0. and wz == f.max_turn_rad_s
    # inside the deadband: no translation, align heading with the presenter
    vx, vy, wz = f.twist(0, _state(2.02, 0.02, 0.), presenter)
    assert (vx, vy) == (0., 0.) and wz == f.max_turn_rad_s
    vx, vy, wz = f.twist(0, _state(2.02, 0.02, math.pi / 2), presenter)
    assert (vx, vy, wz) == (0., 0., 0.)


def test_formation_rejects_colliding_or_leading_slots():
    with pytest.raises(ValueError, match='0.35'):
        Formation.from_dict({'slots': [[-1., 0.], [-1., .2]]}, 2)
    with pytest.raises(ValueError, match='behind'):
        Formation.from_dict({'slots': [[-.2, 0.]]}, 1)
    with pytest.raises(ValueError, match='at least 2 slots'):
        Formation.from_dict({'slots': [[-1., 0.]]}, 2)


# --- choreography file ---------------------------------------------------------------------

def test_choreography_loads_spawns_behind_the_start_pose_and_receipts_everything(tmp_path):
    c = Choreography.load(_choreo_dict(tmp_path), 12)
    spawn = c.spawn_positions()
    assert len(spawn) == 12 and spawn[0] == [-4.8, 1.1] and spawn[11] == [-6., 2.9]
    r = c.receipt()
    assert r['presenter']['pose_source'].startswith('simulation (oracle)')
    assert r['formation']['spawn'] == spawn and r['lighting'] == DEFAULT_LIGHTING
    assert 'not physical admission' in r['scope'] and r['stage_asset'] is None
    # scripted twist for robot 0 at t=0: in its slot, facing +x -> stands
    assert c.script_for(0)(0., _state(-4.8, 1.1, 0.)) == (0., 0., 0.)
    # presenter moved 1 m: robot 0 walks forward
    vx, vy, wz = c.script_for(0)(3. + 1. / .18, _state(-4.8, 1.1, 0.))
    assert vx > 0. and vy == 0.


def test_choreography_requires_a_start_along_plus_x_and_existing_assets(tmp_path):
    with pytest.raises(ValueError, match=r'\+x'):
        Choreography.load(_choreo_dict(tmp_path, presenter={'waypoints': [[0., 0.], [0., 3.]], 'speed_m_s': .2}), 12)
    with pytest.raises(ValueError, match='stage_asset does not exist'):
        Choreography.load(_choreo_dict(tmp_path, stage_asset='missing.usda'), 12)
    with pytest.raises(ValueError, match=r'\(0, 0.5\]'):
        Choreography.load(_choreo_dict(tmp_path, presenter={'waypoints': [[0., 0.], [3., 0.]], 'speed_m_s': .9}), 12)
    with pytest.raises(ValueError, match='unknown lighting key'):
        Choreography.load(_choreo_dict(tmp_path, lighting={'spot': 1.}), 12)


# --- spawn layout --------------------------------------------------------------------------

def test_choreography_layout_takes_explicit_positions_with_a_separation_floor():
    rows = placements(2, 0., 'choreography', 0., positions=[[-1., 0.], [-1., .5]])
    assert rows == {'duck00': ('/World/Duck00', [-1., 0., .125]), 'duck01': ('/World/Duck01', [-1., .5, .125])}
    with pytest.raises(ValueError, match=f'{CHOREOGRAPHY_MIN_SEPARATION_M}'):
        placements(2, 0., 'choreography', 0., positions=[[-1., 0.], [-1., .3]])
    with pytest.raises(ValueError, match='one explicit'):
        placements(2, 0., 'choreography', 0., positions=[[-1., 0.]])
    with pytest.raises(ValueError, match='finite'):
        placements(1, 0., 'choreography', 0., positions=[[float('nan'), 0.]])


# --- controller: scripted twist ------------------------------------------------------------

def _controller():
    from test_mobile_bridge import Clock, _module
    from mobile_support_fixture import support_contract
    clock = Clock()
    c = _module().MobileBridgeController(
        robot_id="duck", source="isolated-bridge", engine="newton", device="cuda:0",
        asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64,
        support_contract=support_contract(), max_linear_speed=0.3, max_angular_speed=1.0,
        max_duration_s=10.0, lease_s=0.5, max_state_age_s=0.5, clock=clock)
    return c, clock


def test_scripted_twist_is_announced_bounded_and_refuses_command_admission():
    from test_mobile_bridge import command, publish
    c, clock = _controller()
    assert c.hello()['scripted_twist'] is False
    generation = c.hello()['generation']
    c.script(lambda t, state: (5., 0., -7.))
    assert c.hello()['scripted_twist'] is True and c.hello()['generation'] == generation + 1
    # nothing completed yet: stands without faulting
    assert c.control_at(0.) == (0., 0., 0.) and c.state()['controller'] != 'fault'
    publish(c)
    assert c.control_at(.005) == (.3, 0., -1.)  # clipped to the admitted limits
    with pytest.raises(ValueError, match='choreography'):
        command(c)
    c.stop()
    assert c.hello()['scripted_twist'] is False and c.control_at(.01) == (0., 0., 0.)


def test_scripted_twist_keeps_the_freshness_rule_and_faults_on_a_broken_script():
    from test_mobile_bridge import publish
    c, clock = _controller()
    publish(c)
    c.script(lambda t, state: (.1, 0., 0.))
    clock.now += .6  # state older than max_state_age_s
    assert c.control_at(.005) == (0., 0., 0.)
    assert c.state()['controller'] == 'fault' and c.hello()['scripted_twist'] is False
    assert c.script_error.startswith('ValueError')
    d, _ = _controller()
    publish(d)
    d.script(lambda t, state: (float('nan'), 0., 0.))
    assert d.control_at(.005) == (0., 0., 0.) and d.state()['controller'] == 'fault'
    assert 'not finite' in d.script_error
    e, _ = _controller()
    with pytest.raises(ValueError, match='callable'):
        e.script(3)


def test_scripted_twist_cannot_coexist_with_an_admitted_command():
    from test_mobile_bridge import command, publish
    c, clock = _controller()
    publish(c)
    command(c)
    with pytest.raises(ValueError, match='coexist'):
        c.script(lambda t, state: (0., 0., 0.))
    assert c.control_at(.005) == (.1, 0., 0.)


# --- presenter proxy (needs pxr) -----------------------------------------------------------

def test_presenter_proxy_writes_root_and_joint_ops(tmp_path):
    pxr = pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, Gf
    from cascade.sim.microduck_choreography import PresenterProxy
    asset = tmp_path / 'presenter.usda'
    stage = Usd.Stage.CreateNew(str(asset))
    root = UsdGeom.Xform.Define(stage, '/Presenter')
    stage.SetDefaultPrim(root.GetPrim())
    for path in ('/Presenter', '/Presenter/HipL', '/Presenter/HipR', '/Presenter/HipL/Knee', '/Presenter/HipR/Knee',
                 '/Presenter/ShoulderL', '/Presenter/ShoulderR'):
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
    UsdGeom.Sphere.Define(stage, '/Presenter/HipL/Knee/Shin')
    stage.Save()
    world = Usd.Stage.CreateInMemory()
    proxy = PresenterProxy(world, str(asset), root='/World/Presenter')
    out = proxy.write((1., 2., math.pi / 2, 0., True), WalkCycle().pose(.325, True))
    prim = world.GetPrimAtPath('/World/Presenter')
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
    assert ops['xformOp:translate'] == Gf.Vec3d(1., 2., WalkCycle().bob_m)
    assert math.isclose(ops['xformOp:rotateXYZ'][2], 90.)
    hip = UsdGeom.Xformable(world.GetPrimAtPath('/World/Presenter/HipL')).GetOrderedXformOps()[1].Get()
    assert math.isclose(hip[1], 25.) and out['joints']['HipL'] == pytest.approx(math.radians(25.))
