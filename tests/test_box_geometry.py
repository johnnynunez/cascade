"""Complete model geometry coverage on synthetic snapshots, no native model."""
from copy import deepcopy
from dataclasses import replace
import itertools

import numpy as np
import pytest

from cascade.eval.box_geometry import BoxGeometryInventory
from cascade.eval.cavity import BoxWall
from cascade.eval.placement import PlacementPolicy, seal_placement_row
from test_box_cavity import cavity


def fixture():
    walls = cavity().walls
    local = np.eye(4); local[0, 3] = .1
    obj = BoxWall(10, tuple(local.flat), (.1, .05, .08))
    policy = PlacementPolicy('a'*64, 'geometry-episode', (10,), tuple(w.geometry_id for w in walls),
                             (20,), .1, (0.,0.,-9.81), .002, 1, 2, 30)
    inventory = BoxGeometryInventory('a'*64, policy.sha256, 1, 2, (obj,), walls)
    support = np.eye(4)
    support[:3, :3] = [[0,-1,0],[1,0,0],[0,0,1]]
    support[:3, 3] = [2,3,0]
    relative = np.eye(4); relative[:3, :3] = support[:3, :3]; relative[:3, 3] = [.1,.2,.5]
    world = {'support':support, 'object':support @ relative}
    bodies, geometries, rotations = {}, [], []
    for name in ('object', 'support'):
        pose = world[name]
        bodies[name] = {'body_id':getattr(policy,name+'_body_id'), 'position_m':pose[:3,3].tolist()}
        for box in getattr(inventory,name+'_boxes'):
            observed = pose @ np.asarray(box.body_from_geometry).reshape(4,4)
            geometries.append({'id':box.geometry_id, 'position_m':observed[:3,3].tolist()})
            rotations.append({'id':box.geometry_id, 'rotation':observed[:3,:3].ravel().tolist()})
    geometries.append({'id':20,'position_m':[5.,5.,5.]})
    row = {'model_identity_sha256':policy.model_identity_sha256, 'policy_sha256':policy.sha256,
           'epoch':policy.epoch, 'native_ngeom':30, 'solver_step':1,
           'constraint_time_s':0., 'advanced_time_s':.002, 'phase':'euler_constraint_before_integration',
           'bodies':bodies, 'geometries':geometries,
           'box_geometry':{'inventory_sha256':inventory.sha256,
               'body_rotations_world':{name:pose[:3,:3].ravel().tolist() for name,pose in world.items()},
               'rotations_world':rotations}}
    calibration = inventory.cavity(interior_point_m=(0.,0.,.5), top_z_m=1.)
    return inventory, policy, calibration, row


def test_rotated_translated_complete_object_hull_is_in_same_solve_support_frame():
    inventory, policy, calibration, row = fixture()
    old = deepcopy(row)
    hull = inventory.object_hulls(seal_placement_row(row), policy)[10]
    expected = np.array(list(itertools.product((.05,.15),(.2,.4),(.42,.58))))
    assert max(min(np.linalg.norm(x-y) for y in expected) for x in hull) < 1e-14
    result = inventory.verify_snapshot(seal_placement_row(row), policy, calibration, support_frame_error_m=.001)
    assert result['status'] == 'confirmed' and result['physical_admission'] is False
    assert row == old


@pytest.mark.parametrize('fault', ['model','epoch','policy','phase','native_count','step','clock',
    'inventory','missing_box','extra_box','duplicate_box','missing_center','duplicate_center',
    'unknown_body','missing_body_rotation','changed_body_rotation','changed_box_rotation','nonrigid',
    'changed_center','nan','unsealed'])
def test_inconsistent_box_witness_cannot_certify_containment(fault):
    inventory, policy, calibration, row = fixture()
    geometry = row['box_geometry']
    if fault == 'model': row['model_identity_sha256'] = 'b'*64
    elif fault == 'epoch': row['epoch'] = 'other'
    elif fault == 'policy': row['policy_sha256'] = 'b'*64
    elif fault == 'phase': row['phase'] = 'after_integration'
    elif fault == 'native_count': row['native_ngeom'] += 1
    elif fault == 'step': row['solver_step'] = True
    elif fault == 'clock': row['advanced_time_s'] += .002
    elif fault == 'inventory': geometry['inventory_sha256'] = 'b'*64
    elif fault == 'missing_box': geometry['rotations_world'].pop()
    elif fault == 'extra_box': geometry['rotations_world'].append({'id':11,'rotation':np.eye(3).ravel().tolist()})
    elif fault == 'duplicate_box': geometry['rotations_world'].append(geometry['rotations_world'][0])
    elif fault == 'missing_center': row['geometries'].pop()
    elif fault == 'duplicate_center': row['geometries'].append(row['geometries'][0])
    elif fault == 'unknown_body': row['bodies']['object']['body_id'] = 3
    elif fault == 'missing_body_rotation': geometry['body_rotations_world'].pop('object')
    elif fault == 'changed_body_rotation': geometry['body_rotations_world']['object'] = np.eye(3).ravel().tolist()
    elif fault == 'changed_box_rotation': geometry['rotations_world'][0]['rotation'] = np.eye(3).ravel().tolist()
    elif fault == 'nonrigid': geometry['rotations_world'][0]['rotation'][0] = 2.
    elif fault == 'changed_center': row['geometries'][0]['position_m'][0] += .01
    sealed = seal_placement_row(row)
    if fault == 'nan': sealed['box_geometry']['rotations_world'][0]['rotation'][0] = float('nan')
    elif fault == 'unsealed': sealed['geometries'][0]['position_m'][0] += .01
    result = inventory.verify_snapshot(sealed, policy, calibration, support_frame_error_m=0.)
    assert result['status'] == 'unverified' and result['physical_admission'] is False


@pytest.mark.parametrize('fault',['missing_shape','wrong_body','wrong_model','wrong_source','wrong_wall','unknown_error'])
def test_calibration_and_inventory_must_share_complete_compiled_identity(fault):
    inventory, policy, calibration, row = fixture()
    error = 0.
    if fault == 'missing_shape': policy = replace(policy, object_geoms=(10,11))
    elif fault == 'wrong_body': calibration = replace(calibration, support_body_id=3)
    elif fault == 'wrong_model': calibration = replace(calibration, model_identity_sha256='b'*64)
    elif fault == 'wrong_source': calibration = replace(calibration, geometry_source_sha256='b'*64)
    elif fault == 'wrong_wall':
        walls = list(calibration.walls); walls[0] = replace(walls[0], geometry_id=29)
        calibration = replace(calibration, walls=tuple(walls))
    elif fault == 'unknown_error': error = None
    assert inventory.verify_snapshot(seal_placement_row(row),policy,calibration,support_frame_error_m=error)['status'] == 'unverified'


def test_complete_geometry_that_protrudes_is_unverified_even_when_its_center_is_inside():
    inventory, policy, calibration, row = fixture()
    # Enlarge the compiled object and bind the new whole inventory consistently.
    inventory = replace(inventory, object_boxes=(replace(inventory.object_boxes[0], half_size_m=(2.,.05,.08)),))
    row['box_geometry']['inventory_sha256'] = inventory.sha256
    calibration = inventory.cavity(interior_point_m=(0.,0.,.5),top_z_m=1.)
    assert inventory.verify_snapshot(seal_placement_row(row),policy,calibration,support_frame_error_m=0.)['status'] == 'unverified'
