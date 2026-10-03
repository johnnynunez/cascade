"""CPU bitmap/contract controls; these do not claim RTX or native admission."""
import hashlib

import cv2
import numpy as np
import pytest

from benchmark.rgbd.ground_texture import (
    GroundTextureBoard, PIXELS, ST, author_ground_texture, png_bytes,
    texture_rectangles, texture_rgb, write_texture,
)
from benchmark.rgbd.planar_reference import Board, reference_from_rgb


def ground_capture(mutation=None):
    """Declared CPU projection hypothesis over a separately rasterized texture."""
    from benchmark.rgbd.planar_reference import digest
    from cascade.sensing.models import MeasurementMetadata, ObservationEnvelope, RgbdPayload
    board = GroundTextureBoard()
    rgb = cv2.resize(texture_rgb(board)[1320:2280, 1800:3080], (640, 480),
                     interpolation=cv2.INTER_AREA)
    k = np.diag([1000., 1000., 1.])
    t = np.diag([1., -1., -1., 1.])
    t[:3, 3] = [-.1, .34, 1.]
    if mutation == 'fx':
        k[0, 0] *= 1.02
    elif mutation == 'fy':
        k[1, 1] *= 1.02
    calibration = digest({'k': k.tolist(), 't': t.tolist(), 'offset': [.5, .5]})
    payload = RgbdPayload(MeasurementMetadata('camera', calibration), 640, 480,
        rgb.tobytes(), np.ones((480,640),dtype='<f4').tobytes(),
        k.flatten().tolist(), t.flatten().tolist(), 'world', (.5, .5))
    capture = ObservationEnvelope('cpu-texture', 'overview', 'synthetic-epoch', 1,
        'synthetic', .1, 10., .1, None, 'synthetic', payload)
    return board, capture


def test_distinct_ground_contract_keeps_historical_mesh_default():
    old = Board()
    board = GroundTextureBoard()
    assert old.origin_xyz_m[2] == .002 and old.description()['surface'] == 'visual_only'
    assert board.origin_xyz_m[2] == 0. and board.sha256 != old.sha256
    assert board.description()['background_below_m'] == 0.
    assert 'collision_plane' in board.description()['surface']
    assert board.description()['texture']['texel_m'] == .0005
    assert ST == ((1., 1.), (0., 1.), (0., 0.), (1., 0.))


def test_all_rectangles_and_centers_have_exact_metric_texel_boundaries():
    board = GroundTextureBoard()
    rows = texture_rectangles(board)
    assert len(rows) == 53
    assert rows[1]['bounds_uv'] == [2160, 1920, 2220, 1980]
    for row in rows:
        u0, v0, u1, v1 = row['bounds_uv']
        assert all(type(v) is int and 0 <= v <= PIXELS for v in row['bounds_uv'])
        actual = [(u0-2000)/2000, (2000-v1)/2000, (u1-2000)/2000, (2000-v0)/2000]
        np.testing.assert_array_equal(actual, row['xy_bounds_m'])
        if row['label'].startswith('square_'):
            assert u1-u0 == v1-v0 == 60
        elif row['label'] != 'background':
            assert u1-u0 == v1-v0 == 30
            assert (u0+u1) % 2 == (v0+v1) % 2 == 0


@pytest.mark.parametrize('board', [Board(), GroundTextureBoard(square_m=.0301),
    GroundTextureBoard(origin_xyz_m=(.0801, .01, 0.)),
    GroundTextureBoard(origin_xyz_m=(.9, .01, 0.))])
def test_wrong_reference_or_nonintegral_or_outside_bitmap_rejected(board):
    with pytest.raises(ValueError):
        texture_rectangles(board)


def test_wrong_height_is_not_a_ground_reference():
    with pytest.raises(ValueError, match='Z=0'):
        GroundTextureBoard(origin_xyz_m=(.08, .01, .002))


def test_png_is_deterministic_exact_rgb_and_bound_to_descriptor(tmp_path):
    board = GroundTextureBoard()
    path = tmp_path/'texture.png'
    receipt = write_texture(path, board=board)
    assert path.read_bytes() == png_bytes(board)
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded[:, :, ::-1], texture_rgb(board))
    with pytest.raises(FileExistsError):
        write_texture(path, board=board)


def test_rgb_only_oracle_receives_explicit_ground_board_and_no_camera_geometry():
    board = GroundTextureBoard()
    # Fixed integer crop of the actual bitmap, not a camera render. It tests
    # authored color orientation and metric origin using no K/T/depth/raycast.
    rgb = cv2.resize(texture_rgb(board)[1320:2280, 1800:3080], (640, 480),
                     interpolation=cv2.INTER_AREA)
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        reference = reference_from_rgb(rgb, board)
    finally:
        cv2.setNumThreads(old_threads)
    assert reference.board is board
    pixels = np.asarray(reference.pixels())
    expected = np.c_[((pixels[:, 0]+.5)*2+1800)/2000-1,
                      1-((pixels[:, 1]+.5)*2+1320)/2000,
                      np.zeros(len(pixels))]
    np.testing.assert_allclose(reference.expected_xyz(), expected, atol=.0001, rtol=0)
    assert len(reference.held_ids) == 17 and len(reference.fit_ids) == 18


def test_aliased_cpu_resample_still_rejects_at_original_fit_thresholds():
    board = GroundTextureBoard()
    rgb = cv2.resize(texture_rgb(board)[1520:2060, 2020:2740], (640, 480),
                     interpolation=cv2.INTER_AREA)
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with pytest.raises(ValueError, match='held-out reference error'):
            reference_from_rgb(rgb, board)
    finally:
        cv2.setNumThreads(old_threads)


def test_cpu_usd_authoring_preserves_physics_purpose_and_adds_no_geometry(tmp_path):
    pytest.importorskip('pxr', reason='OpenUSD SDK unavailable in ordinary CPU environment')
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    plane = UsdGeom.Plane.Define(stage, '/World/Ground')
    plane.CreateAxisAttr('Z')
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    material = UsdShade.Material.Define(stage, '/World/GroundMaterial')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(1.)
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material, materialPurpose='physics')
    path = tmp_path/'texture.png'
    board = GroundTextureBoard()
    write_texture(path, board=board)
    receipt = author_ground_texture(stage, path, board=board)
    assert receipt['physics_binding'] == '/World/Ground.material:binding:physics'
    assert receipt['native_render_validated'] is False
    assert len([p for p in stage.TraverseAll() if p.IsA(UsdGeom.Gprim)]) == 1
    assert material.GetPrim().GetAttribute('physics:dynamicFriction').Get() == 1.
    assert tuple(map(tuple, UsdGeom.PrimvarsAPI(plane).GetPrimvar('st').Get())) == ST
    with pytest.raises(ValueError):
        author_ground_texture(stage, path, board=board)
