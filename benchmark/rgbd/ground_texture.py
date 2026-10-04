"""CPU-authorable planar reference on an existing collision plane.

No camera data enters the bitmap geometry. This prototype neither boots Kit nor
changes a physics importer. Actual RTX primvar support remains a native gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import zlib

import numpy as np

from .planar_reference import Board, digest

PIXELS = 4000
EXTENT_M = 2
ST = ((1., 1.), (0., 1.), (0., 0.), (1., 0.))
GROUND = '/World/Ground'
MATERIAL = '/World/RgbdGroundReferenceMaterial'


@dataclass(frozen=True)
class GroundTextureBoard(Board):
    origin_xyz_m: tuple = (0.08, 0.01, 0.0)

    def __post_init__(self):
        super().__post_init__()
        if self.origin_xyz_m[2] != 0.:
            raise ValueError('existing Ground reference requires world Z=0')

    def description(self):
        # Do not inherit the historical mesh's visual-only/raised-background
        # description: this material lies on the existing collision surface.
        return {
            'schema': 2, 'kind': 'existing_ground_texture',
            'columns': self.columns, 'rows': self.rows, 'square_m': self.square_m,
            'origin_xyz_m': list(self.origin_xyz_m), 'axes': 'world_x_y',
            'units': 'm', 'surface': 'existing_collision_plane_visual_material',
            'background_below_m': 0., 'marker_center_outside_squares': 0.8,
            'marker_halfwidth_squares': 0.25,
            'marker_order': ['red', 'green', 'blue', 'yellow'],
            'texture': {'width': PIXELS, 'height': PIXELS, 'format': 'RGB8_PNG',
                        'row_zero': 'positive_world_y', 'texel_m': 0.0005,
                        'color_space': 'raw', 'wrap_s': 'clamp', 'wrap_t': 'clamp'},
            'ground': {'path': GROUND, 'axis': 'Z', 'width_m': 2., 'length_m': 2.,
                       'world_transform': 'identity', 'st': [list(v) for v in ST],
                       'st_interpolation': 'vertex', 'st_element_size': 1},
        }


def texture_rectangles(board):
    """Exact half-open texel-edge rectangles, rejecting unrepresentable metrics."""
    from .binary_reference import BinaryGroundBoard, binary_rectangles
    from .binary_layout_a import BinaryLayoutABoard, layout_rectangles
    from .checker_accuracy import AccuracyBoard
    from .checker_saddle import SaddleBoard
    if type(board) in (AccuracyBoard, SaddleBoard):
        return layout_rectangles(board.geometry)
    if type(board) is BinaryLayoutABoard:
        return layout_rectangles(board)
    if type(board) is BinaryGroundBoard:
        return binary_rectangles(board)
    if type(board) is not GroundTextureBoard:
        raise ValueError('explicit GroundTextureBoard required')
    x, y = (Fraction(str(v)) for v in board.origin_xyz_m[:2])
    s = Fraction(str(board.square_m))
    rectangles = []

    def edge(value):
        pixel = (value + 1) * PIXELS / EXTENT_M
        if pixel.denominator != 1 or not 0 <= pixel <= PIXELS:
            raise ValueError('every reference edge must be an in-plane integer texel boundary')
        return int(pixel)

    def rect(x0, y0, width, height, rgb, label):
        u0, u1 = edge(x0), edge(x0 + width)
        v0, v1 = PIXELS - edge(y0 + height), PIXELS - edge(y0)
        if u0 >= u1 or v0 >= v1:
            raise ValueError('nonempty reference rectangle required')
        rectangles.append({'label': label, 'bounds_uv': [u0, v0, u1, v1],
                           'rgb8': list(rgb),
                           'xy_bounds_m': list(map(float, (x0, y0, x0+width, y0+height)))})

    pad = Fraction(7, 5) * s
    rect(x-pad, y-pad, board.columns*s+2*pad, board.rows*s+2*pad,
         (255, 255, 255), 'background')
    for r in range(board.rows):
        for c in range(board.columns):
            rgb = (255,)*3 if (r+c) % 2 else (0,)*3
            rect(x+c*s, y+r*s, s, s, rgb, f'square_{r}_{c}')
    half = s / 4
    outside = Fraction(4, 5)
    for (c, r), color, name in zip(
        [(-outside, -outside), (board.columns+outside, -outside),
         (-outside, board.rows+outside), (board.columns+outside, board.rows+outside)],
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)],
        ['red', 'green', 'blue', 'yellow'],
    ):
        # Centers also lie on exact texel boundaries, independently of edges.
        edge(x+c*s), edge(y+r*s)
        rect(x+c*s-half, y+r*s-half, 2*half, 2*half, color, name)
    return rectangles


def texture_rgb(board):
    rectangles = texture_rectangles(board)  # validate before allocating 48 MB
    image = np.full((PIXELS, PIXELS, 3), 96, dtype=np.uint8)
    for row in rectangles:
        u0, v0, u1, v1 = row['bounds_uv']
        image[v0:v1, u0:u1] = row['rgb8']
    return image


def png_bytes(board):
    """Deterministic lossless PNG: fixed filter 0, compression and no metadata."""
    image = texture_rgb(board)

    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', zlib.crc32(kind + data)))

    compressor = zlib.compressobj(level=9)
    compressed = b''.join(compressor.compress(b'\0' + row.tobytes()) for row in image)
    compressed += compressor.flush()
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('!2I5B', PIXELS, PIXELS, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', compressed) + chunk(b'IEND', b''))


def write_texture(path, *, board):
    path = Path(path)
    data = png_bytes(board)
    with path.open('xb') as stream:
        stream.write(data)
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'board_sha256': board.sha256, 'rectangles_sha256': digest(texture_rectangles(board))}


def scene_bindings(stage):
    """Bind all original opinions and the complete added appearance graph.

    No geometry subtree is exempted: even the historical mesh board path is
    included. Only the exact newly authored appearance opinions are separated.
    """
    from .native_bridge import stage_snapshot
    snapshot = stage_snapshot(stage, exclude_path=None)
    appearance = {'ground_st': snapshot['prims'][GROUND]['attributes'].pop('primvars:st', None),
                  'visual_binding': snapshot['prims'][GROUND]['relationships'].pop('material:binding', None),
                  'material_prims': {}}
    for path in list(snapshot['prims']):
        if path == MATERIAL or path.startswith(MATERIAL+'/'):
            appearance['material_prims'][path] = snapshot['prims'].pop(path)
    return {'original_stage_sha256': digest(snapshot), 'appearance_sha256': digest(appearance)}


def validate_ground_texture(stage, texture_path, *, board, receipt):
    """Reject changed texture, mapping, shader, original physics or extra geometry."""
    texture = Path(texture_path).read_bytes()
    if (receipt.get('board_sha256') != board.sha256
            or receipt.get('rectangles_sha256') != digest(texture_rectangles(board))
            or hashlib.sha256(texture).hexdigest() != receipt.get('texture_sha256')
            or texture != png_bytes(board)):
        raise ValueError('ground texture or metric descriptor changed')
    binding = scene_bindings(stage)
    if any(receipt.get(key) != value for key, value in binding.items()):
        raise ValueError('ground appearance or original scene changed')
    return binding


def author_ground_texture(stage, texture_path, *, board):
    """Add appearance only; preserve the existing physics-purpose binding.

    The caller must include the returned texture/board/shader and new composed
    scene/source digests in a NEW model recipe before any subsequent native run.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    texture_path = Path(texture_path).resolve(strict=True)
    if texture_path.stat().st_size > 1024 * 1024:
        raise ValueError('bounded reference PNG required')
    texture = texture_path.read_bytes()
    if texture != png_bytes(board):
        raise ValueError('texture does not encode the declared exact board')
    ground = stage.GetPrimAtPath(GROUND)
    plane = UsdGeom.Plane(ground)
    if (not plane or stage.GetPrimAtPath(MATERIAL)
            or UsdGeom.GetStageMetersPerUnit(stage) != 1.
            or plane.GetAxisAttr().Get() != 'Z'
            or plane.GetWidthAttr().Get() != 2. or plane.GetLengthAttr().Get() != 2.
            or UsdGeom.Xformable(ground).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) != Gf.Matrix4d(1.)
            or not ground.HasAPI(UsdPhysics.CollisionAPI)):
        raise ValueError('unchanged two-meter world Ground collision plane required')
    ancestor = ground
    while ancestor and not ancestor.IsPseudoRoot():
        if any(a.GetNumTimeSamples() for a in ancestor.GetAttributes()
               if a.GetName().startswith('xformOp:') or a.GetName() in ('axis', 'width', 'length')):
            raise ValueError('static Ground geometry and ancestor transforms required')
        ancestor = ancestor.GetParent()
    bindings = UsdShade.MaterialBindingAPI(ground)
    physics_material, physics_rel = bindings.ComputeBoundMaterial('physics')
    if (not physics_material or str(physics_material.GetPath()) != '/World/GroundMaterial'
            or bindings.ComputeBoundMaterial()[0]
            or UsdGeom.PrimvarsAPI(ground).GetPrimvar('st')):
        raise ValueError('original physics-only binding and no existing st required')
    prior_physics = (str(physics_material.GetPath()), str(physics_rel.GetPath()))
    before = scene_bindings(stage)['original_stage_sha256']
    material = UsdShade.Material.Define(stage, MATERIAL)
    surface = UsdShade.Shader.Define(stage, MATERIAL+'/surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    # Baseline unmaterialed Newton color is this same schema default. The board
    # is emissive; no displacement, opacity map or physics-material replacement.
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.18))
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
    surface.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(1.)
    uv = UsdShade.Shader.Define(stage, MATERIAL+'/uv')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    sampler = UsdShade.Shader.Define(stage, MATERIAL+'/texture')
    sampler.CreateIdAttr('UsdUVTexture')
    sampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    for name, value in [('sourceColorSpace', 'raw'), ('wrapS', 'clamp'), ('wrapT', 'clamp')]:
        sampler.CreateInput(name, Sdf.ValueTypeNames.Token).Set(value)
    sampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
    surface.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(sampler.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
    from .binary_layout_a import BinaryLayoutABoard
    from .checker_accuracy import AccuracyBoard
    from .checker_saddle import SaddleBoard
    mapping = board.texture_st() if type(board) in (BinaryLayoutABoard, AccuracyBoard, SaddleBoard) else ST
    UsdGeom.PrimvarsAPI(ground).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(mapping)
    bindings.Bind(material)
    current, current_rel = bindings.ComputeBoundMaterial('physics')
    if (str(current.GetPath()), str(current_rel.GetPath())) != prior_physics:
        raise ValueError('physics material binding changed')
    binding = scene_bindings(stage)
    if binding['original_stage_sha256'] != before:
        raise ValueError('appearance authoring changed original scene')
    return {**binding, 'board': board.description(), 'board_sha256': board.sha256,
            'rectangles_sha256': digest(texture_rectangles(board)),
            'texture_sha256': hashlib.sha256(texture).hexdigest(), 'texture_bytes': len(texture),
            'material_path': MATERIAL, 'shader': 'UsdPreviewSurface',
            'texture_input': 'emissiveColor', 'diffuse_color': [.18, .18, .18],
            'filtering': 'renderer-selected (not measured by CPU prototype)',
            'physics_material': prior_physics[0], 'physics_binding': prior_physics[1],
            'native_render_validated': False, 'physical_admission': False}
