"""Bounded exact observer transport, retaining plain and v1 decoding.

V2 compresses each binary array independently. Vertex float32 bytes are in
column-major order; counts use uint16; indices use signed int16 differences
from the preceding index (initial preceding index zero). Everything is little
endian. Decoding reconstructs the complete original vertex/topology lists.
"""
import base64
import binascii
import math
import struct
import zlib

MARKER = 'KITCHEN_OBSERVER '
COMPRESSED_MARKER = 'KITCHEN_OBSERVER_ZLIB '
MAX_PAYLOAD = 65536
MAX_LINE = 7400
ARRAYS_V1 = 'geometry_arrays_f32_u16_le_v1'
ARRAYS_V2 = 'geometry_arrays_f32_column_u16_i16delta_le_zlib_v2'
ARRAY_KEYS = {'vertices_m', 'face_vertex_counts', 'face_vertex_indices'}


def snapshot_payload(stdout):
    if not isinstance(stdout, str):
        raise ValueError('Observer stdout must be text')
    lines = [line for line in stdout.splitlines()
             if line.startswith(MARKER) or line.startswith(COMPRESSED_MARKER)]
    if len(lines) != 1:
        raise ValueError('Missing, duplicate or truncated observer JSON marker')
    line = lines[0]
    if len(line.encode('utf-8')) >= MAX_LINE:
        raise ValueError('Observer wire line exceeds the bounded bridge transport')
    if line.startswith(MARKER):
        return line[len(MARKER):]
    try:
        packed = base64.b64decode(line[len(COMPRESSED_MARKER):], validate=True)
        if not packed or len(packed) > MAX_LINE:
            raise ValueError('Invalid bounded compressed observer payload')
        decoder = zlib.decompressobj()
        raw = decoder.decompress(packed, MAX_PAYLOAD + 1)
        if (len(raw) > MAX_PAYLOAD or not decoder.eof
                or decoder.unconsumed_tail or decoder.unused_data):
            raise ValueError('Incomplete, concatenated or oversized observer compression stream')
        return raw.decode('utf-8')
    except (binascii.Error, zlib.error, UnicodeError) as exc:
        raise ValueError('Invalid compressed observer payload') from exc


def encode_code():
    """Drop-in generated code; reads only existing observed arrays."""
    return '''import zlib as _obs_zlib
_obs_geom = _gpu_observed["scene_geometry"]["convex_collider"]
_obs_arrays = {}
for _obs_key in ("vertices_m", "face_vertex_counts", "face_vertex_indices"):
    _obs_values = _obs_np.asarray(_obs_geom[_obs_key])
    _obs_shape = list(_obs_values.shape)
    if _obs_key == "vertices_m":
        if _obs_values.ndim != 2 or _obs_values.shape[1] != 3 or not 4 <= len(_obs_values) <= 10000:
            raise RuntimeError("Invalid bounded convex vertex shape")
        _obs_values = _obs_values.T
        _obs_dtype = "<f4"
    else:
        if _obs_values.ndim != 1 or not 1 <= len(_obs_values) <= 60000:
            raise RuntimeError("Invalid bounded convex topology shape")
        _obs_dtype = "<u2"
    _obs_baked = _obs_np.ascontiguousarray(_obs_values, dtype=_obs_dtype)
    if not _obs_np.isfinite(_obs_values).all() or not _obs_np.array_equal(_obs_values, _obs_baked):
        raise RuntimeError("Convex geometry cannot be encoded losslessly")
    if _obs_key == "face_vertex_indices":
        _obs_deltas = _obs_np.diff(_obs_baked.astype(_obs_np.int64), prepend=0)
        _obs_baked = _obs_np.ascontiguousarray(_obs_deltas, dtype="<i2")
        if not _obs_np.array_equal(_obs_deltas, _obs_baked):
            raise RuntimeError("Convex topology delta exceeds signed16 range")
    _obs_arrays[_obs_key] = {"shape": _obs_shape,
        "zlib_base64": _obs_base64.b64encode(_obs_zlib.compress(_obs_baked.tobytes(), 9)).decode("ascii")}
    del _obs_geom[_obs_key]
_obs_geom["geometry_arrays_f32_column_u16_i16delta_le_zlib_v2"] = _obs_arrays
_obs_payload = _obs_json.dumps(_gpu_observed, separators=(",", ":"), allow_nan=False)
if len(_obs_payload.encode("utf-8")) > 65536:
    raise RuntimeError("Observer decoded snapshot exceeds 65536 bytes")
_obs_encoded = _obs_base64.b64encode(_obs_zlib.compress(_obs_payload.encode("utf-8"), 6)).decode("ascii")
_obs_line = "KITCHEN_OBSERVER_ZLIB " + _obs_encoded
if len(_obs_line.encode("utf-8")) >= 7400:
    raise RuntimeError("Observer compressed snapshot exceeds safe bridge stdout size")
print(_obs_line)
'''


def _shape(key, shape):
    if (not isinstance(shape, list) or any(type(n) is not int or n < 1 for n in shape)
            or (key == 'vertices_m' and (len(shape) != 2 or shape[1] != 3 or not 4 <= shape[0] <= 10000))
            or (key != 'vertices_m' and (len(shape) != 1 or shape[0] > 60000))):
        raise ValueError('Compact convex array shape exceeds its bounds')
    return math.prod(shape)


def _array_bytes(text, expected_bytes, *, compressed):
    # Bound standalone calls as well as calls behind the outer payload cap.
    max_packed = MAX_PAYLOAD if compressed else expected_bytes
    if not isinstance(text, str) or len(text) > 4 * ((max_packed + 2) // 3):
        raise ValueError('Compact convex array encoding exceeds its bound')
    try:
        packed = base64.b64decode(text, validate=True)
        if len(packed) > max_packed:
            raise ValueError('Compact convex array packed bytes exceed their bound')
        if compressed:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(packed, expected_bytes + 1)
            if (len(raw) != expected_bytes or not decoder.eof
                    or decoder.unconsumed_tail or decoder.unused_data):
                raise ValueError('Incomplete, concatenated or wrong-sized convex array stream')
        else:
            raw = packed
            if len(raw) != expected_bytes:
                raise ValueError('Compact convex array byte length differs from shape')
        return raw
    except (TypeError, binascii.Error, zlib.error) as exc:
        raise ValueError('Invalid compact convex array encoding') from exc


def expand_geometry(value):
    """Expand v1 or v2 arrays with strict finite/size/topology validation."""
    if not isinstance(value, dict):
        raise ValueError('Observer physics payload must be an object')
    scene = value.get('scene_geometry')
    if scene is None:
        return value
    if not isinstance(scene, dict):
        raise ValueError('Observer scene geometry must be an object')
    geometry = scene.get('convex_collider')
    if geometry is None:
        return value
    if not isinstance(geometry, dict):
        raise ValueError('Convex collider geometry must be an object')
    tags = [tag for tag in (ARRAYS_V1, ARRAYS_V2) if tag in geometry]
    if not tags:
        return value
    if len(tags) != 1:
        raise ValueError('Convex payload mixes compact array versions')
    tag = tags[0]
    arrays = geometry[tag]
    compressed = tag == ARRAYS_V2
    field = 'zlib_base64' if compressed else 'base64'
    if not isinstance(arrays, dict) or set(arrays) != ARRAY_KEYS or ARRAY_KEYS.intersection(geometry):
        raise ValueError('Convex payload has missing, mixed or repeated array representations')
    unpacked = {}
    for key in ('vertices_m', 'face_vertex_counts', 'face_vertex_indices'):
        item = arrays[key]
        if not isinstance(item, dict) or set(item) != {'shape', field}:
            raise ValueError('Invalid compact convex array receipt')
        count = _shape(key, item['shape'])
        size, fmt = (4, 'f') if key == 'vertices_m' else (2, 'H')
        if compressed and key == 'face_vertex_indices':
            fmt = 'h'
        raw = _array_bytes(item[field], count * size, compressed=compressed)
        values = list(struct.unpack('<' + str(count) + fmt, raw))
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Compact convex geometry must be finite')
        if key == 'vertices_m':
            if compressed:
                length = item['shape'][0]
                unpacked[key] = [[values[i], values[length+i], values[2*length+i]] for i in range(length)]
            else:
                unpacked[key] = [values[i:i+3] for i in range(0, count, 3)]
        elif key == 'face_vertex_indices' and compressed:
            indices = []
            previous = 0
            for delta in values:
                index = previous + delta
                if not 0 <= index < len(unpacked['vertices_m']):
                    raise ValueError('Compact convex index delta reconstructs an out-of-range vertex')
                indices.append(index)
                previous = index
            unpacked[key] = indices
        else:
            unpacked[key] = values
    if (sum(unpacked['face_vertex_counts']) != len(unpacked['face_vertex_indices'])
            or any(n < 3 for n in unpacked['face_vertex_counts'])
            or any(i >= len(unpacked['vertices_m']) for i in unpacked['face_vertex_indices'])):
        raise ValueError('Compact convex topology is inconsistent')
    geometry.update(unpacked)
    del geometry[tag]
    return value
