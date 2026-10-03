"""NULL arena descriptors, backed mutation, and allocator ABI controls."""
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.control.fastening import FasteningFault
from cascade.sim import factory_precompile as pre
from test_factory_precompile import fixture as fixture


def owner(pointer):
    field = ctypes.c_void_p(pointer)
    return NS(_address=ctypes.addressof(field), keep_alive=field)


LAYOUT = {'pointer_offsets': {'field': 0}}


def test_null_values_are_not_physical_buffers_but_shape_and_type_are_bound():
    data = owner(None)
    row = pre._mujoco_arena_row(data, 'field', np.full(6, 1, np.int32), LAYOUT)
    assert row == pre._mujoco_arena_row(data, 'field', np.full(6, 987, np.int32), LAYOUT)
    assert row['storage'] == 'absent_native_arena' and row['native_pointer'] is None
    assert 'sha256' not in row
    assert row != pre._mujoco_arena_row(data, 'field', np.zeros(5, np.int32), LAYOUT)
    assert row != pre._mujoco_arena_row(data, 'field', np.zeros(6, np.float64), LAYOUT)


def test_backed_mutation_relocation_and_transition_to_null_are_visible():
    buffer = (ctypes.c_int * 6)(*range(6))
    array = np.ctypeslib.as_array(buffer)
    data = owner(ctypes.addressof(buffer))
    before = pre._mujoco_arena_row(data, 'field', array, LAYOUT)
    array[2] = 9
    after = pre._mujoco_arena_row(data, 'field', array, LAYOUT)
    assert before['sha256'] != after['sha256']
    other = (ctypes.c_int * 6)(*array)
    data.keep_alive.value = ctypes.addressof(other)
    moved = pre._mujoco_arena_row(data, 'field', np.ctypeslib.as_array(other), LAYOUT)
    assert moved['sha256'] == after['sha256'] and moved != after
    data.keep_alive.value = None
    absent = pre._mujoco_arena_row(data, 'field', array.copy(), LAYOUT)
    assert absent != moved


@pytest.mark.parametrize('kind', ['null_alias', 'backed_copy', 'different_pointer'])
def test_inconsistent_storage_never_yields_a_snapshot_row(kind):
    buffer = (ctypes.c_int * 6)(*range(6))
    array = np.ctypeslib.as_array(buffer)
    data = owner(ctypes.addressof(buffer))
    if kind == 'null_alias':
        data.keep_alive.value = None
    elif kind == 'backed_copy':
        array = array.copy()
    else:
        other = (ctypes.c_int * 6)(*range(6))
        data.keep_alive.value = ctypes.addressof(other)
    with pytest.raises(FasteningFault):
        pre._mujoco_arena_row(data, 'field', array, LAYOUT)


def test_empty_descriptor_preserves_native_pointer_without_view_requirement():
    buffer = (ctypes.c_int * 1)(0)
    data = owner(ctypes.addressof(buffer))
    present = pre._mujoco_arena_row(data, 'field', np.empty(0, np.int32), LAYOUT)
    data.keep_alive.value = None
    absent = pre._mujoco_arena_row(data, 'field', np.empty(0, np.int32), LAYOUT)
    assert present['sha256'] == absent['sha256'] and present != absent


@pytest.fixture
def abi(monkeypatch, tmp_path):
    class Data:
        _address = 12345  # Never dereferenced by the admission-only function.
    binary = tmp_path/'_structs.so'
    binary.write_bytes(b'reviewed fake binding for admission-negative controls')
    native = tmp_path/'libmujoco.so.3.12.0'
    native.write_bytes(b'reviewed fake allocator for admission-negative controls')
    header = tmp_path/'mjdata.h'
    header.write_bytes(b'reviewed fake header')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    layout = {'structs_module_sha256': sha(binary),
              'headers_sha256': {'mjdata.h': sha(header)},
              'pointer_bytes': ctypes.sizeof(ctypes.c_void_p), 'byteorder': pre.sys.byteorder,
              'sizeof_mjData': 16, 'pointer_offsets': {'field': 0},
              'native_library': {'basename': native.name, 'sha256': sha(native)}}
    path = tmp_path/'layout.json'
    path.write_text(json.dumps(layout))
    monkeypatch.setattr(pre, '_MJDATA_LAYOUT', path)
    monkeypatch.setattr(pre, '_MJDATA_LAYOUT_SHA256', sha(path))
    monkeypatch.setattr(pre.importlib, 'import_module', lambda _: NS(MjData=Data, __file__=str(binary)))
    monkeypatch.setattr(pre.sys, 'platform', 'linux')
    read = Path.read_text
    maps = [f'1-2 r-xp 0 0:0 1 {native}\n']
    monkeypatch.setattr(Path, 'read_text', lambda p, *a, **k:
                        maps[0] if str(p) == '/proc/self/maps' else read(p, *a, **k))
    return NS(data=Data(), layout=layout, path=path, maps=maps,
              binary=binary, native=native, header=header, sha=sha)


@pytest.mark.parametrize('change', ['layout', 'binary', 'header', 'allocator',
                                   'absent_allocator', 'duplicate_allocator', 'wrong_type'])
def test_unreviewed_abi_fails_before_native_pointer_read(abi, change):
    assert pre._mujoco_arena_layout(abi.data) == abi.layout
    data = abi.data
    if change in ('layout', 'binary', 'header'):
        p = {'layout': abi.path, 'binary': abi.binary, 'header': abi.header}[change]
        p.write_bytes(p.read_bytes()+b'changed')
    elif change == 'allocator':
        abi.native.write_bytes(b'another ABI')
    elif change == 'absent_allocator':
        abi.maps[0] = ''
    elif change == 'duplicate_allocator':
        abi.maps[0] += '2-3 r-xp 0 0:0 1 /elsewhere/libmujoco.so.3.12.0\n'
    else:
        data = NS(_address=abi.data._address)
    with pytest.raises(FasteningFault):
        pre._mujoco_arena_layout(data)


@pytest.fixture
def pinned_mujoco():
    mj = pytest.importorskip('mujoco')
    module = __import__('mujoco._structs', fromlist=['_structs'])
    layout = json.loads(pre._MJDATA_LAYOUT.read_text())
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != layout['structs_module_sha256']:
        pytest.skip('actual SDK controls require the explicitly pinned MuJoCo3.12 Linux wheel')
    return mj


@pytest.mark.parametrize('forwarded', [False, True])
def test_real_arena_snapshot_repeatable_and_physical_changes_still_detected(fixture, pinned_mujoco, forwarded):
    mj = pinned_mujoco
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>''')
    data = mj.MjData(model)
    if forwarded:
        mj.mj_forward(model, data)  # CPU setup only; no steps or compile calls.
    fixture.scene.solver.cpu_data = data
    before = pre.physical_snapshot(fixture.scene, fixture.wp)
    assert before == pre.physical_snapshot(fixture.scene, fixture.wp)
    assert any(row.get('storage') == 'absent_native_arena' for row in before.values())
    data.qpos[0] += .001
    assert before != pre.physical_snapshot(fixture.scene, fixture.wp)
    data.qpos[0] -= .001
    assert before == pre.physical_snapshot(fixture.scene, fixture.wp)
    if forwarded:
        assert data.nefc > 0
        data.efc_force[0] += 1
        assert before != pre.physical_snapshot(fixture.scene, fixture.wp)
    assert data.time == 0


@pytest.mark.parametrize('field', ['contact_pos', 'contact_friction', 'option_gravity',
                                   'warning_number', 'solver_improvement'])
def test_nested_native_contacts_options_and_stats_cannot_escape(fixture, pinned_mujoco, field):
    mj = pinned_mujoco
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/><body pos="0 0 .09"><freejoint/>
      <geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>''')
    data = mj.MjData(model)
    mj.mj_forward(model, data)  # One CPU fixture setup; no simulated time.
    fixture.scene.solver.cpu_data = data
    fixture.scene.solver.cpu_model = model
    before = pre.physical_snapshot(fixture.scene, fixture.wp)
    assert before == pre.physical_snapshot(fixture.scene, fixture.wp)
    assert data.ncon == 4
    if field == 'contact_pos':
        data.contact.pos[0, 0] += .125
    elif field == 'contact_friction':
        data.contact.friction[0, 0] += .125
    elif field == 'option_gravity':
        model.opt.gravity[0] += .125
    elif field == 'warning_number':
        data.warning.number[0] += 1
    else:
        data.solver.improvement[0] += .125
    assert before != pre.physical_snapshot(fixture.scene, fixture.wp)
    assert data.time == 0
