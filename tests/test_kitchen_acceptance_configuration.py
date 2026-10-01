"""Offline evidence/configuration guards; no simulator or physical acceptance."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.config import load_demo_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('kitchen_acceptance_configuration', ROOT / 'benchmark/diagnostics/kitchen_acceptance.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def args(minimum=None, maximum=None, occupancy='nvblox'):
    return SimpleNamespace(query_region_min=minimum, query_region_max=maximum, occupancy=occupancy)


def test_query_expansion_preserves_workspace_and_clearance():
    cfg = load_demo_config(arm='isaac_kitchen_gpu')
    safety = copy.deepcopy(cfg.safety.as_dict())
    report = diag.configure_query_region(cfg, args([.1, -.3, -.01], [.58, .3, .55]))
    assert report['max'] == [.58, .3, .55]
    assert report['safety_before'] == report['safety_after']
    assert cfg.safety.as_dict() == safety
    assert cfg.safety.workspace.max[0] == .50
    assert cfg.safety.min_clearance_m == .03


def test_default_query_preserves_existing_configuration():
    cfg = load_demo_config(arm='isaac_kitchen_gpu')
    before = copy.deepcopy(cfg.occupancy.as_dict())
    report = diag.configure_query_region(cfg, args())
    assert not report['explicit_override']
    assert report['max'] == cfg.safety.workspace.max
    assert cfg.occupancy.as_dict() == before


@pytest.mark.parametrize('query_args', [
    args([.1, -.3, -.01]), args(None, [.58, .3, .55]),
    args([.1, -.3, -.01], [.58, .3, .55], 'none'),
    args([.1, -.3, -.01], [float('nan'), .3, .55]),
    args([.1, -.3, -.01], [float('inf'), .3, .55]),
    args([.1, -.3, -.01], [.1, .3, .55]),
    args([.1, -.3], [.58, .3, .55]),
])
def test_invalid_query_rejected_without_mutating_safety(query_args):
    cfg = load_demo_config(arm='isaac_kitchen_gpu')
    before = copy.deepcopy(cfg._data)
    with pytest.raises(ValueError):
        diag.configure_query_region(cfg, query_args)
    assert cfg._data == before


def records():
    return [{'sequence': i, 'client_started_monotonic': 10+i, 'physics': {
        'server_monotonic': 100+i, 'robot_id': 'robot', 'cameras': {
            name: {'available': True, 'capture_monotonic': 99.8+i,
                   'robot_id': 'robot', 'producer_time_source': 'physics_loop_monotonic'}
            for name in ('cam0', 'side', 'proof')}}} for i in range(3)]


def summary(rows, **kwargs):
    return diag.camera_age_summary(rows, host='127.0.0.1', port=8692,
        robot_id='robot', complete=kwargs.get('complete', True), errors=kwargs.get('errors', []))


def test_three_cameras_use_server_clock_and_keep_all_samples():
    r = summary(records())
    assert r['pass']
    assert r['source'] == ['127.0.0.1', 8692]
    for camera in r['cameras'].values():
        assert camera['samples'] == 3
        assert camera['maximum_age_s'] == pytest.approx(.2)
        assert camera['maximum_observed_capture_gap_s'] == 1


@pytest.mark.parametrize('fault', ['absent', 'wrong_robot', 'wrong_clock', 'stale', 'future',
                                  'nonfinite_capture', 'nonfinite_server', 'regression', 'frozen'])
def test_one_bad_camera_prevents_acceptance_without_dropping_evidence(fault):
    rows = records()
    frame = rows[1]['physics']['cameras']['side']
    if fault == 'absent':
        del rows[1]['physics']['cameras']['side']
    elif fault == 'wrong_robot':
        frame['robot_id'] = 'other'
    elif fault == 'wrong_clock':
        frame['producer_time_source'] = 'client_wall'
    elif fault == 'stale':
        frame['capture_monotonic'] = 98.
    elif fault == 'future':
        frame['capture_monotonic'] = 102.
    elif fault == 'nonfinite_capture':
        frame['capture_monotonic'] = float('nan')
    elif fault == 'nonfinite_server':
        rows[1]['physics']['server_monotonic'] = float('nan')
    elif fault == 'regression':
        frame['capture_monotonic'] = 99.7
    elif fault == 'frozen':
        for row in rows:
            row['physics']['cameras']['side']['capture_monotonic'] = 100.
    r = summary(rows)
    assert not r['pass']
    assert not r['cameras']['side']['pass']
    assert r['cameras']['side']['samples'] == 3
    if fault != 'frozen':
        assert r['cameras']['side']['first_invalid_sample']['sequence'] == 1
    if fault == 'stale':
        assert r['cameras']['side']['maximum_age_s'] == 3.
        assert r['cameras']['side']['samples_over_2s'] == 1


@pytest.mark.parametrize('kwargs', [{'complete': False}, {'errors': ['observer timeout']}])
def test_incomplete_observer_never_passes(kwargs):
    assert not summary(records(), **kwargs)['pass']


def test_manifest_includes_actual_clock_and_camera_auditor_sources():
    hashes = diag.frozen_sources(ROOT / 'demo/scene/kitchen_config.json')
    for name in ('src/cascade/control/isaac_arm.py', 'src/cascade/control/simulation_motion.py',
                 'configs/arms/isaac.yaml', 'demo/kitchen/physics/snapshot_codec.py',
                 'benchmark/diagnostics/nvblox_postclose_recovery.py'):
        assert name in hashes
    diag.check_frozen_sources(hashes)
    name = 'src/cascade/control/simulation_motion.py'
    hashes[name] = '0'*64
    with pytest.raises(RuntimeError, match='source changed'):
        diag.check_frozen_sources(hashes)


@pytest.mark.parametrize('arguments', [
    ['--query-region-min', '.1', '-.3', '-.01'],
    ['--query-region-min', '.1', '-.3', '-.01', '--query-region-max', 'nan', '.3', '.55'],
])
def test_cli_rejects_invalid_query_before_creating_output(tmp_path, arguments):
    with pytest.raises(SystemExit) as exc:
        diag.main(['--port', '8692', '--engine', 'physx', '--occupancy', 'nvblox',
                   '--output', str(tmp_path/'new'), *arguments])
    assert exc.value.code == 2
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('component', ['physics', 'cameras', 'frame'])
@pytest.mark.parametrize('value', [None, [], {}])
def test_malformed_camera_records_remain_serializable(component, value):
    import json
    rows = records()
    if component == 'physics':
        rows[1]['physics'] = value
    elif component == 'cameras':
        rows[1]['physics']['cameras'] = value
    else:
        rows[1]['physics']['cameras']['side'] = value
    report = summary(rows)
    assert not report['pass']
    assert report['cameras']['side']['first_invalid_sample']['sequence'] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
@pytest.mark.parametrize('field', ['capture_monotonic', 'server_monotonic'])
def test_nonfinite_camera_clock_is_preserved_as_typed_evidence(value, field):
    import json
    rows = records()
    if field == 'server_monotonic':
        rows[1]['physics'][field] = value
    else:
        rows[1]['physics']['cameras']['side'][field] = value
    report = summary(rows)
    assert not report['pass']
    encoded = json.dumps(report, allow_nan=False)
    assert 'invalid_value' in encoded
