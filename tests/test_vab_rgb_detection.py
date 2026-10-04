import json
from types import SimpleNamespace

import pytest

from benchmark.vab.detect_rgb import create_detector, detect_archive, weights_digest
from test_vab_rgb_reconstruct import archive


def test_archived_detection_uses_exactly_one_instance_and_two_observed_views(tmp_path):
    directory = archive(tmp_path)
    calls = []
    def detect(frame, **kwargs):
        calls.append(frame)
        return []
    factory_calls = []
    def factory():
        factory_calls.append(True)
        return SimpleNamespace(detect=detect)
    result = detect_archive(directory, factory)
    assert factory_calls == [True] and len(calls) == 2
    assert result['captures'][0]['status'] == 'unverified'
    assert len(result['captures'][1]['views']) == 2
    assert all(v['detections'] == [] for v in result['captures'][1]['views'])
    assert result['physics_steps_advanced'] == result['simulator_models_constructed'] == 0
    assert not result['physical_admission'] and not result['object_identity_verified']
    json.dumps(result, allow_nan=False)


def test_bad_late_capture_refuses_before_detector_construction(tmp_path):
    directory = archive(tmp_path)
    path = directory/'0001.json'
    value = json.loads(path.read_bytes())
    value['views'][1]['capture_sha256'] = '0'*64
    path.write_text(json.dumps(value))
    def factory():
        pytest.fail('no model construction after invalid capture')
    with pytest.raises(ValueError, match='identity differ'):
        detect_archive(directory, factory)


def test_archive_changed_during_detection_is_refused(tmp_path):
    directory = archive(tmp_path)
    def mutate(*args, **kwargs):
        (directory/'0000.json').write_text('{}')
        return []
    with pytest.raises(ValueError, match='changed during'):
        detect_archive(directory, lambda: SimpleNamespace(detect=mutate))


def test_missing_or_pointer_weights_refused_without_loading_model(tmp_path):
    path = tmp_path/'yoloe-11s-seg-pf.pt'
    with pytest.raises(ValueError, match='checkpoint'):
        weights_digest(path)
    path.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='checkpoint'):
        weights_digest(path)


def test_weights_rechecked_at_loading_boundary_after_archive_validation(tmp_path, monkeypatch):
    import cascade.perception.detector as implementation
    path = tmp_path/'yoloe-11s-seg-pf.pt'
    path.write_bytes(b'a' * 1024**2)
    before = weights_digest(path)
    path.write_bytes(b'b' * 1024**2)
    def unexpected(*args, **kwargs):
        pytest.fail('changed checkpoint reached model constructor')
    monkeypatch.setattr(implementation, 'OpenVocabDetector', unexpected)
    with pytest.raises(ValueError, match='checkpoint'):
        create_detector(path, before, 'cpu')
