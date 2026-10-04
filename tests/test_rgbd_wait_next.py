"""Real bounded TCP readers and cache publication; no SDK or physics execution."""
import threading
import time

import pytest

from cascade.sensing import build_sensor_domain
from cascade.sim.mobile_rgbd import calibration_record
from test_sensing_rgbd import calibration, endpoint as endpoint, reader


def waiting_read(cache, client, monkeypatch):
    entered = threading.Event()
    wait = cache._published.wait
    def observed(timeout=None):
        entered.set()
        return wait(timeout)
    monkeypatch.setattr(cache._published, 'wait', observed)
    result = []
    worker = threading.Thread(target=lambda: result.append(client()))
    worker.start()
    assert entered.wait(2), 'reader did not enter publication wait'
    return worker, result


def finish(worker, result):
    worker.join(2)
    assert not worker.is_alive()
    assert len(result) == 1
    return result[0]


def test_opt_in_waits_for_completed_publication_without_controller_access(endpoint, monkeypatch):
    controller, server, cache, profile, _, _, publish, operations = endpoint
    publish()
    client = reader(profile, max_age_s=5., wait_next=True)
    try:
        first = client(); assert first is not None, client.last_error
        worker, result = waiting_read(cache, client, monkeypatch)
        assert not result and cache._frame['step'] == 10
        assert controller.hello()['generation'] == 0 and not server._owners
        publish(11)
        second = finish(worker, result)
        assert second is not None, client.last_error
        assert first.metadata['step'] == 10 and second.metadata['step'] == 11
        assert second.metadata['sim_time_s'] == .055
        request = operations[-1]
        assert request['wait_next']['epoch'] == first.metadata['epoch']
        assert request['wait_next']['after_step'] == 10
        assert 0 < request['wait_next']['timeout_s'] <= profile['timeout_s']
        assert {row['op'] for row in operations} == {'hello', 'frame'}
    finally:
        cache.close(); client.close()


def test_new_capture_after_wait_retains_original_producer_age(endpoint, monkeypatch):
    _, _, cache, profile, _, _, publish, _ = endpoint
    publish(); client = reader(profile, max_age_s=5., wait_next=True)
    try:
        assert client() is not None
        worker, result = waiting_read(cache, client, monkeypatch)
        publish(11, age=6.)
        assert finish(worker, result) is None
        assert 'stale' in client.last_error
        assert client._seen.metadata['step'] == 11
        assert client._seen.metadata['producer_age_s'] >= 6.
    finally:
        cache.close(); client.close()


@pytest.mark.parametrize('closure', ['cache', 'server'])
def test_close_releases_waiter_and_original_worker_bounds(endpoint, monkeypatch, closure):
    _, server, cache, profile, _, _, publish, _ = endpoint
    publish(); client = reader(profile, max_age_s=5., wait_next=True)
    try:
        assert client() is not None
        worker, result = waiting_read(cache, client, monkeypatch)
        (cache if closure == 'cache' else server).close()
        assert finish(worker, result) is None
        server.close()
        assert not server._workers
    finally:
        cache.close(); client.close()


@pytest.mark.parametrize('change,reason', [
    ({'epoch': 'other'}, 'epoch'), ({'after_step': 11}, 'future'),
    ({'after_step': True}, 'after_step'), ({'timeout_s': 1.01}, 'timeout'),
    ({'timeout_s': float('nan')}, 'timeout'), ({'extra': 0}, 'exact'),
])
def test_wait_refuses_unknown_epoch_future_step_and_unbounded_request(endpoint, change, reason):
    _, _, cache, profile, _, _, publish, _ = endpoint
    publish()
    request = {'camera': 'overview', 'modality': 'rgbd', 'wait_next': {
        'epoch': profile['epoch'], 'after_step': 10, 'timeout_s': .1, **change}}
    with pytest.raises((ValueError, TypeError), match=reason):
        cache(request)


def test_wait_deadline_is_not_renewed_even_if_newer_capture_exists(endpoint):
    _, _, cache, profile, _, _, publish, _ = endpoint
    publish(11)
    request = {'camera': 'overview', 'modality': 'rgbd', 'wait_next': {
        'epoch': profile['epoch'], 'after_step': 10, 'timeout_s': 1.},
        '_frame_deadline': time.monotonic()-1.}
    with pytest.raises(RuntimeError, match='deadline'):
        cache(request)
    assert cache._frame['step'] == 11


def test_socket_wait_cannot_supply_transport_deadline_or_cancellation(endpoint):
    from cascade.sim.mobile_frames import _FrameRPC
    _, server, _, profile, _, _, publish, _ = endpoint
    publish()
    client = _FrameRPC(profile)
    try:
        client.connect(1.)
        assert client.request({'op': 'hello', 'role': 'reader'}, timeout_s=1.)['ok']
        response = client.request({'op': 'frame', 'camera': 'overview', 'modality': 'rgbd',
            'wait_next': {'epoch': profile['epoch'], 'after_step': 9, 'timeout_s': 1.},
            '_frame_deadline': -1., '_frame_cancelled': True}, timeout_s=1.)
        assert response['rgbd']['step'] == 10 and not server._owners
    finally:
        client.close()


def test_wait_timeout_does_not_publish_or_refresh_snapshot(endpoint):
    _, _, cache, profile, _, _, publish, _ = endpoint
    publish(age=.2); captured = cache._captured_at
    with pytest.raises(RuntimeError, match='deadline'):
        cache({'camera': 'overview', 'modality': 'rgbd', 'wait_next': {
            'epoch': profile['epoch'], 'after_step': 10, 'timeout_s': .01}})
    assert cache._captured_at == captured and cache._frame['step'] == 10


def test_epoch_change_during_wait_refuses_publication(endpoint, monkeypatch):
    _, _, cache, profile, _, _, publish, _ = endpoint
    publish(); client = reader(profile, max_age_s=5., wait_next=True)
    try:
        assert client() is not None
        worker, result = waiting_read(cache, client, monkeypatch)
        with cache._published:
            cache._frame = {**cache._frame, 'epoch': 'foreign', 'step': 11}
            cache._published.notify_all()
        assert finish(worker, result) is None
        assert client._seen.metadata['step'] == 10
    finally:
        cache.close(); client.close()


def test_opt_in_requires_advertised_capability_and_profile_boolean(endpoint):
    _, server, cache, profile, _, _, publish, operations = endpoint
    publish(); cache.rgbd_wait_next = False
    client = reader(profile, wait_next=True)
    try:
        assert client() is None and 'advertise' in client.last_error
        assert all(row['op'] == 'hello' for row in operations)
    finally:
        client.close()
    for value in (1, 'true', None):
        with pytest.raises(ValueError, match='boolean'):
            reader(profile, wait_next=value)
    domain = build_sensor_domain('sensing', {'kind': 'sensors', 'robot_id': profile['robot_id'],
        'read_timeout_s': 2., 'providers': [{'id': 'overview', 'kind': 'mobile_rgbd',
            'profile': profile, 'camera': 'overview', 'wait_next': True,
            'calibration_sha256': calibration_record(calibration())[1]}]})
    assert domain.close()['ok']


def test_legacy_reader_still_refuses_replayed_capture(endpoint):
    _, _, _, profile, _, _, publish, operations = endpoint
    publish(); client = reader(profile)
    try:
        assert client() is not None
        assert client() is None and 'replay' in client.last_error
        assert all('wait_next' not in row for row in operations)
    finally:
        client.close()
