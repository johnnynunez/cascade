"""A process return is not proof that every runtime teardown stage succeeded."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cascade.apps import demo, mcp_server
from cascade.control.arm_rig import ArmRig


def runtime(*, watcher=None, camera=None, arm_rig=None):
    return SimpleNamespace(arm=None, arm_rig=arm_rig, watcher=watcher, rig=None,
                           camera=camera or SimpleNamespace(close=lambda: None),
                           stream_server=None, viewer=None, beliefs_path=None)


def fail(message):
    def call():
        raise RuntimeError(message)
    return call


def test_stage_error_survives_and_remaining_teardown_runs():
    camera, arm = SimpleNamespace(close=Mock()), SimpleNamespace(disconnect=Mock())
    rt = runtime(watcher=SimpleNamespace(stop=fail('watcher could not close')), camera=camera)
    result = demo.shutdown_runtime(rt, arm)
    assert isinstance(result, dict), 'old teardown discarded the error and returned None'
    assert result['ok'] is False and result['complete'] is False
    assert any(row.get('stage') == 'watcher' and 'watcher could not close' in str(row) for row in result['stages'])
    camera.close.assert_called_once()
    arm.disconnect.assert_called_once()


def test_arm_rig_records_per_arm_error_and_continues():
    second = SimpleNamespace(disconnect=Mock(return_value=None))
    rig = ArmRig([SimpleNamespace(disconnect=fail('bus refused')), second], ['first', 'second'])
    result = rig.disconnect()
    assert isinstance(result, dict), 'old rig printed failure but provided no receipt'
    assert result['ok'] is False
    assert result['stages'][0]['stage'] == 'first' and result['stages'][0]['ok'] is False
    assert result['stages'][1]['ok'] is True
    second.disconnect.assert_called_once()


def test_correct_software_closure_is_explicit_without_physical_claim():
    result = demo.shutdown_runtime(runtime(), SimpleNamespace(disconnect=lambda: None))
    assert isinstance(result, dict)
    assert result['ok'] is True and result['complete'] is True
    assert result['physical_rest_verified'] is False


def test_pending_owned_worker_is_not_clean_teardown():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    try:
        result = demo.shutdown_runtime(runtime(watcher=SimpleNamespace(_thread=thread, stop=lambda: None)),
                                       SimpleNamespace(disconnect=lambda: None))
        assert isinstance(result, dict)
        assert result['ok'] is False and result['pending_threads']
    finally:
        release.set()
        thread.join(2)


def test_mcp_preserves_teardown_failure_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('CASCADE_RUN_DIR', str(tmp_path))
    server = mcp_server.McpSkillServer()
    server._runtime = runtime(watcher=SimpleNamespace(stop=fail('stop failed')))
    server._arm = SimpleNamespace(disconnect=lambda: None)
    result = server.shutdown()
    assert isinstance(result, dict)
    assert result['ok'] is False
    retained = json.loads((tmp_path / 'teardown.json').read_text())
    assert retained['ok'] is False and 'stop failed' in str(retained)


def test_mcp_main_does_not_return_clean_exit_after_teardown_failure(monkeypatch):
    server = SimpleNamespace(_signals=None, shutdown=lambda: {'ok': False, 'complete': False})
    monkeypatch.setattr(mcp_server, 'McpSkillServer', lambda: server)
    monkeypatch.setattr(mcp_server, '_serve_stdio', lambda *_: 0)
    monkeypatch.setattr('sys.argv', ['cascade-mcp'])
    assert mcp_server.main() != 0


@pytest.mark.parametrize('mode', ['mobile', 'composed'])
def test_non_arm_close_receipt_is_preserved_without_arm_commands(mode):
    rt = SimpleNamespace(robot_mode=mode, close=lambda: {'ok': False, 'pending_domains': ['base']})
    arm = SimpleNamespace(disconnect=Mock())
    result = demo.shutdown_runtime(rt, arm)
    assert isinstance(result, dict)
    assert result['ok'] is False and 'pending_domains' in str(result)
    arm.disconnect.assert_not_called()


def test_camera_rig_error_is_not_hidden_by_parent_stage():
    from cascade.perception.stream import CameraRig

    good = SimpleNamespace(name='good', close=Mock(return_value=None))
    cameras = CameraRig([SimpleNamespace(name='bad', close=fail('camera socket')), good])
    rt = runtime()
    rt.rig = cameras
    result = demo.shutdown_runtime(rt, SimpleNamespace(disconnect=lambda: None))
    stage = next(item for item in result['stages'] if item['stage'] == 'cameras')
    assert result['ok'] is False and stage['ok'] is False
    assert 'camera socket' in str(stage)
    good.close.assert_called_once()


def test_failed_receipt_is_sticky_and_does_not_repeat_parking_or_disconnect():
    close = Mock(side_effect=RuntimeError('retained failure'))
    rt, arm = runtime(), SimpleNamespace(disconnect=close)
    first = demo.shutdown_runtime(rt, arm)
    first['stages'].clear()  # callers cannot rewrite the stored history
    close.side_effect = None
    second = demo.shutdown_runtime(rt, arm)
    assert second['ok'] is False and 'retained failure' in str(second)
    close.assert_called_once()


def test_false_park_result_is_retained_and_disconnect_still_runs():
    safe = SimpleNamespace(harness=SimpleNamespace(estopped=False, park_q=[0], kin=None),
                           raw=SimpleNamespace(connected=True), n_joints=1,
                           move_joints=Mock(return_value=False))
    rt = runtime()
    rt.arm = safe
    backend = SimpleNamespace(disconnect=Mock(return_value=None))
    receipt = demo.shutdown_runtime(rt, backend)
    assert receipt['ok'] is False and 'ParkIncomplete' in str(receipt)
    backend.disconnect.assert_called_once()
    assert safe.move_joints.call_args.kwargs['joint_margin'] == 0.0


def test_contradictory_close_with_pending_is_not_completed():
    from cascade.lifecycle import teardown_step

    stage = teardown_step('domain', lambda: {'ok': True, 'pending_providers': ['camera']})
    assert stage['ok'] is False and stage['result']['pending_providers'] == ['camera']


def test_mcp_persistence_failure_is_retained_and_not_clean(tmp_path, monkeypatch):
    blocked = tmp_path / 'file-not-directory'
    blocked.write_text('keep')
    monkeypatch.setenv('CASCADE_RUN_DIR', str(blocked))
    server = mcp_server.McpSkillServer()
    receipt = server.shutdown()
    assert receipt['ok'] is False
    assert receipt['stages'][-1]['stage'] == 'persist_receipt'
    assert server.shutdown() == receipt
    assert blocked.read_text() == 'keep'


@pytest.mark.parametrize('code', [0, 143])
def test_mcp_main_preserves_return_when_software_teardown_completed(monkeypatch, code):
    server = SimpleNamespace(_signals=None, shutdown=lambda: {'ok': True, 'complete': True})
    monkeypatch.setattr(mcp_server, 'McpSkillServer', lambda: server)
    monkeypatch.setattr(mcp_server, '_serve_stdio', lambda *_: code)
    monkeypatch.setattr('sys.argv', ['cascade-mcp'])
    assert mcp_server.main() == code


def test_composed_arm_adapter_preserves_failed_close(monkeypatch):
    from cascade.apps.robot_runtime import DomainAdapter

    adapter = object.__new__(DomainAdapter)
    adapter.profile = {'kind': 'manipulation'}
    adapter.runtime = runtime(watcher=SimpleNamespace(stop=fail('worker failure')))
    adapter.owner = SimpleNamespace(disconnect=lambda: None)
    receipt = adapter.close()
    assert receipt['ok'] is False and 'worker failure' in str(receipt)
