"""A cancelled or timed-out batch must never reuse its delayed response."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('zmq')
pytest.importorskip('msgpack_numpy')
import msgpack
import zmq

from cascade.grasping.graspgenx_backend import GraspGenXClient, GraspGenXError, NoEligibleGrasps
from cascade.types import MotionHalted
from test_graspgenx_backend import _planner, _cube_fix


def server(response, delay=0.):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    port = sock.bind_to_random_port('tcp://127.0.0.1')
    received = threading.Event()
    def worker():
        try:
            if sock.poll(1500):
                sock.recv()
                received.set()
                time.sleep(delay)
                sock.send(msgpack.packb(response, use_bin_type=True))
        finally:
            sock.close()
    thread = threading.Thread(target=worker)
    thread.start()
    return port, received, thread, ctx


def test_bounded_request_success_uses_same_transport_without_actuators():
    port, _, thread, ctx = server({'status': 'ok'})
    client = GraspGenXClient(port=port)
    try:
        assert client.request({'action': 'health'}, deadline=time.monotonic()+1)['status'] == 'ok'
    finally:
        client.close(); thread.join(2); ctx.term()


@pytest.mark.parametrize('cancel', [False, True])
def test_waiting_reply_timeout_or_halt_discards_socket_and_late_reply(cancel):
    port, got, thread, ctx = server({'status': 'late'}, delay=.2)
    client = GraspGenXClient(port=port)
    def check():
        if cancel and got.is_set():
            raise MotionHalted('new instruction')
    started = time.monotonic()
    try:
        with pytest.raises(MotionHalted if cancel else GraspGenXError):
            client.request({'action': 'health'}, deadline=started+.08, check=check)
        assert client._sock is None
    finally:
        client.close(); thread.join(2); ctx.term()


def test_reply_that_arrives_after_deadline_is_never_returned(monkeypatch):
    client = GraspGenXClient()
    now = [0.]
    from cascade.grasping import graspgenx_backend as module
    # Do not monkeypatch process-wide time used by unrelated threads.
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    closed = []
    class Socket:
        def poll(self, timeout, event):
            if event == zmq.POLLIN: now[0] = 1.1
            return event
        def send(self, data, flags): now[0] = .7
        def recv(self, flags): pytest.fail('late response must not be consumed')
        def close(self): closed.append(True)
    client._sock = Socket()
    with pytest.raises(GraspGenXError, match='timed out'):
        client.request({'action': 'health'}, deadline=1.)
    assert closed == [True] and client._sock is None


@pytest.mark.parametrize('kind', ['nan', 'mismatch', 'shape', 'empty'])
def test_bad_or_empty_model_responses_are_not_feasibility_exhaustion(kind, monkeypatch):
    planner = _planner(0, min_score=.25, approach_z_max=-.95)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, :3] = np.diag([1., -1., -1.])
    raw = {'grasps': pose, 'confidences': np.array([.8])}
    if kind == 'nan': raw['confidences'][0] = np.nan
    if kind == 'mismatch': raw['confidences'] = np.array([.8, .9])
    if kind == 'shape': raw['grasps'] = [1, 2]
    if kind == 'empty': raw = {'grasps': np.empty((0,4,4)), 'confidences': []}
    calls = []
    planner._client = SimpleNamespace(request=lambda *a, **kw: calls.append(kw) or raw)
    monkeypatch.setattr('cascade.grasping.graspgenx_backend.np.savez', lambda *a, **kw: None)
    with pytest.raises((GraspGenXError, ValueError)) as err:
        planner.plan(_cube_fix(), deadline=time.monotonic()+1)
    assert not isinstance(err.value, NoEligibleGrasps)
    assert len(calls) == 1


def test_nonempty_valid_model_batch_removed_by_score_is_explicit_feasibility():
    planner = _planner(0, min_score=.25, approach_z_max=-.95)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, :3] = np.diag([1., -1., -1.])
    planner._client = SimpleNamespace(request=lambda *a, **kw: {'grasps': pose, 'confidences': [.1]})
    with pytest.raises(NoEligibleGrasps):
        planner.plan(_cube_fix(), deadline=time.monotonic()+1)


@pytest.mark.parametrize('kind', ['rotation_zero', 'rotation_scaled', 'reflection', 'last_row', 'bool', 'string', 'score_bool', 'score_string'])
def test_low_score_does_not_hide_malformed_pose_or_wire_types(kind):
    planner = _planner(0, min_score=.25, approach_z_max=-.95)
    pose = np.eye(4, dtype=np.float32)[None]
    pose[0, :3, :3] = np.diag([1., -1., -1.])
    scores = np.array([.1])
    if kind == 'rotation_zero': pose[0, :3, :3] = 0
    if kind == 'rotation_scaled': pose[0, :3, :3] *= 2
    if kind == 'reflection': pose[0, :3, :3] = -np.eye(3)
    if kind == 'last_row': pose[0, 3, 0] = .1
    if kind == 'bool': pose = pose.astype(bool)
    if kind == 'string': pose = pose.astype(str)
    if kind == 'score_bool': scores = np.array([False])
    if kind == 'score_string': scores = np.array(['0.1'])
    planner._client = SimpleNamespace(request=lambda *a, **kw: {'grasps': pose, 'confidences': scores})
    with pytest.raises(GraspGenXError, match='malformed') as err:
        planner.plan(_cube_fix(), deadline=time.monotonic()+1)
    assert not isinstance(err.value, NoEligibleGrasps)
