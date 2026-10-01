"""A single absolute deadline covers bridge lock, send and response reads."""
import json
import socket
import threading
import time

import pytest

from cascade.sim.bridge_client import BridgeClient, BridgeError


def connected_pair():
    client, server = socket.socketpair()
    bridge = BridgeClient(timeout_s=1.)
    bridge._sock = client
    return bridge, server


def test_transport_lock_wait_is_bounded_and_does_not_close_other_request():
    bridge, server = connected_pair()
    bridge._lock.acquire()
    started = time.monotonic()
    try:
        with pytest.raises(BridgeError, match='transport lock'):
            bridge.request({'op': 'state'}, timeout_s=.03)
        assert .02 <= time.monotonic() - started < .3
        assert bridge._sock is not None
        server.setblocking(False)
        with pytest.raises(BlockingIOError):
            server.recv(1)
    finally:
        bridge._lock.release()
        bridge.close()
        server.close()


def test_slow_drip_response_cannot_extend_absolute_deadline():
    bridge, server = connected_pair()
    stopped = threading.Event()
    def drip():
        try:
            server.recv(4096)
            for byte in b'{"ok":true,"slow":1}\n':
                if stopped.wait(.01):
                    break
                server.sendall(bytes([byte]))
        except OSError:
            pass
    worker = threading.Thread(target=drip)
    worker.start()
    started = time.monotonic()
    try:
        with pytest.raises(BridgeError, match='I/O failed'):
            bridge.request({'op': 'state'}, timeout_s=.07)
        assert .05 <= time.monotonic() - started < .4
        assert bridge._sock is None
    finally:
        stopped.set()
        worker.join(timeout=.5)
        bridge.close()
        server.close()
    assert not worker.is_alive()


def test_late_reply_cannot_be_consumed_by_a_subsequent_request():
    bridge, server = connected_pair()
    try:
        with pytest.raises(BridgeError):
            bridge.request({'op': 'set_joints', 'q': [1.]}, timeout_s=.03)
        assert json.loads(server.recv(4096))['op'] == 'set_joints'
        assert bridge._sock is None
        with pytest.raises(BridgeError, match='not connected'):
            bridge.request({'op': 'state'}, timeout_s=.03)
        with pytest.raises(BrokenPipeError):
            server.sendall(b'{"ok":true,"for":"old-request"}\n')
        # Explicit reconnect is required; the new connection cannot inherit bytes.
        client2, server2 = socket.socketpair()
        bridge._sock = client2
        server2.sendall(b'{"ok":true,"for":"new-connection"}\n')
        try:
            assert bridge.request({'op':'state'})['for'] == 'new-connection'
        finally:
            server2.close()
    finally:
        bridge.close()
        server.close()


def test_send_blocking_uses_the_same_finite_deadline():
    bridge, server = connected_pair()
    bridge._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    started = time.monotonic()
    try:
        with pytest.raises(BridgeError, match='I/O failed'):
            bridge.request({'op':'test', 'data':'x' * 1000000}, timeout_s=.03)
        assert time.monotonic() - started < .4
        assert bridge._sock is None
    finally:
        bridge.close()
        server.close()


@pytest.mark.parametrize('response', [b'[]\n', b'not-json\n', b'{"ok":true}\nextra'])
def test_malformed_wire_closes_connection(response):
    bridge, server = connected_pair()
    try:
        server.sendall(response)
        with pytest.raises(BridgeError):
            bridge.request({'op':'state'})
        assert bridge._sock is None
    finally:
        bridge.close()
        server.close()


@pytest.mark.parametrize('budget', [0, -1, True, float('nan'), float('inf'), '1'])
def test_invalid_deadline_fails_before_send(budget):
    bridge, server = connected_pair()
    try:
        with pytest.raises(BridgeError, match='finite and positive'):
            bridge.request({'op':'state'}, timeout_s=budget)
        assert bridge._sock is not None
    finally:
        bridge.close()
        server.close()
