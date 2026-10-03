"""A real control-channel EOF withdraws queued mutation before the next solve."""
import json
import selectors
import socket

import pytest
from test_microduck_admission import pending, request, setup
from test_mobile_bridge import channel

from cascade.sim.mobile_bridge import MobileBridgeServer


@pytest.mark.parametrize('operation', ['command_velocity', 'reset_stop'])
@pytest.mark.parametrize('disconnect_at', ['before_drain', 'after_callback', 'before_reply'])
def test_tcp_eof_withdraws_queued_command_and_reset(operation, disconnect_at, monkeypatch):
    import cascade.sim.microduck_admission as module

    fleet, owner, steppers, admission = setup()
    controller, peer = (stepper.controller for stepper in steppers)
    server = MobileBridgeServer(admission.endpoint('duck0'), port=0)
    control = None
    mutations = []
    try:
        if operation == 'reset_stop':
            controller.stop()
        server.start()
        control = channel(server, 'control', 'client')
        with server._guard:
            server_socket = server._owners['client']
            worker, = server._workers

        def disconnect():
            control._sock.shutdown(socket.SHUT_RDWR)
            control.close()
            # Synchronize actual kernel EOF readiness, not the handler's
            # finally/owner_disconnected path (it is still awaiting admission).
            with selectors.DefaultSelector() as selector:
                selector.register(server_socket, selectors.EVENT_READ)
                assert selector.select(timeout=1), 'control FIN did not arrive'

        original = getattr(controller, operation)

        def mutate(arguments):
            result = original(arguments)
            mutations.append(result)
            if disconnect_at == 'after_callback':
                disconnect()
            return result

        monkeypatch.setattr(controller, operation, mutate)
        if disconnect_at == 'before_reply':
            request_type = module._Request

            def delayed_reply(*args):
                item = request_type(*args)
                wait = item.event.wait

                def wait_then_disconnect(timeout):
                    arrived = wait(timeout)
                    if arrived:
                        disconnect()
                    return arrived

                item.event.wait = wait_then_disconnect
                return item

            monkeypatch.setattr(module, '_Request', delayed_reply)

        wire = {'op': operation, **request(controller)}
        # Reset's wire API may omit owner; the real server must bind it.
        if operation == 'reset_stop':
            wire.pop('owner')
        control._sock.sendall((json.dumps(wire) + '\n').encode())
        pending(admission)
        with admission._lock:
            item = admission._pending['duck0']
        if disconnect_at == 'before_drain':
            disconnect()
        admission.drain()
        if disconnect_at == 'before_reply':
            # Wait for the real connection worker to observe the guard before
            # its response; its closure also proves no blocked reader remains.
            worker.join(timeout=1)
            assert not worker.is_alive()
        assert item.error is not None
        assert 'owner channel closed' in str(item.error)
        assert len(mutations) == (0 if disconnect_at == 'before_drain' else 1)
        assert controller._active is None
        assert controller.control_at(owner.physics_clock[1]) == (0., 0., 0.)
        if operation == 'reset_stop' or disconnect_at != 'before_drain':
            assert controller.state()['latched']
        assert peer.hello()['generation'] == 0
        assert owner.step_count == 3
    finally:
        admission.close()
        if control is not None:
            control.close()
        server.close()
        fleet.close()
