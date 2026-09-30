"""A real delayed REP peer reproduces concurrent probe/refresh REQ corruption."""
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import msgpack
import zmq

from cascade.perception.occupancy import OccupancyClient


def test_simultaneous_occupancy_requests_keep_their_own_responses(loopback):
    ready = Queue()

    def server():
        sock = zmq.Context.instance().socket(zmq.REP)
        try:
            sock.setsockopt(zmq.RCVTIMEO, 5000)
            ready.put(sock.bind_to_random_port(f"tcp://{loopback}"))
            for _ in range(4):
                payload = msgpack.unpackb(sock.recv(), raw=False)
                time.sleep(.025)  # force send/recv overlap in the old client
                sock.send(msgpack.packb(payload, use_bin_type=True))
        finally:
            sock.close(0)

    thread = Thread(target=server, daemon=True)
    thread.start()
    client = OccupancyClient(host=loopback, port=ready.get(timeout=2), timeout_ms=2000)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = list(pool.map(lambda i: client.request({"id": i}), range(4)))
        assert replies == [{"id": i} for i in range(4)]
    finally:
        client.close()
        thread.join(timeout=6)
