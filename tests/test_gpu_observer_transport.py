"""Real TCP transport regressions with synthetic replies, never live acceptance."""
import importlib.util
import json
from pathlib import Path
import socketserver
import tempfile
import threading
import time

import pytest

from conftest import loopback_host


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu_observer_transport", ROOT / "demo/kitchen/physics/gpu_proof_audit.py")
gpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu)


def snapshot_reply(sequence):
    sample = {"version": 1, "channel": "physics_tensor", "synthetic_test_only": True,
              "test_sequence": sequence}
    return {"ok": True, "stdout": "KITCHEN_OBSERVER " + json.dumps(sample) + "\n"}


class QueuedBridge(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            request = json.loads(raw)
            self.server.requests.append(request)
            if request["op"] == "ping":
                reply = {"ok": True, "synthetic_test_only": True}
            elif request["op"] == "exec":
                sequence = self.server.exec_count
                self.server.exec_count += 1
                if sequence == 1:
                    self.server.queued.set()
                    if not self.server.release.wait(timeout=45):
                        raise RuntimeError("Test did not release its queued reply")
                    reply = self.server.second_reply
                else:
                    reply = snapshot_reply(sequence)
            else:
                raise AssertionError("Observer sent a non-observation request")
            self.wfile.write(json.dumps(reply).encode() + b"\n")
            self.wfile.flush()


@pytest.fixture
def queued_bridge():
    server = socketserver.ThreadingTCPServer((loopback_host(), 0), QueuedBridge)
    server.daemon_threads = True
    server.requests = []
    server.exec_count = 0
    server.queued = threading.Event()
    server.release = threading.Event()
    server.second_reply = snapshot_reply(1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def observer_dir():
    # The observer intentionally confines its artifacts to its source checkout.
    with tempfile.TemporaryDirectory(prefix=".test-gpu-observer-", dir=ROOT) as directory:
        yield Path(directory) / "evidence"


def observer(server, out):
    return gpu.GpuProofObserver(out, host=server.server_address[0],
                                port=server.server_address[1], interval_s=.05, budget_s=60)


def test_stop_retains_real_tcp_reply_queued_beyond_old_deadlines(queued_bridge, observer_dir):
    witness = observer(queued_bridge, observer_dir)
    with witness:
        assert queued_bridge.queued.wait(timeout=2)
        # Exceed both the former 8-second I/O deadline and 12-second stop join.
        # Keep real sockets and clocks so the prior implementation loses evidence.
        timer = threading.Timer(12.25, queued_bridge.release.set)
        timer.start()
        started = time.monotonic()
        try:
            witness.stop()
            assert time.monotonic() - started >= 12
            assert not witness._thread.is_alive()
            assert witness.complete
            assert witness.errors == []
        finally:
            queued_bridge.release.set()
            timer.cancel()
            timer.join(timeout=2)

    rows = [json.loads(line) for line in (observer_dir / "samples.jsonl").read_text().splitlines()]
    wires = [json.loads(line) for line in (observer_dir / "wire.jsonl").read_text().splitlines()]
    assert rows == witness.records
    assert [row["sequence"] for row in rows] == [0, 1]
    assert [row["physics"]["test_sequence"] for row in rows] == [0, 1]
    assert rows[1]["client_finished_monotonic"] - rows[1]["client_started_monotonic"] >= 12
    assert [wire["reply"] for wire in wires] == [snapshot_reply(0), snapshot_reply(1)]
    assert [request["op"] for request in queued_bridge.requests] == ["ping", "exec", "exec", "ping"]
    metadata = json.loads((observer_dir / "metadata.json").read_text())
    assert metadata["complete"] and metadata["samples"] == 2 and metadata["errors"] == []


def test_server_exec_error_still_stops_observation_without_retry(queued_bridge, observer_dir):
    queued_bridge.second_reply = {"ok": False, "error": "exec timed out waiting for the main loop"}
    witness = observer(queued_bridge, observer_dir)
    with witness:
        assert queued_bridge.queued.wait(timeout=2)
        queued_bridge.release.set()
        witness._thread.join(timeout=2)
        assert not witness._thread.is_alive()

    assert not witness.complete
    assert len(witness.records) == 1
    assert len(witness.errors) == 1 and "exec timed out waiting for the main loop" in witness.errors[0]
    assert [request["op"] for request in queued_bridge.requests] == ["ping", "exec", "exec"]
    metadata = json.loads((observer_dir / "metadata.json").read_text())
    assert not metadata["complete"] and metadata["errors"] == witness.errors
