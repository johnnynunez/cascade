"""Exercise camera recovery with producer clocks and an owned fake transport."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "kitchen_camera_recovery", Path(__file__).resolve().parents[1] / "demo/serve_isaac_view.py")
camera = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera)


def capture(name, stamp):
    return SimpleNamespace(frame_id=0, capture={
        "camera": name, "t": stamp,
        "proprioception": {"robot_id": "/robot", "t": stamp,
                           "time_source": "physics_loop_monotonic"},
    })


def run_rounds(rounds):
    """Each round supplies receiver time and one capture/error per camera."""
    index = 0
    clients = []
    published = []

    class Stop:
        def is_set(self):
            return index >= len(rounds)

        def wait(self, seconds):
            nonlocal index
            published.append({
                "stats": rig.stats(), "state": rig.state(), "wait_s": seconds,
                "frames": {name: rig.latest(name) for name, _ in camera.CAMERAS},
            })
            index += 1
            return self.is_set()

    class Client:
        closed = False

        def connect(self):
            clients.append(self)

        def ping(self):
            return True

        def observation(self, name):
            value = rounds[index][1][name]
            if isinstance(value, Exception):
                raise value
            return value

        def close(self):
            self.closed = True

    def clock():
        return rounds[min(index, len(rounds)-1)][0]

    rig = camera.RecoveringViewRig(client_factory=Client, stop_event=Stop(), clock=clock)
    rig._run()
    assert all(client.closed for client in clients)
    assert all(rig.latest(name) is None for name, _ in camera.CAMERAS)
    return clients, published


@pytest.mark.parametrize("stale", [{"proof"}, {"proof", "cam0", "side"}])
def test_slow_render_recovers_on_same_connection_without_showing_stale_frames(stale):
    rounds = []
    for index, now in enumerate([100., 102.2, 102.3]):
        frames = {name: capture(name, 100. if index == 1 and name in stale else now)
                  for _, name in camera.CAMERAS}
        rounds.append((now, frames))
    clients, published = run_rounds(rounds)
    assert len(clients) == 1
    assert all(row["state"]["connection_generation"] == 1 for row in published)
    assert all(row["state"]["retry_in_s"] == 0 for row in published)
    first, delayed, recovered = published
    for public, source in camera.CAMERAS:
        assert first["stats"][public]["online"]
        assert delayed["stats"][public]["online"] is (source not in stale)
        assert (delayed["frames"][public] is None) is (source in stale)
        if source in stale:
            assert delayed["stats"][public]["frame_id"] == 0
            assert delayed["state"]["capture"][public] is None
        assert recovered["stats"][public]["online"]
        assert recovered["frames"][public].capture["t"] == 102.3
    assert "waiting for fresh cameras" in delayed["state"]["agent_status"]
    assert recovered["state"]["stream_status"] == "online"


@pytest.mark.parametrize("failure", ["socket", "identity", "future", "nonfinite", "backwards"])
def test_bad_transport_or_capture_identity_still_discards_frames_and_reconnects(failure):
    rounds = [(now, {name: capture(name, now) for _, name in camera.CAMERAS})
              for now in [100., 100.1, 100.2]]
    if failure == "socket":
        rounds[1][1]["proof"] = OSError("bridge unavailable")
    elif failure == "identity":
        rounds[1][1]["proof"].capture["camera"] = "another-camera"
    else:
        stamp = {"future": 101., "nonfinite": float("nan"), "backwards": 99.9}[failure]
        rounds[1][1]["proof"] = capture("proof", stamp)
    clients, published = run_rounds(rounds)
    assert len(clients) == 2
    failed = published[1]
    assert all(not value["online"] for value in failed["stats"].values())
    assert all(value is None for value in failed["frames"].values())
    assert failed["state"]["bridge_error"]
    assert failed["wait_s"] == .25
    assert all(value["online"] for value in published[2]["stats"].values())
    assert published[2]["state"]["connection_generation"] == 2
