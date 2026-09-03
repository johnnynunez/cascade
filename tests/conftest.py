import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cascade.config import load_demo_config  # noqa: E402

URDF = REPO / "assets" / "urdf" / "00-arm-rs_asm-v3" / "urdf" / "00-arm-rs_asm-v3.urdf"
# Renamed 2026-07-31 with the asset refresh: the stage is now named after the
# robot (00-arm-rs_asm-v3.usda -> RS-rebot-dev-arm.usda), and its payloads
# follow the same convention (payloads/RS-rebot-dev-arm_{base,meshes,physics}).
USD = REPO / "assets" / "usd" / "RS-rebot-dev-arm" / "RS-rebot-dev-arm.usda"
# Assets are authored in the mirrored joint convention; the SDK and every q
# constant in this repo are local (q_local = -q_asset). See usd_model.
JOINT_SIGNS = [-1, -1, -1, -1, -1, -1]


def has_pinocchio() -> bool:
    try:
        import pinocchio  # noqa: F401

        return True
    except ImportError:
        return False


needs_pin = pytest.mark.skipif(
    not has_pinocchio() or not URDF.exists(),
    reason="pinocchio or RS model not available",
)

#: The SO-101 URDF is vendored (text only, no meshes needed for kinematics), so
#: unlike the RS model it is always present in a checkout.
SO101_URDF = REPO / "assets" / "urdf" / "so101" / "so101.urdf"

needs_pin_so101 = pytest.mark.skipif(
    not has_pinocchio() or not SO101_URDF.exists(),
    reason="pinocchio or SO-101 model not available",
)


@pytest.fixture(autouse=True)
def _isolate_persistent_beliefs(monkeypatch, tmp_path):
    """No test may read or write the SHARED persistent world model.

    `build_runtime` now restores beliefs from `runs/beliefs.json` (or
    `CASCADE_BELIEFS_PATH`) so the world model survives a restart. Without this
    fixture two things go wrong, and both were observed:

    * a developer's real run leaks into the suite -- a remembered object at a
      position where the mock scene has nothing makes `pick_and_place` target
      empty table, fail, and escalate to the LLM, which broke
      `test_orchestrator_reflex_path_never_calls_llm` with an error that looks
      nothing like its cause;
    * tests leak into each other, and into the developer's real memory file.

    Persistence itself is covered directly in
    tests/test_persistent_memory_curriculum.py against a tmp_path, which is
    where that behaviour belongs. Everything else runs with a private, empty
    world -- opt back in per test with monkeypatch if you need it.
    """
    monkeypatch.setenv("CASCADE_BELIEFS_PATH", str(tmp_path / "beliefs.json"))
    monkeypatch.delenv("CASCADE_BELIEFS", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_booth(monkeypatch):
    """A booth-day shell (CASCADE_BOOTH=1 exported) must not silently rerun the
    whole suite under booth tuning — a green run has to certify DEV
    behavior. Booth behavior is opted into per-test via monkeypatch.setenv
    (which overrides this scrub); subprocess tests inherit the scrubbed
    os.environ too."""
    monkeypatch.delenv("CASCADE_BOOTH", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_occupancy(monkeypatch):
    """occupancy is ON by default in configs/demo.yaml (2026-09-03), and this
    venv has the `grasping` extra, so every build_runtime in the suite would
    construct a real OccupancyClient and each WorldWatcher tick would then
    wait out a 500 ms ZMQ timeout against a bridge nobody started. That is
    dead time multiplied by every E2E test, and the cache staying empty means
    the runs certify nothing extra. Occupancy behaviour is covered directly
    by tests/test_occupancy.py (FakeClient + a live bridge subprocess), which
    opts back in via monkeypatch/en-bloc construction."""
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")


@pytest.fixture
def demo_cfg():
    return load_demo_config(camera="mock", arm="mock", llm="mock")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def loopback_host() -> str:
    """A loopback hostname that stdlib HTTP clients can actually reach here.

    Tests bind their servers to 0.0.0.0 and connect back over loopback. On a
    machine with certain VPN clients active (observed with a utun interface
    holding 198.18.0.0/24, the benchmark range some VPNs use for split
    tunnelling) a connection to the LITERAL 127.0.0.1 is intercepted and
    closed -- `http.client.RemoteDisconnected` -- while `localhost` resolves
    and connects normally.

    That is an environment fault, not a product bug: the server is listening
    correctly and a browser reaches it. But hard-coding 127.0.0.1 makes the
    suite red on a developer laptop for a reason that has nothing to do with
    the code, so the address is probed once and cached instead.
    """
    global _LOOPBACK_HOST
    if _LOOPBACK_HOST is not None:
        return _LOOPBACK_HOST

    import http.server
    import threading
    import urllib.request

    class _Ping(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib callback name
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", 0), _Ping)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for host in ("127.0.0.1", "localhost"):
            try:
                urllib.request.urlopen(f"http://{host}:{port}/", timeout=3).read()
                _LOOPBACK_HOST = host
                return host
            except Exception:
                continue
    finally:
        srv.shutdown()
    _LOOPBACK_HOST = "127.0.0.1"   # nothing worked; fail with the usual name
    return _LOOPBACK_HOST


_LOOPBACK_HOST: str | None = None


@pytest.fixture
def loopback():
    """The loopback host name to build test URLs from (see loopback_host)."""
    return loopback_host()
