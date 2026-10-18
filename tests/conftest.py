import ast
import copy
import math
import os
import runpy
import uuid
import select
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def load_isaac_bridge_definitions(names, env):
    """Load bridge handlers and their shutdown guard without booting Kit."""
    path = REPO / "scripts/isaac_bridge.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = set(names) | {"_bridge_should_stop", "_request_shutdown"}
    if "Handler" in wanted or "_refresh_frames" in wanted:
        wanted.add("_gripper_joint_snapshot")
    if "Handler" in wanted:
        wanted.add("_attachment_state_snapshot")
        # Boot state of every bridge: no seeded reset layout armed (ping reads it).
        env.setdefault("_RESET_SEED", None)
    if "_refresh_frames" in wanted or "_settle_props" in wanted:
        wanted.add("_camera_capture_due")
    if "_refresh_frames" in wanted:
        wanted.add("_render_token")
    if "_step_with_frame_history" in wanted:
        wanted.update({"_capture_frame_state", "_invalidate_frame_history", "_gripper_joint_snapshot"})
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    assert {node.name for node in nodes} == wanted
    # Unit tests have their own shutdown state and never consult a live STOP file.
    env.setdefault("os", SimpleNamespace(path=SimpleNamespace(lexists=lambda _path: False), environ={}))
    env.setdefault("_shutdown_requested", False)
    env.setdefault("_camera_video", None)
    env.setdefault("GRIP_IDX", [])
    env.setdefault("_motion_clock_epoch", "test-bridge-epoch")
    env.setdefault("_last_camera_capture_started", None)
    env.setdefault("_pending_camera_publications", set())
    spans = runpy.run_path(str(REPO / "scripts/isaac_python_spans.py"))["PythonSpans"]()
    env.setdefault("_python_spans", spans)
    env.setdefault("_profile_zone", spans.zone)
    env.setdefault("copy", copy)
    env.setdefault("math", math)
    env.setdefault("uuid", uuid)
    constant = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "NEWTON_CONTACT_CHANNEL" for t in n.targets))
    env.setdefault("NEWTON_CONTACT_CHANNEL", ast.literal_eval(constant))
    if "_refresh_frames" in wanted or "_step_with_frame_history" in wanted:
        helper = runpy.run_path(str(REPO / "scripts/isaac_frame_history.py"))
        for name in ("FrameHistory", "ClockDiscontinuity", "reference_key"):
            env.setdefault(name, helper[name])
        env.setdefault("_frame_history", helper["FrameHistory"]())
        env.setdefault("_published_frame_tokens", {})
        env.setdefault("_camera_frame_errors", {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)


def start_sleeping_process(*args, start_new_session=False):
    """Wait until Python has completed exec before recording its identity."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import os,time; os.write(1,b'R'); time.sleep(120)", *args],
        stdout=subprocess.PIPE, start_new_session=start_new_session,
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0], "fixture child did not start"
        assert child.stdout.read(1) == b"R", "fixture child exited before readiness"
        return child
    except BaseException:
        child.kill()
        child.wait(timeout=5)
        raise
    finally:
        child.stdout.close()


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
def _isolate_episodic_memory(monkeypatch, tmp_path):
    """Same isolation for the persisted VISUAL index (B43).

    `memory.persist_episodic` is off in the shipped profile, but a shell that
    exported CASCADE_EPISODIC=1 for a live session would otherwise make every
    build_runtime in the suite read and write the developer's
    `runs/episodic.json` -- remembered appearances leaking between tests and
    into the real store. Tests opt in per-test; the path is private.
    """
    monkeypatch.setenv("CASCADE_EPISODIC_PATH", str(tmp_path / "episodic.json"))
    monkeypatch.delenv("CASCADE_EPISODIC", raising=False)


@pytest.fixture(autouse=True)
def _isolate_learned_memories(monkeypatch, tmp_path):
    """Same isolation for the OTHER shared learned stores.

    ~/.cascade/grasp_memory.json (grasp priors/re-ranking) and
    ~/.cascade/envelope.json (operating envelope + failure model) change
    skill behaviour between otherwise-identical sessions. Observed leak:
    grasp-debugging runs against the real rig config recorded a string of
    air-grasps into the developer's ~/.cascade, whose learned prior then
    z-nudged the E2E suite's grasps into failing --
    test_grasp_and_place_happy_path went red with nothing in the diff.
    The reverse direction (suite writing into the developer's learned
    memory) corrupts real-rig behaviour silently.
    """
    monkeypatch.setenv("CASCADE_GRASP_MEMORY_PATH",
                       str(tmp_path / "grasp_memory.json"))
    monkeypatch.setenv("CASCADE_ENVELOPE_PATH",
                       str(tmp_path / "envelope.json"))


@pytest.fixture(autouse=True)
def _no_ambient_aspire_library(monkeypatch, tmp_path_factory):
    """No test runtime serves a machine's own ASPIRE notes (preflight R24).

    `build_runtime` reads the profile's note library once: the kitchen's
    `${CASCADE_ASPIRE_LIBRARY:-~/.cascade/aspire/sim/library}`, fr3 (and its
    mock / mujoco profiles) `${repo}/runs/fr3/skills_library`, every other
    profile `<repo>/skills_library` (gitignored, filled by learn_from_runs on
    dev and rig checkouts). All of them point at one directory that is never
    created, so every runtime here reads an empty library unless a test hands
    it one; and a shell's CASCADE_RUN_PURPOSE never relabels the suite's runs.
    """
    from cascade.config import PACKAGE_ROOT
    from cascade.skills import notes

    absent = tmp_path_factory.getbasetemp() / "no-aspire-library"
    monkeypatch.setenv("CASCADE_ASPIRE_LIBRARY", str(absent))
    monkeypatch.setattr(notes, "DEFAULT_LIBRARY", absent)
    monkeypatch.delenv("CASCADE_RUN_PURPOSE", raising=False)
    # libraries a shipped profile names inside the checkout (`${repo}/...`)
    shipped = {PACKAGE_ROOT / "runs" / "fr3" / "skills_library"}
    library_root = notes.library_root

    def pinned_root(cfg):
        root = library_root(cfg)
        return absent if root in shipped else root

    monkeypatch.setattr(notes, "library_root", pinned_root)


@pytest.fixture(autouse=True)
def _no_ambient_booth(monkeypatch):
    """A booth-day shell (CASCADE_BOOTH=1 exported) must not silently rerun the
    whole suite under booth tuning — a green run has to certify DEV
    behavior. Booth behavior is opted into per-test via monkeypatch.setenv
    (which overrides this scrub); subprocess tests inherit the scrubbed
    os.environ too."""
    monkeypatch.delenv("CASCADE_BOOTH", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_memory_embedder(monkeypatch):
    """A shell that exported CASCADE_MEMORY_EMBEDDER(=siglip) for a GPU session
    must not turn every build_runtime in the suite into a model load (or an
    EmbedderUnavailable on a box without the `memory-embed` extra). The suite
    certifies the shipped default (no embedder); tests opt in per-test."""
    monkeypatch.delenv("CASCADE_MEMORY_EMBEDDER", raising=False)
    monkeypatch.delenv("CASCADE_MEMORY_EMBEDDER_MODEL", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_programs(monkeypatch):
    """A shell that exported CASCADE_PROGRAMS=1 for a live session must not
    add list_programs / run_program to every MCP catalog in the suite (B42),
    nor point it at a real programs.jsonl. The suite certifies the shipped
    default (tier off); tests opt in per-test."""
    monkeypatch.delenv("CASCADE_PROGRAMS", raising=False)
    monkeypatch.delenv("CASCADE_PROGRAMS_PATH", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_vla_executor(monkeypatch):
    """A shell that exported CASCADE_GRASP_EXECUTOR=vla (and a policy port)
    for a GPU session must not route every grasp in the suite to a policy
    server (B49). The suite certifies the shipped analytic executor; tests
    opt in per test."""
    monkeypatch.delenv("CASCADE_GRASP_EXECUTOR", raising=False)
    monkeypatch.delenv("CASCADE_VLA_PORT", raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_judge(monkeypatch):
    """A shell that exported CASCADE_JUDGE(=vlm) for a live launch must not
    make every launcher test in the suite start a judge pass against a real
    model endpoint (B44), nor point judge_run at a private judge config. The
    suite certifies the shipped default (judge pass off); tests opt in."""
    for name in ("CASCADE_JUDGE", "CASCADE_JUDGE_TIMEOUT_S", "CASCADE_JUDGE_CONFIG"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_observed_finger_gate(monkeypatch):
    """A Spark shell that sourced the installer's env.sh exports
    CASCADE_OBSERVED_FINGER_GATE=1 (install_support.SPARK_RUNTIME_DEFAULTS).
    It must not switch every grasp in the suite to the observed-finger path:
    tests written for the shipped default (gate unset, as on CI) then fail on
    the Spark for an environment reason -- on the mock stack the gate refuses
    the grasp ("observed-finger gate currently requires calibrated Isaac
    reBot"). The suite certifies the gate-off default; gate-on tests set the
    gate themselves with monkeypatch.setenv (which overrides this scrub), so
    they run gate-on in every shell. pytest_configure below scrubs it for the
    whole session too (import-time code, module and session fixtures); this
    per-test scrub also undoes a test's direct os.environ write."""
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)


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


@pytest.fixture(autouse=True)
def _no_live_graspgenx(monkeypatch):
    """No test reaches a live GraspGen-X server. demo.yaml's grasp backend is
    `graspgenx` on 127.0.0.1:5556, and every mock-profile build_runtime probes
    it at startup: on a host that serves the model (the Spark, the rig) the
    suite's probes and plans landed on the real server, beside the robot's own
    requests. CASCADE_GRASPGENX_PORT beats every config layer, so it points at
    a port this process holds bound but not listening (nothing answers: the
    startup probe gives up after its short probe timeout, as on CI where
    nothing listens on 5556), and an ambient
    CASCADE_GRASPGENX_HOST is dropped so that port is local. A test that needs
    a server starts a fake one and sets its own port (monkeypatch.setenv /
    delenv override this); the port stays held until the test ends.
    pytest_configure below pins the same for the whole session, so module and
    session fixtures and import-time code never see 5556 either."""
    from owned_server import held_dead_port

    monkeypatch.delenv("CASCADE_GRASPGENX_HOST", raising=False)
    with held_dead_port() as port:
        monkeypatch.setenv("CASCADE_GRASPGENX_PORT", str(port))
        yield


_GRASPGENX_ENV = ("CASCADE_GRASPGENX_PORT", "CASCADE_GRASPGENX_HOST")
#: Spark defaults the suite certifies OFF for the whole session
#: (`_no_ambient_observed_finger_gate` per test).
_SCRUBBED_ENV = ("CASCADE_OBSERVED_FINGER_GATE",)


def pytest_configure(config):
    """Session-wide B6 pin, set before collection: test modules' import-time
    code and module/session-scoped fixtures run before the per-test
    `_no_live_graspgenx`, so they get a held dead port of their own (held
    until the session ends) and no ambient host. The same holds for an
    ambient observed-finger gate (_SCRUBBED_ENV): scrubbed for the session.
    The ambient values come back in pytest_unconfigure."""
    from owned_server import held_dead_port

    hold = held_dead_port()
    port = hold.__enter__()
    # The held dead port and the ambient value of every variable pinned here
    # (GraspGen-X's and the scrubbed Spark defaults), for pytest_unconfigure.
    config._cascade_session_env_pin = (hold, {name: os.environ.get(name)
                                              for name in _GRASPGENX_ENV + _SCRUBBED_ENV})
    os.environ.pop("CASCADE_GRASPGENX_HOST", None)
    for name in _SCRUBBED_ENV:
        os.environ.pop(name, None)
    os.environ["CASCADE_GRASPGENX_PORT"] = str(port)


def pytest_unconfigure(config):
    pin = getattr(config, "_cascade_session_env_pin", None)
    if pin is None:
        return
    hold, ambient = pin
    for name, value in ambient.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    hold.__exit__(None, None, None)


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
