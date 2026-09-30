"""GraspGen-X backend: wire protocol, pose conversion, and the fallback rule.

`configs/demo.yaml` defaults `grasp.backend: graspgenx`, yet this backend had
NO tests: the real server needs an NVIDIA GPU, a separate venv and downloaded
checkpoints, so on every machine without them the client raised and the runtime
silently fell back to the OBB planner. The whole path -- wire format, frame
conversion, tip offset, approach filtering, width estimation -- was therefore
only ever exercised on the DGX, where a mistake would surface at a demo.

These tests run it against `scripts/serve_graspgenx_stub.py`, which speaks the
same protocol analytically, so the client path is covered on any machine.

The stub is NOT a model: it does not test grasp QUALITY, only that cascade
speaks the protocol correctly and converts frames right. That distinction is
the point -- these tests would still catch an inverted axis or a dropped tip
offset, which is what actually breaks a demo.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest
from conftest import REPO

from cascade.config import Cfg
from cascade.types import Detection, ObjectFix


def _has_wire() -> bool:
    try:
        import msgpack  # noqa: F401
        import msgpack_numpy  # noqa: F401
        import zmq  # noqa: F401

        return True
    except ImportError:
        return False


needs_wire = pytest.mark.skipif(
    not _has_wire(),
    reason="needs the grasping extra: uv pip install -e '.[grasping]'",
)

STUB = REPO / "scripts" / "serve_graspgenx_stub.py"
PORT = 5599          # not 5556: never collide with a real server on the rig


@pytest.fixture(scope="module")
def stub_server():
    """The protocol stub, on a private port."""
    proc = subprocess.Popen(
        [sys.executable, str(STUB), "--port", str(PORT), "--quiet"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait for the bind rather than sleeping a fixed amount.
    deadline = time.monotonic() + 15.0
    import socket

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode()[:400] if proc.stderr else ""
            pytest.skip(f"stub server exited: {err}")
        s = socket.socket()
        s.settimeout(0.2)
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            s.close()
            break
        s.close()
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.skip("stub server did not bind in time")
    yield PORT
    proc.kill()
    proc.wait(timeout=5)


def _cube_fix(centre=(0.22, 0.0, 0.045), half=0.025, n=400) -> ObjectFix:
    rng = np.random.default_rng(0)
    c = np.asarray(centre, dtype=float)
    pts = c + rng.uniform(-half, half, size=(n, 3))
    return ObjectFix(
        label="red cube", position=c, points=pts,
        detection=Detection(label="red cube", conf=0.9, bbox=np.zeros(4)),
        extent=np.array([half * 2] * 3), axes=np.eye(3),
    )


def _planner(port, **over):
    from cascade.grasping.graspgenx_backend import GraspGenXPlanner

    cfg = {"host": "127.0.0.1", "port": port, "timeout_ms": 8000,
           "num_grasps": 16, "tip_offset_m": 0.098}
    cfg.update(over)
    return GraspGenXPlanner(Cfg({"graspgenx": cfg}))


# ── the failure that started this: the client could not even be built ────


def test_the_client_builds_when_the_extra_is_installed():
    """`grasp.backend: graspgenx` is the DEFAULT, so a missing wire dependency
    silently disabled the configured planner on every grasp. The extra now
    declares pyzmq+msgpack-numpy; this test fails loudly if that regresses."""
    from cascade.grasping.graspgenx_backend import GraspGenXClient, GraspGenXError

    if not _has_wire():
        pytest.skip("grasping extra not installed in this venv")
    try:
        GraspGenXClient(port=PORT)
    except GraspGenXError as e:  # pragma: no cover
        pytest.fail(f"client could not be constructed: {e}")


def test_the_extra_is_declared_in_pyproject():
    """The deps must be installable through a supported path, not by hand."""
    text = (REPO / "pyproject.toml").read_text()
    assert "grasping = [" in text, "no `grasping` extra in pyproject.toml"
    block = text.split("grasping = [", 1)[1].split("]", 1)[0]
    assert "pyzmq" in block and "msgpack" in block


# ── protocol + frame conversion ──────────────────────────────────────────


@needs_wire
def test_grasps_come_back_and_are_top_down(stub_server):
    gs = _planner(stub_server).plan(_cube_fix(), max_width_m=0.055)
    assert gs, "no grasps returned"
    for g in gs:
        assert g.approach[2] < 0, f"approach points up: {g.approach}"


@needs_wire
def test_the_grasp_lands_on_the_object(stub_server):
    """Catches a dropped/incorrect tip_offset: the server returns a
    GRIPPER-BASE pose ~10 cm behind the jaw centre, and the client must add it
    back along the approach. Forgetting it puts every grasp 10 cm above the
    object -- jaws close on air, which is exactly the failure this backend is
    meant to avoid."""
    fix = _cube_fix()
    g = _planner(stub_server).plan(fix, max_width_m=0.055)[0]
    lateral = float(np.linalg.norm(g.position[:2] - fix.position[:2]))
    assert lateral < 0.01, f"grasp is {lateral*100:.1f} cm off laterally"
    # within the object's vertical span, not hovering above it
    assert abs(g.position[2] - fix.position[2]) < 0.03, (
        f"grasp z={g.position[2]:.3f} vs object z={fix.position[2]:.3f}: "
        f"tip_offset probably not applied"
    )


@needs_wire
def test_tip_offset_moves_the_grasp_along_the_approach(stub_server):
    """Pin the offset's DIRECTION. A sign error still lands 'near' the object
    but on the wrong side, and only shows up as intermittent air grasps."""
    fix = _cube_fix()
    near = _planner(stub_server, tip_offset_m=0.0).plan(fix, max_width_m=0.055)[0]
    far = _planner(stub_server, tip_offset_m=0.098).plan(fix, max_width_m=0.055)[0]
    # approach is -z, so adding the offset moves the TCP DOWN from the base
    assert far.position[2] < near.position[2], (
        "tip offset moved the grasp the wrong way along the approach"
    )
    assert abs((near.position[2] - far.position[2]) - 0.098) < 1e-3


@needs_wire
def test_the_rotation_is_orthonormal_and_matches_the_approach(stub_server):
    """cascade's Grasp convention is columns [approach, jaw-open, third].
    A permutation error here produces grasps that look plausible and rotate the
    wrist into a limit."""
    g = _planner(stub_server).plan(_cube_fix(), max_width_m=0.055)[0]
    R = np.asarray(g.rotation, dtype=float)
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(R[:, 0], g.approach, atol=1e-6)


@needs_wire
def test_width_is_clamped_to_the_jaw(stub_server):
    """A grasp wider than the jaw can never close. The SO-101's 55 mm jaw is
    the case that matters, since demo.yaml's default sweep describes a 90 mm
    reBot gripper."""
    for jaw in (0.055, 0.09):
        for g in _planner(stub_server).plan(_cube_fix(), max_width_m=jaw):
            assert g.width_m <= jaw + 1e-9, f"{g.width_m} > jaw {jaw}"


@needs_wire
def test_grasps_are_ranked_best_first(stub_server):
    gs = _planner(stub_server).plan(_cube_fix(), max_width_m=0.055)
    q = [g.quality for g in gs]
    assert q == sorted(q, reverse=True), f"not sorted by quality: {q}"


@needs_wire
def test_a_too_small_cloud_is_rejected_not_guessed(stub_server):
    from cascade.grasping.graspgenx_backend import GraspGenXError

    fix = _cube_fix(n=10)
    with pytest.raises(GraspGenXError, match="points"):
        _planner(stub_server).plan(fix, max_width_m=0.055)


# ── the booth rule: a dead server must never break a grasp ───────────────


@needs_wire
def test_a_symmetric_object_does_not_crash_the_planner(stub_server):
    """Regression: a cube's two horizontal extents are EQUAL, and sorting
    (extent, axis_vector) tuples then falls through to comparing numpy arrays
    -> "truth value of an array is ambiguous", returned to the client as an
    opaque server error and silently degraded to OBB.

    The random cloud the other tests use is asymmetric, so it never triggered
    this; the real perception cloud of a cube triggers it every single time.
    Symmetric objects are the common case in a demo, so this is the cloud the
    stub must survive."""
    # The REAL perception cloud of the mock camera's cube, captured from a
    # running demo (tests/assets/mock_cube_cloud.npy). Using the recorded
    # cloud rather than a synthesised one is deliberate: the tie is between
    # two floats equal to the last bit, and a hand-built cloud reproduces it
    # only by accident. This file IS the reproduction.
    pts = np.load(REPO / "tests" / "assets" / "mock_cube_cloud.npy")
    c = pts.mean(axis=0)
    fix = ObjectFix(
        label="red cube", position=c, points=pts,
        detection=Detection(label="red cube", conf=0.9, bbox=np.zeros(4)),
        extent=np.array([0.034, 0.034, 0.0]), axes=np.eye(3),
    )
    gs = _planner(stub_server, num_grasps=100).plan(fix, max_width_m=0.055)
    assert gs, "symmetric object returned no grasps"


@needs_wire
def test_the_grasp_is_above_the_table_not_below_it(stub_server):
    """The client applies `tip_offset_m` to BOTH actions, because the real
    server returns a gripper-base pose either way ("even with centered sweep
    boxes", configs/demo.yaml). A stub that returned a jaw-centre pose for
    `infer_object` would make that correct offset push every grasp 9.8 cm
    below the table -- grasps that look plausible in a list and are
    unreachable in the cell.

    Guards the sign/convention agreement between the two ends of the wire."""
    pts = np.load(REPO / "tests" / "assets" / "mock_cube_cloud.npy")
    c = pts.mean(axis=0)
    fix = ObjectFix(
        label="red cube", position=c, points=pts,
        detection=Detection(label="red cube", conf=0.9, bbox=np.zeros(4)),
        extent=np.array([0.034, 0.034, 0.0]), axes=np.eye(3),
    )
    for g in _planner(stub_server, num_grasps=100).plan(fix, max_width_m=0.055):
        assert g.position[2] > 0.0, (
            f"grasp at z={g.position[2]:.3f} is below the table: the tip "
            f"offset convention disagrees between client and server"
        )
    top = _planner(stub_server, num_grasps=100).plan(fix, max_width_m=0.055)[0]
    assert abs(top.position[2] - c[2]) < 0.03, (
        f"top grasp z={top.position[2]:.3f} is not near the object z={c[2]:.3f}"
    )


@needs_wire
def test_a_dead_server_raises_a_typed_error_promptly():
    """The runtime catches this and falls back to OBB. It must be a
    GraspGenXError (not a hang, not a raw zmq error) or the fallback in
    runtime.py cannot do its job."""
    from cascade.grasping.graspgenx_backend import GraspGenXError

    planner = _planner(5601, timeout_ms=600)   # nothing listening there
    t0 = time.monotonic()
    with pytest.raises(GraspGenXError):
        planner.plan(_cube_fix(), max_width_m=0.055)
    assert time.monotonic() - t0 < 10.0, "dead server took too long to fail"


def test_the_runtime_falls_back_to_obb_when_graspgenx_is_down(monkeypatch):
    """End of the booth rule, at the level that matters: a grasp still gets
    planned. Verified through the real _plan_grasps path with the backend
    pointed at a dead port."""
    pytest.importorskip("pinocchio")
    from pathlib import Path

    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    cfg = load_demo_config(camera="mock_small", arm="so101_mock", llm="mock")
    # point the backend at a port with nothing on it
    cfg._data["grasp"]["backend"] = "graspgenx"
    cfg._data["grasp"]["graspgenx"]["port"] = 5602
    cfg._data["grasp"]["graspgenx"]["timeout_ms"] = 400
    runtime, arm = build_runtime(cfg, Path("/tmp/wrc_ggx_fallback"))
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and runtime.beliefs.find("red cube") is None:
            time.sleep(0.05)
        _, fix = runtime._localize("red cube")
        grasps = runtime._plan_grasps(fix, label="red cube")
        assert grasps, "no grasps at all: the OBB fallback did not run"
    finally:
        shutdown_runtime(runtime, arm)


@needs_wire
def test_required_model_rejects_the_analytic_stub(stub_server):
    from cascade.grasping.graspgenx_backend import GraspGenXError
    with pytest.raises(GraspGenXError, match="stub rejected"):
        _planner(stub_server, required=True).probe()


@needs_wire
@pytest.mark.parametrize("tags", [[], ["obb"], ["diff", "diff"]])
def test_required_diffusion_rejects_missing_or_analytic_provenance(tags):
    from cascade.grasping.graspgenx_backend import GraspGenXError
    planner = _planner(5602, required=True, planner="diffusion", sweep={
        "extents_open": [.09, .02, .045], "extents_mid": [.045, .02, .045]})
    planner._client.request = lambda request: {
        "grasps": np.eye(4)[None], "confidences": np.array([.9]), "branch_tags": tags}
    with pytest.raises(GraspGenXError, match="provenance"):
        planner.plan(_cube_fix())


@needs_wire
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensor_cloud_crosses_wire_and_width_stays_on_its_device(stub_server, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required for the original Spark failure")
    fix = _cube_fix()
    expected = _planner(stub_server).plan(fix)
    fix.points = torch.as_tensor(fix.points, device=device)
    actual = _planner(stub_server).plan(fix)
    assert len(actual) == len(expected)
    np.testing.assert_allclose([g.width_m for g in actual], [g.width_m for g in expected], atol=1e-6)
    np.testing.assert_allclose([g.position for g in actual], [g.position for g in expected], atol=1e-6)
    assert fix.points.device.type == device


@pytest.mark.parametrize("required", [False, True])
def test_runtime_recovers_after_one_failed_inference_without_sticky_obb(required):
    from types import SimpleNamespace
    from cascade.config import load_demo_config
    from cascade.skills.runtime import SkillRuntime
    from cascade.types import Grasp, SkillError

    cfg = load_demo_config()
    cfg._data["grasp"]["graspgenx"]["required"] = required
    learned = Grasp(np.array([.2, 0, .03]), np.eye(3), .05, np.array([0, 0, -1]))

    class FlakyModel:
        status = {"status": "ok"}
        last_latency_s = .5
        calls = 0
        def probe(self):
            self.status = {"status": "ok"}
        def plan(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary timeout")
            return [learned]
        def describe(self):
            return "graspgenx (learned 6-DoF)"

    runtime = SimpleNamespace(cfg=cfg, _max_width=.09, _tool_axis_order="down_open",
                              _graspgenx=FlakyModel(), _graspgenx_down=False,
                              memory=SimpleNamespace(add=lambda *args: None),
                              grasp_memory=SimpleNamespace(prior=lambda *args: None))
    if required:
        with pytest.raises(SkillError, match="required but unavailable"):
            SkillRuntime._plan_grasps(runtime, _cube_fix())
    else:
        assert SkillRuntime._plan_grasps(runtime, _cube_fix())
        assert runtime.grasp_planner_used == "obb (graspgenx down)"
    runtime._graspgenx_retry_after = 0
    result = SkillRuntime._plan_grasps(runtime, _cube_fix())
    assert result[0] is learned
    if required:
        assert len(result) == 1  # no analytic candidates mixed into required mode
    assert runtime._graspgenx.calls == 2
    assert not runtime._graspgenx_down
    assert runtime.grasp_planner_used == "graspgenx (learned 6-DoF)"


@pytest.mark.parametrize("backend", ["obb", "graspgenx"])
def test_explicit_launcher_backend_reaches_every_arm(monkeypatch, backend):
    from cascade.config import load_demo_config
    monkeypatch.setenv("CASCADE_GRASP_BACKEND", backend)
    cfg = load_demo_config(arms=["isaac_kitchen_gpu", "mock"])
    assert cfg.grasp.backend == backend
    assert all(arm["resolved"]["grasp"]["backend"] == backend for arm in cfg.arms)


@needs_wire
@pytest.mark.parametrize("tip", [0.0, 0.05, 0.098])
def test_sweep_conditioning_and_returned_pose_use_the_same_gripper_frame(tip):
    planner = _planner(5602, tip_offset_m=tip, sweep={
        "extents_open": [.09, .02, .045], "extents_mid": [.045, .02, .045],
        "offset_open": [0, 0, 0], "offset_mid": [0, 0, 0],
        "fingertip_depth": .0225})
    fix = _cube_fix()

    def model(request):
        sweep = request["sweep_volume_params"]
        np.testing.assert_allclose(sweep["offset_mid"], sweep["offset_open"])
        assert sweep["fingertip_depth"] == pytest.approx(tip + .0225)
        pose = np.eye(4)
        pose[:3, :3] = np.diag([1, -1, -1])
        # The model positions the conditioned open volume at the object.
        pose[:3, 3] = fix.position - pose[:3, :3] @ np.array(sweep["offset_open"])
        return {"grasps": pose[None], "confidences": np.array([.9]), "branch_tags": ["diff"]}

    planner._client.request = model
    grasp = planner.plan(fix)[0]
    np.testing.assert_allclose(grasp.position, fix.position, atol=1e-7)
