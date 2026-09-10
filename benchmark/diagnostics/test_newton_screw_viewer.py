"""Recording/UI boundary tests, not proof of screw physics.

Run with the isolated Newton interpreter. The physics has its own real tests.
"""
from pathlib import Path
import importlib.util
import json
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def load_runner():
    path = REPO / "scripts/demo_newton_screw.py"
    assert path.is_file(), "Newton screw demo runner is missing"
    spec = importlib.util.spec_from_file_location("screw_viewer_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingBoundary:
    """A clock/state double; never presented as simulation evidence."""
    def __init__(self, *, verified=True, finite=True):
        self.sim_time = 0.0
        self.frame_dt = 0.1
        self.good = verified
        self.finite = finite
        self.model = SimpleNamespace(body_count=1, device="cpu")
        self.state = SimpleNamespace(body_q=SimpleNamespace(numpy=self.positions))

    def positions(self):
        return np.array([[self.sim_time if self.finite else float("nan"),
                          0, 0, 0, 0, 0, 1]], dtype=np.float32)

    def step(self):
        self.sim_time += self.frame_dt

    def metrics(self):
        return {"phase": "fixture", "screw_turns": self.sim_time,
                "axial_mm": self.sim_time, "applied_torque_nm": 0.1,
                "completed": self.sim_time >= 0.2,
                "verified": self.good and self.sim_time >= 0.2}


def test_recording_preserves_sampled_states_and_clock(tmp_path):
    runner = load_runner()
    result = runner.simulate(RecordingBoundary(), tmp_path, max_sim_seconds=1)
    assert result["summary"]["verified"] is True
    archive = np.load(tmp_path / "states.npz")
    np.testing.assert_allclose(archive["times"], [0, 0.1, 0.2])
    np.testing.assert_allclose(archive["body_q"][:, 0, 0], [0, 0.1, 0.2])
    assert json.loads((tmp_path / "summary.json").read_text())["verified"] is True


@pytest.mark.parametrize("case", ["unverified", "nan", "timeout"])
def test_failed_attempt_invalidates_previous_receipt(tmp_path, case):
    runner = load_runner()
    (tmp_path / "summary.json").write_text('{"verified":true}')
    demo = RecordingBoundary(verified=case != "unverified", finite=case != "nan")
    with pytest.raises(RuntimeError):
        runner.simulate(demo, tmp_path, max_sim_seconds=0.05 if case == "timeout" else 1)
    assert json.loads((tmp_path / "summary.json").read_text())["verified"] is False


def test_native_viewer_binds_loopback_and_serves_its_client():
    runner = load_runner()
    newton = importlib.import_module("newton")
    import warp as wp

    setattr(newton, "use_coord_layout_targets", True)
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat(0.0, 0.0, 0.0, 1.0)))
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    viewer = runner.make_viewer(port=0)
    server = viewer._server
    try:
        assert server.get_host() == "127.0.0.1"
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        url = f"http://127.0.0.1:{server.get_port()}"
        with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
            text = response.read().decode()
            assert response.status == 200
            assert "<html" in text.lower()
    finally:
        viewer.close()


def test_replay_builds_controls_and_writes_native_recording(tmp_path):
    """Native UI smoke using an explicitly synthetic recording boundary."""
    runner = load_runner()
    newton = importlib.import_module("newton")
    setattr(newton, "use_coord_layout_targets", True)
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    model = builder.finalize(device="cpu")
    recording = runner.simulate(RecordingBoundary(), tmp_path)
    demo = SimpleNamespace(model=model, frame_dt=0.1, camera_target=(0.1, 0, 0.1))
    runner.replay(demo, recording, tmp_path, port=0, open_browser=False, serve_seconds=0.1)
    assert (tmp_path / "episode.viser").stat().st_size > 0
    assert json.loads((tmp_path / "viewer.json").read_text())["url"].startswith("http://127.0.0.1:")


def test_one_click_bootstrap_uses_a_separate_environment(tmp_path):
    import os
    import subprocess
    import sys

    script = REPO / "scripts/show_newton_screw.sh"
    assert script.is_file(), "one-click Newton launcher is missing"
    binary = tmp_path / "uv"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    result = subprocess.run(["bash", str(script), "--no-open", "--port", "8768"],
                            env=dict(os.environ, PATH=str(tmp_path) + ":" + os.environ["PATH"]),
                            capture_output=True, text=True, check=True)
    command = json.loads(result.stdout)
    assert command[:3] == ["run", "--no-project", "--isolated"]
    assert "newton[sim]==1.5.1" in command
    assert "viser==1.1.0" in command
    assert command[-4:] == [str(REPO / "scripts/demo_newton_screw.py"), "--no-open", "--port", "8768"]
