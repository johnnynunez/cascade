"""scripts/franka_rig.py: the read-only Franka FR3 checks scripts/launch.sh runs.

`bringup` needs rclpy (a sourced ROS 2 install), which CI and this workstation
lack: its target derivation is pinned against configs/arms/fr3.yaml and its
failure is a clear non-zero exit. `ready` is exercised against the REAL robot
server (cascade.apps.mcp_server --http, mock FR3 + mock camera, latched STOP):
the operator contract launch.sh waits for is the server's own live view.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import has_pinocchio

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("franka_rig", REPO / "scripts" / "franka_rig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rig = _load()


def test_bringup_targets_are_the_fr3_profiles_native_endpoints():
    """What the native franka_ros2 bringup must offer the FR3 driver
    (configs/arms/fr3.yaml): both joint-state topics, the arm controller's
    trajectory action and the three Franka Gripper actions, and the arm
    controller itself (ACTIVE)."""
    targets = rig.bringup_targets(rig.load_arm("fr3"))
    assert targets == {
        "topics": ["/franka/joint_states", "/franka_gripper/joint_states"],
        "actions": ["/fr3_arm_controller/follow_joint_trajectory", "/franka_gripper/gripper_action",
                    "/franka_gripper/move", "/franka_gripper/grasp"],
        "controller": "fr3_arm_controller",
    }


def test_bringup_targets_follow_a_namespace_and_absolute_names():
    arm = {"ros_namespace": "/left/", "ros_state_topic": "joint_states", "ros_trajectory_action": "/abs/ctrl/fjt",
           "ros_gripper_action": "gripper/act"}
    assert rig.bringup_targets(arm) == {"topics": ["/left/joint_states"],
                                        "actions": ["/abs/ctrl/fjt", "/left/gripper/act"], "controller": "abs"}


def test_bringup_without_ros_fails_clearly(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "rclpy", None)   # import rclpy -> ImportError
    assert rig.main(["bringup", "--timeout", "0.1"]) == 3
    assert "[franka] bringup:" in capsys.readouterr().err


VIEW = {"cameras": {"d435i": {"fps": 15, "error": None}}, "stopped": True, "resume_supported": True}


@pytest.mark.parametrize("change, problem", [
    ({"stopped": False}, "software STOP is not latched"),
    ({"resume_supported": False}, "Resume & Home is not available"),
    ({"cameras": {}}, "no camera announced"),
    ({"cameras": {"d435i": {"fps": 15, "error": "disconnected"}}}, "camera d435i: disconnected"),
    ({"cameras": {"d435i": {"fps": 0, "error": None}}}, "camera d435i has no frames yet"),
])
def test_the_operator_contract(change, problem):
    assert rig.runtime_ready(VIEW, "d435i") == []
    assert problem in rig.runtime_ready({**VIEW, **change}, "d435i")
    assert rig.runtime_ready(VIEW, "l515") == ["camera l515 not announced"]


def test_wait_ready_polls_until_the_contract_holds():
    views = iter([OSError("refused"), {**VIEW, "stopped": False},
                  {**VIEW, "cameras": {"d435i": {"fps": 15, "error": "disconnected"}}}, VIEW])

    def get(_url):
        v = next(views)
        if isinstance(v, Exception):
            raise v
        return v
    out = rig.wait_ready(8090, 30, "d435i", get=get, sleep=lambda _s: None)
    assert out == {"cameras": ["d435i"], "stopped": True, "resume_supported": True,
                   "operator_page": "http://127.0.0.1:8090/"}


def test_wait_ready_times_out_naming_what_is_missing():
    with pytest.raises(RuntimeError, match="software STOP is not latched"):
        rig.wait_ready(8090, 0, get=lambda _u: {**VIEW, "stopped": False}, sleep=lambda _s: None)


def test_cpu_lists():
    assert rig.cpu_ranges({0, 1, 2, 3, 8}) == "0-3,8"
    assert rig.cpu_ranges({5}) == "5"
    assert rig.narrowed_affinity({0, 1, 2, 3, 4, 5, 6, 7}, range(14)) == "0-7"
    assert rig.narrowed_affinity(range(14), range(14)) == "", "an unrestricted launcher pins nothing"


def test_cpus_follows_taskset():
    r = subprocess.run(["taskset", "-c", "0", sys.executable, str(REPO / "scripts/franka_rig.py"), "cpus"],
                       capture_output=True, text=True, timeout=30, check=False)
    if r.returncode != 0 or (os.cpu_count() or 1) < 2:
        pytest.skip("taskset unavailable or a single-CPU host")
    assert r.stdout.strip() == "0"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not has_pinocchio(), reason="pinocchio not available")
def test_the_real_robot_server_meets_the_contract_with_a_latched_stop(tmp_path):
    """The robot server exactly as launch.sh registers it for the FR3
    (CASCADE_START_STOPPED=1, CASCADE_STREAM=eager), on the mock FR3 and the
    mock camera: its live view reports the camera streaming, the STOP latched
    and Resume & Home available, which is what `franka_rig.py ready` waits
    for before the agent's sandbox comes up."""
    stream_port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASCADE_")}
    env.update(PYTHONPATH=os.pathsep.join(p for p in (str(REPO / "src"), env.get("PYTHONPATH", "")) if p),
               CASCADE_ARM="fr3_mock", CASCADE_CAMERAS="mock", CASCADE_START_STOPPED="1", CASCADE_STREAM="eager",
               CASCADE_STREAM_PORT=str(stream_port), CASCADE_OCCUPANCY="0", CASCADE_VIEW="0",
               CASCADE_RUN_DIR=str(tmp_path / "run"), CASCADE_BELIEFS_PATH=str(tmp_path / "beliefs.json"),
               CASCADE_GRASP_MEMORY_PATH=str(tmp_path / "grasp.json"),
               CASCADE_ENVELOPE_PATH=str(tmp_path / "envelope.json"),
               CASCADE_EPISODIC_PATH=str(tmp_path / "episodic.json"), YOLO_OFFLINE="True")
    token = tmp_path / "mcp.token"
    token.write_text("t" * 32 + "\n")
    token.chmod(0o600)
    port_file = tmp_path / "mcp.port"
    proc = subprocess.Popen([sys.executable, "-m", "cascade.apps.mcp_server", "--http", "127.0.0.1:0",
                             "--http-port-file", str(port_file), "--token-file", str(token)],
                            cwd=tmp_path, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=open(tmp_path / "server.log", "w"))
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited {proc.returncode}: {(tmp_path / 'server.log').read_text()[-2000:]}")
            try:
                out = rig.wait_ready(stream_port, 2, "mock")
                break
            except RuntimeError:
                continue
        else:
            pytest.fail("the live view never met the contract: " + (tmp_path / "server.log").read_text()[-2000:])
        assert out["stopped"] is True and out["resume_supported"] is True and out["cameras"] == ["mock"]
        r = subprocess.run([sys.executable, str(REPO / "scripts/franka_rig.py"), "ready", "--port", str(stream_port),
                            "--camera", "mock", "--timeout", "10"], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["operator_page"] == f"http://127.0.0.1:{stream_port}/"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_owns_names_only_this_launch_owners_live_process(tmp_path):
    """`process_owner.py owns --role R` (launch.sh: reuse only an occupancy
    bridge this launcher started) exits 0 for a live recorded process of
    that role and 1 otherwise."""
    from cascade.apps import process_owner as owners

    script = REPO / "src/cascade/apps/process_owner.py"

    def owns(role):
        return subprocess.run([sys.executable, str(script), "--repo", str(REPO), "--state-root", str(tmp_path),
                               "owns", "--role", role], capture_output=True, text=True, timeout=30).returncode
    assert owns("occupancy_bridge") == 1, "no owner yet"
    state = owners.profile_state_dir(tmp_path, "")
    owner = owners.load_owner(state, REPO, "", create=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "serve_occupancy_bridge.py"])
    try:
        assert owns("occupancy_bridge") == 1
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                owners.register_process(state, owner, child.pid, "occupancy_bridge")
                break
            except (ValueError, OSError):
                time.sleep(0.05)
        assert owns("occupancy_bridge") == 0
        assert owns("graspgenx") == 1
    finally:
        child.terminate()
        child.wait(timeout=10)
    assert owns("occupancy_bridge") == 1, "a dead record is not a live process"
