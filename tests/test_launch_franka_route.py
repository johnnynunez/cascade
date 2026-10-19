"""scripts/launch.sh --sim none --arm fr3 --cameras d435i: the Franka FR3 on the unified route.

The real launcher runs end to end (the OpenShell route of
tests/test_launch_openshell_route.py, whose fixture doubles dependencies,
Docker and scripts/openshell_rig.py). The Franka host boundaries are doubled
here too: the ROS setup scripts are inert files, and scripts/franka_rig.py
(the read-only bringup and readiness checks; its own contract is
tests/test_franka_rig.py) records every call and the ROS marker it ran under.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_delivery import model_http_boundary as _model_http_boundary
from test_launch_openshell_route import linux_launcher as _linux_launcher

launcher_boundary = _launcher_boundary
model_http_boundary = _model_http_boundary
linux_launcher = _linux_launcher

REPO = Path(__file__).resolve().parents[1]

FAKE_FRANKA_RIG = r'''
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FRANKA_RIG_LOG"], "a") as f:
    f.write(json.dumps({"args": args, "ros": os.environ.get("CASCADE_TEST_ROS_SOURCED"),
                        "domain": os.environ.get("ROS_DOMAIN_ID")}) + "\n")
if args[0] == "cpus":
    print(os.environ.get("FAKE_CPUS", ""))
elif args[0] == "bringup":
    if os.environ.get("FAKE_BRINGUP_DOWN"):
        print("[franka] bringup: no JointState on /franka/joint_states", file=sys.stderr)
        raise SystemExit(3)
    print(json.dumps({"controller": "fr3_arm_controller", "controller_state": "active"}))
elif args[0] == "ready":
    if os.environ.get("FAKE_NOT_READY"):
        print("[franka] ready: software STOP is not latched", file=sys.stderr)
        raise SystemExit(3)
    print(json.dumps({"cameras": ["d435i"], "stopped": True, "resume_supported": True}))
'''


@contextmanager
def _brain(model: str):
    """An OpenAI-compatible brain serving `model` with native tool calls."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": model}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
                "name": "cascade__world_state", "arguments": "{}"}}]}}]}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _listening():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def franka(linux_launcher, tmp_path):
    """linux_launcher on the FR3 path: a Bolt brain (a Cosmos still answers
    on the fixture's endpoint), inert ROS setup files, a fake franka_rig.py."""
    h = linux_launcher
    repo, env = h["repo"], dict(h["env"])
    (repo / "scripts/franka_rig.py").write_text(FAKE_FRANKA_RIG)
    ros, ws = tmp_path / "ros setup.bash", tmp_path / "franka setup.bash"
    ros.write_text("export CASCADE_TEST_ROS_SOURCED=1\nunset NOT_SET_ANYWHERE\n[ -z \"$UNBOUND_IN_ROS_SETUP\" ]\n"
                   "true\n")
    ws.write_text("export AMENT_PREFIX_PATH=/fake/franka_ros2_ws/install\n")
    with _brain("nemotron-3.5-bolt") as bolt:
        env.update(CASCADE_NEMOTRON_BOLT_BASE_URL=bolt, CASCADE_ROS_SETUP=str(ros), CASCADE_FRANKA_SETUP=str(ws),
                   FRANKA_RIG_LOG=str(tmp_path / "franka-rig.jsonl"), CASCADE_STREAM_PORT=str(_free_port()))
        for key in ("ROS_DOMAIN_ID", "CASCADE_MCP_CPUS", "RMW_IMPLEMENTATION", "CASCADE_GRASP_BACKEND",
                    "CASCADE_START_STOPPED", "CASCADE_STREAM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.pop(key, None)
        c = list(h["command"])
        i = c.index("--arm")
        command = c[:i] + ["--arm", "fr3", "--cameras", "d435i"] + c[i + 4:]
        yield {**h, "env": env, "command": command, "bolt": bolt}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _lines(path) -> list:
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _run(h, *extra, env=None, command=None):
    return subprocess.run([*(command or h["command"]), *extra], env=env or h["env"], capture_output=True, text=True,
                          timeout=120, check=False)


def _rig(h):
    return _lines(h["env"]["RIG_LOG"])


def _franka_rig(h):
    return _lines(h["env"]["FRANKA_RIG_LOG"])


def _seen_env(h) -> dict:
    return json.loads((Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json").read_text())


def test_fr3_runs_on_the_unified_route_with_the_franka_contract(franka):
    """`--sim none --arm fr3 --cameras d435i`: the default OpenShell instance,
    the bringup checked before anything starts, the robot server started in
    the ROS environment with a latched STOP, the operator contract checked
    on its live view, then the sandbox observe-only with the franka agent
    profile and the Bolt brain, never the Cosmos still answering."""
    h = franka
    p = _run(h)
    assert p.returncode == 0, p.stdout + p.stderr
    rig = _rig(h)
    assert all(c[:2] == ["--instance", ""] for c in rig), "the real robot's route is the default instance"
    assert [c[2] for c in rig if c[2] != "exec"] == ["instance", "install", "gateway", "mcp-start", "up", "instance"]
    (start,) = [c[2:] for c in rig if c[2] == "mcp-start"]
    assert "--ros-env" in start and "--replace" not in start and "--cpus" not in start
    (up,) = [c[2:] for c in rig if c[2] == "up"]
    assert up[up.index("--agent-profile") + 1] == "franka"
    assert up[up.index("--model") + 1] == "nemotron-3.5-bolt", "Cosmos is not the Franka brain"
    assert up[up.index("--brain-port") + 1] == h["bolt"].split(":")[-1].split("/")[0]
    assert "--observe-only" in up and "--allow-motion" not in up
    calls = _franka_rig(h)
    assert [c["args"][0] for c in calls] == ["cpus", "bringup", "ready"]
    bringup = next(c for c in calls if c["args"][0] == "bringup")
    assert bringup["args"][:3] == ["bringup", "--arm", "fr3"]
    assert bringup["ros"] == "1" and bringup["domain"] == "31", "the check runs in the sourced ROS environment"
    ready = next(c for c in calls if c["args"][0] == "ready")
    assert ready["args"][ready["args"].index("--camera") + 1] == "d435i"
    assert ready["args"][ready["args"].index("--port") + 1] == h["env"]["CASCADE_STREAM_PORT"]
    assert "operator:  STOP / Resume & Home http://127.0.0.1:" in p.stdout
    assert "[launch] plan: sim=none arm=fr3 cameras=d435i brain=nemotron-bolt" in p.stdout
    assert "scripts/openshell_rig.py motion unlock" in p.stdout


def test_the_robot_server_gets_the_franka_environment(franka):
    h = franka
    env = {**h["env"], "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp", "CASCADE_START_HOLDING": "cork",
           "CASCADE_GROUNDING_LLM_PROFILE": "local_cosmos"}
    p = _run(h, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    seen = _seen_env(h)
    assert (seen["CASCADE_ARM"], seen["CASCADE_CAMERAS"]) == ("fr3", "d435i")
    assert seen["CASCADE_START_STOPPED"] == "1" and seen["CASCADE_STREAM"] == "eager"
    assert seen["ROS_DOMAIN_ID"] == "31" and seen["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert seen["CASCADE_ROS_SETUP"] == h["env"]["CASCADE_ROS_SETUP"]
    assert seen["CASCADE_FRANKA_SETUP"] == h["env"]["CASCADE_FRANKA_SETUP"]
    assert seen["OMP_NUM_THREADS"] == "4" and seen["OPENBLAS_NUM_THREADS"] == "4"
    assert seen["CASCADE_PERCEPTION_MODEL"] == "nemotron-3.5-bolt", "the brain identifies objects"
    assert seen["CASCADE_PERCEPTION_URL"] == h["bolt"]
    assert "CASCADE_GROUNDING_LLM_PROFILE" not in seen, "no Cosmos grounding on the unified route"
    assert "CASCADE_START_HOLDING" not in seen
    assert seen["CASCADE_OCCUPANCY"] == "0", "--occupancy none still reaches the server"


def test_an_explicit_ros_domain_wins(franka):
    h = franka
    p = _run(h, env={**h["env"], "ROS_DOMAIN_ID": "47"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert _seen_env(h)["ROS_DOMAIN_ID"] == "47"
    assert next(c for c in _franka_rig(h) if c["args"][0] == "bringup")["domain"] == "47"


@pytest.mark.parametrize("source, expected", [("taskset", "0-7"), ("explicit", "2-5"), ("none", None)])
def test_the_robot_server_unit_gets_the_launchers_cpus(franka, source, expected):
    """`taskset -c 0-7 ./scripts/launch.sh ...` pins the sidecars it starts;
    the robot server is a user unit, so its CPUs are passed explicitly.
    CASCADE_MCP_CPUS wins."""
    h = franka
    env = dict(h["env"])
    if source == "taskset":
        env["FAKE_CPUS"] = "0-7"
    elif source == "explicit":
        env.update(FAKE_CPUS="0-7", CASCADE_MCP_CPUS="2-5")
    p = _run(h, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    (start,) = [c[2:] for c in _rig(h) if c[2] == "mcp-start"]
    if expected is None:
        assert "--cpus" not in start
    else:
        assert start[start.index("--cpus") + 1] == expected
        assert f"robot server pinned to CPUs {expected}" in p.stdout


def test_a_bringup_that_does_not_answer_stops_the_launch_before_anything_starts(franka):
    h = franka
    p = _run(h, env={**h["env"], "FAKE_BRINGUP_DOWN": "1"})
    assert p.returncode != 0
    assert "the franka_ros2 bringup does not answer" in p.stderr and "bringup_franka.sh" in p.stderr
    assert [c for c in _rig(h) if c[2] != "instance"] == [], "no sandbox, no robot server"


def test_missing_ros_setup_is_refused(franka, tmp_path):
    h = franka
    p = _run(h, env={**h["env"], "CASCADE_FRANKA_SETUP": str(tmp_path / "absent.bash")})
    assert p.returncode != 0 and "ROS setup missing" in p.stderr
    assert _franka_rig(h)[-1]["args"][0] == "cpus" and [c for c in _rig(h) if c[2] != "instance"] == []


def test_a_server_that_misses_the_operator_contract_never_gets_an_agent(franka):
    h = franka
    p = _run(h, env={**h["env"], "FAKE_NOT_READY": "1"})
    assert p.returncode != 0
    assert "did not reach the Franka operator contract" in p.stderr
    steps = [c[2] for c in _rig(h)]
    assert "mcp-start" in steps and "up" not in steps


def test_a_busy_operator_page_port_stops_the_launch(franka):
    """The old interface (or a previous launch) still holds the live view:
    the launcher never stops it, and refuses before touching the camera."""
    h = franka
    with _listening() as port:
        p = _run(h, env={**h["env"], "CASCADE_STREAM_PORT": str(port)})
    assert p.returncode != 0 and f"port :{port}" in p.stderr and "launch.sh --down" in p.stderr
    assert not any(c["args"][0] == "bringup" for c in _franka_rig(h))
    assert [c for c in _rig(h) if c[2] != "instance"] == []


def test_a_foreign_occupancy_server_is_not_reused(franka):
    """The runtime clears the occupancy map (`clear`); a server from another
    checkout may lack it, so only one this launcher started is reused."""
    h = franka
    c = list(h["command"])
    c[c.index("--occupancy") + 1] = "warp"
    with _listening() as port:
        p = _run(h, env={**h["env"], "CASCADE_OCCUPANCY_PORT": str(port)}, command=c)
    assert p.returncode != 0
    assert f"an occupancy server this launcher did not start answers on :{port}" in p.stderr
    assert [c for c in _rig(h) if c[2] != "instance"] == []


def test_graspgenx_auto_means_the_learned_server(franka):
    """`--graspgenx auto` on the FR3 = a GraspGen-X server on :5556 (as on
    every real arm); the robot server is told to use it, though fr3.yaml's
    own default stays the analytic planner."""
    h = franka
    c = list(h["command"])
    c[c.index("--graspgenx") + 1] = "auto"
    p = _run(h, command=c, env={**h["env"], "CASCADE_GRASPGENX_PORT": str(_free_port())})
    assert p.returncode != 0 and "no GraspGen-X server" in p.stderr
    (h["repo"] / "scripts/check_graspgenx.py").write_text("print('learned inference ok')\n")
    with _listening() as port:
        p = _run(h, command=c, env={**h["env"], "CASCADE_GRASPGENX_PORT": str(port)})
    assert p.returncode == 0, p.stdout + p.stderr
    assert _seen_env(h)["CASCADE_GRASP_BACKEND"] == "graspgenx"
    assert "graspgenx: :%d (external)" % port in p.stdout


def test_graspgenx_none_keeps_the_profiles_analytic_planner(franka):
    h = franka
    assert _run(h).returncode == 0
    assert _seen_env(h)["CASCADE_GRASP_BACKEND"] == "obb"


def test_the_mock_fr3_rehearses_the_contract_without_ros(franka):
    """`--arm fr3_mock --cameras mock`: the same Franka operator route (latched
    STOP, readiness, franka agent profile) with no ROS sourcing or bringup."""
    h = franka
    c = list(h["command"])
    c[c.index("--arm") + 1], c[c.index("--cameras") + 1] = "fr3_mock", "mock"
    p = _run(h, command=c, env={**h["env"], "CASCADE_ROS_SETUP": "/nonexistent/setup.bash"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert [x["args"][0] for x in _franka_rig(h)] == ["cpus", "ready"]
    (start,) = [x[2:] for x in _rig(h) if x[2] == "mcp-start"]
    assert "--ros-env" not in start
    (up,) = [x[2:] for x in _rig(h) if x[2] == "up"]
    assert up[up.index("--agent-profile") + 1] == "franka"
    seen = _seen_env(h)
    assert seen["CASCADE_START_STOPPED"] == "1" and "CASCADE_ROS_SETUP" not in seen and "ROS_DOMAIN_ID" not in seen


def test_other_real_arms_keep_their_route(linux_launcher):
    """reBot-style real arms (here the mock arm) are untouched: no ROS,
    no Franka checks, no latched STOP; the agent profile follows the brain
    (this fixture's brain is Cosmos3-Edge: its compact cosmos profile)."""
    h = linux_launcher
    p = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    (start,) = [c[2:] for c in _lines(h["env"]["RIG_LOG"]) if c[2] == "mcp-start"]
    assert "--ros-env" not in start and "--cpus" not in start
    (up,) = [c[2:] for c in _lines(h["env"]["RIG_LOG"]) if c[2] == "up"]
    assert up[up.index("--model") + 1] == "cosmos3-edge" and up[up.index("--agent-profile") + 1] == "cosmos"
    seen = json.loads((Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json").read_text())
    assert not {"CASCADE_START_STOPPED", "CASCADE_ROS_SETUP", "ROS_DOMAIN_ID", "OMP_NUM_THREADS"} & set(seen)
    assert "franka" not in p.stdout and "operator:" not in p.stdout


def test_dry_run_prints_the_franka_plan_and_touches_nothing(franka):
    h = franka
    p = _run(h, "--dry-run")
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert "franka: FR3 operator route (arm=fr3): software STOP latched at start" in out
    assert "brain=nemotron-bolt" in out and "graspgenx=none" in out
    assert "franka: native franka_ros2 bringup is checked, never started (ROS_DOMAIN_ID=31" in out
    assert "franka: would source" in out and "scripts/franka_rig.py bringup --arm fr3" in out
    assert "with the franka agent profile" in out
    assert _rig(h) == [["--instance", "", "instance"]], "only the instance name check, which touches nothing"
    assert [c["args"][0] for c in _franka_rig(h)] == ["cpus"]
    assert not Path(h["env"]["CASCADE_LAUNCH_STATE"]).exists()


def test_check_reports_ros_bringup_and_the_operator_page(franka):
    h = franka
    p = _run(h, "--check")
    out = p.stdout
    assert "[ok]      ROS setup:" in out
    assert "[ok]      franka_ros2 bringup: joint states, fr3 arm controller and Franka Gripper actions answer" in out
    assert "free for the operator STOP / Resume & Home page" in out
    p = _run(h, "--check", env={**h["env"], "FAKE_BRINGUP_DOWN": "1"})
    assert "[MISSING] franka_ros2 bringup:" in p.stdout and p.returncode == 3
    assert _rig(h) == [] or all(c[2] == "instance" for c in _rig(h))


def test_the_franka_route_needs_openshell(launcher_boundary, tmp_path):
    """On macOS (no OpenShell Docker driver) the FR3 launch refuses instead of
    running its agent on the host without the operator profile."""
    h = launcher_boundary
    (h["repo"] / "scripts/franka_rig.py").write_text("print('')\n")
    c = list(h["command"])
    c[c.index("--arm") + 1], c[c.index("--cameras") + 1] = "fr3_mock", "mock"
    p = subprocess.run(c, env=h["env"], capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode != 0 and "runs its agent inside OpenShell, which needs Linux" in p.stderr
    assert not Path(h["env"]["HOST_LOG"]).exists()


def test_the_launch_header_documents_the_franka_command():
    text = (REPO / "scripts/launch.sh").read_text()
    assert "#   ./scripts/launch.sh --sim none --arm fr3 --cameras d435i       # Franka FR3" in text
    doc = (REPO / "docs/FRANKA_OPENCLAW.md").read_text()
    assert "./scripts/launch.sh --sim none --arm fr3 --cameras d435i" in doc
    assert shutil.which("bash")
    assert os.access(REPO / "scripts/ros_env.sh", os.X_OK)
    assert sys.version_info >= (3, 10)


def _registration(monkeypatch, capsys, *franka):
    """launch.sh's registration heredoc, executed verbatim in this process."""
    import re

    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", (REPO / "scripts/launch.sh").read_text(), re.DOTALL)
    [registration] = [block for block in blocks if '"requestTimeoutMs"' in block]
    monkeypatch.setattr(sys, "argv", ["-", "/fixture/python", "/fixture/repo", "d435i", "fr3", "", "none",
                                      "/fixture/state", "f" * 32, "0", "none", *franka])
    exec(compile(registration, str(REPO / "scripts/launch.sh"), "exec"), {})
    return json.loads(capsys.readouterr().out)["env"]


def test_the_registration_writes_the_franka_environment_only_when_asked(monkeypatch, capsys):
    for key in list(os.environ):
        if key.startswith(("CASCADE_", "ROS_", "RMW_")) or key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CASCADE_ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    monkeypatch.setenv("CASCADE_FRANKA_SETUP", "/ws/install/setup.bash")
    monkeypatch.setenv("ROS_DOMAIN_ID", "31")
    monkeypatch.setenv("CASCADE_STREAM", "lazy")
    env = _registration(monkeypatch, capsys, "1", "1")
    assert env["CASCADE_START_STOPPED"] == "1" and env["CASCADE_STREAM"] == "eager", "the contract wins"
    assert (env["CASCADE_ROS_SETUP"], env["CASCADE_FRANKA_SETUP"], env["ROS_DOMAIN_ID"]) == (
        "/opt/ros/jazzy/setup.bash", "/ws/install/setup.bash", "31")
    mock = _registration(monkeypatch, capsys, "1", "0")
    assert mock["CASCADE_START_STOPPED"] == "1" and "CASCADE_ROS_SETUP" not in mock
    # The pre-Franka caller (ten arguments) and an ordinary arm: no Franka keys.
    for env in (_registration(monkeypatch, capsys), _registration(monkeypatch, capsys, "0", "0")):
        assert not {"CASCADE_START_STOPPED", "CASCADE_ROS_SETUP", "ROS_DOMAIN_ID", "OMP_NUM_THREADS"} & set(env)
        assert env["CASCADE_STREAM"] == "lazy", "an operator value is forwarded as before"
