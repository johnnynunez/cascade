"""scripts/launch.sh on Linux: the agent ALWAYS runs inside OpenShell.

The real launcher runs end to end; only the host boundaries are doubled:
dependencies and the runtime build (the launcher_boundary python), the model
HTTP endpoint, Docker, and scripts/openshell_rig.py (its own contract is
tests/test_openshell_rig.py). The fake rig records every call and runs the
`exec -- openclaw ...` it is asked for against the fake OpenClaw CLI with a
SEPARATE log, so a test can tell an OpenClaw call made inside the sandbox
from one made on the host.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_launch_delivery import launcher_boundary as _launcher_boundary
from test_launch_delivery import model_http_boundary as _model_http_boundary

launcher_boundary = _launcher_boundary
model_http_boundary = _model_http_boundary

REPO = Path(__file__).resolve().parents[1]

FAKE_RIG = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:1] and args[0].startswith("--instance="):        # one word: argparse takes any value
    args = ["--instance", args[0].split("=", 1)[1], *args[1:]]
elif args[:1] == ["--instance"] and args[1].startswith("-"):
    # The real rig's argparse reads a separate value starting with '-' as an option
    # (captured from scripts/openshell_rig.py --instance -sim instance).
    print("usage: openshell_rig.py [-h] [--instance NAME]\n"
          "openshell_rig.py: error: argument --instance: expected one argument", file=sys.stderr)
    sys.exit(2)
with open(os.environ["RIG_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
inst, rest = (args[1], args[2:]) if args[:1] == ["--instance"] else ("", args)
if os.environ.get("RIG_BROKEN"):
    raise SystemExit("rig failed for another reason")
# Every command of the real rig resolves its instance first: refuse a name
# exactly as it does (its reason on stderr, exit NAME_REFUSED: the rig's _cli).
import importlib.util
spec = importlib.util.spec_from_file_location("real_openshell_rig", os.environ["RIG_REAL"])
real = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = real
spec.loader.exec_module(real)
try:
    real.instance(inst)
except real.NameRefused as refused:
    print(refused, file=sys.stderr)
    sys.exit(real.NAME_REFUSED)
home = Path(os.environ["RIG_HOME"]) / (inst or "default")
cmd = rest[0]
if cmd == "instance":
    print(json.dumps({"name": inst, "home": str(home)}))
elif cmd == "mcp-start":
    if os.environ.get("RIG_MCP_FAILS"):
        raise SystemExit("cascade-openshell-mcp is already running")
    seen = Path(os.environ["RIG_HOME"]) / "mcp-env.seen.json"
    seen.write_text(Path(rest[rest.index("--env-file") + 1]).read_text())
    seen.with_suffix(".mode").write_text(oct(Path(rest[rest.index("--env-file") + 1]).stat().st_mode & 0o777))
elif cmd == "up":
    home.mkdir(parents=True, exist_ok=True)
    sandbox = f"cascade-{inst}-agent" if inst else "cascade-agent"
    (home / "deployed.json").write_text(json.dumps({"settings": {"sandbox": sandbox,
                                                                 "chat_port": 17910 if inst else 18799}}))
elif cmd == "exec":
    command = rest[rest.index("--") + 1:]
    os.execvpe(command[0], command, {**os.environ, "HOST_LOG": os.environ["SANDBOX_LOG"]})
'''


@pytest.fixture
def linux_launcher(launcher_boundary, tmp_path):
    """launcher_boundary, on Linux: real uname, the OpenShell route's files."""
    h = launcher_boundary
    repo, env = h["repo"], dict(h["env"])
    shutil.copy2(REPO / "scripts/launch_openshell.sh", repo / "scripts/launch_openshell.sh")
    (repo / "scripts/openshell_rig.py").write_text(FAKE_RIG)
    bindir = Path(env["PATH"].split(os.pathsep)[0])
    (bindir / "uname").write_text('#!/bin/bash\nif [[ "${1:-}" == -s ]]; then echo Linux; exit 0; fi\n'
                                  'exec /usr/bin/uname "$@"\n')
    (bindir / "docker").write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n[[ -z "${DOCKER_BROKEN:-}" ]]\n')
    (bindir / "docker").chmod(0o755)
    # Record every script the launcher runs with its python (then run it).
    py = Path(env["PY"])
    real = py.with_name("python-boundary")
    py.rename(real)
    # A MuJoCo launch: mujoco "imports" and the robot MJCF needs no fetch.
    py.write_text(f'#!{sys.executable}\nimport json, os, subprocess, sys\nargs = sys.argv[1:]\n'
                  'open(os.environ["PY_LOG"], "a").write(json.dumps(args) + "\\n")\n'
                  'if args == ["-c", "import mujoco"]:\n    sys.exit(0)\n'
                  'if args[:1] == ["-"]:\n    source = sys.stdin.read()\n'
                  '    if \'.arm.get("mjcf")\' in source:\n        print("")\n        sys.exit(0)\n'
                  f'    sys.exit(subprocess.run([{str(real)!r}, *args], input=source, text=True).returncode)\n'
                  f'os.execv({str(real)!r}, [{str(real)!r}, *args])\n')
    py.chmod(0o755)
    env.update(RIG_LOG=str(tmp_path / "rig.jsonl"), RIG_HOME=str(tmp_path / "rig-home"), PY_LOG=str(tmp_path / "py.jsonl"),
               SANDBOX_LOG=str(tmp_path / "sandbox.jsonl"), DOCKER_LOG=str(tmp_path / "docker.log"),
               RIG_REAL=str(REPO / "scripts/openshell_rig.py"))
    env.pop("CASCADE_OPENSHELL_INSTANCE", None)
    env.pop("CASCADE_BRAIN_KEY_FILE", None)
    env.pop("CASCADE_MODEL_API_KEY", None)
    (tmp_path / "rig-home").mkdir()
    state = Path(env["CASCADE_LAUNCH_STATE"]) / "profile-isolated-test"
    return {**h, "env": env, "tmp": tmp_path, "state": state}


def _lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _run(h, *extra, env=None):
    return subprocess.run([*h["command"], *extra], env=env or h["env"], capture_output=True, text=True, timeout=120, check=False)


def _rig(h) -> list[list[str]]:
    return _lines(Path(h["env"]["RIG_LOG"]))


def test_linux_runs_the_agent_inside_openshell_and_never_on_the_host(linux_launcher):
    """The real robot (`--sim none`) uses the default instance: the robot
    server runs as its unit bound to this launch, the sandbox comes up
    observe-only, and every OpenClaw call (health, tool probe, the proof's
    turns) runs inside the sandbox. OpenClaw on this host is never touched:
    no `mcp set`, no onboarding, no host gateway."""
    h = linux_launcher
    p = _run(h)
    assert p.returncode == 0, p.stdout + p.stderr
    assert not Path(h["env"]["HOST_LOG"]).exists(), "OpenClaw ran on the host"
    rig = _rig(h)
    assert all(c[:2] == ["--instance", ""] for c in rig), "the real robot's route is the default instance"
    steps = [c[2] for c in rig if c[2] != "exec"]
    assert steps == ["instance", "install", "gateway", "mcp-start", "up", "instance"]
    (start,) = [c[2:] for c in rig if c[2] == "mcp-start"]
    assert "--replace" not in start, "a running default (real robot) server is never replaced"
    owner = (h["state"] / "owner.json")
    assert start[start.index("--") + 1:] == ["--launch-owner", json.loads(owner.read_text())["owner"],
                                             "--launch-state-dir", str(h["state"])]
    (up,) = [c[2:] for c in rig if c[2] == "up"]
    port = h["http"]["base_url"].split(":")[-1].split("/")[0]
    assert up[up.index("--brain-port") + 1] == port and up[up.index("--model") + 1] == "cosmos3-edge"
    assert up[up.index("--context-window") + 1] == "32768", "the brain's context, not the template's"
    assert "--observe-only" in up and "--allow-motion" not in up, "the real robot starts locked"
    inside = _lines(Path(h["env"]["SANDBOX_LOG"]))
    assert ["mcp", "probe", "cascade", "--json"] in inside
    assert any(c[:1] == ["agent"] for c in inside) and any(c[:1] == ["health"] for c in inside)
    assert "agent:     OpenClaw inside OpenShell (instance default, sandbox cascade-agent)" in p.stdout
    assert "chat:      http://127.0.0.1:18799/" in p.stdout
    assert "scripts/openshell_rig.py motion unlock" in p.stdout
    assert (h["state"] / "openshell-instance").read_text() == "\n", "--down knows what to tear down"


def test_the_proof_binds_to_the_one_long_lived_robot_server(linux_launcher):
    """The sandbox's chats all talk to the instance's one robot server, so
    the proof binds to that world rather than waiting for a fresh one."""
    h = linux_launcher
    p = _run(h)
    assert p.returncode == 0, p.stdout + p.stderr
    runs = [c for c in _lines(Path(h["env"]["PY_LOG"])) if c[:1] and c[0].endswith("demo_proof.py") and "--run" in c]
    assert len(runs) == 1 and "--persistent-mcp" in runs[0], runs


def test_the_robot_server_gets_the_registrations_environment_privately(linux_launcher):
    h = linux_launcher
    p = _run(h)
    assert p.returncode == 0, p.stdout + p.stderr
    home = Path(h["env"]["RIG_HOME"])
    env = json.loads((home / "mcp-env.seen.json").read_text())
    assert (home / "mcp-env.seen.mode").read_text() == "0o600"
    assert env["CASCADE_ARM"] == "mock" and env["CASCADE_CAMERAS"] == "mock"
    assert env["CASCADE_OPENCLAW_PROFILE"] == "isolated-test", "the owner check needs the launch's profile"
    assert env["CASCADE_OCCUPANCY"] == "0", "--occupancy none reaches the server"
    assert not (h["state"] / "openshell-mcp-env.json").exists(), "the launcher's copy is removed"


def _sim_command(h, sim="mujoco"):
    c = list(h["command"])
    i = c.index("--sim")
    return c[:i] + ["--sim", sim] + c[i + 6:]   # drop "--arm mock --cameras mock"


def test_a_simulator_gets_its_own_instance_with_motion_allowed(linux_launcher):
    h = linux_launcher
    p = subprocess.run(_sim_command(h), env=h["env"], capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    rig = _rig(h)
    assert {tuple(c[:2]) for c in rig} == {("--instance", "sim")}
    (up,) = [c[2:] for c in rig if c[2] == "up"]
    assert up[up.index("--allow-motion"):up.index("--allow-motion") + 3] == ["--allow-motion", "--for", "12h"]
    assert "--replace" in next(c for c in rig if c[2] == "mcp-start")
    assert "allowed for 12 h (simulator)" in p.stdout


STORES = {"CASCADE_GRASP_MEMORY_PATH": "grasp_memory.json", "CASCADE_ENVELOPE_PATH": "envelope.json",
          "CASCADE_BELIEFS_PATH": "beliefs.json", "CASCADE_EPISODIC_PATH": "episodic.json"}


def test_a_simulators_outcomes_never_reach_the_real_robots_stores(linux_launcher):
    """Grasp memory and the learned envelope default to ~/.cascade, which is
    the real rig's: a simulated pick would re-rank the real robot's grasps.
    A simulator's robot server learns into this launch's own state."""
    h = linux_launcher
    env = {**h["env"], "CASCADE_OPENCLAW_PROFILE": ""}   # the profile-less launch (a profile already isolates)
    for key in STORES:
        env.pop(key, None)
    state = Path(env["CASCADE_LAUNCH_STATE"]) / "unprofiled"
    stale = state / "memory" / "beliefs.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"beliefs": [{"label": "red cube", "position": [0.23, 0.0, 0.09]}]}')
    p = subprocess.run(_sim_command(h), env=env, capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    seen = json.loads((Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json").read_text())
    assert {k: seen.get(k) for k in STORES} == {k: str(state / "memory" / f) for k, f in STORES.items()}
    # A simulator's world is new on every launch, so what an earlier launch
    # remembered is wrong by construction. Found live: a cube "recalled" where
    # the previous launch had left it hijacked the next pick (8 pre-grasp IK
    # failures), and that launch's failures had taught grasp memory a nudge.
    assert (state / "memory").is_dir() and not stale.exists()


def test_the_real_robot_keeps_its_stores_and_an_explicit_store_wins(linux_launcher, tmp_path):
    h = linux_launcher
    env = {**h["env"], "CASCADE_OPENCLAW_PROFILE": ""}
    for key in STORES:
        env.pop(key, None)
    assert _run(h, env=env).returncode == 0
    seen = json.loads((Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json").read_text())
    assert not set(STORES) & set(seen), "--sim none: the robot's own (default) stores"
    mine = str(tmp_path / "mine.json")
    p = subprocess.run(_sim_command(h), env={**env, "CASCADE_GRASP_MEMORY_PATH": mine},
                       capture_output=True, text=True, timeout=120, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads((Path(h["env"]["RIG_HOME"]) / "mcp-env.seen.json").read_text())[
        "CASCADE_GRASP_MEMORY_PATH"] == mine


def test_the_brain_key_is_a_private_file_and_never_an_argument(linux_launcher):
    h = linux_launcher
    env = {**h["env"], "CASCADE_MODEL_API_KEY": "sk-local-brain"}
    p = _run(h, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    (up,) = [c[2:] for c in _rig(h) if c[2] == "up"]
    key = Path(up[up.index("--brain-key-file") + 1])
    assert key.read_text().strip() == "sk-local-brain" and key.stat().st_mode & 0o777 == 0o600
    assert not any("sk-local-brain" in a for c in _rig(h) for a in c)


def test_a_cloud_brain_is_refused_before_anything_starts(linux_launcher):
    """`keep` means a model OpenClaw on the host is logged into; the
    sandbox reaches only this host's loopback, so Linux refuses it."""
    h = linux_launcher
    env = {**h["env"], "CASCADE_COSMOS_BASE_URL": "http://127.0.0.1:0/v1"}
    p = _run(h, env=env)
    assert p.returncode != 0
    assert "a cloud model OpenClaw is logged into never enters the sandbox" in p.stderr
    assert [c for c in _rig(h) if c[2] != "instance"] == []


def test_linux_without_docker_refuses_instead_of_falling_back_to_the_host(linux_launcher):
    h = linux_launcher
    p = _run(h, env={**h["env"], "DOCKER_BROKEN": "1"})
    assert p.returncode != 0 and "install Docker" in p.stderr
    assert _rig(h) == [["--instance", "", "instance"]], "only the name check, which touches nothing"
    assert not Path(h["env"]["HOST_LOG"]).exists()


def test_a_running_real_robot_server_stops_the_launch(linux_launcher):
    h = linux_launcher
    p = _run(h, env={**h["env"], "RIG_MCP_FAILS": "1"})
    assert p.returncode != 0 and "robot MCP server did not start" in p.stderr
    assert "up" not in [c[2] for c in _rig(h)]
    assert (h["state"] / "openshell-instance").exists(), "--down can still clean up what started"


def test_down_tears_down_the_instance_the_launch_brought_up(linux_launcher):
    h = linux_launcher
    assert _run(h).returncode == 0
    p = subprocess.run(["bash", str(h["repo"] / "scripts/launch.sh"), "--down"], env=h["env"],
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _rig(h)[-1] == ["--instance", "", "teardown"], "the default (real robot) instance keeps its gateway"
    assert not (h["state"] / "openshell-instance").exists()


def test_down_of_a_simulator_instance_also_stops_its_gateway(linux_launcher):
    h = linux_launcher
    h["state"].mkdir(parents=True, exist_ok=True)
    (h["state"] / "openshell-instance").write_text("sim\n")
    p = subprocess.run(["bash", str(h["repo"] / "scripts/launch.sh"), "--down"], env=h["env"],
                       capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert _rig(h) == [["--instance", "sim", "teardown", "--gateway"]]


@pytest.mark.parametrize("sim, instance", [("mujoco", "sim"), ("isaac", "sim"), ("none", "default")])
def test_the_plan_names_the_route_and_instance(linux_launcher, sim, instance):
    h = linux_launcher
    extra = ["--arm", "mock", "--cameras", "mock"] if sim == "none" else []
    p = subprocess.run(["bash", str(h["repo"] / "scripts/launch.sh"), "--dry-run", "--sim", sim, *extra],
                       env=h["env"], capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"agent: OpenClaw inside OpenShell (instance {instance}); this host runs no agent" in p.stdout
    assert "the OpenClaw sandbox (OpenShell)" in p.stdout


def test_an_explicit_instance_wins(linux_launcher):
    h = linux_launcher
    p = _run(h, env={**h["env"], "CASCADE_OPENSHELL_INSTANCE": "lab"})
    assert p.returncode == 0, p.stdout + p.stderr
    assert {tuple(c[:2]) for c in _rig(h)} == {("--instance", "lab")}
    assert "--replace" in next(c for c in _rig(h) if c[2] == "mcp-start"), "a named instance is the launcher's own"


def test_an_instance_name_openshell_would_refuse_stops_the_launch_before_anything_starts(linux_launcher):
    """Found live: CASCADE_OPENSHELL_INSTANCE=boltpr failed only at `sandbox
    create` (its sandbox name is past OpenShell's 19 characters), after the
    simulator, the gateway and the robot server had started. The launcher
    asks the rig first (`instance` touches nothing) and stops with its reason,
    a dry run included."""
    h = linux_launcher
    env = {**h["env"], "CASCADE_OPENSHELL_INSTANCE": "boltpr"}
    p = subprocess.run(["bash", str(h["repo"] / "scripts/launch.sh"), "--dry-run", "--sim", "mujoco"],
                       env=env, capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode != 0 and "invalid instance name 'boltpr'" in p.stderr, p.stdout + p.stderr
    p = _run(h, env=env)
    assert p.returncode != 0
    assert "invalid instance name 'boltpr'" in p.stderr and "1-5 characters" in p.stderr, p.stderr
    assert "[launch] ERROR: OpenShell instance 'boltpr' refused: invalid instance name" in p.stderr, p.stderr
    assert _rig(h) == [["--instance", "boltpr", "instance"]] * 2, "nothing but the name check"
    assert not (h["state"] / "owner.json").exists() and not (h["state"] / "openshell-instance").exists()
    assert not Path(h["env"]["HOST_LOG"]).exists() and not Path(h["env"]["SANDBOX_LOG"]).exists()


def test_a_name_starting_with_a_dash_is_refused_as_a_name(linux_launcher):
    """The launcher hands the name over as one word (--instance=NAME), so a
    leading '-' reaches the rig's name check instead of its option parser."""
    h = linux_launcher
    p = _run(h, env={**h["env"], "CASCADE_OPENSHELL_INSTANCE": "-sim"})
    assert p.returncode != 0
    assert "OpenShell instance '-sim' refused: invalid instance name '-sim'" in p.stderr, p.stderr
    assert "expected one argument" not in p.stderr


def test_a_rig_that_fails_otherwise_is_not_reported_as_a_refused_name(linux_launcher):
    h = linux_launcher
    p = _run(h, env={**h["env"], "RIG_BROKEN": "1"})
    assert p.returncode != 0
    assert "could not check OpenShell instance 'default'" in p.stderr, p.stderr
    assert "rig failed for another reason" in p.stderr and "refused" not in p.stderr
    assert not Path(h["env"]["HOST_LOG"]).exists() and not Path(h["env"]["SANDBOX_LOG"]).exists()


def test_check_reports_a_refused_name_as_a_failed_row_and_keeps_the_others(linux_launcher):
    h = linux_launcher
    ok = _run(h, "--check")
    p = _run(h, "--check", env={**h["env"], "CASCADE_OPENSHELL_INSTANCE": "boltpr"})
    assert p.returncode == 3 and "preflight FAILED" in p.stdout, p.stdout + p.stderr
    assert "[MISSING] OpenShell instance 'boltpr' refused: invalid instance name 'boltpr'" in p.stdout, p.stdout
    assert "OpenShell installed" not in p.stdout and "OpenShell not installed yet" not in p.stdout
    rows = [line for line in ok.stdout.splitlines() if line.startswith("  [")]
    assert len(rows) >= 4, ok.stdout
    for row in rows:              # every other row is still reported
        if "OpenShell" not in row:
            assert row in p.stdout, (row, p.stdout)
    assert not Path(h["env"]["HOST_LOG"]).exists() and not Path(h["env"]["SANDBOX_LOG"]).exists()


def test_macos_keeps_openclaw_on_the_host(launcher_boundary):
    """The host route is the platform's choice, not a switch: macOS has no
    OpenShell Docker driver (launcher_boundary reports Darwin)."""
    h = launcher_boundary
    p = subprocess.run(h["command"], env=h["env"], capture_output=True, text=True, timeout=60, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "agent: OpenClaw on this host" in p.stdout
    assert any("mcp" in c and "set" in c for c in _lines(Path(h["env"]["HOST_LOG"])))


def test_there_is_no_switch_back_to_the_host_route_on_linux():
    """No flag or variable puts the agent back on a Linux host."""
    text = (REPO / "scripts/launch.sh").read_text()
    decided = [line for line in text.splitlines() if "AGENT_ROUTE=" in line and not line.lstrip().startswith("#")]
    assert decided == ['if [[ "$(uname -s)" == Linux ]]; then AGENT_ROUTE=openshell; else AGENT_ROUTE=host; fi',
                       '[[ "${CASCADE_INSTALL_PROFILE:-}" != spark ]] || AGENT_ROUTE=host'], decided
