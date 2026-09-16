"""Run the shipped launch boundaries with synthetic processes and TCP state.

These tests never start Isaac, OpenClaw, a model, or an acceptance case.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socketserver
import subprocess
import sys
import threading

import pytest

from conftest import loopback_host


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/launch.sh"


def launch_boundary(tmp_path):
    """Execute the real option block and bridge argument builder in a shell."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = LAUNCH.read_text()
    prefix = source[:source.index("\nlog()  {")]
    begin = source.index("        isaac_args=(")
    end = source.index('        log "starting Isaac Sim bridge', begin)
    argument_builder = source[begin:end]
    entry = scripts / "launch-boundary.sh"
    entry.write_text(prefix + '''
"$BOUNDARY_PY" - "$ARM" "$CAMERAS" "$BRAIN" "$OCCUPANCY" "$GRASPGENX" "$SCENE_CONFIG" <<'PY'
import json, os, sys
print(json.dumps({"selection": sys.argv[1:], "occupancy": os.environ.get("CASCADE_OCCUPANCY")}), flush=True)
PY
ISAAC_PY="$BOUNDARY_PY"
USD="$REPO/robot.usda"
''' + argument_builder + '''
exec "$BOUNDARY_PY" "$BOUNDARY_SOURCE/scripts/isaac_launch.py" --python "$ISAAC_PY" -- \
    "$REPO/scripts/isaac_bridge.py" "${isaac_args[@]}"
''')
    (scripts / "isaac_bridge.py").write_text('''import json, os, sys
keys = ("CASCADE_REQUIRE_CUDA", "CASCADE_PHYSICS_DEVICE", "CASCADE_ISAAC_DT", "PAAI_CAMERA_VIDEO_CONFIG")
print(json.dumps({"argv": sys.argv[1:], "environment": {key: os.environ[key] for key in keys if key in os.environ}}))
''')
    environment = {
        "HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin",
        "CASCADE_INSTALL_PROFILE": "spark", "CASCADE_LAUNCH_STATE": str(tmp_path / "state"),
        "BOUNDARY_PY": sys.executable, "BOUNDARY_SOURCE": str(ROOT),
        "PAAI_CAMERA_VIDEO_CONFIG": str(tmp_path / "optional-video.json"),
    }
    return repo, entry, environment


@pytest.mark.parametrize("arguments", [[], ["--engine", "physx", "--arm", "isaac_kitchen_gpu", "--occupancy", "none"]])
def test_spark_default_passes_brev_engine_profile_and_cuda_to_bridge(tmp_path, arguments):
    repo, entry, environment = launch_boundary(tmp_path)
    result = subprocess.run(["bash", str(entry), *arguments], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    selected, child = map(json.loads, result.stdout.splitlines())
    assert selected == {
        "selection": ["isaac_kitchen_gpu", "isaac,isaac_side,isaac_proof", "qwen", "none", "none",
                      str(repo / "demo/scene/kitchen_config.json")],
        "occupancy": "0",
    }
    assert child["argv"] == ["--port", "8611", "--usd", str(repo / "robot.usda"), "--gui",
                             "--engine", "physx", "--scene-config", str(repo / "demo/scene/kitchen_config.json")]
    assert child["environment"] == {
        "CASCADE_REQUIRE_CUDA": "1", "CASCADE_PHYSICS_DEVICE": "cuda:0",
        "CASCADE_ISAAC_DT": "0.008333333333333333",
    }
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("arguments", [
    ["--engine", "newton"], ["--arm", "isaac"], ["--occupancy", "warp"],
])
def test_spark_rejects_conflicting_event_options_before_mutation(tmp_path, arguments):
    _, entry, environment = launch_boundary(tmp_path)
    result = subprocess.run(["bash", str(entry), *arguments], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Spark event delivery" in result.stderr
    assert not result.stdout
    assert not (tmp_path / "state").exists()
    assert not Path(environment["HOME"]).exists()


def runtime_identity():
    return {
        "ok": True, "engine": "physx", "scene_config": "/fixture/kitchen_config.json",
        "scene_config_sha256": "a" * 64, "physics_dt_s": 1 / 120,
        "physics_gpu": True, "physics_device": "cuda:0", "physics_tensor_device": "cuda:0",
        "gpu_attestation": {
            "required": True, "backend": "physx", "device": "cuda:0", "tensor_device": "cuda:0",
            "tensor_device_ordinal": 0, "cuda_context_present": True, "gpu_dynamics": True,
            "broadphase": "GPU", "cpu_fallback_allowed": False,
        },
    }


@pytest.mark.parametrize("fault", [
    None, "newton", "cpu", "unguarded", "no_context", "cpu_broadphase", "different_tensor",
    "wrong_ordinal", "fallback_allowed", "wrong_scene", "wrong_dt",
])
def test_spark_probe_requires_actual_brev_runtime_identity(monkeypatch, fault, capsys):
    """A healthy socket cannot substitute for the live engine/device/scene."""
    pong = runtime_identity()
    if fault == "newton":
        pong["engine"] = "newton"
    elif fault == "cpu":
        pong["physics_gpu"] = False
    elif fault == "unguarded":
        pong["gpu_attestation"].pop("required")
    elif fault == "no_context":
        pong["gpu_attestation"]["cuda_context_present"] = False
    elif fault == "cpu_broadphase":
        pong["gpu_attestation"]["broadphase"] = "MBP"
    elif fault == "different_tensor":
        pong["physics_tensor_device"] = "cuda:1"
    elif fault == "wrong_ordinal":
        pong["gpu_attestation"]["tensor_device_ordinal"] = 1
    elif fault == "fallback_allowed":
        pong["gpu_attestation"]["cpu_fallback_allowed"] = True
    elif fault == "wrong_scene":
        pong["scene_config_sha256"] = "b" * 64
    elif fault == "wrong_dt":
        pong["physics_dt_s"] = 1 / 60
    requests = []

    class Bridge(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                op = json.loads(line)["op"]
                requests.append(op)
                reply = pong if op == "ping" else {"ok": True, "q": [0.0, 1.2, 1.2, 0.0, 0.75, 0.0]}
                self.wfile.write(json.dumps(reply).encode() + b"\n")
                self.wfile.flush()

    with socketserver.ThreadingTCPServer((loopback_host(), 0), Bridge) as server:
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        worker.start()
        try:
            blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", LAUNCH.read_text(), flags=re.DOTALL)
            [probe] = [block for block in blocks if "Isaac bridge answers:" in block]
            monkeypatch.setenv("CASCADE_INSTALL_PROFILE", "spark")
            monkeypatch.setenv("CASCADE_ISAAC_DT", "0.008333333333333333")
            monkeypatch.setattr(sys, "argv", ["-", str(server.server_address[1]), "physx",
                                               "/fixture/kitchen_config.json", "a" * 64, "2", ""])
            # Preserve the real TCP client while honoring the repository's VPN-safe host fixture.
            from cascade.sim import bridge_client
            real_client = bridge_client.BridgeClient
            monkeypatch.setattr(bridge_client, "BridgeClient",
                                lambda **kwargs: real_client(host=loopback_host(), **kwargs))
            if fault is None:
                exec(compile(probe, str(LAUNCH), "exec"), {})
                assert "state_ok=True" in capsys.readouterr().out
                assert requests == ["ping", "state"]
            else:
                with pytest.raises(AssertionError):
                    exec(compile(probe, str(LAUNCH), "exec"), {})
                assert requests == ["ping"], "invalid identity must fail before later runtime activity"
        finally:
            server.shutdown()
            worker.join(timeout=2)
