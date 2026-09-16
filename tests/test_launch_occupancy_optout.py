"""A launch flag must reach the persistent MCP runtime it configures."""
import json
from pathlib import Path
import subprocess

from test_launch_delivery import launcher_boundary, model_http_boundary


def test_explicit_no_occupancy_reaches_the_registered_mcp(launcher_boundary):
    h = launcher_boundary
    cmd = list(h["command"])
    if "--occupancy" in cmd:
        cmd[cmd.index("--occupancy") + 1] = "none"
    else:
        cmd += ["--occupancy", "none"]
    result = subprocess.run(cmd, env=h["env"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(Path(h["env"]["MCP_CONFIG"]).read_text())
    assert config["env"]["CASCADE_OCCUPANCY"] == "0"
