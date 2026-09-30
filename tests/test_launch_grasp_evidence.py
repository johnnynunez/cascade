"""The existing launcher must opt the persistent native MCP process in."""
import json
from pathlib import Path
import subprocess

import pytest

from test_launch_delivery import launcher_boundary, model_http_boundary


@pytest.mark.parametrize('enabled', [False, True])
def test_grasp_evidence_env_reaches_only_the_existing_mcp_registration(launcher_boundary, tmp_path, enabled):
    h = launcher_boundary
    h['env'].pop('CASCADE_GRASP_EVIDENCE_DIR', None)
    directory = str(tmp_path / 'diagnostic evidence')
    if enabled:
        h['env']['CASCADE_GRASP_EVIDENCE_DIR'] = directory
    result = subprocess.run(h['command'], env=h['env'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads(Path(h['env']['MCP_CONFIG']).read_text())
    assert config['env'].get('CASCADE_GRASP_EVIDENCE_DIR') == (directory if enabled else None)
    assert not Path(directory).exists()  # registration itself does not capture data
