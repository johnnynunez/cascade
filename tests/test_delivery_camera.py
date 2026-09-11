"""Camera configuration contract; actual framing is checked on Spark."""
import ast
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("enabled", [False, True])
def test_recording_camera_is_opt_in_and_preserves_perception(monkeypatch, enabled):
    source = Path(__file__).resolve().parents[1] / "scripts/isaac_bridge.py"
    tree = ast.parse(source.read_text())
    nodes: list[ast.stmt] = [n for n in tree.body if (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CAM_DEFS" for t in n.targets)
    ) or (isinstance(n, ast.If) and "CASCADE_PROOF_CAMERA" in ast.unparse(n.test))]
    monkeypatch.setenv("CASCADE_PROOF_CAMERA", "1" if enabled else "0")
    def camera(path, eye, target, up, **kw):
        return {"path": path, "eye": eye, "target": target, **kw}
    namespace = {"os": os, "_camera": camera}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    cameras = namespace["CAM_DEFS"]
    assert {"cam0", "side", "wrist"}.issubset(cameras)
    assert cameras["cam0"]["eye"] == (0.78, -0.35, 0.60)
    assert cameras["side"]["eye"] == (0.95, 0.75, 0.45)
    assert ("proof" in cameras) is enabled
    if enabled:
        assert cameras["proof"]["path"] == "/World_Cams/proof"
        assert cameras["proof"]["eye"] == (1.1, -1.1, 0.9)
        assert cameras["proof"]["target"] == (0.20, 0.02, 0.30)
