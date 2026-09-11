"""The real GUI framing helper must use the viewport manager after scene warmup."""
import ast
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parents[1]


def helper(monkeypatch, gui=True):
    tree = ast.parse((REPO / "scripts/isaac_bridge.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_frame_gui_viewport"]
    assert len(functions) == 1, "missing canonical GUI framing helper"
    manager = types.ModuleType("isaacsim.core.rendering_manager")
    manager.ViewportManager = types.SimpleNamespace(set_camera_view=Mock())
    monkeypatch.setitem(sys.modules, "isaacsim.core.rendering_manager", manager)
    scope = {"args": types.SimpleNamespace(gui=gui), "BASE_Z": 0.1, "U": 2.0}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "isaac_gui_helper", "exec"), scope)
    return scope["_frame_gui_viewport"], manager.ViewportManager.set_camera_view, tree


def test_gui_camera_uses_rendering_manager_in_stage_units(monkeypatch):
    frame, set_view, _ = helper(monkeypatch)
    frame()
    set_view.assert_called_once()
    args, kwargs = set_view.call_args
    assert args == ("/OmniverseKit_Persp",)
    assert kwargs["eye"] == pytest.approx([2.3, 1.7, 1.6])
    assert kwargs["target"] == pytest.approx([0.4, 0.03, 0.6])


def test_gui_camera_does_not_change_headless_perception(monkeypatch):
    frame, set_view, _ = helper(monkeypatch, gui=False)
    frame()
    set_view.assert_not_called()


def test_gui_camera_framing_runs_after_physics_warmup(monkeypatch):
    _, _, tree = helper(monkeypatch)
    calls = {n.value.func.id: i for i, n in enumerate(tree.body)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)}
    assert calls["_frame_gui_viewport"] > calls["_settle_props"]
