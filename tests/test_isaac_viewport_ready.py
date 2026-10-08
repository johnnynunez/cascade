"""Headless Newton bridges switch the viewport off only after Kit reports app-ready.

On the internal Isaac Sim 6.2 build the full (Newton) app experience loads
`isaacsim.app.setup`, which waits for the viewport's first frame before it
releases app-ready. `SimulationApp(disable_viewport_updates=True)` stops the
viewport at construction, so that frame never arrives and `open_stage()` never
returns (measured 2026-10-07: ~25k frames of "await_viewport: waiting for
viewport handle", no bridge output). These tests pin the gate and the helper
without booting Kit.
"""
import ast
from pathlib import Path
import sys
import types

import pytest

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "scripts/isaac_bridge.py"


def _tree():
    return ast.parse(BRIDGE.read_text())


def _assignment(tree, name):
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    assert len(nodes) == 1, f"expected one top-level assignment of {name}"
    return nodes[0]


@pytest.mark.parametrize("gui, kwargs, expected", [
    (False, {"experience": "/r/apps/isaacsim.exp.full.newton.kit"}, True),   # headless Newton
    (False, {}, False),                                                       # headless PhysX (default app)
    (True, {"experience": "/r/apps/isaacsim.exp.full.newton.kit"}, False),   # GUI keeps the viewport
    (True, {}, False),
])
def test_only_headless_full_experience_launches_defer_the_viewport_switch(gui, kwargs, expected):
    node = _assignment(_tree(), "_defer_viewport_off")
    value = eval(compile(ast.Expression(node.value), "defer", "eval"),
                 {"args": types.SimpleNamespace(gui=gui), "_kwargs": dict(kwargs)})
    assert value is expected


def test_newton_selects_the_full_experience_before_the_gate_is_computed():
    tree = _tree()
    gate = _assignment(tree, "_defer_viewport_off")
    app = _assignment(tree, "app")
    newton_ifs = [node for node in tree.body if isinstance(node, ast.If)
                  and "newton" in ast.unparse(node.test) and "experience" in ast.unparse(node)]
    assert newton_ifs and newton_ifs[0].lineno < gate.lineno < app.lineno


def test_deferred_switch_runs_before_the_stage_is_opened():
    tree = _tree()
    calls = [node for node in tree.body if isinstance(node, ast.If)
             and ast.unparse(node.test) == "_defer_viewport_off"
             and "_disable_viewport_updates_once_ready(app)" in ast.unparse(node)]
    assert len(calls) == 1
    opens = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func).endswith("open_stage")]
    assert opens, "bridge no longer opens a stage?"
    assert calls[0].lineno < min(node.lineno for node in opens)


class _Kit:
    def __init__(self, ready_after):
        self.ready_after = ready_after
        self.polls = 0

    def is_app_ready(self):
        self.polls += 1
        return self.ready_after is not None and self.polls > self.ready_after


class _App:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


def _helper(monkeypatch, kit, viewport, utility_fails=False):
    tree = _tree()
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_disable_viewport_updates_once_ready"]
    assert len(functions) == 1
    kit_app = types.ModuleType("omni.kit.app")
    kit_app.get_app = lambda: kit
    omni = types.ModuleType("omni")
    omni_kit = types.ModuleType("omni.kit")
    omni_viewport = types.ModuleType("omni.kit.viewport")
    omni.kit = omni_kit
    omni_kit.app = kit_app
    omni_kit.viewport = omni_viewport
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit)
    monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app)
    utility = types.ModuleType("omni.kit.viewport.utility")
    if utility_fails:
        def get_active_viewport():
            raise RuntimeError("no viewport extension")
    else:
        def get_active_viewport():
            return viewport
    utility.get_active_viewport = get_active_viewport
    omni_viewport.utility = utility
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", omni_viewport)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", utility)
    scope = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "viewport_helper", "exec"), scope)
    return scope["_disable_viewport_updates_once_ready"]


def test_viewport_renders_until_app_ready_then_stops(monkeypatch, capsys):
    kit, app = _Kit(ready_after=3), _App()
    viewport = types.SimpleNamespace(updates_enabled=True)
    _helper(monkeypatch, kit, viewport)(app, max_updates=50)
    assert app.updates == 3            # rendered exactly until Kit said ready
    assert viewport.updates_enabled is False
    assert "disabled after app-ready" in capsys.readouterr().out


def test_never_ready_leaves_the_viewport_on_and_says_so(monkeypatch, capsys):
    kit, app = _Kit(ready_after=None), _App()
    viewport = types.SimpleNamespace(updates_enabled=True)
    _helper(monkeypatch, kit, viewport)(app, max_updates=7)
    assert app.updates == 7
    assert viewport.updates_enabled is True   # a deadlock is not hidden by turning rendering off
    assert "never reported app-ready" in capsys.readouterr().out


def test_missing_viewport_utility_keeps_rendering(monkeypatch, capsys):
    kit, app = _Kit(ready_after=0), _App()
    viewport = types.SimpleNamespace(updates_enabled=True)
    _helper(monkeypatch, kit, viewport, utility_fails=True)(app, max_updates=5)
    assert viewport.updates_enabled is True
    assert "left ON" in capsys.readouterr().out
