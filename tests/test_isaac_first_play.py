"""Exercise actual first-play ordering without importing Kit or using a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("callback_fails", [False, True])
def test_first_play_completes_deferred_callbacks_before_articulation(callback_fails):
    tree = ast.parse((ROOT / "scripts/isaac_bridge.py").read_text())
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "app_utils.play")
    end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "art" for target in node.targets))
    events = []
    pending = False
    initialized = False
    updates = 0

    def play(*, commit):
        nonlocal pending
        assert commit is False, "first-play callbacks must run inside a normal app update"
        pending = True
        events.append("queued")

    def update():
        nonlocal pending, initialized, updates
        updates += 1
        if pending:
            events.append("physics callback")
            if callback_fails:
                raise RuntimeError("physics callback failed")
            initialized, pending = True, False
        events.append("update complete")

    def articulation(prim):
        assert initialized and not pending
        assert updates == 10
        assert prim == "/fixture/robot"
        events.append("articulation")
        return object()

    scope = {"app_utils": SimpleNamespace(play=play), "app": SimpleNamespace(update=update),
             "args": SimpleNamespace(engine="physx", prim="/fixture/robot"),
             "_fix_gravity": lambda: events.append("gravity"),
             "SimulationManager": SimpleNamespace(get_active_physics_engine=lambda: "physx"),
             "Articulation": articulation}
    code = compile(ast.Module(body=tree.body[start:end + 1], type_ignores=[]), "actual_first_play", "exec")
    if callback_fails:
        with pytest.raises(RuntimeError, match="physics callback failed"):
            exec(code, scope)
        assert events == ["queued", "physics callback"]
        assert updates == 1 and "art" not in scope, "a failed callback must not be retried or bypassed"
    else:
        exec(code, scope)
        assert events == ["queued", "physics callback", *(["update complete"] * 10), "gravity", "articulation"]
