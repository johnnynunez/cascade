"""Exercise the actual native-to-contract assignment without constructing a solver."""
import ast
from dataclasses import replace
import inspect
from types import SimpleNamespace as NS

import numpy as np
import pytest

import cascade.sim.factory_model as model
from cascade.sim.factory_recipe import LEGACY_RECIPE, MARGIN_RECIPE, seating_recipe
from test_factory_owner import model_fixture
from test_fastening_runtime import binding, limits


def constructor_binding(scene):
    # Full constructor doubles would conceal SDK drift. Execute its exact AST
    # binding statement, after the real authoring check, with synthetic labels.
    tree = ast.parse(inspect.getsource(model))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FactoryBoundModel")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    statements = [n for n in init.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == "self" and t.attr == "binding" for t in n.targets)]
    assert len(statements) == 1
    assert isinstance(statements[0].value, ast.Call)
    assert statements[0].value.func.id == "FasteningBinding"
    result = NS(document={"dt_s": float(np.float32(1/600)),
        "authoring": model.authoring_descriptor(scene)}, limits=limits())
    thread = ("nut", "bolt")
    tools = tuple(("nut", "socket_wall_"+str(i)) for i in range(6))
    scene.model.shape_label = ["nut", "bolt", *(p[1] for p in tools)]
    names = ("joint", "socket_spin")
    namespace = dict(vars(model), self=result, scene=scene, thread=thread, tools=tools, names=names)
    code = compile(ast.Module(body=statements, type_ignores=[]), inspect.getfile(model), "exec")
    exec(code, namespace)
    return result.binding


@pytest.mark.parametrize("recipe", [LEGACY_RECIPE, MARGIN_RECIPE])
def test_constructor_binding_converts_validated_native_origin_without_changing_values(recipe):
    scene = model_fixture()
    r = seating_recipe(recipe)
    scene.fixture_recipe = recipe
    scene.fixture_center_xy_m = r.center_xy_m
    scene._center_xy = np.array(r.center_xy_m)
    scene.fixture_position = np.r_[scene._center_xy, 0.]
    scene.ik_margin_rad = r.ik_margin_rad
    scene.intersect_position_control_range = r.intersect_position_control_range
    original = scene.fixture_position.copy()
    assert all(type(v) is np.float64 for v in original)
    result = constructor_binding(scene)
    assert all(type(v) is float for v in result.fixture_origin_m)
    assert result.fixture_origin_m == (*r.center_xy_m, 0.)
    assert result.fixture_recipe == recipe
    assert result.thread_pitch_m == .0025
    assert result.dt_s == float(np.float32(1/600))
    np.testing.assert_array_equal(scene.fixture_position, original)
    assert scene.fixture_position.dtype == np.float64


@pytest.mark.parametrize("bad", [np.float64(.24), np.float32(.24), True, ".24", float("nan"), float("inf")])
def test_public_binding_numeric_contract_still_refuses_non_python_or_invalid_scalars(bad):
    with pytest.raises(ValueError, match="fixture origin"):
        replace(binding(), fixture_origin_m=(bad, 0., 0.))


def test_constructor_binding_does_not_bypass_real_authoring_validation():
    scene = model_fixture()
    scene.fixture_position[0] += .001
    with pytest.raises(model.FasteningFault, match="authoring differs"):
        constructor_binding(scene)
