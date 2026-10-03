"""Numeric and refusal contracts of the generated passive world transform."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def world_reader(cache, base_z=0.):
    path = Path(__file__).resolve().parents[1] / "demo/kitchen/physics/gpu_proof_audit.py"
    spec = importlib.util.spec_from_file_location("geometry_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(module.scene_geometry_snapshot_code(include_open_box=True))
    world = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_world")
    namespace = {"_cache": cache, "_obs_np": np, "BASE_Z": base_z}
    exec(compile(ast.Module(body=[world], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_world"]


class Cache:
    def __init__(self, matrix):
        self.matrix = matrix
        self.reads = []

    def GetLocalToWorldTransform(self, prim):
        self.reads.append(prim)
        return self.matrix


def test_row_vector_rotation_translation_and_base_height():
    # Positive quarter-turn about Z, translation (4,5,6). This deliberately
    # catches transposition/column-vector conventions independent of NumPy.
    matrix = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                       [0., 0., 1., 0.], [4., 5., 6., 1.]])
    cache = Cache(matrix)
    points = np.array([[1., 2., 3.], [-2., 1., -1.]])
    before = points.copy()
    output = world_reader(cache, base_z=.5)("prim", points)
    np.testing.assert_array_equal(output, [[2., 6., 8.5], [3., 3., 4.5]])
    np.testing.assert_array_equal(points, before)
    assert cache.reads == ["prim"]
    assert not np.shares_memory(output, points)


def test_live_translation_read_on_every_sample():
    cache = Cache(np.eye(4))
    read = world_reader(cache)
    first = read("table", np.zeros((1, 3)))
    cache.matrix[3, :3] = [1., 2., 3.]
    second = read("table", np.zeros((1, 3)))
    np.testing.assert_array_equal(first, [[0., 0., 0.]])
    np.testing.assert_array_equal(second, [[1., 2., 3.]])
    assert cache.reads == ["table", "table"]


def test_general_affine_transform_matches_scalar_reference():
    rng = np.random.default_rng(1234)
    points = rng.normal(size=(64, 3))
    for _ in range(12):
        matrix = np.eye(4)
        matrix[:, :3] = rng.normal(size=(4, 3))
        # Independent scalar formula: arbitrary shear, signed scales and
        # translations remain observable to the unchanged scene auditor.
        expected = [[sum(float(p[k]) * float(matrix[k, j]) for k in range(3))
                     + float(matrix[3, j]) - (.125 if j == 2 else 0.)
                     for j in range(3)] for p in points]
        np.testing.assert_allclose(world_reader(Cache(matrix), .125)("prim", points),
                                   expected, rtol=0, atol=2e-15)


@pytest.mark.parametrize("matrix", [np.eye(3), np.full((4, 4), np.nan),
                                    np.full((4, 4), np.inf),
                                    np.diag([1., 1., 1., 2.]),
                                    np.eye(4) + np.diag([.1], k=3)])
def test_bad_world_transform_is_refused(matrix):
    with pytest.raises(RuntimeError, match="finite affine"):
        world_reader(Cache(matrix))("bad", np.zeros((1, 3)))
