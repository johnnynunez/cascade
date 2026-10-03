"""Synthetic reader tests; native tensor parity is recorded separately."""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROPS = ("pink_cube", "green_cube")


def reader_fixture(*, missing_schema=False, wrong_paths=False, wrong_count=False,
                   bad_shape=False, backend_error=False, backend_reorder=False,
                   bad_native=None, transpose=None):
    path = Path(__file__).resolve().parents[1] / "demo/kitchen/observer/observer.py"
    spec = importlib.util.spec_from_file_location("batch_observer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(module.build_snapshot_code(props=PROPS))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    calls = {"constructors": [], "poses": 0, "velocities": 0, "readbacks": 0, "tensor": False}

    class Tensor:
        device = "cuda:synthetic"

        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

        def numpy(self):
            calls["readbacks"] += 1
            return self.value.copy()

    class Rigid:
        def __init__(self, paths, **kwargs):
            assert calls["tensor"] is False
            calls["constructors"].append((paths, kwargs))
            self.paths = [paths] if isinstance(paths, str) else list(paths)
            self.batch = not isinstance(paths, str)
            self.valid = True
            if wrong_paths and self.batch:
                self.paths.reverse()
            native = list(reversed(self.paths)) if backend_reorder else self.paths.copy()
            if bad_native == "duplicate":
                native[-1] = native[0]
            elif bad_native == "foreign":
                native[-1] = "/World_Props/unrequested"
            self._physics_rigid_body_view = SimpleNamespace(prim_paths=native,
                count=len(native) - int(bad_native == "count"))
            if bad_native == "unavailable":
                self._physics_rigid_body_view = None

        def indices(self):
            return [PROPS.index(path.rsplit("/", 1)[1])
                    for path in self._physics_rigid_body_view.prim_paths]

        def __len__(self):
            return len(self.paths) - int(wrong_count and self.batch)

        def is_physics_tensor_entity_valid(self):
            return self.valid

        def get_world_poses(self):
            assert calls["tensor"] and self.batch
            calls["poses"] += 1
            points = [[i + .1, 2., 3.] for i in self.indices()]
            if bad_shape:
                points.pop()
            points = np.asarray(points)
            quaternions = np.asarray([[1., 0., 0., 0.]] * len(self.paths))
            return Tensor(points.T if transpose == "positions" else points), Tensor(
                quaternions.T if transpose == "orientations" else quaternions)

        def get_velocities(self):
            assert calls["tensor"] and self.batch
            calls["velocities"] += 1
            linear = np.asarray([[i + .4, 5., 6.] for i in self.indices()])
            angular = np.asarray([[i + .7, 8., 9.] for i in self.indices()])
            return (Tensor(linear.T if transpose == "velocities" else linear),
                    Tensor(angular.T if transpose == "angular" else angular))

    @contextmanager
    def backend(name, **kwargs):
        assert name == "tensor"
        assert kwargs == {"raise_on_unsupported": True, "raise_on_fallback": True}
        if backend_error:
            raise RuntimeError("unsupported tensor backend")
        calls["tensor"] = True
        try:
            yield
        finally:
            calls["tensor"] = False

    namespace = {
        "_obs_np": np, "_obs_RP": Rigid, "_obs_backend": backend,
        "_tl": SimpleNamespace(is_playing=lambda: True), "engine": "physx",
        "_obs_SM": SimpleNamespace(get_simulation_time=lambda: 2.5, get_num_physics_steps=lambda: 300),
        "stage": SimpleNamespace(GetPrimAtPath=lambda _: SimpleNamespace(HasAPI=lambda _: not missing_schema)),
        "_obs_UP": SimpleNamespace(RigidBodyAPI="rigid"),
        "_obs_PX": SimpleNamespace(PhysxRigidBodyAPI="physx_rigid"),
        "art": SimpleNamespace(is_physics_tensor_entity_valid=lambda: True,
                               get_dof_positions=lambda: Tensor([[1., .2, .8]]),
                               get_dof_velocities=lambda: Tensor([[0., .01, -.01]]),
                               get_dof_limits=lambda: (Tensor([[0., 0., 0.]]), Tensor([[2., 1., 1.]]))),
        "GRIP_IDX": (1, 2), "step": 123, "BASE_Z": .5,
        "args": SimpleNamespace(prim="synthetic_arm"), "names": ["j0", "left", "right"],
        "_obs_time": SimpleNamespace(monotonic=lambda: 10.), "_frames": {},
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_obs_collect"], namespace, calls


def test_one_batch_preserves_named_pose_velocity_fields_and_individual_views():
    read, namespace, calls = reader_fixture()
    sample = read()
    assert sample["physics_step"] == 300 and sample["sim_time"] == 2.5
    assert sample["q"] == [1., .2, .8]
    for index, prop in enumerate(PROPS):
        assert sample["props"][prop] == {
            "position_m": [index + .1, 2., 2.5], "orientation_wxyz": [1., 0., 0., 0.],
            "linear_velocity_m_s": [index + .4, 5., 6.],
            "angular_velocity_rad_s": [index + .7, 8., 9.], "tensor_device": "cuda:synthetic"}
    assert calls["poses"] == calls["velocities"] == 1
    assert calls["readbacks"] == 8  # four articulation arrays + four rigid batches
    assert len(namespace["_kitchen_passive_observer_views_v1"]) == 2
    paths, options = calls["constructors"][-1]
    assert paths == ["/World_Props/" + name for name in PROPS]
    assert options == {"resolve_paths": False, "reset_xform_op_properties": False}
    assert all(not options["reset_xform_op_properties"] for _, options in calls["constructors"])


def test_valid_view_reused_but_every_sample_reads_live_tensor_arrays():
    read, _, calls = reader_fixture()
    read()
    constructors = len(calls["constructors"])
    read()
    assert len(calls["constructors"]) == constructors
    assert calls["poses"] == calls["velocities"] == 2
    assert calls["readbacks"] == 16


def test_invalidated_individual_and_batch_views_are_recreated():
    read, namespace, calls = reader_fixture()
    read()
    old_batch = namespace["_kitchen_passive_observer_batch_v1"]["view"]
    old_individual = namespace["_kitchen_passive_observer_views_v1"]["/World_Props/pink_cube"]
    old_batch.valid = old_individual.valid = False
    read()
    assert len(calls["constructors"]) == 5
    assert namespace["_kitchen_passive_observer_batch_v1"]["view"] is not old_batch
    assert namespace["_kitchen_passive_observer_views_v1"]["/World_Props/pink_cube"] is not old_individual


@pytest.mark.parametrize("error", ["wrong_paths", "wrong_count"])
def test_path_or_count_mismatch_refuses_before_tensor_reads(error):
    read, _, calls = reader_fixture(**{error: True})
    with pytest.raises(RuntimeError, match="path/count binding"):
        read()
    assert calls["poses"] == calls["velocities"] == calls["readbacks"] == 0


def test_missing_schema_never_creates_a_view_or_adds_schema():
    read, _, calls = reader_fixture(missing_schema=True)
    with pytest.raises(RuntimeError, match="refuses to add missing"):
        read()
    assert calls["constructors"] == [] and calls["readbacks"] == 0


def test_bad_tensor_shape_is_not_truncated_or_broadcast():
    read, _, _ = reader_fixture(bad_shape=True)
    with pytest.raises(RuntimeError, match="unexpected shape"):
        read()


@pytest.mark.parametrize("array", ["positions", "orientations", "velocities", "angular"])
def test_transposed_tensor_with_same_element_count_is_refused(array):
    read, _, _ = reader_fixture(transpose=array)
    with pytest.raises(RuntimeError, match="unexpected shape"):
        read()


def test_actual_backend_order_drives_named_output_not_requested_order():
    ordered, _, _ = reader_fixture()
    reversed_rows, _, _ = reader_fixture(backend_reorder=True)
    assert reversed_rows()["props"] == ordered()["props"]


@pytest.mark.parametrize("bad", ["duplicate", "foreign", "count", "unavailable"])
def test_missing_or_ambiguous_native_row_identity_refused_before_reads(bad):
    read, _, calls = reader_fixture(bad_native=bad)
    with pytest.raises(RuntimeError, match="native tensor row identity"):
        read()
    assert calls["poses"] == calls["velocities"] == calls["readbacks"] == 0


def test_unsupported_tensor_backend_is_not_replaced_by_usd():
    read, _, calls = reader_fixture(backend_error=True)
    with pytest.raises(RuntimeError, match="unsupported tensor"):
        read()
    assert calls["poses"] == calls["velocities"] == calls["readbacks"] == 0
