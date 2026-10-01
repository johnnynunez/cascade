"""A successful map reset discards observations and allocator history."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from cascade.perception.occupancy_backends import NvbloxBackend


@pytest.fixture
def native_api(monkeypatch):
    state = {"instances": [], "fail_construct": False, "fail_sync": False}

    class Params:
        def set_projective_integrator_params(self, value):
            self.projective = value

    class Mapper:
        def __init__(self, *, voxel_sizes_m, mapper_parameters):
            if state["fail_construct"]:
                raise RuntimeError("allocation failed")
            self.voxel = voxel_sizes_m
            self.params = mapper_parameters
            self.observations = []
            state["instances"].append(self)

        def add_depth_frame(self, depth, pose, sensor, mask_frame=None):
            self.observations.append(depth)

        def clear(self):
            raise AssertionError("the old allocator must not survive the reset")

    def synchronize():
        if state["fail_sync"]:
            raise RuntimeError("CUDA sync failed")

    modules = {
        "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
            current_device=lambda: 0, synchronize=synchronize)),
        "nvblox_torch": SimpleNamespace(),
        "nvblox_torch.mapper": SimpleNamespace(Mapper=Mapper, QueryType=SimpleNamespace(ESDF=1)),
        "nvblox_torch.mapper_params": SimpleNamespace(MapperParams=Params,
            ProjectiveIntegratorParams=SimpleNamespace),
        "nvblox_torch.constants": SimpleNamespace(constants=SimpleNamespace(esdf_unknown_distance=lambda: -1.0)),
        "nvblox_torch.sensor": SimpleNamespace(Sensor=SimpleNamespace()),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def test_clear_preserves_nondefault_geometry_and_resets_observations(native_api):
    backend = NvbloxBackend(voxel=.007, max_integration_m=1.25,
                           region_min=(.1, -.2, .03), region_max=(.45, .3, .5))
    spec = backend.spec
    previous = backend.mapper
    previous.observations.append("measured surface")
    backend.last_integrate_ms = 12.0
    backend.last_query_ms = 8.0
    backend.clear()
    assert backend.mapper is not previous
    assert backend.mapper.observations == []
    assert backend.mapper.voxel == .007
    assert backend.mapper.params is previous.params
    assert backend.mapper.params.projective.projective_integrator_max_integration_distance_m == 1.25
    assert backend.spec is spec and backend.masked_depth is True
    assert backend.last_integrate_ms == backend.last_query_ms == 0.0


@pytest.mark.parametrize("failure", ["fail_construct", "fail_sync"])
def test_failed_clear_blocks_queries_and_integration_until_explicit_retry(native_api, failure):
    backend = NvbloxBackend(voxel=.005)
    previous = backend.mapper
    previous.observations.append("old map")
    native_api[failure] = True
    with pytest.raises(RuntimeError, match="failed"):
        backend.clear()
    assert backend.mapper is None
    with pytest.raises(RuntimeError, match="clear must complete"):
        backend.query((0, 0, 0), (.1, .1, .1))
    with pytest.raises(RuntimeError, match="clear must complete"):
        backend.integrate_depth(np.ones((2, 2)), np.eye(3), np.eye(4))
    native_api[failure] = False
    backend.clear()
    assert backend.mapper is not previous and backend.mapper.observations == []


@pytest.mark.hardware
def test_cuda_repeated_clear_is_unknown_then_reproduces_the_same_grid():
    pytest.importorskip("nvblox_torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native map reset requires CUDA")
    backend = NvbloxBackend(voxel=.01, max_integration_m=1.25)
    depth = np.full((120, 160), .8, np.float32)
    depth[35:85, 55:105] = .4
    active = np.ones(depth.shape, np.uint8)
    active[35:85, 55:105] = 0
    K = np.array([[150., 0., 80.], [0., 150., 60.], [0., 0., 1.]])
    lo, hi = (-.2, -.2, .1), (.2, .2, .9)
    reference = None
    for _ in range(3):
        backend.clear()
        empty = backend.query(lo, hi)
        assert np.isposinf(empty["grid"]).all()
        assert len(empty["points"]) == 0
        for _ in range(3):
            backend.integrate_masked_depth(depth, K, np.eye(4), active)
        grid = backend.query(lo, hi)["grid"]
        assert np.isfinite(grid).any() and np.isposinf(grid).any()
        if reference is None:
            reference = grid.copy()
        else:
            np.testing.assert_array_equal(grid.view(np.uint32), reference.view(np.uint32))
