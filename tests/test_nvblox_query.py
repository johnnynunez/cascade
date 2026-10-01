"""Exercise ESDF conversion without claiming a CUDA/nvblox runtime test."""

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.occupancy_backends import GridSpec, NvbloxBackend


class HostTensor(np.ndarray):
    """Only replace the tensor/device boundary; query() itself is production code."""

    def cuda(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


@pytest.mark.parametrize("unknown", [-1000.0, 1000.0])
def test_nvblox_query_preserves_3d_clearance_and_unknown_policy(unknown):
    signed = np.array([-0.02, 0.0, 0.03, unknown, 0.05, 0.06, 0.07, 0.08], np.float32)
    esdf_query = object()

    def query_layer(kind, centres):
        assert kind is esdf_query
        assert centres.shape == (8, 4)
        np.testing.assert_array_equal(centres[:, 3], 0)
        np.testing.assert_allclose(centres[0, :3], [0.005, 0.005, 0.005])
        np.testing.assert_allclose(centres[-1, :3], [0.015, 0.015, 0.015])
        return signed.reshape(-1, 1).view(HostTensor)

    backend = object.__new__(NvbloxBackend)
    backend.voxel = 0.01
    backend._unknown = unknown
    backend._QueryType = SimpleNamespace(ESDF=esdf_query)
    backend._torch = SimpleNamespace(
        as_tensor=lambda a: np.asarray(a).view(HostTensor),
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    backend.mapper = SimpleNamespace(query_layer=query_layer)

    result = backend.query([0.005, 0.005, 0.005], [0.015, 0.015, 0.015])

    assert result["grid"].shape == (2, 2, 2)
    assert result["grid"].dtype == np.float32
    np.testing.assert_allclose(
        result["grid"].ravel(), [0, 0, 0.03, np.inf, 0.05, 0.06, 0.07, 0.08]
    )
    np.testing.assert_allclose(result["points"], [[0.005, 0.005, 0.005], [0.005, 0.005, 0.015]])
    assert result["voxel"] == 0.01
    np.testing.assert_allclose(result["origin"], [0.005, 0.005, 0.005])


@pytest.mark.parametrize("voxel", [0.01, 0.005])
def test_nvblox_samples_native_centres_and_brackets_requested_workspace(voxel):
    lo = np.array([0.1, -0.3, -0.01])
    hi = np.array([0.5, 0.3, 0.55])
    spec = GridSpec(lo, hi, voxel, voxel_centres=True)
    positions = spec.centres()
    # Model nvblox's containing-voxel lookup. No boundary rounding may
    # duplicate or skip a voxel, including negative world coordinates.
    indices = np.floor(positions / voxel).astype(int)
    np.testing.assert_allclose(positions, (indices + 0.5) * voxel, atol=1e-7)
    for axis in range(3):
        np.testing.assert_array_equal(np.diff(indices, axis=axis)[..., axis], 1)
    assert np.all(positions[0, 0, 0] <= lo + 1e-7)
    assert np.all(positions[-1, -1, -1] >= hi - 1e-7)
    assert np.all(lo - positions[0, 0, 0] < voxel + 1e-7)
    assert np.all(positions[-1, -1, -1] - hi < voxel + 1e-7)


@pytest.mark.parametrize("region_min", [
    [0.05, 0.0, 0.18],
    [0.051, -0.003, 0.1827],
    [-0.011, -0.019, 0.174],
])
def test_nvblox_clearance_keeps_world_registration_when_query_bounds_change(region_min):
    from cascade.perception.occupancy import OccupancyMap

    voxel = 0.01
    plane_z = 0.505
    esdf_query = object()

    def query_layer(kind, spheres):
        assert kind is esdf_query
        np.testing.assert_array_equal(spheres[:, 3], 0)
        # Emulate upstream's float32 block/containing-voxel lookup, not
        # interpolation at the requested sphere position. Eight voxels/block.
        positions = np.asarray(spheres[:, :3], dtype=np.float32)
        block_size = np.float32(voxel) * np.float32(8)
        block = np.floor(positions / block_size).astype(np.int32)
        local = ((positions - block_size * block.astype(np.float32))
                 * np.float32(1.0 / np.float32(voxel))).astype(np.int32)
        indices = block * 8 + np.minimum(local, 7)
        native_z = (indices[:, 2] + 0.5) * voxel
        # Synthetic planar ESDF: each returned value belongs to its native
        # voxel centre. The query must preserve that position in the client.
        distances = (plane_z - native_z).astype(np.float32)
        return distances.reshape(-1, 1).view(HostTensor)

    backend = object.__new__(NvbloxBackend)
    backend.voxel = voxel
    backend._unknown = -1000.0
    backend._QueryType = SimpleNamespace(ESDF=esdf_query)
    backend._torch = SimpleNamespace(
        as_tensor=lambda a: np.asarray(a).view(HostTensor),
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    backend.mapper = SimpleNamespace(query_layer=query_layer)

    lo = np.array(region_min)
    hi = lo + 0.1
    result = backend.query(lo, hi)
    cache = object.__new__(OccupancyMap)
    cache._grid = result["grid"]
    cache._grid_origin = result["origin"]
    cache._grid_voxel = result["voxel"]
    # One fixed world point and both ROI boundaries must recover the same
    # analytic field, regardless of how the requested bounds align to voxels.
    points = np.array([[0.083, 0.037, 0.211], lo, hi])
    np.testing.assert_allclose(cache._sample_grid(points), plane_z - points[:, 2],
                               rtol=0, atol=2e-7)
