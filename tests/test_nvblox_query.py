"""Exercise ESDF conversion without claiming a CUDA/nvblox runtime test."""

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.occupancy_backends import NvbloxBackend


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
        np.testing.assert_allclose(centres[0, :3], [0, 0, 0])
        np.testing.assert_allclose(centres[-1, :3], [0.01, 0.01, 0.01])
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

    result = backend.query([0, 0, 0], [0.01, 0.01, 0.01])

    assert result["grid"].shape == (2, 2, 2)
    assert result["grid"].dtype == np.float32
    np.testing.assert_allclose(
        result["grid"].ravel(), [0, 0, 0.03, np.inf, 0.05, 0.06, 0.07, 0.08]
    )
    np.testing.assert_allclose(result["points"], [[0, 0, 0], [0, 0, 0.01]])
    assert result["voxel"] == 0.01
    np.testing.assert_array_equal(result["origin"], [0, 0, 0])
