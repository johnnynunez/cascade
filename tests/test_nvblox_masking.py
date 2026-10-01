"""Negotiated masked rays preserve measured free space without inventing depth."""

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.occupancy import OccupancyError
from cascade.perception.occupancy_backends import NvbloxBackend
from cascade.types import SafetyViolation
from test_occupancy_payload import PROP, frame, mapping


@pytest.mark.parametrize("status,native", [
    ({"backend": "nvblox", "masked_depth": True}, True),
    ({"backend": "nvblox"}, False),
    ({"backend": "nvblox", "masked_depth": "true"}, False),
    ({"backend": "warp", "masked_depth": True}, False),
])
def test_only_negotiated_nvblox_receives_measured_excluded_depth(status, native):
    m = mapping()
    m.status = None
    probes = []
    m._client.probe = lambda timeout_ms: probes.append(True) or status
    f = frame()
    m.refresh(f, np.eye(4))
    assert m.last_error is None and probes == [True]
    packet = next(p for p in m._client.requests if p["action"].startswith("integrate"))
    assert packet["action"] == ("integrate_masked_depth" if native else "integrate_depth")
    np.testing.assert_allclose(packet["depth"], [[.5, .6 if native else 0., .8]])
    np.testing.assert_allclose(f.depth_m, [[.5, .6, .8]])
    if native:
        assert packet["active_mask"].dtype == np.uint8
        np.testing.assert_array_equal(packet["active_mask"], [[1, 0, 1]])
    else:
        assert "active_mask" not in packet


def test_native_mask_and_intrinsics_follow_depth_subsampling():
    m = mapping()
    m.status = {"backend": "nvblox", "masked_depth": True}
    m.depth_stride = 2
    f = frame()
    f.depth_m = np.repeat(np.repeat(f.depth_m, 2, axis=0), 2, axis=1)
    f.robot_mask = np.repeat(np.repeat(f.robot_mask, 2, axis=0), 2, axis=1)
    f.payload_mask = f.robot_mask.copy()
    f.prop_masks[PROP] = f.robot_mask.copy()
    f.K[:2] *= 2
    m.refresh(f, np.eye(4))
    assert m.last_error is None
    packet = next(p for p in m._client.requests if p["action"] == "integrate_masked_depth")
    np.testing.assert_allclose(packet["depth"], [[.5, .6, .8]])
    np.testing.assert_array_equal(packet["active_mask"], [[1, 0, 1]])
    np.testing.assert_allclose(packet["K"], np.diag([10., 10., 1.]))


def test_native_history_replays_measured_rays_but_never_old_payload_surfaces():
    m = mapping()
    m.status = {"backend": "nvblox", "masked_depth": True}
    neighbor = "/World_Props/pink_cube"
    m.allowed_contact_paths.add(neighbor)

    def capture(attached, stamp):
        f = frame(attached, stamp)
        f.prop_masks[neighbor] = np.array([[False, False, True]])
        return f

    m.refresh(capture(False, 1.), np.eye(4))
    m.refresh(capture(False, 2.), np.eye(4))
    m._client.requests.clear()
    m.refresh(capture(True, 3.), np.eye(4))
    assert m.last_error is None and m.last_replayed_frames == 2
    packets = [p for p in m._client.requests if p["action"] == "integrate_masked_depth"]
    assert len(packets) == 3
    for packet in packets:
        np.testing.assert_allclose(packet["depth"], [[.5, .6, .8]])
        np.testing.assert_array_equal(packet["active_mask"], [[1, 0, 1]])

    m._client.requests.clear()
    released = capture(False, 4.)
    released.prop_masks[PROP] = np.array([[True, False, False]])
    released.depth_m = np.array([[.7, .9, .8]], np.float32)
    m.refresh(released, np.eye(4))
    assert m.last_error is None and m.last_replayed_frames == 2
    packets = [p for p in m._client.requests if p["action"] == "integrate_masked_depth"]
    for old in packets[:-1]:
        np.testing.assert_allclose(old["depth"], [[.5, .6, .8]])
        np.testing.assert_array_equal(old["active_mask"], [[1, 0, 1]])
    np.testing.assert_allclose(packets[-1]["depth"], [[.7, .9, .8]])
    np.testing.assert_array_equal(packets[-1]["active_mask"], [[1, 1, 1]])


def test_changed_bridge_rejecting_native_action_fails_closed_without_cloud_downgrade():
    m = mapping()
    m._payload_pose_fn = None  # even ordinary robot masking must fail closed
    m.status = {"backend": "nvblox", "masked_depth": True}
    packets = []

    def old_bridge(packet):
        packets.append(packet)
        raise OccupancyError("unknown action 'integrate_masked_depth'")

    m._client.request = old_bridge
    m.refresh(frame(), np.eye(4))
    assert [p["action"] for p in packets] == ["integrate_masked_depth"]
    assert "native depth masking unavailable" in m.last_error
    with pytest.raises(SafetyViolation, match="native depth masking unavailable"):
        m.clearance(np.zeros((1, 3)))


class _HostTensor(np.ndarray):
    def cuda(self):
        return self


def _backend_without_cuda():
    calls = []
    backend = object.__new__(NvbloxBackend)
    backend.masked_depth = True
    backend._torch = SimpleNamespace(
        as_tensor=lambda a: np.asarray(a).view(_HostTensor),
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    backend._Sensor = SimpleNamespace(from_camera=lambda *args: args)
    backend.mapper = SimpleNamespace(
        add_depth_frame=lambda *args, **kwargs: calls.append((args, kwargs)),
        update_esdf=lambda: None,
    )
    return backend, calls


@pytest.mark.parametrize("mask", [
    np.ones((2, 2), np.float32),
    np.ones((2, 2), np.int64),
    np.ones((2, 3), np.uint8),
    np.ones((4,), np.uint8),
    np.full((2, 2), 2, np.uint8),
])
def test_bad_native_mask_is_rejected_before_any_integration(mask):
    backend, calls = _backend_without_cuda()
    with pytest.raises(ValueError, match="binary bool/uint8 image matching depth shape"):
        backend.integrate_masked_depth(np.full((2, 2), .6), np.eye(3), np.eye(4), mask)
    assert calls == []


@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_native_backend_passes_binary_uint8_mask_without_modifying_depth(dtype):
    backend, calls = _backend_without_cuda()
    depth = np.array([[.4, .8], [.4, .8]], np.float32)
    active = np.array([[0, 1], [0, 1]], dtype=dtype)
    backend.integrate_masked_depth(depth, np.eye(3), np.eye(4), active)
    assert len(calls) == 1
    args, kwargs = calls[0]
    np.testing.assert_array_equal(args[0], depth)
    np.testing.assert_array_equal(kwargs["mask_frame"], active)
    assert kwargs["mask_frame"].dtype == np.uint8


@pytest.mark.hardware
def test_cuda_native_mask_observes_front_only_and_preserves_neighbor():
    """Run explicitly with -m hardware in the nvblox CUDA environment."""
    pytest.importorskip("nvblox_torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("native masking requires CUDA")
    depth = np.full((120, 160), .8, np.float32)
    depth[35:85, 55:105] = .4
    active = np.ones(depth.shape, np.uint8)
    active[35:85, 55:105] = 0
    K = np.array([[150., 0., 80.], [0., 150., 60.], [0., 0., 1.]])
    points = np.array([[.005, .005, .205], [.005, .005, .405],
                       [.005, .005, .605], [.155, .005, .705], [.155, .005, .805]])
    results = []
    for native in (False, True):
        backend = NvbloxBackend(voxel=.01)
        for _ in range(3):
            if native:
                backend.integrate_masked_depth(depth, K, np.eye(4), active)
            else:
                backend.integrate_depth(np.where(active, depth, 0.), K, np.eye(4))
        query = backend.query(points.min(axis=0), points.max(axis=0))
        indices = np.rint((points-query["origin"])/query["voxel"]).astype(int)
        results.append(query["grid"][tuple(indices.T)])
    zeroed, masked = results
    assert np.isinf(zeroed[:3]).all()
    assert np.isfinite(masked[0]) and masked[0] > .1  # actual front free space
    assert np.isinf(masked[1:3]).all()  # surface and behind stay unobserved
    assert np.isfinite(masked[3:]).all()
    np.testing.assert_allclose(masked[3:], zeroed[3:], atol=1e-6)
    assert masked[4] <= .02  # the neighboring measured surface stays occupied
