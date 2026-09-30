"""Attached geometry must leave the world map without becoming collision-free."""

import base64
import runpy
import zlib
from pathlib import Path

import numpy as np
import pytest
from test_occupancy import FakeKin, limits

from cascade.perception.occupancy import OccupancyMap
from cascade.safety.harness import SafetyHarness
from cascade.types import Frame, SafetyViolation

HELPERS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/isaac_self_mask.py")
)
PROP = "/World_Props/green_cube"


def contact(left=1.0, right=1.0, counts=(1, 1)):
    return {
        "jaw_forces_n": [[left, 0, 0], [-right, 0, 0]],
        "jaw_contact_counts": counts,
    }


def test_attachment_requires_measured_force_on_both_jaws():
    paths = HELPERS["bilateral_contact_paths"](
        {
            "green_cube": contact(),
            "orange": contact(right=0.0),
            "pink_cube": contact(counts=(1, 0)),
            "lemon": contact(left=0.01),
        }
    )
    assert paths == [PROP]
    with pytest.raises(ValueError, match="contact evidence"):
        HELPERS["bilateral_contact_paths"]({"green_cube": contact(right=np.nan)})


def test_payload_mask_preserves_neighbor_and_distinguishes_robot_pixels():
    packet = HELPERS["encode_robot_mask"](
        np.arange(1, 5).reshape(1, 4),
        {
            "idToLabels": {
                "1": "/Robot/jaw",
                "2": PROP + "/mesh",
                "3": PROP + "_other/mesh",
                "4": "/World_Props/orange",
            }
        },
        "/Robot",
        2.0,
        contact_paths=[PROP],
        payload_tracking=True,
    )
    decode = lambda name: np.frombuffer(
        zlib.decompress(base64.b64decode(packet[name])), np.uint8
    )
    np.testing.assert_array_equal(decode("data"), [1, 1, 0, 0])
    np.testing.assert_array_equal(decode("payload_data"), [0, 1, 0, 0])


class Client:
    def __init__(self):
        self.requests = []

    def request(self, packet):
        self.requests.append(packet)
        axes = np.linspace(-1.0, 1.0, 41)
        xyz = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1)
        return {
            "points": np.array([[0.26, 0.0, 0.6]], dtype=np.float32),
            "grid": np.linalg.norm(xyz - [0.26, 0, 0.6], axis=-1),
            "origin": np.full(3, -1.0),
            "voxel": 0.05,
        }


def frame(attached=True, stamp=1.0):
    mask = np.array([[False, attached, False]])
    return Frame(
        rgb=np.zeros((1, 3, 3), np.uint8),
        depth_m=np.array([[0.5, 0.6, 0.8]], np.float32),
        K=np.diag([10.0, 10.0, 1.0]),
        robot_mask=mask,
        payload_mask=mask.copy(),
        prop_masks={PROP: np.array([[False, True, False]])},
        capture={
            "backend": "isaac", "source": ["test", 1],
            "proprioception": {"backend": "isaac", "robot_id": "/Robot",
                               "time_source": "physics_loop_monotonic", "t": stamp},
            "t": stamp,
            "camera": "cam0",
            "contact_paths": [PROP] if attached else [],
        },
    )


def mapping():
    m = OccupancyMap(
        client=Client(),
        region_min=np.full(3, -1.0),
        region_max=np.ones(3),
        depth_stride=1,
    )
    m.status = {"backend": "test"}
    m.allowed_contact_paths = {PROP}
    tcp = np.eye(4)
    tcp[2, 3] = 0.5
    m.track_payload(lambda _: tcp)
    return m


def test_attach_and_release_clear_ghosts_and_preserve_raw_depth():
    m = mapping()
    f = frame()
    m.refresh(f, np.eye(4))
    assert m.last_error is None
    assert [r["action"] for r in m._client.requests] == [
        "clear",
        "integrate_depth",
        "query",
    ]
    np.testing.assert_allclose(m._client.requests[1]["depth"], [[0.5, 0.0, 0.8]])
    np.testing.assert_allclose(f.depth_m, [[0.5, 0.6, 0.8]])
    moved_tcp = np.eye(4)
    moved_tcp[:3, 3] = [0.2, 0, 0.5]
    np.testing.assert_allclose(
        m.payload_points(moved_tcp), [[0.26, 0.0, 0.6]], atol=1e-7
    )
    m.refresh(frame(stamp=2.0), np.eye(4))
    assert sum(r["action"] == "clear" for r in m._client.requests) == 1
    m.refresh(frame(attached=False, stamp=3.0), np.eye(4))
    assert sum(r["action"] == "clear" for r in m._client.requests) == 2
    assert len(m.payload_points(moved_tcp)) == 0
    before = len(m._client.requests)
    m.refresh(frame(stamp=2.0), np.eye(4))  # delayed second camera cannot reattach
    assert len(m._client.requests) == before


def test_missing_payload_evidence_fails_closed_until_valid_refresh():
    m = mapping()
    f = frame()
    f.payload_mask = None
    m.refresh(f, np.eye(4))
    assert "payload" in m.last_error
    with pytest.raises(SafetyViolation, match="payload"):
        m.clearance(np.zeros((1, 3)))
    assert m._client.requests == []
    m.refresh(frame(stamp=2.0), np.eye(4))
    assert m.last_error is None
    assert m.clearance(np.zeros((1, 3))) is not None


def test_carried_surface_collision_rejects_motion_with_clear_tcp():
    m = mapping()
    m.refresh(frame(), np.eye(4))
    h = SafetyHarness(limits(min_clearance_m=0.03), kinematics=FakeKin(), occupancy=m)
    q = np.array([0.2, 0, 0.5, 0, 0, 0])
    assert m.clearance(q[None, :3])[0] > 0.03
    assert "attached object" in h.vet_pose(q)
    with pytest.raises(SafetyViolation, match="attached object"):
        h.approve(q, q, dt=1e9)


def test_payload_unknown_cells_cannot_become_clear_space():
    m = mapping()
    m.refresh(frame(), np.eye(4))
    m._grid[:] = np.inf
    h = SafetyHarness(limits(min_clearance_m=0.03), kinematics=FakeKin(), occupancy=m)
    q = np.array([0.2, 0, 0.5, 0, 0, 0])
    assert "unobserved" in h.vet_pose(q)
    h.allow_grasp_descent([.26, 0], radius_m=.1, z_min=.4)
    h.approve(q, q, dt=1e9)  # existing intentional-contact zone includes the payload
    h.clear_grasp_exemption()
    with pytest.raises(SafetyViolation, match="unobserved"):
        h.approve(q, q, dt=1e9)


def test_zero_weight_unknown_corner_does_not_invalidate_measured_voxel():
    m = mapping()
    m.refresh(frame(), np.eye(4))
    m._grid[:] = np.inf
    m._grid[20, 20, 20] = .1
    np.testing.assert_allclose(m.payload_clearance(np.zeros((1, 3))), [.1])


def test_replay_preserves_neighbor_and_does_not_resurrect_released_old_pose():
    m = mapping()
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
    depths = [p["depth"] for p in m._client.requests if p["action"] == "integrate_depth"]
    assert len(depths) == 3
    for d in depths:
        np.testing.assert_allclose(d, [[.5, 0., .8]])
    m._client.requests.clear()
    released = capture(False, 4.)
    released.prop_masks[PROP] = np.array([[True, False, False]])
    released.depth_m = np.array([[.7, .9, .8]], np.float32)
    m.refresh(released, np.eye(4))
    assert m.last_error is None and m.last_replayed_frames == 2
    depths = [p["depth"] for p in m._client.requests if p["action"] == "integrate_depth"]
    for old in depths[:-1]:
        assert old[0, 1] == 0  # old prop pose cannot return
        assert old[0, 2] == pytest.approx(.8)  # neighbor is preserved
    np.testing.assert_allclose(depths[-1], [[.7, .9, .8]])


def test_old_other_camera_cannot_release_newer_confirmed_attachment():
    m = mapping()
    m.refresh(frame(False, 1.), np.eye(4))
    m.refresh(frame(True, 3.), np.eye(4))
    m.refresh(frame(True, 12.), np.eye(4))
    stale = frame(False, 9.)
    stale.capture["camera"] = "side"
    before = len(m._client.requests)
    m.refresh(stale, np.eye(4))
    assert len(m._client.requests) == before
    assert m._contact_paths == (PROP,)
    assert len(m.payload_points(np.eye(4))) > 0


def test_reset_discards_history_and_rejects_pre_reset_captures():
    m = mapping()
    m.refresh(frame(True, 1.), np.eye(4))
    assert m._depth_history and m._payload_samples
    assert m.begin_scene_reset()
    assert not m._depth_history and not m._payload_samples
    assert m._grid is None and m._occupied is None
    count = len(m._client.requests)
    m.refresh(frame(True, 3.), np.eye(4))
    assert len(m._client.requests) == count
    with pytest.raises(SafetyViolation, match="post-reset"):
        m.clearance(np.zeros((1, 3)))
    m.finish_scene_reset([frame(False, 5.)])
    m.refresh(frame(True, 4.), np.eye(4))
    assert len(m._client.requests) == count
    m.refresh(frame(False, 6.), np.eye(4))
    assert m.last_error is None and m._body_error is None
    assert m._contact_paths == ()
    assert not m._payload_samples
    assert m._depth_history["cam0"][0]["t"] == 6.
