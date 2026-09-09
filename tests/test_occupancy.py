import numpy as np
import pytest
from conftest import REPO, loopback_host

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import Frame, SafetyViolation


class FakeClient:
    """Stands in for OccupancyClient without a real ZMQ socket."""

    def __init__(self, occupied=None, fail=False):
        self.occupied = occupied if occupied is not None else np.empty((0, 3))
        self.fail = fail
        self.requests: list[dict] = []

    def request(self, payload: dict) -> dict:
        self.requests.append(payload)
        if self.fail:
            raise OccupancyError("bridge down")
        if payload["action"] == "integrate":
            return {}
        if payload["action"] == "query":
            return {"points": self.occupied.astype(np.float32)}
        if payload["action"] == "probe":
            return {"ok": True, "backend": "fake", "device": "cpu", "voxel": 0.02, "esdf": False}
        # An OLD bridge (cloud protocol only): the real one answers an unknown
        # action with an error dict, which the client turns into
        # OccupancyError("... unknown action ...") and downgrades on.
        raise OccupancyError(f"occupancy bridge error: unknown action {payload['action']!r}")

    def probe(self, timeout_ms=300):
        if self.fail:
            raise OccupancyError("bridge down")
        return self.request({"action": "probe"})


def _map(occupied=None, fail=False, **kw):
    return OccupancyMap(
        client=FakeClient(occupied=occupied, fail=fail),
        region_min=np.array([-1, -1, -1]),
        region_max=np.array([1, 1, 1]),
        **kw,
    )


def _frame(depth_val=0.5, size=(8, 8)):
    h, w = size
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), depth_val, dtype=np.float32)
    K = np.array([[50.0, 0, w / 2], [0, 50.0, h / 2], [0, 0, 1]])
    return Frame(rgb=rgb, depth_m=depth, K=K)


def test_from_config_contract():
    # No `occupancy:` block at all -> off (nothing to configure a client from).
    assert OccupancyMap.from_config(None) is None
    # Explicitly disabled -> off.
    assert OccupancyMap.from_config({"enabled": False}) is None


def test_from_config_enabled_by_default(monkeypatch):
    # 2026-09-03: occupancy defaults ON. An empty block builds a map when the
    # wire deps are importable (this venv has the `grasping` extra); the
    # conftest scrub must be lifted to see the real default.
    monkeypatch.delenv("CASCADE_OCCUPANCY", raising=False)
    m = OccupancyMap.from_config({})
    assert m is not None
    # Booth rule at the next layer: no bridge running, so the cache is empty
    # and clearance() abstains rather than blocking.
    assert m.clearance(np.zeros((1, 3))) is None


def test_from_config_env_kill_switch(monkeypatch):
    monkeypatch.setenv("CASCADE_OCCUPANCY", "0")
    assert OccupancyMap.from_config({"enabled": True}) is None
    monkeypatch.setenv("CASCADE_OCCUPANCY", "1")
    assert OccupancyMap.from_config({"enabled": False}) is not None


def test_clearance_none_before_first_refresh():
    m = _map()
    assert m.clearance(np.zeros((1, 3))) is None


def test_clearance_after_refresh():
    occupied = np.array([[0.0, 0.0, 0.0]])
    m = _map(occupied=occupied)
    m.refresh(_frame(), T_base_cam=np.eye(4))
    d = m.clearance(np.array([[0.0, 0.0, 0.05], [1.0, 0.0, 0.0]]))
    assert d is not None
    assert d[0] == pytest.approx(0.05, abs=1e-6)
    assert d[1] == pytest.approx(1.0, abs=1e-6)


def test_stale_cache_is_treated_as_absent():
    m = _map(occupied=np.array([[0.0, 0.0, 0.0]]), max_age_s=0.0)
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.clearance(np.zeros((1, 3))) is None


def test_bridge_error_keeps_last_good_cache_and_records_error():
    m = _map(occupied=np.array([[0.0, 0.0, 0.0]]))
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.last_error is None
    m._client.fail = True
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.last_error is not None
    # stale cache is still readable until max_age_s elapses
    assert m.clearance(np.zeros((1, 3))) is not None


class FakeKin:
    joint_limits = (np.full(6, -3.0), np.full(6, 3.0))

    def fk(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def link_positions(self, q):
        return np.array([[0, 0, 0.2], [0, 0, 0.2], [q[0], q[1], max(q[2], 0.1)]])


def limits(**kw):
    base = dict(
        workspace_min=np.array([-1.0, -1.0, -1.0]),
        workspace_max=np.array([1.0, 1.0, 1.0]),
        table_z=-1.0,  # keep the table checks out of the way of this test
        table_clearance=0.0,
        max_joint_vel=10.0,
        watchdog_s=1e9,
        min_clearance_m=0.05,
    )
    base.update(kw)
    return SafetyLimits(**base)


def test_harness_rejects_a_waypoint_too_close_to_an_occupied_point():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    far = np.array([0.3, 0.0, 0.5, 0, 0, 0])  # tcp = q[:3]
    h.approve(far, far, dt=1e9)  # 0.3 m away, clear
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])  # 0.01 m away, too close
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(far, near, dt=1e9)


def test_harness_ignores_occupancy_without_fresh_data():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    # never refreshed -> clearance() returns None -> check is skipped
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    h.approve(near, near, dt=1e9)


def test_harness_grasp_exemption_covers_occupancy_too():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(near, near, dt=1e9)
    h.allow_grasp_descent(np.array([0.3, 0.0]), radius_m=0.1, z_min=-1.0)
    h.approve(near, near, dt=1e9)


def test_vet_pose_reports_occupancy_violation():
    occ = _map(occupied=np.array([[0.3, 0.0, 0.2]]))
    occ.refresh(_frame(), T_base_cam=np.eye(4))
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=occ)
    q = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    reason = h.vet_pose(q)
    assert reason is not None and "occupancy" in reason


# ─────────────────────────────────────────────────────────────────────────
# REAL WIRE. Everything above uses FakeClient, so it verifies cache rules
# and harness behaviour but never the ZMQ/msgpack protocol itself: a bridge
# that changed its response shape would keep every test above green.
#
# `occupancy.enabled` is ON by default and the bridge ships in this repo, so
# the failure mode is a bridge nobody started -- the same blind spot
# GraspGen-X had. These tests run the client against the actual
# scripts/serve_occupancy_bridge.py process (warp backend).
# ─────────────────────────────────────────────────────────────────────────

BRIDGE = REPO / "scripts" / "serve_occupancy_bridge.py"
BRIDGE_PORT = 5598          # not 5557: never collide with a rig bridge


def _has_wire() -> bool:
    try:
        import msgpack  # noqa: F401
        import msgpack_numpy  # noqa: F401
        import zmq  # noqa: F401

        return True
    except ImportError:
        return False


needs_wire = pytest.mark.skipif(
    not _has_wire(),
    reason="needs the grasping extra: uv pip install -e '.[grasping]'",
)


def _spawn_bridge(port: int):
    """The real occupancy bridge (warp backend), on a private port. Fails the
    test if it cannot start: a bridge that will not come up is a regression,
    not an environment to skip around (the deps are in the `grasping` extra
    plus warp-lang, both in this venv)."""
    import socket
    import subprocess
    import sys
    import time

    proc = subprocess.Popen(
        [sys.executable, str(BRIDGE), "--port", str(port),
         "--voxel-size", "0.02", "--backend", "warp",
         "--region-min", "-1.0", "-1.0", "-1.0", "--region-max", "1.0", "1.0", "1.0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 90.0   # first Warp kernel compile is slow
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            err = proc.stderr.read().decode()[-600:] if proc.stderr else ""
            pytest.fail(f"occupancy bridge exited at startup: {err}")
        s = socket.socket()
        s.settimeout(0.2)
        ok = s.connect_ex((loopback_host(), port)) == 0
        s.close()
        if ok:
            return proc
        time.sleep(0.1)
    proc.kill()
    pytest.fail("occupancy bridge did not bind in time")


@pytest.fixture(scope="module")
def bridge_server():
    """Shared bridge for the protocol tests. NOTE: legacy `integrate` stamps
    points that no ray ever carves, so tests on this fixture see each
    other's obstacles -- geometry-sensitive tests use `fresh_bridge`."""
    proc = _spawn_bridge(BRIDGE_PORT)
    yield BRIDGE_PORT
    proc.kill()
    proc.wait(timeout=5)


@pytest.fixture
def fresh_bridge():
    """A private, EMPTY bridge per test for depth-frame geometry."""
    proc = _spawn_bridge(BRIDGE_PORT + 1)
    yield BRIDGE_PORT + 1
    proc.kill()
    proc.wait(timeout=5)


def _live_map(port, **kw):
    """An OccupancyMap talking to the real bridge over a real socket."""
    from cascade.perception.occupancy import OccupancyClient

    kw.setdefault("region_min", np.array([-1.0, -1.0, -1.0]))
    kw.setdefault("region_max", np.array([1.0, 1.0, 1.0]))
    return OccupancyMap(
        client=OccupancyClient(host=loopback_host(), port=port, timeout_ms=5000),
        **kw,
    )


@needs_wire
def test_the_wire_client_builds_when_the_extra_is_installed():
    """occupancy.py raises OccupancyError if pyzmq/msgpack-numpy are missing,
    which disables the whole map. The `grasping` extra carries them (shared
    with the GraspGen-X client); this fails loudly if that regresses."""
    from cascade.perception.occupancy import OccupancyClient, OccupancyError

    try:
        OccupancyClient(port=BRIDGE_PORT)
    except OccupancyError as e:  # pragma: no cover
        pytest.fail(f"client could not be constructed: {e}")


@needs_wire
def test_integrate_then_query_round_trips_over_the_real_socket(bridge_server):
    """The protocol itself: msgpack-numpy must carry a float32 (N,3) array
    both ways. A dtype or key-name drift breaks here and nowhere else."""
    from cascade.perception.occupancy import OccupancyClient

    c = OccupancyClient(host=loopback_host(), port=bridge_server, timeout_ms=5000)
    wall = np.column_stack([
        np.full(300, 0.30), np.linspace(-0.1, 0.1, 300), np.full(300, 0.15),
    ]).astype(np.float32)
    assert c.request({"action": "integrate", "points": wall}) == {}
    resp = c.request({
        "action": "query",
        "region_min": np.array([0.0, -0.3, 0.0], dtype=np.float32),
        "region_max": np.array([0.6, 0.3, 0.4], dtype=np.float32),
    })
    pts = np.asarray(resp["points"])
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] > 0, "wall integrated but query came back empty"
    # everything the bridge returns must be the wall we put in, voxel-snapped
    assert abs(float(pts[:, 0].mean()) - 0.30) < 0.02
    # and the distance grid: zero on the wall, growing away from it
    grid = np.asarray(resp["grid"]); origin = np.asarray(resp["origin"]); v = float(resp["voxel"])
    assert grid.ndim == 3 and v == pytest.approx(0.02)
    def at(p):
        return float(grid[tuple(np.round((np.asarray(p) - origin) / v).astype(int))])
    assert at((0.30, 0.0, 0.15)) == pytest.approx(0.0, abs=1e-6)
    assert at((0.40, 0.0, 0.15)) == pytest.approx(0.10, abs=0.011)
    c.close()


@needs_wire
def test_the_query_region_actually_filters(bridge_server):
    """region_min/region_max must be honoured by the bridge, not ignored.
    If they were, the harness would receive obstacles from outside the
    workspace and refuse to move for no visible reason."""
    from cascade.perception.occupancy import OccupancyClient

    c = OccupancyClient(host=loopback_host(), port=bridge_server, timeout_ms=5000)
    far = np.array([[5.0, 5.0, 5.0]] * 10, dtype=np.float32)
    c.request({"action": "integrate", "points": far})
    resp = c.request({
        "action": "query",
        "region_min": np.array([0.0, -0.3, 0.0], dtype=np.float32),
        "region_max": np.array([0.6, 0.3, 0.4], dtype=np.float32),
    })
    pts = np.asarray(resp["points"]).reshape(-1, 3)
    assert not len(pts) or float(pts[:, 0].max()) <= 0.6 + 1e-3, (
        "bridge returned points outside the requested region"
    )
    c.close()


@needs_wire
def test_a_real_depth_frame_becomes_clearance(fresh_bridge):
    """End to end over the socket: a depth frame -> integrate -> query ->
    cached grid -> clearance numbers the harness can gate on. On a FRESH
    bridge: the shared one carries earlier tests' stamped walls."""
    m = _live_map(fresh_bridge)
    # camera 1 m up looking down; the frame's flat 0.5 m depth becomes a
    # plane of points at z = 0.5
    T = np.eye(4)
    T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    T[2, 3] = 1.0
    m.refresh(_frame(depth_val=0.5), T_base_cam=T)
    assert m.last_error is None, f"refresh failed over the wire: {m.last_error}"
    assert m._occupied is not None and len(m._occupied) > 0

    # `points` are occupied voxel centres: the surface AND the truncation
    # band just behind it. Anchor on the surface (topmost voxel), not on
    # points[0], which may sit a few voxels below the plane.
    top = m._occupied[np.argmax(m._occupied[:, 2])]
    on_plane = np.asarray(top, dtype=float).reshape(1, 3)
    d_near = m.clearance(on_plane)
    d_far = m.clearance(on_plane + np.array([0.0, 0.0, 0.4]))       # inside the mapped region
    d_out = m.clearance(on_plane + np.array([0.0, 0.0, 2.0]))       # OUTSIDE the mapped region
    assert d_near is not None and d_far is not None and d_out is not None
    assert float(d_near[0]) < 0.03, f"point on the cloud reads {d_near[0]:.3f} m away"
    assert float(d_far[0]) == pytest.approx(0.4, abs=0.03), "0.4 m above the plane should read ~0.4"
    # a point the map does not cover must read 'nothing known' (inf), NEVER a
    # clamped border distance: clamping would hand it an obstacle it has no
    # relation to -- or hide a real one standing right there.
    assert np.isinf(d_out[0])


@needs_wire
def test_the_harness_gates_on_data_that_came_over_the_wire(bridge_server):
    """The point of the whole path: an obstacle that reached the map through
    a real bridge round trip must block a waypoint. The FakeClient tests
    prove the rule; this proves the plumbing feeding it."""
    m = _live_map(bridge_server)
    obstacle = np.array([[0.30, 0.0, 0.20]] * 50, dtype=np.float32)
    m._client.request({"action": "integrate", "points": obstacle})
    m.refresh(_frame(depth_val=0.0), T_base_cam=np.eye(4))   # query-only
    assert m.last_error is None

    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=m)
    far = np.array([0.3, 0.0, 0.9, 0, 0, 0])
    h.approve(far, far, dt=1e9)                     # clear, must pass
    near = np.array([0.3, 0.0, 0.21, 0, 0, 0])      # 0.01 m from the obstacle
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(far, near, dt=1e9)


@needs_wire
def test_a_dead_bridge_degrades_instead_of_freezing_the_arm(bridge_server):
    """The booth rule. A bridge that stops answering must leave the arm
    movable: refresh records last_error, the cache ages out, and clearance
    then returns None (= skip the check) rather than blocking forever."""
    m = _live_map(5597, max_age_s=0.0)   # nothing listening on 5597
    m.refresh(_frame(), T_base_cam=np.eye(4))
    assert m.last_error is not None, "a dead bridge should record an error"
    assert m.clearance(np.zeros((1, 3))) is None, "no data must read as None"

    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=m)
    q = np.array([0.3, 0.0, 0.21, 0, 0, 0])
    h.approve(q, q, dt=1e9)   # must NOT raise


@needs_wire
def test_the_bridge_reports_a_bad_action_as_an_error(bridge_server):
    """Malformed requests must come back as OccupancyError, not a hang or a
    silently empty cloud that would read as 'nothing in the way'."""
    from cascade.perception.occupancy import OccupancyClient, OccupancyError

    c = OccupancyClient(host=loopback_host(), port=bridge_server, timeout_ms=5000)
    with pytest.raises(OccupancyError, match="unknown action"):
        c.request({"action": "definitely_not_an_action"})
    c.close()


@needs_wire
def test_probe_names_the_backend(bridge_server):
    """The startup probe is what makes 'occupancy: warp on cpu' a verified
    statement instead of a config wish."""
    m = _live_map(bridge_server)
    st = m.probe()
    assert st is not None and st["backend"] == "warp" and st["esdf"] is True and st["carving"] is True
    assert "warp" in m.describe()
    dead = _live_map(5597)
    assert dead.probe() is None and dead.probe_error and "timed out" in dead.probe_error
    assert dead.describe().startswith("none")


def _tabletop_frame(cube: bool, W=160, H=120, fx=150.0):
    """Analytic depth of a table 0.60 m below a straight-down camera, with an
    optional 5 cm cube. Camera at base (0.28, 0, 0.60)."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = np.full((H, W), 0.60, dtype=np.float32)
    if cube:
        xc = (us - W / 2) / fx * 0.55
        yc = (vs - H / 2) / fx * 0.55
        d[(np.abs(xc + 0.10) < 0.025) & (np.abs(yc) < 0.025)] = 0.55
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=float)
    return Frame(rgb=np.zeros((H, W, 3), dtype=np.uint8), depth_m=d, K=K)


_T_TOPDOWN = np.array([[0, -1, 0, 0.28], [-1, 0, 0, 0.0], [0, 0, -1, 0.60], [0, 0, 0, 1]], dtype=float)


@needs_wire
def test_depth_frames_build_a_distance_field_over_the_wire(fresh_bridge):
    """integrate_depth + query: the grid must read ~0 on the cube, its height
    above the table over empty table, and the trilinear clearance() must
    match the grid to within a voxel."""
    m = _live_map(fresh_bridge, region_min=np.array([0.0, -0.3, -0.02]),
                  region_max=np.array([0.6, 0.3, 0.4]), depth_stride=1)
    for _ in range(3):
        m.refresh(_tabletop_frame(cube=True), T_base_cam=_T_TOPDOWN)
    assert m.last_error is None, m.last_error
    assert m._grid is not None
    d = m.clearance(np.array([[0.28, 0.10, 0.03],    # inside the cube
                              [0.28, 0.10, 0.10],    # 5 cm above its top
                              [0.28, -0.10, 0.10]]))  # over empty table: 10 cm to the table
    assert d is not None
    assert d[0] < 0.015
    assert d[1] == pytest.approx(0.05, abs=0.015)
    assert d[2] == pytest.approx(0.10, abs=0.015)


@needs_wire
def test_a_removed_object_is_carved_out_of_the_map(fresh_bridge):
    """The property the old accumulator lacked: frames WITHOUT the cube must
    free the voxels it occupied, so clearance above that spot grows back."""
    m = _live_map(fresh_bridge, region_min=np.array([0.0, -0.3, -0.02]),
                  region_max=np.array([0.6, 0.3, 0.4]), depth_stride=1)
    for _ in range(3):
        m.refresh(_tabletop_frame(cube=True), T_base_cam=_T_TOPDOWN)
    before = m.clearance(np.array([[0.28, 0.10, 0.06]]))[0]
    assert before < 0.02, "cube not in the map to begin with"
    for _ in range(12):
        m.refresh(_tabletop_frame(cube=False), T_base_cam=_T_TOPDOWN)
    after = m.clearance(np.array([[0.28, 0.10, 0.06]]))[0]
    assert after == pytest.approx(0.06, abs=0.015), f"cube still in the map: clearance {after:.3f}"


@needs_wire
def test_the_harness_gates_on_a_carved_grid(fresh_bridge):
    m = _live_map(fresh_bridge, region_min=np.array([0.0, -0.3, -0.02]),
                  region_max=np.array([0.6, 0.3, 0.4]), depth_stride=1)
    for _ in range(3):
        m.refresh(_tabletop_frame(cube=True), T_base_cam=_T_TOPDOWN)
    h = SafetyHarness(limits(), kinematics=FakeKin(), occupancy=m)
    far = np.array([0.28, 0.10, 0.30, 0, 0, 0])
    h.approve(far, far, dt=1e9)
    near = np.array([0.28, 0.10, 0.07, 0, 0, 0])   # 2 cm above the cube top
    with pytest.raises(SafetyViolation, match="occupancy"):
        h.approve(far, near, dt=1e9)


# ─────────────────────────────────────────────────────────────────────────
# Backend unit tests (in-process, no socket): the Warp kernels against the
# scipy reference, and the factory contract.
# ─────────────────────────────────────────────────────────────────────────

def _has_warp() -> bool:
    try:
        import warp  # noqa: F401
        import scipy  # noqa: F401

        return True
    except ImportError:
        return False


needs_warp = pytest.mark.skipif(not _has_warp(), reason="needs warp-lang + scipy")


@needs_warp
def test_warp_edt_matches_scipy_exactly():
    from cascade.perception.occupancy_backends import edt_3d, make_backend

    b = make_backend("warp", voxel=0.01, region_min=(0.0, -0.3, -0.02), region_max=(0.6, 0.3, 0.4))
    f = _tabletop_frame(cube=True, W=640, H=480, fx=600.0)
    for _ in range(3):
        b.integrate_depth(f.depth_m, f.K, _T_TOPDOWN)
    out = b.query((0.0, -0.3, -0.02), (0.6, 0.3, 0.4))
    occ = b.occupied_mask()
    assert occ.any(), "nothing occupied: the projective integration did not land"
    ref = edt_3d(occ, 0.01)
    g = out["grid"]
    fin = np.isfinite(g) & np.isfinite(ref)
    assert fin.all()
    assert np.abs(g - ref).max() < 1e-4


@needs_warp
def test_warp_backend_carves_and_reports_timing():
    from cascade.perception.occupancy_backends import make_backend

    b = make_backend("warp", voxel=0.01, region_min=(0.0, -0.3, -0.02), region_max=(0.6, 0.3, 0.4))
    f_cube = _tabletop_frame(cube=True, W=640, H=480, fx=600.0)
    f_empty = _tabletop_frame(cube=False, W=640, H=480, fx=600.0)
    b.integrate_depth(f_cube.depth_m, f_cube.K, _T_TOPDOWN)
    pts = np.argwhere(b.occupied_mask()) * 0.01 + b.spec.origin
    assert (pts[:, 2] > 0.015).sum() > 20, "cube missing"
    for _ in range(10):
        b.integrate_depth(f_empty.depth_m, f_empty.K, _T_TOPDOWN)
    pts = np.argwhere(b.occupied_mask()) * 0.01 + b.spec.origin
    assert (pts[:, 2] > 0.015).sum() == 0, "cube not carved"
    assert b.last_integrate_ms > 0


def test_explicit_backend_never_falls_through(monkeypatch):
    """`nvblox` asked for on a machine without it must raise, not hand back
    another backend under the same name (the client displays that name)."""
    from cascade.perception import occupancy_backends as ob

    import sys
    monkeypatch.setitem(sys.modules, "nvblox_torch", None)   # force ImportError
    with pytest.raises(Exception):
        ob.make_backend("nvblox")
    with pytest.raises(ValueError, match="unknown occupancy backend"):
        ob.make_backend("open3d")
    v = ob.make_backend("voxel", voxel=0.02)
    assert v.name == "voxel" and "no carving" in v.describe()


def test_the_occupancy_extra_installs_the_warp_backend_deps():
    """The one-click launcher installs `cascade[occupancy]`; if that extra
    ever loses warp/scipy the bridge silently starts on the numpy `voxel`
    backend (no carving, no distance field) -- exactly what a fresh clone
    did before the extra existed."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    m = re.search(r"^occupancy = \[(.*?)\]", text, re.M | re.S)
    assert m, "no `occupancy` extra in pyproject.toml"
    deps = m.group(1)
    for need in ("warp-lang", "scipy", "pyzmq", "msgpack-numpy"):
        assert need in deps, f"occupancy extra lacks {need}"
    launch = (Path(__file__).resolve().parents[1] / "scripts" / "launch.sh").read_text()
    assert "occupancy,llm" in launch, "launch.sh does not install the occupancy extra"
