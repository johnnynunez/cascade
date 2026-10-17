"""A bounded map snapshot on every clearance refusal or halt (spec §3.3).

Whenever the TCP/joint, attached-payload or tool-volume map gate returns a
reason, the harness files one report: phase, gate, joints, TCP pose, the
offending sample, its clearance, the margin, the exemption in force, the map's
state and up to `max_voxels` occupied voxel centres near the sample. Reports
are built only after a gate refused (an admitted 50 Hz step never builds one),
deduplicated per drain, bounded to the newest `max_reports`, and a failing
snapshot never changes the refusal. The skill runtime drains them into each
call's trace row (`context.clearance`).

Offline: CartesianKin / case() (tests/test_home_routes.py), the reBot model,
and a real OccupancyMap over the synthetic GridBridge.
"""
import threading

import numpy as np
import pytest
from conftest import needs_pin
from occupancy_grid_bridge import GridBridge, centred_box, depth_frame, make_map
from test_harness_tool_volume import T_CAM, side_grasp_geometry
from test_home_routes import case

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation, SkillError

REPORT_KEYS = {"phase", "gate", "reason", "q", "q_prev", "tcp", "sample", "sample_tcp", "clearance_m",
               "margin_m", "exemption", "map", "voxels"}
#: CartesianKin hovers its TCP at (0.3, 0, 0.3); this box's top is 2 cm under it
UNDER = centred_box([.3, 0., .26], [.04, .04, .04])


def live_map(*boxes, voxel=.01, **kw):
    bridge = GridBridge(list(boxes), voxel=voxel)
    occ = make_map(bridge, unobserved="free", **kw)
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    occ.bridge = bridge
    return occ


def spy(monkeypatch, occ):
    calls, original = [], occ.nearby_occupied

    def recording(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(occ, "nearby_occupied", recording)
    return calls


# ── one report per refusal, per gate (reBot model + GridBridge) ─────────────


@pytest.fixture(scope="module")
def rebot():
    return side_grasp_geometry()


def _gate_scene(rebot, gate):
    """(obstacle box, payload samples in the TCP frame or None, a TCP shift
    away from the box) that make exactly `gate` refuse the reBot's home pose
    (TCP forward, camera up)."""
    T = rebot.kin.fk(rebot.home)
    tcp = T[:3, 3]
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)          # TCP axes = base axes
    if gate == "tcp":                     # 2.5 cm ahead of the fingertips
        return centred_box(tcp + [.035, 0., 0.], [.02, .02, .02]), None, [-.01, 0., 0.]
    if gate == "link 2":                  # 2.5 cm above joint origin links[1]
        return centred_box(rebot.kin.link_positions(rebot.home)[1] + [0., 0., .035], [.02, .02, .02]), None, [-.01, 0., 0.]
    if gate == "wrist camera":            # 1.5 cm above the back of the camera box
        return centred_box(tcp + [-.12, 0., .08 + .025], [.02, .02, .02]), None, [0., 0., -.01]
    if gate == "jaws":                    # 1 cm under one open jaw, 3.7 cm from the TCP
        return centred_box(tcp + [-.02, .04, -.03], [.02, .02, .02]), None, [0., 0., .01]
    if gate == "attached":                # a held sample 10 cm under the TCP, 2 cm above a box
        return centred_box(tcp + [0., 0., -.13], [.02, .02, .02]), np.array([[0., 0., -.1]]), [0., 0., .01]
    raise AssertionError(gate)


def _rebot_harness(rebot, box, payload=None):
    occ = live_map(box)
    if payload is not None:
        occ._payload_samples["scene"] = payload
    limits = SafetyLimits.from_config(rebot.cfg.safety)
    return SafetyHarness(limits, kinematics=rebot.kin, occupancy=occ, tool_volume=rebot.volume), occ


@needs_pin
@pytest.mark.parametrize("gate", ["tcp", "link 2", "attached", "wrist camera", "jaws"])
def test_each_gate_files_exactly_one_report_per_refusal(rebot, gate):
    box, payload, away = _gate_scene(rebot, gate)
    T_prev = rebot.kin.fk(rebot.home)
    T_prev[:3, 3] += away
    ik = rebot.kin.ik(T_prev, rebot.home)
    assert ik.success
    q_prev = np.asarray(ik.q, dtype=float)                 # 1 cm farther from the box
    h, _ = _rebot_harness(rebot, box, payload)
    reason = h.vet_pose(rebot.home)
    expected_start = {"tcp": "point 0 clearance", "link 2": "point 1 clearance",
                      "attached": "attached object point", "wrist camera": "wrist camera clearance",
                      "jaws": "jaws clearance"}[gate]
    assert reason is not None and reason.startswith(expected_start), reason
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == gate and r["reason"] == reason and r["phase"] == "vet_pose"
    assert r["margin_m"] == pytest.approx(rebot.cfg.safety.min_clearance_m) == .03
    assert 0 <= r["clearance_m"] < .03 and r["voxels"]
    # the offending sample is the one the reason names, and its voxels are the box's
    assert f"[{r['sample'][0]:.4f}, {r['sample'][1]:.4f}, {r['sample'][2]:.4f}]" in reason
    lo, hi = np.asarray(box[0]) - .006, np.asarray(box[1]) + .006
    assert all(np.all((lo <= v[:3]) & (v[:3] <= hi)) for v in np.asarray(r["voxels"]))
    assert h.drain_clearance_reports() == []
    # the streamed gate files its own report for a step closing in on it
    h.heartbeat()
    with pytest.raises(SafetyViolation, match=expected_start):
        h.approve(q_prev, rebot.home, 1.0)
    (r2,) = h.drain_clearance_reports()
    assert r2["phase"] == "approve" and r2["gate"] == gate and r2["q_prev"] == q_prev.tolist()


@needs_pin
def test_sample_tcp_is_the_sample_in_the_tcp_frame(rebot):
    """At the side grasp's tilted pregrasp, a joint origin's sample in the
    TCP frame is the inverse TCP pose applied to it (not a base offset)."""
    q = rebot.q_pre
    link = rebot.kin.link_positions(q)[1]
    h, _ = _rebot_harness(rebot, centred_box(link + [0., 0., .035], [.02, .02, .02]))
    assert h.vet_pose(q).startswith("point 1 clearance")
    (r,) = h.drain_clearance_reports()
    T = rebot.kin.fk(q)
    assert not np.allclose(T[:3, :3], np.eye(3), atol=.1)
    np.testing.assert_allclose(r["sample"], link, atol=1e-9)
    np.testing.assert_allclose(r["sample_tcp"], (np.linalg.inv(T) @ [*link, 1.])[:3], atol=1e-9)
    np.testing.assert_allclose(r["tcp"], T, atol=1e-12)


@needs_pin
def test_a_clear_rebot_pose_files_nothing(rebot):
    h, occ = _rebot_harness(rebot, centred_box([.0, .4, .3], [.02, .02, .02]))
    assert h.vet_pose(rebot.home) is None and h.vet_step(rebot.home, rebot.home, .02) is None
    assert h.drain_clearance_reports() == []


# ── report contents and bounds ───────────────────────────────────────────────


def test_report_fields_and_bounds():
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    reason = h.vet_step(q_prev, q, .02)
    assert reason.startswith("point 0 clearance 0.0")
    (r,) = h.drain_clearance_reports()
    assert set(r) == REPORT_KEYS
    assert r["phase"] == "vet_step" and r["gate"] == "tcp" and r["reason"] == reason
    assert r["q"] == q.tolist() and r["q_prev"] == q_prev.tolist()
    assert np.allclose(r["tcp"], np.eye(4) + np.pad(q[:, None], ((0, 1), (3, 0))))
    assert r["sample"] == pytest.approx(q.tolist()) and r["sample_tcp"] == pytest.approx([0., 0., 0.])
    assert r["clearance_m"] == pytest.approx(.02, abs=1e-3) and r["margin_m"] == .03
    assert r["exemption"] is None
    assert r["map"] == {"refreshes": 1, "age_s": r["map"]["age_s"], "voxel_m": .01,
                        "occupied": len(occ._occupied)}
    assert 0 <= r["map"]["age_s"] < 5
    assert 0 < len(r["voxels"]) <= 64 and len(r["voxels"]) == 64          # the box holds > 64 near voxels
    assert all(row[3] <= .05 for row in r["voxels"]) and r["voxels"] == sorted(r["voxels"], key=lambda v: v[3])
    assert all(round(v, 3) == v for row in r["voxels"] for v in row)
    nearest = r["voxels"][0]
    assert nearest[3] == pytest.approx(np.linalg.norm(np.subtract(nearest[:3], q)), abs=1e-3)


def test_the_exemption_in_force_is_recorded():
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q = np.array([.3, 0., .3])
    assert h.vet_pose(q, exempt_xy=[.1, .2], exempt_radius_m=.05, exempt_z_min=.01)
    assert h.drain_clearance_reports()[0]["exemption"] == {"cylinder": [.1, .2, .05, .01]}
    with h.target_column_exemption(([.12, .2], .04, .02), z_max=.4):
        with pytest.raises(SafetyViolation):
            h.approve(q, q - [0., 0., .001], .02)
    assert h.drain_clearance_reports()[0]["exemption"] == {"cylinder": [.12, .2, .04, .02, .4]}


@pytest.mark.parametrize("phase", ["approve", "vet_step", "vet_pose", "route_end"])
def test_each_phase_names_itself(phase):
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .3]), np.array([.3, 0., .299])
    if phase == "approve":
        with pytest.raises(SafetyViolation, match="point 0 clearance"):
            h.approve(q_prev, q, .02)
    elif phase == "vet_step":
        assert h.vet_step(q_prev, q, .02)
    elif phase == "vet_pose":
        assert h.vet_pose(q)
    else:
        assert h.route_end_violation(q)
    (r,) = h.drain_clearance_reports()
    assert r["phase"] == phase and r["q"] == q.tolist()
    assert r["q_prev"] == (q_prev.tolist() if phase in ("approve", "vet_step") else None)


def test_reports_dedupe_and_keep_the_newest_32(monkeypatch):
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]))            # flat top at z 0.28
    _, _, h = case(occupancy=occ)
    calls = spy(monkeypatch, occ)
    q = np.array([.3, 0., .3])
    for _ in range(10):
        assert h.vet_pose(q)
        with pytest.raises(SafetyViolation):
            h.approve(q + [0., 0., .001], q, .02)                           # closing in by 1 mm
    assert len(h.drain_clearance_reports()) == 1                            # 20 refusals, one sample
    assert len(calls) == 1                                                  # and one snapshot built
    # the same sample on a refreshed map is new evidence
    assert h.vet_pose(q)
    occ.refresh(depth_frame(), T_CAM)
    assert h.vet_pose(q)
    assert [r["map"]["refreshes"] for r in h.drain_clearance_reports()] == [1, 2]
    # 40 distinct samples (2 cm apart): the newest 32 are kept, in order
    goals = [np.array([.12 + .02 * (k % 20), .05 * (k // 20), .3]) for k in range(40)]
    for g in goals:
        assert h.vet_pose(g)
    reports = h.drain_clearance_reports()
    np.testing.assert_allclose([r["sample"] for r in reports], [g.tolist() for g in goals[8:]])
    # the refusal that ends a call is always kept: a repeat of an evicted
    # sample is filed again, a repeat of a kept one is already there
    for g in goals:
        assert h.vet_pose(g)
    assert h.vet_pose(goals[0]) and h.vet_pose(goals[-1])
    reports = h.drain_clearance_reports()
    assert len(reports) == 32 and reports[-1]["sample"] == pytest.approx(goals[0].tolist())
    assert any(r["sample"] == pytest.approx(goals[-1].tolist()) for r in reports)


def test_the_map_config_bounds_the_reports():
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]),
                   refusal_snapshot={"radius_m": .02, "max_voxels": 8, "max_reports": 4})
    _, _, h = case(occupancy=occ)
    for k in range(10):
        assert h.vet_pose(np.array([.12 + .02 * k, 0., .3]))
    reports = h.drain_clearance_reports()
    assert len(reports) == 4
    assert all(0 < len(r["voxels"]) <= 8 and all(v[3] <= .02 for v in r["voxels"]) for r in reports)


# ── cost on the motion path (Review Focus 4) and failure isolation ─────────


def test_admitted_stream_never_builds_a_snapshot(monkeypatch):
    occ = live_map(centred_box([.3, .1, .3], [.04, .04, .04]))               # its face at y = 0.08
    safe, raw, h = case(occupancy=occ)
    calls = spy(monkeypatch, occ)
    assert safe.move_planned([.3, .045, .3]) and calls == []                 # streams 3.5 cm from it
    assert len(raw.commands) > 10 and h.drain_clearance_reports() == []
    with pytest.raises(SkillError):
        safe.move_planned([.3, .06, .3])                                     # would end 2 cm from it
    reports = h.drain_clearance_reports()
    assert 1 <= len(calls) == len(reports)
    assert {r["phase"] for r in reports} <= {"vet_step", "route_end"}


def test_a_halt_mid_stream_files_one_approve_report(monkeypatch):
    occ = live_map(centred_box([-.2, -.4, .5], [.02, .02, .02]))             # far from the route
    safe, raw, h = case(occupancy=occ)
    calls, approve = spy(monkeypatch, occ), h.approve
    seen = []

    def approve_then_appear(q_prev, q_next, dt, *a, **k):
        seen.append(1)
        if len(seen) == 20:      # an obstacle appears 2 cm ahead of the moving TCP
            occ.bridge.boxes.append(centred_box(np.asarray(q_next) + [0., .03, 0.], [.02, .02, .02]))
            occ.refresh(depth_frame(), T_CAM)
        return approve(q_prev, q_next, dt, *a, **k)

    monkeypatch.setattr(h, "approve", approve_then_appear)
    with pytest.raises((SkillError, SafetyViolation)):
        safe.move_planned([.3, .2, .3])
    (r,) = h.drain_clearance_reports()
    assert r["phase"] == "approve" and r["gate"] == "tcp" and r["map"]["refreshes"] == 2
    assert len(calls) == 1 and r["voxels"]


@pytest.mark.parametrize("error", [RuntimeError, SafetyViolation])
def test_snapshot_failure_keeps_the_refusal(monkeypatch, error):
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    expected = {"vet_step": h.vet_step(q_prev, q, .02), "vet_pose": h.vet_pose(q)}
    assert all(expected.values())
    h.drain_clearance_reports()

    def broken(*_a, **_k):
        raise error("snapshot exploded")

    monkeypatch.setattr(occ, "nearby_occupied", broken)
    assert h.vet_step(q_prev, q, .02) == expected["vet_step"]
    (r,) = h.drain_clearance_reports()
    assert "snapshot exploded" in r["snapshot_error"] and r["voxels"] == [] and r["gate"] == "tcp"
    assert set(r) == REPORT_KEYS | {"snapshot_error"}
    assert h.vet_pose(q) == expected["vet_pose"]
    assert "snapshot_error" in h.drain_clearance_reports()[0]
    with pytest.raises(SafetyViolation) as exc:
        h.approve(q_prev, q, .02)
    assert str(exc.value) == expected["vet_step"]
    assert "snapshot_error" in h.drain_clearance_reports()[0]


def test_a_broken_report_never_changes_the_refusal():
    from test_clearance_escape import VOXEL, PointsMap

    class BrokenMap(PointsMap):
        @property
        def refreshes(self):                  # read before any snapshot work
            raise RuntimeError("map bookkeeping exploded")

    q = np.array([.3, 0., .3])
    _, _, good = case(occupancy=PointsMap(VOXEL))
    expected = good.vet_step(q, q - [0., 0., .001], .02)
    _, _, h = case(occupancy=BrokenMap(VOXEL))
    assert expected and h.vet_step(q, q - [0., 0., .001], .02) == expected
    assert h.vet_pose(q) == good.vet_pose(q)
    with pytest.raises(SafetyViolation) as exc:
        h.approve(q, q - [0., 0., .001], .02)
    assert str(exc.value) == expected


def test_refusals_without_a_measured_clearance():
    from types import SimpleNamespace

    q = np.array([.3, 0., .3])
    unobserved = SimpleNamespace(clearance=lambda p: np.full(len(p), np.nan), required=True, unknown_blocks=True)
    _, _, h = case(occupancy=unobserved)
    reason = h.vet_pose(q)
    assert reason.startswith("point 0 has unobserved clearance")
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "tcp" and r["clearance_m"] is None and r["sample"] == q.tolist()
    # no answer at all from a map that must give one: no sample to name
    unavailable = SimpleNamespace(clearance=lambda p: None, required=True, unknown_blocks=True)
    _, _, h = case(occupancy=unavailable)
    reason = h.vet_pose(q)
    assert reason.startswith("clearance unavailable")
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "arm" and r["sample"] is None and r["clearance_m"] is None
    assert r["voxels"] == [] and "snapshot_error" not in r and r["reason"] == reason


def test_a_map_without_nearby_occupied_still_refuses_and_reports():
    from test_clearance_escape import VOXEL, points_map

    _, _, h = case(occupancy=points_map(VOXEL))
    q = np.array([.3, 0., .3])
    reason = h.vet_step(q, q - [0., 0., .001], .02)
    assert reason.startswith("point 0 clearance")
    (r,) = h.drain_clearance_reports()
    assert r["reason"] == reason and r["voxels"] == [] and "nearby_occupied" in r["snapshot_error"]


# ── the map side: nearby_occupied and refusal_snapshot ─────────────────────


def test_nearby_occupied_is_the_gate_cache_sorted_and_bounded():
    occ = live_map(UNDER)
    p = np.array([.3, 0., .3])
    rows = occ.nearby_occupied(p, .05, 10)
    assert rows.shape == (10, 4)
    assert np.all(np.diff(rows[:, 3]) >= 0) and np.all(rows[:, 3] <= .05)
    np.testing.assert_allclose(rows[:, 3], np.linalg.norm(rows[:, :3] - p, axis=1))
    cache = {tuple(np.round(v, 6)) for v in np.asarray(occ._occupied, dtype=float)}
    assert all(tuple(np.round(v, 6)) in cache for v in rows[:, :3])
    everything = occ.nearby_occupied(p, .05, 10_000)
    d_all = np.linalg.norm(np.asarray(occ._occupied, dtype=float) - p, axis=1)
    assert len(everything) == int((d_all <= .05).sum()) > 10
    assert occ.nearby_occupied(p, .005, 10).shape == (0, 4)
    assert make_map(GridBridge([UNDER])).nearby_occupied(p, .05, 10).shape == (0, 4)   # no cache yet


def test_nearby_occupied_reads_under_the_refresh_lock():
    occ = live_map(UNDER)
    held, done = threading.Event(), threading.Event()
    release = threading.Event()

    def hold():
        with occ._refresh_lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert held.wait(5)
    reader = threading.Thread(target=lambda: (occ.nearby_occupied([.3, 0., .3], .05, 4), done.set()))
    reader.start()
    try:
        assert not done.wait(.2)          # waits for the commit in progress
    finally:
        release.set()
        holder.join(5)
    assert done.wait(5)
    reader.join(5)


def test_refusal_snapshot_defaults_and_bad_values(monkeypatch):
    from cascade.config import load_demo_config
    from cascade.perception.occupancy import RefusalSnapshot

    assert OccupancyMap().refusal_snapshot == RefusalSnapshot(.05, 64, 32) == RefusalSnapshot()
    cfg = load_demo_config(camera="mock", arm="mock", llm="mock")
    assert cfg.occupancy.refusal_snapshot.as_dict() == {"radius_m": .05, "max_voxels": 64, "max_reports": 32}
    custom = OccupancyMap(refusal_snapshot={"radius_m": .02, "max_voxels": 8, "max_reports": 4})
    assert custom.refusal_snapshot == RefusalSnapshot(.02, 8, 4)
    assert OccupancyMap(refusal_snapshot={"max_reports": 5}).refusal_snapshot == RefusalSnapshot(.05, 64, 5)
    for bad in ({"radius_m": 0}, {"radius_m": -.01}, {"radius_m": float("nan")}, {"radius_m": "5cm"},
                {"radius_m": 1.0}, {"max_voxels": 0}, {"max_voxels": 2.5}, {"max_voxels": True},
                {"max_voxels": 100_000}, {"max_reports": 0}, {"max_reports": 1.5}, {"max_reports": 10_000},
                {"radius": .05}, "5cm", [.05, 64, 32]):
        with pytest.raises(OccupancyError, match="refusal_snapshot"):
            OccupancyMap(refusal_snapshot=bad)
    # from_config hands the block over (a fake client: nothing is contacted)
    import cascade.perception.occupancy as occupancy_mod

    monkeypatch.setenv("CASCADE_OCCUPANCY", "1")
    monkeypatch.setattr(occupancy_mod, "OccupancyClient", lambda **_kw: GridBridge([]))
    built = OccupancyMap.from_config(cfg.occupancy)
    assert built is not None and built.refusal_snapshot == RefusalSnapshot(.05, 64, 32)
    block = cfg.occupancy.as_dict()
    block["refusal_snapshot"] = {"radius_m": .03, "max_voxels": 16, "max_reports": 8}
    from cascade.config import Cfg

    assert OccupancyMap.from_config(Cfg(block)).refusal_snapshot == RefusalSnapshot(.03, 16, 8)
    block["refusal_snapshot"] = {"max_reports": -1}
    with pytest.raises(OccupancyError, match="refusal_snapshot"):
        OccupancyMap.from_config(Cfg(block))


# ── the runtime drains each call's reports into its trace row ──────────────


@needs_pin
def test_runtime_drains_reports_into_the_trace(demo_cfg, tmp_path):
    from cascade.apps.demo import build_runtime, shutdown_runtime

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        h = runtime.arm.harness
        q_home = np.asarray(runtime.arm.get_state().q, dtype=float)
        tcp = h.kin.fk(q_home)[:3, 3]
        h.occupancy = live_map(centred_box(tcp + [0., 0., -.08], [.04, .04, .02]))   # its top 7 cm under the TCP
        n0 = len(arm.commands)
        r = runtime.execute("move_relative", {"direction": "down", "distance_m": .06})
        assert r["ok"] is False and len(arm.commands) == n0
        row = runtime.trace.rows()[-1]
        assert row["skill"] == "move_relative"
        reports = row["context"]["clearance"]
        assert reports[0]["gate"] == "tcp" and all(set(rep) >= REPORT_KEYS for rep in reports)
        assert all(rep["voxels"] for rep in reports)
        # a refusal filed outside any call is stale: the next call drains it first
        assert h.vet_pose(np.asarray(reports[0]["q"], dtype=float))
        r = runtime.execute("move_relative", {"direction": "up", "distance_m": .02})
        assert r["ok"] is True, r
        assert "clearance" not in runtime.trace.rows()[-1]["context"]
    finally:
        shutdown_runtime(runtime, arm)
