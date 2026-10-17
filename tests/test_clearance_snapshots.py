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
import re
import threading
import time

import numpy as np
import pytest
from conftest import needs_pin, needs_pin_so101
from occupancy_grid_bridge import GridBridge, centred_box, depth_frame, make_map
from test_harness_tool_volume import T_CAM, side_grasp_geometry
from test_home_routes import case

from cascade.perception.occupancy import OccupancyError, OccupancyMap
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation, SkillError

REPORT_KEYS = {"phase", "gate", "reason", "q", "q_prev", "tcp", "sample", "sample_tcp", "clearance_m",
               "margin_m", "exemption", "map", "voxels", "gate_refreshes", "gate_cache_clears", "snapshot"}
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
    assert r["map"] == {"refreshes": 1, "cache_clears": 0, "age_s": r["map"]["age_s"], "voxel_m": .01,
                        "occupied": len(occ._occupied)}
    assert r["gate_refreshes"] == 1 and r["gate_cache_clears"] == 0
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
    # sample is filed again, a repeat of a kept one moves its report last
    for g in goals:
        assert h.vet_pose(g)
    assert h.vet_pose(goals[0]) and h.vet_pose(goals[-1])
    reports = h.drain_clearance_reports()
    assert len(reports) == 32 and reports[-1]["sample"] == pytest.approx(goals[-1].tolist())
    assert reports[-2]["sample"] == pytest.approx(goals[0].tolist())
    assert sum(r["sample"] == pytest.approx(goals[-1].tolist()) for r in reports) == 1


def test_the_map_config_bounds_the_reports():
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]),
                   refusal_snapshot={"radius_m": .02, "max_voxels": 8, "max_reports": 4})
    _, _, h = case(occupancy=occ)
    for k in range(10):
        assert h.vet_pose(np.array([.12 + .02 * k, 0., .3]))
    reports = h.drain_clearance_reports()
    assert len(reports) == 4
    assert all(0 < len(r["voxels"]) <= 8 and all(v[3] <= .02 for v in r["voxels"]) for r in reports)


def test_every_report_names_its_snapshot_query_and_bounds():
    """The offline replay certifies a snapshot only within its bounds around
    the point it was taken at (A6 fix ruling 8): every report records that
    query centre and the bounds it used, and a streamed halt that takes a
    report over keeps the FIRST centre (the one its voxels were measured
    from) while it rewrites the sample."""
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]),            # flat top at z 0.28
                   refusal_snapshot={"radius_m": .03, "max_voxels": 16, "max_reports": 8})
    _, _, h = case(occupancy=occ)
    q = np.array([.3, 0., .3])
    assert h.vet_pose(q)
    (r,) = h.drain_clearance_reports()
    assert r["snapshot"] == {"centre": r["sample"], "radius_m": .03, "max_voxels": 16}
    assert r["sample"] == pytest.approx(q.tolist())
    assert h.vet_pose(q)
    halt_q = q + [.002, 0., 0.]                              # another pose in the same 1 cm cell
    _, exc = _refuse(h, "approve", halt_q + [0., 0., .001], halt_q)
    assert exc is not None
    (kept,) = h.drain_clearance_reports()
    assert kept["phase"] == "approve" and kept["sample"] == pytest.approx(halt_q.tolist())
    assert kept["snapshot"] == {"centre": pytest.approx(q.tolist()), "radius_m": .03, "max_voxels": 16}
    assert all(np.linalg.norm(np.subtract(v[:3], kept["snapshot"]["centre"])) == pytest.approx(v[3], abs=1e-3)
               for v in kept["voxels"])
    # the shipped bounds by default; a report with no sample has no centre
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    occ._last_refresh = time.monotonic() - occ.max_age_s - 100.0
    with pytest.raises(SafetyViolation, match="stale"):
        h.vet_pose(q)
    (raised,) = h.drain_clearance_reports()
    assert raised["raised"] == "SafetyViolation"
    assert raised["snapshot"] == {"centre": None, "radius_m": .05, "max_voxels": 64}


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
    record = {"refreshes": 4, "age_s": .1, "voxel_m": .01, "occupied": 7}
    unavailable = SimpleNamespace(clearance=lambda p: None, required=True, unknown_blocks=True,
                                  refreshes=4, clearance_record=lambda: dict(record))
    _, _, h = case(occupancy=unavailable)
    reason = h.vet_pose(q)
    assert reason.startswith("clearance unavailable")
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "arm" and r["sample"] is None and r["clearance_m"] is None
    assert r["voxels"] == [] and "snapshot_error" not in r and r["reason"] == reason
    assert r["map"] == record and "raised" not in r


def test_a_mid_step_refusal_reports_its_interior_sample():
    from test_clearance_escape import points_map

    voxel = np.array([.3, 0., .25])            # 5 mm above it -> 6 mm below it in one step
    _, _, h = case(occupancy=points_map(voxel))
    q_prev, q = voxel + [0., 0., .005], voxel - [0., 0., .006]
    reason = h.vet_step(q_prev, q, .02)
    assert "mid-step" in reason
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "tcp" and r["clearance_m"] == pytest.approx(.00225)
    assert f"[{r['sample'][0]:.4f}, {r['sample'][1]:.4f}, {r['sample'][2]:.4f}]" in reason
    assert q[2] < r["sample"][2] < q_prev[2]               # the interior sample, not an end


@needs_pin
def test_a_tool_volume_without_an_answer_names_its_set(rebot):
    from types import SimpleNamespace

    occupancy = SimpleNamespace(clearance=lambda p: np.full(len(p), 1.), clearance_excluding=lambda p, **_k: None,
                                required=True, unknown_blocks=True)
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin, occupancy=occupancy,
                      tool_volume=rebot.volume)
    reason = h.vet_pose(rebot.home)
    assert reason.startswith("tool volume clearance unavailable")
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "tool volume" and r["sample"] is None and r["reason"] == reason


def test_the_map_record_and_its_voxels_are_one_commit(monkeypatch):
    """A refresh that tries to commit while the report is built waits for
    it: the report's refresh count and its voxels belong to the same map."""
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    original = occ.nearby_occupied
    occ.bridge.boxes[:] = [centred_box([.3, .2, .26], [.04, .04, .04])]     # the next map moves it away
    refresher = []

    def racing(*args, **kwargs):
        if not refresher:
            refresher.append(threading.Thread(target=occ.refresh, args=(depth_frame(), T_CAM)))
            refresher[0].start()
            refresher[0].join(1.0)                # commits now unless the report holds the map
        return original(*args, **kwargs)

    monkeypatch.setattr(occ, "nearby_occupied", racing)
    assert h.vet_pose(np.array([.3, 0., .3]))
    refresher[0].join(10)
    (r,) = h.drain_clearance_reports()
    assert occ.refreshes == 2 and r["map"]["refreshes"] == 1
    assert r["voxels"] and all(abs(v[1]) < .05 for v in r["voxels"])        # the first map's box


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


# ── a required map that raises, a newer map, dedupe order, frames ──────────


def _age_free(text):
    """A refusal text with a stale map's age (it grows between two reads) normalised."""
    return re.sub(r"\(\d+\.\d+ s old", "(N s old", text)


def _break_map(occ, fault):
    """Make a required map refuse by RAISING out of clearance() (spec §7 A2
    counts these refusals too)."""
    if fault == "stale":
        occ._last_refresh = time.monotonic() - occ.max_age_s - 7200.0
    elif fault == "body mask":
        occ._body_error = "robot body pose not yet validated by a masked depth refresh"
    else:                                                    # a failed refresh
        occ.bridge.fail = "bridge timed out"
        occ.refresh(depth_frame(), T_CAM)
        assert occ.last_error


def _raises_of(monkeypatch, occ, name):
    """Every SafetyViolation (the object) map method `name` raises."""
    raised, original = [], getattr(occ, name)

    def recording(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except SafetyViolation as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(occ, name, recording)
    return raised


def _refuse(h, phase, q_prev, q):
    """(refusal text, the SafetyViolation raised or None) of one gate run."""
    h.heartbeat()
    try:
        if phase == "vet_step":
            return h.vet_step(q_prev, q, .02), None
        if phase == "vet_pose":
            return h.vet_pose(q), None
        h.approve(q_prev, q, .02)
    except SafetyViolation as exc:
        return str(exc), exc
    return None, None


FAR = centred_box([.0, .4, .3], [.02, .02, .02])        # nowhere near any pose below


@pytest.mark.parametrize("fault", ["stale", "body mask", "failed refresh"])
@pytest.mark.parametrize("phase", ["vet_pose", "vet_step", "approve"])
def test_a_required_map_that_raises_files_one_report(monkeypatch, fault, phase):
    """A stale map, a body-mask fault or a failed refresh refuses by raising:
    the same exception object leaves the gate (vet_step returns its text),
    and exactly one sample-less report carries the map record."""
    occ = live_map(FAR)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    _break_map(occ, fault)
    with pytest.raises(SafetyViolation) as direct:
        occ.clearance(q[None, :])
    raised = _raises_of(monkeypatch, occ, "clearance")
    reason, exc = _refuse(h, phase, q_prev, q)
    assert len(raised) == 1 and reason == str(raised[0])
    assert (exc is None) == (phase == "vet_step") and (exc is None or exc is raised[0])   # never rebuilt
    assert _age_free(reason) == _age_free(str(direct.value))                             # the map's own text
    (r,) = h.drain_clearance_reports()
    assert set(r) == REPORT_KEYS | {"raised"}
    assert r["raised"] == "SafetyViolation" and r["gate"] == "arm" and r["phase"] == phase
    assert r["reason"] == reason and r["voxels"] == [] and r["sample"] is None and r["clearance_m"] is None
    assert r["map"]["refreshes"] == occ.refreshes == r["gate_refreshes"] == 1
    if fault == "stale":
        assert r["map"]["age_s"] > 7200                                  # the evidence of a stale halt
    assert h.drain_clearance_reports() == []


@pytest.mark.parametrize("phase", ["vet_pose", "vet_step", "approve"])
@pytest.mark.parametrize("gate", ["attached", pytest.param("tool volume", marks=needs_pin)])
def test_every_gate_files_a_report_when_its_map_query_raises(request, monkeypatch, gate, phase):
    """The held-object and tool-volume gates query the map themselves
    (payload_clearance, clearance_excluding): a raise there files a report
    under that gate's set name and leaves as the same object."""
    if gate == "attached":
        occ = live_map(FAR)
        occ._payload_samples["scene"] = np.array([[0., 0., -.1]])
        _, _, h = case(occupancy=occ)
        q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
        method = "payload_clearance"
    else:
        rebot = request.getfixturevalue("rebot")
        h, occ = _rebot_harness(rebot, FAR)
        q_prev = q = rebot.home
        method = "clearance_excluding"
    fired = []

    def refusing(*_a, **_k):
        fired.append(SafetyViolation(f"occupancy unsafe: {method} refused (test)"))
        raise fired[-1]

    monkeypatch.setattr(occ, method, refusing)
    reason, exc = _refuse(h, phase, q_prev, q)
    assert len(fired) == 1 and reason == str(fired[0])
    assert (exc is None) == (phase == "vet_step") and (exc is None or exc is fired[0])
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == gate and r["raised"] == "SafetyViolation" and r["reason"] == reason
    assert r["phase"] == phase and r["sample"] is None and r["voxels"] == [] and r["map"]["refreshes"] == 1


def test_a_raising_map_files_one_report_and_the_halt_has_the_last_word():
    occ = live_map(FAR)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    _break_map(occ, "stale")
    for _ in range(5):
        assert _refuse(h, "vet_pose", q_prev, q)[1] is not None
        assert _refuse(h, "vet_step", q_prev, q)[0]
    reason, exc = _refuse(h, "approve", q_prev, q)
    assert exc is not None
    (r,) = h.drain_clearance_reports()                       # one per gate and map commit
    assert r["phase"] == "approve" and r["reason"] == reason and r["raised"] == "SafetyViolation"


def test_a_gate_without_a_report_context_files_nothing():
    """spatial_moves' camera-box check calls the arm gate with no ctx: its
    refusals (returned or raised) stay report-free, as before."""
    occ = live_map(FAR)
    _, _, h = case(occupancy=occ)
    _break_map(occ, "stale")
    with pytest.raises(SafetyViolation):
        h._occupancy_violation(np.array([[.3, 0., .3]]), None)
    assert h.drain_clearance_reports() == []


@pytest.mark.parametrize("phase", ["vet_pose", "approve"])
def test_a_refresh_between_the_gate_read_and_its_report_is_flagged(monkeypatch, phase):
    """The gate holds no map lock from its read to the report (the 50 Hz
    path stays as it was): a refresh committed in between makes the
    snapshot newer than what the gate read, and the report says so."""
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    reason, _ = _refuse(h, phase, q_prev, q)
    (r,) = h.drain_clearance_reports()
    assert r["gate_refreshes"] == r["map"]["refreshes"] == 1 and "snapshot_newer_map" not in r
    original = occ.clearance

    def read_then_commit(points):
        out = original(points)
        if occ.refreshes == 1:
            occ.refresh(depth_frame(), T_CAM)               # lands after the gate's read
        return out

    monkeypatch.setattr(occ, "clearance", read_then_commit)
    assert _refuse(h, phase, q_prev, q)[0] == reason          # decided on the gate's own read
    (r,) = h.drain_clearance_reports()
    assert r["gate_refreshes"] == 1 and r["map"]["refreshes"] == 2 and r["snapshot_newer_map"] is True


def test_a_halt_that_takes_over_a_report_rechecks_the_newer_map_flag(monkeypatch):
    """A preflight read commit 1 and filed a snapshot of commit 2 (flagged);
    a streamed halt at the same cell that read commit 2 takes the report
    over, and its account is exact again."""
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    original = occ.clearance

    def read_then_commit(points):
        out = original(points)
        if occ.refreshes == 1:
            occ.refresh(depth_frame(), T_CAM)
        return out

    monkeypatch.setattr(occ, "clearance", read_then_commit)
    assert h.vet_pose(q)
    monkeypatch.setattr(occ, "clearance", original)
    reason, exc = _refuse(h, "approve", q_prev, q)
    assert exc is not None
    (r,) = h.drain_clearance_reports()
    assert r["phase"] == "approve" and r["reason"] == reason
    assert r["gate_refreshes"] == r["map"]["refreshes"] == 2 and "snapshot_newer_map" not in r


@pytest.mark.parametrize("clear", ["fence_capture_refresh", "begin_scene_reset"])
@pytest.mark.parametrize("phase", ["vet_pose", "approve"])
def test_a_cache_clear_between_the_gate_read_and_its_report_is_flagged(monkeypatch, clear, phase):
    """Payload tracking drops the map cache without a refresh (a fenced or
    retained-payload capture refresh, a scene reset, a background replay):
    the refresh count stays, yet the snapshot is not of the map the gate
    read. The report says so structurally (`cache_clears`)."""
    occ = live_map(UNDER)
    _, _, h = case(occupancy=occ)
    q_prev, q = np.array([.3, 0., .31]), np.array([.3, 0., .3])
    original = occ.clearance
    if clear == "begin_scene_reset":
        occ._payload_pose_fn = lambda *_: None          # a payload-tracking map (begin_scene_reset's guard)

    def read_then_clear(points):
        out = original(points)
        if clear == "fence_capture_refresh":
            occ.fence_capture_refresh("test: the capture refresh failed")
        else:
            assert occ.begin_scene_reset()
        return out

    monkeypatch.setattr(occ, "clearance", read_then_clear)
    reason, _ = _refuse(h, phase, q_prev, q)
    assert reason and reason.startswith("point 0 clearance")             # decided on the gate's own read
    (r,) = h.drain_clearance_reports()
    assert r["gate_refreshes"] == r["map"]["refreshes"] == 1              # no refresh counted
    assert r.get("snapshot_newer_map") is True and r["voxels"] == []
    assert r["map"]["cache_clears"] == r["gate_cache_clears"] + 1 == 1


def test_every_cache_drop_counts_a_clear():
    """The cache is dropped in one place (`_drop_cache`), which counts it:
    no other code path sets the grid or the occupied cloud to None."""
    import inspect
    import re as _re

    from cascade.perception import occupancy

    src = inspect.getsource(occupancy.OccupancyMap)
    drops = [m.start() for m in _re.finditer(r"self\._(grid|occupied)\s*=\s*(self\._(grid|occupied)\s*=\s*)?None", src)]
    body = inspect.getsource(occupancy.OccupancyMap._drop_cache)
    assert len(drops) == 1 and _re.search(r"self\._grid = self\._occupied = None", body), drops
    occ = live_map(UNDER)
    before = occ.cache_clears
    occ.fence_capture_refresh("test")
    assert occ.cache_clears == before + 1 and occ.clearance_record()["cache_clears"] == before + 1
    assert occ._grid is None and occ._occupied is None and occ._last_refresh is None


def test_a_returned_unavailable_and_a_later_raise_at_one_commit_keep_two_reports():
    """Both are sample-less refusals of the same gate at the same commit; the
    raise is a different event (`raised`) and keeps its own report."""

    class Flaky:
        required, status, refreshes, unknown_blocks = True, None, 3, True
        raising = False

        def clearance(self, points):
            if self.raising:
                raise SafetyViolation("required occupancy unavailable: the bridge stopped answering (test)")
            return None

        def nearby_occupied(self, point, radius_m, limit):
            return np.zeros((0, 4))

        def clearance_record(self):
            return {"refreshes": 3, "age_s": 0.0, "voxel_m": .01, "occupied": 0}

    occ = Flaky()
    _, _, h = case(occupancy=occ)
    q = np.array([.3, 0., .3])
    assert h.vet_pose(q) == "clearance unavailable (fresh observed distance grid required)"
    occ.raising = True
    with pytest.raises(SafetyViolation):
        h.vet_pose(q)
    first, second = h.drain_clearance_reports()
    assert "raised" not in first and first["reason"].startswith("clearance unavailable")
    assert second["raised"] == "SafetyViolation" and second["reason"].startswith("required occupancy unavailable")
    assert first["gate"] == second["gate"] == "arm" and first["map"]["refreshes"] == second["map"]["refreshes"]


def test_the_same_refusal_after_a_cache_drop_keeps_its_own_report():
    """A cache drop without a refresh is another map state: the refusal
    filed after it gets its own report (its map record shows the drop), not
    the earlier report moved last with the earlier record."""

    class Dropped:
        required, status, refreshes, unknown_blocks, cache_clears = True, None, 3, True, 0

        def clearance(self, points):
            return None

        def nearby_occupied(self, point, radius_m, limit):
            return np.zeros((0, 4))

        def clearance_record(self):
            return {"refreshes": 3, "cache_clears": self.cache_clears, "age_s": 0.0, "voxel_m": .01, "occupied": 0}

    occ = Dropped()
    _, _, h = case(occupancy=occ)
    q = np.array([.3, 0., .3])
    assert h.vet_pose(q).startswith("clearance unavailable")
    occ.cache_clears = 1
    assert h.vet_pose(q).startswith("clearance unavailable")
    first, second = h.drain_clearance_reports()
    assert (first["gate_cache_clears"], second["gate_cache_clears"]) == (0, 1)
    assert second["map"]["cache_clears"] == 1 and "snapshot_newer_map" not in second


def test_after_a_halt_takeover_the_voxel_distances_stay_measured_from_the_snapshot_centre(monkeypatch):
    """A streamed halt rewrites `sample`; the snapshot (voxels and their `d`
    column) stays the first sample's: `d` is measured from
    `snapshot.centre`, never from the new sample (docs say so)."""
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]))
    _, _, h = case(occupancy=occ)
    q = np.array([.3, 0., .3])
    halt_q = q + [.004, .003, 0.]                             # the same 1 cm cell
    assert h.vet_pose(q)
    reason, exc = _refuse(h, "approve", halt_q + [0., 0., .001], halt_q)
    assert exc is not None
    (r,) = h.drain_clearance_reports()
    centre, sample = np.asarray(r["snapshot"]["centre"]), np.asarray(r["sample"])
    assert np.allclose(centre, q) and np.allclose(sample, halt_q)
    vox = np.asarray(r["voxels"], dtype=float)
    np.testing.assert_allclose(vox[:, 3], np.round(np.linalg.norm(vox[:, :3] - centre, axis=1), 3), atol=1.5e-3)
    assert np.max(np.abs(vox[:, 3] - np.linalg.norm(vox[:, :3] - sample, axis=1))) > 2e-3


def test_a_streamed_halt_takes_over_the_matching_report_and_moves_last(monkeypatch):
    """One report per gate, 1 cm spot and map commit. A repeat moves it last
    (the refusal that ended the call is the last one), and a streamed halt
    (approve) rewrites what happened while the snapshot stays: same cell,
    same commit, so no new voxels."""
    occ = live_map(centred_box([.31, .03, .26], [.44, .16, .04]))             # flat top at z 0.28
    _, _, h = case(occupancy=occ)
    calls = spy(monkeypatch, occ)
    q, other = np.array([.3, 0., .3]), np.array([.2, 0., .3])
    halt_q = q + [.002, 0., 0.]                              # another pose in the same 1 cm cell
    preflight = h.vet_pose(q)
    assert preflight and h.vet_pose(other)
    reason, exc = _refuse(h, "approve", halt_q + [0., 0., .001], halt_q)
    assert exc is not None and reason != preflight
    assert h.vet_pose(other)                                 # a repeat: moves last, nothing rewritten
    assert h.vet_step(q + [0., 0., .001], q, .02)            # not a halt: moves last, keeps the halt
    reports = h.drain_clearance_reports()
    assert [r["phase"] for r in reports] == ["vet_pose", "approve"]
    assert reports[0]["sample"] == pytest.approx(other.tolist())
    kept = reports[-1]
    assert kept["reason"] == reason and kept["q"] == halt_q.tolist()
    assert kept["q_prev"] == (halt_q + [0., 0., .001]).tolist()
    assert kept["sample"] == pytest.approx(halt_q.tolist()) and kept["tcp"][0][3] == pytest.approx(halt_q[0])
    assert len(calls) == 2 and kept["voxels"]                # one snapshot per cell: its voxels stayed


def test_reports_carry_the_base_pose_and_every_gate_reads_the_map_frame():
    """`sample` is in the frame the gate queried the map in, the MAP (table)
    frame for every gate: `base_pose @` FK for the TCP and joint origins
    (#348), a held object and the tool parts; `sample_tcp` is the TCP frame
    (base_pose @ FK) for every gate, and the snapshot is centred on the
    map-frame sample."""
    from cascade.types import pose_to_transform

    base = pose_to_transform([0., .22, 0., 0., 0., .3])
    q = np.array([.3, 0., .3])
    _, _, h0 = case()
    T = h0.kin.fk(q)
    tcp_map = (base @ T)[:3, 3]
    # the TCP gate: a box 2 cm under the TCP where it physically is
    h = SafetyHarness(h0.limits, h0.kin, occupancy=live_map(centred_box(tcp_map - [0., 0., .04], [.04, .04, .04])),
                      base_pose=base)
    assert h.vet_pose(q).startswith("point 0 clearance")
    (r,) = h.drain_clearance_reports()
    assert r["base_pose"] == np.round(base, 6).tolist() and r["base_pose"] != np.eye(4).tolist()
    np.testing.assert_allclose(r["sample"], tcp_map, atol=1e-12)
    np.testing.assert_allclose(r["sample_tcp"], [0., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(r["sample_tcp"], (np.linalg.inv(base @ T) @ [*r["sample"], 1.])[:3], atol=1e-12)
    assert r["snapshot"]["centre"] == r["sample"] and r["voxels"]
    assert r["voxels"][0][3] == pytest.approx(.02, abs=.006)      # the box top, measured from the lifted TCP
    # the same box at the TCP's raw base-frame numbers: 22 cm from the arm
    h = SafetyHarness(h0.limits, h0.kin, occupancy=live_map(UNDER), base_pose=base)
    assert h.vet_pose(q) is None and h.drain_clearance_reports() == []
    # a held sample 10 cm under the TCP: queried in the table frame
    held = (base @ T @ [0., 0., -.1, 1.])[:3]
    occ = live_map(centred_box(held - [0., 0., .03], [.04, .04, .02]))     # its top 2 cm under it
    occ._payload_samples["scene"] = np.array([[0., 0., -.1]])
    h = SafetyHarness(h0.limits, h0.kin, occupancy=occ, base_pose=base)
    assert h.vet_pose(q).startswith("attached object point")
    (r,) = h.drain_clearance_reports()
    assert r["gate"] == "attached" and r["base_pose"] == np.round(base, 6).tolist()
    np.testing.assert_allclose(r["sample"], held, atol=1e-9)
    np.testing.assert_allclose(r["sample_tcp"], [0., 0., -.1], atol=1e-9)
    np.testing.assert_allclose(r["sample_tcp"], (np.linalg.inv(base @ np.asarray(r["tcp"])) @ [*held, 1.])[:3],
                               atol=1e-9)


def test_clearance_record_is_the_map_state_a_report_carries(monkeypatch):
    assert OccupancyMap().clearance_record() == {"refreshes": 0, "cache_clears": 0, "age_s": None, "voxel_m": None,
                                                 "occupied": None}
    occ = live_map(UNDER)
    rec = occ.clearance_record()
    assert rec["refreshes"] == 1 and rec["voxel_m"] == .01 and rec["occupied"] == len(occ._occupied) > 0
    assert 0 <= rec["age_s"] < 5 and round(rec["age_s"], 3) == rec["age_s"]
    # the harness takes the record from the map, never from its private fields
    sentinel = {"refreshes": 1, "age_s": 1.5, "voxel_m": .02, "occupied": 3}
    monkeypatch.setattr(occ, "clearance_record", lambda: dict(sentinel))
    _, _, h = case(occupancy=occ)
    assert h.vet_pose(np.array([.3, 0., .3]))
    (r,) = h.drain_clearance_reports()
    assert r["map"] == sentinel and r["voxels"] and "snapshot_error" not in r

    class NoRecord:                         # a stand-in without the method still gets its voxels
        required, status, refreshes = False, None, 3

        def clearance(self, points):
            return np.full(len(points), .01)

        def nearby_occupied(self, point, radius_m, limit):
            return np.array([[.3, 0., .29, .01]])

    _, _, h = case(occupancy=NoRecord())
    assert h.vet_pose(np.array([.3, 0., .3]))
    (r,) = h.drain_clearance_reports()
    assert r["map"] is None and "clearance_record" in r["snapshot_error"]
    assert r["voxels"] == [[.3, 0., .29, .01]] and "snapshot_newer_map" not in r


def _nearby_reference(cloud, point, radius_m, limit):
    """nearby_occupied as first shipped: the float64 box and radius cuts on the whole cloud."""
    p = np.asarray(point, dtype=float).reshape(3)
    r = float(radius_m)
    pts = np.asarray(cloud, dtype=float)
    pts = pts[np.all(np.abs(pts - p) <= r, axis=1)]
    d = np.linalg.norm(pts - p, axis=1)
    pts, d = pts[d <= r], d[d <= r]
    order = np.argsort(d, kind="stable")[: int(limit)]
    return np.column_stack([pts[order], d[order]])


@pytest.mark.parametrize("n", [10_000, 50_000, 200_000])
def test_nearby_occupied_equals_the_float64_cut(n):
    """The float32 pre-cut is only a speed-up: random clouds in the reBot
    workspace plus voxels one float32 step either side of the radius."""
    rng = np.random.default_rng(n)
    lo, hi = np.array([.1, -.3, -.01]), np.array([.5, .3, .55])
    cloud = (lo + (hi - lo) * rng.random((n, 3))).astype(np.float32)
    p = cloud[0].astype(float)                                # exactly representable: edges are exact
    edge = []
    for x in (p[0] - .05, p[0] + .05, p[0] - .02, p[0] + .02):
        x32 = np.float32(x)
        edge += [[v, p[1], p[2]] for v in (np.nextafter(x32, np.float32(-1)), x32, np.nextafter(x32, np.float32(1)))]
    cloud = np.vstack([cloud, np.asarray(edge, dtype=np.float32)])
    occ = OccupancyMap()
    occ._occupied = cloud
    queries = [p] + [lo + (hi - lo) * rng.random(3) for _ in range(12)] + [cloud[k].astype(float)
                                                                          for k in rng.integers(0, n, 6)]
    for point in queries:
        for radius, limit in ((.05, 64), (.02, 8), (.2, 10 ** 6)):
            got = occ.nearby_occupied(point, radius, limit)
            want = _nearby_reference(cloud, point, radius, limit)
            assert got.shape == want.shape and np.array_equal(got, want)
    assert len(occ.nearby_occupied(p, .05, 10 ** 6)) > 1


def test_nearby_occupied_keeps_float64_edge_voxels():
    """A float64 cache (a stand-in, or a future bridge): a voxel whose x lies
    one float64 step beyond the rounded bound yet inside the exact cut (the
    float64 subtraction rounds it in). The pre-cut's widening keeps it."""
    for p0, r, x in ((0.02559258618412316, .03, -0.004407413815876838),
                     (0.05511730714318658, .05, 0.005117307143186574),
                     (-0.07784265105538583, .05, -0.027842651055385827),
                     (0.03532548277782002, .03, 0.005325482777820022)):
        p = np.array([p0, .1, .2])
        occ = OccupancyMap()
        occ._occupied = np.array([[x, .1, .2], [p0, .1, .2]])
        got = occ.nearby_occupied(p, r, 8)
        assert len(got) == 2 and np.array_equal(got, _nearby_reference(occ._occupied, p, r, 8))


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


@needs_pin_so101
def test_two_arms_each_row_carries_only_its_own_calls_reports(tmp_path):
    """A refusal on the right arm: its row carries only that call's reports
    (the right arm's base pose, its own map), none of the reports either arm
    filed between calls, and the next clear call of either arm carries none."""
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    cfg = load_demo_config(arms=["so101_left", "so101_right"], camera="mock", llm="mock")
    runtime, arm = build_runtime(cfg, tmp_path / "run", view=False, serve=False)
    try:
        arms = {name: runtime.arm_rig.get(name) for name in ("so101_left", "so101_right")}
        home = {name: np.asarray(a.get_state().q, dtype=float) for name, a in arms.items()}
        # each TCP where it physically is: the map is in the table frame (#348)
        tcp = {name: (a.harness.base_pose @ a.harness.kin.fk(home[name]))[:3, 3] for name, a in arms.items()}
        # each arm reads its own map; the right one has a box 4.5 cm under its TCP
        maps = {"so101_left": live_map(FAR),
                "so101_right": live_map(centred_box(tcp["so101_right"] + [0., 0., -.055], [.04, .04, .02]))}
        for name, a in arms.items():
            a.harness.occupancy = maps[name]

        def file_between_calls():
            """One refusal per arm outside any call, on a 2 cm-voxel map."""
            for name, a in arms.items():
                a.harness.occupancy = live_map(centred_box(tcp[name] + [0., 0., -.03], [.04, .04, .02]), voxel=.02)
                assert a.harness.vet_pose(home[name])
                a.harness.occupancy = maps[name]

        file_between_calls()
        r = runtime.execute("move_relative", {"direction": "down", "distance_m": .03, "arm": "so101_right"})
        assert r["ok"] is False and "occupancy map" in r["error"]
        assert all(np.allclose(a.get_state().q, home[name]) for name, a in arms.items())
        reports = runtime.trace.rows()[-1]["context"]["clearance"]
        right_base = np.round(arms["so101_right"].harness.base_pose, 6).tolist()
        assert right_base != np.round(arms["so101_left"].harness.base_pose, 6).tolist()
        assert reports and all(rep["base_pose"] == right_base for rep in reports)
        assert all(rep["map"]["voxel_m"] == .01 and rep["voxels"] for rep in reports)   # this call's map only
        assert {rep["gate"] for rep in reports} <= {"tcp"} | {f"link {i}" for i in range(2, 8)}
        for name in arms:
            file_between_calls()
            r = runtime.execute("move_home", {"arm": name})
            assert r["ok"] is True, r
            row = runtime.trace.rows()[-1]
            assert row["skill"] == "move_home" and "clearance" not in row["context"]
    finally:
        shutdown_runtime(runtime, arm)
