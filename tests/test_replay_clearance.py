"""scripts/replay_clearance.py: re-vet recorded clearance refusals offline (spec §6.3).

Every clearance refusal files a bounded map snapshot into its trace row
(`context.clearance`, SafetyHarness.drain_clearance_reports). The replay
rebuilds an occupancy stand-in from each snapshot's voxels (`ReplayMap`, exact
cube distances, never the recorded `d` column) and re-runs the harness at the
recorded joints:

- `old`: the static strict map check at `q` (vet_pose's map gates, no escape);
- `L1`: `vet_step(q_prev, q)` (the escape rule; static without a `q_prev`);
- `L4`: `old` with an override tool volume (the rebuilt camera box + parts).

The reports here are REAL: a reBot harness (URDF kinematics, shipped tool
volume) over an OccupancyMap on the synthetic GridBridge refuses, its report is
drained and written as a trace row with json.dumps. The obstacle faces sit
EPS inside a voxel boundary, so the map's exact box distance and the replay's
cube distance differ by exactly EPS.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_pin
from occupancy_grid_bridge import GridBridge, depth_frame, make_map
from test_harness_tool_volume import T_CAM, side_grasp_geometry

from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "replay_clearance.py"

#: the GridBridge lattice of make_map (REGION_MIN -0.30, -0.45, -0.02): voxel
#: boundaries on whole centimetres, centres at x.xx5 (exact at the snapshot's 1 mm)
V = 0.01
#: obstacle faces sit this far inside a voxel boundary: one voxel layer, and the
#: map's box distance is the replay's cube distance + EPS
EPS = 1e-4
#: an obstacle face 2.03 cm ahead of the recorded TCP (x = 0.4498)
FACE_X = 0.47
X_REC, X_IN, X_OUT, X_TAKE = 0.4498, 0.4598, 0.4398, 0.4538
#: the URDF parts and a cut camera box (configs/arms/rebot_rs.yaml comment):
#: lever 4's rebuilt volume
PARTS = {"tool_volume": {"housing": {"min": [-0.160, -0.095, -0.045], "max": [-0.070, 0.095, 0.045]},
                         "fingers": {"min": [-0.090, -0.085, -0.020], "max": [0.0, 0.085, 0.020]}},
         "wrist_camera": {"front_m": 0.06, "bottom_m": 0.035}}


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location("replay_clearance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebot():
    return side_grasp_geometry()


# ── recording real refusals ─────────────────────────────────────────────────


def live_map(*boxes):
    occ = make_map(GridBridge(list(boxes), voxel=V), unobserved="free")
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return occ


def recorder(rebot, *boxes, base_pose=None, payload=None):
    occ = live_map(*boxes)
    if payload is not None:
        occ._payload_samples["scene"] = np.asarray(payload, dtype=float)
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin, occupancy=occ,
                      base_pose=base_pose, tool_volume=rebot.volume)
    return h, occ


def at_x(rebot, x):
    """The home orientation with the TCP moved to x (IK, 1e-9)."""
    T = rebot.kin.fk(rebot.home)
    T[0, 3] = x
    ik = rebot.kin.ik(T, rebot.home, tol=1e-9)
    assert ik.success
    q = np.asarray(ik.q, dtype=float)
    assert abs(rebot.kin.fk(q)[0, 3] - x) < 1e-7
    return q


def ahead_box(rebot, half_y=0.04):
    """A wall 2.03 cm ahead of the TCP at X_REC (face EPS inside the 0.47 boundary)."""
    z = rebot.kin.fk(rebot.home)[2, 3]
    return (FACE_X + EPS, -half_y, z - 0.04), (FACE_X + 0.04, half_y, z + 0.04)


def voxel_box(centre):
    """One occupied voxel: a box EPS inside the voxel at `centre`."""
    c = np.asarray(centre, dtype=float)
    return tuple(c - V / 2 + EPS), tuple(c + V / 2 - EPS)


def lattice(p):
    """The voxel centre nearest p on make_map's lattice (x.xx5)."""
    return np.floor(np.asarray(p, dtype=float) / V) * V + V / 2


def row(step, skill, reports):
    return {"step": step, "t": 1.0, "skill": skill, "tier": None, "args": {},
            "context": {"arm": "default", "held_object": None, "clearance": reports},
            "duration_ms": 1.0, "result": {"ok": False}}


def write_trace(path, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r if isinstance(r, str) else json.dumps(r) + "\n" for r in rows))
    return path


def vet_pose_refusal(rebot, q_rec):
    """A real vet_pose TCP refusal at q_rec, its report as a trace would carry it."""
    h, _ = recorder(rebot, ahead_box(rebot))
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("point 0 clearance"), reason
    (rep,) = h.drain_clearance_reports()
    return json.loads(json.dumps(rep))


def approve_refusal(rebot, q_from, q_to, *boxes, exemption=None, z_max=None):
    """A real streamed refusal of the closing step q_from -> q_to."""
    h, _ = recorder(rebot, *(boxes or (ahead_box(rebot),)))
    h.heartbeat()
    with h.target_column_exemption(exemption, z_max), pytest.raises(SafetyViolation) as exc:
        h.approve(q_from, q_to, 1.0)
    (rep,) = h.drain_clearance_reports()
    assert rep["reason"] == str(exc.value)
    return json.loads(json.dumps(rep))


# ── the brief's case (spec §6.3, R6) ────────────────────────────────────────


@needs_pin
def test_a_recorded_refusal_replays_old_refused_and_l1_allowed(rc, rebot, tmp_path):
    q_rec, q_in = at_x(rebot, X_REC), at_x(rebot, X_IN)
    rep = vet_pose_refusal(rebot, q_rec)
    assert rep["phase"] == "vet_pose" and rep["gate"] == "tcp" and rep["q_prev"] is None
    # R6: emulates a pre-PR-A trace -- the recorded step came from 1 cm CLOSER
    # (q_in), so q strictly retreats from the wall
    rep["q_prev"] = q_in.tolist()
    trace = write_trace(tmp_path / "mcp_1_x" / "trace.jsonl", row(4, "move_relative", [rep]))
    (loaded,) = rc.load_reports([trace])
    assert (loaded["run"], loaded["step"], loaded["skill"]) == ("mcp_1_x", 4, "move_relative")

    out = rc.replay(loaded, arm="rebot_rs")
    assert out["old"].startswith("refused") and f"{rep['clearance_m']:.3f}" in out["old"]
    assert out["old"].startswith("refused: point 0 clearance")
    assert out["L1"] == "allowed"               # the recorded step strictly retreated (built that way)
    assert out["L4"] == out["old"]              # no override volume: the shipped one
    assert out["partial_map"] is True and out["replayable"] is True and out["approximate"] == []
    summary = rc.summarize([out])
    assert summary == {"old": {"allowed": 0, "refused": 1}, "L1": {"allowed": 1, "refused": 0},
                       "L4": {"allowed": 0, "refused": 1}}


@needs_pin
def test_the_replay_clearance_is_the_cube_distance_of_the_snapshot(rc, rebot):
    """Exactness: the wall's face sits EPS inside a voxel boundary, so the
    recorded (exact box) clearance and the replay's cube distance differ by
    exactly EPS; a centre distance would read half a voxel farther."""
    rep = vet_pose_refusal(rebot, at_x(rebot, X_REC))
    assert rep["clearance_m"] == pytest.approx(FACE_X + EPS - X_REC, abs=1e-6)
    out = rc.replay(rep, arm="rebot_rs")
    assert out["clearance_m"] == rep["clearance_m"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"] - EPS, abs=1e-6)
    assert out["old"] == "refused: " + rep["reason"].replace(
        f"{rep['clearance_m']:.9f}", f"{out['replay_clearance_m']:.9f}").replace(
        f"{(0.03 - rep['clearance_m']) * 1000:.6f}", f"{(0.03 - out['replay_clearance_m']) * 1000:.6f}")


@needs_pin
def test_a_streamed_closing_step_replays_l1_refused_as_recorded(rc, rebot):
    """R6 second row: a real approve refusal (the step closes in from outside
    the margin) stays refused under the escape rule, for the same point."""
    q_out, q_rec = at_x(rebot, X_OUT), at_x(rebot, X_REC)
    rep = approve_refusal(rebot, q_out, q_rec)
    assert rep["phase"] == "approve" and rep["q_prev"] == q_out.tolist()
    assert "inside the margin" not in rep["reason"]
    out = rc.replay(rep, arm="rebot_rs")
    assert out["L1"].startswith("refused: point 0 clearance") and f"{rep['clearance_m']:.3f}" in out["L1"]
    assert "inside the margin" not in out["L1"]
    assert out["old"].startswith("refused: point 0 clearance")
    assert out["phase"] == "approve" and out["gate"] == "tcp" and out["recorded"] == rep["reason"]


@needs_pin
def test_a_step_the_live_rule_refused_never_replays_as_admitted(rc, rebot):
    """A streamed or preflight refusal was decided by the step rule itself on
    the whole map: when the snapshot model admits the step (here: a q_prev
    that retreats in the model), L1 keeps the recorded refusal and says so."""
    rep = approve_refusal(rebot, at_x(rebot, X_OUT), at_x(rebot, X_REC))
    rep["q_prev"] = at_x(rebot, X_IN).tolist()
    out = rc.replay(rep, arm="rebot_rs")
    assert out["L1"] == "refused: " + rep["reason"] and out["approximate"] == ["L1 kept as recorded"]
    for phase in ("vet_step",):
        assert rc.replay({**rep, "phase": phase}, arm="rebot_rs")["L1"] == out["L1"]
    # a static refusal given an earlier q_prev (a pre-PR-A trace) is replayed as computed
    static = rc.replay({**rep, "phase": "vet_pose"}, arm="rebot_rs")
    assert static["L1"] == "allowed" and static["approximate"] == []


@needs_pin
def test_the_cli_prints_one_line_per_report_and_the_summary(rc, rebot, tmp_path, capsys):
    q_rec, q_in, q_out = at_x(rebot, X_REC), at_x(rebot, X_IN), at_x(rebot, X_OUT)
    first = vet_pose_refusal(rebot, q_rec)
    first["q_prev"] = q_in.tolist()             # emulates a pre-PR-A trace (R6)
    second = approve_refusal(rebot, q_out, q_rec)
    trace = write_trace(tmp_path / "mcp_7" / "trace.jsonl",
                        row(0, "move_relative", [first]),
                        {"step": 1, "skill": "list_objects", "context": {"arm": "default"}},
                        "not json\n",
                        row(2, "move_home", [second]))
    assert rc.main([str(trace.parent), "--arm", "rebot_rs"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["step"], r["skill"], r["L1"] == "allowed") for r in lines[:-1]] == [
        (0, "move_relative", True), (2, "move_home", False)]
    assert lines[-1] == {"old": {"allowed": 0, "refused": 2}, "L1": {"allowed": 1, "refused": 1},
                         "L4": {"allowed": 0, "refused": 2}}


# ── loading traces ──────────────────────────────────────────────────────────


def test_load_reports_reads_every_clearance_entry_with_its_row(rc, tmp_path):
    a, b, c = {"gate": "tcp", "q": [0.0]}, {"gate": "jaws", "q": [1.0]}, {"gate": "link 2", "q": [2.0]}
    one = write_trace(tmp_path / "mcp_a" / "trace.jsonl",
                      row(3, "grasp_object", [a, b]), "\n", "{broken\n", "[1, 2]\n",
                      {"step": 4, "skill": "describe_scene", "context": None},
                      {"step": 5, "skill": "move_home", "context": {"clearance": "not a list"}},
                      row(6, "move_home", ["not a report", c]))
    two = write_trace(tmp_path / "mcp_b" / "trace.jsonl", row(0, "move_relative", [a]))
    got = rc.load_reports([one, two.parent])
    assert [(r["run"], r["step"], r["skill"], r["gate"]) for r in got] == [
        ("mcp_a", 3, "grasp_object", "tcp"), ("mcp_a", 3, "grasp_object", "jaws"),
        ("mcp_a", 6, "move_home", "link 2"), ("mcp_b", 0, "move_relative", "tcp")]
    assert got[1]["q"] == [1.0]


# ── the other gates ─────────────────────────────────────────────────────────


@needs_pin
def test_a_wrist_camera_refusal_replays_and_lever_4_takes_the_override_volume(rc, rebot, tmp_path, capsys):
    """One voxel inside today's slab, in front of the cut camera box and above
    the fingers: the shipped volume refuses it, the rebuilt one (lever 4) not."""
    q_rec = at_x(rebot, X_REC)
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    centre = lattice(tcp + [-0.015, 0.0, 0.07])
    h, _ = recorder(rebot, voxel_box(centre))
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("wrist camera clearance"), reason
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["gate"] == "wrist camera" and len(rep["voxels"]) == 1

    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: wrist camera clearance") and out["L4"] == out["old"]
    # the map re-measures a tool sample to the voxel CENTRE minus half a voxel;
    # the cube distance is never larger, and at most (sqrt 3 - 1) / 2 voxel less
    gap = rep["clearance_m"] - out["replay_clearance_m"]
    assert -1e-9 <= gap <= V * (np.sqrt(3) - 1) / 2 + 1e-9
    parts = rc.tool_volume_from_override("rebot_rs", PARTS)
    assert parts.names == ("wrist camera", "housing", "fingers", "jaws")
    out4 = rc.replay(rep, arm="rebot_rs", tool_volume_override=parts)
    assert out4["L4"] == "allowed" and out4["old"] == out["old"]
    assert out4["L4_volume"] == "override" and out["L4_volume"] == "shipped"
    # the same through the CLI
    trace = write_trace(tmp_path / "mcp_9" / "trace.jsonl", row(1, "grasp_object", [rep]))
    override = tmp_path / "parts.json"
    override.write_text(json.dumps(PARTS))
    assert rc.main([str(trace), "--tool-volume", str(override)]) == 0
    *lines, summary = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["L4"] == "allowed"
    assert summary == {"old": {"allowed": 0, "refused": 1}, "L1": {"allowed": 0, "refused": 1},
                       "L4": {"allowed": 1, "refused": 0}}


def test_a_bad_override_volume_is_refused_loudly(rc):
    with pytest.raises(ValueError, match="unknown"):
        rc.tool_volume_from_override("rebot_rs", {"tool_volumes": {}})
    with pytest.raises(ValueError, match="open fingers unsampled"):
        rc.tool_volume_from_override("rebot_rs", {"wrist_camera": {"front_m": 0.06}})


@needs_pin
def test_an_attached_payload_refusal_replays_its_held_sample(rc, rebot):
    """The held sample (the report's sample_tcp) goes back on the TCP: the
    replay refuses it, under every lever (attached samples never escape)."""
    q_rec, q_in = at_x(rebot, X_REC), at_x(rebot, X_IN)
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    payload = [[0.0, 0.0, -0.10]]
    # a slab whose top face sits EPS under the voxel boundary 1.77 cm below the held sample
    top = np.floor((tcp[2] - 0.115) / V) * V
    slab = (tcp[0] - 0.04, -0.04, top - 0.03), (tcp[0] + 0.04, 0.04, top - EPS)
    h, _ = recorder(rebot, slab, payload=payload)
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("attached object point 0 clearance"), reason
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    rep["q_prev"] = q_in.tolist()               # a step sideways would not help a held object
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: attached object point 0 clearance")
    assert out["L1"].startswith("refused: attached object point 0 clearance")
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"] - EPS, abs=1e-6)


@needs_pin
def test_a_ceiling_on_the_recorded_exemption_is_kept(rc, rebot):
    """A refusal under the air-grasp exit's capped column (A3: z_max): the TCP
    sits ABOVE the ceiling, inside the cylinder, so it is checked; without the
    ceiling the replay would exempt it and call the pose allowed."""
    q_out, q_rec = at_x(rebot, X_OUT), at_x(rebot, X_REC)
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    column = (tcp[:2], 0.05, 0.30)
    # a narrow wall: every voxel of it lies inside the column's 5 cm
    rep = approve_refusal(rebot, q_out, q_rec, ahead_box(rebot, half_y=0.02), exemption=column,
                          z_max=float(tcp[2]) - 0.02)
    assert len(rep["exemption"]["cylinder"]) == 5
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: point 0 clearance")
    assert out["L1"].startswith("refused: point 0 clearance") and out["approximate"] == []
    # the step itself, replayed (as a static refusal given its q_prev), keeps the ceiling too
    assert rc.replay({**rep, "phase": "vet_pose"}, arm="rebot_rs")["L1"].startswith("refused: point 0 clearance")
    # the same report without its ceiling is exempt (the cylinder holds the TCP)
    unbounded = copy.deepcopy(rep)
    unbounded["exemption"]["cylinder"] = unbounded["exemption"]["cylinder"][:4]
    assert rc.replay(unbounded, arm="rebot_rs")["old"] == "allowed"
    # and the step is replayed under that exemption: the column admits it
    # (as a static refusal given its q_prev: a recorded step stays refused)
    assert rc.replay({**unbounded, "phase": "vet_pose"}, arm="rebot_rs")["L1"] == "allowed"


@needs_pin
def test_a_tool_part_clearance_is_read_as_its_gate_reads_it(rc):
    """The replayed clearance of a tool part drops the table band and the
    exemption column, as the gate does; a column under an `episode` key the
    replay cannot rebuild is said so."""
    model_q = [0.0] * 6
    report = {"phase": "vet_pose", "gate": "jaws", "reason": "jaws clearance 0.025 m below 0.030 m",
              "q": model_q, "q_prev": None, "sample": [0.3, 0.0, 0.03], "sample_tcp": [0.0, 0.0, 0.0],
              "clearance_m": 0.025, "margin_m": 0.03,
              "exemption": {"cylinder": [0.3, 0.0, 0.004, 0.04], "episode": {"keys": 3}},
              "map": {"refreshes": 1, "age_s": 0.1, "voxel_m": 0.01, "occupied": 3},
              "voxels": [[0.305, 0.005, 0.005, 0.026],     # the table, 2 cm under the sample
                         [0.3, 0.0, 0.05, 0.02],           # inside the exemption column, 1.5 cm above
                         [0.33, 0.005, 0.035, 0.031]]}     # the neighbour, 2.5 cm away
    out = rc.replay(report, arm="rebot_rs")
    assert out["replay_clearance_m"] == pytest.approx(0.025, abs=1e-12)
    assert out["approximate"] == ["exemption keys not replayed: ['episode']"]


@needs_pin
def test_the_recorded_base_pose_places_the_tool_volume(rc, rebot):
    """An arm bolted 20 cm along y: its tool volume is read in the table frame
    (base_pose @), so the replay must lift it the same way."""
    q_rec = at_x(rebot, X_REC)
    base_pose = np.eye(4)
    base_pose[1, 3] = 0.20
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    centre = lattice(tcp + [-0.015, 0.20, 0.07])
    h, _ = recorder(rebot, voxel_box(centre), base_pose=base_pose)
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("wrist camera clearance"), reason
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["base_pose"] == base_pose.tolist()
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: wrist camera clearance")
    rep.pop("base_pose")
    assert rc.replay(rep, arm="rebot_rs")["old"] == "allowed"


# ── what cannot be replayed exactly ─────────────────────────────────────────


@needs_pin
def test_a_map_that_raised_is_not_replayable_and_counts_as_refused(rc, rebot):
    h, occ = recorder(rebot, ahead_box(rebot))
    occ._last_refresh = time.monotonic() - occ.max_age_s - 100.0
    with pytest.raises(SafetyViolation, match="stale"):
        h.vet_pose(at_x(rebot, X_REC))
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["raised"] == "SafetyViolation" and rep["voxels"] == []
    out = rc.replay(rep, arm="rebot_rs")
    assert out["replayable"] is False and "raised" in out["not_replayable"]
    assert all(out[lever].startswith("refused: not replayable") for lever in ("old", "L1", "L4"))
    assert rep["reason"] in out["old"]
    assert rc.summarize([out]) == {lever: {"allowed": 0, "refused": 1} for lever in ("old", "L1", "L4")}


def test_reports_without_a_measured_sample_are_not_replayable(rc):
    base = {"phase": "vet_pose", "gate": "tcp", "reason": "r", "q": [0.0] * 6, "q_prev": None,
            "sample": [0.4, 0.0, 0.4], "clearance_m": 0.02, "exemption": None,
            "map": {"refreshes": 1, "age_s": 0.1, "voxel_m": 0.01, "occupied": 1},
            "voxels": [[0.425, 0.005, 0.405, 0.025]], "margin_m": 0.03}
    cases = {
        "no offending sample": {"sample": None, "clearance_m": None, "voxels": [], "gate": "arm"},
        "unobserved": {"clearance_m": None},
        "no voxels": {"voxels": [], "snapshot_error": "RuntimeError: boom"},
        "voxel size": {"map": None},
        "joints": {"q": [0.0] * 5},
        "q_prev": {"q_prev": [float("nan")] * 6},
        "bad sample": {"sample": [0.4, 0.0]},
        "sample_tcp": {"gate": "attached", "sample_tcp": None},
        "exemption": {"exemption": {"cylinder": [0.3, 0.0, 0.05]}},
        "bad voxels": {"voxels": [[0.425, 0.005]]},
    }
    for why, change in cases.items():
        out = rc.replay({**base, **change}, arm="rebot_rs")
        assert out["replayable"] is False and why in out.get("not_replayable", ""), (why, out)
        assert out["old"] == out["L1"] == out["L4"] and out["old"].startswith("refused: not replayable")


@needs_pin
def test_a_snapshot_of_a_newer_map_is_flagged_approximate(rc, rebot, monkeypatch):
    h, occ = recorder(rebot, ahead_box(rebot))
    original = occ.clearance

    def read_then_refresh(points):
        try:
            return original(points)
        finally:
            if not getattr(read_then_refresh, "done", False):
                read_then_refresh.done = True
                occ.refresh(depth_frame(), T_CAM)

    monkeypatch.setattr(occ, "clearance", read_then_refresh)
    assert h.vet_pose(at_x(rebot, X_REC)) is not None
    (rep,) = h.drain_clearance_reports()
    assert rep["snapshot_newer_map"] is True
    out = rc.replay(json.loads(json.dumps(rep)), arm="rebot_rs")
    assert out["approximate"] == ["snapshot_newer_map"] and out["old"].startswith("refused")


@needs_pin
def test_a_halt_that_took_over_a_report_is_flagged_and_replayed_from_the_centres(rc, rebot):
    """A streamed halt in the same 1 cm cell rewrites the report's sample but
    keeps the first snapshot, whose `d` column was measured from the first
    sample. The replay uses the centres only, and says the report moved."""
    q_rec, q_out, q_take = at_x(rebot, X_REC), at_x(rebot, X_OUT), at_x(rebot, X_TAKE)
    h, _ = recorder(rebot, ahead_box(rebot))
    assert h.vet_pose(q_rec) is not None
    h.heartbeat()
    with pytest.raises(SafetyViolation):
        h.approve(q_out, q_take, 1.0)
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["phase"] == "approve" and rep["sample"][0] == pytest.approx(X_TAKE, abs=1e-7)
    out = rc.replay(rep, arm="rebot_rs")
    assert out["approximate"] == ["takeover"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"] - EPS, abs=1e-6)
    assert out["L1"].startswith("refused: point 0 clearance")


@needs_pin
def test_the_replay_never_connects_and_never_changes_its_input(rc, rebot, tmp_path, monkeypatch, capsys):
    rep = vet_pose_refusal(rebot, at_x(rebot, X_REC))
    rep["q_prev"] = at_x(rebot, X_IN).tolist()
    trace = write_trace(tmp_path / "mcp_3" / "trace.jsonl", row(0, "move_relative", [rep]))

    def no_network(*args, **kwargs):
        raise AssertionError("the replay opened a connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    kept = copy.deepcopy(rep)
    rc.replay(rep, arm="rebot_rs")
    assert rep == kept
    assert rc.main([str(trace)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["L1"] == {"allowed": 1, "refused": 0}


# ── ReplayMap ───────────────────────────────────────────────────────────────


def test_replay_map_distances_are_exact_to_the_voxel_cubes(rc):
    m = rc.ReplayMap(np.array([[0.005, 0.005, 0.005, 0.123]]), 0.01)
    pts = np.array([[0.03, 0.005, 0.005],          # 2 cm off a face
                    [0.02, 0.02, 0.02],            # off the corner (0.01, 0.01, 0.01)
                    [0.007, 0.004, 0.009],         # inside the cube
                    [0.005, -0.015, 0.025]])       # off an edge
    np.testing.assert_allclose(m.clearance(pts), [0.02, np.sqrt(3) * 0.01, 0.0, np.hypot(0.015, 0.015)],
                               atol=1e-12)
    np.testing.assert_allclose(m.payload_clearance(pts), m.clearance(pts))
    assert m.required is False and m.unknown_blocks is False
    assert m.status == {"voxel": 0.01} and m.refreshes == 0
    empty = rc.ReplayMap(np.zeros((0, 4)), 0.01)
    assert np.all(np.isinf(empty.clearance(pts))) and empty.nearby_occupied(pts[0], 0.05, 8).shape == (0, 4)
    for bad in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rc.ReplayMap(np.zeros((0, 4)), bad)
    with pytest.raises(ValueError):
        rc.ReplayMap(np.array([[0.0, float("nan"), 0.0, 0.0]]), 0.01)


def test_replay_map_excludes_the_table_band_and_the_target_column(rc):
    table, target, high, neighbour = [0.105, 0.005, 0.005], [0.205, 0.005, 0.105], [0.205, 0.005, 0.305], \
        [0.305, 0.005, 0.105]
    m = rc.ReplayMap(np.array([table, target, high, neighbour]), 0.01)
    probe = np.array([[0.105, 0.005, 0.045], [0.205, 0.005, 0.145], [0.205, 0.005, 0.265],
                      [0.305, 0.005, 0.145]])
    every = m.clearance(probe)
    np.testing.assert_allclose(every, [0.035, 0.035, 0.035, 0.035], atol=1e-12)
    column = (np.array([0.205, 0.005]), 0.03, 0.05)
    got = m.clearance_excluding(probe, below_z=0.02, cylinder=column, within=0.03)
    # the table voxel and the whole column go; the neighbour stays
    assert got[0] > 0.035 and got[1] > 0.035 and got[2] > 0.035
    assert got[3] == pytest.approx(0.035, abs=1e-12)
    capped = m.clearance_excluding(probe, below_z=0.02, cylinder=(*column, 0.2), within=0.03)
    assert capped[2] == pytest.approx(0.035, abs=1e-12)        # above the ceiling: counts again
    assert capped[1] > 0.035
    assert m.clearance_excluding(probe, within=0.03)[0] == pytest.approx(0.035, abs=1e-12)


def test_replay_map_snapshot_api(rc):
    vox = np.array([[0.0, 0.0, 0.0, 9.0], [0.03, 0.0, 0.0, 9.0], [0.0, 0.01, 0.0, 9.0], [0.2, 0.0, 0.0, 9.0]])
    m = rc.ReplayMap(vox, 0.01)
    rows = m.nearby_occupied([0.0, 0.0, 0.0], 0.05, 64)
    np.testing.assert_allclose(rows, [[0.0, 0.0, 0.0, 0.0], [0.0, 0.01, 0.0, 0.01], [0.03, 0.0, 0.0, 0.03]])
    assert m.nearby_occupied([0.0, 0.0, 0.0], 0.05, 2).shape == (2, 4)
    assert m.clearance_record() == {"refreshes": 0, "age_s": 0.0, "voxel_m": 0.01, "occupied": 4}
    np.testing.assert_allclose(m._occupied, vox[:, :3])
    T = np.array([[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]])
    assert rc.ReplayMap(vox, 0.01).payload_points(T).shape == (0, 3)
    held = rc.ReplayMap(vox, 0.01, payload_tcp=[[0.1, 0.0, -0.1]])
    np.testing.assert_allclose(held.payload_points(T), [[1.0, 2.1, 2.9]], atol=1e-12)


def test_replay_map_reads_are_reentrant_under_its_refresh_lock(rc):
    """The harness holds the map's refresh lock while it builds a clearance
    report (clearance_record + nearby_occupied): a plain Lock would deadlock."""
    m = rc.ReplayMap(np.array([[0.0, 0.0, 0.0, 0.0]]), 0.01)
    done = []

    def report():
        with m._refresh_lock:
            done.append((m.clearance_record()["occupied"], len(m.nearby_occupied([0.0, 0.0, 0.0], 0.05, 8))))

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    worker.join(5.0)
    assert done == [(1, 1)]
