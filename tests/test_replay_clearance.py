"""scripts/replay_clearance.py: re-vet recorded clearance refusals offline (spec §6.3).

Every clearance refusal files a bounded map snapshot into its trace row
(`context.clearance`, SafetyHarness.drain_clearance_reports). The replay
rebuilds the live map's distance grid from each snapshot's voxels (`ReplayMap`:
the lattice EDT, read trilinearly as OccupancyMap reads its grid; never the
recorded `d` column) and re-runs the harness at the recorded joints:

- `old`: the static strict map check at `q` (vet_pose's map gates, no escape);
- `L1`: the RECORDED verdict for a refusal the step rule itself decided
  (approve / vet_step); `vet_step(q_prev, q)` (static without a `q_prev`) for a
  static refusal (vet_pose / route_end);
- `L4`: `old` with an override tool volume (the rebuilt camera box + parts).

Each computed verdict is three-valued: the snapshot alone, and the snapshot
with the certified bound for every voxel it left out (`R - |p - centre|`),
must agree, else it reads "undecidable".

The reports here are REAL: a reBot harness (URDF kinematics, shipped tool
volume) over an OccupancyMap on `EdtBridge` (each grid node reads its distance
to the nearest occupied voxel centre, as nvblox's ESDF and the warp / voxel EDT
do) refuses, its report is drained and written as a trace row with json.dumps.
One test records on the box-distance GridBridge (a test fake), whose distances
voxel centres cannot give back.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import socket
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_pin
from occupancy_grid_bridge import TABLE, GridBridge, depth_frame, make_map
from scipy import ndimage
from test_harness_tool_volume import OBJ, T_CAM, side_grasp_geometry

from cascade.perception.occupancy import OccupancyMap, RefusalSnapshot
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.types import SafetyViolation

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "replay_clearance.py"

#: the lattice of make_map (REGION_MIN -0.30, -0.45, -0.02): voxel boundaries
#: on whole centimetres, centres at x.xx5 (exact at the snapshot's 1 mm)
V = 0.01
#: obstacle faces sit this far inside a voxel boundary: the voxel just in front
#: of the face stays free
EPS = 1e-4
#: an obstacle face just past 0.47: its first occupied centres at x = 0.475,
#: 2.52 cm ahead of the recorded TCP (x = 0.4498) on the EDT grid
FACE_X = 0.47
X_REC, X_IN, X_OUT, X_TAKE = 0.4498, 0.4598, 0.4398, 0.4538
#: the URDF parts and a cut camera box (configs/arms/rebot_rs.yaml comment):
#: lever 4's rebuilt volume
PARTS = {"tool_volume": {"housing": {"min": [-0.160, -0.095, -0.045], "max": [-0.070, 0.095, 0.045]},
                         "fingers": {"min": [-0.090, -0.085, -0.020], "max": [0.0, 0.085, 0.020]}},
         "wrist_camera": {"front_m": 0.06, "bottom_m": 0.035}}
#: a test-only snapshot holding the whole scene (shipped: 5 cm, 64 voxels), so
#: an admission can be certified: no voxel it left out lies within 1 m
WHOLE = RefusalSnapshot(radius_m=1.0, max_voxels=4096, max_reports=32)
BUCKETS = ("allowed", "refused", "undecidable", "not_replayable", "no_snapshot")
NUMBER = re.compile(r"-?\d+\.\d+")


def buckets(**counts):
    return {**dict.fromkeys(BUCKETS, 0), **counts}


def same_words(a: str, b: str) -> bool:
    """The same text up to its numbers."""
    return NUMBER.sub("#", a) == NUMBER.sub("#", b)


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location("replay_clearance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rebot():
    return side_grasp_geometry()


class EdtBridge(GridBridge):
    """The real backends' distance grid (nvblox's ESDF at its voxel centres,
    the warp / voxel EDT): each grid node reads its distance to the nearest
    OCCUPIED voxel centre. Occupied = GridBridge's (centres within v/2 of a
    box)."""

    def _grid(self, lo, hi):
        out = super()._grid(lo, hi)
        assert not self.unknown
        points = np.asarray(out["points"], dtype=float)
        if len(points):
            grid = out["grid"]
            occupied = np.zeros(grid.shape, dtype=bool)
            idx = np.round((points - np.asarray(out["origin"], dtype=float)) / self.voxel).astype(int)
            occupied[tuple(idx.T)] = True
            out["grid"] = ndimage.distance_transform_edt(~occupied, sampling=self.voxel).astype(np.float32)
        return out


# ── recording real refusals ─────────────────────────────────────────────────


def live_map(*boxes, bridge=EdtBridge, snapshot=None):
    occ = make_map(bridge(list(boxes), voxel=V), unobserved="free")
    if snapshot is not None:
        occ.refusal_snapshot = snapshot
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return occ


def recorder(rebot, *boxes, base_pose=None, payload=None, snapshot=None, bridge=EdtBridge):
    occ = live_map(*boxes, bridge=bridge, snapshot=snapshot)
    if payload is not None:
        occ._payload_samples["scene"] = np.asarray(payload, dtype=float)
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin, occupancy=occ,
                      base_pose=base_pose, tool_volume=rebot.volume)
    return h, occ


def at_tcp(rebot, x, dy=0.0, dz=0.0):
    """The home orientation with the TCP moved to x (and dy, dz off home's) (IK, 1e-9)."""
    T = rebot.kin.fk(rebot.home)
    T[0, 3] = x
    T[1, 3] += dy
    T[2, 3] += dz
    ik = rebot.kin.ik(T, rebot.home, tol=1e-9)
    assert ik.success
    q = np.asarray(ik.q, dtype=float)
    assert np.allclose(rebot.kin.fk(q)[:3, 3], T[:3, 3], atol=1e-7)
    return q


def at_x(rebot, x):
    return at_tcp(rebot, x)


def ahead_box(rebot, half_y=0.04):
    """A wall ahead of the TCP: its face EPS past 0.47, first occupied centres at 0.475."""
    z = rebot.kin.fk(rebot.home)[2, 3]
    return (FACE_X + EPS, -half_y, z - 0.04), (FACE_X + 0.04, half_y, z + 0.04)


def voxel_box(centre):
    """One occupied voxel: a box EPS inside the voxel at `centre`."""
    c = np.asarray(centre, dtype=float)
    return tuple(c - V / 2 + EPS), tuple(c + V / 2 - EPS)


def lattice(p):
    """The voxel centre nearest p on make_map's lattice (x.xx5)."""
    return np.floor(np.asarray(p, dtype=float) / V) * V + V / 2


def row(step, skill, reports, result=None):
    return {"step": step, "t": 1.0, "skill": skill, "tier": None, "args": {},
            "context": {"arm": "default", "held_object": None, "clearance": reports},
            "duration_ms": 1.0, "result": result or {"ok": False}}


def write_trace(path, *rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r if isinstance(r, str) else json.dumps(r) + "\n" for r in rows))
    return path


def vet_pose_refusal(rebot, q_rec, *, snapshot=None):
    """A real vet_pose TCP refusal at q_rec, its report as a trace would carry it."""
    h, _ = recorder(rebot, ahead_box(rebot), snapshot=snapshot)
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("point 0 clearance"), reason
    (rep,) = h.drain_clearance_reports()
    return json.loads(json.dumps(rep))


def approve_refusal(rebot, q_from, q_to, *boxes, exemption=None, z_max=None, snapshot=None, bridge=EdtBridge):
    """A real streamed refusal of the closing step q_from -> q_to."""
    h, _ = recorder(rebot, *(boxes or (ahead_box(rebot),)), snapshot=snapshot, bridge=bridge)
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
    rep = vet_pose_refusal(rebot, q_rec, snapshot=WHOLE)
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
    assert out["L1_source"] == "model" and out["model_mismatch"] is False
    assert out["L4"] == out["old"]              # no override volume: the shipped one
    assert out["partial_map"] is True and out["replayable"] is True and out["approximate"] == []
    summary = rc.summarize([out])
    assert summary == {"old": buckets(refused=1), "L1": buckets(allowed=1), "L4": buckets(refused=1),
                       "model_mismatch": 0}


@needs_pin
def test_an_admission_the_snapshot_cannot_certify_reads_undecidable(rc, rebot):
    """The same refusal with the SHIPPED snapshot (5 cm, 64 voxels, capped
    here): the step retreats from every voxel the snapshot holds, but a voxel
    it left out (no nearer than its farthest kept one, minus the 1 mm
    rounding) could sit at the shoulder. Never counted as allowed."""
    rep = vet_pose_refusal(rebot, at_x(rebot, X_REC))
    assert rep["snapshot"] == {"centre": rep["sample"], "radius_m": 0.05, "max_voxels": 64}
    assert len(rep["voxels"]) == 64
    rep["q_prev"] = at_x(rebot, X_IN).tolist()
    out = rc.replay(rep, arm="rebot_rs")
    assert out["L1"].startswith("undecidable: the snapshot admits it")
    assert out["old"].startswith("refused: point 0 clearance")
    far = max(v[3] for v in rep["voxels"])
    assert out["certified"] == {"centre": rep["sample"], "radius_m": pytest.approx(far - 0.001),
                                "bounds": "recorded"}
    assert rc.summarize([out])["L1"] == buckets(undecidable=1)


@needs_pin
@pytest.mark.parametrize("x, dy, dz", [(X_REC, 0.0, 0.0), (X_TAKE, 0.0037, -0.0042), (0.4467, -0.0081, 0.0066)])
def test_the_replayed_clearance_is_the_live_grid_read_of_the_snapshot(rc, rebot, x, dy, dz):
    """Exactness: the live gate reads the EDT grid trilinearly; the replay
    rebuilds that grid from the snapshot's centres and reads it the same way
    (a cube distance would read about half a voxel less)."""
    q = at_tcp(rebot, x, dy, dz)
    rep = vet_pose_refusal(rebot, q)
    out = rc.replay(rep, arm="rebot_rs")
    assert out["clearance_m"] == rep["clearance_m"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)
    assert out["old"].startswith("refused: point 0 clearance") and same_words(out["old"], "refused: " + rep["reason"])


@needs_pin
def test_a_streamed_closing_step_replays_l1_as_recorded(rc, rebot):
    """R6 second row: a real approve refusal (the step closes in from outside
    the margin) keeps its recorded verdict; the model agrees with it."""
    q_out, q_rec = at_x(rebot, X_OUT), at_x(rebot, X_REC)
    rep = approve_refusal(rebot, q_out, q_rec)
    assert rep["phase"] == "approve" and rep["q_prev"] == q_out.tolist()
    assert "inside the margin" not in rep["reason"]
    out = rc.replay(rep, arm="rebot_rs")
    assert out["L1"] == "refused: " + rep["reason"] and out["L1_source"] == "recorded"
    assert out["L1_model"].startswith("refused: point 0 clearance") and out["model_mismatch"] is False
    assert out["old"].startswith("refused: point 0 clearance")
    assert out["phase"] == "approve" and out["gate"] == "tcp" and out["recorded"] == rep["reason"]


@needs_pin
def test_a_step_the_live_rule_refused_keeps_its_recorded_verdict(rc, rebot):
    """A streamed or preflight refusal was decided by the step rule itself on
    the whole map: L1 is that verdict, whatever the snapshot model says. A
    model that admits the step (here: a q_prev that retreats in the model) is
    a model mismatch. Any phase but the four gate phases is not replayable."""
    rep = approve_refusal(rebot, at_x(rebot, X_OUT), at_x(rebot, X_REC), snapshot=WHOLE)
    rep["q_prev"] = at_x(rebot, X_IN).tolist()
    for phase in ("approve", "vet_step"):
        out = rc.replay({**rep, "phase": phase}, arm="rebot_rs")
        assert out["L1"] == "refused: " + rep["reason"] and out["L1_source"] == "recorded"
        assert out["L1_model"] == "allowed" and out["model_mismatch"] is True
        assert rc.summarize([out])["model_mismatch"] == 1
    # a static refusal given an earlier q_prev (a pre-PR-A trace) is computed
    static = rc.replay({**rep, "phase": "vet_pose"}, arm="rebot_rs")
    assert static["L1"] == "allowed" and static["L1_source"] == "model" and static["model_mismatch"] is False
    # fail closed: a phase the harness never writes never reaches the model
    for phase in ("Approve", "halt", None):
        bad = rc.replay({**rep, "phase": phase}, arm="rebot_rs")
        assert bad["replayable"] is False and "phase" in bad["not_replayable"], bad
        assert all(bad[lever].startswith("refused: not replayable") for lever in rc.LEVERS)
    missing = {k: v for k, v in rep.items() if k != "phase"}
    assert rc.replay(missing, arm="rebot_rs")["replayable"] is False


@needs_pin
def test_the_cli_prints_one_line_per_report_and_the_summary(rc, rebot, tmp_path, capsys):
    q_rec, q_in, q_out = at_x(rebot, X_REC), at_x(rebot, X_IN), at_x(rebot, X_OUT)
    first = vet_pose_refusal(rebot, q_rec, snapshot=WHOLE)
    first["q_prev"] = q_in.tolist()             # emulates a pre-PR-A trace (R6)
    second = approve_refusal(rebot, q_out, q_rec)
    trace = write_trace(tmp_path / "mcp_7" / "trace.jsonl",
                        row(0, "move_relative", [first]),
                        {"step": 1, "skill": "list_objects", "context": {"arm": "default"}, "result": {"ok": True}},
                        "\n",
                        row(2, "move_home", [second]))
    assert rc.main([str(trace.parent), "--arm", "rebot_rs"]) == 0
    out, err = capsys.readouterr()
    lines = [json.loads(line) for line in out.splitlines()]
    assert [(r["step"], r["skill"], r["L1"] == "allowed") for r in lines[:-1]] == [
        (0, "move_relative", True), (2, "move_home", False)]
    assert lines[-1] == {"old": buckets(refused=2), "L1": buckets(allowed=1, refused=1),
                         "L4": buckets(refused=2), "model_mismatch": 0}
    assert "2 report(s): 2 replayable" in err and "0 refusal row(s) without context.clearance" in err
    assert "L1/L2 are measured by the live acceptance (spec §7 A1), not by this replay" in err


# ── reading traces: robustness (fail loud, never stop the run) ──────────────


def test_load_reports_reads_every_clearance_entry_with_its_row(rc, tmp_path):
    a, b, c = {"gate": "tcp", "q": [0.0]}, {"gate": "jaws", "q": [1.0]}, {"gate": "link 2", "q": [2.0]}
    one = write_trace(tmp_path / "mcp_a" / "trace.jsonl",
                      row(3, "grasp_object", [a, b]), "\n", "{broken\n", "[1, 2]\n",
                      {"step": 4, "skill": "describe_scene", "context": None},
                      {"step": 5, "skill": "move_home", "context": {"clearance": "not a list"}},
                      row(6, "move_home", ["not a report", c]))
    two = write_trace(tmp_path / "mcp_b" / "trace.jsonl", row(0, "move_relative", [a]))
    counts = {}
    got = rc.load_reports([one, two.parent], counts)
    assert [(r["run"], r["step"], r["skill"], r.get("gate")) for r in got] == [
        ("mcp_a", 3, "grasp_object", "tcp"), ("mcp_a", 3, "grasp_object", "jaws"),
        ("mcp_a", 6, "move_home", None), ("mcp_a", 6, "move_home", "link 2"), ("mcp_b", 0, "move_relative", "tcp")]
    assert got[1]["q"] == [1.0]
    assert got[2]["malformed"] == "report entry is not a mapping: 'not a report'"
    # a broken line, a non-object line and an unreadable clearance field
    assert counts == {"skipped_lines": 3, "malformed_entries": 1, "no_snapshot": 0, "lost": 0}
    assert rc.load_reports([two]) == [{**a, "run": "mcp_b", "step": 0, "skill": "move_relative"}]


@needs_pin
def test_unreadable_input_is_counted_and_exits_non_zero_without_stopping_the_run(rc, rebot, tmp_path, capsys):
    good = approve_refusal(rebot, at_x(rebot, X_OUT), at_x(rebot, X_REC))
    trace = write_trace(tmp_path / "mcp_5" / "trace.jsonl",
                        "not json\n", "[1, 2]\n",
                        {"step": 1, "skill": "move_home", "context": {"clearance": "not a list"}},
                        row(2, "grasp_object", ["not a report", {**good, "exemption": [0.3, 0.0, 0.05, 0.0]}, good]))
    assert rc.main([str(trace)]) == 1
    out, err = capsys.readouterr()
    *lines, summary = [json.loads(line) for line in out.splitlines()]
    assert [r["replayable"] for r in lines] == [False, False, True]
    assert lines[0]["not_replayable"] == "report entry is not a mapping: 'not a report'"
    assert lines[1]["not_replayable"].startswith("exemption is not a mapping")
    assert lines[2]["L1"] == "refused: " + good["reason"]
    assert summary["old"] == buckets(refused=1, not_replayable=2)
    assert "3 unreadable line(s)" in err and "1 malformed report entr" in err


@needs_pin
def test_a_report_whose_replay_raises_never_stops_the_run(rc, rebot, tmp_path, capsys, monkeypatch):
    good = approve_refusal(rebot, at_x(rebot, X_OUT), at_x(rebot, X_REC))
    trace = write_trace(tmp_path / "mcp_6" / "trace.jsonl", row(0, "move_home", [good, good]))
    real, calls = rc.replay, []

    def first_one_raises(report, **kwargs):
        calls.append(report)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(report, **kwargs)

    monkeypatch.setattr(rc, "replay", first_one_raises)
    assert rc.main([str(trace)]) == 1
    out, err = capsys.readouterr()
    *lines, summary = [json.loads(line) for line in out.splitlines()]
    assert lines[0]["replayable"] is False and lines[0]["not_replayable"] == "replay failed: RuntimeError: boom"
    assert lines[0]["old"].startswith("refused: not replayable") and lines[0]["run"] == "mcp_6"
    assert lines[1]["replayable"] is True and len(calls) == 2
    assert summary["L1"] == buckets(refused=1, not_replayable=1)
    assert "1 report(s) whose replay raised" in err


@needs_pin
def test_refusal_rows_without_a_snapshot_are_counted(rc, rebot, tmp_path, capsys):
    """Spec §7 A2: every refusal and halt must carry its snapshot. A failed
    row whose error names a map clearance gate but carries no
    `context.clearance` (a pre-A4 trace, a lost report) is `no_snapshot`."""
    good = approve_refusal(rebot, at_x(rebot, X_OUT), at_x(rebot, X_REC))

    def failed(step, error, context):
        return {"step": step, "skill": "move_home", "context": context, "result": {"ok": False, "error": error}}

    trace = write_trace(
        tmp_path / "mcp_8" / "trace.jsonl",
        failed(0, "approach unsafe: point 0 clearance 0.020000000 m below 0.030000000 m (shortfall 10.000000 mm; "
                  "occupancy map) at [0.4000, 0.0000, 0.3000]", {"arm": "default"}),
        failed(1, "SafetyViolation: required occupancy unavailable: distance grid is stale (2.00 s old, "
                  "max_age_s 1.5)", None),
        failed(2, "wrist camera clearance 0.012 m below 0.030 m (occupancy map) at [0.4, 0.0, 0.5]",
               {"clearance": [], "clearance_lost": 1}),                 # a report the harness lost
        failed(3, "no reachable grasp: IK failed", {}),
        {"step": 4, "skill": "move_home", "context": {}, "result": {"ok": True}},
        row(5, "move_home", [good], {"ok": False, "error": good["reason"]}))
    assert rc.main([str(trace)]) == 0
    out, err = capsys.readouterr()
    summary = json.loads(out.splitlines()[-1])
    assert summary["old"] == summary["L4"] == buckets(refused=1, no_snapshot=3)
    assert "3 refusal row(s) without context.clearance" in err


def test_summarize_fails_closed(rc):
    rows = [{"replayable": True, "old": "something else", "L1": "allowed", "L4": "undecidable: x"},
            {"old": "allowed", "L1": "allowed", "L4": "allowed"}]
    assert rc.summarize(rows, no_snapshot=2) == {
        "old": buckets(refused=1, not_replayable=1, no_snapshot=2),
        "L1": buckets(allowed=1, not_replayable=1, no_snapshot=2),
        "L4": buckets(undecidable=1, not_replayable=1, no_snapshot=2), "model_mismatch": 0}


# ── the other gates ─────────────────────────────────────────────────────────


@needs_pin
def test_a_wrist_camera_refusal_replays_and_lever_4_takes_the_override_volume(rc, rebot, tmp_path, capsys):
    """One voxel inside today's slab, in front of the cut camera box and above
    the fingers: the shipped volume refuses it, the rebuilt one (lever 4) not;
    certified only by a snapshot that rules out every other voxel."""
    q_rec = at_x(rebot, X_REC)
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    centre = lattice(tcp + [-0.015, 0.0, 0.07])
    reports = {}
    for name, snapshot in (("whole", WHOLE), ("shipped", None)):
        h, _ = recorder(rebot, voxel_box(centre), snapshot=snapshot)
        reason = h.vet_pose(q_rec)
        assert reason is not None and reason.startswith("wrist camera clearance"), reason
        (rep,) = h.drain_clearance_reports()
        reports[name] = json.loads(json.dumps(rep))
    rep = reports["whole"]
    assert rep["gate"] == "wrist camera" and len(rep["voxels"]) == 1

    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: wrist camera clearance") and out["L4"] == out["old"]
    # the gate re-measures a tool sample below the margin to the voxel centre
    # minus half a voxel; so does the replay
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)
    parts = rc.tool_volume_from_override("rebot_rs", PARTS)
    assert parts.names == ("wrist camera", "housing", "fingers", "jaws")
    out4 = rc.replay(rep, arm="rebot_rs", tool_volume_override=parts)
    assert out4["L4"] == "allowed" and out4["old"] == out["old"]
    assert out4["L4_volume"] == "override" and out["L4_volume"] == "shipped"
    # the shipped snapshot cannot rule out a voxel by the housing or the fingers
    shipped = rc.replay(reports["shipped"], arm="rebot_rs", tool_volume_override=parts)
    assert shipped["L4"].startswith("undecidable: the snapshot admits it")
    # the same through the CLI
    trace = write_trace(tmp_path / "mcp_9" / "trace.jsonl", row(1, "grasp_object", [rep]))
    override = tmp_path / "parts.json"
    override.write_text(json.dumps(PARTS))
    assert rc.main([str(trace), "--tool-volume", str(override)]) == 0
    *lines, summary = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["L4"] == "allowed"
    assert summary == {"old": buckets(refused=1), "L1": buckets(refused=1), "L4": buckets(allowed=1),
                       "model_mismatch": 0}


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
    # a slab whose top face sits EPS under the voxel boundary below the held sample
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
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)


@needs_pin
def test_a_ceiling_on_the_recorded_exemption_is_kept(rc, rebot):
    """A refusal under the air-grasp exit's capped column (A3: z_max): the TCP
    sits ABOVE the ceiling, inside the cylinder, so it is checked; without the
    ceiling the replay would exempt it and call the pose allowed."""
    q_out, q_rec = at_x(rebot, X_OUT), at_x(rebot, X_REC)
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    column = (tcp[:2], 0.05, 0.30)
    rep = approve_refusal(rebot, q_out, q_rec, ahead_box(rebot, half_y=0.02), exemption=column,
                          z_max=float(tcp[2]) - 0.02, snapshot=WHOLE)
    assert len(rep["exemption"]["cylinder"]) == 5
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: point 0 clearance")
    assert out["L1"] == "refused: " + rep["reason"] and out["approximate"] == []
    # the step itself, replayed (as a static refusal given its q_prev), keeps the ceiling too
    assert rc.replay({**rep, "phase": "vet_pose"}, arm="rebot_rs")["L1"].startswith("refused: point 0 clearance")
    # the same report without its ceiling is exempt (the cylinder holds the TCP)
    unbounded = copy.deepcopy(rep)
    unbounded["exemption"]["cylinder"] = unbounded["exemption"]["cylinder"][:4]
    loose = rc.replay(unbounded, arm="rebot_rs")
    assert loose["old"] == "allowed" and loose["model_mismatch"] is True
    # a recorded STATIC refusal the model admits is a model mismatch too
    static = rc.replay({**unbounded, "phase": "vet_pose", "q_prev": None}, arm="rebot_rs")
    assert static["old"] == static["L1"] == "allowed" and static["model_mismatch"] is True
    # and the step is replayed under that exemption: the column admits it
    assert rc.replay({**unbounded, "phase": "vet_pose"}, arm="rebot_rs")["L1"] == "allowed"


@needs_pin
def test_a_tool_part_clearance_is_read_as_its_gate_reads_it(rc):
    """The replayed clearance of a tool part below the margin is re-measured
    as the gate does: to the nearest centre that is not in the table band nor
    in the exemption column, minus half a voxel. A column under an `episode`
    key the replay cannot rebuild is said so."""
    model = rc._arm_config("rebot_rs")
    assert 0.005 <= model.limits.table_z + model.volume.table_band_m < 0.035      # the band holds the table voxel
    report = {"phase": "vet_pose", "gate": "jaws", "reason": "jaws clearance 0.025 m below 0.030 m",
              "q": [0.0] * 6, "q_prev": None, "sample": [0.305, 0.005, 0.025], "sample_tcp": [0.0, 0.0, 0.0],
              "clearance_m": 0.025, "margin_m": 0.03,
              "exemption": {"cylinder": [0.305, 0.005, 0.004, 0.04], "episode": {"keys": 3}},
              "map": {"refreshes": 1, "age_s": 0.1, "voxel_m": 0.01, "occupied": 3},
              "voxels": [[0.305, 0.005, 0.005, 0.02],      # the table, 2 cm under the sample
                         [0.305, 0.005, 0.045, 0.02],      # inside the exemption column, 2 cm above
                         [0.335, 0.005, 0.035, 0.032]]}    # the neighbour
    out = rc.replay(report, arm="rebot_rs")
    # to the neighbour's centre as the live cache holds it (float32)
    neighbour = np.asarray([0.335, 0.005, 0.035], dtype=np.float32).astype(float)
    expected = float(np.linalg.norm(np.asarray(report["sample"]) - neighbour)) - V / 2
    assert out["replay_clearance_m"] == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(np.hypot(0.03, 0.01) - V / 2, abs=1e-8)
    assert out["approximate"] == ["exemption keys not replayed: ['episode']"]
    assert out["certified"]["bounds"] == "profile"     # no recorded bounds: the profile's


@needs_pin
def test_the_recorded_base_pose_places_the_tool_volume(rc, rebot):
    """An arm bolted 20 cm along y: its tool volume is read in the table frame
    (base_pose @), so the replay must lift it the same way."""
    q_rec = at_x(rebot, X_REC)
    base_pose = np.eye(4)
    base_pose[1, 3] = 0.20
    tcp = rebot.kin.fk(q_rec)[:3, 3]
    centre = lattice(tcp + [-0.015, 0.20, 0.07])
    h, _ = recorder(rebot, voxel_box(centre), base_pose=base_pose, snapshot=WHOLE)
    reason = h.vet_pose(q_rec)
    assert reason is not None and reason.startswith("wrist camera clearance"), reason
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["base_pose"] == base_pose.tolist()
    out = rc.replay(rep, arm="rebot_rs")
    assert out["old"].startswith("refused: wrist camera clearance")
    rep.pop("base_pose")
    assert rc.replay(rep, arm="rebot_rs")["old"] == "allowed"


@needs_pin
def test_a_box_distance_map_is_not_recoverable_and_the_replay_stays_conservative(rc, rebot):
    """GridBridge (a test fake) measures exact BOX distances, which voxel
    centres cannot give back: the rebuilt grid reads farther. A recorded
    refusal the model then admits never counts as admitted: L1 keeps the
    recorded verdict, an admitting `old` is a model mismatch, and with the
    shipped snapshot it reads undecidable."""
    x_far, x_rec = FACE_X + EPS - 0.0395, FACE_X + EPS - 0.0295        # box distance 3.95 / 2.95 cm
    q_far, q_rec = at_x(rebot, x_far), at_x(rebot, x_rec)
    for snapshot, verdict in ((WHOLE, "allowed"), (None, "undecidable")):
        rep = approve_refusal(rebot, q_far, q_rec, bridge=GridBridge, snapshot=snapshot)
        assert rep["clearance_m"] == pytest.approx(0.0295, abs=1e-6)
        out = rc.replay(rep, arm="rebot_rs")
        assert out["replay_clearance_m"] > 0.03 > rep["clearance_m"]       # the centres' grid reads farther
        assert out["L1"] == "refused: " + rep["reason"] and out["L1_source"] == "recorded"
        assert out["old"].startswith(verdict), out["old"]
        assert out["model_mismatch"] is (verdict == "allowed")


# ── what cannot be replayed exactly ─────────────────────────────────────────


@needs_pin
def test_a_map_that_raised_is_not_replayable(rc, rebot):
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
    assert rc.summarize([out]) == {**{lever: buckets(not_replayable=1) for lever in ("old", "L1", "L4")},
                                   "model_mismatch": 0}


def test_reports_without_a_measured_sample_are_not_replayable(rc):
    base = {"phase": "vet_pose", "gate": "tcp", "reason": "r", "q": [0.0] * 6, "q_prev": None,
            "sample": [0.4, 0.0, 0.4], "clearance_m": 0.02, "exemption": None,
            "map": {"refreshes": 1, "age_s": 0.1, "voxel_m": 0.01, "occupied": 2},
            "voxels": [[0.425, 0.005, 0.405, 0.025], [0.435, 0.005, 0.405, 0.035]], "margin_m": 0.03}
    cases = {
        "no offending sample": {"sample": None, "clearance_m": None, "voxels": [], "gate": "arm"},
        "unobserved": {"clearance_m": None},
        "no voxels": {"voxels": [], "snapshot_error": "RuntimeError: boom"},
        "voxel size": {"map": None},
        "map record": {"map": "not a record"},
        "joints": {"q": [0.0] * 5},
        "q_prev": {"q_prev": [float("nan")] * 6},
        "bad sample": {"sample": [0.4, 0.0]},
        "sample_tcp": {"gate": "attached", "sample_tcp": None},
        "exemption cylinder": {"exemption": {"cylinder": [0.3, 0.0, 0.05]}},
        "exemption is not a mapping": {"exemption": [0.3, 0.0, 0.05, 0.0]},
        "bad voxels": {"voxels": [[0.425, 0.005]]},
        # a NaN `d` would leave a capped snapshot's certified radius uncapped
        "finite [x, y, z, d]": {"voxels": [[0.425, 0.005, 0.405, float("nan")], [0.435, 0.005, 0.405, 0.035]]},
        "phase": {"phase": "Approve"},
        "margin_m": {"margin_m": 0.02},
        "lattice": {"map": {"refreshes": 1, "age_s": 0.1, "voxel_m": 0.02, "occupied": 2}},
        "snapshot record": {"snapshot": {"centre": [0.4, 0.0, 0.4], "radius_m": -1.0, "max_voxels": 64}},
    }
    for why, change in cases.items():
        out = rc.replay({**base, **change}, arm="rebot_rs")
        assert out["replayable"] is False and why in out.get("not_replayable", ""), (why, out)
        assert out["old"] == out["L1"] == out["L4"] and out["old"].startswith("refused: not replayable")
    assert rc.replay({**base, "margin_m": None}, arm="rebot_rs")["not_replayable"].startswith("margin_m")
    garbage = rc.replay("garbage", arm="rebot_rs")
    assert garbage["replayable"] is False and garbage["not_replayable"].startswith("report entry is not a mapping")


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
def test_a_halt_that_took_over_a_report_is_certified_around_its_snapshot_centre(rc, rebot):
    """A streamed halt in the same 1 cm cell rewrites the report's sample but
    keeps the first snapshot, taken around the FIRST sample (its recorded
    `snapshot.centre`). The replay reads the grid at the new sample and
    certifies around the recorded centre. A report from before the centre was
    recorded is certified around its sample with the radius shrunk by the
    1 cm dedupe cell's diagonal (any approve report: the takeover heuristic
    can miss one)."""
    q_rec, q_out, q_take = at_x(rebot, X_REC), at_x(rebot, X_OUT), at_x(rebot, X_TAKE)
    h, _ = recorder(rebot, ahead_box(rebot))
    assert h.vet_pose(q_rec) is not None
    h.heartbeat()
    with pytest.raises(SafetyViolation):
        h.approve(q_out, q_take, 1.0)
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    assert rep["phase"] == "approve" and rep["sample"][0] == pytest.approx(X_TAKE, abs=1e-7)
    assert rep["snapshot"]["centre"][0] == pytest.approx(X_REC, abs=1e-7)
    out = rc.replay(rep, arm="rebot_rs")
    assert out["approximate"] == ["takeover"]
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-6)
    assert out["L1"] == "refused: " + rep["reason"]
    radius = min(0.05, max(v[3] for v in rep["voxels"]) - 0.001)       # capped: the farthest kept one
    assert out["certified"] == {"centre": rep["snapshot"]["centre"], "radius_m": pytest.approx(radius),
                                "bounds": "recorded"}
    old_style = {k: v for k, v in rep.items() if k != "snapshot"}
    out = rc.replay(old_style, arm="rebot_rs")
    assert out["approximate"] == ["takeover"]
    assert out["certified"] == {"centre": rep["sample"], "radius_m": pytest.approx(radius - 0.01 * np.sqrt(3)),
                                "bounds": "profile"}
    # no takeover the heuristic can see, still an approve report: shrunk; a static one: not
    plain = approve_refusal(rebot, q_out, q_rec)
    plain.pop("snapshot")
    far = min(0.05, max(v[3] for v in plain["voxels"]) - 0.001)
    out = rc.replay(plain, arm="rebot_rs")
    assert out["approximate"] == [] and out["certified"]["radius_m"] == pytest.approx(far - 0.01 * np.sqrt(3))
    out = rc.replay({**plain, "phase": "vet_pose", "q_prev": None}, arm="rebot_rs")
    assert out["certified"]["radius_m"] == pytest.approx(far)


@needs_pin
def test_the_replay_never_connects_and_never_changes_its_input(rc, rebot, tmp_path, monkeypatch, capsys):
    rep = vet_pose_refusal(rebot, at_x(rebot, X_REC), snapshot=WHOLE)
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
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["L1"] == buckets(allowed=1)


# ── vet_pose parity (ruling 4: `_static_reason` copies vet_pose's map gates) ──


def _near_box(rebot, q, rng):
    T = rebot.kin.fk(q)
    cands = [T[:3, 3], *rebot.kin.link_positions(q)[1:], *rebot.volume.points(T)[::7]]
    p = cands[int(rng.integers(len(cands)))]
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    size = rng.uniform(0.01, 0.06, 3)
    centre = p + direction * (rng.uniform(0.005, 0.04) + np.abs(direction) @ size / 2)
    return tuple(centre - size / 2), tuple(centre + size / 2)


@needs_pin
def test_the_static_replay_check_is_vet_poses_map_verdict(rc, rebot):
    """Random poses x exemption x held payload on real maps: wherever
    vet_pose reached its map gates, `_static_reason` (what `old` and L4 run)
    gives the same verdict and text; where it passed them, None. Any edit of
    vet_pose's map gates must update the replay (global constraints, B4-B6)."""
    rng = np.random.default_rng(20261017)
    limits = SafetyLimits.from_config(rebot.cfg.safety)
    seen = Counter()
    for scene in range(30):
        q0 = (rebot.home, rebot.q_pre, rebot.q_grasp)[scene % 3] + rng.normal(0, 0.1, 6)
        boxes = [_near_box(rebot, q0, rng) for _ in range(int(rng.integers(1, 3)))]
        if scene % 3 == 2:
            boxes.append(TABLE)
        occ = live_map(*boxes)
        if scene % 2:
            occ._payload_samples["scene"] = np.array([[0.0, 0.0, -0.1], [0.02, 0.0, -0.08]])
        h = SafetyHarness(limits, kinematics=rebot.kin, occupancy=occ, tool_volume=rebot.volume)
        exempt = None if scene % 4 < 2 else (OBJ[:2].copy(), 0.07, -0.06 if scene % 8 < 4 else 0.0)
        for _ in range(6):
            q = q0 + rng.normal(0, 0.01, 6)
            reason = h.vet_pose(q, *(() if exempt is None else exempt))
            filed = h.drain_clearance_reports()
            static = rc._static_reason(h, q, exempt)
            assert h.drain_clearance_reports() == []                 # the replay's check files nothing
            if filed:
                assert static == reason
                seen["map refusal " + filed[0]["gate"].split(" ")[0]] += 1
            elif reason is None:
                assert static is None
                seen["clear"] += 1
            else:
                seen["refused before the map"] += 1
    assert sum(n for k, n in seen.items() if k.startswith("map refusal")) >= 30 and seen["clear"] >= 10, seen
    assert {"map refusal tcp", "map refusal attached"} <= set(seen), seen


# ── ReplayMap ───────────────────────────────────────────────────────────────


def _corner_blend(point, value):
    """Trilinear blend of value(node) over the lattice cell (x.xx5) holding point."""
    p = np.asarray(point, dtype=float)
    lo = np.floor((p - V / 2) / V) * V + V / 2
    t = (p - lo) / V
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1]) * (t[2] if dz else 1 - t[2])
                out += w * value(lo + V * np.array([dx, dy, dz]))
    return out


def test_replay_map_reads_the_live_grid_rebuilt_from_its_voxels(rc):
    c = np.array([0.005, 0.005, 0.005])
    m = rc.ReplayMap(np.array([[*c, 0.123]]), V)
    pts = np.array([[0.035, 0.005, 0.005],         # a node 3 cm off: exact
                    [0.030, 0.005, 0.005],         # between nodes 2 and 3 cm off: 2.5 (a cube: 2.0)
                    [0.020, 0.020, 0.020],         # a cell off the diagonal
                    [0.007, 0.004, 0.009]])        # inside the voxel's cell
    expected = [_corner_blend(p, lambda n: np.linalg.norm(n - c)) for p in pts]
    np.testing.assert_allclose(m.clearance(pts), expected, atol=1e-12)
    np.testing.assert_allclose(m.clearance(pts)[:2], [0.03, 0.025], atol=1e-12)
    np.testing.assert_allclose(m.payload_clearance(pts), m.clearance(pts))
    assert m.required is False and m.unknown_blocks is False
    assert m.status == {"voxel": V} and m.refreshes == 0
    empty = rc.ReplayMap(np.zeros((0, 4)), V)
    assert np.all(np.isinf(empty.clearance(pts))) and empty.nearby_occupied(pts[0], 0.05, 8).shape == (0, 4)
    for bad in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            rc.ReplayMap(np.zeros((0, 4)), bad)
    with pytest.raises(ValueError):
        rc.ReplayMap(np.array([[0.0, float("nan"), 0.0, 0.0]]), V)


def test_a_certified_replay_map_never_reads_past_what_its_snapshot_rules_out(rc):
    """With a certificate (centre, R) every node reads at most the least
    distance to a voxel the snapshot left out, R - |node - centre| (never
    below 0): the lower bound of the live read."""
    c = np.array([0.005, 0.005, 0.005])
    centre, radius = np.array([0.0, 0.0, 0.0]), 0.05
    m = rc.ReplayMap(np.array([[*c, 0.0]]), V, certified=(centre, radius))
    pts = np.array([[0.035, 0.005, 0.005], [0.030, 0.012, 0.005], [0.105, 0.005, 0.005], [0.3, 0.3, 0.3]])
    expected = [_corner_blend(p, lambda n: min(np.linalg.norm(n - c), max(radius - np.linalg.norm(n - centre), 0.0)))
                for p in pts]
    np.testing.assert_allclose(m.clearance(pts), expected, atol=1e-12)
    assert m.clearance(pts)[3] == 0.0                                       # far beyond the ball: anything
    plain = rc.ReplayMap(np.array([[*c, 0.0]]), V)
    assert np.all(m.clearance(pts) <= plain.clearance(pts)) and m.clearance(pts)[2] < plain.clearance(pts)[2]


def test_replay_map_tool_reads_exclude_the_table_band_and_the_target_column(rc):
    """`clearance_excluding` as OccupancyMap's: the grid read, and below
    `within` the nearest centre outside the band and the column (ceiling
    included) minus half a voxel, or `within` when none is in reach."""
    table, target, high, neighbour = [0.105, 0.005, 0.005], [0.205, 0.005, 0.105], [0.205, 0.005, 0.305], \
        [0.305, 0.005, 0.105]
    m = rc.ReplayMap(np.array([table, target, high, neighbour]), V)
    probe = np.array([[0.105, 0.005, 0.03], [0.205, 0.005, 0.13], [0.205, 0.005, 0.28], [0.305, 0.005, 0.13],
                      [0.405, 0.005, 0.13]])
    every = m.clearance(probe)
    np.testing.assert_allclose(every[:4], [0.025] * 4, atol=1e-12)
    column = (np.array([0.205, 0.005]), 0.03, 0.05)
    got = m.clearance_excluding(probe, below_z=0.02, cylinder=column, within=0.03)
    # the table voxel and the whole column go (nothing else in reach: `within`);
    # the neighbour stays, re-measured to its centre (float32, as the live
    # cache holds it) minus half a voxel
    def f32(z):
        return float(np.float32(z))

    np.testing.assert_allclose(got[:4], [0.03, 0.03, 0.03, 0.13 - f32(0.105) - V / 2], rtol=0, atol=1e-12)
    assert got[4] == every[4] > 0.03                                        # not below `within`: the grid read
    capped = m.clearance_excluding(probe, below_z=0.02, cylinder=(*column, 0.2), within=0.03)
    assert capped[2] == pytest.approx(f32(0.305) - 0.28 - V / 2, abs=1e-12)  # above the ceiling: counts again
    assert capped[1] == pytest.approx(0.03, abs=1e-12)
    np.testing.assert_allclose(m.clearance_excluding(probe, within=0.03), every)
    np.testing.assert_allclose(m.clearance_excluding(probe, below_z=0.02, cylinder=column), every)
    # a certificate bounds the re-measure too: a voxel left out could be nearer
    certified = rc.ReplayMap(np.array([table, target, high, neighbour]), V,
                             certified=(np.array([0.105, 0.005, 0.03]), 0.004))
    assert certified.clearance_excluding(probe[:1], below_z=0.02, cylinder=column, within=0.03)[0] == 0.0


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


# ── pre-final: the replay mirrors the live read at its edges (A6 review) ────


def _region_map(*boxes, region_max_x=0.45, snapshot=WHOLE):
    """A live map whose region ends at x = region_max_x (last voxel centre
    half a voxel inside it): the reBot's TCP at X_REC lies within v/2 of that
    face, where the live read clamps to the border node."""
    occ = OccupancyMap(client=EdtBridge(list(boxes), voxel=V), region_min=np.array([-0.30, -0.45, -0.02]),
                       region_max=np.array([region_max_x, 0.45, 0.60]), required=True, unobserved="free")
    occ.probe()
    occ.refusal_snapshot = snapshot
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None and occ.usable()
    return occ


@needs_pin
def test_a_sample_in_the_region_clamp_band_replays_the_live_read(rc, rebot):
    """Live, `_sample_grid` clamps a point within v/2 outside the border node
    centres to the border node; an unclamped replay read 3.07 cm where the
    live gate refused at 2.67 cm. The report records the grid (origin, shape)
    and the replay mirrors the clamp."""
    q = at_x(rebot, X_REC)
    occ = _region_map(voxel_box([0.425, 0.015, 0.445]))
    h = SafetyHarness(SafetyLimits.from_config(rebot.cfg.safety), kinematics=rebot.kin, occupancy=occ,
                      tool_volume=rebot.volume)
    assert h.vet_pose(q)
    rep = json.loads(json.dumps([r for r in h.drain_clearance_reports() if r["gate"] == "tcp"][-1]))
    assert rep["clearance_m"] < 0.03
    out = rc.replay(rep, arm="rebot_rs")
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-9)


@needs_pin
def test_a_node_exactly_at_the_margin_reads_as_the_live_float32_grid(rc, rebot):
    """The live grid holds float32 node values (float32(0.03) < 0.03: refused)
    on a float32 origin; a float64 replay read 0.0300000004 for a TCP the live
    gate refused at 0.0299999953."""
    home = rebot.kin.fk(rebot.home)[:3, 3]
    node = np.array([0.445, -0.005, 0.435])
    q = at_tcp(rebot, node[0], dy=node[1] - home[1], dz=node[2] - home[2])
    h, _ = recorder(rebot, voxel_box(node + [0.0, -0.03, 0.0]), snapshot=WHOLE)
    assert h.vet_pose(q)
    rep = json.loads(json.dumps(h.drain_clearance_reports()[-1]))
    assert rep["gate"] == "tcp" and rep["clearance_m"] < 0.03
    out = rc.replay(rep, arm="rebot_rs")
    assert out["replay_clearance_m"] == pytest.approx(rep["clearance_m"], abs=1e-12)
    assert out["replay_clearance_m"] < 0.03


def test_the_replay_read_is_the_live_read_inside_and_at_the_region_faces(rc):
    """Same voxels, same grid record: ReplayMap.clearance == OccupancyMap.clearance
    for points inside the region, in the v/2 clamp band at every face and
    beyond it (+inf), to float32 node precision."""
    rng = np.random.default_rng(7)
    boxes = [voxel_box(lattice(c)) for c in rng.uniform([0.38, -0.1, 0.3], [0.44, 0.1, 0.5], (6, 3))]
    occ = _region_map(*boxes, region_max_x=0.45)
    rec = occ.clearance_record()
    rows = occ.nearby_occupied(np.array([0.41, 0.0, 0.4]), 1.0, 4096)
    replay = rc.ReplayMap(np.round(rows, 3), V, grid=(rec["grid_origin"], rec["grid_shape"]))
    lo = np.asarray(rec["grid_origin"], dtype=float) - V / 2
    hi = lo + np.asarray(rec["grid_shape"]) * V
    pts = np.vstack([rng.uniform([0.36, -0.12, 0.28], [0.45, 0.12, 0.52], (400, 3)),
                     np.column_stack([rng.uniform(hi[0] - V, hi[0] + V, 200),                # the +x face band
                                      rng.uniform(-0.12, 0.12, 200), rng.uniform(0.28, 0.52, 200)]),
                     np.column_stack([rng.uniform(0.36, 0.45, 200), rng.uniform(-0.12, 0.12, 200),
                                      rng.uniform(lo[2] - V, lo[2] + V, 200)])])             # the -z face band
    live, mirrored = occ.clearance(pts), replay.clearance(pts)
    assert np.array_equal(np.isinf(live), np.isinf(mirrored)) and np.isinf(live).sum() > 20
    np.testing.assert_allclose(mirrored[np.isfinite(live)], live[np.isfinite(live)], rtol=0, atol=1e-9)


def test_band_and_cylinder_exclusions_compare_float32_centres_as_live(rc):
    """The live cache holds float32 voxel centres: float32(0.07) > 0.07, so a
    centre at z 0.07 is ABOVE a band top of 0.07 (an obstacle), and a centre
    at x 0.07 lies OUTSIDE a 0.07 m cylinder (not exempt). A float64 replay
    dropped both (fail-open)."""
    m = rc.ReplayMap(np.array([[0.07, 0.0, 0.07]]), V)
    sample = np.array([[0.07, 0.0, 0.085]])                   # 1.5 cm over the centre
    d = m.clearance_excluding(sample, below_z=0.07, cylinder=None, within=0.03)
    assert d[0] == pytest.approx(0.015 - V / 2, abs=1e-6)        # counted: re-measured to the centre
    d = m.clearance_excluding(sample, below_z=None, cylinder=(np.zeros(2), 0.07, 0.0), within=0.03)
    assert d[0] == pytest.approx(0.015 - V / 2, abs=1e-6)


def _report_with_voxels(rc, voxels, voxel_m, **grid):
    rep = {"phase": "vet_pose", "gate": "tcp", "reason": "r", "q": [0.0] * 6, "q_prev": None,
           "sample": list(voxels[0][:3]), "clearance_m": 0.02, "exemption": None, "margin_m": 0.03,
           "voxels": voxels, "map": {"refreshes": 1, "age_s": 0.0, "voxel_m": voxel_m, "occupied": len(voxels),
                                     **grid},
           "snapshot": {"centre": list(voxels[0][:3]), "radius_m": 0.05, "max_voxels": 64}}
    return rc._why_not_replayable(rep, rc._arm_config("rebot_rs"))


def test_voxels_off_the_recorded_map_grid_or_a_bad_grid_are_not_replayable(rc):
    vox = [[0.405, 0.005, 0.405], [0.415, 0.005, 0.405]]
    assert _report_with_voxels(rc, vox, 0.01, grid_origin=[-0.295, -0.445, -0.015], grid_shape=[96, 90, 62]) is None
    why = _report_with_voxels(rc, vox, 0.01, grid_origin=[-0.29, -0.445, -0.015], grid_shape=[96, 90, 62])
    assert why == "voxel centres off the recorded map grid", why
    for bad in ({"grid_origin": [0.0, 0.0], "grid_shape": [1, 1, 1]}, {"grid_origin": [0.0] * 3, "grid_shape": [0, 1, 1]},
                {"grid_origin": [0.0] * 3, "grid_shape": None}, {"grid_origin": [0.0] * 3, "grid_shape": [1.5, 1, 1]}):
        why = _report_with_voxels(rc, vox, 0.01, **bad)
        assert why is not None and why.startswith("bad voxels or map grid"), (bad, why)


@needs_pin
def test_an_admission_without_the_recorded_grid_is_undecidable(rc, rebot):
    """An older report (no `grid_origin` / `grid_shape`) cannot mirror the
    live clamp at the region faces: what would certify "allowed" reads
    undecidable."""
    q_rec = at_x(rebot, X_REC)
    centre = lattice(rebot.kin.fk(q_rec)[:3, 3] + [-0.015, 0.0, 0.07])
    h, _ = recorder(rebot, voxel_box(centre), snapshot=WHOLE)
    assert h.vet_pose(q_rec)
    (rep,) = h.drain_clearance_reports()
    rep = json.loads(json.dumps(rep))
    parts = rc.tool_volume_from_override("rebot_rs", PARTS)
    assert rc.replay(rep, arm="rebot_rs", tool_volume_override=parts)["L4"] == "allowed"
    for key in ("grid_origin", "grid_shape"):
        rep["map"].pop(key)
    out = rc.replay(rep, arm="rebot_rs", tool_volume_override=parts)
    assert out["L4"] == rc.NO_REGION and out["old"].startswith("refused")


def test_a_lattice_the_millimetre_rounding_broke_is_not_replayable(rc):
    """1.5 cm voxels: centres at .0075 steps, rounded to 1 mm, sit 0.5 mm off
    the recorded lattice. A 1 mm tolerance passed them; the certificate needs
    the lattice to ~1e-6 m."""
    rounded = [[0.407, 0.0, 0.4], [0.422, 0.0, 0.4], [0.438, 0.0, 0.4]]   # 0.4075, 0.4225, 0.4375 at 1 mm
    why = _report_with_voxels(rc, rounded, 0.015)
    assert why is not None and "lattice" in why, why
    assert _report_with_voxels(rc, [[0.405, 0.0, 0.4], [0.42, 0.0, 0.4], [0.435, 0.0, 0.4]], 0.015) is None


def test_a_snapshot_whose_steps_share_a_factor_is_not_replayable(rc):
    """voxel_m recorded as half the real voxel: every centre step is an even
    number of recorded voxels. A GCD >= 2 over the integer steps catches it
    (fail closed); one voxel, or steps with GCD 1, replay."""
    why = _report_with_voxels(rc, [[0.405, 0.005, 0.405], [0.425, 0.005, 0.405], [0.405, 0.045, 0.405]], 0.01)
    assert why is not None and "share a factor" in why, why
    assert _report_with_voxels(rc, [[0.405, 0.005, 0.405], [0.415, 0.005, 0.405]], 0.01) is None
    assert _report_with_voxels(rc, [[0.405, 0.005, 0.405]], 0.01) is None


def test_reports_the_harness_did_not_keep_are_counted_structurally(rc, tmp_path, capsys):
    """`context.clearance_lost` (a filing exception, eviction at the report
    cap) counts as `no_snapshot` whatever the error text says; the wording
    match is kept only for rows with neither key (pre-A4 traces)."""
    trace = write_trace(
        tmp_path / "mcp_10" / "trace.jsonl",
        {"step": 0, "skill": "move_home", "context": {"clearance_lost": 2},
         "result": {"ok": False, "error": "this attempt FAILED: the arm could not move"}},
        {"step": 1, "skill": "move_home", "context": {"clearance": [], "clearance_lost": 1},
         "result": {"ok": False, "error": "point 0 clearance 0.02 m below 0.03 m (occupancy map)"}},
        {"step": 2, "skill": "move_home", "context": {},
         "result": {"ok": False, "error": "point 0 clearance 0.02 m below 0.03 m (occupancy map)"}})
    counts = {}
    assert rc.load_reports([trace], counts) == []
    assert counts["no_snapshot"] == 4 and counts["lost"] == 3
    rc.main([str(trace)])
    assert "3 report(s) the harness did not keep" in capsys.readouterr().err
