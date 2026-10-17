"""Re-vet recorded clearance refusals from their map snapshots (offline only).

Every clearance refusal of the TCP/joint, held-payload or tool-volume map gate
files a bounded map snapshot into its trace row: `context.clearance` (spec
2026-10-16 nvblox object-aware §3.3, `SafetyHarness.drain_clearance_reports`).
This rebuilds the live map's distance grid from each snapshot (`ReplayMap`)
and re-runs the arm's own SafetyHarness at the recorded joints, per lever
(spec §6.3):

- `old`: the static strict map check at `q` -- vet_pose's map gates (TCP and
  joint origins, the held payload, the tool volume) under the recorded
  exemption (its z_max ceiling included), no escape rule;
- `L1`: a refusal the live gate decided with the step rule itself (phase
  approve or vet_step) keeps its RECORDED verdict (`L1_source: "recorded"`):
  that rule saw the whole map. The model's own `vet_step(q_prev, q)` verdict
  is `L1_model`; a model that admits such a step, or that admits any
  recorded refusal under `old`, counts in `model_mismatch` (a fidelity
  number, never an admission). A static refusal (vet_pose, route_end) gets
  L1 from the model (`L1_source: "model"`): `vet_step(q_prev, q)` with the
  recorded exemption in force (dt = 0: the velocity cap is not replayed), or
  the static check without a `q_prev` (every post-PR-A static report). So
  this replay does NOT measure L1 or L2: the escape steps the rule admits
  file no report. Spec §7 A1 (live acceptance) measures them, or a future
  trace counter of admitted escapes;
- `L4`: `old` with an override tool volume (`--tool-volume`, the rebuilt
  camera box + measured parts); without one it is `old` with the shipped
  volume.

Distance model. The live gates read `OccupancyMap.clearance`: a trilinear
read of the bridge's grid, whose nodes are the voxel centres and whose values
are distances between voxel centres (nvblox's ESDF, the warp / voxel EDT). The
replay rebuilds that lattice EDT from the snapshot's centres (lattice origin
= any centre) and reads it the same way; a tool sample below the margin is
re-measured to the nearest centre outside the table band and the exemption
column, minus half a voxel, as `clearance_excluding` does. That is exact on
those backends up to the snapshot's 1 mm rounding (`replay_clearance_m` is
the replayed read at the recorded sample). A box-distance map (the GridBridge
test fake) cannot be recovered from voxel centres: the rebuilt grid reads
farther there. A snapshot holds no unknown voxels: the live read takes an
unobserved grid corner as far (1e3), the replay has no such corner, so near
unobserved space it reads LOW (toward refused). A sample outside the live
map's region reads +inf there, finite here (toward refused too).

Three-valued verdicts. A snapshot holds only the occupied voxels within
`radius_m` of its query centre, at most `max_voxels` of them (the nearest);
the report records both and the centre (`snapshot`; without it, the
profile's `occupancy.refusal_snapshot` and the sample). Every voxel it left
out lies at least `R - |p - centre|` from a point p, where R is `radius_m`,
or the farthest kept distance minus the 1 mm rounding when the cap was hit,
less the 1 cm dedupe cell's diagonal for an approve report without a recorded
centre (a streamed halt may have taken it over from another sample in that
cell). Each computed lever runs twice: on the snapshot alone and with that
certified bound on every grid node and re-measure (`certified`). Both allowed
-> "allowed", both refused -> "refused: <the harness's reason>", otherwise
"undecidable: ...". The 5 cm / 64-voxel shipped snapshot rarely certifies an
admission for the arm or a ~15 cm tool: such rows read "undecidable", never
"allowed" (`partial_map: true`). Reports whose snapshot is of a newer map than
the gate read, or that a halt took over, are flagged `approximate`.

Not replayable (every lever "refused: not replayable (why): <recorded
reason>", its own bucket): a map that refused by RAISING (stale, failed, body
mask: `raised`); a phase the harness never writes; no measured sample, no
voxels or voxel size; an unobserved (NaN) clearance; joints that do not fit
`--arm`; a `margin_m` other than the replaying profile's; voxel centres off a
`voxel_m` lattice (1 mm rounding allowed); a malformed exemption or snapshot
record; a report entry that is not a mapping.

Never connects to anything (no bridge, arm, camera or service) and never
writes: it reads trace files and the arm profile, prints one JSON line per
report, then the summary {lever: {allowed, refused, undecidable,
not_replayable, no_snapshot}, model_mismatch}. `no_snapshot` counts failed
rows whose error names a map clearance gate but that carry no
`context.clearance` (a pre-A4 trace, a lost report): the spec §7 A2 metric.
stderr: totals; unreadable lines, malformed report entries and reports whose
replay raised are counted there and make the exit status 1 (each such report
still gets a not-replayable row and the run goes on).

    uv run --extra kinematics python scripts/replay_clearance.py runs/mcp_*/trace.jsonl \\
        [--arm rebot_rs] [--tool-volume parts.json]

`parts.json` overrides the profile's `safety.tool_volume` and
`relative_motion.wrist_camera` blocks, e.g. the URDF parts named in
configs/arms/rebot_rs.yaml with a cut camera box:

    {"tool_volume": {"housing": {"min": [-0.160, -0.095, -0.045], "max": [-0.070, 0.095, 0.045]},
                     "fingers": {"min": [-0.090, -0.085, -0.020], "max": [0.0, 0.085, 0.020]}},
     "wrist_camera": {"front_m": 0.06, "bottom_m": 0.035}}
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))     # this tree's cascade, never another checkout's

from cascade.config import Cfg, load_demo_config
from cascade.perception.occupancy import _refusal_snapshot
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import build_tool_volume

LEVERS = ("old", "L1", "L4")
BUCKETS = ("allowed", "refused", "undecidable", "not_replayable", "no_snapshot")
#: phases whose live gate already ran the step rule (L1) on the whole map:
#: their L1 is the recorded verdict
STEP_PHASES = ("approve", "vet_step")
#: phases whose live gate is the static check: every lever is computed
STATIC_PHASES = ("vet_pose", "route_end")
#: without a recorded snapshot centre, a kept voxel whose `d` differs from its
#: distance to the report's sample by more than this (centres and `d` are
#: rounded to 1 mm) was measured from an earlier sample: a halt took it over
TAKEOVER_TOL_M = 0.002
#: the snapshot rounds distances and centres to 1 mm
ROUNDING_M = 0.001
#: centres on one voxel lattice: each rounded by up to 0.5 mm
LATTICE_TOL_M = ROUNDING_M + 1e-6
#: the harness dedupes reports by their sample rounded to 1 cm: a streamed
#: halt takes a report over from a sample in the same cell, at most its
#: diagonal away
DEDUPE_CELL_DIAGONAL_M = 0.01 * float(np.sqrt(3.0))
#: the trace-row error texts of a map clearance refusal or halt (harness reasons)
MAP_REFUSAL = re.compile(r"occupancy map|required occupancy unavailable|occupancy unsafe"
                         r"|clearance unavailable \(fresh observed distance grid required\)")
#: the override file's blocks
_OVERRIDE_KEYS = {"tool_volume", "wrist_camera"}
_COUNTS = ("skipped_lines", "malformed_entries", "no_snapshot")


class ReplayMap:
    """An occupancy stand-in holding one refusal snapshot's voxels, read as
    the live map reads its grid.

    `clearance(points)` is `OccupancyMap._sample_grid`'s read of the lattice
    EDT rebuilt from the snapshot: each node of the voxel-centre lattice
    (origin: any centre) holds its distance to the nearest snapshot centre,
    blended trilinearly over the 8 nodes around a point (+inf with no voxels).
    `clearance_excluding` is `OccupancyMap.clearance_excluding`: that read,
    and for a sample below `within` the nearest centre outside the table band
    (`below_z`) and the `cylinder` (z_max ceiling included) minus half a
    voxel, or `within` when none lies within `within` + a voxel.

    `certified=(centre, R)`: the snapshot left out no voxel within R of
    `centre`, so every node value and re-measured distance is capped by
    R - |p - centre| (never below 0): the lower bound of the live read.

    The rest of the map surface the harness reads: `payload_points` /
    `payload_clearance` (the held samples given in the TCP frame),
    `nearby_occupied`, `clearance_record`, `refreshes`, `status["voxel"]`
    (the escape's interior sampling) and a REENTRANT `_refresh_lock`: the
    harness holds it while it files a report and then calls
    `clearance_record` and `nearby_occupied`, which take it again.
    `required = False` (an answer, never a raise) and `unknown_blocks =
    False` (the snapshot holds no observation support).
    """

    required = False
    unknown_blocks = False

    def __init__(self, voxels, voxel_m: float, *, payload_tcp=None, certified=None):
        voxel_m = float(voxel_m)
        if not (np.isfinite(voxel_m) and voxel_m > 0.0):
            raise ValueError(f"voxel_m must be finite and > 0, got {voxel_m!r}")
        rows = np.asarray(voxels, dtype=float)
        centres = rows.reshape(len(rows), -1)[:, :3] if rows.size else np.zeros((0, 3))
        if centres.shape[1] != 3 or not np.all(np.isfinite(centres)):
            raise ValueError("voxels must be finite rows [x, y, z(, d)]")
        self._occupied = np.array(centres, dtype=float)
        self._origin = self._occupied[0].copy() if len(self._occupied) else np.zeros(3)
        self._bound = None
        if certified is not None:
            centre, radius = np.asarray(certified[0], dtype=float).reshape(3), float(certified[1])
            if not (np.all(np.isfinite(centre)) and np.isfinite(radius)):
                raise ValueError(f"a certificate is a finite (centre, radius), got {certified!r}")
            self._bound = (centre, radius)
        self.voxel_m = voxel_m
        self.status = {"voxel": voxel_m}
        self.refreshes = 0
        self._refresh_lock = threading.RLock()
        held = np.zeros((0, 3)) if payload_tcp is None else np.asarray(payload_tcp, dtype=float)
        self._payload_tcp = held.reshape(-1, 3)

    def _ruled_out(self, points) -> np.ndarray:
        """The least distance from each point to a voxel the snapshot left
        out: R - |p - centre| (never below 0) with a certificate, +inf
        without one (the snapshot is the whole map)."""
        if self._bound is None:
            return np.full(len(points), np.inf)
        centre, radius = self._bound
        return np.maximum(radius - np.linalg.norm(points - centre, axis=1), 0.0)

    @staticmethod
    def _nearest(points, centres) -> np.ndarray:
        """Distance from each point to its nearest centre, +inf with none."""
        out = np.full(len(points), np.inf)
        if not len(centres):
            return out
        step = max(1, 2 ** 18 // len(centres))
        for start in range(0, len(points), step):
            chunk = points[start:start + step]
            out[start:start + step] = np.linalg.norm(chunk[:, None, :] - centres[None, :, :], axis=-1).min(axis=1)
        return out

    def _grid_read(self, points) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(self._occupied) and self._bound is None:
            return np.full(len(pts), np.inf)
        v = self.voxel_m
        f = (pts - self._origin) / v
        i0 = np.floor(f)
        t = f - i0
        corners = np.array([[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=float)
        keys, inverse = np.unique((i0[:, None, :] + corners[None, :, :]).reshape(-1, 3), axis=0,
                                  return_inverse=True)
        nodes = self._origin + keys * v
        value = np.minimum(self._nearest(nodes, self._occupied), self._ruled_out(nodes))
        node_d = value[np.asarray(inverse).reshape(-1)].reshape(len(pts), 8)
        weight = np.prod(np.where(corners[None, :, :] == 1.0, t[:, None, :], 1.0 - t[:, None, :]), axis=-1)
        return (weight * node_d).sum(axis=1)

    def clearance(self, points) -> np.ndarray:
        return self._grid_read(points)

    def clearance_excluding(self, points, *, below_z=None, cylinder=None, within=None) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        dist = self.clearance(pts)
        if within is None or (below_z is None and cylinder is None):
            return dist
        dist = np.array(dist, dtype=float)
        near = np.isfinite(dist) & (dist < within)
        if not near.any() or not len(self._occupied):
            return dist
        cand = self._occupied
        if below_z is not None:
            cand = cand[cand[:, 2] > float(below_z)]
        if cylinder is not None and len(cand):
            center, radius, z_min = cylinder[:3]
            inside = ((np.linalg.norm(cand[:, :2] - np.asarray(center, dtype=float)[:2], axis=1) <= float(radius))
                      & (cand[:, 2] >= float(z_min) - 1e-6))
            if len(cylinder) > 3:
                inside &= cand[:, 2] <= float(cylinder[3]) + 1e-6
            cand = cand[~inside]
        sub = pts[near]
        exact = np.minimum(self._nearest(sub, cand), self._ruled_out(sub))
        reach = float(within) + self.voxel_m
        dist[near] = np.where(exact <= reach, np.maximum(exact - self.voxel_m / 2.0, 0.0), float(within))
        return dist

    def payload_points(self, T_base_tcp) -> np.ndarray:
        T = np.asarray(T_base_tcp, dtype=float)
        return self._payload_tcp @ T[:3, :3].T + T[:3, 3]

    def payload_clearance(self, points) -> np.ndarray:
        return self.clearance(points)

    def nearby_occupied(self, point, radius_m: float, limit: int) -> np.ndarray:
        with self._refresh_lock:
            cloud = self._occupied
        if not len(cloud) or int(limit) <= 0:
            return np.zeros((0, 4))
        p = np.asarray(point, dtype=float).reshape(3)
        d = np.linalg.norm(cloud - p, axis=1)
        keep = d <= float(radius_m)
        pts, d = cloud[keep], d[keep]
        order = np.argsort(d, kind="stable")[: int(limit)]
        return np.column_stack([pts[order], d[order]])

    def clearance_record(self) -> dict:
        with self._refresh_lock:
            return {"refreshes": int(self.refreshes), "age_s": 0.0, "voxel_m": self.voxel_m,
                    "occupied": len(self._occupied)}


# ── reading traces ──────────────────────────────────────────────────────────


def _json_row(line: str):
    """The trace row a line holds (a JSON object), or None."""
    try:
        row = json.loads(line)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


def _row_reports(row):
    """A row's `context.clearance`: its list, None when it has none, False
    when it is there but not a list."""
    context = row.get("context")
    reports = context.get("clearance") if isinstance(context, dict) else None
    if reports is None or isinstance(reports, list):
        return reports
    return False


def _map_refusal(row: dict) -> bool:
    """Is this row a failed call whose error names a map clearance gate?"""
    result = row.get("result")
    return (isinstance(result, dict) and result.get("ok") is False
            and bool(MAP_REFUSAL.search(str(result.get("error", "")))))


def load_reports(paths, counts: dict | None = None) -> list[dict]:
    """Every `context.clearance` entry of the given trace.jsonl files (or run
    directories holding one), oldest first, each with its row's `run` (the
    run directory's name), `step` and `skill`. An entry that is not a
    mapping comes back as {run, step, skill, "malformed": why} (replay() says
    it is not replayable). `counts` (optional) accumulates
    `skipped_lines` (a line that is not a JSON object, or whose clearance
    field is not a list), `malformed_entries` and `no_snapshot` (a failed row
    naming a map clearance gate without any report)."""
    counts = {} if counts is None else counts
    for key in _COUNTS:
        counts.setdefault(key, 0)
    out: list[dict] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "trace.jsonl"
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = _json_row(line)
            reports = None if row is None else _row_reports(row)
            if row is None or reports is False:
                counts["skipped_lines"] += 1
                continue
            if not reports:
                counts["no_snapshot"] += int(_map_refusal(row))
                continue
            where = {"run": path.parent.name, "step": row.get("step"), "skill": row.get("skill")}
            for report in reports:
                if isinstance(report, dict):
                    out.append({**report, **where})
                else:
                    counts["malformed_entries"] += 1
                    out.append({**where, "malformed": f"report entry is not a mapping: {report!r:.80}"})
    return out


# ── the arm ─────────────────────────────────────────────────────────────────


@functools.cache
def _arm_config(arm: str) -> SimpleNamespace:
    """The profile as `apps.demo._build_arm` reads it: the arm's own
    resolved view, its limits and its shipped tool volume (no kinematics),
    plus the map's snapshot bounds (for reports that do not record them)."""
    cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
    acfg = cfg.arm
    view = acfg.get("resolved") or cfg
    occupancy = cfg.get("occupancy")
    snapshot = _refusal_snapshot(occupancy.get("refusal_snapshot") if occupancy is not None else None)
    return SimpleNamespace(name=arm, acfg=acfg, view=view, n_joints=int(acfg.get("n_joints", 6)),
                           limits=SafetyLimits.from_config(view.safety), volume=build_tool_volume(view, acfg),
                           snapshot=snapshot)


@functools.cache
def _kinematics(arm: str):
    """The arm's kinematics, built as `apps.demo._build_arm` builds it."""
    from cascade.control.kinematics import Kinematics

    acfg = _arm_config(arm).acfg
    return Kinematics(model_path=acfg.model, ee_frame=acfg.get("ee_frame", "gripper_end"),
                      n_controlled=_arm_config(arm).n_joints, joint_signs=acfg.get("joint_signs"),
                      ik_task_weights=acfg.get("ik_task_weights"))


def tool_volume_from_override(arm: str, override: dict):
    """The tool volume `arm` would build with `override` = {"tool_volume":
    {...}, "wrist_camera": {...}} merged over its `safety.tool_volume` and
    `relative_motion.wrist_camera` blocks (lever 4). Validated by
    `build_tool_volume` itself: a bad block raises ValueError."""
    if not isinstance(override, dict) or not override:
        raise ValueError(f"a tool-volume override is a non-empty mapping of {sorted(_OVERRIDE_KEYS)}, "
                         f"got {override!r}")
    unknown = sorted(set(override) - _OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"tool-volume override: unknown key(s) {unknown}; allowed {sorted(_OVERRIDE_KEYS)}")
    model = _arm_config(arm)
    data = model.view.as_dict()
    for key, (section, block) in (("tool_volume", ("safety", "tool_volume")),
                                  ("wrist_camera", ("relative_motion", "wrist_camera"))):
        if key in override:
            if not isinstance(override[key], dict):
                raise ValueError(f"override {key} must be a mapping, got {override[key]!r}")
            merged = dict(data.setdefault(section, {}).get(block) or {})
            merged.update(override[key])
            data[section][block] = merged
    volume = build_tool_volume(Cfg(data), model.acfg)
    if volume is None:
        raise ValueError("the override leaves the arm without a tool volume")
    return volume


# ── replay ──────────────────────────────────────────────────────────────────


def _joints(value, n: int, name: str):
    """(joint vector, None) or (None, why it is unusable)."""
    try:
        q = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, f"{name} is not a joint vector"
    if q.shape != (n,):
        return None, f"{name} has {q.size} joints, the arm has {n}"
    if not np.all(np.isfinite(q)):
        return None, f"{name} is not finite"
    return q, None


def _point(value) -> bool:
    """Is `value` three finite numbers?"""
    try:
        p = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return p.shape == (3,) and bool(np.all(np.isfinite(p)))


def _snapshot_record(report: dict):
    """The recorded `snapshot` as (centre or None, radius_m, max_voxels), or
    None when the report has none (older reports). Raises TypeError /
    ValueError when it is malformed."""
    record = report.get("snapshot")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise TypeError(f"not a mapping: {record!r:.80}")
    centre, radius, cap = record.get("centre"), record.get("radius_m"), record.get("max_voxels")
    if centre is not None and not _point(centre):
        raise ValueError(f"centre {centre!r:.80}")
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not (np.isfinite(radius) and radius > 0):
        raise ValueError(f"radius_m {radius!r}")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError(f"max_voxels {cap!r}")
    if centre is None and report.get("sample") is not None:
        raise ValueError("a sample but no centre")
    return (None if centre is None else np.asarray(centre, dtype=float)), float(radius), int(cap)


def _why_not_replayable(report: dict, model) -> str | None:
    if report.get("malformed"):
        return str(report["malformed"])
    phase = report.get("phase")
    if phase not in STEP_PHASES + STATIC_PHASES:
        return f"phase {phase!r} is not a gate phase {STEP_PHASES + STATIC_PHASES}"
    if report.get("raised"):
        return f"the map raised {report['raised']}: no voxels"
    _, why = _joints(report.get("q"), model.n_joints, "q")
    if why is None and report.get("q_prev") is not None:
        _, why = _joints(report.get("q_prev"), model.n_joints, "q_prev")
    if why is not None:
        return why
    if report.get("sample") is None:
        return "no offending sample: the map gave no answer"
    if not _point(report.get("sample")):
        return "bad sample"
    if report.get("gate") == "attached" and not _point(report.get("sample_tcp")):
        return "bad sample_tcp: the held sample cannot be placed"
    if report.get("clearance_m") is None:
        return "unobserved clearance: the snapshot holds no observation support"
    margin, own = report.get("margin_m"), float(model.limits.min_clearance_m)
    try:
        same = abs(float(margin) - own) <= 1e-9
    except (TypeError, ValueError):
        same = False
    if not same:
        return f"margin_m {margin!r} recorded, the replaying profile's min_clearance_m is {own:g}"
    exemption = report.get("exemption")
    if exemption is not None and not isinstance(exemption, dict):
        return f"exemption is not a mapping: {exemption!r:.80}"
    cylinder = (exemption or {}).get("cylinder")
    if cylinder is not None:
        try:
            vals = np.asarray(cylinder, dtype=float)
        except (TypeError, ValueError):
            vals = np.zeros(0)
        if vals.shape not in ((4,), (5,)) or not np.all(np.isfinite(vals[:4])):
            return f"bad exemption cylinder {cylinder!r}"
    if not report.get("voxels"):
        error = report.get("snapshot_error")
        return "no voxels in the snapshot" + (f" ({error})" if error else "")
    record = report.get("map")
    voxel = record.get("voxel_m") if isinstance(record, dict) else None
    try:
        ok = voxel is not None and np.isfinite(float(voxel)) and float(voxel) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return "no voxel size in the map record"
    try:
        rows = np.asarray(report["voxels"], dtype=float)
        if rows.ndim != 2 or rows.shape[1] not in (3, 4) or not np.all(np.isfinite(rows)):
            raise ValueError(f"rows of shape {rows.shape}, finite [x, y, z, d] expected")
        ReplayMap(rows, voxel)
    except (TypeError, ValueError) as exc:
        return f"bad voxels: {exc}"
    voxel = float(voxel)
    steps = (rows[:, :3] - rows[0, :3]) / voxel
    if np.any(np.abs(steps - np.round(steps)) * voxel > LATTICE_TOL_M):
        return f"voxel centres off the {voxel:g} m lattice (voxel_m does not fit them)"
    try:
        _snapshot_record(report)
    except (TypeError, ValueError) as exc:
        return f"bad snapshot record: {exc}"
    return None


def _exemption(report: dict):
    """The recorded exemption as the harness holds it: (center_xy, radius,
    z_min[, z_max]) or None."""
    cylinder = (report.get("exemption") or {}).get("cylinder")
    if cylinder is None:
        return None
    vals = [float(v) for v in cylinder]
    return (np.asarray(vals[:2]), vals[2], vals[3], *vals[4:5])


def _static_reason(h: SafetyHarness, q, exempt) -> str | None:
    """vet_pose's map gates at `q`, in its order, under `exempt` with its
    ceiling (vet_pose itself takes no z_max): TCP + joint origins, the held
    payload, the tool volume. No escape rule; no report filed (no ctx).
    Pinned to vet_pose by a parity test: an edit of vet_pose's map gates
    updates this copy (global constraints, B4-B6)."""
    return (h._occupancy_violation(h._clearance_points(q), exempt)
            or h._payload_occupancy_violation(q, exempt)
            or h._tool_volume_violation(h.kin.fk(q), exempt))


def _step_reason(h: SafetyHarness, q_prev, q, exempt) -> str | None:
    """`vet_step(q_prev, q)` with the recorded exemption as the grasp
    exemption in force (the approach / exit column)."""
    cylinder = None if exempt is None else exempt[:3]
    z_max = exempt[3] if exempt is not None and len(exempt) > 3 else None
    with h.target_column_exemption(cylinder, z_max):
        return h.vet_step(q_prev, q, 0.0)


def _verdict(snapshot_only: str | None, certified: str | None) -> str:
    """One three-valued verdict from the same check on the snapshot alone
    and with the certified bound on what it left out."""
    if snapshot_only is None and certified is None:
        return "allowed"
    if snapshot_only is not None and certified is not None:
        return f"refused: {snapshot_only}"
    if snapshot_only is None:
        return f"undecidable: the snapshot admits it, a voxel it left out could refuse it ({certified})"
    return f"undecidable: the snapshot refuses it ({snapshot_only}), the voxels it left out could change that"


def _approximate(report: dict, sample: np.ndarray, centre) -> list[str]:
    flags = []
    if report.get("snapshot_newer_map"):
        flags.append("snapshot_newer_map")
    if centre is not None:
        moved = float(np.linalg.norm(centre - sample)) > 1e-9
    else:
        vox = np.asarray(report["voxels"], dtype=float)
        moved = vox.shape[1] >= 4 and float(np.max(np.abs(np.linalg.norm(vox[:, :3] - sample, axis=1)
                                                          - vox[:, 3]))) > TAKEOVER_TOL_M
    if moved:
        flags.append("takeover")
    extra = sorted(set(report.get("exemption") or {}) - {"cylinder"})
    if extra:
        flags.append(f"exemption keys not replayed: {extra}")
    return flags


def _certificate(report: dict, model, sample: np.ndarray, flags) -> dict:
    """Where the snapshot rules voxels out: none it left out lies within
    `radius_m` of `centre` (module docstring, "Three-valued verdicts")."""
    recorded = _snapshot_record(report)
    if recorded is not None:
        centre, radius, cap = recorded
        bounds = "recorded"
    else:
        centre, radius, cap = None, float(model.snapshot.radius_m), int(model.snapshot.max_voxels)
        bounds = "profile"
    vox = np.asarray(report["voxels"], dtype=float)
    if len(vox) >= cap:             # capped: what it left out is no nearer than its farthest kept voxel
        if vox.shape[1] >= 4:
            farthest = float(vox[:, 3].max())
        else:
            farthest = 0.0 if centre is None else float(np.linalg.norm(vox[:, :3] - centre, axis=1).max())
        radius = min(radius, farthest - ROUNDING_M)
    if centre is None:
        centre = sample
        if report.get("phase") == "approve" or "takeover" in flags:
            radius -= DEDUPE_CELL_DIAGONAL_M
    return {"centre": np.asarray(centre, dtype=float).tolist(), "radius_m": max(float(radius), 0.0),
            "bounds": bounds}


def _not_replayable(report, why: str) -> dict:
    """The row of a report nothing can be replayed from: every lever refused."""
    report = report if isinstance(report, dict) else {}
    refused = f"refused: not replayable ({why}): {report.get('reason')}"
    return {"run": report.get("run"), "step": report.get("step"), "skill": report.get("skill"),
            "phase": report.get("phase"), "gate": report.get("gate"), "recorded": report.get("reason"),
            "clearance_m": report.get("clearance_m"), "replay_clearance_m": None,
            **{lever: refused for lever in LEVERS}, "L1_model": refused,
            "L1_source": "recorded" if report.get("phase") in STEP_PHASES else "model",
            "L4_volume": None, "partial_map": True, "replayable": False, "not_replayable": why,
            "approximate": [], "model_mismatch": False, "certified": None}


def replay(report: dict, *, arm: str, tool_volume_override=None) -> dict:
    """One report re-vetted at its recorded joints on its own snapshot:
    `old`, `L1`, `L4` ("allowed" / "refused: ..." / "undecidable: ..."),
    `L1_source`, `L1_model`, `model_mismatch`, plus what was recorded, the
    replayed clearance at the recorded sample, the `certified` ball,
    `partial_map: True`, `replayable` and `approximate`. Never changes
    `report`."""
    if not isinstance(report, dict):
        report = {"malformed": f"report entry is not a mapping: {report!r:.80}"}
    model = _arm_config(arm)
    why = _why_not_replayable(report, model)
    if why is not None:
        out = _not_replayable(report, why)
        out["L4_volume"] = "shipped" if tool_volume_override is None else "override"
        return out

    kin = _kinematics(arm)
    phase = report["phase"]
    q = np.asarray(report["q"], dtype=float)
    q_prev = None if report.get("q_prev") is None else np.asarray(report["q_prev"], dtype=float)
    sample = np.asarray(report["sample"], dtype=float).reshape(3)
    exempt = _exemption(report)
    base_pose = report.get("base_pose")
    payload = report.get("sample_tcp") if report.get("gate") == "attached" else None
    recorded = _snapshot_record(report)
    flags = _approximate(report, sample, None if recorded is None else recorded[0])
    certificate = _certificate(report, model, sample, flags)
    bound = (certificate["centre"], certificate["radius_m"])

    def harnesses(volume):
        """The same harness over the snapshot alone and over the snapshot
        with the certified bound."""
        return tuple(SafetyHarness(model.limits, kinematics=kin, base_pose=base_pose, tool_volume=volume,
                                   occupancy=ReplayMap(report["voxels"], report["map"]["voxel_m"],
                                                       payload_tcp=payload, certified=certified))
                     for certified in (None, bound))

    shipped = harnesses(model.volume)
    old = _verdict(*(_static_reason(h, q, exempt) for h in shipped))
    l1_model = old if q_prev is None else _verdict(*(_step_reason(h, q_prev, q, exempt) for h in shipped))
    if phase in STEP_PHASES:
        # The live gate ran this very rule on the WHOLE map and refused.
        l1, source = f"refused: {report.get('reason')}", "recorded"
    else:
        l1, source = l1_model, "model"
    l4 = old if tool_volume_override is None else _verdict(
        *(_static_reason(h, q, exempt) for h in harnesses(tool_volume_override)))

    h = shipped[0]
    gate = str(report.get("gate"))
    if gate in ("tcp", "attached") or gate.startswith("link "):
        d = h.occupancy.clearance(sample[None, :])[0]
    else:   # a tool part: obstacles attributed by where they are, as the gate reads them
        band = 0.0 if model.volume is None else float(model.volume.table_band_m)
        d = h.occupancy.clearance_excluding(sample[None, :], below_z=float(model.limits.table_z) + band,
                                            cylinder=exempt, within=float(model.limits.min_clearance_m))[0]
    return {"run": report.get("run"), "step": report.get("step"), "skill": report.get("skill"),
            "phase": phase, "gate": report.get("gate"), "recorded": report.get("reason"),
            "clearance_m": report.get("clearance_m"), "replay_clearance_m": float(d) if np.isfinite(d) else None,
            "old": old, "L1": l1, "L4": l4, "L1_source": source, "L1_model": l1_model,
            "L4_volume": "shipped" if tool_volume_override is None else "override",
            "partial_map": True, "replayable": True, "approximate": flags,
            # every recorded refusal refused live: a model that admits it is wrong
            "model_mismatch": old == "allowed" or (phase in STEP_PHASES and l1_model == "allowed"),
            "certified": certificate}


def _bucket(out: dict, lever: str) -> str:
    if not out.get("replayable", False):
        return "not_replayable"
    verdict = str(out.get(lever))
    if verdict == "allowed":
        return "allowed"
    return "undecidable" if verdict.startswith("undecidable") else "refused"


def summarize(results, *, no_snapshot: int = 0) -> dict:
    """{lever: {allowed, refused, undecidable, not_replayable, no_snapshot},
    "model_mismatch": n} over replay() results; `no_snapshot` = the refusal
    rows without a report (load_reports' count). Anything that is not
    "allowed" or "undecidable" counts as refused."""
    summary = {lever: dict.fromkeys(BUCKETS, 0) for lever in LEVERS}
    for out in results:
        for lever in LEVERS:
            summary[lever][_bucket(out, lever)] += 1
    for lever in LEVERS:
        summary[lever]["no_snapshot"] = int(no_snapshot)
    summary["model_mismatch"] = sum(bool(out.get("model_mismatch")) for out in results)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="trace.jsonl files or run directories")
    parser.add_argument("--arm", default="rebot_rs", help="the arm profile the traces were recorded on")
    parser.add_argument("--tool-volume", metavar="JSON",
                        help="lever 4: override {tool_volume, wrist_camera} blocks (default: L4 = old)")
    args = parser.parse_args(argv)
    override = None
    if args.tool_volume:
        override = tool_volume_from_override(args.arm, json.loads(Path(args.tool_volume).read_text()))
    counts: dict = {}
    reports = load_reports(args.paths, counts)
    results, failed = [], 0
    for report in reports:
        try:
            out = replay(report, arm=args.arm, tool_volume_override=override)
        except Exception as exc:  # noqa: BLE001 -- one bad report never stops the run
            failed += 1
            out = _not_replayable(report, f"replay failed: {type(exc).__name__}: {exc}")
        print(json.dumps(out))
        results.append(out)
    print(json.dumps(summarize(results, no_snapshot=counts["no_snapshot"])))
    print(f"{len(results)} report(s): {sum(r['replayable'] for r in results)} replayable, "
          f"{sum(bool(r['approximate']) for r in results)} approximate, "
          f"{sum(bool(r['model_mismatch']) for r in results)} model mismatch", file=sys.stderr)
    print(f"{counts['no_snapshot']} refusal row(s) without context.clearance (spec §7 A2)", file=sys.stderr)
    print("L1 of an approve/vet_step refusal is its recorded verdict: L1/L2 are measured by the live "
          "acceptance (spec §7 A1), not by this replay", file=sys.stderr)
    skipped, malformed = counts["skipped_lines"], counts["malformed_entries"]
    if skipped or malformed or failed:
        print(f"input errors: {skipped} unreadable line(s), {malformed} malformed report "
              f"entr{'y' if malformed == 1 else 'ies'}, {failed} report(s) whose replay raised", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
