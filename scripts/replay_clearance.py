"""Re-vet recorded clearance refusals from their map snapshots (offline only).

Every clearance refusal of the TCP/joint, held-payload or tool-volume map gate
files a bounded map snapshot into its trace row: `context.clearance` (spec
2026-10-16 nvblox object-aware §3.3, `SafetyHarness.drain_clearance_reports`).
This rebuilds an occupancy stand-in from each snapshot (`ReplayMap`: the
voxels near the offending sample, exact distances to their cubes) and re-runs
the arm's own SafetyHarness at the recorded joints, per lever (spec §6.3):

- `old`: the static strict map check at `q` -- vet_pose's map gates (TCP and
  joint origins, the held payload, the tool volume) under the recorded
  exemption (its z_max ceiling included), no escape rule;
- `L1`: `vet_step(q_prev, q)`, today's escape rules (TCP/links strictly
  retreating, the tool-volume retreat); the static check without a `q_prev`
  (vet_pose and route_end reports). dt = 0: the velocity cap is not replayed.
  A report the live gate filed while running this very rule (phase approve
  or vet_step) stays refused: a replayed admission there is the snapshot
  model's, flagged `approximate: ["L1 kept as recorded"]`. L1 admits only a
  recorded static refusal given an earlier `q_prev` (a pre-PR-A trace);
- `L4`: `old` with an override tool volume (`--tool-volume`, the rebuilt
  camera box + measured parts); without one it is `old` with the shipped
  volume.

Each verdict is "allowed" or "refused: <the harness's reason>". The map is
PARTIAL (`partial_map: true`): only the snapshot's voxels exist, at most
`refusal_snapshot.max_voxels` within `radius_m` of the recorded sample, so an
"allowed" means "allowed by what the refusal recorded", never that the move
is safe (lever 4's new parts far from that sample, the housing behind the
camera, see no voxel the snapshot left out). Exactness against the live gate:

- the snapshot rounds voxel centres to 1 mm (<= 0.9 mm per distance);
- a cube distance reads LESS than the map's own, toward "refused": the
  nvblox / EDT grid measures TCP and joint origins to voxel CENTRES (about
  0.5-0.9 voxel more), the tool-volume gate to the centre minus half a voxel
  (at most (sqrt 3 - 1)/2 voxel more);
- the voxels' recorded `d` column is never used (a streamed halt that took a
  report over keeps the first sample's `d`; such a report is flagged
  `approximate: ["takeover"]`, and one whose snapshot is of a newer map than
  the gate read `["snapshot_newer_map"]`).

A report the map refused by RAISING (stale, failed, body mask: `raised`), one
without a measured sample, voxels or voxel size, and one whose joints do not
fit `--arm` is not replayable: every lever reads "refused: not replayable
(why): <recorded reason>", counted as refused.

Never connects to anything (no bridge, arm, camera or service) and never
writes: it reads trace files and the arm profile, prints one JSON line per
report, then the summary `{lever: {allowed, refused}}` (stderr: totals).

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
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))     # this tree's cascade, never another checkout's

from cascade.config import Cfg, load_demo_config
from cascade.safety.harness import SafetyHarness, SafetyLimits
from cascade.safety.tool_volume import build_tool_volume

LEVERS = ("old", "L1", "L4")
#: phases whose live gate already ran the step rule (L1) on the whole map
STEP_PHASES = ("approve", "vet_step")
#: a kept voxel whose recorded `d` differs from its distance to the report's
#: sample by more than this (centres and `d` are rounded to 1 mm) belongs to
#: an earlier sample: a streamed halt took the report over
TAKEOVER_TOL_M = 0.002
#: the override file's blocks
_OVERRIDE_KEYS = {"tool_volume", "wrist_camera"}


class ReplayMap:
    """An occupancy stand-in holding one refusal snapshot's voxels.

    Clearance is the exact Euclidean distance to the nearest voxel CUBE (side
    `voxel_m`, centred on a snapshot voxel; 0 inside, +inf with no voxels).
    The rest of the map surface the harness reads: `clearance_excluding`
    (drops voxels at or below `below_z` and inside the `cylinder`, z_max
    ceiling included, then the exact cube distance: never NaN, `within` is
    not a cap), `payload_points` / `payload_clearance` (the held samples
    given in the TCP frame), `nearby_occupied`, `clearance_record`,
    `refreshes`, `status["voxel"]` (the escape's interior sampling) and a
    REENTRANT `_refresh_lock`: the harness holds it while it files a report
    and then calls `clearance_record` and `nearby_occupied`, which take it
    again. `required = False` (an answer, never a raise) and
    `unknown_blocks = False` (the snapshot holds no observation support).
    """

    required = False
    unknown_blocks = False

    def __init__(self, voxels, voxel_m: float, *, payload_tcp=None):
        voxel_m = float(voxel_m)
        if not (np.isfinite(voxel_m) and voxel_m > 0.0):
            raise ValueError(f"voxel_m must be finite and > 0, got {voxel_m!r}")
        rows = np.asarray(voxels, dtype=float)
        centres = rows.reshape(len(rows), -1)[:, :3] if rows.size else np.zeros((0, 3))
        if centres.shape[1] != 3 or not np.all(np.isfinite(centres)):
            raise ValueError("voxels must be finite rows [x, y, z(, d)]")
        self._occupied = np.array(centres, dtype=float)
        self.voxel_m = voxel_m
        self.status = {"voxel": voxel_m}
        self.refreshes = 0
        self._refresh_lock = threading.RLock()
        held = np.zeros((0, 3)) if payload_tcp is None else np.asarray(payload_tcp, dtype=float)
        self._payload_tcp = held.reshape(-1, 3)

    def _cube_distance(self, points, centres) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if not len(centres):
            return np.full(len(pts), np.inf)
        gap = np.maximum(np.abs(pts[:, None, :] - centres[None, :, :]) - self.voxel_m / 2.0, 0.0)
        return np.linalg.norm(gap, axis=-1).min(axis=1)

    def clearance(self, points) -> np.ndarray:
        return self._cube_distance(points, self._occupied)

    def clearance_excluding(self, points, *, below_z=None, cylinder=None, within=None) -> np.ndarray:
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
        return self._cube_distance(points, cand)

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


def load_reports(paths) -> list[dict]:
    """Every `context.clearance` entry of the given trace.jsonl files (or run
    directories holding one), oldest first, each with its row's `run` (the
    run directory's name), `step` and `skill`. Unparseable lines and rows
    without reports are skipped."""
    out: list[dict] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / "trace.jsonl"
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            context = row.get("context")
            reports = context.get("clearance") if isinstance(context, dict) else None
            if not isinstance(reports, list):
                continue
            for report in reports:
                if isinstance(report, dict):
                    out.append({**report, "run": path.parent.name, "step": row.get("step"),
                                "skill": row.get("skill")})
    return out


# ── the arm ─────────────────────────────────────────────────────────────────


@functools.cache
def _arm_config(arm: str) -> SimpleNamespace:
    """The profile as `apps.demo._build_arm` reads it: the arm's own
    resolved view, its limits and its shipped tool volume (no kinematics)."""
    cfg = load_demo_config(arm=arm, camera="mock", llm="mock")
    acfg = cfg.arm
    view = acfg.get("resolved") or cfg
    return SimpleNamespace(name=arm, acfg=acfg, view=view, n_joints=int(acfg.get("n_joints", 6)),
                           limits=SafetyLimits.from_config(view.safety), volume=build_tool_volume(view, acfg))


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


def _why_not_replayable(report: dict, n: int) -> str | None:
    if report.get("raised"):
        return f"the map raised {report['raised']}: no voxels"
    _, why = _joints(report.get("q"), n, "q")
    if why is None and report.get("q_prev") is not None:
        _, why = _joints(report.get("q_prev"), n, "q_prev")
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
    cylinder = (report.get("exemption") or {}).get("cylinder")
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
        ReplayMap(report["voxels"], voxel)
    except (TypeError, ValueError) as exc:
        return f"bad voxels: {exc}"
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
    payload, the tool volume. No escape rule; no report filed (no ctx)."""
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


def _verdict(reason: str | None) -> str:
    return "allowed" if reason is None else f"refused: {reason}"


def _approximate(report: dict, sample: np.ndarray) -> list[str]:
    flags = []
    if report.get("snapshot_newer_map"):
        flags.append("snapshot_newer_map")
    vox = np.asarray(report["voxels"], dtype=float).reshape(len(report["voxels"]), -1)
    if vox.shape[1] >= 4:
        moved = np.abs(np.linalg.norm(vox[:, :3] - sample, axis=1) - vox[:, 3])
        if float(moved.max()) > TAKEOVER_TOL_M:
            flags.append("takeover")
    extra = sorted(set(report.get("exemption") or {}) - {"cylinder"})
    if extra:
        flags.append(f"exemption keys not replayed: {extra}")
    return flags


def replay(report: dict, *, arm: str, tool_volume_override=None) -> dict:
    """One report re-vetted at its recorded joints on its own snapshot:
    `old`, `L1`, `L4` ("allowed" / "refused: ..."), plus what was recorded,
    the replayed clearance at the recorded sample, `partial_map: True`,
    `replayable` and `approximate`. Never changes `report`."""
    out = {"run": report.get("run"), "step": report.get("step"), "skill": report.get("skill"),
           "phase": report.get("phase"), "gate": report.get("gate"), "recorded": report.get("reason"),
           "clearance_m": report.get("clearance_m"), "replay_clearance_m": None,
           "L4_volume": "shipped" if tool_volume_override is None else "override",
           "partial_map": True, "replayable": True, "approximate": []}
    model = _arm_config(arm)
    why = _why_not_replayable(report, model.n_joints)
    if why is not None:
        refused = f"refused: not replayable ({why}): {report.get('reason')}"
        out.update({lever: refused for lever in LEVERS}, replayable=False, not_replayable=why)
        return out

    kin = _kinematics(arm)
    q = np.asarray(report["q"], dtype=float)
    q_prev = None if report.get("q_prev") is None else np.asarray(report["q_prev"], dtype=float)
    sample = np.asarray(report["sample"], dtype=float).reshape(3)
    exempt = _exemption(report)
    base_pose = report.get("base_pose")
    payload = report.get("sample_tcp") if report.get("gate") == "attached" else None

    def harness(volume) -> SafetyHarness:
        occupancy = ReplayMap(report["voxels"], report["map"]["voxel_m"], payload_tcp=payload)
        return SafetyHarness(model.limits, kinematics=kin, occupancy=occupancy, base_pose=base_pose,
                             tool_volume=volume)

    h = harness(model.volume)
    old = _static_reason(h, q, exempt)
    out["old"] = _verdict(old)
    flags = _approximate(report, sample)
    if q_prev is None:
        out["L1"] = out["old"]
    else:
        out["L1"] = _verdict(_step_reason(h, q_prev, q, exempt))
        if out["L1"] == "allowed" and report.get("phase") in STEP_PHASES:
            # The live gate ran this very rule on the WHOLE map and refused:
            # an admission here comes from the snapshot model (partial map,
            # cube distances read up to ~1 voxel lower, unevenly), so it is
            # never counted as the rule admitting the step.
            out["L1"] = f"refused: {report.get('reason')}"
            flags.append("L1 kept as recorded")
    out["L4"] = (out["old"] if tool_volume_override is None
                 else _verdict(_static_reason(harness(tool_volume_override), q, exempt)))

    gate = str(report.get("gate"))
    if gate in ("tcp", "attached") or gate.startswith("link "):
        d = h.occupancy.clearance(sample[None, :])[0]
    else:   # a tool part: obstacles attributed by where they are, as the gate reads them
        band = 0.0 if model.volume is None else float(model.volume.table_band_m)
        d = h.occupancy.clearance_excluding(sample[None, :], below_z=float(model.limits.table_z) + band,
                                            cylinder=exempt, within=float(model.limits.min_clearance_m))[0]
    out["replay_clearance_m"] = float(d) if np.isfinite(d) else None
    out["approximate"] = flags
    return out


def summarize(results) -> dict:
    """{lever: {"allowed": n, "refused": m}} over replay() results."""
    summary = {lever: {"allowed": 0, "refused": 0} for lever in LEVERS}
    for out in results:
        for lever in LEVERS:
            summary[lever]["allowed" if out[lever] == "allowed" else "refused"] += 1
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
    results = []
    for report in load_reports(args.paths):
        out = replay(report, arm=args.arm, tool_volume_override=override)
        print(json.dumps(out))
        results.append(out)
    print(json.dumps(summarize(results)))
    print(f"{len(results)} report(s): {sum(r['replayable'] for r in results)} replayable, "
          f"{sum(bool(r['approximate']) for r in results)} approximate", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
