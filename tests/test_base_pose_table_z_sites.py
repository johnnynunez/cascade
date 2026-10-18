"""base_pose Option 2: the grasp path's `table_z` sites on a RAISED base (spec
docs/superpowers/specs/2026-10-17-base-pose-runtime-frame-design.md, §3.7, §6.4).

`safety.table_z` is a base-frame height. Every site that compares it with a
perceived point reads that point in the RUNNING arm's frame, so a base 3 cm
above the table (`base_pose` z 0.03, `safety.table_z: -0.03`, workspace floor
-0.05) measures, floors and boxes the cube exactly as an arm on the table
does. The verifier's lift fallback counts from the WORLD table (R4).

Rig: so101_right at RAISED, camera fixed in the table frame and shifted 10 cm
in -y: mock_small's cube stands on the world table (z = 0) at W (0.198, -0.10),
5 cm tall; in the arm frame it is at B (0.198, 0.12), its top at z 0.02, the
table at z -0.03.
"""

from __future__ import annotations

import numpy as np
import pytest
from base_pose_oracle import (  # noqa: F401  (fixtures by name)
    RAISED,
    lift,
    lifted_tcps,
    make_runtime,
    moved_cfg,
    raised_table,
    raw_arm,
    transform,
)

T = transform(RAISED)
WORLD_OBJECT_XY = (0.198, -0.10)


@pytest.fixture(autouse=True)
def _gate_off(monkeypatch):
    """The analytic grasp on MockArm: the real observed-finger gate needs a
    calibrated Isaac reBot, so CASCADE_OBSERVED_FINGER_GATE is pinned off
    (its on-path is covered by tests/test_base_pose_runtime_frame.py's gate
    double rows)."""
    monkeypatch.delenv("CASCADE_OBSERVED_FINGER_GATE", raising=False)
#: mock_small's cube height (`box_height_m`).
CUBE_M = 0.05


def _raised_rig(*, box_height_m=None, grasp=None):
    cfg = raised_table(moved_cfg("so101_right", RAISED, move_cameras=False, shift_xy=(0.0, -0.10)))
    data = cfg._data
    if box_height_m is not None:
        for cam in [data.get("camera"), *(data.get("cameras") or [])]:
            cam = getattr(cam, "_data", cam)
            if isinstance(cam, dict) and "box_height_m" in cam:
                cam["box_height_m"] = float(box_height_m)
    for key, value in (grasp or {}).items():
        resolved = data["arm"].get("resolved")
        resolved = getattr(resolved, "_data", resolved) or {}
        for section in filter(None, (data.get("grasp"), resolved.get("grasp"))):
            getattr(section, "_data", section)[key] = value
    return cfg


def _fix_spy(rt, monkeypatch):
    """Every fix the grasp planner receives (arm frame)."""
    fixes = []
    real = rt._plan_grasps

    def plan(fix, *a, **k):
        fixes.append(fix)
        return real(fix, *a, **k)

    monkeypatch.setattr(rt, "_plan_grasps", plan)
    return fixes


def test_the_rig_stands_the_cube_on_the_world_table(make_runtime):  # noqa: F811
    """Premise: the arm's own table plane is the world table."""
    rt = make_runtime(_raised_rig())
    assert float(rt.arm.harness.limits.table_z) == pytest.approx(-0.03)
    assert rt._table_z_world() == pytest.approx(0.0, abs=1e-12)


def test_object_height_is_measured_from_the_arms_table(make_runtime, monkeypatch):  # noqa: F811
    rt = make_runtime(_raised_rig())
    notes = []
    real = rt._note_measurement

    def note(**values):
        notes.append(values)
        return real(**values)

    monkeypatch.setattr(rt, "_note_measurement", note)
    rt.execute("grasp_object", {"label": "red cube"})                # main: 0.08 (a world top over B's table)
    heights = [n["object_height_m"] for n in notes if "object_height_m" in n]
    assert heights, notes
    assert heights[0] == pytest.approx(CUBE_M, abs=0.003)


def test_grasp_support_height_is_the_world_support_in_the_arm_frame(make_runtime, monkeypatch):  # noqa: F811
    rt = make_runtime(_raised_rig(grasp={"place_support_clearance_m": 0.01}))
    fixes = _fix_spy(rt, monkeypatch)
    supports = []
    real = rt._grasp_support_height

    def support(points):
        supports.append(real(points))
        return supports[-1]

    monkeypatch.setattr(rt, "_grasp_support_height", support)
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is True, result
    assert supports and fixes
    world_low = float(lift(T, fixes[0].points)[:, 2].min())
    assert supports[0] == pytest.approx(max(world_low, 0.0) - 0.03, abs=1e-6)     # main: 0.03 higher


def test_a_low_object_is_grasped_at_the_floor_above_the_world_table(make_runtime):  # noqa: F811
    """A 1 cm tall cube: its centre (z 0.005) is under the grasp floor; the
    TCP descends to `min_grasp_z_m` (0.008) above the WORLD table."""
    rt = make_runtime(_raised_rig(box_height_m=0.01))
    n0 = len(raw_arm(rt.arm).commands)
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is True, result
    tcps = lifted_tcps(rt, rt.arm, n0, T)
    assert len(tcps)
    lowest = float(tcps[:, 2].min())
    assert lowest == pytest.approx(float(rt.cfg.grasp.get("min_grasp_z_m", 0.008)), abs=0.002)    # main: ~3 cm higher


def test_the_clearance_box_sits_on_the_world_table(make_runtime, monkeypatch):  # noqa: F811
    """so101 has no clearance hulls (the B73 vet never runs on it): the box
    the vet would build from this grasp's fix and support plane (the arm's
    `table_z`, as `_angled_clearance_check` passes it) stands on the world
    table. The cloud is cut to the cube's top face: a top-only cloud cannot
    locate the bottom, so the box reaches down to the support plane (with
    side points the box ends at the lowest one, in either frame)."""
    import dataclasses

    from cascade.grasping import gripper_clearance as gc

    rt = make_runtime(_raised_rig())
    fixes = _fix_spy(rt, monkeypatch)
    rt.execute("grasp_object", {"label": "red cube"})
    assert fixes
    points = np.asarray(fixes[0].points, dtype=float)
    top_only = dataclasses.replace(fixes[0], points=points[points[:, 2] >= points[:, 2].max() - 0.004])
    box = gc.target_box(top_only, float(rt.arm.harness.limits.table_z))
    bottom = lift(T, [*box.centre[:2], float(box.centre[2] - box.half[2])])
    assert bottom[2] == pytest.approx(0.0, abs=1e-3)
    top = lift(T, [*box.centre[:2], float(box.centre[2] + box.half[2])])
    assert top[2] == pytest.approx(CUBE_M, abs=0.003)


class _BoxGrounder:
    """A VLM grounder double that names mock_small's box (the detector missed)."""

    def ground(self, frame, query):
        from cascade.types import Detection

        return Detection(label=query, conf=0.9, bbox=np.array([301, 308, 339, 346], dtype=np.float32))


def test_vlm_bbox_shave_drops_the_table_plane(make_runtime, monkeypatch):  # noqa: F811
    """The VLM's box covers table pixels too; the shave above `table_z` drops
    them in the arm frame: every fix point is above the arm's table, and the
    fix is the cube's height."""
    rt = make_runtime(_raised_rig())
    rt.cfg._data["grounder"] = {"base_url": "http://127.0.0.1:9/v1", "model": "double"}   # never dialled
    rt._grounder = _BoxGrounder()
    rt.cfg._data.setdefault("perception_loop", {})["belief_fallback_age_s"] = -1.0       # no memory fix
    monkeypatch.setattr(rt.detector, "detect", lambda *a, **k: [])
    grounded = []
    real = rt._vlm_ground_fix

    def ground(*a, **k):
        grounded.append(real(*a, **k))
        return grounded[-1]

    monkeypatch.setattr(rt, "_vlm_ground_fix", ground)
    rt.execute("grasp_object", {"label": "red cube"})
    assert grounded and grounded[0] is not None, "the VLM grounded the cube"
    fix = grounded[0][1]
    z = np.asarray(fix.points, dtype=float)[:, 2]
    assert float(z.min()) > -0.03 + 0.005, z.min()
    assert float(z.max() - z.min()) <= CUBE_M + 0.003


def test_the_verifier_lift_counts_from_the_world_table(make_runtime, monkeypatch):  # noqa: F811   (R4)
    """With no comparable start pose the lift check counts from the table:
    the RUNNING arm's world table (z 0), not its base-frame `table_z` (-0.03):
    a cube centre at world z 0.025 rose 2.5 cm, not 5.5 cm."""
    rt = make_runtime(_raised_rig())
    monkeypatch.setattr(rt.effects, "_comparable_start", lambda before, channel: None)
    after = [*WORLD_OBJECT_XY, 0.025]
    monkeypatch.setattr(rt.effects, "_best_pose", lambda label: (list(after), "belief"))
    result = rt.execute("grasp_object", {"label": "red cube"})
    assert result["ok"] is True, result
    measured = result["postcondition"]["measured"]
    assert measured["z_before"] == pytest.approx(0.0, abs=1e-9), result["postcondition"]
    assert measured["rise_m"] == pytest.approx(0.025, abs=1e-9)
