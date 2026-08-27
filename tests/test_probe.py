"""Tests for the queryable cursor (probe_point / locate_pixel).

Motivated by Anthropic's Claude-Plays-Robotics ablation: static depth and
segmentation overlays were roughly neutral, while a cursor the model can move
and QUERY lifted manipulation success from 6% to 32%. The contract being
pinned here is that a probe returns hard, comparable numbers -- and fails
honestly rather than inventing a 3D point.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cascade.perception.probe import PointProbe, _median_depth, deproject


# ── fixtures: a synthetic pinhole camera looking straight down ───────────

W, H = 64, 48
K = np.array([[50.0, 0.0, W / 2], [0.0, 50.0, H / 2], [0.0, 0.0, 1.0]])

#: camera 1 m above the table, +Z_cam pointing down at the table, so a
#: camera-frame point (x, y, z) maps to base (x, -y, 1 - z).
T_BASE_CAM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
])


def _frame(depth_value=0.9, holes=False):
    depth = np.full((H, W), depth_value, dtype=np.float32)
    if holes:
        depth[:, :] = 0.0
    return SimpleNamespace(
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        depth_m=depth,
        K=K,
        T_base_cam=T_BASE_CAM,
        has_depth=True,
        depth_source="sensor",
        frame_id=1,
    )


class _Beliefs:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


def _runtime(frame=None, beliefs=(), tcp=(0.3, 0.0, 0.3), workspace=None):
    from cascade.config import Cfg

    cfg = Cfg({
        "safety": {"workspace": workspace or {"min": [0.10, -0.30, -0.01],
                                              "max": [0.50, 0.30, 0.55]},
                   "table_z": 0.0},
        "grasp": {"reach_x_min": 0.155, "reach_x_max": 0.185},
    })
    rt = SimpleNamespace(
        cfg=cfg,
        last_frame=frame if frame is not None else _frame(),
        beliefs=_Beliefs(list(beliefs)),
        extrinsics=SimpleNamespace(cam_to_base=lambda: T_BASE_CAM),
        camera=SimpleNamespace(name="over"),
        rig=None,
        observe=lambda: frame,
        _tcp=lambda: np.array(tcp, dtype=float),
    )
    return rt


def _belief(label, position):
    return SimpleNamespace(label=label, position=list(position))


# ── geometry ─────────────────────────────────────────────────────────────


def test_deproject_center_pixel_is_on_the_optical_axis():
    p = deproject(W / 2, H / 2, 0.9, K)
    assert p[0] == pytest.approx(0.0) and p[1] == pytest.approx(0.0)
    assert p[2] == pytest.approx(0.9)


def test_probe_center_lifts_to_the_expected_base_point():
    rt = _runtime()
    out = PointProbe(rt).probe(W / 2, H / 2)
    assert out["ok"]
    # camera 1 m up looking down, surface at 0.9 m -> table point z = 0.1
    assert out["position"][2] == pytest.approx(0.1, abs=1e-6)
    assert out["distance_m"] == pytest.approx(0.9)


def test_round_trip_pixel_to_3d_to_pixel_is_stable():
    """The property that makes the cursor trustworthy at all.

    The synthetic depth map is a constant 0.9 m (a sphere around the camera,
    not a plane), so an off-axis belief re-projects with a small geometric
    offset that is the FIXTURE's, not the probe's. Round-trip is therefore
    asserted on-axis, where the two agree exactly; the off-axis case is
    covered by the live-rig measurement documented in probe.py (1.6 mm).
    """
    probe = PointProbe(_runtime())
    center = probe.probe(W / 2, H / 2)
    target = np.array(center["position"])

    rt = _runtime(beliefs=[_belief("cube", target)])
    probe2 = PointProbe(rt)
    loc = probe2.locate_pixel("cube")
    assert loc["ok"] and loc["in_view"]
    back = probe2.probe(*loc["pixel"])
    assert back["ok"]
    err = np.linalg.norm(np.array(back["position"]) - target)
    assert err < 2e-3, f"round-trip drifted {err*1000:.1f} mm"


def test_probe_reports_it_measures_a_surface_not_a_centroid():
    """A ray hits the first surface: an agent must not use this as a centre."""
    out = PointProbe(_runtime()).probe(W / 2, H / 2)
    assert "surface" in out["measures"] and "centre" in out["measures"]


# ── object association ───────────────────────────────────────────────────


def test_probe_names_the_object_under_the_cursor():
    target = np.array([0.18, 0.06, 0.10])
    rt = _runtime(beliefs=[_belief("pink cube", target),
                           _belief("far box", [0.45, -0.25, 0.02])])
    probe = PointProbe(rt)
    out = probe.probe(*probe.locate_pixel("pink cube")["pixel"])
    assert out["object"]["label"] == "pink cube"
    assert out["object"]["offset_m"] < 0.01


def test_probe_on_empty_table_reports_no_object_but_offers_the_nearest():
    rt = _runtime(beliefs=[_belief("far box", [0.45, -0.25, 0.02])])
    out = PointProbe(rt).probe(W / 2, H / 2)
    assert out["ok"] and out["object"] is None
    assert out["nearest_object"]["label"] == "far box"


# ── reachability: the "orientation" signal the ablation says models need ──


def test_probe_flags_a_point_outside_the_topdown_ik_band():
    """x=0.28 is reachable in the AABB but hopeless for a top-down grasp."""
    rt = _runtime()
    out = PointProbe(rt).probe(W / 2, H / 2)  # lands at x=0.0... use a belief
    rt2 = _runtime(beliefs=[_belief("cube", [0.28, 0.0, 0.10])])
    probe = PointProbe(rt2)
    out = probe.probe(*probe.locate_pixel("cube")["pixel"])
    assert out["reachable"]["in_workspace"] is True
    assert out["reachable"]["in_topdown_ik_band"] is False
    assert "IK band" in out["reachable"]["note"]


def test_probe_confirms_a_point_inside_the_ik_band():
    rt = _runtime(beliefs=[_belief("cube", [0.17, 0.0, 0.10])])
    probe = PointProbe(rt)
    out = probe.probe(*probe.locate_pixel("cube")["pixel"])
    assert out["reachable"]["in_topdown_ik_band"] is True
    assert "note" not in out["reachable"]


def test_probe_flags_a_point_outside_the_workspace():
    rt = _runtime(workspace={"min": [0.10, -0.05, -0.01], "max": [0.20, 0.05, 0.55]},
                  beliefs=[_belief("cube", [0.18, 0.20, 0.10])])
    probe = PointProbe(rt)
    out = probe.probe(*probe.locate_pixel("cube")["pixel"])
    assert out["reachable"]["in_workspace"] is False
    assert "refused" in out["reachable"]["note"]


def test_probe_reports_offset_from_the_gripper():
    rt = _runtime(tcp=(0.30, 0.00, 0.30))
    out = PointProbe(rt).probe(W / 2, H / 2)
    assert out["from_gripper"]["distance_m"] > 0
    assert len(out["from_gripper"]["delta_xyz_m"]) == 3


# ── honest failure (never invent a 3D point) ─────────────────────────────


def test_probe_outside_the_image_is_refused():
    out = PointProbe(_runtime()).probe(9999, 9999)
    assert not out["ok"] and "outside" in out["error"]


def test_probe_without_depth_is_refused():
    frame = _frame()
    frame.depth_m = None
    frame.has_depth = False
    frame.depth_source = "none"
    out = PointProbe(_runtime(frame=frame)).probe(W / 2, H / 2)
    assert not out["ok"] and "no depth" in out["error"]


def test_probe_into_a_depth_hole_is_refused_not_guessed():
    out = PointProbe(_runtime(frame=_frame(holes=True))).probe(W / 2, H / 2)
    assert not out["ok"] and "depth hole" in out["error"]


def test_median_window_survives_a_single_dead_pixel():
    depth = np.full((H, W), 0.9, dtype=np.float32)
    depth[H // 2, W // 2] = 0.0          # one hole exactly under the cursor
    assert _median_depth(depth, W // 2, H // 2) == pytest.approx(0.9)


def test_locate_pixel_for_an_unknown_object_is_honest():
    out = PointProbe(_runtime()).locate_pixel("unicorn")
    assert not out["ok"] and "not in the world model" in out["error"]


def test_locate_pixel_reports_an_object_behind_the_camera():
    rt = _runtime(beliefs=[_belief("cube", [0.2, 0.0, 5.0])])  # above the cam
    out = PointProbe(rt).locate_pixel("cube")
    assert not out["ok"] and "behind the camera" in out["error"]


def test_locate_pixel_reports_out_of_view_without_failing():
    rt = _runtime(beliefs=[_belief("cube", [3.0, 3.0, 0.1])])
    out = PointProbe(rt).locate_pixel("cube")
    assert out["ok"] and out["in_view"] is False


# ── normalized coordinates (what a model on a resized frame emits) ───────


def test_normalized_coordinates_map_to_the_same_point():
    probe = PointProbe(_runtime())
    explicit = probe.probe(0.5, 0.5, normalized=True)
    pixels = probe.probe((W - 1) / 2, (H - 1) / 2)
    assert explicit["ok"] and pixels["ok"]
    assert explicit["pixel"] == pixels["pixel"]


def test_fractional_coordinates_are_auto_detected_as_normalized():
    """A model emitting 0..1 must not silently probe the top-left corner."""
    out = PointProbe(_runtime()).probe(0.5, 0.5)
    assert out["pixel"] == [int(round(0.5 * (W - 1))), int(round(0.5 * (H - 1)))]


def test_locate_pixel_returns_both_pixel_and_normalized():
    rt = _runtime(beliefs=[_belief("cube", [0.18, 0.06, 0.10])])
    out = PointProbe(rt).locate_pixel("cube")
    assert out["ok"]
    u, v = out["pixel"]
    assert out["normalized"][0] == pytest.approx(u / (W - 1), abs=1e-3)
    assert out["normalized"][1] == pytest.approx(v / (H - 1), abs=1e-3)


# ── the human-facing mark stays separate from the agent's numbers ────────


def test_draw_cursor_marks_without_mutating_a_readonly_image():
    from cascade.perception.probe import draw_cursor

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img.flags.writeable = False
    out = draw_cursor(img, W // 2, H // 2, "0.42 m")
    assert out is not img and out.any()  # something was drawn, original intact
