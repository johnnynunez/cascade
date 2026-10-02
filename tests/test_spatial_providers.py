"""Metric replay, invalidation and conservative planning through real domains."""
from dataclasses import replace
import math

import pytest

from cascade.spatial.frames import SpatialStamp, TransformSample, FrameTree
from cascade.spatial.grid import GridSnapshot, integrate_scan, plan_route, ray_cells
from cascade.spatial.memory import LandmarkObservation, SpatialMemory
from cascade.config import load_robot_config
from cascade.apps.robot_runtime import build_robot_runtime


def stamp(t=1., **kwargs):
    return SpatialStamp(**{**dict(map_id="room", epoch="one", clock_id="physics", time_s=t,
                               source_id="fixture", source_sha256="a"*64,
                               calibration_id="cal-v1", measurement_kind="synthetic"), **kwargs})


def edge(t=1., **kwargs):
    return TransformSample(**{**dict(parent="map", child="camera", translation_m=(1., 2., 3.),
                                    rotation_wxyz=(1., 0., 0., 0.), stamp=stamp(t)), **kwargs})


def tree():
    frames = FrameTree("room", "one", "physics", "map")
    frames.add(edge())
    return frames


def lookup(frames, target="map", source="camera", time_s=1., **kwargs):
    return frames.lookup(target, source, **{**dict(time_s=time_s, epoch="one", clock_id="physics"), **kwargs})


def test_transform_direction_inverse_rotation_and_capture_time():
    frames = tree()
    frames.add(edge(2., translation_m=(2., 2., 3.), rotation_wxyz=(math.sqrt(.5), 0, 0, math.sqrt(.5))))
    assert lookup(frames).point((1, 0, 0)) == pytest.approx((2, 2, 3))
    point = lookup(frames, time_s=2.).point((1, 0, 0))
    assert point == pytest.approx((2, 3, 3))
    assert lookup(frames, target="camera", source="map", time_s=2.).point(point) == pytest.approx((1, 0, 0))
    # A capture at t=1 must not use the robot's new pose at t=2.
    assert lookup(frames).point((1, 0, 0)) == pytest.approx((2, 2, 3))


@pytest.mark.parametrize("kwargs", [dict(time_s=.9), dict(time_s=1.21), dict(epoch="reset"),
                                         dict(clock_id="wall"), dict(source="implicit-base")])
def test_transform_rejects_missing_stale_reset_or_foreign_clock(kwargs):
    with pytest.raises(ValueError):
        lookup(tree(), **kwargs)


def test_tree_refuses_topology_provider_calibration_replay_and_bounds():
    frames = tree()
    bad = [edge(), edge(.9), edge(2., stamp=stamp(2., calibration_id="new")),
           edge(2., stamp=stamp(2., epoch="reset")), edge(2., stamp=stamp(2., source_id="other")),
           edge(child="map", parent="camera"), edge(child="orphan", parent="unknown")]
    for sample in bad:
        with pytest.raises(ValueError):
            frames.add(sample)
    with pytest.raises(ValueError):
        edge(rotation_wxyz=(2, 0, 0, 0))
    bounded = FrameTree("room", "one", "physics", "map", max_frames=2, history=1)
    bounded.add(edge()); bounded.add(edge(2.))
    with pytest.raises(ValueError):
        bounded.add(edge(2., child="extra"))
    with pytest.raises(ValueError):
        lookup(bounded)


def test_relative_transform_does_not_depend_on_cancelled_ancestor_age():
    frames = tree()
    frames.add(edge(parent="camera", child="lens", translation_m=(0, 0, .1), static=True))
    relative = lookup(frames, target="camera", source="lens", time_s=100.)
    assert relative.point((0, 0, 0)) == pytest.approx((0, 0, .1))
    assert len(relative.samples) == 1
    assert lookup(frames, target="camera", source="camera", time_s=100.).point((1, 2, 3)) == (1, 2, 3)
    with pytest.raises(ValueError, match="stale"):
        lookup(frames, source="lens", time_s=100.)


def test_memory_keeps_identical_labels_separate_and_invalidates_epoch():
    frames = tree()
    memory = SpatialMemory(frames)
    a = LandmarkObservation("view-a", "cup", "camera", (1, 0, 0), stamp(), .9)
    b = replace(a, observation_id="view-b", point_m=(2, 0, 0))
    memory.observe(a); memory.observe(b)
    values = memory.query("cup", epoch="one", clock_id="physics", time_s=1.)
    assert [v["point_map_m"] for v in values] == [[2., 2., 3.], [3., 2., 3.]]
    assert all(not v["physical_admission"] for v in values)
    values[0]["point_map_m"][0] = 100
    assert memory.query("cup", epoch="one", clock_id="physics", time_s=1.)[0]["point_map_m"][0] == 2
    with pytest.raises(ValueError, match="replay"):
        memory.observe(a)
    with pytest.raises(ValueError, match="calibration"):
        memory.observe(replace(a, observation_id="new", stamp=stamp(calibration_id="wrong")))
    with pytest.raises(ValueError, match="context"):
        memory.query("cup", epoch="reset", clock_id="physics", time_s=1.)
    assert not memory.query("cup", epoch="one", clock_id="physics", time_s=100.)


def test_planar_scan_free_occupied_unknown_and_replay_refusal():
    grid = GridSnapshot("map", stamp(0.), .1, (0., 0.), 31, 31, [-1]*(31*31))
    frames = tree()
    result = integrate_scan(grid, frames, frame_id="camera", stamp=stamp(), bearings_rad=[0., math.pi/2],
                            ranges_m=[.5, None], max_range_m=.8)
    assert result.value(result.cell((1.5, 2.))) == 100
    assert result.value(result.cell((1., 2.4))) == 0
    assert result.value(result.cell((.3, .3))) == -1
    assert grid.value(grid.cell((1.5, 2.))) == -1  # immutable previous map
    assert result.sha256 != grid.sha256
    with pytest.raises(ValueError, match="replay"):
        integrate_scan(result, frames, frame_id="camera", stamp=stamp(), bearings_rad=[0], ranges_m=[.5], max_range_m=1.)
    tilted = tree()
    tilted.add(edge(2., rotation_wxyz=(math.sqrt(.5), math.sqrt(.5), 0, 0)))
    with pytest.raises(ValueError, match="horizontal"):
        integrate_scan(grid, tilted, frame_id="camera", stamp=stamp(2.), bearings_rad=[0], ranges_m=[.5], max_range_m=1.)


def test_subcell_rays_do_not_clear_unobserved_obstacle_or_corner_touch():
    grid = GridSnapshot("map", stamp(0.), 1., (0, 0), 3, 3, [-1, -1, -1, 100, -1, -1, -1, -1, -1])
    frames = FrameTree("room", "one", "physics", "map")
    frames.add(edge(translation_m=(.99, .01, 0)))
    dx, dy = .02, 1.79
    result = integrate_scan(grid, frames, frame_id="camera", stamp=stamp(),
                            bearings_rad=[math.atan2(dy, dx)], ranges_m=[math.hypot(dx, dy)], max_range_m=2.)
    assert result.value((0, 1)) == 100  # actual ray never entered this cell
    assert result.value((1, 1)) == 100
    assert result.value((1, 0)) == 0
    assert list(ray_cells((.5, .5), (1.5, 1.5))) == [(0, 0), (1, 1)]
    assert list(ray_cells((1., .5), (0., .5))) == [(0, 0)]
    assert list(ray_cells((.5, .5), (1., .5))) == [(0, 0)]


def test_partial_scan_cannot_refresh_old_free_geometry():
    grid = grid_fixture()
    frames = FrameTree("room", "one", "physics", "map")
    frames.add(edge(100., translation_m=(.5, .5, 0)))
    result = integrate_scan(grid, frames, frame_id="camera", stamp=stamp(100.),
                            bearings_rad=[0], ranges_m=[.2], max_range_m=.5)
    assert result.stamp.time_s == 100. and result.oldest_capture_time_s == 1.
    assert result.capture_sources[0] == grid.stamp
    assert result.transform_sources[0].stamp.time_s == 100.
    with pytest.raises(ValueError, match="stale"):
        plan(result, time_s=100.)


def grid_fixture():
    w, h = 31, 21
    cells = [100 if x == 15 and not 6 <= y <= 14 else 0 for y in range(h) for x in range(w)]
    return GridSnapshot("map", stamp(), .1, (0, 0), w, h, cells)


def plan(grid, **kwargs):
    return plan_route(grid, **{**dict(start_xy_m=(.45, .45), goal_xy_m=(2.65, 1.45),
            footprint_radius_m=.1, clearance_m=.02, position_error_m=.01, time_s=1., epoch="one",
            clock_id="physics", expected_map_sha256=grid.sha256), **kwargs})


def test_plan_routes_through_opening_with_full_clearance_and_identity():
    grid = grid_fixture(); route = plan(grid)
    assert route["map_sha256"] == grid.sha256 and not route["physical_admission"]
    assert route["unknown_policy"] == "blocked" and route["execution"] == "not_executed"
    points = route["waypoints_xy_m"]
    assert points[0] == (.45, .45) and points[-1] == (2.65, 1.45)
    # Independent metric assertion: the circle never touches the wall squares.
    for px, py in points:
        for y in range(grid.height):
            if grid.value((15, y)) != 100:
                continue
            dx = max(1.5-px, 0, px-1.6)
            dy = max(y*.1-py, 0, py-(y+1)*.1)
            assert math.hypot(dx, dy) > .13


@pytest.mark.parametrize("kwargs", [dict(epoch="new"), dict(clock_id="wall"), dict(time_s=3.),
                                        dict(expected_map_sha256="b"*64), dict(max_expansions=1),
                                        dict(start_xy_m=(.01, .01)), dict(position_error_m=2.)])
def test_plan_rejects_unsafe_inputs(kwargs):
    with pytest.raises(ValueError):
        plan(grid_fixture(), **kwargs)


def test_unknown_wall_blocks_route_and_input_cells_cannot_mutate_snapshot():
    grid = grid_fixture(); cells = list(grid.cells)
    for y in range(grid.height):
        cells[y*grid.width+15] = -1
    changed = replace(grid, cells=cells)
    cells[:] = [0]*len(cells)
    with pytest.raises(ValueError, match="no route"):
        plan(changed)


def test_real_composed_profile_memory_frames_plan_and_stop_reset(tmp_path):
    rt, _ = build_robot_runtime(load_robot_config("spatial_replay"), tmp_path)
    try:
        resources = rt.execute("list_resources")
        assert all(r["synthetic"] and r["controller_id"] is None for r in resources["resources"])
        map_data = rt.execute("spatial.get_map")
        args = dict(epoch="episode-1", clock_id="replay-seconds", time_s=2.)
        memories = rt.execute("spatial.recall", {**args, "label": "cup"})
        assert memories["ok"] and len(memories["result"]) == 3
        assert memories["result"][0]["point_map_m"] == pytest.approx(memories["result"][1]["point_map_m"])
        assert memories["result"][2]["point_map_m"] != memories["result"][0]["point_map_m"]
        result = rt.execute("spatial.plan_route", {**args, "expected_map_sha256": map_data["map_sha256"],
                "start_xy_m": [.45, .45], "goal_xy_m": [2.65, 1.45], "footprint_radius_m": .1,
                "clearance_m": .02, "position_error_m": .01})
        assert result["ok"] and not result["physical_admission"]
        rt.stop(); assert rt.reset_stop()["ok"]
        assert rt.execute("spatial.get_map")["map_sha256"] == map_data["map_sha256"]
        assert not rt.execute("spatial.recall", {**args, "epoch": "new-map", "label": "cup"})["ok"]
        assert not rt.motion_skills
    finally:
        rt.close()
