"""CPU contract tests; the SDK double is not native/GPU acceptance evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from cascade.config import Cfg
from cascade.planning import PlanningError, make_motion_planner
from cascade.planning import cumotion as adapter


class FakeSDK:
    """Only the cuMotion 1.1.0 interfaces used by the adapter are implemented."""

    Obstacle = NS(Type=NS(CUBOID="cuboid"), Attribute=NS(SIDE_LENGTHS="size"))
    TrajectoryOptimizer = NS(CSpaceTarget=lambda q: q.copy(),
                             Results=NS(Status=NS(SUCCESS="success")))
    Pose3 = staticmethod(lambda T: T.copy())

    def __init__(self):
        self.names = ("b", "a")
        self.base, self.tools = "base", ["tip"]
        self.parameters, self.obstacles, self.requests = [], [], []
        self.world_ready = False
        self.status = "success"
        self.failure = None
        self.mutate_trajectory = lambda t: None
        self.collision = False
        self.spheres = 4
        self.parameter_ok = True
        self.kin = NS(base_frame_name=lambda: self.base,
                      cspace_coord_limits=lambda i: NS(lower=-2., upper=2.),
                      cspace_coord_velocity_limit=lambda i: 2.)
        self.robot = NS(kinematics=lambda: self.kin,
                        cspace_coord_name=lambda i: self.names[i],
                        num_cspace_coords=lambda: len(self.names),
                        tool_frame_names=lambda: self.tools)

    def load_robot_from_memory(self, xrdf, urdf):
        self.loaded = (xrdf, urdf)
        return self.robot

    def create_world(self):
        def update():
            self.world_ready = True
        return NS(add_obstacle=lambda o, T: self.obstacles.append((o, T)),
                  add_world_view=lambda: NS(update=update))

    def create_obstacle(self, kind):
        obj = NS(kind=kind)
        obj.set_attribute = lambda key, value: setattr(obj, key, value.copy())
        return obj

    def create_robot_world_inspector(self, robot, view):
        assert self.world_ready
        return NS(num_world_collision_spheres=lambda: self.spheres,
                  num_self_collision_spheres=lambda: self.spheres,
                  in_self_collision=lambda q: self.collision,
                  in_collision_with_obstacle=lambda q: False)

    def create_default_trajectory_optimizer_config(self, robot, tool, view):
        assert self.world_ready and robot is self.robot and tool in self.tools
        def set_param(key, value):
            self.parameters.append((key, value))
            return self.parameter_ok
        return NS(set_param=set_param)

    def create_trajectory_optimizer(self, config):
        def plan(start, goal):
            self.requests.append((start.copy(), goal.copy()))
            if self.failure:
                raise self.failure
            trajectory = NS(
                num_cspace_coords=lambda: len(start),
                domain=lambda: NS(lower=7., upper=8.),
                min_position=lambda: np.minimum(start, goal),
                max_position=lambda: np.maximum(start, goal),
                max_velocity_magnitude=lambda: np.abs(goal - start),
                eval=lambda t, derivative_order=0: (
                    start + (goal - start) * (t - 7) if derivative_order == 0 else goal - start),
            )
            self.mutate_trajectory(trajectory)
            return NS(status=lambda: self.status, trajectory=lambda: trajectory)
        return NS(plan_to_cspace_target=plan)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    urdf, xrdf = tmp_path / "robot.urdf", tmp_path / "robot.xrdf"
    urdf.write_text("<robot name='contract-double'/>")
    xrdf.write_text("format: xrdf\nformat_version: 1.0\n")
    config = dict(type="cumotion", urdf=str(urdf), xrdf=str(xrdf),
                  joint_names=["a", "b"], joint_signs=[-1, 1],
                  base_frame="base", tool_frame="tip", obstacles=[], sample_dt_s=0.2)
    sdk = FakeSDK()
    monkeypatch.setattr(adapter, "_sdk", lambda: sdk)
    return config, sdk


def test_public_factory_maps_names_signs_and_nonzero_time_origin(setup):
    cfg, sdk = setup
    start, goal = np.array([.2, .3]), np.array([-.2, .4])
    with make_motion_planner(Cfg(cfg)) as planner:
        result = planner.plan(start, goal)
        record = result.as_dict()
        np.testing.assert_allclose(sdk.requests[0][0], [.3, -.2])
        np.testing.assert_allclose(sdk.requests[0][1], [.4, .2])
        np.testing.assert_allclose(result.positions[0], start)
        np.testing.assert_allclose(result.positions[-1], goal)
        np.testing.assert_allclose(result.velocities, np.tile([-.4, .1], (6, 1)))
        assert result.times_s == pytest.approx([0, .2, .4, .6, .8, 1])
        assert record["execution_authorized"] is False and record["status"] == "candidate"
        assert sdk.parameters == [("enable_self_collision", True), ("enable_world_collision", True),
                                  ("enable_self_collision", True), ("enable_world_collision", True),
                                  ("trajopt/pbo/enabled", False),
                                  *list(adapter.PATH_POSITION_WEIGHTS.items())]
    assert planner._optimizer is None and planner._world is None and planner._robot is None
    with pytest.raises(PlanningError, match="closed"):
        planner.plan(start, goal)
    np.testing.assert_array_equal(start, [.2, .3])
    np.testing.assert_array_equal(goal, [-.2, .4])
    json.dumps(record, allow_nan=False)


def test_obstacles_model_and_result_are_copied_and_hashed(setup):
    cfg, sdk = setup
    box = dict(name="table", size_m=[1., 2., .1], T_base_box=np.eye(4).tolist())
    box["T_base_box"][2][3] = -.05
    cfg["obstacles"] = [box]
    with make_motion_planner(cfg) as planner:
        first = planner.plan([0, 0], [.1, .1])
        box["size_m"][0] = 99
        box["T_base_box"][2][3] = 99
        Path(cfg["urdf"]).write_text("changed after construction")
        second = planner.plan([0, 0], [.1, .1])
        assert first == second
        assert sdk.obstacles[0][0].size.tolist() == [1, 2, .1]
        assert sdk.obstacles[0][1][2, 3] == -.05
        assert "contract-double" in sdk.loaded[1]
    with make_motion_planner(cfg) as changed:
        assert changed.model_sha256 != first.model_sha256
        assert changed.scene_sha256 != first.scene_sha256


@pytest.mark.parametrize("patch,match", [
    ({"type": "auto"}, "explicitly"),
    ({"device": "cuda:1"}, "unknown"),
    ({"joint_names": ["a", "a"]}, "unique"),
    ({"joint_names": ["a", "missing"]}, "c-space"),
    ({"joint_signs": [True, 1]}, "joint_signs"),
    ({"joint_signs": [0, 1]}, "joint_signs"),
    ({"base_frame": "other"}, "base_frame mismatch"),
    ({"tool_frame": "other"}, "tool_frame"),
    ({"obstacles": None}, "explicitly"),
    ({"obstacles": [{"type": "mesh"}]}, "cuboid"),
    ({"sample_dt_s": float("nan")}, "finite"),
    ({"max_samples": True}, "max_samples"),
    ({"joint_margin": -1}, "nonnegative"),
    ({"max_duration_s": 0}, "positive"),
    ({"urdf": "/nonexistent/cumotion-test-model"}, "readable"),
])
def test_invalid_configuration_fails_without_solver_calls(setup, patch, match):
    cfg, sdk = setup
    cfg.update(patch)
    with pytest.raises(PlanningError, match=match):
        make_motion_planner(cfg)
    assert not sdk.requests


@pytest.mark.parametrize("kind", ["reflection", "scale", "nan", "size", "duplicate"])
def test_invalid_cuboids_rejected_before_sdk(setup, kind, monkeypatch):
    cfg, _ = setup
    box = dict(name="box", size_m=[1, 1, 1], T_base_box=np.eye(4).tolist())
    if kind == "reflection": box["T_base_box"][0][0] = -1
    if kind == "scale": box["T_base_box"][1][1] = 2
    if kind == "nan": box["T_base_box"][2][3] = float("nan")
    if kind == "size": box["size_m"][0] = 0
    cfg["obstacles"] = [box, copy.deepcopy(box)] if kind == "duplicate" else [box]
    monkeypatch.setattr(adapter, "_sdk", lambda: pytest.fail("SDK loaded for invalid geometry"))
    with pytest.raises(PlanningError):
        make_motion_planner(cfg)


@pytest.mark.parametrize("input", [[float("nan"), 0], [0], [[0, 0]], [2., 0], [float("inf"), 0]])
def test_invalid_requests_never_enter_native_solver(setup, input):
    cfg, sdk = setup
    with make_motion_planner(cfg) as planner:
        with pytest.raises(PlanningError):
            planner.plan(input, [0, 0])
        assert not sdk.requests
        planner.plan([0, 0], [.1, .1])  # bad caller data does not poison the SDK


@pytest.mark.parametrize("fault,match", [
    ("status", "TRAJECTORY_OPTIMIZATION_FAILURE"),
    ("exception", "native error"),
    ("zero_duration", "duration"),
    ("huge_duration", "duration"),
    ("nan_domain", "finite"),
    ("wrong_count", "count"),
    ("nan_position", "finite"),
    ("limit", "extrema"),
    ("velocity", "extrema"),
    ("collision", "collision"),
    ("endpoint", "endpoints"),
    ("jump", "between samples"),
    ("samples", "sample budget"),
])
def test_native_failure_never_publishes_or_retries(setup, fault, match):
    cfg, sdk = setup
    if fault == "status": sdk.status = "TRAJECTORY_OPTIMIZATION_FAILURE"
    if fault == "exception": sdk.failure = RuntimeError("native error")
    if fault == "collision": sdk.collision = True
    if fault == "samples": cfg["max_samples"] = 2
    def corrupt(t):
        if fault == "zero_duration": t.domain = lambda: NS(lower=7., upper=7.)
        if fault == "huge_duration": t.domain = lambda: NS(lower=7., upper=1000.)
        if fault == "nan_domain": t.domain = lambda: NS(lower=float("nan"), upper=8.)
        if fault == "wrong_count": t.num_cspace_coords = lambda: 99
        if fault == "nan_position": t.eval = lambda *_: np.array([float("nan"), 0])
        if fault == "limit": t.max_position = lambda: np.array([4., 0])
        if fault == "velocity": t.max_velocity_magnitude = lambda: np.array([5., 0])
        if fault == "endpoint": t.eval = lambda *_: np.array([.5, .5])
        if fault == "jump":
            original = t.eval
            t.eval = lambda time, order: np.array([1.5, 0]) if order == 0 and 7.3 < time < 7.7 else original(time, order)
    sdk.mutate_trajectory = corrupt
    with make_motion_planner(cfg) as planner:
        with pytest.raises(PlanningError, match=match):
            planner.plan([0, 0], [.1, .1])
        with pytest.raises(PlanningError, match="faulted"):
            planner.plan([0, 0], [.1, .1])
        assert len(sdk.requests) == 1


@pytest.mark.parametrize("fault", ["geometry", "parameter", "limits"])
def test_incomplete_native_capability_is_not_accepted(setup, fault):
    cfg, sdk = setup
    if fault == "geometry": sdk.spheres = 0
    if fault == "parameter": sdk.parameter_ok = False
    if fault == "limits": sdk.kin.cspace_coord_velocity_limit = lambda _: float("inf")
    with pytest.raises(PlanningError):
        make_motion_planner(cfg)
    assert not sdk.requests


def test_loader_reports_missing_and_wrong_version_without_import(monkeypatch):
    def missing(_):
        raise adapter.metadata.PackageNotFoundError("cumotion")
    monkeypatch.setattr(adapter.metadata, "version", missing)
    monkeypatch.setattr(adapter, "import_module", lambda _: pytest.fail("unexpected import"))
    with pytest.raises(PlanningError, match="SDK unavailable"):
        adapter._sdk()
    monkeypatch.setattr(adapter.metadata, "version", lambda _: "9.9.9")
    with pytest.raises(PlanningError, match="1.1.0 required"):
        adapter._sdk()


def test_public_cli_writes_candidate_and_protects_existing_output(setup, tmp_path):
    from cascade.apps.plan_motion import main
    cfg, _ = setup
    cfg["urdf"], cfg["xrdf"] = "robot.urdf", "robot.xrdf"
    config, output = tmp_path / "planner.yaml", tmp_path / "plan.json"
    config.write_text(yaml.safe_dump(cfg))
    args = ["--config", str(config), "--start", "0", "0", "--goal", ".1", ".1", "--output", str(output)]
    assert main(args) == 0
    before = output.read_bytes()
    assert json.loads(before)["execution_authorized"] is False
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_cli_failure_leaves_no_candidate_file(setup, tmp_path):
    from cascade.apps.plan_motion import main
    cfg, sdk = setup
    sdk.status = "FAIL"
    config, output = tmp_path / "planner.yaml", tmp_path / "plan.json"
    config.write_text(yaml.safe_dump(cfg))
    assert main(["--config", str(config), "--start", "0", "0", "--goal", ".1", ".1", "--output", str(output)]) == 1
    assert not output.exists()


def test_optional_imports_and_cli_help_do_not_load_sdk():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'cumotion', 'torch', 'pinocchio', 'isaacsim'}:
            raise AssertionError('unexpected optional import: ' + name)
sys.meta_path.insert(0, Block())
from cascade.planning import make_motion_planner
from cascade.apps.plan_motion import main
main(['--help'])
"""
    completed = subprocess.run([sys.executable, "-c", code, source], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
