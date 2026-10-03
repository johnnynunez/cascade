"""CPU authoring and binding controls, never native dynamics/admission."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.apps.factory_runtime import prepare_factory_model, validate_factory_profile
from cascade.config import load_robot_config
from cascade.control.fastening import FasteningFault
from cascade.sim.factory_model import FactoryBoundModel, authoring_descriptor, check_authored_joint, model_fingerprint
from cascade.sim.factory_recipe import LEGACY_RECIPE, MARGIN_RECIPE, seating_recipe
from cascade.sim.newton_screw_seating import SeatingScene
from test_factory_config import no_sdk
from test_factory_owner import model_fixture
from test_fastening_runtime import binding


@pytest.mark.parametrize("profile,recipe", [("factory_m20_mounted", LEGACY_RECIPE),
    ("factory_m20_mounted_margin_v2", MARGIN_RECIPE)])
def test_recipes_remain_separate_unprepared_passive_profiles(monkeypatch, profile, recipe):
    from cascade.apps.robot_runtime import describe_robot
    no_sdk(monkeypatch)
    cfg = load_robot_config(profile)
    value = cfg.domains.fastening.as_dict()
    assert value["recipe"] == recipe
    assert value["device"] is value["model_identity_sha256"] is None
    validate_factory_profile(value)
    assert describe_robot(cfg)["fastening"].resources[0].admission == "unvalidated"


@pytest.mark.parametrize("recipe", [LEGACY_RECIPE, MARGIN_RECIPE])
def test_preparation_passes_the_actual_configured_recipe_without_inferred_defaults(monkeypatch, tmp_path, recipe):
    import cascade.sim.factory_observation as observation
    import cascade.sim.factory_model as model
    import cascade.sim.newton_screw_seating as seating
    monkeypatch.setattr(observation, "sdk_sources", lambda: {})
    seen = []
    def construct(*args, **kwargs):
        seen.append(kwargs)
        return NS(model=NS(device=NS(is_cuda=True)))
    monkeypatch.setattr(seating, "SeatingScene", construct)
    monkeypatch.setattr(model, "FactoryBoundModel", lambda scene: scene)
    profile = load_robot_config("factory_m20_mounted").domains.fastening.as_dict()
    prepare_factory_model(profile | {"device":"cuda:0", "recipe":recipe}, tmp_path)
    assert seen == [{"device":"cuda:0", "drive":False, "substeps":10, "recipe":recipe}]


@pytest.mark.parametrize("bad", [None, True, "v2", "factory_m20_fixed_axis_v3"])
def test_unknown_recipe_refuses_before_native_import_or_scene_construction(monkeypatch, bad):
    no_sdk(monkeypatch)
    with pytest.raises(ValueError, match="recipe"):
        SeatingScene("missing", "missing", "missing", recipe=bad)


def test_v2_binding_requires_the_declared_origin_and_keeps_pitch_fixed():
    original = binding()
    new = replace(original, fixture_recipe=MARGIN_RECIPE, fixture_origin_m=(.23, 0., 0.))
    assert new.sha256 != original.sha256
    for changed in ({"fixture_origin_m":(.24, 0., 0.)}, {"thread_pitch_m":.003}):
        with pytest.raises(ValueError): replace(new, **changed)


@pytest.mark.parametrize("recipe", [LEGACY_RECIPE, MARGIN_RECIPE])
def test_seat_ring_uses_the_same_declared_xy_as_the_thread_fixture(monkeypatch, tmp_path, recipe):
    # Configuration-only doubles; no mesh/collision/force claim.
    scene = object.__new__(SeatingScene)
    scene._configure_recipe(recipe)
    scene._center_xy = np.array(scene.fixture_center_xy_m)
    mesh = NS(vertices=np.zeros((3,3)), faces=np.array([[0,1,2]]))
    monkeypatch.setitem(sys.modules, "trimesh", NS(creation=NS(annulus=lambda **_: mesh)))
    scene.newton = NS(Mesh=lambda *_: NS(build_sdf=lambda **_: None))
    scene.wp = NS(vec3=lambda *v: v, quat_identity=lambda: (0.,0.,0.,1.),
                  transform=lambda p,q: (p,q))
    written = []
    builder = NS(ShapeConfig=lambda **kwargs: kwargs,
                 add_shape_mesh=lambda *args,**kwargs: written.append((args,kwargs)))
    scene._add_fixture(builder,tmp_path)
    args,kwargs = written[0]
    assert args == (-1,)
    assert kwargs['xform'] == ((*scene.fixture_center_xy_m,.0215),(0.,0.,0.,1.))
    assert kwargs['label'] == 'seat_ring'


def test_joint_diagnostic_keeps_the_actual_rejected_values_and_margin():
    position = float(np.float32(1.644682043802835))
    with pytest.raises(FasteningFault) as exc:
        check_authored_joint("wrist_flex", position, -1.65806, 1.65806, .02)
    text = str(exc.value)
    for field in ("joint=wrist_flex", f"q_rad={position:.17g}",
                  f"lower_rad={-1.65806:.17g}", f"upper_rad={1.65806:.17g}", "margin_rad=0.02"):
        assert field in text
    # A margin failure stays a failure; neither clipping nor reinterpretation.
    assert position > 1.65806-.02


@pytest.mark.parametrize("field,value", [
    ("_center_xy", np.array([.23, 0.])), ("fixture_center_xy_m", (.23, 0.)),
    ("fixture_position", np.array([.23, 0., 0.])), ("ik_margin_rad", .02),
    ("intersect_position_control_range", True), ("socket_offset", np.array([.012,0.,-.110])),
    ("_initial_nut_z", .068), ("_heights", np.linspace(.028, .069, 43)),
])
def test_model_rechecks_consumed_authoring_parameters(field, value):
    scene = model_fixture()
    setattr(scene, field, value)
    with pytest.raises(FasteningFault): authoring_descriptor(scene)


@pytest.mark.parametrize("field", ["entry", "bottom", "_targets", "ik_ranges"])
def test_changed_authoring_cannot_keep_the_same_model_binding(field):
    scene = model_fixture()
    model = object.__new__(FactoryBoundModel)
    model.scene, model._fingerprint = scene, model_fingerprint(scene)
    model._authoring = authoring_descriptor(scene)
    model.check_immutable()
    getattr(scene, field).flat[0] += .001
    with pytest.raises(FasteningFault, match="authoring changed"):
        model.check_immutable()


def cpu_scene(recipe):
    mj = pytest.importorskip("mujoco")
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    robot = Path(__file__).resolve().parents[1]/"assets/mjcf/so101/so101.xml"
    if not robot.is_file():
        pytest.skip("CPU SO-101 FK requires scripts/fetch_robot_assets.py so101")
    scene = object.__new__(SeatingScene)
    scene._configure_recipe(recipe)
    scene.mujoco, scene.least_squares, scene.robot_asset = mj, least_squares, robot
    scene._setup_ik()
    scene._initial_nut_z = .069
    scene._center_xy = np.array(scene.fixture_center_xy_m)
    scene.fixture_position = np.r_[scene._center_xy, 0.]
    scene.entry = scene._ik(np.r_[scene._center_xy, .069], [0.,0.,0.,1.6,.17])
    scene.bottom = scene._ik(np.r_[scene._center_xy, .045], scene.entry)
    scene._prepare_follower()
    return scene


def test_legacy_cpu_authoring_retains_original_initial_margin_refusal():
    scene = cpu_scene(LEGACY_RECIPE)
    assert scene.fixture_center_xy_m == (.24,0.) and scene.ik_margin_rad == .001
    assert not scene.intersect_position_control_range
    np.testing.assert_allclose(scene.entry, [.0013053729364881708, -.029871997967254822,
        -.0440137190411909, 1.644682043802835, .08538733142210371], atol=1e-7, rtol=0)
    j = scene.ik_model.joint('wrist_flex')
    ctrl = scene.ik_model.actuator('wrist_flex').ctrlrange
    with pytest.raises(FasteningFault, match="joint=wrist_flex"):
        check_authored_joint('wrist_flex', float(np.float32(scene.entry[3])),
                             max(j.range[0],ctrl[0]), min(j.range[1],ctrl[1]), .02)


def test_v2_all_43_targets_preserve_geometry_refs_residual_and_unchanged_guard_margin():
    scene = cpu_scene(MARGIN_RECIPE)
    assert seating_recipe(MARGIN_RECIPE).ik_margin_rad == .025
    np.testing.assert_array_equal(scene.ik_model.qpos0[scene.ik_indices], np.zeros(5))
    assert scene._targets.shape == (43,5)
    for height, q in zip(scene._heights, scene._targets, strict=True):
        scene.ik_data.qpos[scene.ik_indices] = q
        scene.mujoco.mj_kinematics(scene.ik_model, scene.ik_data)
        r = scene.ik_data.xmat[scene.ik_gripper].reshape(3,3)
        p = scene.ik_data.xpos[scene.ik_gripper]+r@scene.socket_offset
        assert np.linalg.norm(np.r_[p-[.23,0.,height], .15*(r@[0.,0.,-1.]-[0.,0.,-1.])]) <= 1e-5
        for i,name in enumerate(scene.arm_joints):
            lo,hi = scene.ik_ranges[i]
            check_authored_joint(name, float(np.float32(q[i])), lo, hi, .02)
            assert lo+.025 <= q[i] <= hi-.025
    descriptor = authoring_descriptor(scene)
    assert descriptor['center_xy_m'] == (.23,0.)
    assert descriptor['fixture_origin_m'] == [.23,0.,0.]
    assert descriptor['follower_targets_rad'] == scene._targets.tolist()


def test_v2_cannot_claim_the_legacy_target_reachable_by_clipping():
    scene = cpu_scene(MARGIN_RECIPE)
    with pytest.raises(ValueError, match="not reachable"):
        scene._ik(np.array([.24,0.,.069]), [0.,0.,0.,1.6,.17])


def test_nonzero_reference_is_not_silently_copied_to_newton_initial_q(monkeypatch):
    scene = cpu_scene(MARGIN_RECIPE)
    model = scene.ik_model
    model.qpos0[scene.ik_indices[0]] = .1
    monkeypatch.setattr(scene.mujoco.MjModel, "from_xml_path", lambda _: model)
    with pytest.raises(ValueError, match="zero arm joint references"):
        scene._setup_ik()
