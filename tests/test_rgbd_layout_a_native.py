"""Passive entrypoint, USD appearance and model-binding controls for layout A."""

import copy
import json
from types import SimpleNamespace as NS

import pytest

from benchmark.rgbd import layout_a_native_bridge as module
from benchmark.rgbd.binary_layout_a import BinaryLayoutABoard
from benchmark.rgbd.binary_reference import detector_recipe
from benchmark.rgbd.ground_texture import MATERIAL, validate_ground_texture
from cascade.sim.mobile_identity import build_model_identity
from test_mobile_identity import recipe_inputs as recipe_inputs

SOURCE = module.REPO


@pytest.fixture(autouse=True)
def current_cpu_consumer(monkeypatch, tmp_path):
    """Bind the test interpreter, never require Linux bytes on macOS/ARM."""
    root = tmp_path / "recipe"
    for name in module.SOURCES:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((SOURCE / name).read_bytes())
    (root / module.CONSUMER_SOURCE).write_bytes(
        module.common.canonical(detector_recipe())
    )
    monkeypatch.setattr(module, "REPO", root)


def declaration():
    path = module.REPO / module.CONSUMER_SOURCE
    sha = module.common.sha(path)
    return path, sha, module.load_consumer(path, sha)


def test_fixture_identifies_rotated_geometry_and_new_asset_only():
    _, _, text = declaration()
    descriptor = module.texture_descriptor(text)
    board = module.board_from_fixture(descriptor, text)
    assert type(board) is BinaryLayoutABoard
    assert descriptor["board"]["schema"] == 4
    assert (
        descriptor["texture_sha256"]
        == "2cc54a33ea46e6486353fd057519f93f5fc194443f0ac89e6f22c2a230a3d583"
    )
    for key in (
        "yaw_deg",
        "frame_id",
        "tag_centers_local_m",
        "origin_xyz_m",
        "tag_size_m",
        "observed_resolution",
    ):
        changed = copy.deepcopy(descriptor)
        changed["board"].pop(key)
        with pytest.raises(ValueError, match="consumer declaration"):
            module.board_from_fixture(changed, text)


@pytest.mark.parametrize("kind", ["png", "declaration", "missing"])
def test_changed_input_fails_before_authoring(kind):
    path, sha, text = declaration()
    if kind == "png":
        p = module.REPO / module.TEXTURE
        p.write_bytes(p.read_bytes() + b"\0")
        with pytest.raises(ValueError, match="PNG differs"):
            module.texture_descriptor(text)
    elif kind == "declaration":
        path.write_bytes(path.read_bytes() + b" ")
        with pytest.raises(ValueError, match="source/hash"):
            module.load_consumer(path, sha)
    else:
        path.unlink()
        with pytest.raises(OSError):
            module.load_consumer(path, sha)


def test_cross_sdk_author_can_generate_same_bits_without_claiming_detection():
    _, _, text = declaration()
    original = module.texture_descriptor(text)
    value = json.loads(text)
    value["opencv_version"] = "99.0.0"
    foreign = module.common.canonical(value).decode()
    authored = module.texture_descriptor(foreign)
    assert authored["texture_sha256"] == original["texture_sha256"]
    assert authored["board_sha256"] != original["board_sha256"]
    with pytest.raises(ValueError, match="active implementation"):
        module.board_from_fixture(authored, foreign)


def test_new_model_rehashes_bound_support_geometry_bitmap_and_entrypoint(
    recipe_inputs, monkeypatch
):
    admission, native, paths = recipe_inputs
    original = build_model_identity(admission, native, **paths)
    for name in module.SOURCES:
        p = paths["repo"] / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((module.REPO / name).read_bytes())
    monkeypatch.setattr(module, "REPO", paths["repo"])
    _, _, text = declaration()
    module.extend_admission(admission, original, text)
    current = build_model_identity(admission, native, **paths)
    assert current["model_identity_sha256"] != original["model_identity_sha256"]
    assert set(module.SOURCES) <= set(current["recipe"]["source_sha256"])
    p = paths["repo"] / "benchmark/rgbd/homography_support.py"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        build_model_identity(admission, native, **paths)


def test_check_only_does_not_construct_native_or_write_output(
    monkeypatch, tmp_path, capsys
):
    path, sha, _ = declaration()
    args = NS(camera_rgbd=True, check_only=True, out=tmp_path / "absent")
    bridge = NS(
        parse_args=lambda _: args,
        admit=lambda _: {"source_sha256": {}},
        run=lambda *a, **k: pytest.fail("native"),
    )
    monkeypatch.setattr(module.common, "bind_bridge", lambda: bridge)
    monkeypatch.setattr(
        module.common,
        "reference_identity",
        lambda *a: {"model_identity_sha256": "a" * 64},
    )
    monkeypatch.setattr(module, "reference_backend", lambda *a: pytest.fail("backend"))
    argv = [
        "--reference-identity",
        "unused",
        "--reference-identity-sha256",
        "a" * 64,
        "--reference-model-sha256",
        "a" * 64,
        "--consumer-detector",
        str(path),
        "--consumer-detector-sha256",
        sha,
    ]
    assert module.main(argv) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert (
        receipt["no_native_run"]
        and not receipt["physical_acceptance"]
        and not args.out.exists()
    )
    assert receipt["ground_reference_descriptor"]["board"]["schema"] == 4
    assert (
        receipt["generator"]["role"] == "bitmap_authoring_only_not_consumer_detection"
    )


def test_variant_preserves_full_native_comparison_and_parent_camera_order(
    monkeypatch, tmp_path
):
    path, sha, text = declaration()
    descriptor = module.texture_descriptor(text)
    events = []
    reference = {
        "recipe": {
            "native": {
                "native_labels": ["ground", "foot"],
                "native_body_properties": {"mass": [1.0]},
                "native_model_properties": {"shape_body": [0, 1]},
            },
            "bam": {},
        },
        "model_identity_sha256": "a" * 64,
    }

    class Parent:
        def __init__(self):
            self.receipt = copy.deepcopy(reference["recipe"]["native"]) | {"bam": {}}
            self.admission = {
                "ground_reference_descriptor": descriptor,
                "planar_reference_identity": reference,
            }
            self.args = NS(out=tmp_path)

        def _checkpoint(self):
            events.append("checkpoint")

        def _create_camera(self, stage):
            events.append("camera")

        def open(self):
            self._create_camera(object())
            events.extend(["export", "bootstrap"])

    monkeypatch.setattr(module, "author_checked_layout", lambda *a: descriptor)
    monkeypatch.setattr(
        module.ground, "ground_snapshot", lambda stage: {"appearance": "same"}
    )
    backend = module.reference_backend(Parent, path, sha)()
    backend.open()
    assert events == [
        "checkpoint",
        "checkpoint",
        "camera",
        "checkpoint",
        "export",
        "bootstrap",
        "checkpoint",
    ]
    for field, value in [
        ("native_labels", ["ground", "foot", "extra"]),
        ("native_body_properties", {"mass": [2.0]}),
        ("native_model_properties", {"shape_body": [0, 2]}),
    ]:
        backend.receipt = copy.deepcopy(reference["recipe"]["native"]) | {"bam": {}}
        backend.receipt[field] = value
        with pytest.raises(ValueError, match=field):
            backend.open()


def usd_stage():
    pytest.importorskip(
        "pxr", reason="OpenUSD SDK unavailable in ordinary CPU environment"
    )
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Plane.Define(stage, "/World/Ground")
    ground.GetAxisAttr().Set("Z")
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    material = UsdShade.Material.Define(stage, "/World/GroundMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(0.8)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(
        material, materialPurpose="physics"
    )
    return stage


def test_cpu_usd_actual_rotated_st_preserves_geometry_and_physics_binding():
    stage = usd_stage()
    from pxr import UsdGeom, UsdShade

    before = module.scene_bindings(stage)["original_stage_sha256"]
    _, _, text = declaration()
    expected = module.texture_descriptor(text)
    result = module.author_checked_layout(stage, expected, text)
    assert result["original_stage_sha256"] == before
    board = module.board_from_fixture(result, text)
    actual = (
        UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Ground")).GetPrimvar("st").Get()
    )
    assert tuple(map(tuple, actual)) == board.texture_st()
    bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Ground"))
    assert (
        str(bound.ComputeBoundMaterial("physics")[0].GetPath())
        == "/World/GroundMaterial"
    )
    assert [str(p.GetPath()) for p in stage.TraverseAll() if p.IsA(UsdGeom.Gprim)] == [
        "/World/Ground"
    ]
    assert (
        json.loads(
            stage.GetPrimAtPath(MATERIAL).GetCustomDataByKey(module.ground.METADATA_KEY)
        )
        == expected
    )
    assert not result["native_render_validated"]


@pytest.mark.parametrize("mutation", ["st", "shader", "metadata", "ancestor"])
def test_cpu_usd_postbootstrap_changes_refuse_without_masking_ancestor_binding(
    tmp_path, mutation
):
    stage = usd_stage()
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    _, _, text = declaration()
    expected = module.texture_descriptor(text)
    fixture = module.author_checked_layout(stage, expected, text)
    before = module.ground.ground_snapshot(stage)
    if mutation == "st":
        st = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/Ground")).GetPrimvar("st")
        values = list(st.Get())
        values[0] = Gf.Vec2f(0, 0)
        st.Set(values)
    elif mutation == "shader":
        stage.GetPrimAtPath(MATERIAL + "/texture").GetAttribute("inputs:wrapS").Set(
            "repeat"
        )
    elif mutation == "metadata":
        stage.GetPrimAtPath(MATERIAL).SetCustomDataByKey(
            module.ground.METADATA_KEY, "{}"
        )
    else:
        replacement = UsdShade.Material.Define(stage, "/World/Replacement")
        shader = UsdShade.Shader.Define(stage, "/World/Replacement/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.18)
        )
        replacement.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World")).Bind(
            replacement, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        assert (
            str(
                UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/Ground"))
                .ComputeBoundMaterial()[0]
                .GetPath()
            )
            == "/World/Replacement"
        )
    with pytest.raises(ValueError, match="appearance or original"):
        validate_ground_texture(
            stage,
            module.REPO / module.TEXTURE,
            board=BinaryLayoutABoard(consumer_detector_json=text),
            receipt=fixture,
        )
    backend = module.ground.reference_backend(object)()
    backend._ground_stage = stage
    backend._ground_authored = before
    backend.args = NS(out=tmp_path)
    backend.receipt = {}
    with pytest.raises(ValueError, match="appearance/geometry changed"):
        backend._check_ground("after_bootstrap")
