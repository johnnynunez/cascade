"""Bound declarations and passive entrypoint controls; no native renderer."""

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from benchmark.rgbd import binary_native_bridge as module
from benchmark.rgbd import binary_reference as binary
from benchmark.rgbd.ground_texture import MATERIAL
from benchmark.rgbd.planar_reference import reference_from_rgb
from cascade.sim.mobile_identity import build_model_identity
from test_mobile_identity import recipe_inputs as recipe_inputs

SOURCE_REPO = module.REPO


@pytest.fixture(autouse=True)
def cpu_contract_for_actual_test_interpreter(monkeypatch, tmp_path):
    """Portable unit fixture; the prepared native plan keeps its separate pin.

    Never require macOS/ARM to impersonate this host's frozen Linux consumer.
    The real cross-interpreter check-only uses the committed declaration.
    """
    root = tmp_path / "consumer-contract"
    for relative in module.SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((SOURCE_REPO / relative).read_bytes())
    (root / module.CONSUMER_SOURCE).write_bytes(
        module.common.canonical(binary.detector_recipe())
    )
    monkeypatch.setattr(module, "REPO", root)


def declaration():
    path = module.REPO / module.CONSUMER_SOURCE
    return (
        path,
        module.common.sha(path),
        module.load_consumer(path, module.common.sha(path)),
    )


def test_default_cpu_recipe_and_explicit_frozen_descriptor_have_same_geometry():
    _, _, content = declaration()
    board = binary.BinaryGroundBoard(consumer_detector_json=content)
    assert board.description() == binary.BinaryGroundBoard().description()
    assert (
        module.board_from_fixture(module.texture_descriptor(content), content) == board
    )
    copied = board.consumer_detector()
    copied["opencv_version"] = "99.0.0"
    assert board.consumer_detector()["opencv_version"] != "99.0.0"


@pytest.mark.parametrize(
    "mutation", ["version", "parameters", "codebook", "implementation"]
)
def test_consumer_implementation_mismatch_refused_even_without_campaign(
    monkeypatch, mutation
):
    _, _, content = declaration()
    changed = json.loads(content)
    if mutation == "version":
        changed["opencv_version"] = "99.0.0"
    elif mutation == "parameters":
        changed["parameters"]["errorCorrectionRate"] = 1.0
    elif mutation == "codebook":
        changed["dictionary_bytes_sha256"] = "a" * 64
    else:
        changed["opencv_package_sha256"] = "a" * 64
    board = binary.BinaryGroundBoard(
        consumer_detector_json=module.common.canonical(changed).decode()
    )
    monkeypatch.setattr(
        binary,
        "detect_binary_corners",
        lambda *a: pytest.fail("mismatch reached detection"),
    )
    with pytest.raises(ValueError, match="active implementation"):
        reference_from_rgb(np.zeros((1, 1, 3), np.uint8), board)


def test_cross_version_declaration_can_author_exact_bits_but_cannot_detect():
    _, _, content = declaration()
    changed = json.loads(content)
    changed["opencv_version"] = (
        "4.14.0" if changed["opencv_version"] != "4.14.0" else "5.0.0"
    )
    other = module.common.canonical(changed).decode()
    assert (
        module.texture_descriptor(other)["texture_sha256"]
        == module.texture_descriptor(content)["texture_sha256"]
    )
    assert (
        module.texture_descriptor(other)["board_sha256"]
        != module.texture_descriptor(content)["board_sha256"]
    )
    with pytest.raises(ValueError, match="active implementation"):
        module.board_from_fixture(module.texture_descriptor(other), other)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "unknown_parameter",
        "nan",
        "invalid_version",
        "mutable_dict",
        "noncanonical",
        "boolean_parameter",
        "boolean_policy",
    ],
)
def test_malformed_declaration_is_not_an_implicit_default(mutation):
    _, _, content = declaration()
    value = json.loads(content)
    if mutation == "missing":
        del value["opencv_version"]
    elif mutation == "extra":
        value["other"] = 1
    elif mutation == "unknown_parameter":
        value["parameters"]["other"] = 1
    elif mutation == "nan":
        value["parameters"]["errorCorrectionRate"] = float("nan")
    elif mutation == "invalid_version":
        value["opencv_version"] = "foreign"
    elif mutation == "boolean_parameter":
        value["parameters"]["errorCorrectionRate"] = False
    elif mutation == "boolean_policy":
        value["expected_ids"][0] = False
    encoded = (
        value
        if mutation == "mutable_dict"
        else json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if mutation != "noncanonical" else (", ", ": "),
        )
    )
    with pytest.raises(ValueError, match="consumer detector"):
        binary.BinaryGroundBoard(consumer_detector_json=encoded)


def test_changed_dictionary_and_png_cannot_inherit_declared_authoring(
    monkeypatch, tmp_path
):
    _, _, content = declaration()
    changed = json.loads(content)
    changed["dictionary_bytes_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="codebook"):
        module.texture_descriptor(module.common.canonical(changed).decode())
    original = module.REPO
    path = tmp_path / module.TEXTURE
    path.parent.mkdir(parents=True)
    path.write_bytes((original / module.TEXTURE).read_bytes() + b"\0")
    monkeypatch.setattr(module, "REPO", tmp_path)
    with pytest.raises(ValueError, match="PNG differs"):
        module.texture_descriptor(content)


def test_authenticated_declaration_source_is_mandatory_and_immutable(
    monkeypatch, tmp_path
):
    _, expected, content = declaration()
    path = tmp_path / module.CONSUMER_SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(content)
    monkeypatch.setattr(module, "REPO", tmp_path)
    assert module.load_consumer(path, expected) == content
    path.write_text(content + "\n")
    with pytest.raises(ValueError, match="source/hash"):
        module.load_consumer(path, expected)
    path.unlink()
    with pytest.raises(OSError):
        module.load_consumer(path, expected)


def test_model_rehashes_all_new_helpers_bitmap_and_declaration(
    recipe_inputs, monkeypatch
):
    admission, native, paths = recipe_inputs
    original = build_model_identity(admission, native, **paths)
    source = module.REPO
    for relative in module.SOURCES:
        path = paths["repo"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / relative).read_bytes())
    monkeypatch.setattr(module, "REPO", paths["repo"])
    _, _, content = declaration()
    module.extend_admission(admission, original, content)
    current = build_model_identity(admission, native, **paths)
    assert current["model_identity_sha256"] != original["model_identity_sha256"]
    assert set(module.SOURCES) <= set(current["recipe"]["source_sha256"])
    path = paths["repo"] / module.CONSUMER_SOURCE
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="source changed"):
        build_model_identity(admission, native, **paths)


def test_check_only_never_constructs_backend_and_reports_real_generator(
    monkeypatch, tmp_path, capsys
):
    path, expected, _ = declaration()
    args = NS(camera_rgbd=True, check_only=True, out=tmp_path / "absent")
    bridge = NS(
        parse_args=lambda _: args,
        admit=lambda _: {"source_sha256": {}},
        run=lambda *a, **kw: pytest.fail("native run"),
    )
    monkeypatch.setattr(module.common, "bind_bridge", lambda: bridge)
    monkeypatch.setattr(
        module.common,
        "reference_identity",
        lambda *a: {"model_identity_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        module, "reference_backend", lambda *a: pytest.fail("backend built")
    )
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
        expected,
    ]
    assert module.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["no_native_run"]
        and result["generator"]["role"]
        == "bitmap_authoring_only_not_consumer_detection"
    )
    assert module.CONSUMER_SOURCE in result["source_sha256"] and not args.out.exists()
    with pytest.raises(SystemExit):
        module.main(argv[:-4])


def test_injected_variant_keeps_camera_bootstrap_guards_and_refuses_late_declaration_change(
    monkeypatch, tmp_path
):
    path, expected, content = declaration()
    descriptor = module.texture_descriptor(content)
    events = []
    reference = {
        "recipe": {"native": {"native_labels": ["ground", "foot"]}, "bam": {}},
        "model_identity_sha256": "a" * 64,
    }

    class Parent:
        def __init__(self):
            self.receipt = {"native_labels": ["ground", "foot"], "bam": {}}
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

    monkeypatch.setattr(
        module, "author_checked_binary", lambda stage, value, c: descriptor
    )
    monkeypatch.setattr(
        module.ground, "ground_snapshot", lambda stage: {"appearance": "constant"}
    )
    backend = module.reference_backend(Parent, path, expected)()
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
    backend.receipt["native_labels"].append("unexpected")
    with pytest.raises(ValueError, match="native.native_labels"):
        backend.open()
    monkeypatch.setattr(
        module,
        "load_consumer",
        lambda *a: (_ for _ in ()).throw(ValueError("source/hash changed")),
    )
    with pytest.raises(ValueError, match="source/hash changed"):
        backend.open()


def test_cpu_usd_consumes_exact_consumer_and_actual_authoring_environment():
    pytest.importorskip("pxr", reason="OpenUSD absent in ordinary CPU environment")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    plane = UsdGeom.Plane.Define(stage, "/World/Ground")
    plane.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
    material = UsdShade.Material.Define(stage, "/World/GroundMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(
        material, materialPurpose="physics"
    )
    _, _, content = declaration()
    descriptor = module.texture_descriptor(content)
    fixture = module.author_checked_binary(stage, descriptor, content)
    prim = stage.GetPrimAtPath(MATERIAL)
    assert json.loads(prim.GetCustomDataByKey(module.ground.METADATA_KEY)) == descriptor
    assert (
        json.loads(prim.GetCustomDataByKey(module.GENERATOR_METADATA))
        == fixture["generator"]
    )
    assert fixture["physics_material"] == "/World/GroundMaterial"
    assert [str(p.GetPath()) for p in stage.TraverseAll() if p.IsA(UsdGeom.Gprim)] == [
        "/World/Ground"
    ]
