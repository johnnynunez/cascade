"""CPU contracts for owned rendering and exact Isaac snapshot publication."""
import copy
from contextlib import nullcontext
from dataclasses import replace
import importlib
from multiprocessing import Pipe
from pathlib import Path
import struct
import threading
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.sim.ovrtx_masks import body_masks, body_label, decode_id_map, semantic_layer
from cascade.sim.ovrtx_renderer import SceneSnapshot, _snapshot
from cascade.sim.ovrtx_worker import serve
from cascade.sim.render_binding import valid_render_binding
from cascade.types import Frame


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("isaac_ovrtx")


def labels_buffer(labels):
    table, strings = bytearray(), bytearray()
    for identity, label in labels.items():
        encoded = label.encode() + b"\0"
        table.extend(struct.pack("<6I", identity, 0, 0, 0, len(encoded), len(labels)*24 + len(strings)))
        strings.extend(encoded)
    return np.frombuffer(table + strings + struct.pack("<I", len(labels)), np.uint8)


def test_semantic_masks_are_exact_body_ids_and_reject_unmapped_pixels():
    ids = np.array([[[1], [2]], [[0], [3]]], np.uint32)
    buf = labels_buffer({1: f"class: {body_label('/robot/link')};",
                         2: f"class: {body_label('/World_Props/cube')};", 3: "class: wall;"})
    masks = body_masks(ids, buf, (2, 2), ["/robot/link", "/World_Props/cube"])
    assert masks["/robot/link"].tolist() == [[True, False], [False, False]]
    assert masks["/World_Props/cube"].sum() == 1
    with pytest.raises(ValueError, match="unmapped"):
        body_masks(ids + 1, buf, (2, 2), masks)
    ambiguous = labels_buffer({1: f"class: {body_label('/robot/link')}; class: other;", 2: "class: wall;", 3: "class: wall;"})
    with pytest.raises(ValueError, match="ambiguous"):
        body_masks(ids, ambiguous, (2, 2), masks)


@pytest.mark.parametrize("mutate", [
    lambda a: a[:-1],
    lambda a: np.concatenate([a[:-4], np.array([255]*4, np.uint8)]),
    lambda a: np.concatenate([a[:20], np.zeros(4, np.uint8), a[24:]]),
])
def test_corrupt_identifier_map_fails_closed(mutate):
    with pytest.raises((ValueError, UnicodeError)):
        decode_id_map(mutate(labels_buffer({1: "class: /robot;"})))


def test_semantics_override_private_geometry_and_validate_paths():
    layer = semantic_layer({"/robot/link/mesh": "/robot/link"})
    assert 'over "mesh" (' in layer and 'prepend apiSchemas = ["SemanticsAPI:class"]' in layer
    assert f'semanticData = "{body_label("/robot/link")}"' in layer
    assert body_label("/Robot") != body_label("/robot")
    with pytest.raises(ValueError):
        semantic_layer({'/robot/evil"': "/robot"})


@pytest.fixture
def packet_source():
    t = 10.
    state = {"version": 1, "backend": "isaac", "robot_id": "/robot", "q": [.1]*6,
             "producer_epoch": "physics-epoch", "t": t, "joint_convention": "asset",
             "time_source": "physics_loop_monotonic",
             "gripper_joints": {"version": 1, "names": ["joint_left", "joint_right"],
                                "position_m": [.01, .01], "lower_m": [0., 0.], "upper_m": [.02, .02]}}
    contact = {"error": None, "tracking": True, "paths": ["/World_Props/cube"],
               "scene_prop_paths": ["/World_Props/cube"]}
    pose = np.eye(4)
    pose[2, 3] = 1.
    snapshot = SceneSnapshot("physics_tensor", "physics-epoch", 23, .4, t,
        {"/robot": np.eye(4)}, {"cam0": pose},
        {"proprioception": state, "contact_state": contact, "snapshot_finished_monotonic": 10.1})
    normalized, digest, _, _ = _snapshot(snapshot, snapshot.transforms, snapshot.cameras)
    frame = Frame(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), np.float32), np.eye(3),
                  T_base_cam=pose.copy(), capture={"scene_state": normalized, "scene_state_sha256": digest,
                  "scene_sha256": "a"*64, "producer_epoch": "renderer-epoch", "camera": "cam0",
                  "render_reference": dict(ordinal=2, step_start_s=0., sensor_start_s=.1,
                                           sensor_end_s=.1, step_end_s=.1)},
                  prop_masks={"/robot": np.array([[1, 0], [0, 0]], bool),
                              "/World_Props/cube": np.array([[0, 1], [0, 0]], bool)})
    return snapshot, frame


def test_bridge_preserves_captured_state_and_base_transform(bridge, packet_source):
    snapshot, frame = packet_source
    packet = bridge.bridge_packet(frame, snapshot, robot_id="/robot", base_z=.2)
    assert packet["proprioception"] == snapshot.metadata["proprioception"]
    assert packet["T_base_cam"][2][3] == .8
    assert packet["render_reference"]["source"] == "ovrtx_snapshot"
    assert "numerator" not in packet["render_reference"]
    assert valid_render_binding(packet)
    assert packet["robot_pixel_mask"]["contact_paths"] == ["/World_Props/cube"]
    packet["proprioception"]["q"][0] = 2.
    assert not valid_render_binding(packet)
    assert snapshot.metadata["proprioception"]["q"][0] == .1


@pytest.mark.parametrize("field,value", [("snapshot_sha256", "bad"), ("producer_epoch", "changed"),
    ("history_physics_step", True), ("renderer_epoch", ""), ("snapshot_finished_monotonic", 9.)])
def test_malformed_ovrtx_binding_does_not_authorize_motion(bridge, packet_source, field, value):
    snapshot, frame = packet_source
    packet = bridge.bridge_packet(frame, snapshot, robot_id="/robot", base_z=0.)
    packet["render_reference"][field] = value
    assert not valid_render_binding(packet)


def test_wrong_snapshot_even_at_same_epoch_or_sequence_is_rejected(bridge, packet_source):
    snapshot, frame = packet_source
    changed = copy.deepcopy(snapshot.transforms)
    changed["/robot"][0, 3] = .5
    with pytest.raises(ValueError, match="supplied physical snapshot"):
        bridge.bridge_packet(frame, replace(snapshot, transforms=changed), robot_id="/robot", base_z=0.)


def test_unknown_contacts_preserve_error_not_empty_attachment(bridge, packet_source):
    snapshot, frame = packet_source
    with pytest.raises(ValueError, match="Contact state unavailable"):
        bridge.encode_masks(frame.prop_masks, "/robot", {"error": "lost tensor"}, 10.)
    contact = copy.deepcopy(snapshot.metadata["contact_state"])
    contact["paths"] = ["/World_Props/absent"]
    with pytest.raises(ValueError, match="omits captured"):
        bridge.encode_masks(frame.prop_masks, "/robot", contact, 10.)


def test_owned_worker_stops_after_partial_failure_and_closes_owner():
    parent, child = Pipe()
    events = []
    class Renderer:
        def __init__(self, **kwargs):
            events.append("create")
        def open(self):
            events.append("open")
        def render(self, snapshot):
            events.append("render")
            raise ValueError("partial publication")
        def close(self):
            events.append("close")
    thread = threading.Thread(target=serve, args=(child, Renderer))
    thread.start()
    parent.send(("open", {}))
    assert parent.poll(2) and parent.recv() == (True, None)
    parent.send(("render", None))
    assert parent.poll(2) and parent.recv() == (False, "ValueError: partial publication")
    thread.join(2)
    assert not thread.is_alive() and events == ["create", "open", "render", "close"]
    parent.close()


def test_opt_in_selection_survives_owned_isaac_launcher(bridge):
    launch = importlib.import_module("isaac_launch")
    values = {"CASCADE_CAMERA_RENDERER": "ovrtx", "CASCADE_OVRTX_PYTHON": "/sdk/bin/python",
              "CASCADE_OVRTX_OUTPUT": "/run/render", "CASCADE_OVRTX_DEVICE": "1"}
    env = launch.clean_environment(values | {"PYTHONHOME": "/foreign"}, source=None)
    assert all(env[k] == v for k, v in values.items())
    assert "PYTHONHOME" not in env


def test_tensor_pose_rotation_and_wrist_mount_are_metric(bridge):
    T = bridge.pose_matrix([1, 2, 3], [np.sqrt(.5), 0, 0, np.sqrt(.5)])
    np.testing.assert_allclose(T @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)
    with pytest.raises(ValueError, match="normalized"):
        bridge.pose_matrix([1, 2, 3], [2, 0, 0, 0])


def test_producer_binds_current_tensor_wrist_and_rejects_changed_repeat(bridge, monkeypatch, packet_source):
    monkeypatch.setitem(sys.modules, "isaacsim.core.experimental.utils.backend",
                        NS(use_backend=lambda value: nullcontext()))
    source = bridge.IsaacOvrtx.__new__(bridge.IsaacOvrtx)
    source.latest = None
    source.base_z = .2
    source.error = None
    source.paths = ("/robot",)
    source.wrist_link = "/robot"
    source.wrist_mount = np.eye(4)
    source.wrist_mount[0, 3] = .06
    source.specs = [NS(name="wrist", T_base_cam=np.eye(4))]
    positions = np.array([[.1, .2, .3]])
    source._bodies = NS(is_physics_tensor_entity_valid=lambda: True,
        get_world_poses=lambda: (NS(numpy=lambda: positions.copy()), NS(numpy=lambda: np.array([[1, 0, 0, 0]]))))
    payload = copy.deepcopy(packet_source[0].metadata)
    payload.pop("snapshot_finished_monotonic")
    payload["wrist_T"] = np.eye(4).tolist()  # stale pre-update camera intentionally differs.
    args = dict(epoch="physics-epoch", physics_step=23, simulation_time=.4, started=10., payload=payload)
    source.capture(**args)
    np.testing.assert_allclose(source.latest.cameras["wrist"][:3, 3], [.16, .2, .3])
    assert source.latest.metadata["wrist_T"][2][3] == pytest.approx(.1)
    first = source.latest
    args["started"] = 12.
    args["payload"]["proprioception"]["t"] = 12.
    source.capture(**args)
    assert source.latest is first and source.latest.captured_monotonic == 10.
    positions[0, 0] += .01
    with pytest.raises(ValueError, match="Contradictory"):
        source.capture(**args)
    source._bodies.is_physics_tensor_entity_valid = lambda: False
    with pytest.raises(ValueError, match="fallback forbidden"):
        source.capture(**args)


def test_startup_readiness_accepts_complete_ovrtx_physics_binding(bridge, packet_source):
    from cascade.sim.startup_readiness import _packet
    snapshot, frame = packet_source
    packet = bridge.bridge_packet(frame, snapshot, robot_id="/robot", base_z=0.)
    frame.capture = packet | {"backend": "isaac", "source": ("localhost", 1234)}
    _packet(frame, ("localhost", 1234), "/robot", "physics-epoch", 6)
    frame.capture["render_reference"]["renderer"]["sensor_end_s"] += .1
    with pytest.raises(Exception, match="render/state association"):
        _packet(frame, ("localhost", 1234), "/robot", "physics-epoch", 6)


def test_export_flattens_private_scene_and_resets_nested_body_stacks(bridge, monkeypatch, tmp_path):
    pytest.importorskip("pxr", reason="Native OpenUSD serialization contract")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    parent = UsdGeom.Xform.Define(stage, "/World")
    parent.AddTranslateOp().Set(Gf.Vec3d(.5, 0, 0))
    for path, x in (("/World/Robot", .2), ("/World/Robot/Link", .1)):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdGeom.Cube.Define(stage, path + "/mesh")
    cam = UsdGeom.Camera.Define(stage, "/World_Cams/cam0")
    cam.AddTranslateOp().Set(Gf.Vec3d(0, 0, 2))
    before = stage.GetRootLayer().ExportToString()
    created = []
    def bodies(paths):
        created.append(paths)
        return NS(paths=paths)
    monkeypatch.setattr(bridge, "PhysicsBodies", bodies)
    source = bridge.IsaacOvrtx(stage, {"cam0": (None, [[100., 0, 20], [0, 100., 20], [0, 0, 1]])},
        python="unused", output=tmp_path, robot_id="/World/Robot", base_z=0., width=40, height=40)
    assert stage.GetRootLayer().ExportToString() == before
    assert created == [["/World/Robot", "/World/Robot/Link"]]
    frozen = Usd.Stage.Open(str(tmp_path / "render-scene.usda"))
    for path, x in (("/World/Robot", .2), ("/World/Robot/Link", .1)):
        xf = UsdGeom.Xformable(frozen.GetPrimAtPath(path))
        assert not xf.GetResetXformStack()  # Kit only exports the unedited layer.
        assert xf.GetLocalTransformation().ExtractTranslation()[0] == pytest.approx(x)
    from cascade.sim.ovrtx_renderer import _camera_layer
    layer = __import__("pxr.Sdf", fromlist=["Layer"]).Layer.CreateAnonymous()
    layer.ImportFromString(_camera_layer(source.config["scene"], source.specs, 0,
        source.config["semantic_paths"], source.config["world_paths"], source.config["instance_paths"]))
    render_stage = Usd.Stage.Open(layer)
    for path in source.paths:
        assert UsdGeom.Xformable(render_stage.GetPrimAtPath(path)).GetResetXformStack()
    assert source.config["semantic_paths"]["/World/Robot/Link/mesh"] == "/World/Robot/Link"
    assert (tmp_path / "scene-manifest.json").exists()


def test_tensor_view_uses_its_actual_path_order_without_usd_authoring(bridge, monkeypatch):
    values = np.array([[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0]], np.float32)
    view = NS(prim_paths=["/b", "/a"], check=lambda: True,
              get_transforms=lambda: NS(numpy=lambda: values.copy()))
    simulation = NS(is_valid=True, create_rigid_body_view=lambda paths: view)
    manager = NS(_physics_sim_view__warp=simulation)
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", NS(SimulationManager=manager))
    bodies = bridge.PhysicsBodies(["/a", "/b"])
    p, q = bodies.get_world_poses()
    np.testing.assert_array_equal(p, [[4, 5, 6], [1, 2, 3]])
    np.testing.assert_array_equal(q, [[0, 0, 0, 1], [1, 0, 0, 0]])
    manager._physics_sim_view__warp = NS(is_valid=False)
    assert not bodies.is_physics_tensor_entity_valid()
