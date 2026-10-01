"""CPU contract tests; these do not stand in for the optional RTX smoke."""
import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cascade.config import Cfg, load_profile
from cascade.perception.camera_base import CameraError, make_camera
from cascade.perception.freshness import newer_capture
from cascade.sim.ovrtx_renderer import CameraSpec, OvrtxError, OvrtxRenderer, SceneSnapshot, _camera_layer


class FakeSDK:
    def __init__(self):
        self.events = []
        self.failure = None
        self.frames = 1
        self.count = 1
        self.depth = np.full((3, 4, 1), 2., dtype=np.float32)
        self.color = np.broadcast_to(np.array([10, 20, 30, 255], np.uint8), (3, 4, 4)).copy()
        self.start = None
        self.exposure = False
        self.t = 0.
        self.maps = 0
        self.unmaps = 0
        owner = self

        class Op:
            def wait(self):
                if owner.failure == "wait":
                    raise RuntimeError("publication failed")

        class Query:
            def result(self):
                return NS(total_prim_count=owner.count)

            def release(self):
                owner.events.append("release_query")
                return Op()

        class Dictionary:
            def __init__(self, stage):
                pass

            def create_path_list_from_strings(self, paths):
                return paths

            def destroy_path_list(self, paths):
                owner.events.append("destroy_paths")

            def destroy(self):
                owner.events.append("destroy_dictionary")

        class Stage:
            def __init__(self, name):
                owner.events.append("create_stage")

            def query_from_path_list(self, paths):
                return Query()

            def advance_write_floor(self, ordinal, scope):
                owner.events.append(("seal", ordinal))
                return Op()

            def write_attribute(self, query, attr, **kwargs):
                owner.events.append(("write", attr, kwargs["ordinal"], kwargs["tensors"].copy()))
                return Op()

            def destroy(self):
                owner.events.append("destroy_stage")

        class Var:
            def __init__(self, data):
                self.data = data

            def map(self, device):
                owner.maps += 1
                if owner.failure == "read" and owner.maps % 2 == 0:
                    raise RuntimeError("depth readback failed")
                return self

            def __dlpack__(self, *args, **kwargs):
                return self.data.__dlpack__(*args, **kwargs)

            def __dlpack_device__(self):
                return self.data.__dlpack_device__()

            def unmap(self):
                owner.unmaps += 1

        class Products(dict):
            pass

        class Renderer:
            def __init__(self, config):
                owner.events.append(("create_renderer", config))

            def attach_ovstage(self, stage):
                owner.events.append("attach")

            def step(self, *, render_products, delta_time, ordinal):
                owner.events.append(("step", ordinal))
                if owner.failure == "step":
                    raise RuntimeError("render failed")
                products = Products()
                products.simulation_start_time = owner.t
                owner.t += delta_time
                products.simulation_end_time = owner.t
                start = owner.t if owner.start is None else owner.start
                for p in render_products:
                    frame = NS(start_time=start, end_time=start + (.01 if owner.exposure else 0),
                               render_vars={f"{p}/Color": Var(owner.color), f"{p}/Depth": Var(owner.depth)})
                    products[p] = NS(frames=[frame] * owner.frames)
                return products

            def detach_ovstage(self):
                owner.events.append("detach")

            def destroy(self):
                owner.events.append("destroy_renderer")

        self.modules = (NS(Renderer=Renderer, RendererConfig=NS, Device=NS(CPU=0)),
                        NS(Stage=Stage, PathDictionary=Dictionary, Scope=NS(ALL=0),
                           population=NS(open_usd_from_string=lambda *a, **k: owner.events.append("load"))))


@pytest.fixture
def setup(tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text('#usda 1.0\ndef Xform "World" {}\n')
    sdk = FakeSDK()
    spec = CameraSpec("cam", 4, 3, 5., 5., np.eye(4))
    renderer = OvrtxRenderer(scene, [spec], dynamic_paths=["/World/Cube"], _modules=sdk.modules)
    snap = SceneSnapshot("physics-owner", "epoch", 1, 1., 10., {"/World/Cube": np.eye(4)})
    yield renderer, snap, sdk
    renderer.close()


def test_packet_depth_color_pose_and_provenance_are_one_transaction(setup):
    r, s, sdk = setup
    pose = np.eye(4)
    pose[:3, 3] = [.1, .2, .3]
    frame = r.render(replace(s, cameras={"cam": pose}))["cam"]
    assert tuple(frame.rgb[0, 0]) == (30, 20, 10)
    assert frame.depth_m.dtype == np.float32 and (frame.depth_m == 2).all()
    np.testing.assert_array_equal(frame.K, [[5, 0, 1.5], [0, 5, 1.], [0, 0, 1]])
    np.testing.assert_array_equal(frame.T_base_cam, pose)
    assert frame.t == 10. and frame.capture["scene_state"]["sim_time"] == 1.
    assert "proprioception" not in frame.capture
    assert frame.robot_mask is None and frame.payload_mask is None
    assert sdk.maps == sdk.unmaps == 2
    assert sdk.events[-1] == ("step", 2)
    writes = [e for e in sdk.events if isinstance(e, tuple) and e[0] == "write"]
    assert len(writes) == 2
    np.testing.assert_array_equal(writes[-1][3][0], (pose @ np.diag([1, -1, -1, 1])).T)


def test_duplicate_preserves_entire_packet_and_no_sdk_step(setup):
    r, s, sdk = setup
    first = r.render(s)["cam"]
    before = copy.deepcopy(sdk.events)
    first.rgb[:] = 0
    second = r.render(s)["cam"]
    assert tuple(second.rgb[0, 0]) == (30, 20, 10)
    assert second.t == 10. and second.frame_id == 1
    assert len(sdk.events) == len(before)
    assert not newer_capture(second, first)


def test_freshness_barrier_binds_renderer_epoch(setup):
    r, s, _ = setup
    first = r.render(s)["cam"]
    second = r.render(replace(s, sequence=2, sim_time=2., captured_monotonic=11.))["cam"]
    assert newer_capture(second, first)
    second.capture["producer_epoch"] = "other-owner"
    with pytest.raises(ValueError, match="identity changed"):
        newer_capture(second, first)


@pytest.mark.parametrize("change", [dict(epoch="new"), dict(source="other"), dict(sequence=0),
    dict(sequence=2, sim_time=.9), dict(sequence=2, captured_monotonic=9.),
    dict(metadata={"changed": True}), dict(cameras={"cam": np.diag([1., -1., -1., 1.])})])
def test_identity_clocks_and_contradictory_duplicate_rejected_before_writes(setup, change):
    r, s, sdk = setup
    r.render(s)
    n = len(sdk.events)
    with pytest.raises(OvrtxError):
        r.render(replace(s, **change))
    assert len(sdk.events) == n


@pytest.mark.parametrize("change", [dict(sequence=True), dict(sim_time=float("nan")),
    dict(captured_monotonic=-1), dict(transforms={}), dict(metadata={"nan": float("nan")}),
    dict(cameras={"unknown": np.eye(4)}), dict(transforms={"/World/Cube": np.zeros((4, 4))})])
def test_invalid_snapshot_does_not_initialize_sdk(setup, change):
    r, s, sdk = setup
    with pytest.raises(OvrtxError):
        r.render(replace(s, **change))
    assert not sdk.events


@pytest.mark.parametrize("failure", ["step", "read"])
def test_partial_transaction_is_terminal_until_new_owner(setup, failure):
    r, s, sdk = setup
    sdk.failure = failure
    with pytest.raises(OvrtxError, match="transaction failed"):
        r.render(s)
    n = len(sdk.events)
    sdk.failure = None
    with pytest.raises(OvrtxError, match="invalid"):
        r.render(s)
    assert len(sdk.events) == n


@pytest.mark.parametrize("frames", [0, 2])
def test_no_missing_or_interpolated_frame_relabel(setup, frames):
    r, s, sdk = setup
    sdk.frames = frames
    with pytest.raises(OvrtxError, match="instantaneous"):
        r.render(s)


@pytest.mark.parametrize("start", [-1., float("nan"), 9.])
def test_render_window_invalid_fails_closed(setup, start):
    r, s, sdk = setup
    sdk.start = start
    with pytest.raises(OvrtxError):
        r.render(s)


def test_noninstantaneous_exposure_rejected(setup):
    r, s, sdk = setup
    sdk.exposure = True
    with pytest.raises(OvrtxError, match="exposure"):
        r.render(s)


def test_nonfinite_background_zero_and_valid_metric_not_rescaled(setup):
    r, s, sdk = setup
    sdk.depth[0, :, 0] = [float("inf"), float("nan"), -1., 20.]
    d = r.render(s)["cam"].depth_m
    assert (d[0] == 0).all() and (d[1:] == 2).all()


def test_missing_dynamic_body_does_not_upsert_and_cleans_resources(setup):
    r, s, sdk = setup
    sdk.count = 0
    with pytest.raises(OvrtxError, match="missing"):
        r.render(s)
    assert sdk.events[-3:] == ["detach", "destroy_stage", "destroy_renderer"]
    assert "release_query" in sdk.events


def test_close_order_idempotent_and_no_reopen(setup):
    r, s, sdk = setup
    r.render(s)
    r.close()
    n = len(sdk.events)
    r.close()
    assert len(sdk.events) == n
    assert sdk.events[-3:] == ["detach", "destroy_stage", "destroy_renderer"]
    with pytest.raises(OvrtxError, match="closed"):
        r.open()


def test_camera_layer_calibration_and_optical_axis_conversion(setup):
    r, _, _ = setup
    layer = _camera_layer(r.scene, r.cameras, 0)
    assert 'DistanceToImagePlaneSD' in layer and 'DepthSD"' not in layer
    assert 'horizontalAperture = 19.199999999999999' in layer
    assert 'verticalAperture = 14.4' in layer
    assert 'uint[] deviceIds = [0]' in layer
    assert 'shutter:close = 0' in layer


def test_generated_and_demo_usda_parse_with_real_openusd(setup):
    Sdf = pytest.importorskip("pxr.Sdf", reason="optional USD syntax check; GPU smoke also loads the layer")
    r, _, _ = setup
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(_camera_layer(r.scene, r.cameras, 0))
    cfg = load_profile("cameras", "ovrtx")
    assert Sdf.Layer.FindOrOpen(cfg.scene_usd) is not None


def test_factory_is_opt_in_and_sdk_is_lazy(monkeypatch):
    monkeypatch.setattr("importlib.import_module", lambda _: pytest.fail("native SDK imported"))
    cfg = load_profile("cameras", "ovrtx")
    camera = make_camera(cfg)
    assert camera.has_depth
    camera.open()
    camera.close()


@pytest.mark.parametrize("changes", [{"scene_mode": "live"}, {"cx": 2}, {"depth": False},
    {"fy": 250}, {"meters_per_unit": .01}, {"extrinsics": {"mode": "eye_in_hand", "T": np.eye(4).tolist()}}])
def test_profile_rejects_unsupported_capabilities(changes):
    cfg = load_profile("cameras", "ovrtx").as_dict()
    cfg.update(changes)
    with pytest.raises(CameraError):
        make_camera(Cfg(cfg))


def test_root_scene_provenance_cannot_change_before_open(setup):
    r, s, sdk = setup
    r.scene.write_text('#usda 1.0\ndef Xform "Other" {}\n')
    with pytest.raises(OvrtxError, match="Root USD changed"):
        r.render(s)
    assert not sdk.events


def test_future_snapshot_rejected_without_native_initialization(setup):
    r, s, sdk = setup
    with pytest.raises(OvrtxError, match="future"):
        r.render(replace(s, captured_monotonic=1e20))
    assert not sdk.events


def test_unequal_focal_lengths_rejected_before_native_sdk():
    with pytest.raises(OvrtxError, match="square pixels"):
        CameraSpec("unsupported", 160, 120, 180., 135., np.eye(4))
