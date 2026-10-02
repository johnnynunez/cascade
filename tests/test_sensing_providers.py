"""Independent transport tests use software fixtures, never native physics."""
from dataclasses import replace
import socket
import time

import pytest

from cascade.sensing import (MobileRgbSensorProvider, MobileStateSensorProvider, SensorError,
                             SensorHub)
from cascade.sim.mobile_bridge import MobileBridgeController, MobileBridgeServer
from mobile_support_fixture import support_contract
from test_mobile_bridge import publish


def profile(port=45678):
    return {"robot_id": "duck", "source": "isolated-bridge", "engine": "newton", "type": "isaac",
            "device": "cuda:0", "asset_sha256": "a" * 64, "policy_sha256": "b" * 64,
            "model_identity_sha256": "e" * 64, "support_contract": support_contract(),
            "bridge_host": "127.0.0.1", "bridge_port": port, "timeout_s": .5,
            "cameras": {"overview": {"max_age_s": 2., "max_jpeg_bytes": 100_000, "max_pixels": 10_000}}}


def test_construct_inspect_and_close_do_not_connect_or_construct_an_actuator(monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    def forbidden(*args, **kwargs):
        raise AssertionError("passive provider metadata must not dial or construct actuator")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(IsaacBase, "__init__", forbidden)
    for modality in ("imu", "proprioception", "solved_contact"):
        provider = MobileStateSensorProvider(modality, profile(), modality, read_timeout_s=1.)
        assert provider.descriptor.measurement_kind == "physics"
        provider.close()
    provider = MobileRgbSensorProvider("overview", profile(), "overview", read_timeout_s=1.)
    assert provider.descriptor.modality == "rgb"
    provider.close()


def test_real_passive_state_socket_never_steps_or_claims_control(monkeypatch):
    from cascade.control.isaac_base import IsaacBase
    def forbidden(*args, **kwargs):
        raise AssertionError("sensor read cannot construct an actuator")
    monkeypatch.setattr(IsaacBase, "__init__", forbidden)
    controller = MobileBridgeController(
        robot_id="duck", source="isolated-bridge", engine="newton", device="cuda:0",
        asset_sha256="a" * 64, policy_sha256="b" * 64, model_identity_sha256="e" * 64,
        support_contract=support_contract(), max_linear_speed=.2, max_angular_speed=.8,
        max_duration_s=5., lease_s=.3, max_state_age_s=5., max_action_wall_s=2.)
    server = MobileBridgeServer(controller, port=0)
    server.start()
    hub = SensorHub()
    try:
        publish(controller, q=list(range(14)), angular_velocity=[.1, .2, .3])
        before = controller.state()["state"]
        for modality in ("imu", "proprioception", "solved_contact"):
            provider = MobileStateSensorProvider(modality, profile(server.address[1]), modality,
                                                 max_age_s=5., read_timeout_s=2.)
            hub.register(provider)
            value = hub.read(modality)
            assert value.sequence == before["step"] and value.capture_time_s == before["sim_time_s"]
            assert value.model_identity_sha256 == before["model_identity_sha256"]
            assert value.epoch == before["epoch"]
            assert value.producer_age_s > 0
            if modality == "imu":
                assert value.payload.angular_velocity_rad_s == (.1, .2, .3)
                assert value.payload.linear_acceleration_m_s2 is None
            elif modality == "proprioception":
                assert value.payload.position_rad == tuple(range(14))
                assert value.payload.effort_nm is None
            else:
                assert value.payload.observation.contacts[0].normal_force_n == 3
            with pytest.raises(SensorError, match="replay"):
                hub.read(modality)
        after = controller.state()["state"]
        for key in ("epoch", "step", "sim_time_s", "generation", "joint_positions", "controller_status"):
            assert after[key] == before[key]
        assert not server._owners
    finally:
        assert hub.close(2)["ok"]
        server.close()


def test_adapter_preserves_age_and_receipt_exactly(monkeypatch):
    from cascade.sim.base_truth import BaseTruthReader
    from test_mobile_effects import state
    value = replace(state(1), robot_id="duck", source="isolated-bridge", angular_velocity_body=(.1, .2, .3),
                    received_monotonic_s=10., producer_age_s=.32)
    monkeypatch.setattr(BaseTruthReader, "__call__", lambda self: value)
    provider = MobileStateSensorProvider("imu", profile(), "imu", read_timeout_s=1.)
    result = provider.read()
    assert result.received_monotonic_s == 10.
    assert result.producer_age_s == .32
    assert result.age_s(10.1) == pytest.approx(.42)
    assert result.sequence == value.step and result.capture_time_s == value.sim_time_s
    provider.close()


def test_mobile_camera_exports_rgb_only_with_original_capture_age(monkeypatch):
    import cv2
    import numpy as np
    from cascade.sim.mobile_frames import MobileFrame, MobileFrameReader
    ok, jpeg = cv2.imencode(".jpg", np.zeros((4, 6, 3), dtype=np.uint8))
    assert ok
    captured = time.monotonic() - .1
    metadata = {"robot_id": "duck", "source": "isolated-bridge", "epoch": "camera-episode",
                "camera": "overview", "step": 42, "sim_time_s": .21, "width": 6, "height": 4,
                "producer_age_s": .2, "model_identity_sha256": "e" * 64}
    frame = MobileFrame(metadata, jpeg.tobytes(), captured, 2)
    monkeypatch.setattr(MobileFrameReader, "__call__", lambda self, camera: frame)
    provider = MobileRgbSensorProvider("overview", profile(), "overview", read_timeout_s=1.)
    result = provider.read()
    assert result.sequence == 42 and result.capture_time_s == .21
    assert result.received_monotonic_s == captured and result.producer_age_s == .2
    assert result.payload.data == frame.jpeg and result.payload.modality == "rgb"
    assert not hasattr(result.payload, "depth_m_f32le")
    assert result.payload.metadata.calibration_id is None
    provider.close()
