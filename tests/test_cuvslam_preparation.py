"""Explicit pre-acquisition preparation; CPU contract only, never imports the SDK."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from cascade.spatial import cuvslam
from cascade.spatial.domain import build_spatial_domain
from cascade.sensing.hub import SensorError
from test_spatial_cuvslam import rig as rig


def declaration(rig):
    p = rig.obs.payload
    return dict(schema="cascade.rgbd-localization-intrinsics.v1", width=p.width, height=p.height,
        intrinsics=list(p.intrinsics), pixel_center_offset_uv=list(p.pixel_center_offset_uv),
        frame_id=p.metadata.frame_id, calibration_id=p.metadata.calibration_id,
        model_identity_sha256=rig.obs.model_identity_sha256)


def domain(rig, declared=None):
    profile = dict(kind="spatial", robot_id="duck",
                   cuvslam={**rig.config, "preparation_intrinsics": declaration(rig) if declared is None else declared})
    result = build_spatial_domain("prepared", profile, sensor_domains={"sensors": NS(hub=rig.hub)})
    result._clock = lambda: rig.now[0]
    return result


def test_preparation_is_explicit_passive_at_construction_and_reads_no_capture(rig, monkeypatch):
    with monkeypatch.context() as passive:
        passive.setattr(cuvslam, "CuVslamProcess", lambda *a, **k: pytest.fail("passive construction started worker"))
        d = domain(rig)
    try:
        assert "prepare_localization" in {s["name"] for s in d.tool_specs}
        assert "prepare_localization" not in {s["name"] for s in rig.domain.tool_specs}
        with monkeypatch.context() as no_capture:
            no_capture.setattr(d, "_retained", lambda *a: pytest.fail("preparation read a capture"))
            ready = d.execute("prepare_localization", {})
            assert ready["ok"] and not ready["physical_admission"]
            assert d.execute("get_localization", {})["tracking_state"] == "uninitialized"
        assert d._watermark is None and not rig.sdk.calls
        rig.now[0] = 10.1
        result = d.execute("track_capture", rig.publish(replace(rig.obs, received_monotonic_s=10.1)))
        assert result["ok"] and result["sensor_epoch"] == rig.obs.epoch
        assert result["pose"]["position_error_m"] is None
        assert len(rig.sdk.calls) == 1 and d.resources[0].admission == "unvalidated"
    finally:
        assert d.close()["ok"]


@pytest.mark.parametrize("change", ["old_capture", "intrinsics", "offset"])
def test_first_tracking_capture_must_follow_preparation_and_match_declaration(rig, change):
    d = domain(rig)
    try:
        assert d.execute("prepare_localization", {})["ok"]
        obs = rig.obs
        if change == "old_capture":
            obs = replace(obs, producer_age_s=.1)
        elif change == "intrinsics":
            obs = replace(obs, payload=replace(obs.payload, intrinsics=(3., 0., 1.5, 0., 2., 1.5, 0., 0., 1.)))
        else:
            obs = replace(obs, payload=replace(obs.payload, pixel_center_offset_uv=(0., 0.)))
        result = d.execute("track_capture", rig.publish(obs))
        assert not result["ok"] and result["map_epoch_invalidated"]
        assert not rig.sdk.calls and d._tracker is None
    finally:
        assert d.close()["ok"]


@pytest.mark.parametrize("change", ["epoch", "gap", "replay"])
def test_first_real_capture_pins_continuity_without_a_synthetic_warmup_frame(rig, change):
    d = domain(rig)
    try:
        assert d.execute("prepare_localization", {})["ok"]
        first = rig.publish()
        assert d.execute("track_capture", first)["ok"]
        obs = replace(rig.obs, sequence=8, capture_time_s=.4)
        if change == "epoch":
            obs = replace(obs, epoch="other-sensor-epoch")
            # SensorHub rejects the changed epoch before the estimator can see it.
            with pytest.raises(SensorError, match="epoch"):
                rig.publish(obs)
            args = {"epoch": obs.epoch, "sequence": obs.sequence, "capture_sha256": obs.sha256}
        elif change == "gap":
            obs = replace(obs, capture_time_s=.451)
            args = rig.publish(obs)
        else:
            args = first
        result = d.execute("track_capture", args)
        assert not result["ok"] and result["map_epoch_invalidated"]
        assert len(rig.sdk.calls) == 1 and d._tracker is None
    finally:
        assert d.close()["ok"]


@pytest.mark.parametrize("change", [
    {"frame_id": "foreign"}, {"calibration_id": "d"*64}, {"model_identity_sha256": "f"*64},
    {"width": True}, {"height": 0}, {"width": 1024*1024},
    {"pixel_center_offset_uv": [.25, .25]}, {"intrinsics": [1.]}, {"extra": 0},
])
def test_declared_geometry_and_source_binding_are_validated_before_worker_creation(rig, change, monkeypatch):
    monkeypatch.setattr(cuvslam, "CuVslamProcess", lambda *a, **k: pytest.fail("invalid declaration started worker"))
    with pytest.raises(ValueError):
        domain(rig, declaration(rig) | change)


def test_preparation_copies_intrinsics_and_stop_during_warmup_never_publishes_ready(rig, monkeypatch):
    declared = declaration(rig)
    d = domain(rig, declared)
    declared["intrinsics"][0] = 999.
    process = cuvslam.CuVslamProcess
    def stopped(settings, **kwargs):
        assert settings[0]["intrinsics"][0] == 2.
        worker = process(settings, **kwargs)
        worker.warmup = lambda: d.stop()
        return worker
    monkeypatch.setattr(cuvslam, "CuVslamProcess", stopped)
    try:
        result = d.execute("prepare_localization", {})
        assert not result["ok"] and result["map_epoch_invalidated"]
        assert d._prepared_at is None and d._tracker is None and not rig.sdk.calls
        assert not d.execute("get_localization", {})["ok"]
    finally:
        assert d.close()["ok"]
