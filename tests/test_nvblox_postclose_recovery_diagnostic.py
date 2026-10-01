"""Labelled mapper faults retain the real runtime's original camera anchors."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark/diagnostics"))
from nvblox_postclose_recovery import PostCloseMapFault, require_simulation_clock
from cascade.perception.occupancy import OccupancyError
from test_payload_close_barrier import capture, integrate
from test_occupancy_payload import PROP, mapping


def test_old_executor_cannot_enter_physical_probe():
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="executor is required before reset or motion"):
        require_simulation_clock(runtime)


def test_clock_preflight_preserves_validated_producer_identity_and_errors():
    evidence = dict(version=1, source=("127.0.0.1", 8692), robot_id="rebot",
                    engine="physx", clock="physics", epoch="epoch-1",
                    sim_time=1., physics_step=120, physics_dt_s=1 / 120)
    reads = []
    raw = SimpleNamespace(validate_simulation_clock=lambda: reads.append(True) or evidence)
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=raw))
    assert require_simulation_clock(runtime) == evidence and reads == [True]
    def inconsistent_clock():
        raise RuntimeError("producer physics epoch changed")
    raw.validate_simulation_clock = inconsistent_clock
    with pytest.raises(RuntimeError, match="producer physics epoch changed"):
        require_simulation_clock(runtime)


@pytest.mark.parametrize("result", [None, {}, {"version": True}, {"version": 2}])
def test_unsupported_clock_validator_contract_fails_closed(result):
    raw = SimpleNamespace(validate_simulation_clock=lambda: result)
    runtime = SimpleNamespace(arm=SimpleNamespace(raw=raw))
    with pytest.raises(RuntimeError, match="validation contract"):
        require_simulation_clock(runtime)


def fixture():
    m = mapping()
    for camera in ("cam0", "side", "proof"):
        integrate(m, camera, 1., attached=False)
    closed = []
    runtime = SimpleNamespace(arm=SimpleNamespace(harness=SimpleNamespace(occupancy=m)),
                              _close_two_stage=lambda profile: closed.append(profile))
    return runtime, m, closed


def test_fault_arms_only_after_real_close_and_preserves_anchors_through_rebuild(tmp_path):
    runtime, m, closed = fixture()
    original_request = m._client.request
    with PostCloseMapFault(runtime, tmp_path) as fault:
        m._client.request({"action": "integrate_depth"})
        assert not fault.failures and not fault.active
        runtime._close_two_stage("actual material profile")
        assert closed == ["actual material profile"] and fault.active
        for camera in ("cam0", "side", "proof"):
            integrate(m, camera, 3.)
        assert m._body_error and not m._integrated_captures
        with pytest.raises(OccupancyError, match="post-close payload geometry deadline"):
            m.wait_payload_ready([capture(c, 2.) for c in ("cam0", "side", "proof")],
                                 deadline=time.monotonic() + .02, guard=lambda: None)
        report = fault.report()
        assert len(fault.failures) == 3 and report["all_original_anchors_retained"]
        assert set(report["preclose_anchors"]) == {"cam0", "side", "proof"}
        for camera, item in report["preclose_anchors"].items():
            archive = np.load(tmp_path / ("preclose-anchor-" + camera + ".npz"))
            assert item["t"] == 1. and "depth" in archive and "robot" in archive
        # The injected outage is exclusively integration; query/probe remain
        # available and the actual mapper's RPC timeout is not modified.
        m._client.request({"action": "query"})
    assert m._client.request == original_request
    for camera in ("cam0", "side", "proof"):
        integrate(m, camera, 4.)
    result = m.wait_payload_ready([capture(c, 3.) for c in ("cam0", "side", "proof")],
                                  deadline=time.monotonic() + .1, guard=lambda: None)
    assert result["contact_paths"] == [PROP]
    assert fault.report()["all_original_anchors_retained"]


def test_failed_close_does_not_activate_mapper_fault_and_restores_both_hooks(tmp_path):
    runtime, m, closed = fixture()
    def failed(profile):
        raise RuntimeError("actual close failed")
    runtime._close_two_stage = failed
    request = m._client.request
    with pytest.raises(RuntimeError, match="actual close failed"):
        with PostCloseMapFault(runtime, tmp_path) as fault:
            runtime._close_two_stage("profile")
    assert not fault.failures and fault.started_monotonic is None
    assert runtime._close_two_stage is failed and m._client.request == request


def test_missing_mapping_camera_refuses_before_close(tmp_path):
    runtime, m, closed = fixture()
    m._depth_history.pop("side")
    with pytest.raises(RuntimeError, match="three measured pre-close"):
        with PostCloseMapFault(runtime, tmp_path):
            runtime._close_two_stage("profile")
    assert not closed
