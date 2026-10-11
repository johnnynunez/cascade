"""The map's live state is reported honestly: live/stale/failed, age, voxels.

After the 2026-10-11 rig incident the operator could not tell from the MCP
`world_state`, the capability matrix or the live view /state that the
clearance gate was off (CASCADE_OCCUPANCY=0) or how old the map was. Every
surface now carries `backends.occupancy_state` and the capability detail
says whether the map can authorize a step right now.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from occupancy_grid_bridge import TABLE, GridBridge, centred_box, depth_frame, make_map

T_CAM = np.eye(4)


def _live_map(**kw):
    bridge = GridBridge([TABLE, centred_box([0.3, 0.1, 0.05], [0.05, 0.05, 0.1])], voxel=0.02)
    occ = make_map(bridge, unobserved="free", **kw)
    occ.refresh(depth_frame(), T_CAM)
    assert occ.last_error is None
    return occ, bridge


def test_a_fresh_required_map_reports_live_with_its_age_and_voxels():
    occ, _ = _live_map()
    r = occ.report()
    assert r["state"] == "live" and r["live"] is True and r["bridge"] is True
    assert r["required"] is True and r["unobserved"] == "free"
    assert 0.0 <= r["age_s"] < 1.0 and r["max_age_s"] == occ.max_age_s
    assert r["occupied_voxels"] == len(occ._occupied) > 0
    assert r["refreshes"] == 1 and r["last_error"] is None
    assert "nvblox on cuda:0" in r["backend"]
    occ.refresh(depth_frame(), T_CAM)
    assert occ.report()["refreshes"] == 2


def test_a_stale_map_reports_stale_and_not_live():
    occ, _ = _live_map(max_age_s=0.5)
    occ._last_refresh -= 2.0
    r = occ.report()
    assert r["state"] == "stale" and r["live"] is False
    assert r["age_s"] >= 2.0
    assert "stale" in r["detail"] and "max_age_s 0.5" in r["detail"]


def test_a_failed_refresh_reports_failed_with_the_error():
    occ, bridge = _live_map()
    bridge.fail = "occupancy bridge timed out (500 ms)"
    occ.refresh(depth_frame(), T_CAM)
    r = occ.report()
    assert r["state"] == "failed" and r["live"] is False
    assert "timed out" in r["last_error"] and "timed out" in r["detail"]


def test_a_map_waiting_for_its_first_masked_refresh_says_so():
    occ = make_map(GridBridge([TABLE]), unobserved="free")
    r = occ.report()
    assert r["state"] == "waiting" and r["live"] is False and r["age_s"] is None
    assert r["occupied_voxels"] == 0 and r["refreshes"] == 0


def test_a_standby_body_mask_fault_is_reported_not_hidden():
    occ = make_map(GridBridge([TABLE]), unobserved="free")

    def no_state(*_):
        raise RuntimeError("robot body pose unavailable: arm in standby")

    occ.add_robot_body_mask(no_state)
    occ.refresh(depth_frame(), T_CAM)
    r = occ.report()
    assert r["state"] == "failed" and "standby" in r["detail"]


def test_no_bridge_reports_no_bridge():
    class Dead(GridBridge):
        def request(self, payload, timeout_ms=None):
            from cascade.perception.occupancy import OccupancyError

            raise OccupancyError("connection refused")

    from cascade.perception.occupancy import OccupancyMap

    occ = OccupancyMap(client=Dead(), region_min=np.zeros(3), region_max=np.ones(3))
    occ.probe()
    r = occ.report()
    assert r["state"] == "no bridge" and r["live"] is False and r["bridge"] is False


def test_the_report_never_waits_on_a_refresh_in_flight():
    """capability_matrix runs on every dashboard poll: the report must not
    take the refresh locks (a hung bridge would hang the dashboard)."""
    import threading

    occ, _ = _live_map()
    with occ._refresh_lock, occ._io_lock:
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", occ.report()))
        t.start()
        t.join(1.0)
        assert "r" in out, "report() blocked on the refresh locks"


# ── the surfaces: runtime.backends, capability matrix, /state, world_state ──


def _runtime(occ):
    harness = SimpleNamespace(occupancy=occ)
    return SimpleNamespace(arm=SimpleNamespace(harness=harness),
                           cfg=SimpleNamespace(grasp={"backend": "obb"}), grasp_planner_used=None)


def test_backends_carry_the_occupancy_state():
    from cascade.skills.runtime import SkillRuntime

    occ, _ = _live_map()
    b = SkillRuntime.backends(_runtime(occ))
    assert b["occupancy_live"] is True and b["occupancy_state"]["state"] == "live"
    assert SkillRuntime.backends(_runtime(None))["occupancy_state"] is None


@pytest.mark.parametrize("stale", [False, True])
def test_the_capability_cell_says_whether_the_map_authorizes_a_step_now(stale):
    from cascade.apps.capabilities import _occupancy_capability
    from cascade.skills.runtime import SkillRuntime

    occ, _ = _live_map(max_age_s=0.5)
    if stale:
        occ._last_refresh -= 2.0
    cell = _occupancy_capability(SkillRuntime.backends(_runtime(occ)))
    assert cell["available"] is True                 # the bridge answers: no tool is withheld
    assert cell["state"] == ("stale" if stale else "live")
    assert cell["required"] is True and cell["occupied_voxels"] > 0
    if stale:
        assert "stale" in cell["detail"] and "motion is refused" in cell["detail"]
    else:
        assert "live" in cell["detail"] and "occupied voxels" in cell["detail"] and " s old" in cell["detail"]


def test_world_state_and_the_live_view_state_show_it(demo_cfg, tmp_path):
    from cascade.apps.demo import _runtime_state, build_runtime, shutdown_runtime
    from cascade.apps.mcp_server import McpSkillServer

    runtime, arm = build_runtime(demo_cfg, tmp_path / "run")
    try:
        occ, _ = _live_map()
        runtime.arm.harness.occupancy = occ
        state = _runtime_state(runtime)                # the live view's /state
        assert state["backends"]["occupancy_state"]["state"] == "live"
        assert state["capabilities"]["occupancy"]["state"] == "live"
        server = SimpleNamespace(withheld_tools=dict)
        world = McpSkillServer._world_state(server, runtime)
        assert world["backends"]["occupancy_state"]["occupied_voxels"] > 0
        assert world["capabilities"]["occupancy"]["required"] is True
    finally:
        shutdown_runtime(runtime, arm)


def test_live_in_the_report_is_exactly_usable_for_a_required_map():
    """A latched body-mask fault outlives a later depthless commit (which
    clears last_error): the report must still say failed, as usable() does."""
    occ, _ = _live_map()
    assert occ.report()["live"] is occ.usable() is True
    occ._body_error = "robot body pose unavailable for the #0 mesh mask"
    assert occ.last_error is None
    r = occ.report()
    assert r["live"] is occ.usable() is False and r["state"] == "failed" and "mesh mask" in r["detail"]
