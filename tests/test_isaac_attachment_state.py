"""Reuse exact, non-ambiguous completed-update history without sensor reads."""
import copy

import pytest

from test_isaac_frame_snapshot import capture_bridge  # noqa: F401
from test_isaac_contact_frame_history import configure, contacts


def ready(b):
    configure(b)
    b.env["_gpu_contact_snapshot"] = contacts(True, b)
    b.publish()
    step = b.physics_index[0]
    clock = dict(epoch=b.env["_motion_clock_epoch"], physics_step=step, sim_time=step / 1000)
    return clock


def test_contact_from_current_history_is_owned_and_adds_no_sensor_reads(capture_bridge):
    b = capture_bridge; clock = ready(b)
    b.env["_gpu_contact_snapshot"] = lambda *_: pytest.fail("extra contact sensor read")
    b.env["art"].get_dof_positions = lambda: pytest.fail("extra q read")
    sample = b.env["_attachment_state_snapshot"](b.q[0], clock)
    assert sample["tracking"] and sample["error"] is None
    assert sample["paths"] == ["/World_Props/pink_cube"]
    assert sample["q"] == b.q[0, :6].tolist()
    assert sample["gripper_joints"]["position_m"] == [.02, .04]
    b.q[:] = 100
    assert sample["q"] != b.q[0, :6].tolist()


@pytest.mark.parametrize("change", ["q", "jaw", "step", "epoch", "time", "missing", "poison"])
def test_current_state_must_match_unambiguous_history(capture_bridge, change):
    b = capture_bridge; clock = ready(b)
    if change == "q": b.q[0, 0] += .001
    elif change == "jaw": b.q[0, 6] += .001
    elif change == "step": clock["physics_step"] += 1  # Same q is insufficient.
    elif change == "epoch": clock["epoch"] = "another"
    elif change == "time": clock["sim_time"] += .0000001
    elif change == "missing": b.env["_frame_history"].clear()
    else:
        history = b.env["_frame_history"]
        entry = history.resolve(reference=[b.physics_index[0] * 1000, 1000000],
                                simulation_time=clock["sim_time"], epoch=clock["epoch"])
        entry["payload"]["contact_state"]["paths"] = []
        assert not history.record(**entry)
        assert history._last["physics_step"] == clock["physics_step"]  # _last is NOT authority.
    result = b.env["_attachment_state_snapshot"](b.q[0], clock)
    assert result["tracking"] is False and result["paths"] is None and result["error"]


@pytest.mark.parametrize("change", ["off", "error", "wrong_step", "wrong_channel"])
def test_contact_unavailable_is_not_an_empty_usable_attachment(capture_bridge, change):
    b = capture_bridge; ready(b)
    if change == "off": b.env["os"].environ["CASCADE_ISAAC_CONTACT_MASK"] = "0"
    else:
        real = contacts(True, b)
        def read(name):
            if change == "error": raise RuntimeError("contact failure")
            value = real(name)
            if change == "wrong_step": value["physics_step"] -= 1
            else: value["channel"] = "wrong"
            return value
        b.env["_gpu_contact_snapshot"] = read
    b.publish()
    clock = dict(epoch=b.env["_motion_clock_epoch"], physics_step=b.physics_index[0],
                 sim_time=b.physics_index[0] / 1000)
    value = b.env["_attachment_state_snapshot"](b.q[0], clock)
    assert value["paths"] is None and value["tracking"] is False and value["error"]


def test_fresh_empty_attachment_remains_distinct_from_unknown(capture_bridge):
    b = capture_bridge; ready(b)
    b.env["_gpu_contact_snapshot"] = contacts(False, b)
    b.publish()
    clock = dict(epoch=b.env["_motion_clock_epoch"], physics_step=b.physics_index[0],
                 sim_time=b.physics_index[0] / 1000)
    value = b.env["_attachment_state_snapshot"](b.q[0], clock)
    assert value["paths"] == [] and value["tracking"] is True and value["error"] is None
