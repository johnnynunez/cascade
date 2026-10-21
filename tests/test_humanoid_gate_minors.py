"""#434 re-review minors (CPU only, fake bridges on ephemeral ports, no live port).

6. a JSON integer too big for a float (``float()`` raises OverflowError) is a STOP row, not an exception.
"""
from __future__ import annotations

import json
import socket

import pytest

from cascade.skills import humanoid_locomanip as hl

from test_humanoid_gate_hardening import CARRY, ScriptedBridge
from test_humanoid_locomanip import IN_BIN, action, make_gate, profile, state, walk_script

HUGE = 10 ** 400  # a JSON integer float() cannot hold: OverflowError
LIVE_PORTS = {5555, 5556, 5557, 5558, 5566, 8080, 8081, 8091, 8092, 8093, 8611, 18481, 18482, 18741, 17710, 17810,
              17910}


@pytest.fixture(autouse=True)
def _no_live_port(monkeypatch):
    """Socket guard: the fakes listen on ephemeral ports; a live service port is never reached."""
    real = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and len(address) >= 2 and address[1] in LIVE_PORTS:
            raise OSError(f"test socket guard: connect to {address!r} refused")
        return real(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)


# ------------------------------------------------------- 6. an integer too big for a float is a STOP row
def _set(data, key, index, value):
    if index is None:
        data[key] = value
    else:
        data[key][index] = value
    return data


@pytest.mark.parametrize("key, index", [("joint_targets", 2), ("joint_targets", 1), ("joint_targets", 4), ("nav", 0),
                                        ("torso_rpy", 2), ("base_height", None)],
                         ids=["arm", "knee_wbc", "finger", "nav", "torso", "base_height"])
def test_a_command_integer_too_big_for_a_float_is_a_stop_row(key, index):
    gate = make_gate()
    d = gate.check(1, state(1), _set(action(), key, index, HUGE))
    assert d.verdict == "STOP" and d.clamps == () and gate.latched is not None
    assert [r.name for r in d.rows if not r.ok] == ["state_finite"] and "malformed step" in d.reason


@pytest.mark.parametrize("key, index", [("joint_pos", 2), ("joint_vel", 1), ("pelvis_pos", 2),
                                        ("pelvis_quat_wxyz", 0), ("floor_z", None), ("sim_time_s", None)])
def test_a_measured_integer_too_big_for_a_float_is_a_stop_row_in_check_and_admit(key, index):
    for decide in (lambda g, st: g.check(1, st, action()), lambda g, st: g.admit(1, st)):
        gate = make_gate()
        d = decide(gate, _set(state(1), key, index, HUGE))
        assert d.verdict == "STOP" and "malformed step" in d.reason and gate.latched is not None


def test_a_huge_integer_mid_segment_stops_at_that_step_and_keeps_the_bridge_session(tmp_path):
    """Through the domain the step is answered STOP (hold steps run, the segment ends normally): the bridge
    session stays open, so the operator's reset clears it and the robot runs again."""
    def bad(i, step):
        if i == 2:
            step = json.loads(json.dumps(step))
            step["action"]["joint_targets"][2] = HUGE
        return step

    fake = ScriptedBridge(walk_script(5), bad, final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    try:
        out = domain.execute("carry_object", CARRY)
        assert out["stopped"] and out["gate"]["first_stop"]["step"] == 2, out
        assert fake.verdicts == ["pass", "pass", "stop", "stop", "stop"] and fake.applied == 2
        assert domain._session_closed is None and "malformed step" in domain.gate.latched
        assert domain.reset_stop()["ok"] is True and domain.gate.latched is None
    finally:
        domain.close()
