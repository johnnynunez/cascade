"""#431 review Important A: a connect made on an operator's confirmation only connects and holds,
whatever tool triggered it.

The re-reviewer's probe (`confirmed_trigger_probe.py`): the LazyArm raised `ConnectedAndHolding`
once, AFTER its connect; a best-effort read that materialized it first (the release-kind
pre-snapshot's jaw read, `agent/effects.py` `_frac`) swallowed it, and the same call then went on
with a connected arm: `open_gripper` opened the jaw with the operator's hands on the arm, and
`handover` / `place_at` answered "not holding anything" (marker cleared, confirmation used, no
let-go ask), so the agent's next call moved.

Now (1) the LazyArm refuses every command, arm and jaw, from that connect until the agent has been
told and its next motion call starts (or, in a command-line tool, until the operator types 'go'),
(2) best-effort reads re-raise it, and (3) execute() answers 'let go of the arm and retry' for any
motion call that woke the arm on a confirmation, whatever the skill made of it.

Every `_MOTION_SKILLS` tool, on both reBot transports, as the FIRST call after a confirmation over
a held reBot: no bus write after the connect, and the let-go answer (a tool that never reaches the
arm keeps the marker and the confirmation). No test touches CAN: the bus is a recording fake.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from conftest import needs_pin
from test_held_arm_across_restart import _hold_a_rebot
from test_held_arm_followups import TARGET, _bus, _private_stores, _reads  # noqa: F401
from test_rebot_clear_error import no_sleep  # noqa: F401

from cascade.config import load_demo_config
from cascade.control import arm_hold
from cascade.control.lazy_arm import LazyArm

LET_GO = "ConnectedAndHolding: connected and holding; let go of the arm and retry"
#: bus events that write nothing (a parameter read) or only mark the connect in the log
QUIET = {"param_read", "connect_begin", "connect_end"}
#: The motion tools that never reach the arm as a first call with these arguments: refused before
#: any arm access (people guard, nothing to undo, no snapshot, no depth at the pixel, move_to_rest's
#: own held refusal) or a search that moves nothing. They leave the marker and the confirmation.
NEVER_CONNECTS = {"grasp_at_pixel", "move_to_rest", "restore_scene", "search_for_object", "throw", "undo_move"}


def _tools():
    from cascade.skills.runtime import _MOTION_SKILLS, TOOL_SPECS

    specs = {s["name"]: s for s in TOOL_SPECS}
    return sorted(t for t in _MOTION_SKILLS if t in specs), specs


TOOLS, SPECS = _tools()


def _arg(schema):
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "string":
        return "red cube"
    if kind == "integer":
        return 1
    if kind == "number":
        return 0.05
    if kind == "boolean":
        return False
    if kind == "array":
        item = schema.get("items", {})
        return [0.3, 0.0, 0.1] if item.get("type") in ("number", None) else ["red cube"]
    return {}


def _args(tool):
    params = SPECS[tool].get("parameters") or SPECS[tool].get("input_schema") or {}
    props = params.get("properties", {})
    return {k: _arg(props.get(k, {})) for k in params.get("required", [])}


def _mark_connects(monkeypatch, log):
    """Bracket every driver connect in the log: its own frames (the fault clear, the hold) are the
    connect's; anything written after it is a command."""
    from cascade.control.rebot_rs_arm import RebotRSArm
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    for cls in (RebotRSArm, RebotRSMotorBridgeArm):
        def connect(self, _orig=cls.connect):
            log.append(("connect_begin",))
            try:
                return _orig(self)
            finally:
                log.append(("connect_end",))

        monkeypatch.setattr(cls, "connect", connect)


def _commands(events):
    """Bus writes outside a connect (joint targets, MIT frames, jaw frames, mode writes, ...)."""
    out, inside = [], False
    for e in events:
        if e[0] == "connect_begin":
            inside = True
        elif e[0] == "connect_end":
            inside = False
        elif not inside and e[0] not in QUIET:
            out.append(e[0])
    return out


def _runtime(monkeypatch, tmp_path, backend):
    from cascade.apps import demo
    from cascade.control.rebot_rs_arm import RebotRSArm
    from cascade.control.rebot_rs_mb_arm import RebotRSMotorBridgeArm

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path / "run"))
    log, make = _bus(monkeypatch, backend)
    _hold_a_rebot(log, make)
    assert arm_hold.read("can0") is not None, "an earlier session left the arm holding"
    cls = RebotRSArm if backend == "rebot_rs" else RebotRSMotorBridgeArm
    cfg = load_demo_config(camera="mock", arm=backend, llm="mock")
    occ = cfg._data.get("occupancy")
    if isinstance(occ, dict) and occ.get("required"):
        occ["required"] = False
    monkeypatch.setattr(demo, "make_arm", lambda acfg, kinematics=None: cls(acfg))
    rt, arm = demo.build_runtime(cfg, tmp_path / "run2", view=False, serve=False, lazy_arm=True)
    _mark_connects(monkeypatch, log)
    return rt, arm, log


@needs_pin
@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
@pytest.mark.parametrize("tool", TOOLS)
def test_the_first_call_after_a_confirmation_only_connects_and_holds(monkeypatch, tmp_path, no_sleep,  # noqa: F811
                                                                      backend, tool):
    from cascade.apps.demo import shutdown_runtime

    rt, arm, log = _runtime(monkeypatch, tmp_path, backend)
    try:
        arm_hold.confirm("holding the arm")
        n0 = len(log)
        out = rt.execute(tool, _args(tool))
        sent = log[n0:]
        assert _commands(sent) == [], f"{tool}: written after the connect: {_commands(sent)} ({out})"
        connected = rt.arm.raw.connected
        assert connected is (tool not in NEVER_CONNECTS), (tool, out)
        if connected:
            assert ("connect_end",) in sent and [e[0] for e in sent].count("clear_error") == 7, sent
            assert out["ok"] is False and out.get("refused") == "connected_on_confirmation", out
            assert out["error"].startswith(LET_GO), out
            assert "let go" in out["next_action"], out["next_action"]
            assert arm_hold.read("can0") is None and arm_hold.pending_confirmations() == [], \
                "the connect used the confirmation and cleared the marker: the arm is connected and holds"
        else:
            assert "ConnectedAndHolding" not in str(out), out
            assert arm_hold.read("can0") is not None and len(arm_hold.pending_confirmations()) == 1, \
                "nothing connected: the marker and the confirmation are as they were"
    finally:
        shutdown_runtime(rt, arm)


@needs_pin
@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
@pytest.mark.parametrize("tool", ["open_gripper", "handover", "place_at"])
def test_after_the_let_go_answer_the_next_call_goes_through(monkeypatch, tmp_path, no_sleep, backend, tool):  # noqa: F811
    """The answer is given once: the retry runs as the first call after a cold connect would (the
    jaw tools reach the jaw; nothing held, handover and place_at say so)."""
    from cascade.apps.demo import shutdown_runtime

    rt, arm, log = _runtime(monkeypatch, tmp_path, backend)
    try:
        arm_hold.confirm("holding the arm")
        first = rt.execute(tool, _args(tool))
        assert first.get("refused") == "connected_on_confirmation", first
        n1 = len(log)
        again = rt.execute(tool, _args(tool))
        assert again.get("refused") is None and "ConnectedAndHolding" not in str(again), again
        if tool == "open_gripper":
            assert _commands(log[n1:]), "the retry opens the jaw"
        else:
            assert "not holding anything" in again["error"], again
    finally:
        shutdown_runtime(rt, arm)


@needs_pin
@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_confirmed_connect_outside_a_motion_call_answers_the_next_motion_call(monkeypatch, tmp_path, no_sleep,  # noqa: F811
                                                                                backend):
    """Whatever woke the arm (here a reader that swallowed the answer), the agent is told before
    anything moves: the next motion call sends nothing and answers let go; the one after moves."""
    from cascade.apps.demo import shutdown_runtime

    rt, arm, log = _runtime(monkeypatch, tmp_path, backend)
    try:
        _reads(monkeypatch, log, {2: 1.2, 3: 1.2})  # held up where the last session left it
        arm_hold.confirm("holding the arm")
        with pytest.raises(arm_hold.ConnectedAndHolding):
            rt.arm.raw.get_state()
        n0 = len(log)
        out = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        assert _commands(log[n0:]) == [] and out.get("refused") == "connected_on_confirmation", out
        n1 = len(log)
        again = rt.execute("rotate_joint", {"joint": 1, "degrees": 5.0})
        assert again.get("refused") is None and TARGET in [e[0] for e in log[n1:]], again
    finally:
        shutdown_runtime(rt, arm)


# ── the LazyArm latch ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_a_swallowed_answer_still_refuses_every_command(monkeypatch, no_sleep, backend):  # noqa: F811
    """A best-effort read that swallowed the connect's answer does not open the arm to commands:
    joint targets, streams and every jaw command refuse until the agent was told and its next call
    starts; reads, the soft stop and the hard e-stop still go through."""
    log, make = _bus(monkeypatch, backend)
    _mark_connects(monkeypatch, log)
    arm_hold.record("can0", arm="0", reason="held")
    arm_hold.confirm("holding the arm")
    lazy = LazyArm(make, profile_type=backend, hold_key="can0")
    with pytest.raises(arm_hold.ConnectedAndHolding):
        lazy.get_state()  # the swallowed one
    n0 = len(log)
    q = np.zeros(6)
    commands = [lambda: lazy.send_joint_target(q), lambda: lazy.set_gripper(0.5),
                lambda: lazy.stream_to(q, duration_s=0.1), lambda: lazy.stream_path([q, q], 0.1),
                lambda: lazy.wait_settled(q, 0.01, 0.1), lambda: lazy.open_gripper(),
                lambda: lazy.close_gripper_two_stage()]
    for command in commands:
        with pytest.raises(arm_hold.ConnectedAndHolding, match="let go of the arm and retry"):
            command()
    assert _commands(log[n0:]) == [], log[n0:]
    assert lazy.holding_on_confirmation and lazy.get_state() is not None, "reads go through"
    assert callable(lazy.read_positions_for_monitor) and callable(lazy.contact_joint_reader)
    lazy.next_call()
    with pytest.raises(arm_hold.ConnectedAndHolding):
        lazy.send_joint_target(q)  # nobody was told yet: the next call is refused too
    lazy.told_to_let_go()
    lazy.next_call()
    assert not lazy.holding_on_confirmation
    lazy.send_joint_target(q)
    assert TARGET in [e[0] for e in log[n0:]]


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_the_operator_lets_go_at_the_terminal(monkeypatch, no_sleep, backend):  # noqa: F811
    log, make = _bus(monkeypatch, backend)
    arm_hold.record("can0", arm="0", reason="held")
    arm_hold.confirm("holding the arm")
    lazy = LazyArm(make, profile_type=backend, hold_key="can0")
    with pytest.raises(arm_hold.ConnectedAndHolding):
        lazy.send_joint_target(np.zeros(6))
    lazy.let_go()  # 'go' typed at the terminal (calibrate)
    n0 = len(log)
    lazy.send_joint_target(np.zeros(6))
    assert [e[0] for e in log[n0:]].count(TARGET) == 1


@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_an_unconfirmed_connect_latches_nothing(monkeypatch, no_sleep, backend):  # noqa: F811
    log, make = _bus(monkeypatch, backend)
    lazy = LazyArm(make, profile_type=backend, hold_key="can0")
    lazy.get_state()
    assert not lazy.holding_on_confirmation
    lazy.send_joint_target(np.zeros(6))
    assert [e[0] for e in log].count(TARGET) == 1


def test_releasing_the_arm_drops_the_latch(monkeypatch, no_sleep):  # noqa: F811
    log, make = _bus(monkeypatch, "rebot_rs")
    arm_hold.record("can0", arm="0", reason="held")
    arm_hold.confirm("holding the arm")
    lazy = LazyArm(make, profile_type="rebot_rs", hold_key="can0")
    with pytest.raises(arm_hold.ConnectedAndHolding):
        lazy.get_state()
    lazy.disconnect()
    assert not lazy.holding_on_confirmation and not lazy.connected


def test_the_jaw_pre_snapshot_never_swallows_the_answer():
    from cascade.agent.effects import PostconditionChecker

    def jaw():
        raise arm_hold.connected_and_holding("can0")

    for kw in ({"gripper_frac": jaw}, {"holding": jaw}):
        checker = PostconditionChecker(object_pose=lambda label: None, **kw)
        with pytest.raises(arm_hold.ConnectedAndHolding):
            checker.snapshot(None, "open_gripper")
    quiet = PostconditionChecker(object_pose=lambda label: None, gripper_frac=lambda: 1 / 0, holding=lambda: 1 / 0)
    assert quiet.snapshot(None, "open_gripper") == {}, "any other failed read is still best effort"


@needs_pin
@pytest.mark.parametrize("backend", ["rebot_rs", "rebot_rs_mb"])
def test_whatever_swallows_the_answer_the_call_still_answers_let_go(monkeypatch, tmp_path, no_sleep, backend):  # noqa: F811
    """Any best-effort path that eats the connect's answer (here the whole pre-snapshot): the skill then
    finds nothing to hand over, and the call still answers 'let go', not 'not holding anything'."""
    from cascade.apps.demo import shutdown_runtime

    rt, arm, log = _runtime(monkeypatch, tmp_path, backend)
    try:
        snapshot = rt.effects.snapshot

        def swallowing(*a, **k):
            try:
                return snapshot(*a, **k)
            except arm_hold.ConnectedAndHolding:
                return {}

        monkeypatch.setattr(rt.effects, "snapshot", swallowing)
        arm_hold.confirm("holding the arm")
        n0 = len(log)
        out = rt.execute("handover", {})
        assert _commands(log[n0:]) == [] and rt.arm.raw.connected, log[n0:]
        assert out.get("refused") == "connected_on_confirmation" and out["error"].startswith(LET_GO), out
    finally:
        shutdown_runtime(rt, arm)
