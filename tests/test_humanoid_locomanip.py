"""CPU tests for the humanoid loco-manipulation route: gate, verifier, bridge domain, composition.

A fake bridge (loopback TCP, the real JSON-lines protocol) stands in for Isaac Lab-Arena + GR00T:
it streams scripted measured states and proposed actions and applies an action only after the
gate's ``pass``, so the gate, the stop path and the verifier run exactly as in a live session.
"""
from __future__ import annotations

import copy
import json
import math
import socket
import threading

import pytest

from cascade.safety.humanoid_gate import HumanoidGateLimits, HumanoidSafetyGate, JointModel, tilt_deg
from cascade.skills import humanoid_locomanip as hl
from cascade.skills.humanoid_bridge import PROTOCOL, BridgeBusy, BridgeClient
from cascade.skills.humanoid_verifier import verify_carry

JOINTS = ["left_hip_pitch_joint", "left_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
          "left_hand_index_0_joint"]
LIMITS = {"max_pelvis_tilt_deg": 30.0, "min_pelvis_height_m": 0.45, "joint_position_tolerance_rad": 0.05,
          "max_joint_velocity_fraction": 1.0, "workspace_xy_m": [[-1.5, -2.6], [1.5, 1.2]],
          "max_nav_command": [0.45, 0.2, 0.45], "base_height_command_m": [0.55, 0.8],
          "command_joint_tolerance_rad": 0.02, "policy_joint_exclude": [".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
          "finger_joint_pattern": ".*_hand_.*"}
VERIFIER = {"max_object_speed_m_s": 0.05, "min_hand_clearance_m": 0.03, "z_tolerance_m": 0.03}
JOINT_MODEL = {"names": JOINTS, "lower": [-2.0, 0.0, -3.0, -3.0, -1.5708], "upper": [2.0, 2.5, 2.5, 2.5, 0.0],
               "velocity": [20.0, 20.0, 30.0, 30.0, 12.0]}
BIN = {"pos": [-0.245, -1.627, -0.264], "quat_wxyz": [0.0, 0.0, 0.0, 1.0],
       "aabb_local": [[-0.2, -0.15, 0.0], [0.2, 0.15, 0.12]], "lin_vel": [0, 0, 0]}
BOX_HALF = 0.06


def box(pos, vel=(0, 0, 0)):
    return {"pos": list(pos), "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "lin_vel": list(vel),
            "aabb_local": [[-BOX_HALF, -BOX_HALF, -BOX_HALF], [BOX_HALF, BOX_HALF, BOX_HALF]]}


def state(step=0, *, pelvis=(0.0, 0.18, 0.74), quat=(1.0, 0.0, 0.0, 0.0), q=None, qd=None, box_pos=(0.58, 0.18, 0.07),
          box_vel=(0, 0, 0), hands=None):
    return {"step": step, "sim_time_s": step * 0.02, "floor_z": 0.0, "pelvis_pos": list(pelvis),
            "pelvis_quat_wxyz": list(quat), "pelvis_lin_vel": [0, 0, 0],
            "joint_pos": list(q or [0.0, 0.3, 0.0, 0.0, 0.0]), "joint_vel": list(qd or [0.0] * 5),
            "hands": hands or {"left_hand_palm_link": [0.2, 0.4, 0.9], "right_hand_palm_link": [0.2, -0.1, 0.9]},
            "objects": {"brown_box": box(box_pos, box_vel)}, "destinations": {"blue_sorting_bin": copy.deepcopy(BIN)}}


def action(nav=(0.2, 0.0, 0.0), height=0.75, targets=None):
    return {"joint_targets": list(targets or [0.0, 0.3, 0.1, 0.1, 0.0]), "nav": list(nav), "base_height": height,
            "torso_rpy": [0.0, 0.0, 0.0]}


def tilted(deg):
    half = math.radians(deg) / 2
    return (math.cos(half), math.sin(half), 0.0, 0.0)


# ---------------------------------------------------------------------------------------- gate
def make_gate():
    return HumanoidSafetyGate(HumanoidGateLimits.from_profile(LIMITS), JointModel.from_dict(JOINT_MODEL))


def test_gate_passes_nominal_step_and_reports_every_gate():
    d = make_gate().check(3, state(3), action())
    assert d.passed and d.verdict == "PASS"
    names = [r.name for r in d.rows]
    assert names == ["state_finite", "pelvis_tilt_deg", "pelvis_height_m", "joint_position_rad",
                     "joint_velocity_frac", "workspace_m", "command_joint_rad", "command_nav",
                     "command_base_height_m"]
    assert d.clamps == () and "clamps" not in d.as_dict()


@pytest.mark.parametrize("kwargs, act, gate", [
    ({"quat": tilted(35)}, None, "pelvis_tilt_deg"),
    ({"pelvis": (0.0, 0.18, 0.3)}, None, "pelvis_height_m"),
    ({"q": [0.0, 2.6, 0.0, 0.0, 0.0]}, None, "joint_position_rad"),
    ({"qd": [0.0, 0.0, 31.0, 0.0, 0.0]}, None, "joint_velocity_frac"),
    ({"pelvis": (2.0, 0.0, 0.74)}, None, "workspace_m"),
    ({}, action(targets=[0.0, 0.3, 2.6, 0.0, 0.0]), "command_joint_rad"),
    ({}, action(nav=(0.6, 0.0, 0.0)), "command_nav"),
    ({}, action(height=0.4), "command_base_height_m"),
])
def test_gate_stops_on_each_limit(kwargs, act, gate):
    d = make_gate().check(1, state(1, **kwargs), act or action())
    assert d.verdict == "STOP"
    assert d.reason.startswith(gate)
    assert [r.name for r in d.rows if not r.ok] == [gate]


def test_gate_ignores_leg_targets_owned_by_the_whole_body_controller():
    d = make_gate().check(1, state(1), action(targets=[5.0, -5.0, 0.0, 0.0, 0.0]))
    assert d.passed


@pytest.mark.parametrize("target, applied", [(0.3, 0.0), (5.0, 0.0), (-2.0, -1.5708)])
def test_finger_targets_beyond_a_stop_are_clamped_and_recorded(target, applied):
    # left_hand_index_0_joint range [-1.5708, 0.0]: the articulation stops the finger there anyway
    d = make_gate().check(1, state(1), action(targets=[0.0, 0.3, 0.0, 0.0, target]))
    assert d.passed and [r.name for r in d.rows if not r.ok] == []
    assert len(d.clamps) == 1
    c = d.clamps[0]
    assert (c.joint, c.index, c.commanded, c.applied) == ("left_hand_index_0_joint", 4, target, applied)
    assert d.as_dict()["clamps"] == [{"joint": "left_hand_index_0_joint", "index": 4, "commanded": target,
                                      "applied": applied}]


def test_finger_targets_inside_the_range_are_not_touched():
    d = make_gate().check(1, state(1), action(targets=[0.0, 0.3, 0.0, 0.0, -0.7]))
    assert d.passed and d.clamps == ()


def test_arm_targets_are_never_clamped_beyond_the_tolerance_they_stop():
    arm = make_gate().check(1, state(1), action(targets=[0.0, 0.3, 2.5 + 0.03, 0.0, 0.0]))
    assert arm.verdict == "STOP" and arm.reason.startswith("command_joint_rad") and arm.clamps == ()
    ok = make_gate().check(1, state(1), action(targets=[0.0, 0.3, 2.5 + 0.015, 0.0, 0.0]))
    assert ok.passed and ok.clamps == ()


@pytest.mark.parametrize("kwargs, gate", [
    ({"q": [0.0, 0.3, 0.0, 0.0, 0.06]}, "joint_position_rad"),      # finger measured past its stop
    ({"q": [0.0, 0.3, 0.0, 0.0, -1.63]}, "joint_position_rad"),
    ({"qd": [0.0, 0.0, 0.0, 0.0, 12.5]}, "joint_velocity_frac"),    # finger faster than its limit
])
def test_measured_finger_violation_still_stops_even_with_a_clamped_command(kwargs, gate):
    d = make_gate().check(1, state(1, **kwargs), action(targets=[0.0, 0.3, 0.0, 0.0, 0.3]))
    assert d.verdict == "STOP" and d.reason.startswith(gate) and "left_hand_index_0_joint" in d.reason
    assert d.clamps == ()  # nothing is applied after a STOP


def test_profile_has_no_finger_command_tolerance():
    with pytest.raises(ValueError):
        HumanoidGateLimits.from_profile(dict(LIMITS, command_finger_tolerance_rad=0.25))


def test_gate_refuses_non_finite_and_malformed_steps():
    s = state(1)
    s["pelvis_pos"][2] = float("nan")
    assert make_gate().check(1, s, action()).verdict == "STOP"
    bad = state(1)
    del bad["joint_vel"]
    d = make_gate().check(1, bad, action())
    assert d.verdict == "STOP" and "malformed" in d.rows[-1].detail


def test_gate_stop_latches_until_reset_and_estop_latches():
    gate = make_gate()
    assert gate.check(1, state(1, quat=tilted(40)), action()).verdict == "STOP"
    again = gate.check(2, state(2), action())
    assert again.verdict == "STOP" and "pelvis_tilt_deg" in again.reason  # nominal step, still latched
    gate.reset()
    assert gate.check(3, state(3), action()).passed
    gate.estop("operator")
    assert gate.check(4, state(4), action()).reason == "operator"


def test_tilt_of_identity_and_quarter_turn():
    assert tilt_deg((1, 0, 0, 0)) == pytest.approx(0.0)
    assert tilt_deg(tilted(90)) == pytest.approx(90.0)
    assert tilt_deg((0, 0, 0, 1)) == pytest.approx(0.0)  # yaw only


@pytest.mark.parametrize("key, value", [("max_pelvis_tilt_deg", 95), ("workspace_xy_m", [[1, 0], [0, 1]]),
                                        ("base_height_command_m", [0.8, 0.5]), ("policy_joint_exclude", ["("])])
def test_gate_limits_refuse_bad_profiles(key, value):
    bad = dict(LIMITS, **{key: value})
    with pytest.raises(Exception):
        HumanoidGateLimits.from_profile(bad)
    with pytest.raises(ValueError):
        HumanoidGateLimits.from_profile({k: v for k, v in LIMITS.items() if k != "min_pelvis_height_m"})


# ------------------------------------------------------------------------------------ verifier
LIM = HumanoidGateLimits.from_profile(LIMITS)
IN_BIN = (-0.25, -1.62, -0.264 + 0.02 + BOX_HALF)


def test_verifier_confirms_object_resting_in_destination_released_upright():
    v = verify_carry(state(9, box_pos=IN_BIN), "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)
    assert v["status"] == "confirmed", v
    assert set(v["criteria"]) == {"object_in_destination", "object_at_rest", "object_released", "robot_upright"}
    assert v["measurements"]["footprint_margin_m"] > 0


@pytest.mark.parametrize("kwargs, failed", [
    ({"box_pos": (0.58, 0.18, 0.07)}, "object_in_destination"),            # still on the shelf
    ({"box_pos": (-0.25, -1.62, 0.6)}, "object_in_destination"),            # held above the bin
    ({"box_pos": IN_BIN, "box_vel": (0.2, 0, 0)}, "object_at_rest"),
    ({"box_pos": IN_BIN, "hands": {"l": [-0.25, -1.62 + BOX_HALF + 0.01, IN_BIN[2]], "r": [0, 0, 1]}},
     "object_released"),
    ({"box_pos": IN_BIN, "quat": tilted(60)}, "robot_upright"),
])
def test_verifier_refutes_with_the_failed_criterion(kwargs, failed):
    v = verify_carry(state(9, **kwargs), "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)
    assert v["status"] == "refuted" and failed in v["reason"]


def test_verifier_unverified_when_unmeasurable():
    assert verify_carry(None, "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)["status"] == "unverified"
    s = state(9, box_pos=IN_BIN)
    del s["hands"]
    assert verify_carry(s, "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)["status"] == "unverified"
    s = state(9, box_pos=IN_BIN)
    s["objects"]["brown_box"]["pos"][0] = float("inf")
    assert verify_carry(s, "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)["status"] == "unverified"


def test_verifier_uses_the_destination_frame():
    s = state(9, box_pos=IN_BIN)
    s["destinations"]["blue_sorting_bin"]["aabb_local"] = [[-0.3, -0.02, 0.0], [0.3, 0.02, 0.12]]  # narrow in y
    s["objects"]["brown_box"]["pos"][1] = -1.627 + 0.1
    v = verify_carry(s, "brown_box", "blue_sorting_bin", verifier=VERIFIER, gate_limits=LIM)
    assert v["status"] == "refuted"  # 180 deg yaw keeps the narrow axis on world y


# ------------------------------------------------------------------------------ fake bridge
class FakeBridge:
    """Real protocol, scripted world. ``script``: list of (state, action) per policy step; the last
    state is the terminal one. Records every gate verdict it received."""

    def __init__(self, script, *, final=None, end_reason="benchmark_success_signal", rest=2, busy=False):
        self.script, self.final, self.end_reason, self.rest = script, final, end_reason, rest
        self.verdicts, self.segments, self.applied = [], [], 0
        self.applied_targets = []  # joint targets as applied, after the gate's clamps
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(2)
        self.port = self.sock.getsockname()[1]
        self.busy = busy
        self.current = state(0)
        threading.Thread(target=self._serve, daemon=True).start()

    def hello(self):
        return {"type": "hello", "protocol": PROTOCOL, "env": "fake", "scene": "fake scene", "physics": "newton",
                "physics_device": "cpu", "policy": {"name": "Fake", "label": "fake GR00T"},
                "objects": ["brown_box"], "destinations": ["blue_sorting_bin"], "control_hz": 50,
                "joints": JOINT_MODEL}

    def _serve(self):
        while True:
            conn, _ = self.sock.accept()
            f = conn.makefile("r")
            send = lambda m: conn.sendall((json.dumps(m) + "\n").encode())  # noqa: E731
            if self.busy:
                send({"type": "busy"})
                conn.close()
                continue
            send(self.hello())
            try:
                for line in f:
                    msg = json.loads(line)
                    if msg["type"] == "session":
                        send({"type": "ready", "state": self.current})
                    elif msg["type"] == "observe":
                        send({"type": "state", "state": self.current})
                    elif msg["type"] == "segment":
                        self.segments.append(msg)
                        self._segment(msg, f, send)
                    elif msg["type"] == "close":
                        break
            finally:
                conn.close()

    def _segment(self, msg, f, send):
        reason = self.end_reason
        for i, (st, act) in enumerate(self.script):
            send({"type": "step", "i": i, "phase": "policy", "state": st, "action": act})
            reply = json.loads(f.readline())
            self.verdicts.append(reply["verdict"])
            if reply["verdict"] != "pass":
                reason = "gate_stop"
                break
            targets = list(act["joint_targets"])
            for index, value in reply.get("clamp", []):
                targets[index] = value
            self.applied_targets.append(targets)
            self.applied += 1
            self.current = st
        for k in range(self.rest):
            send({"type": "step", "i": 10_000 + k, "phase": "hold", "state": self.current, "action": action(nav=(0, 0, 0))})
            self.verdicts.append(json.loads(f.readline())["verdict"])
        if reason != "gate_stop" and self.final is not None:
            self.current = self.final
        send({"type": "segment_end", "reason": reason, "steps": self.applied, "sim_time_s": 1.0,
              "final": self.current, "arena": {"terms": {"success": reason != "gate_stop"}}, "recording": None})


def profile(port):
    return {"kind": "humanoid_locomanip", "robot_id": "unitree_g1_arena_locomanip",
            "bridge": {"host": "127.0.0.1", "ports": [port], "connect_wait_s": 0, "reply_timeout_s": 10},
            "scene": {"objects": {"brown_box": "brown box"}, "destinations": {"blue_sorting_bin": "blue bin"}},
            "instruction_template": "Pick up the {object} from the shelf, and place it into the {destination} on the table located at the right of the shelf.",
            "max_steps": 50, "rest_steps": 2, "segment_wall_s": 60, "safety": copy.deepcopy(LIMITS),
            "verifier": dict(VERIFIER)}


def walk_script(n=5, *, tilt_at=None):
    out = []
    for i in range(n):
        q = tilted(40) if tilt_at is not None and i >= tilt_at else (1.0, 0.0, 0.0, 0.0)
        out.append((state(i, quat=q, pelvis=(0.0, 0.18 - 0.3 * i, 0.74)), action()))
    return out


def test_carry_object_confirmed_through_gate_and_verifier(tmp_path):
    fake = FakeBridge(walk_script(), final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert out["ok"] and out["verified"] and out["outcome"] == "confirmed", out
    assert out["postcondition"]["status"] == "confirmed"
    assert out["gate"]["verdict"] == "PASS" and out["gate"]["steps_checked"] == 5 + 2
    assert fake.segments[0]["instruction"] == ("Pick up the brown box from the shelf, and place it into the blue bin "
                                               "on the table located at the right of the shelf.")
    assert fake.verdicts == ["pass"] * 7
    seg = json.loads((tmp_path / "segments" / "001" / "segment.json").read_text())
    assert seg["verdict"]["status"] == "confirmed" and seg["benchmark_signal"] == {"terms": {"success": True}}
    assert "benchmark" not in json.dumps(out)  # the benchmark's own signal never reaches the agent
    assert len((tmp_path / "segments" / "001" / "gate.jsonl").read_text().splitlines()) == 1 + 7
    domain.close()


def test_finger_clamps_reach_the_bridge_and_are_logged(tmp_path):
    script = [(state(i, pelvis=(0.0, 0.18 - 0.3 * i, 0.74)), action(targets=[0.0, 0.3, 0.1, 0.1, 0.05 * i]))
              for i in range(4)]
    fake = FakeBridge(script, final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert out["gate"]["verdict"] == "PASS" and out["outcome"] == "confirmed", out
    # step 0 commands the stop itself (0.0): untouched; steps 1-3 press past it: applied at the stop
    assert [t[4] for t in fake.applied_targets] == [0.0, 0.0, 0.0, 0.0]
    assert [t[2] for t in fake.applied_targets] == [0.1] * 4  # arm targets are never edited
    assert out["gate"]["finger_targets_clamped"] == {"steps": 3, "targets": 3, "worst_excess_rad": 0.15,
                                                     "worst_joint": "left_hand_index_0_joint"}
    lines = [json.loads(x) for x in (tmp_path / "segments" / "001" / "gate.jsonl").read_text().splitlines()]
    policy = [x for x in lines if x["phase"] == "policy"]
    assert "clamps" not in policy[0] and policy[2]["clamps"] == [
        {"joint": "left_hand_index_0_joint", "index": 4, "commanded": 0.1, "applied": 0.0}]
    assert policy[2]["action"]["joint_targets"][4] == 0.1  # the proposed action is logged unedited
    seg = json.loads((tmp_path / "segments" / "001" / "segment.json").read_text())
    assert seg["finger_clamps"]["targets"] == 3
    domain.close()


def test_gate_stop_halts_the_policy_and_latches(tmp_path):
    fake = FakeBridge(walk_script(6, tilt_at=3), final=state(99, box_pos=IN_BIN))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert not out["ok"] and out["stopped"] and out["outcome"] == "stopped"
    assert out["gate"]["first_stop"]["step"] == 3 and "pelvis_tilt_deg" in out["error"]
    assert fake.applied == 3 and fake.verdicts[:4] == ["pass", "pass", "pass", "stop"]
    assert out["postcondition"]["status"] in {"refuted", "unverified"}
    again = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert again["refused"] and "reset_stop" in again["error"] and len(fake.segments) == 1
    domain.reset_stop()
    fake.current = state(50)
    fake.script = walk_script(2)
    assert domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})["gate"]["verdict"] == "PASS"
    domain.close()


def test_emergency_stop_from_another_thread_reaches_the_next_step(tmp_path):
    fake = FakeBridge(walk_script(4))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    domain.stop()
    out = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert out["refused"] and fake.segments == []
    domain.close()


def test_pre_motion_gate_refuses_a_fallen_robot(tmp_path):
    fake = FakeBridge(walk_script(4))
    fake.current = state(0, quat=tilted(70))
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    out = domain.execute("carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert out["refused"] and "pre-motion" in out["error"] and fake.segments == []
    domain.close()


def test_read_tools_measure_without_moving(tmp_path):
    fake = FakeBridge(walk_script(2))
    fake.current = state(7, box_pos=IN_BIN)
    domain = hl.build(profile(fake.port), tmp_path, domain_id="carry")
    scene = domain.execute("describe_scene", {})
    assert scene["ok"] and scene["relations"] == {"brown_box in blue_sorting_bin": "confirmed"}
    assert scene["robot"]["upright"] and scene["executor"] == "fake GR00T" and scene["physics"] == "newton"
    v = domain.execute("verify_placement", {"object": "brown_box", "destination": "blue_sorting_bin"})
    assert v["ok"] and v["verdict"] == "confirmed"
    assert domain.execute("carry_object", {"object": "red_box", "destination": "blue_sorting_bin"})["ok"] is False
    assert fake.segments == []
    domain.close()


def test_busy_bridge_is_refused(tmp_path):
    fake = FakeBridge(walk_script(1), busy=True)
    with pytest.raises(BridgeBusy):
        BridgeClient("127.0.0.1", fake.port)


def test_profile_validation_refuses_remote_bridges_and_bad_templates():
    p = profile(8765)
    hl.validate_profile(p)
    for key, value in (("bridge", dict(p["bridge"], host="10.0.0.5")), ("instruction_template", "carry it"),
                       ("max_steps", 0), ("unknown", 1)):
        bad = dict(p, **{key: value})
        with pytest.raises(ValueError):
            hl.validate_profile(bad)


# ------------------------------------------------------------- composition, MCP surface, agent
def test_shipped_profile_composes_with_namespaced_tools():
    from cascade.apps.robot_runtime import robot_tool_descriptors
    from cascade.config import load_robot_config

    cfg = load_robot_config("g1_arena_locomanip")
    tools = robot_tool_descriptors(cfg)
    assert tools["carry.carry_object"].effect == "motion"
    assert tools["carry.describe_scene"].effect == "read" and tools["carry.verify_placement"].effect == "read"
    assert list(tools["carry.carry_object"].parameters["properties"]["object"]["enum"]) == ["brown_box"]


def test_agent_orchestrator_drives_the_composed_runtime(tmp_path, monkeypatch):
    from cascade.agent.llm import LLMResponse, MockLLM, ToolCall
    from cascade.agent.orchestrator import AgentOrchestrator
    from cascade.apps.robot_runtime import build_robot_runtime
    from cascade.config import Cfg, load_robot_config

    fake = FakeBridge(walk_script(3), final=state(99, box_pos=IN_BIN))
    cfg = load_robot_config("g1_arena_locomanip").as_dict()
    cfg["domains"]["carry"]["bridge"]["ports"] = [fake.port]
    runtime, _ = build_robot_runtime(Cfg(cfg), tmp_path)
    llm = MockLLM([
        LLMResponse("", [ToolCall("carry.describe_scene", {}, "1")]),
        LLMResponse("", [ToolCall("carry.carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"}, "2")]),
        LLMResponse("", [ToolCall("carry.verify_placement", {"object": "brown_box", "destination": "blue_sorting_bin"}, "3")]),
        LLMResponse("", [ToolCall("task_done", {"success": True, "summary": "box in the bin, verified"}, "4")]),
    ])
    report = AgentOrchestrator(llm, runtime, decompose=False, verify_milestones=False).run_task(
        "put the brown box in the blue bin")
    assert [e["tool"] for e in report.tool_log] == ["carry.describe_scene", "carry.carry_object",
                                                    "carry.verify_placement", "task_done"]
    assert report.success
    runtime.close()


def test_mcp_server_exposes_and_runs_the_humanoid_tools(tmp_path, monkeypatch):
    """CASCADE_ROBOT=g1_arena_locomanip: passive catalog, then a real call through the MCP server."""
    from cascade.apps.mcp_server import McpSkillServer
    from cascade.skills import humanoid_bridge

    fake = FakeBridge(walk_script(3), final=state(99, box_pos=IN_BIN))
    monkeypatch.delenv("CASCADE_BASE", raising=False)
    monkeypatch.setenv("CASCADE_ROBOT", "g1_arena_locomanip")
    monkeypatch.setenv("CASCADE_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(hl, "connect_first_free",
                        lambda host, ports, **kw: humanoid_bridge.BridgeClient(host, fake.port, **{
                            k: v for k, v in kw.items() if k == "reply_timeout_s"}))
    server = McpSkillServer()
    tools = {tool["name"] for tool in server.list_tools()}
    assert {"carry.describe_scene", "carry.carry_object", "carry.verify_placement", "emergency_stop"} <= tools
    assert server._runtime is None  # listing is passive
    out = json.loads(server.call_tool("carry.carry_object", {"object": "brown_box", "destination": "blue_sorting_bin"})
                     ["content"][-1]["text"])
    assert out["postcondition"]["status"] == "confirmed" and out["gate"]["verdict"] == "PASS"
    assert fake.verdicts == ["pass"] * 5


def _bridge_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "arena_locomanip_bridge.py"
    spec = importlib.util.spec_from_file_location("arena_locomanip_bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib only at import time; Arena is imported inside main()
    return module


class _IndexTablePolicy:
    """Arena's stock GR00T client shape: fixed simulator joint index tables, no name-based sync."""

    robot_state_joints_config = {"a": 0, "b": 1}


class _NameSyncPolicy(_IndexTablePolicy):
    def sync_joint_order(self, names):
        self.robot_state_joints_config = {n: i for i, n in enumerate(names)}
        return {"joints": len(names), "yaml_table_matches": False, "joints_at_other_indices": 2}


@pytest.mark.parametrize("backend", ["newton", "physx"])
def test_bridge_joint_order_is_synced_by_name_and_index_tables_are_refused_on_newton(backend):
    module = _bridge_module()
    bridge = module.Bridge.__new__(module.Bridge)
    bridge.joint_names = ["b", "a"]
    bridge.physics = {"backend": backend}
    bridge.policy = _NameSyncPolicy()
    assert bridge._sync_joint_order()["joints_at_other_indices"] == 2
    assert bridge.policy.robot_state_joints_config == {"b": 0, "a": 1}
    bridge.policy = _IndexTablePolicy()
    if backend == "newton":
        with pytest.raises(SystemExit, match="PhysX-order index table"):
            bridge._sync_joint_order()
    else:
        assert bridge._sync_joint_order() is None
    bridge.policy = object()  # a policy without joint tables is not touched
    assert bridge._sync_joint_order() is None


def test_bridge_observer_cameras_are_never_moved_during_an_episode():
    """A camera moved every step changed Newton's closed loop (policies that succeed without it
    scored ~0 with it): the bridge only has fixed observers, posed once in the scene config."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "scripts" / "arena_locomanip_bridge.py").read_text()
    assert "set_world_poses" not in source
    assert "_place_observer" not in source
    module = _bridge_module()
    for _, eye, target in module.observer_views():
        assert len(eye) == 3 and len(target) == 3
