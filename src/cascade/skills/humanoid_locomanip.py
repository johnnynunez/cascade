"""Humanoid loco-manipulation domain: a learned whole-body policy as a CASCADE skill executor.

``carry_object(object, destination)`` turns its arguments into the policy's language instruction
and runs ONE policy segment on the bridge-owned simulator (Isaac Lab-Arena, GR00T policy server;
see :mod:`cascade.skills.humanoid_bridge`). CASCADE stays the authority around the policy:

1. **pre-motion admission**: the humanoid gate (:mod:`cascade.safety.humanoid_gate`) vets the
   measured state before anything moves; a failure refuses the call;
2. **per-step gate**: every control step's measured state and proposed action are vetted before
   the bridge may apply the action; finger targets beyond a joint stop are clamped onto the joint
   range (sent with the ``pass``, applied by the bridge, logged per step and counted in the
   result). The first STOP latches; the bridge then holds (zero navigation command, last targets)
   and ends the segment; the result says ``stopped``;
3. **independent verifier** (:mod:`cascade.skills.humanoid_verifier`) on the measured state after
   the rest window: confirmed / refuted / unverified with measurements. The benchmark's own
   success signal is recorded to the segment file for comparison only; it never reaches the
   verdict or the agent.

``describe_scene`` and ``verify_placement`` are read-only (one measured snapshot, no physics step).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import threading
import time

from ..robotics.contracts import ResourceDescriptor, ToolDescriptor
from ..safety.humanoid_gate import GATES, HumanoidGateLimits, HumanoidSafetyGate, JointModel, tilt_deg
from .humanoid_bridge import BridgeError, connect_first_free
from .humanoid_verifier import placement_measurements, verify_carry

KIND = "humanoid_locomanip"
RESOURCE = "g1_whole_body"


def tool_specs(profile: dict) -> list[dict]:
    scene = profile["scene"]
    objects, dests = sorted(scene["objects"]), sorted(scene["destinations"])
    obj_help = "; ".join(f"{k} = {v}" for k, v in sorted(scene["objects"].items()))
    dest_help = "; ".join(f"{k} = {v}" for k, v in sorted(scene["destinations"].items()))
    return [
        {"name": "describe_scene",
         "description": ("Look at the humanoid's scene: the objects and destinations that exist, their measured "
                         "positions, whether an object already rests in a destination, the robot's posture "
                         "(pelvis height and tilt) and which learned policy executes carry_object. Reads one "
                         "measured simulator snapshot; nothing moves."),
         "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
        {"name": "carry_object",
         "description": ("Have the humanoid walk to `object`, pick it up, carry it and put it into `destination`, "
                         "executed by a learned language-conditioned loco-manipulation policy (one policy segment, "
                         "up to ~30 s simulated). Every control step is gated by the humanoid safety harness "
                         "(fall/tilt, pelvis height, joint position and velocity limits, workspace, command "
                         "limits); a gate STOP halts the policy and the result says stopped. Afterwards an "
                         "independent verifier measures whether the object rests inside the destination, "
                         "released, with the robot upright: confirmed, refuted or unverified. The policy is "
                         f"learned and can fail. Objects: {obj_help}. Destinations: {dest_help}."),
         "parameters": {"type": "object", "properties": {
             "object": {"type": "string", "enum": objects},
             "destination": {"type": "string", "enum": dests}},
             "required": ["object", "destination"], "additionalProperties": False}},
        {"name": "verify_placement",
         "description": ("Independently check, from one fresh measured snapshot, whether `object` rests inside "
                         "`destination`, released, with the robot upright. Returns confirmed, refuted or "
                         "unverified with the measurements. Nothing moves."),
         "parameters": {"type": "object", "properties": {
             "object": {"type": "string", "enum": objects},
             "destination": {"type": "string", "enum": dests}},
             "required": ["object", "destination"], "additionalProperties": False}},
    ]


MOTION_SKILLS = frozenset({"carry_object"})


def resources(domain_id: str, profile: dict) -> tuple[ResourceDescriptor, ...]:
    bridge = profile["bridge"]
    return (ResourceDescriptor(f"{domain_id}/{RESOURCE}", "humanoid_whole_body", profile["robot_id"],
                               capabilities=("learned_loco_manipulation",),
                               controller_id=f"locomanip-bridge:{bridge['host']}:{'+'.join(map(str, bridge['ports']))}",
                               writer_id=f"{domain_id}/{RESOURCE}", synthetic=False, admission="unvalidated",
                               metadata={"executor": "learned policy behind a bridge",
                                         "gate": "cascade.safety.humanoid_gate", "verifier":
                                         "cascade.skills.humanoid_verifier", "physical_admission": False}),)


def _r(values, n=3):
    return [round(float(v), n) for v in values]


class LocomanipDomain:
    motion_skills = MOTION_SKILLS

    def __init__(self, profile: dict, directory: Path, *, domain_id: str, client=None, seed: int | None = None):
        self.profile, self.domain_id = profile, domain_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tool_specs = tool_specs(profile)
        self.resources = resources(domain_id, profile)
        ids = tuple(r.resource_id for r in self.resources)
        self.tool_descriptors = tuple(ToolDescriptor(
            f"{domain_id}.{s['name']}", s["description"], s["parameters"], domain_id, s["name"],
            effect="motion" if s["name"] in MOTION_SKILLS else "read", requires=ids,
            writes=ids if s["name"] in MOTION_SKILLS else ()) for s in self.tool_specs)
        self.limits = HumanoidGateLimits.from_profile(profile["safety"])
        self.verifier = dict(profile["verifier"])
        self.gate = HumanoidSafetyGate(self.limits)
        self._io = threading.Lock()  # one bridge conversation at a time
        self._segments = 0
        self._session_closed: str | None = None  # why the bridge session was closed mid-segment
        bridge = profile["bridge"]
        self.client = client or connect_first_free(bridge["host"], bridge["ports"], wait_s=bridge["connect_wait_s"],
                                                   reply_timeout_s=bridge["reply_timeout_s"])
        hello = self.client.hello
        self.gate.bind(JointModel.from_dict(hello["joints"]))
        self.hello = {k: hello.get(k) for k in ("env", "physics", "physics_device", "policy", "objects",
                                                "destinations", "control_hz", "scene")}
        missing = (set(profile["scene"]["objects"]) - set(hello.get("objects") or ())) | (
            set(profile["scene"]["destinations"]) - set(hello.get("destinations") or ()))
        if missing:
            self.client.close()
            raise BridgeError(f"bridge scene lacks profile entries {sorted(missing)}")
        ready = self.client.request({"type": "session", "seed": seed}, "ready",
                                    timeout_s=float(bridge["reply_timeout_s"]) * 4)
        # the bridge picks a seed when none is requested; record the one the scene was reset with
        self._write("session.json", {"hello": hello, "ready_state": ready.get("state"), "seed": ready.get("seed", seed),
                                     "limits": self.limits.as_dict(), "verifier": self.verifier})

    # -- lifecycle ---------------------------------------------------------------------------
    def begin_task(self):
        pass

    def stop(self):
        self.gate.estop("emergency stop requested")
        return {"ok": True, "latched": True, "physical_stop_verified": False,
                "detail": "the bridge receives STOP at its next control step and holds"}

    def reset_stop(self):
        if self._session_closed is not None:
            # The bridge session ended mid-segment on a stream this side no longer trusted: nothing can
            # run on it, so the stop stays latched until the operator rebuilds the domain (a restart).
            return {"ok": False, "latched": True,
                    "error": f"bridge session closed mid-segment ({self._session_closed}); the stop stays latched "
                             f"until the operator restarts the robot runtime"}
        return {"ok": True, "cleared": self.gate.reset()}  # the stop it cleared, for the operator's log

    def close(self):
        with self._io:
            self.client.close()
        return {"ok": True}

    # -- helpers -----------------------------------------------------------------------------
    def _write(self, name, value):
        (self.directory / name).write_text(json.dumps(value, indent=1, default=str) + "\n")

    def observe(self) -> dict:
        with self._io:
            return self.client.request({"type": "observe"}, "state")["state"]

    def truth(self) -> dict:
        """Measured physics snapshot for offline checks (Harness² hidden checks)."""
        return self.observe()

    def instruction(self, obj: str, dest: str) -> str:
        scene = self.profile["scene"]
        return self.profile["instruction_template"].format(object=scene["objects"][obj],
                                                           destination=scene["destinations"][dest])

    def _robot(self, state):
        tilt = tilt_deg(state["pelvis_quat_wxyz"])
        height = float(state["pelvis_pos"][2]) - float(state.get("floor_z", 0.0))
        return {"pelvis_position_m": _r(state["pelvis_pos"]), "pelvis_tilt_deg": round(tilt, 1),
                "pelvis_height_m": round(height, 3),
                "upright": tilt <= self.limits.max_pelvis_tilt_deg and height >= self.limits.min_pelvis_height_m}

    # -- tools -------------------------------------------------------------------------------
    def execute(self, name, args):
        try:
            descriptor = next((t for t in self.tool_descriptors if t.local_name == name), None)
            if descriptor is None:
                raise ValueError(f"unknown humanoid tool {name!r}")
            descriptor.validate_arguments(args)
            if name == "describe_scene":
                return self._describe()
            if name == "verify_placement":
                state = self.observe()
                verdict = verify_carry(state, args["object"], args["destination"], verifier=self.verifier,
                                       gate_limits=self.limits)
                return {"ok": True, "verdict": verdict["status"], "reason": verdict["reason"],
                        "criteria": _compact(verdict.get("criteria")), "measured_at_step": state.get("step")}
            return self._carry(args["object"], args["destination"])
        except BridgeError as exc:
            self.gate.estop(f"bridge fault: {exc}")
            return {"ok": False, "execution_ok": False, "verified": False, "outcome": "unverified",
                    "error": f"bridge fault: {exc}",
                    "postcondition": {"status": "unverified", "reason": "bridge fault; nothing verified"}}
        except ValueError as exc:
            return {"ok": False, "execution_ok": False, "verified": False, "error": str(exc),
                    "postcondition": {"status": "unverified", "reason": str(exc)}}

    def _describe(self):
        state = self.observe()
        scene = self.profile["scene"]
        objects = []
        for key, label in sorted(scene["objects"].items()):
            entry = (state.get("objects") or {}).get(key)
            objects.append({"name": key, "label": label,
                            "position_m": _r(entry["pos"]) if entry else None})
        dests = []
        for key, label in sorted(scene["destinations"].items()):
            entry = (state.get("destinations") or {}).get(key)
            dests.append({"name": key, "label": label, "position_m": _r(entry["pos"]) if entry else None})
        relations = {}
        for o in scene["objects"]:
            for d in scene["destinations"]:
                v = verify_carry(state, o, d, verifier=self.verifier, gate_limits=self.limits)
                relations[f"{o} in {d}"] = v["status"]
        policy = self.hello.get("policy") or {}
        return {"ok": True, "scene": self.hello.get("scene") or self.hello.get("env"),
                "physics": self.hello.get("physics"), "executor": policy.get("label") or policy.get("name"),
                "objects": objects, "destinations": dests, "relations": relations,
                "robot": self._robot(state), "safety_stop": self.gate.latched, "measured_at_step": state.get("step")}

    def _carry(self, obj, dest):
        self._segments += 1
        seg_dir = self.directory / "segments" / f"{self._segments:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)
        instruction = self.instruction(obj, dest)
        policy = self.hello.get("policy") or {}
        base = {"instruction": instruction, "executor": policy.get("label") or policy.get("name"),
                "physics": self.hello.get("physics"), "segment_dir": str(seg_dir)}
        latched = self.gate.latched
        if latched is not None:
            return {**base, "ok": False, "execution_ok": False, "verified": False, "refused": True,
                    "outcome": "refused",
                    "error": f"stopped: ask the operator to clear the stop (humanoid safety stop latched: {latched})",
                    "postcondition": {"status": "unverified", "reason": "refused before motion"}}
        t0 = time.monotonic()
        wall = float(self.profile["segment_wall_s"])
        gate_log = (seg_dir / "gate.jsonl").open("w")
        summary = {g: None for g in GATES}
        clamped = {"steps": 0, "targets": 0, "worst_excess_rad": 0.0, "worst_joint": ""}
        first_stop = None
        steps = 0
        end = None
        streaming = False  # True once the bridge runs the segment and waits on our gate replies
        try:
            with self._io:
                start = self.client.request({"type": "observe"}, "state")["state"]
                admission = self.gate.admit(start.get("step", 0) if isinstance(start, dict) else None, start)
                gate_log.write(json.dumps({"phase": "admission", **admission.as_dict()}) + "\n")
                if not admission.passed:
                    return {**base, "ok": False, "execution_ok": False, "verified": False, "refused": True,
                            "outcome": "refused", "error": f"pre-motion gate STOP: {admission.reason}",
                            "gate": {"admission": admission.as_dict()},
                            "postcondition": {"status": "unverified", "reason": "refused before motion"}}
                self.client.send({"type": "segment", "id": self._segments, "object": obj, "destination": dest,
                                  "instruction": instruction, "max_steps": int(self.profile["max_steps"]),
                                  "rest_steps": int(self.profile["rest_steps"])})
                streaming = True
                while True:
                    msg = self.client.receive()
                    if msg["type"] == "segment_end":
                        end = msg
                        break
                    if msg["type"] != "step":
                        raise BridgeError(f"unexpected {msg['type']!r} during a segment")
                    if time.monotonic() - t0 > wall:
                        self.gate.estop(f"segment wall budget {wall:.0f} s exceeded")
                    # A malformed step (index, state or action; a step without an action) is a gate
                    # STOP that latches, and the bridge is still answered: it switches to its hold
                    # and ends the segment instead of waiting for a reply that never comes.
                    state = msg.get("state")
                    decision = self.gate.check(msg.get("i"), state, msg.get("action"))
                    # on pass the bridge applies the action with these finger targets replaced
                    self.client.send({"type": "gate", "i": msg.get("i"), "verdict": "pass" if decision.passed else "stop",
                                      "reason": decision.reason,
                                      "clamp": [[c.index, c.applied] for c in decision.clamps]})
                    if decision.clamps:
                        clamped["steps"] += 1
                        clamped["targets"] += len(decision.clamps)
                        worst = max(decision.clamps, key=lambda c: c.excess_rad)
                        if worst.excess_rad > clamped["worst_excess_rad"]:
                            clamped["worst_excess_rad"] = round(worst.excess_rad, 4)
                            clamped["worst_joint"] = worst.joint
                    steps += 1
                    phase = msg.get("phase", "policy")
                    gate_log.write(json.dumps({"phase": phase,
                                               "sim_time_s": state.get("sim_time_s") if isinstance(state, dict) else None,
                                               **decision.as_dict(), "action": msg.get("action")}) + "\n")
                    for row in decision.rows:
                        if row.value is None:
                            continue
                        prev = summary.get(row.name)
                        worse = (prev is None or (row.value < prev["value"] if row.name == "pelvis_height_m"
                                                  else row.value > prev["value"]))
                        if worse:
                            summary[row.name] = {"value": round(row.value, 4), "limit": row.limit, "step": decision.step}
                    if not decision.passed and first_stop is None:
                        first_stop = {"step": decision.step, "reason": decision.reason, "phase": phase}
        except BaseException as exc:
            if streaming:
                # Mid-segment the bridge may be waiting for a gate reply on a stream this side no
                # longer trusts (a malformed line, an unexpected message, a local fault): latch,
                # then end the session (the bridge reads a close, never a pass), so nothing more
                # is applied and the bridge is not left hanging.
                self.gate.estop(f"segment aborted: {type(exc).__name__}: {exc}")
                self._session_closed = f"{type(exc).__name__}: {exc}"
                self.client.close()
            raise
        finally:
            gate_log.close()
        final = end.get("final")
        verdict = verify_carry(final, obj, dest, verifier=self.verifier, gate_limits=self.limits)
        stopped = first_stop is not None or end.get("reason") == "gate_stop"
        record = {"object": obj, "destination": dest, **base, "end": {k: end.get(k) for k in ("reason", "steps", "sim_time_s")},
                  "benchmark_signal": end.get("arena"), "gate_summary": summary, "finger_clamps": clamped,
                  "first_stop": first_stop,
                  "verdict": verdict, "final_state": final, "recording": end.get("recording"),
                  "wall_s": round(time.monotonic() - t0, 1)}
        self._write(f"segments/{self._segments:03d}/segment.json", record)
        status = verdict["status"]
        result = {**base, "ok": status == "confirmed" and not stopped,
                  "execution_ok": not stopped and end.get("reason") != "bridge_fault",
                  "verified": status == "confirmed", "stopped": stopped,
                  "outcome": "stopped" if stopped else status,
                  "segment": {"ended_by": _agent_reason(end.get("reason")), "control_steps": end.get("steps"),
                              "sim_time_s": end.get("sim_time_s")},
                  "gate": {"verdict": "STOP" if stopped else "PASS", "steps_checked": steps, "first_stop": first_stop,
                           "finger_targets_clamped": clamped,
                           "worst": {k: v for k, v in summary.items() if v is not None}},
                  "postcondition": {"status": status, "reason": verdict["reason"],
                                    "criteria": _compact(verdict.get("criteria"))}}
        if stopped:
            result["error"] = f"humanoid safety gate STOP: {first_stop['reason'] if first_stop else end.get('reason')}"
        return result


def _agent_reason(reason):
    """What ended the segment, as the agent may see it: the benchmark's own success/failure terms
    are not evidence, so they are reported only as "the episode terminated"."""
    if reason in ("gate_stop", "time_out", "budget"):
        return {"gate_stop": "safety gate stop", "time_out": "time budget (30 s)", "budget": "step budget"}[reason]
    return "episode terminated"


def _compact(criteria):
    if not criteria:
        return {}
    return {k: {kk: v.get(kk) for kk in ("ok", "value", "limit") if kk in v} for k, v in criteria.items()}


def validate_profile(profile) -> None:
    required = {"kind", "bridge", "scene", "instruction_template", "max_steps", "rest_steps", "segment_wall_s",
                "safety", "verifier"}
    if not isinstance(profile, dict) or required - set(profile) or set(profile) - required - {"robot_id"}:
        raise ValueError(f"humanoid_locomanip requires exactly {sorted(required)} (+ robot_id)")
    if profile["kind"] != KIND:
        raise ValueError("not a humanoid_locomanip profile")
    bridge = profile["bridge"]
    if (not isinstance(bridge, dict) or set(bridge) != {"host", "ports", "connect_wait_s", "reply_timeout_s"}
            or bridge["host"] not in ("127.0.0.1", "localhost", "::1")
            or not isinstance(bridge["ports"], list) or not bridge["ports"]
            or any(type(p) is not int or not 0 < p < 65536 for p in bridge["ports"])):
        raise ValueError("bridge requires a loopback host, ports, connect_wait_s and reply_timeout_s")
    for key in ("connect_wait_s", "reply_timeout_s"):
        if type(bridge[key]) not in (int, float) or not math.isfinite(bridge[key]) or bridge[key] < 0:
            raise ValueError(f"bridge {key} must be a nonnegative number")
    scene = profile["scene"]
    if (not isinstance(scene, dict) or set(scene) != {"objects", "destinations"}
            or any(not isinstance(scene[k], dict) or not scene[k] for k in scene)
            or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
                   for part in scene.values() for k, v in part.items())):
        raise ValueError("scene requires nonempty objects and destinations maps (name -> spoken label)")
    template = profile["instruction_template"]
    if not isinstance(template, str) or "{object}" not in template or "{destination}" not in template:
        raise ValueError("instruction_template must contain {object} and {destination}")
    for key in ("max_steps", "rest_steps"):
        if type(profile[key]) is not int or profile[key] < 0 or (key == "max_steps" and profile[key] == 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if type(profile["segment_wall_s"]) not in (int, float) or not 0 < profile["segment_wall_s"] <= 3600:
        raise ValueError("segment_wall_s must be in (0, 3600]")
    HumanoidGateLimits.from_profile(profile["safety"])
    from .humanoid_verifier import validate_verifier_limits
    validate_verifier_limits(profile["verifier"])


def describe(domain_id: str, profile: dict):
    validate_profile(profile)
    return resources(domain_id, profile), tool_specs(profile)


def build(profile: dict, directory: Path, *, domain_id: str) -> LocomanipDomain:
    validate_profile(profile)
    return LocomanipDomain(copy.deepcopy(profile), directory, domain_id=domain_id)
