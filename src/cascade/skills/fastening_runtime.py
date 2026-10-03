"""Optional mounted-fastener domain for ordinary RobotRuntime dispatch.

The injected actuator owns guarded writes/lifecycle. The separate passive reader
supplies solved states, never the actuator's completion claim. No ArmBase, pickup,
automatic engagement, seating or preload capability is implied by this domain.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

from ..control.fastening import FasteningFault, FasteningPermit, check_solve
from ..robotics.contracts import ResourceDescriptor, ToolDescriptor
from ..sim.threading_verification import ThreadContract, verify_threading


TURN_SPEC = {"name": "turn_screw",
    "description": "One measured tightening turn of the configured pre-engaged nut, then observed rest. "
        "Requires the mounted socket. Does not acquire a tool or establish seating/preload.",
    "parameters": {"type": "object", "properties": {"turns": {"type": "number", "const": 1.0},
        "direction": {"type": "string", "const": "tighten"}},
        "required": ["turns", "direction"], "additionalProperties": False}}


class FasteningDomain:
    motion_skills = frozenset({"turn_screw"})

    def __init__(self, actuator, reader, *, controller_id, domain_id="fastening",
                 clock=time.monotonic):
        self.domain_id, self.actuator, self.reader, self.clock = domain_id, actuator, reader, clock
        self.binding, self.limits = actuator.binding, actuator.limits
        if type(actuator.synthetic) is not bool:
            raise ValueError("actuator must explicitly declare synthetic status")
        resource = domain_id + "/mounted_arm_spindle"
        self.resources = (ResourceDescriptor(resource, "mounted_fastening", self.binding.robot_id,
            capabilities=("preengaged_thread_turn",), controller_id=controller_id, writer_id=resource,
            synthetic=actuator.synthetic, admission="software_only" if actuator.synthetic else "unvalidated",
            metadata={"binding_sha256": self.binding.sha256, "mounted_tool": True,
                      "preengaged_fastener": True, "seating": False, "pickup": False}),)
        self.tool_descriptors = (ToolDescriptor(domain_id + ".turn_screw",
            TURN_SPEC["description"], TURN_SPEC["parameters"], domain_id, "turn_screw",
            effect="motion", requires=(resource,), writes=(resource,)),)
        self.tool_specs = [t.as_spec() for t in self.tool_descriptors]

    def begin_task(self):
        pass  # A new task never resets the actuator latch or its physics epoch.

    def stop(self):
        return self.actuator.stop()

    def reset_stop(self):
        return self.actuator.reset_stop()

    def close(self):
        return self.actuator.close()

    def execute(self, name, args):
        result = {"ok": False, "execution_ok": False, "verified": False,
                  "synthetic": self.actuator.synthetic, "seating_verified": False,
                  "preload_verified": False, "physical_stop_verified": False,
                  "postcondition": {"status": "unverified", "reason": "no admitted solve interval"}}
        permit = None
        stopped = False
        delivery_attempted = False
        try:
            if name != "turn_screw":
                raise FasteningFault("unsupported fastening operation")
            self.tool_descriptors[0].validate_arguments(args)
            generation = self.actuator.generation
            delivery_attempted = True
            admission_started = self.clock()
            permit = self.actuator.request_turn(expected_generation=generation, **args)
            if (not isinstance(permit, FasteningPermit) or permit.binding_sha256 != self.binding.sha256 or
                    type(permit.generation) is not int or permit.generation != generation + 1 or
                    type(permit.admission_step) is not int or permit.admission_step < 0 or
                    not isinstance(permit.epoch, str) or not permit.epoch or
                    not 0 <= admission_started <= permit.admitted_monotonic_s <= self.clock() or
                    not 0 < permit.deadline_monotonic_s-permit.admitted_monotonic_s <= self.limits.max_command_wall_s+1e-9 or
                    not 0 < permit.end_simulation_time_s-permit.admission_time_s <= self.limits.max_command_sim_s+1e-9):
                raise FasteningFault("invalid controller admission receipt")
            result["admission"] = asdict(permit)
            samples, previous, cursor = [], None, permit.admission_step
            angle, last_angle = 0., None
            contract = ThreadContract(pitch_m=self.binding.thread_pitch_m, requested_turns=1., direction="tighten")
            threading = None
            while self.clock() < permit.deadline_monotonic_s:
                batch = self.reader(cursor, timeout_s=min(.05, max(0., permit.deadline_monotonic_s-self.clock())))
                if not batch:
                    continue
                for row in batch:
                    check_solve(row, self.binding, self.limits, self.clock(), epoch=permit.epoch, previous=previous)
                    if row.step != cursor + 1 or row.generation != permit.generation:
                        raise FasteningFault("post-admission solve sequence/generation mismatch")
                    if not permit.admission_time_s < row.simulation_time_s <= permit.end_simulation_time_s:
                        raise FasteningFault("solve outside admitted command interval")
                    if not math.isclose(row.simulation_time_s,
                            permit.admission_time_s+(row.step-permit.admission_step)*self.binding.dt_s,
                            rel_tol=1e-6, abs_tol=1e-9):
                        raise FasteningFault("observer clock differs from admission")
                    cursor, previous = row.step, row
                    # A pending read may return after the wall budget. Check
                    # its physical/channel faults above, but grant no new credit.
                    if self.clock() >= permit.deadline_monotonic_s:
                        raise FasteningFault("observer returned after command wall deadline")
                    if row.captured_monotonic_s < permit.admitted_monotonic_s:
                        continue
                    samples.append(row.thread_sample(self.binding))
                    x, y, z, w = row.fastener_quaternion_xyzw
                    current_angle = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
                    if last_angle is not None:
                        angle += math.atan2(math.sin(current_angle-last_angle), math.cos(current_angle-last_angle))
                    last_angle = current_angle
                    if len(samples) >= 3 and -angle >= 2*math.pi*(1-contract.turn_tolerance):
                        threading = verify_threading(samples, contract)
                        break
                    if row.simulation_time_s >= permit.end_simulation_time_s:
                        threading = verify_threading(samples, contract)
                        break
                if threading is not None:
                    break
            if threading is None:
                threading = verify_threading(samples, contract)
                result["postcondition"] = threading
                result["error"] = "one measured turn not completed within the admitted deadline"
                return result
            result["postcondition"] = threading
            result["execution_ok"] = threading["status"] == "confirmed"
            stop_started = self.clock()
            stop = self.actuator.stop()
            stopped = True
            result["stop"] = stop
            if stop.get("ok") is not True or type(stop.get("generation")) is not int:
                raise FasteningFault("stop was not acknowledged")
            rest = self._observe_rest(cursor, previous, permit, stop, stop_started)
            result["rest"] = rest
            result["physical_stop_verified"] = rest["status"] == "confirmed" and not self.actuator.synthetic
            if threading["status"] != "confirmed":
                return result
            result["postcondition"] = {"status": rest["status"],
                "reason": "independent threading and rest observed" if rest["status"] == "confirmed"
                          else "threading observed but stop/rest not confirmed",
                "threading": threading, "rest": rest, "seating_verified": False}
            result["ok"] = result["verified"] = rest["status"] == "confirmed"
            return result
        except Exception as exc:
            result["error"] = str(exc)
            # Threading may have passed before a broken rest channel. Do not
            # leave its confirmed sub-verdict as the overall tool postcondition.
            prior = result["postcondition"]
            result["postcondition"] = {"status": "unverified", "reason": str(exc), "threading": prior}
            return result
        finally:
            if delivery_attempted and not stopped:
                try:
                    result["stop"] = self.actuator.stop()
                except Exception as exc:
                    result["stop"] = {"ok": False, "error": str(exc), "physical_stop_verified": False}

    def _observe_rest(self, cursor, previous, permit, stop, stop_started):
        # Budget begins at stop delivery, never at a delayed logging return.
        accepted = stop.get("accepted_monotonic_s")
        if (type(accepted) not in (int, float) or not math.isfinite(accepted) or
                not 0 <= stop_started <= accepted <= self.clock() or
                stop.get("generation") != permit.generation + 1):
            raise FasteningFault("stop receipt lacks a valid local clock")
        deadline = accepted + self.limits.rest_timeout_wall_s
        end = previous.simulation_time_s + self.limits.rest_timeout_sim_s
        start_rest = None
        count = 0
        while self.clock() < deadline:
            batch = self.reader(cursor, timeout_s=min(.05, max(0., deadline-self.clock())))
            for row in batch:
                check_solve(row, self.binding, self.limits, self.clock(), epoch=permit.epoch, previous=previous)
                if row.step != cursor + 1:
                    raise FasteningFault("rest stream skipped solves")
                cursor, previous = row.step, row
                if row.simulation_time_s > end:
                    raise FasteningFault("physical rest deadline exceeded")
                if self.clock() >= deadline:
                    raise FasteningFault("observer returned after rest wall deadline")
                # Already-produced in-flight rows carry the original generation.
                # They remain safety observations but cannot earn stop credit.
                if row.generation == permit.generation and count == 0:
                    continue
                if row.generation != stop["generation"]:
                    raise FasteningFault("unexpected generation during stop observation")
                if row.captured_monotonic_s < accepted:
                    continue
                count += 1
                quiet = (row.commanded_spindle_effort_nm == row.spindle_effort_nm == 0. and
                    row.fastener_linear_speed_m_s <= self.limits.rest_linear_speed_m_s and
                    row.tool_linear_speed_m_s <= self.limits.rest_linear_speed_m_s and
                    max(row.fastener_angular_speed_rad_s, row.tool_angular_speed_rad_s,
                        abs(row.spindle_speed_rad_s), *(abs(v) for v in row.joint_velocity_rad_s))
                    <= self.limits.rest_angular_speed_rad_s)
                if not quiet:
                    start_rest = None
                elif start_rest is None:
                    start_rest = row.simulation_time_s
                if start_rest is not None and row.simulation_time_s-start_rest >= self.limits.rest_window_sim_s-1e-9:
                    return {"status": "confirmed", "samples": count, "last_step": row.step,
                            "generation": row.generation, "window_sim_s": row.simulation_time_s-start_rest,
                            "zero_spindle_effort": True}
        raise FasteningFault("wall budget expired before independently observed rest")
