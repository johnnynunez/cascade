"""Bounded free finger motion verified from passive completed-solve samples."""
from dataclasses import asdict

from ..control.hand import HandFault, check_hand_sample, quiet
from ..robotics.contracts import ToolDescriptor


HAND_SPECS = [
    {"name": "get_hand_state", "description": "Read the latest completed joint state and enabled contact ledger; does not step physics.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "move_fingers", "description": "Move all 16 configured finger joints within the fixed free-motion envelope and verify retained rest. Any enabled loaded contact vetoes this task; no grasp or tactile calibration is implied.",
     "parameters": {"type": "object", "properties": {"positions_rad": {"type": "array", "minItems": 16,
         "maxItems": 16, "items": {"type": "number"}}}, "required": ["positions_rad"], "additionalProperties": False}},
]


class HandDomain:
    motion_skills = frozenset({"move_fingers"})

    def __init__(self, controller, resources, *, domain_id="hand"):
        self.controller, self.resources, self.domain_id = controller, resources, domain_id
        self.tool_specs = HAND_SPECS
        ids = tuple(r.resource_id for r in resources)
        self.tool_descriptors = tuple(ToolDescriptor(domain_id+"."+s["name"], s["description"],
            s["parameters"], domain_id, s["name"], effect="motion" if s["name"] in self.motion_skills else "read",
            requires=ids, writes=ids if s["name"] in self.motion_skills else ()) for s in HAND_SPECS)

    def begin_task(self):
        pass

    def stop(self):
        return self.controller.stop()

    def reset_stop(self):
        return self.controller.reset_stop()

    def close(self):
        return self.controller.close()

    def execute(self, name, args):
        c = self.controller
        result = {"ok": False, "execution_ok": False, "verified": False,
            "synthetic": c.backend.synthetic, "physical_stop_verified": False,
            "postcondition": {"status": "unverified", "reason": "no completed hand task"}}
        attempted = stopped = False
        try:
            descriptor = next((t for t in self.tool_descriptors if t.local_name == name), None)
            if descriptor is None:
                raise HandFault("unknown hand operation")
            descriptor.validate_arguments(args)
            if name == "get_hand_state":
                rows = c.read()
                if not rows:
                    raise HandFault("no completed hand sample available")
                return {"ok": True, "synthetic": c.backend.synthetic,
                    "joint_names": c.backend.joint_names, "sample": asdict(rows[-1]),
                    "measurement_kind": "synthetic" if c.backend.synthetic else "physics",
                    "tactile_calibration": None}
            attempted = True
            permit = c.admit(args["positions_rad"])
            target = tuple(args["positions_rad"])
            result["admission"] = permit
            cursor, previous, since = permit["step"], None, None
            new_generation_seen = False
            reached = None
            while c.clock() < permit["deadline"] and reached is None:
                for row in c.read(cursor):
                    check_hand_sample(row, c.backend, c.clock(), previous=previous)
                    if row.step != cursor+1 or row.step > permit["end_step"]:
                        raise HandFault("hand approach stream skipped or exceeded its physical budget")
                    cursor, previous = row.step, row
                    if c.clock() >= permit["deadline"]:
                        raise HandFault("hand approach observation arrived after its original deadline")
                    if row.generation == permit["generation"]-1 and not new_generation_seen:
                        continue
                    if row.generation != permit["generation"]:
                        raise HandFault("hand command generation was revoked")
                    new_generation_seen = True
                    if row.captured_monotonic_s < permit["accepted"]:
                        continue
                    at_target = max(abs(q-goal) for q, goal in zip(row.position_rad, target)) <= c.limits.target_tolerance_rad
                    if at_target and quiet(row, c.limits):
                        if since is None:
                            since = row.simulation_time_s
                        if row.simulation_time_s-since >= c.limits.quiet_window_sim_s-1e-9:
                            reached = row
                            break
                    else:
                        since = None
            if reached is None:
                raise HandFault("hand did not reach target/rest within its original budget")
            result["execution_ok"] = True
            ack = c.stop()
            result["stop"], stopped = ack, True
            if not ack["ok"] or ack["generation"] != permit["generation"]+1:
                raise HandFault("hand stop was not acknowledged in the next generation")
            rest = self._rest(permit, ack, reached, target)
            result.update(ok=True, verified=True, physical_stop_verified=not c.backend.synthetic,
                rest=rest, postcondition={"status": "confirmed", "reason": "observed bounded finger targets retained through servo-hold rest",
                    "target_error_rad": rest["target_error_rad"], "rest": rest})
        except Exception as exc:
            try:
                reason = str(exc)
            except Exception:
                reason = type(exc).__name__+": unreadable exception"
            result.update(error=reason, postcondition={"status": "unverified", "reason": reason})
        finally:
            if attempted and not stopped:
                try:
                    result["stop"] = c.stop()
                except Exception:
                    result.update(ok=False, verified=False, physical_stop_verified=False,
                        stop={"ok": False, "error": "hand stop acknowledgement failed"})
        return result

    def _rest(self, permit, ack, previous, target):
        c = self.controller
        deadline = ack["accepted_monotonic_s"]+c.limits.rest_wall_s
        end = previous.simulation_time_s+c.limits.rest_sim_s
        cursor, since, new_generation_seen = previous.step, None, False
        while c.clock() < deadline:
            for row in c.read(cursor):
                check_hand_sample(row, c.backend, c.clock(), previous=previous)
                cursor, previous = row.step, row
                if c.clock() >= deadline or row.simulation_time_s > end:
                    raise HandFault("hand rest observation exceeded its original deadline")
                if row.generation == permit["generation"] and not new_generation_seen:
                    continue
                if row.generation != ack["generation"]:
                    raise HandFault("hand rest generation changed")
                new_generation_seen = True
                if row.captured_monotonic_s < ack["accepted_monotonic_s"]:
                    continue
                error = max(abs(q-goal) for q, goal in zip(row.position_rad, target))
                if (row.commanded_position_rad != tuple(ack["targets_rad"])
                        or error > c.limits.target_tolerance_rad or not quiet(row, c.limits)):
                    raise HandFault("hand target or rest was lost after the stop ACK")
                if since is None:
                    since = row.simulation_time_s
                elapsed = row.simulation_time_s-since
                if elapsed >= c.limits.quiet_window_sim_s-1e-9:
                    c.confirm_generation(ack["generation"], deadline)
                    return {"status": "confirmed", "window_sim_s": elapsed, "last_step": row.step,
                        "generation": row.generation, "target_error_rad": error,
                        "hold": "last position-servo targets", "motor_power_off": False}
        raise HandFault("hand stop lacks measured retained rest")
