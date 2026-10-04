"""Fixed mounted approach, observed shoulder loading and motor-off retention."""
from dataclasses import asdict
import math

from ..control.fastening import FasteningFault, FasteningPermit, check_solve
from ..control.fastening_seat import loaded_seat, retained_seat
from ..sim.threading_verification import ThreadContract, verify_threading


SEAT_SPEC = {"name": "seat_fastener",
    "description": "Seat the configured initially pre-engaged Factory M20 nut after a measured "
        "15-turn approach, half a second of shoulder loading and two seconds of motor-off retention. "
        "Requires the mounted socket; does not acquire a tool or verify calibrated preload.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}


def execute_seating(domain, args):
    actuator, binding, limits, clock = domain.actuator, domain.binding, domain.limits, domain.clock
    task = limits.seating
    result = {"ok": False, "execution_ok": False, "verified": False,
        "synthetic": actuator.synthetic, "seating_verified": False, "preload_verified": False,
        "physical_stop_verified": False,
        "postcondition": {"status": "unverified", "reason": "no admitted seating interval"}}
    attempted = stopped = False
    try:
        if task is None:
            raise FasteningFault("shoulder seating is not configured")
        descriptor = next(tool for tool in domain.tool_descriptors if tool.local_name == "seat_fastener")
        descriptor.validate_arguments(args)
        generation = actuator.generation
        started = clock()
        attempted = True
        permit = actuator.request_seating(expected_generation=generation)
        if (not isinstance(permit, FasteningPermit) or permit.operation != "seat"
                or permit.binding_sha256 != binding.sha256 or permit.generation != generation+1
                or not 0 <= started <= permit.admitted_monotonic_s <= clock()
                or not 0 < permit.deadline_monotonic_s-permit.admitted_monotonic_s <= task.max_command_wall_s+1e-9
                or not 0 < permit.end_simulation_time_s-permit.admission_time_s <= task.max_command_sim_s+1e-9):
            raise FasteningFault("invalid seating admission receipt")
        result["admission"] = asdict(permit)
        samples, previous, cursor = [], None, permit.admission_step
        loaded_since = loaded_at = None
        while clock() < permit.deadline_monotonic_s and loaded_at is None:
            batch = domain.reader(cursor, timeout_s=min(.05, max(0., permit.deadline_monotonic_s-clock())))
            for row in batch:
                check_solve(row, binding, limits, clock(), epoch=permit.epoch,
                            previous=previous, stage="seating_observation")
                if row.step != cursor+1 or row.generation != permit.generation:
                    raise FasteningFault("seating solve sequence/generation mismatch")
                if (not permit.admission_time_s < row.simulation_time_s <= permit.end_simulation_time_s
                        or not math.isclose(row.simulation_time_s,
                            permit.admission_time_s+(row.step-permit.admission_step)*binding.dt_s,
                            rel_tol=1e-6, abs_tol=1e-9)):
                    raise FasteningFault("seating solve outside admitted physical interval")
                cursor, previous = row.step, row
                if clock() >= permit.deadline_monotonic_s:
                    raise FasteningFault("seating observation returned after command deadline")
                if row.captured_monotonic_s < permit.admitted_monotonic_s:
                    continue
                samples.append(row.thread_sample(binding))
                if not loaded_seat(row, task):
                    loaded_since = None
                elif loaded_since is None:
                    loaded_since = row.simulation_time_s
                if loaded_since is not None and row.simulation_time_s-loaded_since >= task.loaded_window_sim_s-1e-9:
                    loaded_at = row.simulation_time_s
                    break
        if loaded_at is None:
            raise FasteningFault("no measured loaded shoulder window within the original seating budget")
        # Stop immediately, before any potentially expensive terminal analysis.
        stop_started = clock()
        stop = actuator.stop()
        result["stop"] = stop
        stopped = True
        accepted = stop.get("accepted_monotonic_s")
        if (stop.get("ok") is not True or type(stop.get("generation")) is not int
                or stop["generation"] != permit.generation+1
                or type(accepted) not in (int, float) or not math.isfinite(accepted)
                or not stop_started <= accepted <= clock()):
            raise FasteningFault("seating stop lacks a valid generation/clock ACK")
        admitted_count = len(samples)
        rest = _observe_seated_rest(domain, permit, stop, previous, cursor, samples)
        contract = ThreadContract(pitch_m=binding.thread_pitch_m, requested_turns=task.minimum_turns)
        approach = verify_threading(samples[:admitted_count], contract)
        terminal = verify_threading(samples, contract)
        result.update(pre_stop_threading=approach, final_threading=terminal, rest=rest,
            loaded_window_sim_s=loaded_at-loaded_since)
        if clock() >= accepted+task.rest_timeout_wall_s:
            raise FasteningFault("seating verification exceeded its original rest wall budget")
        result["execution_ok"] = approach["status"] == "confirmed"
        confirmed = result["execution_ok"] and terminal["status"] == "confirmed"
        result.update(ok=confirmed, verified=confirmed, seating_verified=confirmed,
            physical_stop_verified=not actuator.synthetic,
            postcondition={"status": "confirmed" if confirmed else "refuted",
                "reason": "measured threading, loaded shoulder and retained motor-off rest" if confirmed
                    else "threading approach or its retention did not satisfy the declared contract",
                "threading": terminal, "rest": rest, "seating_verified": confirmed, "preload_verified": False})
    except Exception as exc:
        try:
            reason = str(exc)
        except BaseException:
            reason = type(exc).__name__+": exception message unavailable"
        result.update(error=reason, postcondition={"status": "unverified", "reason": reason})
    finally:
        if attempted and not stopped:
            try:
                result["stop"] = actuator.stop()
            except Exception:
                result["stop"] = {"ok": False, "physical_stop_verified": False}
    return result


def _observe_seated_rest(domain, permit, stop, previous, cursor, samples):
    binding, limits, clock = domain.binding, domain.limits, domain.clock
    task = limits.seating
    accepted = stop["accepted_monotonic_s"]
    deadline = accepted+task.rest_timeout_wall_s
    end_sim = previous.simulation_time_s+limits.rest_timeout_sim_s
    zero_since = None
    count = 0
    while clock() < deadline:
        batch = domain.reader(cursor, timeout_s=min(.05, max(0., deadline-clock())))
        for row in batch:
            check_solve(row, binding, limits, clock(), epoch=permit.epoch,
                        previous=previous, stage="seating_rest_observation")
            if row.step != cursor+1:
                raise FasteningFault("seating rest stream skipped solves")
            cursor, previous = row.step, row
            if row.simulation_time_s > end_sim or clock() >= deadline:
                raise FasteningFault("seating rest observation exceeded its original deadline")
            samples.append(row.thread_sample(binding))
            if row.generation == permit.generation and zero_since is None:
                continue
            if row.generation != stop["generation"]:
                raise FasteningFault("seating rest generation changed")
            if row.captured_monotonic_s < accepted:
                continue
            if row.commanded_spindle_effort_nm != 0. or row.spindle_effort_nm != 0.:
                raise FasteningFault("nonzero spindle effort after acknowledged seating stop")
            if zero_since is None:
                zero_since = row.simulation_time_s
            count += 1
            elapsed = row.simulation_time_s-zero_since
            if elapsed >= task.motor_off_window_sim_s-task.retained_window_sim_s-1e-9:
                quiet = (row.fastener_linear_speed_m_s <= limits.rest_linear_speed_m_s
                    and row.tool_linear_speed_m_s <= limits.rest_linear_speed_m_s
                    and max(row.fastener_angular_speed_rad_s, row.tool_angular_speed_rad_s,
                        abs(row.spindle_speed_rad_s), *(abs(v) for v in row.joint_velocity_rad_s))
                        <= limits.rest_angular_speed_rad_s)
                if not quiet or not retained_seat(row, task):
                    raise FasteningFault("seating support/rest was lost after the settling interval")
            if elapsed >= task.motor_off_window_sim_s-1e-9:
                return {"status": "confirmed", "samples": count, "last_step": row.step,
                    "generation": row.generation, "zero_spindle_effort": True,
                    "motor_off_window_sim_s": elapsed, "retained_window_sim_s": task.retained_window_sim_s}
    raise FasteningFault("wall budget expired before observed seating retention")
