"""Passive recording diagnostics, not a locomotion or balanced-rest verdict."""

from types import SimpleNamespace
import math

from cascade.agent.base_effects import BasePostconditionChecker
from cascade.control.mobile_base import BaseState
from cascade.control.mobile_support import support_contract


def inspect_states(rows, identity, *, epoch, limits):
    contract = support_contract(identity["support_contract"])
    model = identity["model_identity_sha256"]
    dt = identity["recipe"]["native"]["actual_physics_dt"]
    if type(dt) not in (int, float) or not math.isfinite(dt) or dt != 0.005:
        raise ValueError("recorded recipe requires the original5ms step")
    if contract["model_identity_sha256"] != model:
        raise ValueError("recorded support contract differs from model")
    checker = SimpleNamespace(_support_contract=contract)
    alarms, known_empty = [], []
    if len(rows) != 80:
        alarms.append({"reason": "exactly80 recorded post-bootstrap solves required"})
    for index, row in enumerate(rows):
        try:
            state = BaseState.from_dict(row["controller"])
            if (
                state.step != index + 3
                or state.step != row["step"]
                or state.sim_time_s != row["sim_time"]
                or state.epoch != epoch
                or state.model_identity_sha256 != model
                or not math.isclose(
                    state.sim_time_s, state.step * dt, rel_tol=0, abs_tol=1e-7
                )
            ):
                raise ValueError("original bootstrap step/time/epoch/model join failed")
            if (
                row["fallen"] != state.fallen
                or state.fallen
                or state.latched
                or state.controller_status != "ready"
                or state.generation != 0
                or row["permission_generation_at_sample"] != 0
            ):
                raise ValueError("recorded controller/fallen/generation alarm")
            value = state.as_dict()
            if (
                not limits["min_height_m"]
                <= state.position_world[2]
                <= limits["max_height_m"]
                or BasePostconditionChecker._tilt(value) > limits["max_tilt_rad"]
            ):
                raise ValueError("recorded height/tilt outside existing limits")
            verdict, reason = BasePostconditionChecker._support(
                checker, value, require_load=False
            )
            if verdict != "confirmed":
                raise ValueError(reason)
            if not value["support"]["contacts"]:
                known_empty.append(state.step)
        except (ValueError, TypeError, KeyError) as exc:
            alarms.append(
                {"row_index": index, "step": row.get("step"), "reason": str(exc)}
            )
    return {
        "passed": not alarms,
        "alarms": alarms,
        "known_empty_contact_steps": known_empty,
        "bootstrap_solve_count": 2,
        "observed_row_count": len(rows),
        "scope": "Original recorded epoch/steps and fault/contact-channel diagnostics; empty contacts are not measured support.",
        "physical_admission": False,
        "balanced_rest_verified": False,
    }
