"""Observed shoulder seating for the explicit mounted Factory M20 fixture.

This is a separate, longer task than one tightening turn. Its force threshold
establishes simulated shoulder support, never calibrated bolt preload.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

from .fastening import FasteningFault, FasteningSolve, _index, _number, _vector


SEATING_RECIPE = "factory_m20_shoulder_seat_v1"


@dataclass(frozen=True)
class SeatingLimits:
    max_command_sim_s: float = 45.
    max_command_wall_s: float = 1200.
    rest_timeout_wall_s: float = 90.
    loaded_window_sim_s: float = .5
    motor_off_window_sim_s: float = 2.
    retained_window_sim_s: float = 1.
    minimum_turns: float = 15.
    shoulder_height_m: float = .023
    nut_half_height_m: float = .008
    shoulder_inner_radius_m: float = .011
    shoulder_outer_radius_m: float = .018
    gap_tolerance_m: float = .0003
    minimum_shoulder_force_n: float = .1
    minimum_loaded_effort_nm: float = .045
    maximum_loaded_speed_rad_s: float = .08

    def __post_init__(self):
        for field in fields(self):
            _number(getattr(self, field.name), field.name, positive=True)
        if (self.shoulder_inner_radius_m >= self.shoulder_outer_radius_m
                or self.retained_window_sim_s > self.motor_off_window_sim_s
                or self.loaded_window_sim_s >= self.max_command_sim_s):
            raise ValueError("inconsistent shoulder seating limits")


@dataclass(frozen=True)
class ShoulderContact:
    candidate: int
    point_world_m: tuple[float, float, float]
    normal_a_to_b_world: tuple[float, float, float]
    normal_force_n: float
    force_on_fastener_world_n: tuple[float, float, float]

    def __post_init__(self):
        _index(self.candidate, "seat candidate")
        for name in ("point_world_m", "normal_a_to_b_world", "force_on_fastener_world_n"):
            object.__setattr__(self, name, _vector(getattr(self, name), 3, name))
        if not math.isclose(math.hypot(*self.normal_a_to_b_world), 1., abs_tol=1e-5, rel_tol=0.):
            raise ValueError("seat contact normal is not unit length")
        _number(self.normal_force_n, "seat normal force", positive=True)


@dataclass(frozen=True)
class SeatingSolve(FasteningSolve):
    shoulder_contacts: tuple[ShoulderContact, ...]

    def __post_init__(self):
        super().__post_init__()
        contacts = tuple(self.shoulder_contacts)
        if (any(type(row) is not ShoulderContact for row in contacts)
                or len({row.candidate for row in contacts}) != len(contacts)):
            raise ValueError("invalid or duplicated shoulder witnesses")
        object.__setattr__(self, "shoulder_contacts", contacts)


def seating_solve(solve, raw_contacts, binding):
    """Bind positive seat witnesses to the complete, already-decoded ledger."""
    seat_pair = binding.seat_contact_pair
    fastener = next(iter(set(seat_pair).intersection(binding.thread_contact_pair)))
    contacts = tuple(ShoulderContact(row["candidate"], row["point_world_m"],
        row["normal_a_to_b_world"], row["normal_force_n"],
        tuple(force if row["shape_b"] == fastener else -force for force in row["force_on_b_world_n"]))
        for row in raw_contacts if row["status"] == "solved" and row["normal_force_n"] > 0
        and tuple(sorted((row["shape_a"], row["shape_b"]))) == seat_pair)
    return SeatingSolve(**{field.name: getattr(solve, field.name) for field in fields(FasteningSolve)},
                        shoulder_contacts=contacts)


def check_seating_solve(row, binding):
    if type(row) is not SeatingSolve:
        raise FasteningFault("seating requires same-solve shoulder observations")
    required = {index for index, pair in enumerate(row.contacts)
        if pair.normal_force_n > 0 and tuple(sorted((pair.collider_a, pair.collider_b))) == binding.seat_contact_pair}
    if {contact.candidate for contact in row.shoulder_contacts} != required:
        raise FasteningFault("shoulder witnesses do not cover the complete positive seat ledger")
    for contact in row.shoulder_contacts:
        if contact.normal_force_n != row.contacts[contact.candidate].normal_force_n:
            raise FasteningFault("shoulder force differs from solved contact ledger")
        pair = row.contacts[contact.candidate]
        fastener = next(iter(set(binding.seat_contact_pair).intersection(binding.thread_contact_pair)))
        compression = sum(force*normal for force, normal in zip(
            contact.force_on_fastener_world_n, contact.normal_a_to_b_world, strict=True))
        if pair.collider_b != fastener:
            compression = -compression
        if not math.isclose(compression, contact.normal_force_n, rel_tol=1e-5, abs_tol=1e-7):
            raise FasteningFault("shoulder force vector is inconsistent with the fastener side")


def shoulder_force(row, limits):
    force = 0.
    origin = row.fixture_position_m
    for contact in row.shoulder_contacts:
        x, y, z = (value-base for value, base in zip(contact.point_world_m, origin, strict=True))
        if (limits.shoulder_inner_radius_m <= math.hypot(x, y) <= limits.shoulder_outer_radius_m
                and abs(z-limits.shoulder_height_m) <= limits.gap_tolerance_m
                and abs(contact.normal_a_to_b_world[2]) >= .95
                and contact.force_on_fastener_world_n[2] > 0):
            force += contact.force_on_fastener_world_n[2]
    return force


def retained_seat(row, limits):
    # The axis-relative gap is the fixed fixture experiment's declared measure;
    # complete contact points, threading tilt and collision checks remain separate.
    gap = row.fastener_position_m[2]-row.fixture_position_m[2]-limits.nut_half_height_m-limits.shoulder_height_m
    return abs(gap) < limits.gap_tolerance_m and shoulder_force(row, limits) >= limits.minimum_shoulder_force_n


def loaded_seat(row, limits):
    return (retained_seat(row, limits) and row.tool_contacts > 0
        and row.spindle_effort_nm >= limits.minimum_loaded_effort_nm
        and row.commanded_spindle_effort_nm >= limits.minimum_loaded_effort_nm
        and row.fastener_angular_speed_rad_s < limits.maximum_loaded_speed_rad_s)
