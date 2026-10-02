"""Read-only verification of a threaded fastener from solved body poses.

The input is a sequence of actual fastener and fixture poses, not arm joint
commands. A passing result establishes signed rotation/advance consistent
with the declared pitch and witnessed contacts. It never establishes seating,
preload, tightening torque, or the fidelity of the supplied collision model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ThreadSample:
    epoch: str
    step: int
    time_s: float
    fastener_id: str
    fixture_id: str
    fastener_position_m: Sequence[float]
    fastener_quaternion_xyzw: Sequence[float]
    fixture_position_m: Sequence[float]
    fixture_quaternion_xyzw: Sequence[float]
    thread_contacts: int
    tool_contacts: int


@dataclass(frozen=True)
class ThreadContract:
    """Right-handed thread along fixture +Z; clockwise advances toward -Z."""

    pitch_m: float
    requested_turns: float = 1.0
    direction: str = "tighten"
    pitch_tolerance_m: float = 0.0003
    radial_tolerance_m: float = 0.001
    tilt_tolerance_rad: float = 0.05
    fixture_translation_tolerance_m: float = 0.0001
    fixture_rotation_tolerance_rad: float = 0.002
    turn_tolerance: float = 0.05
    max_angular_speed_rad_s: float = 10.0
    minimum_contact_fraction: float = 0.8


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("invalid physics quaternion")
    norm = float(np.linalg.norm(q))
    if abs(norm - 1.0) > 0.001:
        raise ValueError("physics quaternion is not normalized")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _position(value):
    p = np.asarray(value, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("invalid physics position")
    return p


def _angle(rotation):
    return math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def verify_threading(samples: Sequence[ThreadSample], contract: ThreadContract) -> dict:
    """Return confirmed/refuted/unverified without changing any simulation state.

    The producer must sample faster than the Nyquist bound given by the
    independently enforced angular speed limit. Epoch/identity changes and
    gaps that permit a hidden full turn make the measurement unverified.
    Contacts count solved contact witnesses; broad-phase pairs are insufficient.
    """
    result = {"status": "unverified", "threading_verified": False,
              "seating_verified": False, "measured": {}}
    try:
        for name, value in vars(contract).items():
            if name == "direction":
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if contract.direction not in ("tighten", "loosen"):
            raise ValueError("direction must be tighten or loosen")
        if contract.minimum_contact_fraction > 1 or contract.turn_tolerance >= contract.requested_turns:
            raise ValueError("invalid contact fraction or turn tolerance")
        if contract.pitch_tolerance_m >= contract.pitch_m * (contract.requested_turns - contract.turn_tolerance):
            raise ValueError("pitch tolerance could accept zero axial advancement")
        if len(samples) < 3:
            raise ValueError("at least three physical observations are required")
        first = samples[0]
        identity = (first.epoch, first.fastener_id, first.fixture_id)
        if not all(isinstance(item, str) and item for item in identity):
            raise ValueError("missing physics epoch or body identity")
        if first.fastener_id == first.fixture_id:
            raise ValueError("fastener and fixture must be different bodies")
        reference_p = _position(first.fixture_position_m)
        reference_r = _rotation(first.fixture_quaternion_xyzw)
        times, angles, axial, radial, tilt = [], [], [], [], []
        fixture_shift, fixture_angle, contacts = [], [], []
        previous = None
        for sample in samples:
            if (sample.epoch, sample.fastener_id, sample.fixture_id) != identity:
                raise ValueError("physics epoch or body identity changed")
            if (isinstance(sample.step, bool) or not isinstance(sample.step, int)
                    or sample.step < 0 or not math.isfinite(sample.time_s)):
                raise ValueError("invalid physical clock")
            if previous is not None and (sample.step <= previous.step or sample.time_s <= previous.time_s):
                raise ValueError("physical clock did not advance")
            for count in (sample.thread_contacts, sample.tool_contacts):
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("invalid solved contact count")
            body_r = _rotation(sample.fastener_quaternion_xyzw)
            fixture_r = _rotation(sample.fixture_quaternion_xyzw)
            fixture_p = _position(sample.fixture_position_m)
            relative_p = fixture_r.T @ (_position(sample.fastener_position_m) - fixture_p)
            relative_r = fixture_r.T @ body_r
            times.append(sample.time_s)
            angles.append(math.atan2(relative_r[1, 0], relative_r[0, 0]))
            axial.append(float(relative_p[2]))
            radial.append(float(np.linalg.norm(relative_p[:2])))
            tilt.append(math.acos(float(np.clip(relative_r[2, 2], -1.0, 1.0))))
            fixture_shift.append(float(np.linalg.norm(fixture_p - reference_p)))
            fixture_angle.append(_angle(reference_r.T @ fixture_r))
            contacts.append(sample.thread_contacts > 0 and sample.tool_contacts > 0)
            previous = sample
        dt = np.diff(times)
        bound = contract.max_angular_speed_rad_s * dt
        if np.any(bound >= math.pi):
            raise ValueError("sampling gap permits ambiguous fastener rotation")
        angles = np.unwrap(angles)
        if np.any(np.abs(np.diff(angles)) > bound + 1e-6):
            raise ValueError("observed fastener exceeded declared angular speed bound")
        sign = -1.0 if contract.direction == "tighten" else 1.0
        turns = sign * (angles - angles[0]) / (2 * math.pi)
        advance = sign * (np.asarray(axial) - axial[0])
        pitch_error = np.abs(advance - contract.pitch_m * turns)
        angular_travel = np.abs(np.diff(angles))
        witnessed = np.asarray(contacts[:-1]) | np.asarray(contacts[1:])
        total_travel = float(angular_travel.sum())
        fraction = float(angular_travel[witnessed].sum() / total_travel) if total_travel > 1e-9 else 0.0
        measured = {"turns": float(turns[-1]), "axial_advance_m": float(advance[-1]),
                    "max_pitch_error_m": float(pitch_error.max()),
                    "max_radial_offset_m": max(radial), "max_axis_tilt_rad": max(tilt),
                    "max_fixture_translation_m": max(fixture_shift),
                    "max_fixture_rotation_rad": max(fixture_angle),
                    "contact_rotation_fraction": fraction, "samples": len(samples),
                    "elapsed_s": float(times[-1] - times[0])}
        result["measured"] = measured
        checks = {
            "requested_fastener_rotation": turns[-1] >= contract.requested_turns - contract.turn_tolerance,
            "axial_advance": advance[-1] >= contract.pitch_m * (contract.requested_turns - contract.turn_tolerance) - contract.pitch_tolerance_m,
            "thread_pitch": pitch_error.max() <= contract.pitch_tolerance_m,
            "axis_alignment": max(radial) <= contract.radial_tolerance_m and max(tilt) <= contract.tilt_tolerance_rad,
            "fixed_fixture": max(fixture_shift) <= contract.fixture_translation_tolerance_m and max(fixture_angle) <= contract.fixture_rotation_tolerance_rad,
            "thread_and_tool_contact": fraction >= contract.minimum_contact_fraction,
        }
        result["checks"] = {name: bool(value) for name, value in checks.items()}
        result["threading_verified"] = all(checks.values())
        result["status"] = "confirmed" if result["threading_verified"] else "refuted"
        result["reason"] = ("measured fastener advancement follows thread pitch"
                            if result["threading_verified"] else
                            "; ".join(name for name, passed in checks.items() if not passed))
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        result["reason"] = str(exc)
    return result
