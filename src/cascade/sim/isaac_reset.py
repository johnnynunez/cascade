"""Validate measured Isaac reset replies; never synthesize missing evidence.

The bridge runs in its own Isaac environment. Its reset helper must report
physics positions for EVERY returned prop, in robot-base metres, using the
existing settle tolerance (2 cm). An old stdout-only acknowledgement is not
a successful reset, even when its transport-level `ok` is true.
"""
from __future__ import annotations

import math

from .bridge_client import BridgeError
from .scene_seed import validate_reset_seed


def validate_isaac_reset(response: dict) -> dict:
    """Return the unchanged reply, or raise on absent/contradictory evidence."""
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise BridgeError(f"Isaac reset did not succeed: {response!r}")
    names = response.get("props_reset")
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise BridgeError("Isaac reset requires a non-empty list of unique prop names")
    verification = response.get("reset_verification")
    if (not isinstance(verification, dict) or verification.get("channel") != "physics"
            or verification.get("frame") != "robot_base"):
        raise BridgeError("Isaac reset is missing physics read-back in robot-base metres")
    tolerance = verification.get("tolerance_m")
    if (type(tolerance) not in (int, float) or not math.isfinite(tolerance)
            or not 0 < tolerance <= 0.02):
        raise BridgeError("Isaac reset has an invalid spawn tolerance")
    checks = verification.get("props")
    if not isinstance(checks, dict) or set(checks) != set(names):
        raise BridgeError("Isaac reset needs a physics check for every prop")
    for name in names:
        check = checks[name]
        if not isinstance(check, dict):
            raise BridgeError(f"Isaac reset {name}: missing read-back")
        vectors = [check.get("position_m"), check.get("spawn_position_m")]
        for vector in vectors:
            if (not isinstance(vector, list) or len(vector) != 3
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise BridgeError(f"Isaac reset {name}: expected a finite 3D position")
        measured_error = math.dist(*vectors)
        reported_error = check.get("error_m")
        if (not math.isfinite(measured_error) or measured_error > tolerance
                or type(reported_error) not in (int, float) or not math.isfinite(reported_error)
                or reported_error < 0 or reported_error > tolerance
                or not math.isclose(measured_error, reported_error, rel_tol=1e-6, abs_tol=1e-9)
                or check.get("finite") is not True or check.get("within_tolerance") is not True):
            raise BridgeError(f"Isaac reset {name}: read-back does not confirm spawn within tolerance")
    if "seed" in response or "seed_fallbacks" in response:
        _validate_seed_stamp(response, names)
    return response


def _validate_seed_stamp(response: dict, names: list) -> None:
    """A seeded reset (bridge op set_reset_seed) stamps its seed and the props
    that kept their default spawn; an unseeded reply carries neither key."""
    if "seed" not in response:
        raise BridgeError("Isaac reset reports seed_fallbacks without a seed")
    try:
        seed = validate_reset_seed(response["seed"])
    except ValueError as exc:
        raise BridgeError(f"Isaac reset seed is invalid: {exc}") from None
    fallbacks = response.get("seed_fallbacks", [] if seed is None else None)
    if (not isinstance(fallbacks, list)
            or any(not isinstance(n, str) or n not in names for n in fallbacks)
            or len(set(fallbacks)) != len(fallbacks) or (seed is None and fallbacks)):
        raise BridgeError("Isaac reset seed_fallbacks must list unique reset props of a seeded reset")
