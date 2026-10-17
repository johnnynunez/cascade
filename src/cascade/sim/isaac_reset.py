"""Validate measured Isaac reset replies; never synthesize missing evidence.

The bridge runs in its own Isaac environment. Its reset helper must report
physics positions for EVERY returned prop, in robot-base metres, using the
existing settle tolerance (2 cm). An old stdout-only acknowledgement is not
a successful reset, even when its transport-level `ok` is true.

`seeded_reset` is the one way a harness sets a reset seed on a bridge.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

from .bridge_client import BridgeClient, BridgeError
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


@contextmanager
def seeded_reset(port, seed, *, host="127.0.0.1", timeout_s=10.0):
    """Hold a bridge's reset seed for one block, then always clear it.

    Harnesses (acceptance A11, verifier B5/B6) set seeds only through this:
    every `reset_props` inside the block places the props on the seed's
    layout (bridge op `set_reset_seed`), and a `finally` sends
    `set_reset_seed null`, also when the block raises or the bridge refuses
    the seed, so no seed leaks into a later run on that bridge. `port` has
    no default: the live demo bridge listens on 8611, a harness names its
    own bridge. Yields the open BridgeClient (closed after the block).

    The clear always runs on a fresh connection (a reply the block left
    unread, e.g. after Ctrl-C, can never pass for the clear's), and a ping
    after it must show no `reset_seed` (a set that landed late re-arms it).
    Once the clear is confirmed, the block's own exception propagates
    unchanged, KeyboardInterrupt included.

    Raises ValueError for a bad port or seed (before connecting), and
    BridgeError when the bridge refuses the seed or cannot confirm the clear.
    """
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"seeded_reset needs the bridge's TCP port (an int in 1..65535), got {port!r}")
    if validate_reset_seed(seed) is None:
        raise ValueError("seeded_reset needs an integer reset seed; null is what it restores afterwards")
    client = BridgeClient(host, port, timeout_s=timeout_s)
    client.connect()
    try:
        try:
            reply = client.request({"op": "set_reset_seed", "seed": seed})
            if reply.get("seed") != seed:
                raise BridgeError(f"bridge did not confirm reset seed {seed}: {reply!r}")
            yield client
        finally:
            _clear_reset_seed(client, port)
    finally:
        client.close()


def _clear_reset_seed(client: BridgeClient, port: int) -> None:
    """`set_reset_seed null` on a fresh connection, then a ping that proves it held."""
    client.close()  # drop the block's socket, and any reply still unread on it
    try:
        client.connect()
        cleared = client.request({"op": "set_reset_seed", "seed": None})
    except BridgeError as exc:
        raise BridgeError(f"could not clear the reset seed on port {port}: {exc}") from exc
    if "seed" not in cleared or cleared["seed"] is not None:
        raise BridgeError(f"bridge did not confirm the reset seed clear on port {port}: {cleared!r}")
    try:
        pinged = client.request({"op": "ping"})
    except BridgeError as exc:
        raise BridgeError(f"could not check the reset seed clear on port {port}: {exc}") from exc
    if "reset_seed" in pinged:
        raise BridgeError(
            f"reset seed {pinged['reset_seed']!r} is armed again on port {port} after its clear "
            "(a set_reset_seed that landed late); clear it with set_reset_seed null")
