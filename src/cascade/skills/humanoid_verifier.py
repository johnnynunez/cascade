"""Independent postcondition verifier for humanoid loco-manipulation (carry an object to a place).

It judges the MEASURED simulator state the bridge reports after a segment (and after its rest
window), never the policy's output, the agent's claim or the benchmark's own success metric:

- ``object_in_destination``: the object's geometric centre lies over the destination's footprint
  (destination frame, its local bounding box) and between the destination's floor and its rim
  plus the object's half height;
- ``object_at_rest``: object linear speed below ``max_object_speed_m_s``;
- ``object_released``: every hand point is at least ``min_hand_clearance_m`` from the object's
  oriented box (a hand still holding or touching it fails);
- ``robot_upright``: pelvis tilt and height inside the humanoid gate's fall limits.

Verdict: ``confirmed`` when every criterion is measured and passes, ``refuted`` when a measured
criterion fails, ``unverified`` when anything needed is missing, malformed or non-finite.
"""
from __future__ import annotations

import math

from ..safety.humanoid_gate import tilt_deg

_VERIFIER_KEYS = {"max_object_speed_m_s", "min_hand_clearance_m", "z_tolerance_m"}


def validate_verifier_limits(data) -> dict:
    if not isinstance(data, dict) or set(data) != _VERIFIER_KEYS:
        raise ValueError(f"humanoid verifier requires exactly {sorted(_VERIFIER_KEYS)}")
    out = {}
    for key in sorted(_VERIFIER_KEYS):
        value = data[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"verifier {key} must be a positive finite number")
        out[key] = float(value)
    return out


def _q_conj_rotate(q, v):
    """Rotate world vector ``v`` into the frame whose orientation is unit quaternion ``q`` (w,x,y,z)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, -x / n, -y / n, -z / n  # inverse rotation
    # v' = q v q*  (expanded)
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))


def _q_rotate(q, v):
    w, x, y, z = q
    return _q_conj_rotate((w, -x, -y, -z), v)


def _vec(value, n, name):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{name} must have {n} values")
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} is not finite")
    return out


def _box(entry, name):
    pos = _vec(entry.get("pos"), 3, f"{name}.pos")
    quat = _vec(entry.get("quat_wxyz"), 4, f"{name}.quat_wxyz")
    aabb = entry.get("aabb_local")
    if not isinstance(aabb, (list, tuple)) or len(aabb) != 2:
        raise ValueError(f"{name}.aabb_local must be [[min], [max]]")
    lo, hi = _vec(aabb[0], 3, f"{name}.aabb_local"), _vec(aabb[1], 3, f"{name}.aabb_local")
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError(f"{name}.aabb_local is not ordered")
    return pos, quat, lo, hi


def _point_to_box(point, pos, quat, lo, hi):
    """Distance from a world point to an oriented box (0 inside)."""
    local = _q_conj_rotate(quat, tuple(p - c for p, c in zip(point, pos)))
    d = [max(l - v, 0.0, v - h) for v, l, h in zip(local, lo, hi)]
    return math.sqrt(sum(x * x for x in d))


def object_centre(entry, name="object"):
    pos, quat, lo, hi = _box(entry, name)
    mid = tuple((a + b) / 2 for a, b in zip(lo, hi))
    off = _q_rotate(quat, mid)
    return tuple(p + o for p, o in zip(pos, off)), tuple((b - a) / 2 for a, b in zip(lo, hi))


def placement_measurements(state: dict, obj: str, destination: str) -> dict:
    """Geometric relation of ``obj`` to ``destination`` (raises ValueError when unmeasurable)."""
    objects, dests = state.get("objects") or {}, state.get("destinations") or {}
    if obj not in objects:
        raise ValueError(f"no measured pose for object {obj!r}")
    if destination not in dests:
        raise ValueError(f"no measured pose for destination {destination!r}")
    centre, half = object_centre(objects[obj], obj)
    dpos, dquat, dlo, dhi = _box(dests[destination], destination)
    local = _q_conj_rotate(dquat, tuple(c - p for c, p in zip(centre, dpos)))
    # signed xy margin to the footprint edge: >0 inside
    margin = min(local[0] - dlo[0], dhi[0] - local[0], local[1] - dlo[1], dhi[1] - local[1])
    return {"object_centre_world_m": [round(v, 4) for v in centre],
            "object_centre_in_destination_m": [round(v, 4) for v in local],
            "footprint_margin_m": round(margin, 4),
            "destination_floor_z_local_m": round(dlo[2], 4), "destination_rim_z_local_m": round(dhi[2], 4),
            "object_half_height_m": round(half[2], 4)}


def verify_carry(state: dict | None, obj: str, destination: str, *, verifier: dict, gate_limits) -> dict:
    """Verdict on "``obj`` rests inside ``destination``, released, robot upright"."""
    criteria: dict[str, dict] = {}
    try:
        if not isinstance(state, dict):
            raise ValueError("no measured final state")
        m = placement_measurements(state, obj, destination)
        local = m["object_centre_in_destination_m"]
        z_lo = m["destination_floor_z_local_m"] - verifier["z_tolerance_m"]
        z_hi = m["destination_rim_z_local_m"] + m["object_half_height_m"] + verifier["z_tolerance_m"]
        inside = m["footprint_margin_m"] >= 0.0 and z_lo <= local[2] <= z_hi
        criteria["object_in_destination"] = {
            "ok": inside, "value": m["footprint_margin_m"], "limit": 0.0,
            "detail": f"centre {local} m in {destination} frame; z band [{round(z_lo, 3)}, {round(z_hi, 3)}]"}
        speed = math.sqrt(sum(v * v for v in _vec(state["objects"][obj].get("lin_vel"), 3, "lin_vel")))
        criteria["object_at_rest"] = {"ok": speed <= verifier["max_object_speed_m_s"], "value": round(speed, 4),
                                      "limit": verifier["max_object_speed_m_s"]}
        pos, quat, lo, hi = _box(state["objects"][obj], obj)
        hands = state.get("hands")
        if not isinstance(hands, dict) or not hands:
            raise ValueError("no measured hand positions")
        clearance = min(_point_to_box(_vec(p, 3, f"hand {k}"), pos, quat, lo, hi) for k, p in sorted(hands.items()))
        criteria["object_released"] = {"ok": clearance >= verifier["min_hand_clearance_m"],
                                       "value": round(clearance, 4), "limit": verifier["min_hand_clearance_m"]}
        tilt = tilt_deg(_vec(state.get("pelvis_quat_wxyz"), 4, "pelvis_quat_wxyz"))
        height = _vec(state.get("pelvis_pos"), 3, "pelvis_pos")[2] - float(state.get("floor_z", 0.0))
        upright = tilt <= gate_limits.max_pelvis_tilt_deg and height >= gate_limits.min_pelvis_height_m
        criteria["robot_upright"] = {"ok": upright, "value": round(tilt, 2), "limit": gate_limits.max_pelvis_tilt_deg,
                                     "detail": f"pelvis tilt {tilt:.1f} deg, height {height:.3f} m "
                                               f"(min {gate_limits.min_pelvis_height_m} m)"}
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "unverified", "reason": f"unmeasurable: {exc}", "criteria": criteria,
                "channel": "simulator physics state (bridge snapshot)"}
    failed = [k for k, v in criteria.items() if not v["ok"]]
    if failed:
        return {"status": "refuted", "reason": "failed: " + ", ".join(failed), "criteria": criteria,
                "measurements": m, "channel": "simulator physics state (bridge snapshot)"}
    return {"status": "confirmed", "reason": f"{obj} rests in {destination}, released, robot upright",
            "criteria": criteria, "measurements": m, "channel": "simulator physics state (bridge snapshot)"}
