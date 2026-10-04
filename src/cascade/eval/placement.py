"""Independent release/support checks over an explicit completed-solve window.

This checks observed release and retained support, not containment, perception,
controller safety, or benchmark success. The native reader remains a trusted
boundary; a digest detects changed records, not fabricated physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .trials import digest_json, require_digest


@dataclass(frozen=True)
class PlacementPolicy:
    model_identity_sha256: str
    epoch: str
    object_geoms: tuple[int, ...]
    support_geoms: tuple[int, ...]
    robot_geoms: tuple[int, ...]
    object_mass_kg: float
    gravity_world_m_s2: tuple[float, float, float]
    solver_dt_s: float
    object_body_id: int
    support_body_id: int
    native_ngeom: int
    rest_s: float = .5
    robot_clearance_m: float = .005
    linear_speed_m_s: float = .01
    angular_speed_rad_s: float = .1
    drift_m: float = .003
    force_fraction_min: float = .8
    force_fraction_max: float = 1.2
    constraint_recipe: str = "uncoupled_free_rigid_objects_v1"

    def __post_init__(self):
        require_digest(self.model_identity_sha256)
        if self.constraint_recipe != "uncoupled_free_rigid_objects_v1":
            raise ValueError("unsupported placement constraint recipe")
        if not isinstance(self.epoch, str) or not self.epoch:
            raise ValueError("placement epoch is required")
        groups = (self.object_geoms, self.support_geoms, self.robot_geoms)
        _integer(self.object_body_id, "object body", 1)
        _integer(self.support_body_id, "support body", 1)
        _integer(self.native_ngeom, "native geometry count", 1)
        if self.object_body_id == self.support_body_id:
            raise ValueError("placement body identities must be disjoint")
        for group in groups:
            if (type(group) is not tuple or not group
                    or any(type(g) is not int or not 0 <= g < self.native_ngeom for g in group)
                    or len(set(group)) != len(group)):
                raise ValueError("placement groups require unique immutable geometry IDs")
        if len(set(sum(groups, ()))) != sum(map(len, groups)):
            raise ValueError("placement geometry groups must be disjoint")
        for name in ("object_mass_kg", "solver_dt_s", "rest_s", "robot_clearance_m",
                     "linear_speed_m_s", "angular_speed_rad_s", "drift_m",
                     "force_fraction_min", "force_fraction_max"):
            _number(getattr(self, name), name, positive=True)
        if self.force_fraction_min >= self.force_fraction_max or self.rest_s < self.solver_dt_s:
            raise ValueError("invalid placement force or time interval")
        gravity = _vector(self.gravity_world_m_s2, 3, "gravity")
        if (type(self.gravity_world_m_s2) is not tuple or not math.isfinite(float(np.linalg.norm(gravity)))
                or np.linalg.norm(gravity) <= 0):
            raise ValueError("nonzero immutable gravity is required")

    @property
    def sha256(self):
        return digest_json(asdict(self))


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"invalid {name}")
    return value


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"invalid {name} shape")
    return np.array([_number(x, name) for x in value], dtype=float)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def seal_placement_row(row):
    """Copy JSON data and bind every channel without keeping SDK array views."""
    import json
    value = json.loads(json.dumps(row, allow_nan=False))
    if "snapshot_sha256" in value:
        raise ValueError("placement row is already sealed")
    value["snapshot_sha256"] = digest_json(value)
    return value


def verify_placement_window(rows, policy, *, first_solver_step, last_solver_step):
    """Require every solve in the caller's original window; never hunt a suffix.

    Geometry bounding spheres provide a conservative separation proof. Contact
    forces act on geom B, and are bound to pre-integration Euler constraint time.
    Missing/inconsistent coverage is unverified; measured violations refute.
    """
    measured = {}
    def verdict(status, reason):
        return {"status": status, "reason": reason, "measured": measured,
                "policy_sha256": policy.sha256,
                "checks": {"released_supported_rest": status},
                "containment": "unverified"}

    try:
        _integer(first_solver_step, "first step", 1)
        _integer(last_solver_step, "last step", 1)
        if last_solver_step <= first_solver_step:
            raise ValueError("placement window must advance")
        if not isinstance(rows, (list, tuple)) or len(rows) != last_solver_step-first_solver_step+1:
            raise ValueError("placement window has missing or extra solves")
        if (last_solver_step-first_solver_step)*policy.solver_dt_s < policy.rest_s-1e-10:
            raise ValueError("placement window is shorter than required rest")
        object_ids, support_ids, robot_ids = map(set, (policy.object_geoms, policy.support_geoms, policy.robot_geoms))
        all_ids = object_ids | support_ids | robot_ids
        up = -np.array(policy.gravity_world_m_s2)
        weight = policy.object_mass_kg*float(np.linalg.norm(up))
        up /= np.linalg.norm(up)
        previous_time = None
        origins = {}
        minimum_clearance, minimum_support, maximum_support = math.inf, math.inf, -math.inf
        violation = None
        for expected, row in enumerate(rows, first_solver_step):
            if not isinstance(row, dict):
                raise ValueError("placement row must be a record")
            require_digest(row["snapshot_sha256"])
            if digest_json({k: v for k, v in row.items() if k != "snapshot_sha256"}) != row["snapshot_sha256"]:
                raise ValueError("placement snapshot changed")
            if (row["model_identity_sha256"] != policy.model_identity_sha256 or row["epoch"] != policy.epoch
                    or row["policy_sha256"] != policy.sha256):
                raise ValueError("placement identity or policy changed")
            if (_integer(row["solver_step"], "solver step", 1) != expected
                    or _integer(row["native_ngeom"], "native geometry count", 1) != policy.native_ngeom
                    or row["phase"] != "euler_constraint_before_integration"
                    or row["coupling_admission"] != policy.constraint_recipe
                    or row["coverage"] != "all_native_contact_candidates"):
                raise ValueError("placement step, phase, or coverage is inconsistent")
            t = _number(row["constraint_time_s"], "constraint time")
            end = _number(row["advanced_time_s"], "advanced time")
            if (t < 0 or not math.isclose(end-t, policy.solver_dt_s, rel_tol=0, abs_tol=1e-9)
                    or (previous_time is not None and not math.isclose(t-previous_time, policy.solver_dt_s, rel_tol=0, abs_tol=1e-9))):
                raise ValueError("placement physical clock is inconsistent")
            previous_time = t
            if row["native_warnings"] != [] or row["external_forces_zero"] is not True:
                raise ValueError("native warnings or applied external forces invalidate placement")
            bodies = row["bodies"]
            if set(bodies) != {"object", "support"}:
                raise ValueError("placement body coverage is incomplete")
            for name, body in bodies.items():
                if _integer(body["body_id"], "body identity", 1) != getattr(policy, name+"_body_id"):
                    raise ValueError("placement body identity changed")
                xyz = _vector(body["position_m"], 3, "body position")
                linear = _vector(body["linear_velocity_m_s"], 3, "body linear velocity")
                angular = _vector(body["angular_velocity_rad_s"], 3, "body angular velocity")
                origins.setdefault(name, xyz)
                if (np.linalg.norm(linear) > policy.linear_speed_m_s
                        or np.linalg.norm(angular) > policy.angular_speed_rad_s
                        or np.linalg.norm(xyz-origins[name]) > policy.drift_m):
                    violation = violation or "object or support did not retain rest"
            geoms = {}
            for geom in row["geometries"]:
                gid = _integer(geom["id"], "geometry ID")
                if gid in geoms or gid >= _integer(row["native_ngeom"], "native geometry count", 1):
                    raise ValueError("duplicate placement geometry")
                geoms[gid] = (_vector(geom["position_m"], 3, "geometry position"),
                              _number(geom["bound_radius_m"], "bounding radius", positive=True))
            if set(geoms) != all_ids:
                raise ValueError("placement geometry coverage is incomplete")
            clearance = min(float(np.linalg.norm(geoms[a][0]-geoms[b][0]))-geoms[a][1]-geoms[b][1]
                            for a in object_ids | support_ids for b in robot_ids)
            _number(clearance, "computed clearance")
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < policy.robot_clearance_m:
                violation = violation or "robot separation was not established"
            contacts = row["contacts"]
            if _integer(row["ncon"], "contact count") != len(contacts):
                raise ValueError("placement contact coverage is incomplete")
            upward_force = 0.
            for index, contact in enumerate(contacts):
                if _integer(contact["index"], "contact index") != index:
                    raise ValueError("placement contact order is incomplete")
                a, b = (_integer(contact[key], "contact geometry") for key in ("geom_a", "geom_b"))
                if a == b or max(a, b) >= _integer(row["native_ngeom"], "native geometry count", 1):
                    raise ValueError("invalid contact pair")
                force = _vector(contact["force_on_b_world_n"], 3, "contact force")
                address = _integer(contact["efc_address"], "constraint address", -1)
                _number(contact["distance_m"], "contact distance")
                frame = np.array([_vector(axis, 3, "contact frame") for axis in contact["frame_rows"]])
                wrench = _vector(contact["wrench_on_b_contact"], 6, "contact wrench")
                _vector(contact["position_m"], 3, "contact position")
                if (type(contact["dimension"]) is not int or contact["dimension"] not in (1, 3, 4, 6)
                        or frame.shape != (3, 3) or not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-7)
                        or not math.isclose(float(np.linalg.det(frame)), 1., rel_tol=0, abs_tol=1e-7)
                        or not np.allclose(frame.T @ wrench[:3], force, rtol=1e-10, atol=1e-10)
                        or wrench[0] < -1e-10
                        or address >= _integer(row["nefc"], "constraint count")):
                    raise ValueError("placement contact frame, force, or constraint is inconsistent")
                if address == -1 and np.any(force != 0):
                    raise ValueError("inactive contact has a nonzero force")
                pair = {a, b}
                if pair & robot_ids and pair & (object_ids | support_ids) and address >= 0:
                    violation = violation or "robot contact persisted after release"
                if a in object_ids and b in support_ids:
                    upward_force += float(-force @ up)
                elif b in object_ids and a in support_ids:
                    upward_force += float(force @ up)
                elif pair & object_ids and not pair <= object_ids and np.linalg.norm(force) > 1e-10:
                    violation = violation or "object load came from outside the declared support"
            fraction = upward_force/weight
            _number(fraction, "computed support fraction")
            minimum_support, maximum_support = min(minimum_support, fraction), max(maximum_support, fraction)
            if not policy.force_fraction_min <= fraction <= policy.force_fraction_max:
                violation = violation or "declared support did not retain the object weight"
        measured.update(first_solver_step=first_solver_step, last_solver_step=last_solver_step,
                        window_sim_s=(len(rows)-1)*policy.solver_dt_s,
                        minimum_robot_clearance_m=minimum_clearance,
                        minimum_support_weight_fraction=minimum_support,
                        maximum_support_weight_fraction=maximum_support)
        return verdict("refuted" if violation else "confirmed", violation or "complete released support/rest window")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return verdict("unverified", str(exc))
