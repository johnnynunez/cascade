"""Read-only sensor domain for the composed robot runtime."""
from __future__ import annotations

from ..robotics.contracts import ResourceDescriptor, identifier
from .hub import SensorHub
from .models import ImuPayload, MeasurementMetadata, ProprioceptionPayload
from .providers import MobileRgbSensorProvider, MobileStateSensorProvider, SyntheticSensorProvider


class SensorDomain:
    motion_skills = frozenset()

    def __init__(self, domain_id, hub):
        self.domain_id = identifier(domain_id, "sensor domain")
        if not isinstance(hub, SensorHub):
            raise ValueError("SensorHub required")
        self.hub = hub
        hub.seal()
        self.resources = tuple(ResourceDescriptor(
            resource_id=f"{domain_id}/{descriptor.sensor_id}", kind="sensor",
            robot_id=descriptor.robot_id, capabilities=(descriptor.modality,),
            synthetic=descriptor.measurement_kind == "synthetic",
            admission="software_only" if descriptor.measurement_kind == "synthetic" else "unvalidated",
            metadata=descriptor.as_dict()) for descriptor in hub.descriptors)
        self.tool_specs = [
            {"name": "list_sensors", "description": "List declared passive sensors and provenance; no connections.",
             "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "read_sensor", "description": "Read a fresh immutable capture without stepping or actuating.",
             "parameters": {"type": "object", "properties": {"sensor_id": {"type": "string"}},
                            "required": ["sensor_id"], "additionalProperties": False}},
        ]

    def execute(self, local_name, args):
        try:
            if not isinstance(args, dict):
                raise ValueError("sensor arguments must be an object")
            if local_name == "list_sensors" and not args:
                return {"ok": True, "sensors": [descriptor.as_dict() for descriptor in self.hub.descriptors]}
            if local_name == "read_sensor" and set(args) == {"sensor_id"}:
                identifier(args["sensor_id"], "sensor_id")
                return {"ok": True, "observation": self.hub.read(args["sensor_id"]).as_dict()}
            raise ValueError("unknown sensor tool or invalid arguments")
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}

    def stop(self):
        return {"ok": True, "actuation": False}

    def reset_stop(self):
        # A stop reset must never reset observation epochs or replay watermarks.
        return {"ok": True, "actuation": False}

    def close(self):
        return self.hub.close()


def build_sensor_domain(domain_id, profile, *, providers=None):
    """Build declared passive providers; optional injected providers are prebuilt.

    Profiles never reference another domain's actuator or lazy runtime object.
    Unknown keys are errors, so a requested calibration/modality cannot be lost.
    """
    if not isinstance(profile, dict) or profile.get("kind") != "sensors":
        raise ValueError("explicit sensors domain profile required")
    allowed = {"kind", "robot_id", "providers", "max_age_s", "read_timeout_s"}
    if set(profile) - allowed:
        raise ValueError("unknown sensor domain settings")
    robot_id = identifier(profile["robot_id"], "robot_id")
    declarations = profile.get("providers", [])
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 16:
        raise ValueError("declare 1..16 sensor providers")
    injected = {} if providers is None else dict(providers)
    hub = SensorHub()
    seen = set()
    for entry in declarations:
        if not isinstance(entry, dict):
            raise ValueError("sensor provider declaration must be an object")
        name = identifier(entry.get("id"), "sensor_id")
        if name in seen:
            raise ValueError("duplicate sensor provider")
        seen.add(name)
        kind = entry.get("kind")
        common = {"id", "kind", "max_age_s", "read_timeout_s", "calibration_id"}
        options = {"max_age_s": entry.get("max_age_s", profile.get("max_age_s", .5)),
                   "read_timeout_s": entry.get("read_timeout_s", profile.get("read_timeout_s", .25))}
        if kind == "injected":
            if set(entry) != {"id", "kind"} or name not in injected:
                raise ValueError("injected sensor requires an exact prebuilt provider")
            provider = injected.pop(name)
        elif kind == "synthetic":
            if set(entry) - (common | {"modality", "frame_id", "values", "period_s", "saturated"}):
                raise ValueError("unknown synthetic sensor setting")
            metadata = MeasurementMetadata(entry["frame_id"], entry.get("calibration_id"), entry.get("saturated"))
            classes = {"imu": ImuPayload, "proprioception": ProprioceptionPayload}
            if entry.get("modality") not in classes or not isinstance(entry.get("values"), dict):
                raise ValueError("synthetic sensor requires explicit supported modality and values")
            payload = classes[entry["modality"]](metadata=metadata, **entry["values"])
            provider = SyntheticSensorProvider(name, robot_id, payload,
                                               period_s=entry.get("period_s", .02), **options)
        elif kind in ("mobile_state", "mobile_rgb"):
            extra = {"profile", "modality"} if kind == "mobile_state" else {"profile", "camera"}
            if set(entry) - (common | extra):
                raise ValueError("unknown mobile sensor setting")
            native = entry["profile"]
            if native["robot_id"] != robot_id:
                raise ValueError("mobile sensor robot identity mismatch")
            options["calibration_id"] = entry.get("calibration_id")
            if kind == "mobile_state":
                provider = MobileStateSensorProvider(name, native, entry["modality"], **options)
            else:
                provider = MobileRgbSensorProvider(name, native, entry["camera"], **options)
        else:
            raise ValueError("unknown sensor provider kind")
        if provider.descriptor.sensor_id != name or provider.descriptor.robot_id != robot_id:
            raise ValueError("injected sensor identity mismatch")
        hub.register(provider)
    if injected:
        raise ValueError("unused injected sensor provider")
    return SensorDomain(domain_id, hub)
