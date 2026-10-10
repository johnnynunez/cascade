"""Read-only sensor domain for the composed robot runtime."""
from __future__ import annotations

from ..robotics.contracts import ResourceDescriptor, identifier
from .alignment import STATUSES, AlignmentPolicy, absent_channels, align_capture
from .hub import SensorError, SensorHub
from .models import (GeneralizedJointStatePayload, ImuPayload, MeasurementMetadata,
                     ProprioceptionPayload, JointStatePayload)
from .providers import (MobileRgbSensorProvider, MobileRgbdSensorProvider,
                        MobileStateSensorProvider, SyntheticSensorProvider)


#: Opt-in (`alignment:` in the sensors profile, backlog B50).
READ_ALIGNED_SPEC = {
    "name": "read_aligned",
    "description": ("Read one fresh reference capture (e.g. a camera) and pair each other sensor's admitted "
                    "capture nearest its capture time: aligned / stale / missing / uncertain with the measured "
                    "skew. Never interpolated; stale and missing values are withheld; no stepping or actuation."),
    "parameters": {"type": "object", "properties": {
        "reference": {"type": "string"},
        "sensor_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 15}},
        "required": ["reference"], "additionalProperties": False},
}


class SensorDomain:
    motion_skills = frozenset()

    def __init__(self, domain_id, hub, *, alignment=None):
        self.domain_id = identifier(domain_id, "sensor domain")
        if not isinstance(hub, SensorHub):
            raise ValueError("SensorHub required")
        if alignment is not None and not isinstance(alignment, AlignmentPolicy):
            raise ValueError("AlignmentPolicy required")
        self.hub = hub
        self.alignment = alignment
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
        if alignment is not None:
            self.tool_specs.append(dict(READ_ALIGNED_SPEC))

    def execute(self, local_name, args):
        try:
            if not isinstance(args, dict):
                raise ValueError("sensor arguments must be an object")
            if local_name == "list_sensors" and not args:
                return {"ok": True, "sensors": [descriptor.as_dict() for descriptor in self.hub.descriptors]}
            if local_name == "read_sensor" and set(args) == {"sensor_id"}:
                identifier(args["sensor_id"], "sensor_id")
                observation = self.hub.read(args["sensor_id"])
                return {"ok": True, "observation": observation.as_dict(),
                        "capture_sha256": observation.sha256}
            if (local_name == "read_aligned" and self.alignment is not None and "reference" in args
                    and set(args) <= {"reference", "sensor_ids"}):
                return self._read_aligned(args)
            raise ValueError("unknown sensor tool or invalid arguments")
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}

    def _read_aligned(self, args):
        """One FRESH reference capture; each paired sensor read once (a refusal,
        e.g. a replay when its producer has nothing newer, is reported, not
        fatal), then its admitted capture nearest the reference's capture time
        from the hub's bounded history. A failed reference read refuses the
        call: no older capture is substituted."""
        descriptors = {d.sensor_id: d for d in self.hub.descriptors}
        reference_id = identifier(args["reference"], "reference")
        if reference_id not in descriptors:
            raise ValueError("unknown reference sensor")
        ids = args.get("sensor_ids", [s for s in descriptors if s != reference_id])
        if not isinstance(ids, list) or not 1 <= len(ids) <= 15:
            raise ValueError("sensor_ids must list 1..15 sensors")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate sensor_ids")
        for sensor_id in ids:
            if identifier(sensor_id, "sensor_id") not in descriptors:
                raise ValueError("unknown sensor in sensor_ids")
            if sensor_id == reference_id:
                raise ValueError("the reference cannot be paired with itself")
        reference = self.hub.read(reference_id)
        pairings, counts = {}, {status: 0 for status in STATUSES}
        for sensor_id in ids:
            read_error = None
            try:
                self.hub.read(sensor_id)
            except SensorError as exc:
                read_error = f"{type(exc).__name__}: {exc}"[:400]
            pairing = align_capture(reference, self.hub.history(sensor_id), self.alignment,
                                    now=self.hub.now(), max_age_s=descriptors[sensor_id].max_age_s)
            entry = {**pairing.as_dict(), "read_error": read_error}
            candidate = pairing.candidate
            if candidate is not None:
                entry.update(sequence=candidate.sequence, epoch=candidate.epoch,
                             clock_domain=candidate.clock_domain, capture_time_s=candidate.capture_time_s,
                             capture_sha256=candidate.sha256)
            if pairing.sample is not None:
                entry.update(observation=candidate.as_dict(), absent=absent_channels(candidate.payload))
            pairings[sensor_id] = entry
            counts[pairing.status] += 1
        return {"ok": True,
                "reference": {"sensor_id": reference_id, "sequence": reference.sequence, "epoch": reference.epoch,
                              "clock_domain": reference.clock_domain,
                              "capture_time_s": reference.capture_time_s, "capture_sha256": reference.sha256,
                              "observation": reference.as_dict()},
                "policy": self.alignment.as_dict(), "pairings": pairings, "counts": counts}

    def stop(self):
        return {"ok": True, "actuation": False}

    def reset_stop(self):
        # A stop reset must never reset observation epochs or replay watermarks.
        return {"ok": True, "actuation": False}

    def close(self):
        return self.hub.close()


def build_sensor_domain(domain_id, profile, *, providers=None, embodiment=None):
    """Build declared passive providers; optional injected providers are prebuilt.

    Profiles never reference another domain's actuator or lazy runtime object.
    Unknown keys are errors, so a requested calibration/modality cannot be lost.
    """
    if not isinstance(profile, dict) or profile.get("kind") != "sensors":
        raise ValueError("explicit sensors domain profile required")
    allowed = {"kind", "robot_id", "providers", "max_age_s", "read_timeout_s", "alignment"}
    if set(profile) - allowed:
        raise ValueError("unknown sensor domain settings")
    # Opt-in capture-time pairing (B50); parsed before any provider is built.
    alignment = AlignmentPolicy.from_profile(profile["alignment"]) if "alignment" in profile else None
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
            classes = {"imu": ImuPayload, "proprioception": ProprioceptionPayload,
                       "joint_state": JointStatePayload, "generalized_joint_state": GeneralizedJointStatePayload}
            if entry.get("modality") not in classes or not isinstance(entry.get("values"), dict):
                raise ValueError("synthetic sensor requires explicit supported modality and values")
            values = dict(entry["values"])
            if entry["modality"] in {"joint_state", "generalized_joint_state"}:
                if embodiment is None:
                    raise ValueError("joint observation profiles require an embodiment declaration")
                from ..robotics.embodiment import EmbodimentDescriptor
                values.setdefault("embodiment_sha256", EmbodimentDescriptor.from_dict(embodiment).sha256)
            payload = classes[entry["modality"]](metadata=metadata, **values)
            provider = SyntheticSensorProvider(name, robot_id, payload,
                                               period_s=entry.get("period_s", .02), **options)
        elif kind == 'mobile_rgbd':
            if set(entry) - ((common - {'calibration_id'}) | {'profile', 'camera', 'calibration_sha256', 'max_pixels', 'wait_next'}):
                raise ValueError('unknown RGB-D sensor setting; calibration requires a producer-checked SHA')
            native = entry['profile']
            if native['robot_id'] != robot_id:
                raise ValueError('RGB-D sensor robot identity mismatch')
            provider = MobileRgbdSensorProvider(name, native, entry['camera'],
                calibration_sha256=entry['calibration_sha256'], max_pixels=entry.get('max_pixels', 640*480),
                wait_next=entry.get('wait_next', False), **options)
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
        if embodiment is not None:
            from .embodiment import EmbodimentBoundProvider
            provider = EmbodimentBoundProvider(provider, domain_id, embodiment)
            if kind == "synthetic":
                provider.validate_payload(payload)
        hub.register(provider)
    if injected:
        raise ValueError("unused injected sensor provider")
    return SensorDomain(domain_id, hub, alignment=alignment)
