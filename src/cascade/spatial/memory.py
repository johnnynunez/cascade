"""Remember observations with their capture frame; retrieval never admits motion."""
from dataclasses import dataclass
import threading

from ..robotics.contracts import identifier
from ..sensing.models import number, vector
from .frames import SpatialStamp, sha256


@dataclass(frozen=True)
class LandmarkObservation:
    observation_id: str
    label: str
    frame_id: str
    point_m: tuple
    stamp: SpatialStamp
    confidence: float

    def __post_init__(self):
        identifier(self.observation_id, "observation_id"); identifier(self.frame_id, "frame_id")
        if not isinstance(self.label, str) or not 1 <= len(self.label) <= 256:
            raise ValueError("bounded label required")
        object.__setattr__(self, "point_m", vector(self.point_m, 3, "point_m"))
        value = number(self.confidence, "confidence", minimum=0)
        if value > 1 or not isinstance(self.stamp, SpatialStamp):
            raise ValueError("confidence or stamp invalid")
        object.__setattr__(self, "confidence", value)


class SpatialMemory:
    def __init__(self, frames, *, capacity=1024):
        if type(capacity) is not int or not 1 <= capacity <= 10000:
            raise ValueError("invalid spatial memory capacity")
        self.frames, self.capacity = frames, capacity
        self._entries = {}
        self._lock = threading.Lock()

    def observe(self, observation, *, max_transform_age_s=.2):
        if not isinstance(observation, LandmarkObservation) or observation.stamp.context != self.frames.context:
            raise ValueError("observation context mismatch")
        s = observation.stamp
        result = self.frames.lookup(self.frames.root, observation.frame_id,
                                   time_s=s.time_s, epoch=s.epoch, clock_id=s.clock_id,
                                   max_age_s=max_transform_age_s)
        # The camera's extrinsic calibration must match the observation; other
        # edges can legitimately carry different odometry calibrations.
        child_edge = next((v for v in result.samples if v.child == observation.frame_id), None)
        if child_edge is not None and child_edge.stamp.calibration_id != s.calibration_id:
            raise ValueError("observation calibration does not match its frame")
        entry = {"observation_id": observation.observation_id, "label": observation.label,
                 "frame_id": observation.frame_id, "point_capture_m": list(observation.point_m),
                 "point_map_m": list(result.point(observation.point_m)), "stamp": s.as_dict(),
                 "confidence": observation.confidence, "transform": result.as_dict(),
                 "use": "search_hint_requires_fresh_observation", "physical_admission": False}
        entry["sha256"] = sha256(entry)
        with self._lock:
            if observation.observation_id in self._entries:
                raise ValueError("observation ID replay")
            if len(self._entries) >= self.capacity:
                raise ValueError("spatial memory full")
            self._entries[observation.observation_id] = entry
        return entry["sha256"]

    def query(self, label, *, epoch, clock_id, time_s, max_age_s=60.):
        if (epoch, clock_id) != self.frames.context[1:]:
            raise ValueError("memory context mismatch")
        now, age = number(time_s, "query time", minimum=0), number(max_age_s, "max_age_s", minimum=0)
        import copy
        with self._lock:
            return [copy.deepcopy(v) for v in self._entries.values() if v["label"] == label
                    and 0 <= now - v["stamp"]["time_s"] <= age]
