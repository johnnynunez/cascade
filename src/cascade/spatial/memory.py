"""Remember observations with their capture frame; retrieval never admits motion."""
from dataclasses import dataclass
import threading
import json

from ..robotics.contracts import identifier, freeze_json, plain_json
from ..sensing.models import number, vector
from .frames import SpatialStamp, sha256


@dataclass(frozen=True)
class LandmarkObservation:
    observation_id: str
    label: str
    frame_id: str
    point_m: tuple
    stamp: SpatialStamp
    confidence: float | None
    provenance: object = None

    def __post_init__(self):
        identifier(self.observation_id, "observation_id"); identifier(self.frame_id, "frame_id")
        if not isinstance(self.label, str) or not 1 <= len(self.label) <= 256:
            raise ValueError("bounded label required")
        object.__setattr__(self, "point_m", vector(self.point_m, 3, "point_m"))
        value = None if self.confidence is None else number(self.confidence, "confidence", minimum=0)
        if (value is not None and value > 1) or not isinstance(self.stamp, SpatialStamp):
            raise ValueError("confidence or stamp invalid")
        object.__setattr__(self, "confidence", value)
        if self.provenance is not None:
            value = freeze_json(self.provenance)
            if not isinstance(plain_json(value), dict) or len(json.dumps(plain_json(value))) > 8192:
                raise ValueError("landmark provenance must be a bounded object")
            object.__setattr__(self, "provenance", value)


class SpatialMemory:
    def __init__(self, frames, *, capacity=1024):
        if type(capacity) is not int or not 1 <= capacity <= 10000:
            raise ValueError("invalid spatial memory capacity")
        self.frames, self.capacity = frames, capacity
        self._entries = {}
        self._lock = threading.Lock()

    def observe(self, observation, *, max_transform_age_s=.2, capture_frames=None):
        if not isinstance(observation, LandmarkObservation) or observation.stamp.context != self.frames.context:
            raise ValueError("observation context mismatch")
        # A retained capture can carry its own measured transform rather than
        # using the mutable latest frame history. The map context/root must be
        # identical; the ordinary calibration and age checks still apply.
        frames = self.frames if capture_frames is None else capture_frames
        from .frames import FrameTree
        if not isinstance(frames, FrameTree) or frames.context != self.frames.context or frames.root != self.frames.root:
            raise ValueError("capture frame context mismatch")
        s = observation.stamp
        result = frames.lookup(frames.root, observation.frame_id, time_s=s.time_s,
                               epoch=s.epoch, clock_id=s.clock_id, max_age_s=max_transform_age_s)
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
        if observation.provenance is not None:
            entry["provenance"] = plain_json(observation.provenance)
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

    def history(self, label, *, epoch, clock_id):
        """Original entries, explicitly historical; no supplied 'now' or freshness."""
        if (epoch, clock_id) != self.frames.context[1:]:
            raise ValueError("memory context mismatch")
        if not isinstance(label, str) or not 1 <= len(label) <= 256:
            raise ValueError("bounded label required")
        import copy
        with self._lock:
            return [copy.deepcopy(v) for v in self._entries.values() if v["label"] == label]
