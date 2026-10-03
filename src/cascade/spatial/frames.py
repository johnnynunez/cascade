"""Source-bound SE(3) transforms at capture time; no implicit frame aliases."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
import hashlib
import json
import threading

import numpy as np

from ..robotics.contracts import identifier
from ..sensing.models import digest, number, vector


def sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SpatialStamp:
    map_id: str
    epoch: str
    clock_id: str
    time_s: float
    source_id: str
    source_sha256: str
    calibration_id: str
    measurement_kind: str

    def __post_init__(self):
        for key in ("map_id", "epoch", "clock_id", "source_id", "calibration_id"):
            identifier(getattr(self, key), key)
        object.__setattr__(self, "time_s", number(self.time_s, "capture time", minimum=0))
        if digest(self.source_sha256) is None:
            raise ValueError("a source SHA256 is required")
        if self.measurement_kind not in {"synthetic", "physics", "measured", "estimated"}:
            raise ValueError("unknown spatial measurement kind")

    @property
    def context(self):
        return self.map_id, self.epoch, self.clock_id

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransformSample:
    """T_parent_child maps child points into parent, meters and wxyz rotation.

    Static samples are calibrated extrinsics within one epoch, not inferred
    world poses. Dynamic samples use bounded zero-order hold; no extrapolation
    beyond max_age_s and no silent interpolation across localization jumps.
    """
    parent: str
    child: str
    translation_m: tuple
    rotation_wxyz: tuple
    stamp: SpatialStamp
    static: bool = False
    position_error_m: float | None = 0.
    angular_error_rad: float | None = 0.

    def __post_init__(self):
        identifier(self.parent, "parent frame"); identifier(self.child, "child frame")
        if self.parent == self.child or not isinstance(self.stamp, SpatialStamp):
            raise ValueError("distinct frames and typed stamp required")
        if type(self.static) is not bool:
            raise ValueError("static must be boolean")
        object.__setattr__(self, "translation_m", vector(self.translation_m, 3, "translation_m"))
        q = vector(self.rotation_wxyz, 4, "rotation_wxyz")
        if abs(sum(v*v for v in q) - 1.) > 1e-6:
            raise ValueError("rotation must be a unit quaternion")
        object.__setattr__(self, "rotation_wxyz", q)
        for key in ("position_error_m", "angular_error_rad"):
            if getattr(self, key) is not None:
                object.__setattr__(self, key, number(getattr(self, key), key, minimum=0))

    @property
    def matrix(self):
        w, x, y, z = self.rotation_wxyz
        value = np.eye(4)
        value[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
        value[:3, 3] = self.translation_m
        return value

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransformResult:
    target: str
    source: str
    matrix: tuple
    samples: tuple[TransformSample, ...]

    def point(self, value):
        p = vector(value, 3, "point_m")
        return tuple(float(v) for v in np.asarray(self.matrix)[:3] @ (*p, 1.))

    def as_dict(self):
        return {"target": self.target, "source": self.source, "matrix": self.matrix,
                "samples": [v.as_dict() for v in self.samples],
                "physical_admission": False}


class FrameTree:
    """A bounded, single-parent measured frame tree scoped to a map epoch."""

    def __init__(self, map_id, epoch, clock_id, root, *, history=128, max_frames=256):
        self.context = tuple(identifier(v) for v in (map_id, epoch, clock_id))
        self.root = identifier(root, "root frame")
        if type(history) is not int or not 1 <= history <= 4096:
            raise ValueError("history must be in 1..4096")
        if type(max_frames) is not int or not 2 <= max_frames <= 4096:
            raise ValueError("max_frames must be in 2..4096")
        self.history, self.max_frames = history, max_frames
        self._samples = {}
        self._lock = threading.RLock()

    def add(self, sample):
        if not isinstance(sample, TransformSample) or sample.stamp.context != self.context:
            raise ValueError("transform context mismatch")
        with self._lock:
            if sample.child == self.root:
                raise ValueError("root cannot have a parent")
            known = {self.root, *self._samples}
            if sample.parent not in known:
                raise ValueError("declare parent before child")
            existing = self._samples.get(sample.child)
            if existing:
                old = existing[-1]
                if (old.parent != sample.parent or old.static != sample.static or
                        old.stamp.calibration_id != sample.stamp.calibration_id or
                        old.stamp.source_id != sample.stamp.source_id or
                        old.stamp.measurement_kind != sample.stamp.measurement_kind):
                    raise ValueError("frame topology, calibration or provider changed; start a new epoch")
                if old.static or sample.stamp.time_s <= old.stamp.time_s:
                    raise ValueError("duplicate/static replacement or out-of-order transform")
            elif len(known) >= self.max_frames:
                raise ValueError("frame capacity exceeded")
            self._samples.setdefault(sample.child, deque(maxlen=self.history)).append(sample)

    def lookup(self, target, source, *, time_s, epoch, clock_id, max_age_s=.2):
        target, source = identifier(target), identifier(source)
        when = number(time_s, "lookup time", minimum=0)
        age = number(max_age_s, "max_age_s", minimum=0)
        if (epoch, clock_id) != self.context[1:]:
            raise ValueError("transform epoch or clock mismatch")
        with self._lock:
            if target not in {self.root, *self._samples} or source not in {self.root, *self._samples}:
                raise ValueError("unknown transform frame")

            def ancestors(frame):
                result = [frame]
                while frame != self.root:
                    frame = self._samples[frame][-1].parent
                    result.append(frame)
                return result
            target_chain, source_chain = ancestors(target), ancestors(source)
            common = next(v for v in source_chain if v in set(target_chain))

            def chain(frame):
                edges = []
                while frame != common:
                    candidates = self._samples[frame]
                    selected = next((v for v in reversed(candidates) if v.stamp.time_s <= when), None)
                    if selected is None or (not selected.static and when - selected.stamp.time_s > age):
                        raise ValueError("missing or stale transform at capture time")
                    edges.append(selected)
                    frame = selected.parent
                return edges

            a, b = chain(target), chain(source)
            ma, mb = np.eye(4), np.eye(4)
            for edge in a:
                ma = edge.matrix @ ma
            for edge in b:
                mb = edge.matrix @ mb
            result = np.linalg.inv(ma) @ mb
            return TransformResult(target, source, tuple(tuple(float(v) for v in row) for row in result),
                                   tuple((*a, *b)))
