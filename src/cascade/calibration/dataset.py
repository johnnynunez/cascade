"""Schema-v1 hand-eye calibration record: one JSON file per camera.

Ported from WRC ``calibration/calib_dataset.py`` (CalibrationSample /
CalibrationResult / save_calibration / load_calibration), reconciled with
cascade's existing extrinsic record (``perception/calibration.py``):

* The quality metrics travel WITH the matrix and the loader refuses a record
  whose gate fails (``load_hand_eye`` returns None, exactly like
  ``load_extrinsic``) -- and the gate is RECOMPUTED from the stored metrics
  with today's thresholds, so neither a hand-edited ``"acceptable": true``
  nor a file written before a threshold was tightened gets through.
* Matrices are stored under the names that say what they are
  (``T_cam2base`` + ``T_marker2gripper`` for eye-to-hand, ``T_cam2gripper``
  + ``T_marker2base`` for eye-in-hand). WRC stored eye-in-hand results under
  ``T_cam2base``/``T_marker2gripper`` and its runtime loader then read the
  MARKER transform as the camera transform.
* Intrinsics live inline in the JSON (WRC wrote them to a ``.npz`` sidecar,
  one more file to lose); the samples are kept so a fit can be audited or
  re-solved later, with the measured joints they came from.
* Writes are atomic (temp file + ``os.replace``), as for the belief store.
* One file format, several METHODS: the marker joint solve
  (``joint_se3_lm_huber``) and markerless depth ICP
  (``depth_icp_markerless``, eye-to-hand only). The gate is per method
  (``assess_record``) and an unknown method never passes: a record whose
  quality cannot be judged is treated as one that failed. A markerless
  record has no marker, so its marker spec and marker transform are absent
  rather than filled with something that looks like one.

Units: metres and radians; matrices are 4x4 ``T_a2b`` (points in a -> b).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .frames import is_se3
from .handeye import EYE_IN_HAND, EYE_TO_HAND, MODES, HandEyeSample, assess_hand_eye

KIND = "cascade.hand_eye"
SCHEMA_VERSION = 1
METHOD = "joint_se3_lm_huber"
FRAME_CONVENTION = "cascade-base-v1"
FRAME_CONVENTION_NOTE = (
    "robot BASE frame (+x away from the robot, +y left, metres); TCP = FK of the "
    "MEASURED local joints at the arm profile's ee_frame (joint_signs baked in); "
    "camera frame = OpenCV (x right, y down, z forward); marker frame = OpenCV "
    "ArUco (origin at the marker centre, x right, y up, z out of the printed face)"
)

_KEYS = {
    EYE_TO_HAND: ("T_cam2base", "T_marker2gripper"),
    EYE_IN_HAND: ("T_cam2gripper", "T_marker2base"),
}


@dataclass(frozen=True)
class MarkerSpec:
    """The physical target. Measure the PRINTED size: printers scale."""

    dictionary: str = "4x4_50"
    marker_id: int = 0
    size_m: float = 0.10

    def to_json(self) -> dict:
        return {"dictionary": self.dictionary, "id": int(self.marker_id),
                "size_m": float(self.size_m)}

    @classmethod
    def from_json(cls, d: dict) -> "MarkerSpec":
        return cls(str(d["dictionary"]), int(d["id"]), float(d["size_m"]))


@dataclass(frozen=True)
class DepthPose:
    """A markerless sample as recorded: the measured joints, their FK and
    how well that pose's depth explained the fitted arm (the depth image
    itself is saved next to the trace, ``depth_file``, not in the JSON)."""

    label: str
    q: tuple | None
    T_gripper2base: np.ndarray
    n_visible: int = 0
    n_inliers: int = 0
    rmse_m: float = float("nan")
    offset_m: float = float("nan")
    offset_deg: float = float("nan")
    depth_file: str = ""


def assess_record(method: str, metrics: dict) -> list[str]:
    """The gate for ``method``; an unknown method is a reason in itself."""
    if method == METHOD:
        return assess_hand_eye(metrics)
    from .markerless import METHOD as MARKERLESS
    from .markerless import assess_markerless

    if method == MARKERLESS:
        return assess_markerless(metrics)
    return [f"unknown calibration method {method!r}: no quality gate to judge it by"]


@dataclass(frozen=True)
class HandEyeRecord:
    mode: str
    T_hand_eye: np.ndarray
    T_marker: np.ndarray | None
    metrics: dict
    marker: MarkerSpec | None
    samples: tuple = ()
    outlier_indices: tuple = ()
    camera: str = ""
    camera_serial: str = ""
    arm: str = ""
    ee_frame: str = ""
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    image_size: tuple | None = None   # (w, h)
    method: str = METHOD
    created_at: str = ""
    note: str = ""
    stored_acceptable: bool | None = field(default=None, compare=False)

    # ── named views (mode-checked, like HandEyeFit) ──────────────────────

    def _need(self, mode, name):
        if self.mode != mode:
            raise AttributeError(f"{name} is only defined for {mode} records (this is {self.mode})")

    @property
    def T_cam2base(self):
        self._need(EYE_TO_HAND, "T_cam2base")
        return self.T_hand_eye

    @property
    def T_marker2gripper(self):
        self._need(EYE_TO_HAND, "T_marker2gripper")
        return self.T_marker

    @property
    def T_cam2gripper(self):
        self._need(EYE_IN_HAND, "T_cam2gripper")
        return self.T_hand_eye

    @property
    def T_marker2base(self):
        self._need(EYE_IN_HAND, "T_marker2base")
        return self.T_marker

    # ── the gate ─────────────────────────────────────────────────────────

    @property
    def markerless(self) -> bool:
        from .markerless import METHOD as MARKERLESS

        return self.method == MARKERLESS

    @property
    def rejection_reasons(self) -> list[str]:
        reasons = assess_record(self.method, self.metrics)
        if self.stored_acceptable is False and not reasons:
            reasons = ["record was saved as rejected"]
        return reasons

    @property
    def acceptable(self) -> bool:
        return not self.rejection_reasons

    @property
    def inlier_flags(self) -> list[bool]:
        bad = set(self.outlier_indices)
        return [i not in bad for i in range(len(self.samples))]

    def with_outliers(self, outliers) -> "HandEyeRecord":
        return replace(self, outlier_indices=tuple(int(i) for i in outliers))

    def summary(self) -> str:
        m = self.metrics
        verdict = "OK" if self.acceptable else "REJECTED"
        if self.markerless:
            from .markerless import markerless_summary

            return markerless_summary(m, self.acceptable, self.camera)
        return (f"[{verdict}] {self.mode} camera={self.camera or '?'}: "
                f"{int(m.get('n_inliers', 0))}/{int(m.get('n_samples', 0))} inliers, "
                f"translation rmse {float(m.get('translation_rmse_m', math.nan)) * 1000:.1f} mm, "
                f"rotation rmse {float(m.get('rotation_rmse_deg', math.nan)):.2f} deg, "
                f"rotation spread {float(m.get('rotation_spread_deg', math.nan)):.1f} deg")

    # ── JSON ─────────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        k_he, k_m = _KEYS[self.mode]
        reasons = self.rejection_reasons
        flags = self.inlier_flags
        return {
            "kind": KIND,
            "schema_version": SCHEMA_VERSION,
            "frame_convention": FRAME_CONVENTION,
            "frame_convention_note": FRAME_CONVENTION_NOTE,
            "mode": self.mode,
            "camera": self.camera,
            "camera_serial": self.camera_serial,
            "arm": self.arm,
            "ee_frame": self.ee_frame,
            k_he: np.asarray(self.T_hand_eye, dtype=float).tolist(),
            **({} if self.T_marker is None
               else {k_m: np.asarray(self.T_marker, dtype=float).tolist()}),
            "marker": None if self.marker is None else self.marker.to_json(),
            "intrinsics": None if self.K is None else {
                "K": np.asarray(self.K, dtype=float).tolist(),
                "D": [] if self.D is None else np.asarray(self.D, dtype=float).ravel().tolist(),
                "image_size": None if self.image_size is None else list(self.image_size),
            },
            "method": self.method,
            "metrics": {k: _json_number(v) for k, v in self.metrics.items()},
            "acceptable": not reasons,
            "rejection_reasons": reasons,
            "outlier_indices": [int(i) for i in self.outlier_indices],
            "samples": [_sample_json(s, flags[i]) for i, s in enumerate(self.samples)],
            "created_at": self.created_at,
            # Wall clock: a persisted record, never an in-process timestamp.
            "saved_at": time.time(),
            "note": self.note,
        }

    @classmethod
    def from_json(cls, d: dict) -> "HandEyeRecord":
        if not isinstance(d, dict) or d.get("kind") != KIND:
            raise ValueError(f"not a cascade hand-eye record (kind={d.get('kind') if isinstance(d, dict) else None!r})")
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported hand-eye schema_version {d.get('schema_version')!r} "
                             f"(this loader reads {SCHEMA_VERSION}); re-run the calibration")
        mode = d.get("mode")
        if mode not in MODES:
            raise ValueError(f"bad hand-eye mode {mode!r}")
        from .markerless import METHOD as MARKERLESS

        method = str(d.get("method", METHOD))
        if method not in (METHOD, MARKERLESS):
            raise ValueError(f"unknown calibration method {method!r} (this loader reads "
                             f"{METHOD!r} and {MARKERLESS!r}); there is no gate to judge it by")
        markerless = method == MARKERLESS
        if markerless and mode != EYE_TO_HAND:
            raise ValueError(f"a {MARKERLESS} record must be eye_to_hand (the arm is the target; "
                             f"a wrist camera does not see it), got {mode!r}")
        k_he, k_m = _KEYS[mode]
        T_he = _se3(d.get(k_he), k_he)
        T_m = None if markerless and d.get(k_m) is None else _se3(d.get(k_m), k_m)
        intr = d.get("intrinsics") or None
        K = D = size = None
        if intr:
            K = np.asarray(intr["K"], dtype=float).reshape(3, 3)
            D = np.asarray(intr.get("D") or [], dtype=float)
            size = tuple(int(v) for v in intr["image_size"]) if intr.get("image_size") else None
        parse = _depth_pose_from_json if markerless else _sample_from_json
        samples = tuple(parse(s) for s in d.get("samples", []))
        marker = d.get("marker") if markerless else d["marker"]
        return cls(
            mode=mode, T_hand_eye=T_he, T_marker=T_m,
            metrics=dict(d.get("metrics") or {}),
            marker=None if marker is None else MarkerSpec.from_json(marker),
            samples=samples,
            outlier_indices=tuple(int(i) for i in d.get("outlier_indices", [])),
            camera=str(d.get("camera", "")), camera_serial=str(d.get("camera_serial", "")),
            arm=str(d.get("arm", "")), ee_frame=str(d.get("ee_frame", "")),
            K=K, D=D, image_size=size, method=method,
            created_at=str(d.get("created_at", "")), note=str(d.get("note", "")),
            stored_acceptable=bool(d.get("acceptable", False)),
        )


def _se3(value, name) -> np.ndarray:
    try:
        T = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} is not a 4x4 matrix") from e
    if not is_se3(T):
        raise ValueError(f"{name} is not a finite rigid transform")
    return T


def _json_number(v):
    """Metric values as plain JSON (non-finite -> null, which the gates
    read as missing, i.e. refused)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.floating, np.integer)):
        v = float(v)
        return v if math.isfinite(v) else None
    return v


def _sample_json(s, inlier: bool) -> dict:
    if isinstance(s, DepthPose):
        return {
            "label": s.label,
            "q": None if s.q is None else [float(v) for v in s.q],
            "T_gripper2base": np.asarray(s.T_gripper2base, dtype=float).tolist(),
            "n_visible": int(s.n_visible),
            "n_inliers": int(s.n_inliers),
            "rmse_m": _json_number(s.rmse_m),
            "offset_m": _json_number(s.offset_m),
            "offset_deg": _json_number(s.offset_deg),
            "depth_file": s.depth_file,
            "inlier": bool(inlier),
        }
    return {
        "label": s.label,
        "q": None if s.q is None else [float(v) for v in s.q],
        "T_gripper2base": np.asarray(s.T_gripper2base, dtype=float).tolist(),
        "T_marker2cam": np.asarray(s.T_marker2cam, dtype=float).tolist(),
        "reprojection_px": float(s.reprojection_px),
        "inlier": bool(inlier),
    }


def _sample_from_json(d: dict) -> HandEyeSample:
    return HandEyeSample(
        T_gripper2base=_se3(d["T_gripper2base"], "sample T_gripper2base"),
        T_marker2cam=_se3(d["T_marker2cam"], "sample T_marker2cam"),
        label=str(d.get("label", "")),
        q=None if d.get("q") is None else tuple(float(v) for v in d["q"]),
        reprojection_px=float(d.get("reprojection_px", 0.0)),
    )


def _depth_pose_from_json(d: dict) -> DepthPose:
    def num(key):
        v = d.get(key)
        return float("nan") if v is None else float(v)
    return DepthPose(
        label=str(d.get("label", "")),
        q=None if d.get("q") is None else tuple(float(v) for v in d["q"]),
        T_gripper2base=_se3(d["T_gripper2base"], "sample T_gripper2base"),
        n_visible=int(d.get("n_visible", 0)), n_inliers=int(d.get("n_inliers", 0)),
        rmse_m=num("rmse_m"), offset_m=num("offset_m"), offset_deg=num("offset_deg"),
        depth_file=str(d.get("depth_file", "")),
    )


def record_from_fit(fit, samples, *, marker: MarkerSpec, camera: str = "",
                    camera_serial: str = "", arm: str = "", ee_frame: str = "",
                    K=None, D=None, image_size=None, note: str = "") -> HandEyeRecord:
    """Wrap a ``HandEyeFit`` and the samples it was solved from."""
    return HandEyeRecord(
        mode=fit.mode, T_hand_eye=np.asarray(fit.T_hand_eye, dtype=float),
        T_marker=np.asarray(fit.T_marker, dtype=float), metrics=dict(fit.metrics),
        marker=marker, samples=tuple(samples), outlier_indices=tuple(fit.outliers),
        camera=camera, camera_serial=str(camera_serial or ""), arm=arm, ee_frame=ee_frame,
        K=None if K is None else np.asarray(K, dtype=float),
        D=None if D is None else np.asarray(D, dtype=float),
        image_size=None if image_size is None else tuple(int(v) for v in image_size),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), note=note,
    )


def save_hand_eye(path, record: HandEyeRecord) -> Path:
    """Write the record atomically; returns the path written."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record.to_json(), f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_hand_eye(path) -> HandEyeRecord:
    """Parse a record WITHOUT the acceptance gate (inspection, --verify).

    Raises FileNotFoundError when absent, ValueError when malformed.
    """
    path = Path(path).expanduser()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e
    return HandEyeRecord.from_json(data)


def load_hand_eye(path, *, trust_unacceptable: bool = False) -> HandEyeRecord | None:
    """The record, or None when there is nothing to trust.

    None for a missing file or a rejected record (unless explicitly
    overridden); ValueError for a malformed one. "No extrinsics" degrades
    visibly (no 3D fusion); "wrong extrinsics" degrades invisibly.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    record = read_hand_eye(path)
    if not record.acceptable and not trust_unacceptable:
        return None
    return record
