"""Opt-in evidence for one grasp attempt; never a control input.

CASCADE_GRASP_EVIDENCE_DIR enables receipts. Existing observations and commands
are copied in memory; filesystem work happens only after the attempt returns or
raises. No arm/camera read, retry, FK/IK solve, or actuator command is introduced.
Isaac hooks record existing driver feedback and accepted transport commands.
Other backends retain the runtime/selection evidence, without a driver trace.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

_ACTIVE: ContextVar[Attempt | None] = ContextVar("grasp_evidence", default=None)
_MAX_EVENTS = 30000
_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_SOURCE_FILES = (
    "src/cascade/grasping/evidence.py", "src/cascade/grasping/selector.py",
    "src/cascade/grasping/graspgenx_backend.py", "src/cascade/grasping/force.py",
    "src/cascade/skills/runtime.py", "src/cascade/memory/grasp_memory.py",
    "src/cascade/control/isaac_arm.py", "src/cascade/control/arm_base.py",
    "src/cascade/safety/harness.py", "src/cascade/apps/mcp_server.py", "scripts/launch.sh",
    "src/cascade/grasping/observed_scene.py", "src/cascade/control/simulation_motion.py",
    "src/cascade/grasping/planning_budget.py", "src/cascade/control/lazy_arm.py",
    "configs/arms/isaac_kitchen_gpu.yaml",
    "src/cascade/control/motion_profile.py", "assets/grasp_geometry/rebot_rs_fingers.json",
)


def _snapshot(value):
    """Copy observations now: candidates and feedback arrays can be mutated."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _snapshot(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(v) for v in value]
    if hasattr(value, "as_dict"):
        return _snapshot(value.as_dict())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    return value


def _stderr(receipt):
    try:
        print("[grasp-evidence] " + json.dumps(receipt, allow_nan=False), file=sys.stderr)
    except Exception:
        pass  # stderr can itself be closed; never replace a motion exception


class Attempt:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.id = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
        self.phase = "entry"
        self.events = []
        self.arrays = {}
        self.array_bytes = 0
        self.errors = []
        self.dropped_events = 0
        self.started = time.monotonic()
        self.started_unix_ns = time.time_ns()

    def error(self, operation, exc):
        self.errors.append({"operation": operation, "type": type(exc).__name__, "message": str(exc)})

    def record(self, kind, data):
        if len(self.events) >= _MAX_EVENTS:
            self.dropped_events += 1
            return
        self.events.append({"kind": kind, "phase": self.phase,
                            "elapsed_s": time.monotonic() - self.started,
                            "data": _snapshot(data)})

    def array(self, name, value):
        # GPU -> host transfer is explicit and opt-in, just as at GGX's wire
        # boundary. This is a copy of an existing cloud, never a new capture.
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be evidence")
        if self.array_bytes + value.nbytes > _MAX_ARRAY_BYTES:
            raise ValueError("attempt array evidence exceeds 64 MiB")
        key = f"{len(self.arrays):04d}_{name}"
        self.arrays[key] = value.copy()
        self.array_bytes += value.nbytes
        self.record("array", {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)})

    def finish(self):
        """Publish a receipt even if array writing failed; surface failures."""
        document = {"schema": "cascade.grasp-attempt.v1", "attempt_id": self.id,
                    "pid": os.getpid(), "directory": str(self.directory),
                    "started_monotonic_s": self.started, "started_unix_ns": self.started_unix_ns,
                    "finished_unix_ns": time.time_ns(),
                    "duration_s": time.monotonic() - self.started,
                    "scope": "existing observations only; no added hardware reads or commands",
                    "events": self.events, "dropped_events": self.dropped_events,
                    "logging_errors": self.errors}
        repo = Path(__file__).resolve().parents[3]
        source = {"observation": "on-disk files at post-attempt flush; not attestation of imported bytecode",
                  "repo": str(repo), "sha256": {}}
        document["source"] = source
        try:
            for relative in _SOURCE_FILES:
                source["sha256"][relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for key, command in (("git_head", ["rev-parse", "HEAD"]),
                                 ("git_tracked_status_porcelain", ["status", "--porcelain", "--untracked-files=no"])):
                source[key] = subprocess.run(["git", "-C", str(repo), *command],
                                             capture_output=True, text=True, check=True,
                                             timeout=2.).stdout.strip()
        except Exception as exc:
            self.error("source_at_flush", exc)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.arrays:
                cloud_path = self.directory / f"{self.id}.npz"
                # Uncompressed: avoid CPU contention with simulation during
                # the post-attempt write. allow_pickle=False is safe on read.
                with cloud_path.open("xb") as stream:
                    np.savez(stream, **self.arrays)
                document["arrays"] = {"path": cloud_path.name,
                                      "sha256": hashlib.sha256(cloud_path.read_bytes()).hexdigest()}
        except Exception as exc:
            self.error("write_arrays", exc)
        document["logging_ok"] = not self.errors and self.dropped_events == 0
        try:
            path = self.directory / f"{self.id}.json"
            serialized = json.dumps(_json_value(document), indent=2, allow_nan=False)
            with path.open("x") as stream:
                stream.write(serialized + "\n")
            _stderr({"attempt_id": self.id, "receipt": str(path),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "logging_ok": document["logging_ok"], "logging_errors": self.errors,
                     "dropped_events": self.dropped_events})
        except Exception as exc:
            self.error("write_receipt", exc)
            _stderr({"attempt_id": self.id, "receipt": None, "logging_ok": False,
                     "logging_errors": self.errors, "dropped_events": self.dropped_events,
                     "terminal_event": _json_value(self.events[-1]) if self.events else None})


def event(kind, **data):
    attempt = _ACTIVE.get()
    if attempt is not None:
        try:
            attempt.record(kind, data)
        except Exception as exc:
            attempt.error(f"event:{kind}", exc)


def phase(name):
    attempt = _ACTIVE.get()
    if attempt is not None:
        attempt.phase = name
        event("phase")


def array(name, value):
    attempt = _ACTIVE.get()
    if attempt is not None:
        try:
            attempt.array(name, value)
        except Exception as exc:
            attempt.error(f"array:{name}", exc)


def localized(frame, fix, extrinsics=None):
    if _ACTIVE.get() is None:
        return
    try:
        array("localized_points_base_m", fix.points)
        # Preserve the exact already-read observation, including nearby objects.
        # This is an opt-in copy, not a camera read or a new segmentation call.
        array("frame_rgb_bgr", frame.rgb)
        if frame.depth_m is not None:
            array("frame_depth_m", frame.depth_m)
        if getattr(frame, "robot_mask", None) is not None:
            array("frame_robot_mask", frame.robot_mask)
        if getattr(fix.detection, "mask", None) is not None:
            array("frame_target_mask", fix.detection.mask)
        T = frame.T_base_cam
        if T is None and extrinsics is not None:
            if getattr(extrinsics, "mode", None) == "eye_to_hand":
                T = extrinsics.T  # static calibration; never call a live FK hook
            else:
                _ACTIVE.get().error("frame_T_base_cam", ValueError(
                    "capture-time transform absent; live eye-in-hand FK is not evidence"))
        array("frame_K", frame.K)
        if T is not None:
            array("frame_T_base_cam", T)
        event("localized", label=fix.label, position_base_m=fix.position,
              extent_m=fix.extent, axes_base=fix.axes, fix_t=fix.t,
              detection_label=fix.detection.label, confidence=fix.detection.conf,
              bbox=fix.detection.bbox, frame_id=frame.frame_id, frame_t=frame.t,
              capture=frame.capture, K=frame.K, T_base_cam=frame.T_base_cam,
              effective_T_base_cam=T, depth_source=frame.depth_source)
    except Exception as exc:
        _ACTIVE.get().error("localized", exc)


def exception(kind, exc):
    attempt = _ACTIVE.get()
    if attempt is not None:
        try:
            event(kind, exception_type=type(exc).__name__, message=str(exc),
                  traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception as error:
            attempt.error(f"exception:{kind}", error)


def ik_result(kind, result):
    attempt = _ACTIVE.get()
    if attempt is not None:
        try:
            event(kind, success=result.success, q=getattr(result, "q", None),
                  error=getattr(result, "error", None))
        except Exception as exc:
            attempt.error(f"ik:{kind}", exc)


def ggx_response(response, request_attempt):
    attempt = _ACTIVE.get()
    if attempt is not None:
        try:
            event("ggx_response", request_attempt=request_attempt, branch_tags=response.get("branch_tags"))
            array("ggx_raw_poses", response["grasps"])
            array("ggx_raw_scores", response["confidences"])
        except Exception as exc:
            attempt.error("ggx_response", exc)


def record_attempt(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        directory = os.environ.get("CASCADE_GRASP_EVIDENCE_DIR", "").strip()
        if not directory:
            return function(self, *args, **kwargs)
        try:
            attempt = Attempt(directory)
        except Exception as exc:
            _stderr({"logging_ok": False, "operation": "create_attempt", "error": str(exc)})
            return function(self, *args, **kwargs)
        token = _ACTIVE.set(attempt)
        try:
            event("request", label=args[0] if args else kwargs.get("label"),
                  material=args[1] if len(args) > 1 else kwargs.get("material"),
                  spatial_hint=args[2] if len(args) > 2 else kwargs.get("spatial_hint"))
            try:
                event("configuration", arm=self.cfg.arm, grasp=self.cfg.grasp)
            except Exception as exc:
                attempt.error("configuration", exc)
            result = function(self, *args, **kwargs)
            event("result", result=result)
            return result
        except BaseException as exc:
            exception("attempt_exception", exc)
            raise
        finally:
            _ACTIVE.reset(token)
            try:
                attempt.finish()
            except Exception as exc:
                _stderr({"attempt_id": attempt.id, "logging_ok": False,
                         "operation": "finish", "error": str(exc)})
    return wrapped
