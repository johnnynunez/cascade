"""Opt-in campaign selection evidence; never a physical task verifier.

Inspect existing camera caches and the original planner/stream handoff. No
extra frame/state requests, SDK construction, trajectory changes or retries.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ARM_PROFILES = ("isaac_kitchen_gpu", "isaac_kitchen_cumotion")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_selection(arm_profile, renderer):
    if arm_profile not in ARM_PROFILES or renderer not in (None, "isaac", "ovrtx"):
        raise ValueError("unsupported kitchen arm profile or renderer expectation")
    if arm_profile == "isaac_kitchen_cumotion" and renderer is None:
        raise ValueError("cuMotion kitchen selection requires an explicit --camera-renderer")


class BackendEvidence:
    """Retain selected-backend evidence at phase and native-stream boundaries.

The original SafeArm approvals and physical executor remain authoritative.
Camera cache checks identify the renderer; they do not replace freshness,
contact, occupancy, measured tracking or independent physical acceptance.
"""

    def __init__(self, runtime, cfg, *, arm_profile, renderer, bounded=False, attach=True):
        validate_selection(arm_profile, renderer)
        self.runtime, self.cfg = runtime, cfg
        self.arm_profile, self.renderer = arm_profile, renderer
        self.arm, self.raw = runtime.arm, runtime.arm.raw
        self.planner = self.arm.motion_planner
        self.cumotion = arm_profile == "isaac_kitchen_cumotion"
        self.phase = "setup"
        self.bounded = bounded
        self.errors_omitted = 0
        self.records, self.selection_checks, self.errors = [], [], []
        self._original_stream = None
        self._native = None
        if self.cumotion:
            from cascade.planning.cumotion import CumotionPlanner
            from cascade.planning.runtime import RuntimeMotionPlanner

            # The ordinary builder has already prepared the explicit SDK.
            # Never initialize it here or accept a config flag as execution.
            if (type(self.planner) is not RuntimeMotionPlanner
                    or type(self.planner._planner) is not CumotionPlanner):
                raise RuntimeError("selected cuMotion runtime was not prepared by the ordinary builder")
            self._native = self.planner._planner
            config = cfg.arm.motion_planner.as_dict()
            if config != self.planner.config:
                raise RuntimeError("effective runtime planner differs from selected profile")
            self.model_sha256 = _digest({key: Path(config[key]).read_text()
                                        for key in ("urdf", "xrdf")})
            self.scene_sha256 = _digest({"base_frame": config["base_frame"],
                                        "obstacles": self._native._boxes(config["obstacles"])})
        elif self.planner is not None or cfg.arm.get("motion_planner") is not None:
            raise RuntimeError("legacy kitchen profile unexpectedly selected a planner")
        self.check()
        if attach:
            self.attach()

    def attach(self):
        if self.cumotion and self._original_stream is None:
            self._original_stream = self.raw.stream_profile
            self.raw.stream_profile = self._stream

    def check(self):
        """Read cached selection evidence only, without waiting or freshening it."""
        try:
            if self.bounded and len(self.selection_checks) >= 256:
                raise RuntimeError("backend selection evidence capacity exceeded")
            if (self.runtime.arm is not self.arm or self.arm.raw is not self.raw
                    or self.arm.motion_planner is not self.planner):
                raise RuntimeError("campaign arm or planner binding changed")
            if self.cumotion and (
                    self.planner._closed or self.planner._planner is not self._native
                    or self._native._closed or self._native._poisoned
                    or self._native.model_sha256 != self.model_sha256
                    or self._native.scene_sha256 != self.scene_sha256):
                raise RuntimeError("selected cuMotion owner or model/world binding changed")
            captures = {}
            if self.renderer is not None:
                from cascade.sim.startup_readiness import _packet

                expected_source = (str(self.cfg.arm.get("bridge_host", "127.0.0.1")),
                                   int(self.cfg.arm.bridge_port))
                expected_cameras = {c["sim_camera"] for c in self.cfg._data["cameras"]}
                for stream in self.runtime.rig:
                    frame = stream.latest()
                    capture = getattr(frame, "capture", None)
                    if not isinstance(capture, dict):
                        raise RuntimeError("selected renderer has no cached capture evidence")
                    name, state = capture.get("camera"), capture.get("proprioception")
                    ref = capture.get("render_reference")
                    expected_ref = "ovrtx_snapshot" if self.renderer == "ovrtx" else "rpFabricTime"
                    if (name not in expected_cameras or name in captures
                            or capture.get("renderer") != self.renderer
                            or not isinstance(state, dict)
                            or not isinstance(ref, dict) or ref.get("source") != expected_ref
                            or not isinstance(state.get("producer_epoch"), str)
                            or not state["producer_epoch"]):
                        raise RuntimeError("selected renderer capture identity or binding differs")
                    # Reuse ordinary readiness's packet contract, including
                    # rpFabricTime tokens, joints/jaws and actual RGB-D. This
                    # function reads only this cached Frame and issues no RPC.
                    _packet(frame, expected_source, self.cfg.arm.bridge_robot_id,
                            state["producer_epoch"], int(self.cfg.arm.n_joints))
                    captures[name] = deepcopy(capture)
                if set(captures) != expected_cameras:
                    raise RuntimeError("selected renderer evidence is incomplete for configured cameras")
            record = {"phase": self.phase, "renderer": self.renderer, "captures": captures}
            self.selection_checks.append(record)
            return record
        except Exception as exc:
            self._error(f"{type(exc).__name__}: {exc}")
            raise

    def _stream(self, profile, *, planned_state, **kwargs):
        from cascade.control.arm_base import PREFLIGHT_MAX_DRIFT_RAD
        from cascade.planning.cumotion import MotionPlan, SDK_VERSION
        from cascade.planning.trajectory import TrajectoryProfile

        if self.bounded and len(self.records) >= 128:
            self._error("backend curve evidence capacity exceeded")
            raise RuntimeError("backend curve evidence capacity exceeded")
        record = {"index": len(self.records), "phase": self.phase,
                  "executor_called": False, "completed": False}
        self.records.append(record)
        try:
            selection = self.check()
            if type(profile) is not TrajectoryProfile or type(profile.plan) is not MotionPlan:
                raise RuntimeError("selected cuMotion stream lacks its native candidate")
            plan = profile.plan
            if (plan.sdk_version != SDK_VERSION or plan.model_sha256 != self.model_sha256
                    or plan.scene_sha256 != self.scene_sha256
                    or plan.joint_names != self._native.joint_names
                    or plan.base_frame != self._native.base_frame
                    or plan.tool_frame != self._native.tool_frame
                    or not isinstance(plan.request_sha256, str) or len(plan.request_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in plan.request_sha256)
                    or profile.start.shape != np.asarray(planned_state.q).shape
                    or not np.isfinite(planned_state.q).all() or not np.isfinite(profile.start).all()
                    or np.max(abs(profile.start - planned_state.q)) > PREFLIGHT_MAX_DRIFT_RAD):
                raise RuntimeError("cuMotion candidate does not match its runtime model or measured request")
            # Record the candidate separately from executor completion. Neither
            # is a claim about grip, payload motion, support or placement.
            record.update(plan=plan.as_dict(), duration_physics_s=profile.duration_s,
                          target_count=len(profile.targets),
                          safety_edge_count=sum(len(t.checks) for t in profile.targets),
                          planned_state_clock=deepcopy(planned_state.physics_clock),
                          renderer_selection_sha256=_digest(selection), executor_called=True)
            result = self._original_stream(profile, planned_state=planned_state, **kwargs)
            record["completed"] = result is True
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            self._error(record["error"])
            raise

    def _error(self, error):
        if not self.bounded or len(self.errors) < 128:
            self.errors.append(error)
        else:
            self.errors_omitted += 1

    def close(self):
        if self._original_stream is not None:
            self.raw.stream_profile = self._original_stream
            self._original_stream = None

    def report(self):
        pick = [r for r in self.records if r["phase"] == "pick"]
        passed = not self.errors and (not self.cumotion or (
            bool(pick) and all(r["executor_called"] and r["completed"] for r in self.records)))
        return deepcopy({"arm_profile": self.arm_profile, "expected_renderer": self.renderer,
                         "pass": passed, "physical_acceptance": False,
                         "scope": "selected backend and original curve handoff only; independent physics audit required",
                         "selection_checks": self.selection_checks, "curves": self.records,
                         "errors": self.errors})


def selection_from_environment(cfg, environment):
    """Pure opt-in validation, before any SDK/camera/arm construction."""
    renderer = environment.get("CASCADE_KITCHEN_CAMERA_RENDERER")
    if renderer is None:
        return None
    if (renderer not in ("isaac", "ovrtx")
            or environment.get("CASCADE_REQUIRE_CUDA") != "1"
            or cfg.arm.get("name") != "isaac_kitchen_cumotion"
            or cfg.arm.get("type") != "isaac"
            or len(cfg.get("arms", [])) != 1
            or cfg.arm.get("motion_planner") is None
            or cfg.arm.motion_planner.get("type") != "cumotion"):
        raise ValueError("explicit kitchen backend evidence requires one isaac_kitchen_cumotion arm, renderer and strict CUDA")
    return renderer


@contextmanager
def ordinary_call(runtime, skill, context):
    """One ordinary motion call; scoped checks, never a physical verdict.

    The builder prepared the SDK. Read cached cameras only. LazyArm exposes
    stream_profile without materializing the driver. Calls retain ordinary
    SafeArm approvals, deadlines, cancellation and independent postconditions.
    """
    renderer = getattr(runtime, "_kitchen_camera_renderer", None)
    if renderer is None:
        yield
        return
    evidence = None
    context["selected_backends"] = {"physical_acceptance": False,
        "arm_profile": "isaac_kitchen_cumotion", "expected_renderer": renderer,
        "skill": skill, "scope": "selection and curve handoff only"}
    try:
        evidence = BackendEvidence(runtime, runtime.cfg,
            arm_profile="isaac_kitchen_cumotion", renderer=renderer, bounded=True, attach=False)
        evidence.attach()
        evidence.phase = skill
        yield
    except BaseException as exc:
        # Do not stringify an arbitrary cancellation; preserving it is primary.
        context["selected_backends"]["exception_type"] = type(exc).__name__
        raise
    finally:
        if evidence is not None:
            evidence.close()
            # The campaign's 'pick' summary is not ordinary runtime admission.
            report = evidence.report()
            report.pop("pass")
            report.update(skill=skill, errors_omitted=evidence.errors_omitted,
                record_limits={"curves":128,"selection_checks":256,"errors":128})
            context["selected_backends"].update(report)
