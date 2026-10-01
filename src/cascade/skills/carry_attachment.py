"""Retain the NV post-close attachment through lift and placement transport.

Uses feedback already requested by the Isaac motion executor. This observes
bilateral attachment, not grip force or mechanical stability. Loss/unknown
stops the attempt; it never opens, homes, retries or clears a held marker.
"""
from __future__ import annotations

import copy
import numpy as np

from ..control.simulation_motion import PhysicsClock
from ..grasping import evidence
from ..types import SafetyViolation
from .contact_episode import _backend, _identity


class AttachmentInvalid(SafetyViolation):
    """A retained attachment no longer authorizes transport."""


class RetainedAttachment:
    def __init__(self, runtime, episode):
        self.failure, self.last, self.status, self.paths = None, None, "unknown", ()
        # Initialization errors are terminal too; they cannot vanish on retry.
        runtime._carry_attachment = self
        self.runtime, self.arm = runtime, runtime.arm
        self.raw, self.backend = self.arm.raw, _backend(self.arm)
        self.harness = self.arm.harness
        self.occupancy = self.harness.occupancy
        self.identity = _identity(runtime)
        self.generation = episode["halt_generation"]
        self.reset_generation = self.occupancy.scene_reset_generation
        self.signs = self.backend._signs.copy()
        self.config_signs = copy.deepcopy(runtime.cfg.arm.get("joint_signs"))
        self.source = self.backend._client._addr
        self.robot = self.backend._cfg.get("bridge_robot_id")
        self.clock = PhysicsClock(self.source, self.robot)
        self.paths = tuple(episode["expected_paths"] or ())
        self.barrier = copy.deepcopy(episode["barrier"])
        self.status = "attached"
        self.jaw_identity = None
        markers = (self.barrier or {}).get("integrated", [])
        epochs = {m.get("producer_epoch") for m in markers}
        if (len(self.paths) != 1 or not markers or len(epochs) != 1
                or not all(isinstance(e, str) and e for e in epochs)
                or any(tuple(m.get("source", ())) != self.source or m.get("robot_id") != self.robot
                       or m.get("clock") != "physics_loop_monotonic" for m in markers)):
            self.reject("post-close attachment binding is incomplete")
        self.epoch = next(iter(epochs))

    def reject(self, reason, *, status="unknown"):
        if self.failure is None:
            self.failure, self.status = str(reason), status
            evidence.event("carry_attachment_rejected", reason=self.failure,
                           attachment_status=self.status, expected_paths=self.paths,
                           last_valid=self.last)
        raise AttachmentInvalid(self.failure)

    def guard(self):
        if self.failure is not None:
            raise AttachmentInvalid(self.failure)
        r = self.runtime
        if (r.arm is not self.arm or self.arm.raw is not self.raw
                or _backend(self.arm) is not self.backend or self.arm.harness is not self.harness
                or self.harness.occupancy is not self.occupancy
                or _identity(r) != self.identity
                or self.occupancy.scene_reset_generation != self.reset_generation
                or self.backend._client._addr != self.source
                or not np.array_equal(self.backend._signs, self.signs)
                or not np.array_equal(r.cfg.arm.get("joint_signs"), self.config_signs)):
            self.reject("retained attachment arm, map, or producer identity changed")
        try:
            self.harness._check_halt_generation(self.generation)
            if self.harness.estopped:
                raise SafetyViolation("e-stop during attachment transport")
        except SafetyViolation as exc:
            self.reject(str(exc))

    def observe(self, state):
        """Validate one existing atomic state; never request another sample."""
        self.guard()
        try:
            self.clock.observe(state.physics_clock)
            a = state.attachment
            if (not isinstance(a, dict) or type(a.get("version")) is not int or a["version"] != 1
                    or a.get("backend") != "isaac" or a.get("source") != self.source
                    or a.get("robot_id") != self.robot or a.get("producer_epoch") != self.epoch
                    or state.physics_clock["epoch"] != self.epoch
                    or type(a.get("physics_step")) is not int or a["physics_step"] != self.clock.step
                    or type(a.get("sim_time")) not in (int, float) or a["sim_time"] != self.clock.time
                    or a.get("joint_convention") != "asset"
                    or a.get("channel") != "completed_update_bilateral_contact"
                    or a.get("sensor_channel") != {"physx": "physx_gpu_contact_tensor",
                                                    "newton": "newton_mjwarp_contact_force"}.get(state.physics_clock["engine"])
                    or a.get("tracking") is not True or a.get("error") is not None):
                self.reject("attachment feedback is absent, invalid, or not from this physical state")
            aq = np.asarray(a.get("q"))
            q = np.asarray(state.q)
            if (aq.shape != (self.backend.n_joints,) or aq.dtype.kind not in "fiu"
                    or not np.isfinite(aq).all() or q.shape != aq.shape
                    or not np.array_equal(aq * self.signs, q)
                    or a.get("gripper_joints") != state.gripper_joints):
                self.reject("attachment articulation differs from atomic feedback")
            jaws = state.gripper_joints
            from .release_episode import _jaws
            self.jaw_identity = _jaws(jaws, expected=self.jaw_identity)
            paths = a.get("paths")
            if (not isinstance(paths, list) or any(not isinstance(p, str) or not p for p in paths)
                    or len(set(paths)) != len(paths)):
                self.reject("attachment paths are unknown")
            if tuple(sorted(paths)) != self.paths:
                self.reject("observed bilateral attachment was lost or changed", status="lost")
            current = {"physics_clock": copy.deepcopy(state.physics_clock),
                       "q": q.tolist(), "gripper_joints": copy.deepcopy(jaws), "paths": list(paths)}
            if (self.last is not None and self.last["physics_clock"]["physics_step"] == self.clock.step
                    and current != self.last):
                self.reject("attachment feedback changed without a physical step")
            self.guard()
            self.last = current
        except AttachmentInvalid:
            raise
        except Exception as exc:
            self.reject("invalid retained attachment feedback: " + str(exc))


def arm(runtime, episode, state=None):
    """Only an explicit Isaac capability + original NV barrier may arm this."""
    check(runtime)
    if episode is None or getattr(_backend(runtime.arm), "attachment_feedback_version", None) != 1:
        return state if state is not None else runtime.arm.get_state()
    try:
        retained = RetainedAttachment(runtime, episode)
        # Bind before the existing post-close read, so its timeout is retained
        # too. This substitutes that read; it adds no request.
        if state is None:
            state = runtime.arm.get_state()
        retained.observe(state)
    except AttachmentInvalid:
        raise
    except Exception as exc:
        active(runtime).reject("invalid post-close attachment authority: " + str(exc))
    evidence.event("carry_attachment_armed", paths=retained.paths, state=retained.last)
    return state


def active(runtime):
    return getattr(runtime, "_carry_attachment", None)


def check(runtime):
    retained = active(runtime)
    if retained is not None:
        retained.guard()


def observe(runtime, state):
    retained = active(runtime)
    if retained is not None:
        retained.observe(state)


def move(runtime, q, **kwargs):
    retained = active(runtime)
    if retained is None:
        return runtime.arm.move_joints(q, **kwargs)
    retained.guard()
    if any(kwargs.get(key) is not None for key in ("before_stream", "feedback_guard")):
        retained.reject("attachment transport cannot replace another safety callback")
    if (kwargs.get("_halt_generation") is not None
            and kwargs["_halt_generation"] != retained.generation):
        retained.reject("attachment transport cancellation token differs")
    kwargs.update(_halt_generation=retained.generation, before_stream=retained.guard,
                  feedback_guard=retained.observe)
    try:
        result = runtime.arm.move_joints(q, **kwargs)
        retained.guard()
        if not result:
            retained.reject("attachment transport did not settle")
        return result
    except AttachmentInvalid:
        raise
    except Exception as exc:
        retained.reject("attachment transport interrupted: " + str(exc))


def failure_result(runtime, error, **extra):
    retained = active(runtime)
    return {"ok": False, "stage": "carry_attachment", "error": str(error),
            "attachment_status": retained.status if retained is not None else "unknown",
            "grip_verified": False, "home_skipped": True, "retry_allowed": False,
            "retained_held_marker": getattr(runtime, "held_object", None),
            "note": "Attachment lost or unavailable; held marker retained only as a precaution. "
                    "No release, retry or return-home was authorized.", **extra}
