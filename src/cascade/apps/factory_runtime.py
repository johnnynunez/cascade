"""Explicit Factory domain construction for the ordinary composed MCP runtime.

Discovery imports no SDK and opens no devices. The shipped profile is unpinned
and therefore cannot construct a native owner. A model pin is an identity check,
not physical admission; fresh solved measurements still decide every action.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import threading
import time

from ..control.fastening import FasteningFault, check_solve
from ..robotics.contracts import ResourceDescriptor
from ..skills.fastening_runtime import FasteningDomain, TURN_SPEC
from ..sim.factory_recipe import seating_recipe

ROBOT_ID = "so101_factory_m20"
CONTROLLER_ID = "factory_newton:private_m20"


def _retain_error_note(error, message):
    """Evidence must not replace the primary exception on Python 3.10 either."""
    note = getattr(error, "add_note", None)
    try:
        if callable(note):
            note(message)
            return
    except Exception:
        pass
    try:
        logging.getLogger(__name__).error(message)
    except Exception:
        pass  # A broken logging sink cannot replace the original failure.


def validate_factory_profile(profile):
    required = {"kind", "recipe", "assets", "robot_asset", "device", "model_identity_sha256"}
    if not isinstance(profile, dict) or set(profile)-required-{"robot_id", "precompile"} or required-set(profile):
        raise ValueError("fastening requires the explicit fixed Factory profile fields")
    seating_recipe(profile["recipe"])
    if "precompile" in profile:
        from ..sim.factory_precompile import PRECOMPILE_RECIPE
        from ..sim.factory_recipe import MARGIN_RECIPE
        if profile["precompile"] != PRECOMPILE_RECIPE or profile["recipe"] != MARGIN_RECIPE:
            raise ValueError("unreviewed Factory precompile recipe")
    if (profile["kind"] != "fastening"
            or profile.get("robot_id", ROBOT_ID) != ROBOT_ID):
        raise ValueError("only the mounted fixed-axis SO-101 Factory M20 recipe is implemented")
    for name in ("assets", "robot_asset"):
        if not isinstance(profile[name], str) or not Path(profile[name]).is_absolute():
            raise ValueError("Factory asset paths must be explicit absolute paths")
    device = profile["device"]
    if device is not None and (not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device)):
        raise ValueError("Factory device must be an explicit CUDA ordinal or null (unprepared)")
    digest = profile["model_identity_sha256"]
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("Factory model identity must be an exact SHA256 or null (unprepared)")


def factory_description(domain_id, profile):
    validate_factory_profile(profile)
    resource = domain_id + "/mounted_arm_spindle"
    return (ResourceDescriptor(resource, "mounted_fastening", ROBOT_ID,
        capabilities=("preengaged_thread_turn",), controller_id=CONTROLLER_ID,
        writer_id=resource, synthetic=False, admission="unvalidated",
        metadata={"model_identity_sha256": profile["model_identity_sha256"],
            "mounted_tool": True, "preengaged_fastener": True, "seating": False, "pickup": False}),), [TURN_SPEC]


def prepare_factory_model(profile, cache_dir):
    """Explicit SDK/model construction, zero solves and no control upload.

    Call only inside an owned native process. This is also the construction
    boundary for a separately supervised passive model-preparation campaign.
    It never fetches assets or substitutes an installed SDK implementation.
    """
    validate_factory_profile(profile)
    if profile["device"] is None:
        raise FasteningFault("Factory device is unprepared; select an explicit CUDA ordinal before construction")
    from ..sim.factory_observation import sdk_sources
    sdk_sources()  # Refuse an unreviewed implementation before model construction.
    from ..sim.newton_screw_seating import SeatingScene
    from ..sim.factory_model import FactoryBoundModel
    scene = SeatingScene(profile["assets"], profile["robot_asset"], Path(cache_dir),
                         device=profile["device"], drive=False, substeps=10,
                         recipe=profile["recipe"])
    if not scene.model.device.is_cuda:
        raise FasteningFault("requested CUDA Factory model did not resolve to a CUDA device")
    if "precompile" in profile:
        try:
            model = FactoryBoundModel(scene, precompile=profile["precompile"])
        except BaseException as error:
            try:
                if hasattr(scene, "precompile_receipt"):
                    _write(Path(cache_dir).parent/"precompile.json", scene.precompile_receipt)
            except Exception as persist_error:
                _retain_error_note(error, f"precompile receipt persistence also failed: {persist_error!r}")
            raise
        _write(Path(cache_dir).parent/"precompile.json", scene.precompile_receipt)
        return model
    return FactoryBoundModel(scene)


def _new_owner(model):
    from ..sim.factory_owner import FactoryNewtonBackend, FactorySolveOwner
    return FactorySolveOwner(FactoryNewtonBackend(model))


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")


class RecordedFactoryDomain(FasteningDomain):
    def __init__(self, owner, directory, *, domain_id):
        self.owner, self.directory = owner, Path(directory)
        self._record_lock = threading.Lock()
        self._record_error = None
        super().__init__(owner.controller, owner.journal.read,
                         controller_id=CONTROLLER_ID, domain_id=domain_id)

    def flush_records(self):
        # Never called by the owner/write/stop path. A slow disk cannot delay
        # priority stop delivery. Overflow already fails in the native owner.
        with self._record_lock:
            try:
                rows = self.owner.records()
                with (self.directory/"solves.jsonl").open("a") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            except Exception as exc:
                self._record_error = f"{type(exc).__name__}: {exc}"
                self.actuator.stop()
                raise

    def execute(self, name, args):
        if self._record_error:
            return {"ok": False, "execution_ok": False, "verified": False,
                "postcondition": {"status": "unverified", "reason": "sticky raw evidence persistence fault"},
                "evidence_error": self._record_error}
        result = super().execute(name, args)
        try:
            self.flush_records()
            _write(self.directory/"last-action.json", result)
        except Exception as exc:
            self._record_error = f"{type(exc).__name__}: {exc}"
            self.actuator.stop()
            result.update(ok=False, verified=False,
                evidence_error=str(exc), postcondition={"status": "unverified", "reason": "raw evidence persistence failed"})
        return result

    def reset_stop(self):
        if self._record_error:
            raise FasteningFault("raw evidence persistence fault requires a new owned epoch")
        return super().reset_stop()

    def close(self):
        result = super().close()
        try:
            self.flush_records()
        except Exception as exc:
            result.update(ok=False, evidence_error=str(exc))
        if self._record_error:
            result.update(ok=False, evidence_error=self._record_error)
        _write(self.directory/"closure.json", result)
        return result


def _ready(owner, *, clock=time.monotonic, timeout_s=10.):
    """Observe a real quiet interval while the initial spindle latch remains on."""
    end = clock()+timeout_s
    previous = start = None
    cursor = 0
    while clock() < end:
        rows = owner.journal.read(cursor, timeout_s=min(.05, max(0., end-clock())))
        for row in rows:
            check_solve(row, owner.backend.binding, owner.backend.limits, clock(),
                        previous=previous, epoch=None if previous is None else previous.epoch,
                        stage="startup_readiness")
            if row.step != cursor+1 or row.generation != 0:
                raise FasteningFault("startup stream lost solves or actuator was admitted early")
            cursor, previous = row.step, row
            if clock() >= end:
                raise FasteningFault("readiness observation returned after startup deadline")
            limits = owner.backend.limits
            quiet = (row.commanded_spindle_effort_nm == row.spindle_effort_nm == 0.
                and max(row.fastener_linear_speed_m_s, row.tool_linear_speed_m_s) <= limits.rest_linear_speed_m_s
                and max(abs(row.spindle_speed_rad_s), row.fastener_angular_speed_rad_s,
                        row.tool_angular_speed_rad_s, *(abs(v) for v in row.joint_velocity_rad_s))
                    <= limits.rest_angular_speed_rad_s)
            if not quiet or not row.thread_contacts or not row.tool_contacts:
                start = None
            elif start is None:
                start = row.simulation_time_s
            if start is not None and row.simulation_time_s-start >= limits.rest_window_sim_s-1e-9:
                return {"ready": True, "epoch": row.epoch, "binding_sha256": row.binding_sha256,
                    "step": row.step, "captured_monotonic_s": row.captured_monotonic_s,
                    "generation": 0, "latched": True, "quiet_interval_sim_s": row.simulation_time_s-start,
                    "physical_task_admission": False}
    raise FasteningFault("startup did not produce quiet pre-engaged independent observations")


def build_factory_runtime(profile, directory, *, domain_id):
    validate_factory_profile(profile)
    expected = profile["model_identity_sha256"]
    if expected is None:
        raise FasteningFault("Factory model identity is unprepared; no native owner was constructed")
    if profile["device"] is None:
        raise FasteningFault("Factory device is unprepared; select an explicit CUDA ordinal before construction")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Never append another epoch to an earlier run's raw evidence.
    marker = directory/"factory-profile.json"
    with marker.open("x") as stream:
        json.dump(profile, stream, sort_keys=True, indent=2, allow_nan=False)
    domain = None
    try:
        model = prepare_factory_model(profile, directory/"sdf-cache")
        _write(directory/"model.json", model.document)
        if model.binding.model_sha256 != expected:
            raise FasteningFault("actual Factory model differs from the exact profile pin")
        owner = _new_owner(model)
        if owner.backend.synthetic is not False:
            raise FasteningFault("configured native Factory builder cannot substitute a synthetic owner")
        domain = RecordedFactoryDomain(owner, directory, domain_id=domain_id)
        owner.start()
        readiness = _ready(owner)
        domain.flush_records()
        _write(directory/"readiness.json", readiness)
        return domain
    except BaseException as exc:
        failure = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        if domain is not None:
            try:
                failure["closure"] = domain.close()
            except BaseException as closing:
                failure["closure"] = {"ok": False, "error": f"{type(closing).__name__}: {closing}"}
        try:
            _write(directory/"startup-failure.json", failure)
        except BaseException as persistence:
            failure["persistence_error"] = f"{type(persistence).__name__}: {persistence}"
            # Preserve the causal startup exception even if the output device
            # is gone. The owning process's stderr/trace can retain this note.
            _retain_error_note(exc, "Factory startup failure receipt: " + json.dumps(failure, sort_keys=True))
        raise
