"""Explicit fixed-hand composition. Passive discovery never imports MuJoCo."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from ..control.hand import HandController, HandFault, quiet
from ..robotics.contracts import ResourceDescriptor
from ..sensing.models import digest
from ..sim.leap_hand import JOINTS, RECIPE
from ..skills.hand_runtime import HAND_SPECS, HandDomain


def validate_hand_profile(profile):
    required = {"kind", "recipe", "asset_root", "model_identity_sha256"}
    if not isinstance(profile, dict) or required-set(profile) or set(profile)-required-{"robot_id"}:
        raise ValueError("hand profile requires the exact fixed-hand fields")
    if (profile["kind"] != "hand" or profile["recipe"] != RECIPE
            or profile.get("robot_id", "leap_hand_right") != "leap_hand_right"):
        raise ValueError("only the declared fixed right LEAP hand recipe is implemented")
    if not isinstance(profile["asset_root"], str) or not Path(profile["asset_root"]).is_absolute():
        raise ValueError("hand asset root must be explicit and absolute")
    digest(profile["model_identity_sha256"])


def hand_description(domain_id, profile):
    validate_hand_profile(profile)
    resource = domain_id+"/fingers"
    return (ResourceDescriptor(resource, "articulated_hand", "leap_hand_right",
        capabilities=("bounded_finger_motion", "solved_contact_observation"),
        controller_id="mujoco-hand:"+str(Path(profile["asset_root"]).resolve()), writer_id=resource,
        synthetic=False, admission="unvalidated", metadata={"model_identity_sha256": profile["model_identity_sha256"],
            "joint_names": JOINTS, "root": "fixed", "grasping": False, "tactile_calibration": None}),), HAND_SPECS


def prepare_hand_model(profile):
    validate_hand_profile(profile)
    from ..sim.leap_hand import LeapHandBackend
    return LeapHandBackend(profile["asset_root"])


def _save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class RecordedHandDomain(HandDomain):
    def __init__(self, controller, resources, directory, *, domain_id):
        super().__init__(controller, resources, domain_id=domain_id)
        self.directory = directory
        self._recorded = False

    def close(self):
        result = super().close()
        if not self._recorded and result["owner_thread_closed"]:
            with (self.directory/"solves.jsonl").open("x") as stream:
                for row in self.controller.records():
                    stream.write(json.dumps(asdict(row), sort_keys=True, allow_nan=False)+"\n")
            _save(self.directory/"closure.json", result)
            self._recorded = True
        return result


def build_hand_runtime(profile, directory, *, domain_id):
    validate_hand_profile(profile)
    if profile["model_identity_sha256"] is None:
        raise HandFault("hand model is unprepared; an explicit preparation pin is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise HandFault("hand output directory must be empty and private")
    controller = domain = None
    try:
        backend = prepare_hand_model(profile)
        _save(directory/"model.json", backend.document)
        if backend.model_sha256 != profile["model_identity_sha256"] or backend.synthetic:
            raise HandFault("constructed hand differs from its native model pin")
        controller = HandController(backend)
        resources, _ = hand_description(domain_id, profile)
        domain = RecordedHandDomain(controller, resources, directory, domain_id=domain_id)
        controller.start()
        deadline, since, cursor = time.monotonic()+3., None, 0
        while time.monotonic() < deadline:
            for row in controller.read(cursor):
                cursor = row.step
                if row.generation != 0 or row.commanded_position_rad != backend.initial_targets:
                    raise HandFault("hand startup changed its stopped initial hold")
                if not quiet(row, backend.limits):
                    since = None
                elif since is None:
                    since = row.simulation_time_s
                if since is not None and row.simulation_time_s-since >= backend.limits.quiet_window_sim_s-1e-9:
                    if time.monotonic() >= deadline:
                        raise HandFault("hand readiness returned after its original deadline")
                    _save(directory/"readiness.json", {"ready": True, "latched": True, "generation": 0,
                        "step": row.step, "epoch": row.epoch, "model_identity_sha256": backend.model_sha256})
                    return domain
        raise HandFault("hand did not establish observed stopped readiness")
    except BaseException as exc:
        try:
            reason = str(exc)
        except BaseException:
            reason = type(exc).__name__+": unreadable exception"
        try:
            closure = domain.close() if domain is not None else None
        except BaseException as close_error:
            closure = {"ok": False, "error_type": type(close_error).__name__}
        try:
            _save(directory/"startup-failure.json", {"ready": False, "error": reason, "closure": closure})
        except BaseException:
            exc.add_note("The hand startup failure could not be persisted; closure was attempted.")
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare a source-pinned fixed hand; does not start its owner or advance physics.")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile = {"kind": "hand", "recipe": RECIPE, "asset_root": str(args.asset_root.resolve()),
        "model_identity_sha256": None}
    model = prepare_hand_model(profile)
    _save(args.output_dir/"model.json", model.document)
    _save(args.output_dir/"prepared-profile.json", profile | {"model_identity_sha256": model.model_sha256})
    print(json.dumps({"model_identity_sha256": model.model_sha256, "physics_steps": 0,
        "owner_started": False, "physical_admission": False}))


if __name__ == "__main__":
    main()
