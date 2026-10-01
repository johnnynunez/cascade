#!/usr/bin/env python3
"""Report one successful real descent as unsettled to exercise its withdrawal.

The injected false return is labelled; measured joints, depth, contact masks,
physics and safety decisions are untouched. This reproduces the failure branch,
not the unrecorded first cause of historical trial13. No grasp retry occurs.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import traceback

from nvblox_camera_recovery import (ROOT, OBJECTS, ActuatorTrace, close_without_motion,
    fresh_camera_checks, fresh_state, phase, reset_window, write_json)


class DescentSettleFault:
    def __init__(self, runtime):
        self.runtime = runtime
        self.original = runtime.arm.move_joints
        self.evidence = None
        self.previous_target = None
        self.retreat_calls = []

    def __enter__(self):
        def move(*args, **kwargs):
            # bias_compensate is the existing descent call-site marker. Only
            # inject after a completed move and before any jaw-close command.
            if self.evidence is not None:
                self.retreat_calls.append({"target_q": args[0].copy(),
                    "exemption": copy.deepcopy(self.runtime.arm.harness._grasp_exempt)})
            result = self.original(*args, **kwargs)
            if (self.evidence is None and kwargs.get("bias_compensate") is True
                    and result is True and not self.runtime.held_object
                    and getattr(self.runtime, "_held_provisional", None) is None):
                self.evidence = {"actual_move_result": result, "injected_result": False,
                    "target_q": args[0].copy(), "pregrasp_q": self.previous_target,
                    "exemption": copy.deepcopy(self.runtime.arm.harness._grasp_exempt)}
                return False
            if result is True and self.evidence is None:
                self.previous_target = args[0].copy()
            return result
        self.runtime.arm.move_joints = move
        return self

    def __exit__(self, *_):
        self.runtime.arm.move_joints = self.original


def run(args, out):
    import numpy as np
    from kitchen_acceptance import load_proof, retarget_ports, install_command_trace
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config
    cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_proof"], arm="isaac_kitchen_gpu", llm="mock")
    retarget_ports(cfg._data, args.port)
    cfg._data["occupancy"].update(port=args.occupancy_port, track_payload=True,
        allowed_contact_paths=["/World_Props/" + name for name in OBJECTS])
    for camera in cfg._data["cameras"]:
        camera["map_depth"] = True
    cfg._data["grasp"]["pregrasp_offset_m"] = .10
    receipt = {"pass": False, "scope": __doc__, "object": args.object, "errors": [], "phases": {}}
    receipt["effective_configuration"] = {"arm": cfg.arm.as_dict(), "grasp": cfg.grasp.as_dict(),
                                           "safety": cfg.safety.as_dict()}
    paths = sorted(set([Path(__file__), ROOT / "benchmark/diagnostics/nvblox_camera_recovery.py",
        ROOT / "benchmark/diagnostics/kitchen_acceptance.py", args.scene_config]
        + list((ROOT / "src/cascade").rglob("*.py")) + list((ROOT / "demo/kitchen/physics").glob("*.py"))
        + list((ROOT / "configs").rglob("*.yaml"))))
    receipt["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    runtime = recorder = observer = None
    try:
        runtime, _ = build_runtime(cfg, out / "runtime", lazy_arm=True)
        install_command_trace(runtime, out / "skills.jsonl")
        recorder = ActuatorTrace(runtime, out / "actuators.jsonl")
        occ = runtime.arm.harness.occupancy
        receipt["occupancy_probe"] = occ.probe(timeout_ms=2000)
        if (receipt["occupancy_probe"].get("backend") != "nvblox"
                or not str(receipt["occupancy_probe"].get("device", "")).startswith("cuda:")):
            raise RuntimeError("real CUDA nvblox required")
        proof = load_proof()
        expected = proof.shared.load_expected_scene_geometry(args.scene_config)
        observer = proof.shared.GpuProofObserver(out / "witness", port=args.port, interval_s=.15,
            budget_s=600, object_name=args.object, destination_name="open box", expected_scene_geometry=expected)
        with observer:
            with observer._lock:
                identities = {(r["physics"]["engine"], r["physics"]["robot_id"]) for r in observer.records}
            if identities != {(args.engine, cfg.arm.bridge_robot_id)}:
                raise RuntimeError("observer does not match requested engine/robot")
            setup = phase(observer, runtime, recorder, "setup_reset", lambda: runtime.execute("reset_scene", {}))
            receipt["phases"]["setup"] = setup
            if not setup["result"].get("ok"):
                raise RuntimeError("initial public reset failed")
            with DescentSettleFault(runtime) as fault:
                grasp = phase(observer, runtime, recorder, "faulted_grasp",
                    lambda: runtime.execute("grasp_object", {"label": args.object.replace("_", " ")}))
            receipt["phases"]["faulted_grasp"] = grasp
            receipt["injection"] = fault.evidence
            receipt["retreat_calls"] = fault.retreat_calls
            receipt["exemption_after_failure"] = runtime.arm.harness._grasp_exempt
            receipt["held_after_failure"] = runtime.held_object
            receipt["provisional_after_failure"] = runtime._held_provisional
            receipt["tcp_after_failure"] = runtime.kin.fk(runtime.arm.get_state().q)[:3, 3]
            # Preserve the failed grasp and perform only the normal public
            # reset. There is no retry or replacement grasp in this probe.
            reset = phase(observer, runtime, recorder, "recovered_reset", lambda: runtime.execute("reset_scene", {}))
            receipt["phases"]["recovered_reset"] = reset
            observer.settle(simulation_seconds=.65, wall_timeout=40)
            receipt["final_state"] = fresh_state(observer, runtime)
            with observer._lock:
                records = copy.deepcopy(observer.records)
            receipt["reset_physics"] = reset_window(records,
                after=observer.marks["recovered_reset_end"]["monotonic"], cfg=cfg,
                expected_props=expected["prop_dimensions_m"], engine=args.engine)
            calls = [e for e in recorder.events if e["phase"] == "faulted_grasp" and e["event"] == "begin"]
            jaw_calls = [e for e in calls if e["layer"] == "actuator" and e["call"] == "set_gripper"]
            before, after = grasp["before"]["physics"], grasp["after"]["physics"]
            receipt["prop_drift_m"] = {name: float(np.linalg.norm(np.asarray(value["position_m"]) - after["props"][name]["position_m"]))
                for name, value in before["props"].items()}
            error = grasp["result"].get("error", "")
            def same_exemption(other):
                initial = fault.evidence["exemption"] if fault.evidence else None
                return (initial is not None and other is not None
                    and np.array_equal(initial[0], other[0]) and initial[1:] == other[1:])
            actual_q = np.asarray(after["q"][:len(cfg.arm.home_q)]) * np.asarray(cfg.arm.joint_signs)
            pregrasp = fault.evidence.get("pregrasp_q") if fault.evidence else None
            receipt["retreat_endpoint_error_rad"] = (float(np.max(np.abs(actual_q - pregrasp)))
                if pregrasp is not None else None)
            fresh_checks = fresh_camera_checks(reset["result"], {"cam0", "side", "proof"})
            receipt["checks"] = {
                "fault_injected_after_real_successful_descent": fault.evidence is not None,
                "original_failure_and_retreat_reported": grasp["result"].get("ok") is False
                    and "did not settle at grasp pose; pre-close retreat completed" in error,
                "no_close_or_drop_command": len(jaw_calls) == 1 and jaw_calls[0]["args"][0] == runtime._grip_open,
                "empty_state_preserved": receipt["held_after_failure"] is None and receipt["provisional_after_failure"] is None,
                "contact_scope_cleared_after_retreat": receipt["exemption_after_failure"] is None,
                "same_contact_scope_throughout_retreat": bool(fault.retreat_calls)
                    and all(same_exemption(c["exemption"]) for c in fault.retreat_calls),
                "retreat_targets_original_pregrasp": pregrasp is not None and bool(fault.retreat_calls)
                    and np.array_equal(fault.retreat_calls[-1]["target_q"], pregrasp),
                "actual_pregrasp_reached": receipt["retreat_endpoint_error_rad"] is not None
                    and receipt["retreat_endpoint_error_rad"] <= float(cfg.arm.get("settle_tol", .02)),
                "public_reset_completed": reset["result"].get("ok") is True,
                "reset_physics": receipt["reset_physics"]["pass"],
                "fresh_map": receipt["final_state"]["map"]["has_grid"] and occ._grid is not None
                    and not occ.scene_reset_pending and not occ.is_stale() and occ._body_error is None and occ.last_error is None,
                "all_camera_depths_rebuilt": set(receipt["final_state"]["map"]["camera_history"]) == {"cam0", "side", "proof"},
                **fresh_checks,
            }
    except Exception:
        receipt["errors"].append(traceback.format_exc())
    finally:
        if runtime is not None:
            if recorder:
                recorder.phase = "cleanup_without_motion"
            receipt["cleanup_errors"] = close_without_motion(runtime)
            receipt["cleanup_calls"] = recorder.calls("cleanup_without_motion") if recorder else []
        receipt.setdefault("checks", {}).update(
            source_unchanged=all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in receipt["source_sha256"].items()),
            no_diagnostic_errors=not receipt["errors"] and not receipt.get("cleanup_errors"),
            observer_complete=bool(observer and observer.complete and not observer.errors),
            cleanup_sent_no_commands=not receipt.get("cleanup_calls"))
        receipt["pass"] = all(receipt["checks"].values())
        receipt["artifacts_sha256"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*") if p.is_file() and p.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8691)
    parser.add_argument("--occupancy-port", type=int, default=25559)
    parser.add_argument("--engine", choices=("physx", "newton"), default="physx")
    parser.add_argument("--object", choices=OBJECTS, default="orange")
    parser.add_argument("--scene-config", type=Path, default=ROOT / "demo/scene/kitchen_config.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    out, args.scene_config = args.output.resolve(), args.scene_config.resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        parser.error("output must be a NEW directory inside this checkout")
    if not args.scene_config.is_relative_to(ROOT) or not args.scene_config.is_file():
        parser.error("scene-config must exist inside the checkout")
    if not all(1 <= p <= 65535 for p in (args.port, args.occupancy_port)):
        parser.error("ports must be 1..65535")
    with open(f"/tmp/cascade-acceptance-{args.port}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True)
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.update(CASCADE_BRIDGE_PORT=str(args.port), CASCADE_OCCUPANCY="1", CASCADE_INSTALL_PROFILE="spark",
            CASCADE_BELIEFS="0", CASCADE_ISAAC_PIXEL_MASK="1", CASCADE_PROOF_CAMERA="1", CASCADE_REQUIRE_CUDA="1",
            YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True", CASCADE_GRASP_MEMORY_PATH=str(out / "grasp-memory.json"))
        os.chdir(ROOT / "models")
        receipt = run(args, out)
    print(out, "PASS" if receipt["pass"] else "FAIL", flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
