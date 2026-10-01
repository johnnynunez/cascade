#!/usr/bin/env python3
"""Rehydrate one recorded failed contact episode and request public reset once.

This diagnostic restores software state from historical command/witness files;
it does NOT claim ordinary process-restart recovery. The original pregrasp,
contact cylinder and provisional subject are unchanged. Stationary feedback is
bound explicitly to a later retained-state capture and checked live to 1 mrad.
Only fresh, real camera/ESDF evidence can authorize the runtime's withdrawal.
No setup motion, fresh grasp, simulator restart, release or prop reset precedes
that guarded withdrawal. Cleanup only closes consumers and Isaac sockets.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from nvblox_camera_recovery import (ROOT, OBJECTS, ActuatorTrace, close_without_motion,
    fresh_camera_checks, fresh_state, phase, reset_window, write_json)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bilateral(physics, prop, *, filters=None):
    import numpy as np
    contact = physics["contacts"]
    forces = np.asarray(contact["jaw_forces_n"], dtype=float)
    counts = np.asarray(contact["jaw_contact_counts"], dtype=int)
    if (contact["channel"] != "physx_gpu_contact_tensor"
            or not contact["device"].startswith("cuda:")
            or contact["sensor_paths"] != [prop]
            or contact["physics_step"] != physics["physics_step"]
            or forces.shape != (2, 3) or not np.isfinite(forces).all()
            or not (np.linalg.norm(forces, axis=1) > .1).all()
            or counts.shape != (2,) or not (counts > 0).all()
            or (filters is not None and contact["filter_paths"] != filters)):
        raise ValueError("retained object lacks same-step measured bilateral CUDA jaw contacts")


def load_episode(campaign_path, case_dir, capture_dir, source_ref):
    """File-only reconstruction; no runtime, bridge or actuator access."""
    import numpy as np
    from cascade.perception.freshness import capture_marker
    from types import SimpleNamespace
    campaign = json.loads(campaign_path.read_text())
    receipt = json.loads((case_dir / "receipt.json").read_text())
    capture = json.loads((capture_dir / "metadata.json").read_text())
    if (campaign.get("source_unchanged") is not True or campaign["pass"] is not False
            or receipt["pass"] is not False or receipt["pick_result"].get("stage") != "grasp"
            or not any(Path(c["receipt"]).resolve() == (case_dir / "receipt.json").resolve()
                       for c in campaign["cases"])):
        raise ValueError("historical campaign does not bind this failed grasp")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", source_ref + "^{commit}"],
                                     cwd=ROOT, text=True).strip()
    for name, digest in campaign["source_sha256"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe source manifest path")
        data = subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("historical source mismatch: " + name)
    rows = read_rows(case_dir / "commands.jsonl")
    begins = {r["id"]: r for r in rows if r.get("event") == "begin"}
    failure = next(r for r in rows if r.get("event") == "exception"
                   and r.get("call") == "move_joints" and r.get("provisional") and r.get("exemption"))
    lift = begins[failure["id"]]
    pre = np.asarray(lift["args"][0], dtype=float)
    successful_approaches = [r for r in rows if r.get("event") == "end"
        and r.get("call") == "move_joints" and r.get("result") is True
        and r["monotonic"] < lift["monotonic"]
        and begins[r["id"]].get("exemption") is None
        and np.array_equal(np.asarray(begins[r["id"]]["args"][0]), pre)]
    if (not successful_approaches or pre.shape != (6,) or not np.isfinite(pre).all()
            or "occupancy unsafe" not in failure["traceback"]
            or "timed out (500 ms)" not in failure["traceback"].lower()
            or lift["provisional"] != failure["provisional"]
            or lift["exemption"] != failure["exemption"]):
        raise ValueError("failed first lift is not bound to its successful original pregrasp/contact scope")
    prop = "/World_Props/" + receipt["object"]
    if (not capture.get("identity") or not capture.get("observer_complete")
            or capture.get("observer_errors") or set(capture["cameras"]) != {"cam0", "side", "proof"}):
        raise ValueError("retained capture is incomplete")
    markers = {}
    for name, item in capture["cameras"].items():
        if sha(capture_dir / (name + ".npz")) != item["sha256"]:
            raise ValueError("retained depth archive hash mismatch")
        marker = capture_marker(SimpleNamespace(capture=item["capture"]))
        if (marker["camera"] != name or marker["source"] != ["127.0.0.1", campaign["port"]]
                or item["capture"]["contact_paths"] != [prop]):
            raise ValueError("retained camera source/contact identity mismatch")
        markers[name] = marker
    held_rows = read_rows(capture_dir / "witness/samples.jsonl")
    if len(held_rows) != capture["sample_count"] or held_rows[-1]["physics"] != capture["last"]:
        raise ValueError("retained physics does not match its raw witness")
    first_after_failure = next(r for r in read_rows(case_dir / "witness/samples.jsonl")
                              if r["client_started_monotonic"] > failure["monotonic"])
    for row in held_rows:
        physics = row["physics"]
        if physics["robot_id"] != markers["cam0"]["robot_id"] or physics["engine"] != campaign["engine"]:
            raise ValueError("retained physics identity changed")
        bilateral(physics, prop, filters=capture["last"]["contacts"]["filter_paths"])
    files = [campaign_path, case_dir / "receipt.json", case_dir / "commands.jsonl",
             case_dir / "witness/samples.jsonl", capture_dir / "metadata.json",
             capture_dir / "witness/samples.jsonl"] + list(capture_dir.glob("*.npz"))
    return {"historical_matching_ref": commit,
        "source_ref_scope": "All manifested files match this commit; hashes do not uniquely identify the historical HEAD.",
        "historical_manifest": campaign["source_sha256"],
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "engine": campaign["engine"], "port": campaign["port"], "object": receipt["object"],
        "provisional": lift["provisional"], "q_pre": pre, "cylinder": lift["exemption"],
        "original_failure": failure, "markers": markers,
        "stationary_feedback_source": "later read-only retained-state capture; not the immediate exception feedback",
        "immediate_failure_q_asset": first_after_failure["physics"]["q"][:6],
        "stationary_physics": capture["last"],
        "settling_delta_rad": float(np.max(np.abs(np.asarray(capture["last"]["q"][:6])
                                                   - first_after_failure["physics"]["q"][:6]))),
        "placement_configuration": receipt["placement_configuration"]}


def validate_live(runtime, physics, historical):
    import numpy as np
    def vector(value, length, label):
        result = np.asarray(value, dtype=float)
        if result.shape != (length,) or not np.isfinite(result).all():
            raise ValueError(label + " must have finite measured coordinates")
        return result
    old = historical["stationary_physics"]
    if (physics["robot_id"] != old["robot_id"] or physics["engine"] != old["engine"]
            or runtime.cfg.arm.bridge_robot_id != old["robot_id"]
            or physics["physics_step"] <= old["physics_step"]):
        raise ValueError("live physics is not the continuing retained scene")
    def static_geometry(value):
        geometry = copy.deepcopy(value["scene_geometry"])
        geometry["convex_collider"]["physx_support"].pop("physics_step", None)
        return geometry
    if (static_geometry(physics) != static_geometry(old)
            or physics["joint_names"] != old["joint_names"]
            or physics["gripper"]["indices"] != old["gripper"]["indices"]
            or physics["spawn_positions_m"] != old["spawn_positions_m"]):
        raise ValueError("live scene geometry, actual jaws or spawn identity changed")
    bilateral(physics, "/World_Props/" + historical["object"], filters=old["contacts"]["filter_paths"])
    actual = vector(physics["q"], 8, "live arm/jaws")
    prior = vector(old["q"], 8, "retained arm/jaws")
    arm_error = float(np.max(np.abs(actual[:6] - prior[:6])))
    jaw_error = float(np.max(np.abs(actual[6:] - prior[6:])))
    if arm_error > .001 or jaw_error > .001:
        raise ValueError(f"live arm/jaws moved from retained capture: {arm_error:.6f} rad, {jaw_error:.6f} m")
    if (physics["gripper"]["indices"] != [6, 7]
            or not np.allclose(vector(physics["gripper"]["q"], 2, "actual jaw feedback"), actual[6:], atol=1e-9)):
        raise ValueError("live jaw feedback is not bound to actual joints")
    if set(physics["props"]) != set(old["props"]):
        raise ValueError("live prop identities changed")
    props = {name: float(np.linalg.norm(vector(p["position_m"], 3, "live prop position")
                        - vector(old["props"][name]["position_m"], 3, "retained prop position")))
             for name, p in physics["props"].items()}
    if not props or max(props.values()) > .001:
        raise ValueError("live props moved from retained capture")
    orientation = {}
    for name, prop in physics["props"].items():
        a = vector(prop["orientation_wxyz"], 4, "live prop orientation")
        b = vector(old["props"][name]["orientation_wxyz"], 4, "retained prop orientation")
        if not (.99 <= np.linalg.norm(a) <= 1.01 and .99 <= np.linalg.norm(b) <= 1.01):
            raise ValueError("invalid retained/live prop quaternion")
        dot = np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))
        orientation[name] = float(2. * np.arccos(np.clip(abs(dot), 0., 1.)))
    if max(orientation.values()) > .001:
        raise ValueError("live prop orientation moved from retained capture")
    q = prior[:6] * vector(runtime.cfg.arm.joint_signs, 6, "configured joint signs")
    measured = vector(runtime.arm.get_state().q, 6, "runtime joint feedback")
    if np.max(np.abs(measured - q)) > .001:
        raise ValueError("runtime feedback does not match independent retained physics")
    return {"max_arm_delta_rad": arm_error, "max_jaw_delta_m": jaw_error,
            "prop_position_delta_m": props, "prop_orientation_delta_rad": orientation, "local_q": q}


def rehydrate(runtime, historical, feedback):
    from cascade.skills import contact_episode
    from cascade.perception.freshness import capture_marker
    import numpy as np
    harness = runtime.arm.harness
    cameras = [c.stream for c in runtime.watcher._cams if c.maps_depth]
    keys, by_stream = {}, {}
    for camera in cameras:
        marker = capture_marker(camera.get_frame())
        expected = historical["markers"].get(marker["camera"])
        if (expected is None or {k: v for k, v in marker.items() if k != "t"}
                != {k: v for k, v in expected.items() if k != "t"} or marker["t"] <= expected["t"]):
            raise ValueError("fresh runtime camera does not match retained producer identity")
        key = harness.occupancy._capture_key(marker)
        keys[marker["camera"]], by_stream[id(camera)] = key, key
    if set(keys) != set(historical["markers"]) or len(cameras) != len(keys):
        raise ValueError("runtime does not have every retained map camera")
    xy, radius, z_min = historical["cylinder"]
    episode = {"arm": runtime.arm, "raw": runtime.arm.raw, "backend": contact_episode._backend(runtime.arm),
        "identity": contact_episode._identity(runtime), "halt_generation": harness._halt_generation,
        "q_pre": historical["q_pre"].copy(), "cylinder": (np.asarray(xy), radius, z_min),
        "expected_paths": ("/World_Props/" + historical["object"],), "q_failure": feedback["local_q"].copy(),
        "barrier": None, "source_identity": (tuple(historical["markers"]["cam0"]["source"]),
            historical["markers"]["cam0"]["robot_id"], historical["markers"]["cam0"]["clock"]),
        "camera_keys": keys, "camera_streams": tuple(cameras), "stream_capture_keys": by_stream,
        "floor_markers": list(historical["markers"].values())}
    runtime._contact_episode = episode
    runtime._held_provisional = tuple(historical["provisional"])
    harness._pending_contact_episode = episode
    harness._contact_scope.value = None
    if harness._grasp_exempt is not None or runtime.held_object:
        raise ValueError("rehydration must not invent global exemption or held promotion")
    return episode


def retained_retreat_phase(observer, runtime, recorder, historical, out, original):
    """Observe the runtime's recovery boundary without changing its commands."""
    import numpy as np
    import time
    previous_phase = recorder.phase
    before = fresh_state(observer, runtime)
    recorder.phase = "original_contact_withdrawal"
    began = time.monotonic()
    observer.mark("original_contact_withdrawal_begin")
    record = {"before": before}
    runtime._diagnostic_contact_retreat = record
    try:
        record["result"] = original(runtime)
        return record["result"]
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        observer.mark("original_contact_withdrawal_end")
        ended = time.monotonic()
        try:
            record["after"] = fresh_state(observer, runtime)
        finally:
            recorder.phase = previous_phase
        record["calls"] = recorder.calls("original_contact_withdrawal")
        q = np.asarray(record["after"]["physics"]["q"][:6]) * np.asarray(runtime.cfg.arm.joint_signs)
        record["pregrasp_error_rad"] = float(np.max(np.abs(q - historical["q_pre"])))
        record["settle_tolerance_rad"] = float(runtime.cfg.arm.get("settle_tol", .02))
        commands = [r for r in read_rows(out / "skills.jsonl") if began <= r["monotonic"] <= ended
                    and r.get("event") == "begin" and r.get("call") == "move_joints"]
        record["motion_requests"] = commands
        after_physics = record["after"]["physics"]
        try:
            bilateral(after_physics, "/World_Props/" + historical["object"],
                      filters=historical["stationary_physics"]["contacts"]["filter_paths"])
            contact_verified = True
        except (ValueError, KeyError, TypeError) as exc:
            contact_verified = False
            record["contact_error"] = str(exc)
        record["object_lift_m"] = float(after_physics["props"][historical["object"]]["position_m"][2]
            - before["physics"]["props"][historical["object"]]["position_m"][2])
        record["checks"] = {
            "observed_original_pregrasp": np.isfinite(record["pregrasp_error_rad"])
                and record["pregrasp_error_rad"] <= record["settle_tolerance_rad"],
            "no_jaw_or_prop_reset_during_retreat": not any(c["layer"] == "actuator"
                and c["call"] in {"set_gripper", "reset_props"} for c in record["calls"]),
            "actual_joint_commands_observed": any(c["layer"] == "actuator"
                and c["call"] == "send_joint_target" for c in record["calls"]),
            "final_target_is_original_pregrasp": bool(commands)
                and np.array_equal(commands[-1]["args"][0], historical["q_pre"]),
            "original_cylinder_throughout_retreat": bool(commands)
                and all(c["exemption"] == historical["cylinder"] for c in commands),
            "runtime_retreat_completed": record.get("result", {}).get("retreated_to_original_pregrasp") is True,
            "same_object_still_in_measured_bilateral_contact": contact_verified,
            "object_lift_observed_before_home_or_release": np.isfinite(record["object_lift_m"])
                and record["object_lift_m"] >= .015,
            "physics_advanced_during_retreat": after_physics["physics_step"] > before["physics"]["physics_step"],
        }
        record["pass"] = all(record["checks"].values())
        if not record["pass"] and "error" not in record:
            from cascade.types import SkillError
            raise SkillError("independent physical contact retreat validation failed")


def rebuilt_camera_map(occupancy, result):
    """Deliveries only count after the same source has entered a successful ESDF."""
    with occupancy._refresh_lock:
        committed = copy.deepcopy(occupancy._integrated_captures)
        floors = result.get("observation_freshness", [])
        checked = []
        for item in floors:
            floor = item["floor"]
            key = occupancy._capture_key(floor)
            entry = committed.get(key)
            checked.append({"camera": floor["camera"], "floor": floor,
                "committed": entry["marker"] if entry else None,
                "pass": bool(entry and entry["marker"]["t"] > floor["t"])})
        return {"cameras": checked, "pass": len(checked) == 3
                and {r["camera"] for r in checked} == {"cam0", "side", "proof"}
                and all(r["pass"] for r in checked)}


def run(args, out):
    from kitchen_acceptance import load_proof, retarget_ports, install_command_trace
    from cascade.apps.demo import build_runtime
    from cascade.config import load_demo_config
    historical = load_episode(args.campaign, args.case, args.capture, args.historical_ref)
    cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_proof"], arm="isaac_kitchen_gpu", llm="mock")
    retarget_ports(cfg._data, historical["port"])
    cfg._data["occupancy"].update(port=args.occupancy_port, track_payload=True,
        allowed_contact_paths=["/World_Props/" + name for name in OBJECTS])
    cfg._data["grasp"].update(historical["placement_configuration"])
    cfg._data["grasp"]["graspgenx"]["port"] = args.grasp_port
    for camera in cfg._data["cameras"]:
        camera["map_depth"] = True
    receipt = {"pass": False, "scope": __doc__, "historical_episode": historical, "errors": [], "phases": {}}
    paths = sorted(set([Path(__file__), ROOT / "benchmark/diagnostics/nvblox_camera_recovery.py",
        ROOT / "benchmark/diagnostics/kitchen_acceptance.py", args.scene_config]
        + list((ROOT / "src/cascade").rglob("*.py")) + list((ROOT / "demo/kitchen/physics").glob("*.py"))
        + list((ROOT / "configs").rglob("*.yaml"))))
    receipt["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    runtime = observer = recorder = None
    try:
        runtime, _ = build_runtime(cfg, out / "runtime", lazy_arm=True)
        install_command_trace(runtime, out / "skills.jsonl")
        recorder = ActuatorTrace(runtime, out / "actuators.jsonl")
        occ = runtime.arm.harness.occupancy
        status = occ.probe(timeout_ms=500)
        receipt["occupancy_probe"] = status
        if (not status or status.get("backend") != "nvblox" or not status.get("masked_depth")
                or not str(status.get("device", "")).startswith("cuda:")):
            raise ValueError("real native masked CUDA nvblox required")
        proof = load_proof()
        expected = proof.shared.load_expected_scene_geometry(args.scene_config)
        observer = proof.shared.GpuProofObserver(out / "witness", port=historical["port"], interval_s=.15,
            budget_s=600, object_name=historical["object"], destination_name="green square", expected_scene_geometry=expected)
        with observer:
            before = fresh_state(observer, runtime)
            with observer._lock:
                complete_physics = copy.deepcopy(next(r["physics"] for r in observer.records
                                                      if r["sequence"] == before["sequence"]))
            feedback = validate_live(runtime, complete_physics, historical)
            receipt["live_binding"] = feedback
            episode = rehydrate(runtime, historical, feedback)
            receipt["setup_sent_no_commands"] = not recorder.calls("setup")
            if not receipt["setup_sent_no_commands"]:
                raise ValueError("setup unexpectedly issued actuator commands")
            if (not all(sha(ROOT / p) == h for p, h in receipt["source_sha256"].items())
                    or not all(sha(ROOT / p) == h for p, h in historical["inputs_sha256"].items())):
                raise ValueError("source or historical evidence changed before reset request")
            from cascade.skills import contact_episode
            original = contact_episode.recover
            contact_episode.recover = lambda rt: retained_retreat_phase(observer, rt, recorder, historical, out, original)
            try:
                reset = phase(observer, runtime, recorder, "rehydrated_reset", lambda: runtime.execute("reset_scene", {}))
            finally:
                contact_episode.recover = original
            receipt["phases"]["reset"] = reset
            receipt["phases"]["contact_retreat"] = getattr(runtime, "_diagnostic_contact_retreat", None)
            receipt["postclose_geometry"] = episode.get("barrier")
            observer.settle(simulation_seconds=.65, wall_timeout=40)
            receipt["final_state"] = fresh_state(observer, runtime)
            with observer._lock:
                records = copy.deepcopy(observer.records)
            receipt["reset_physics"] = reset_window(records,
                after=observer.marks["rehydrated_reset_end"]["monotonic"], cfg=cfg,
                expected_props=expected["prop_dimensions_m"], engine=historical["engine"])
            receipt["camera_map_commits"] = rebuilt_camera_map(occ, reset["result"])
            receipt["checks"] = {
                "setup_sent_no_commands": receipt["setup_sent_no_commands"],
                "public_reset_completed": reset["result"].get("ok") is True,
                "original_contact_retreat_completed": reset["result"].get("contact_recovery", {}).get("retreated_to_original_pregrasp") is True,
                "independent_retreat_and_command_evidence": bool(receipt["phases"]["contact_retreat"]
                    and receipt["phases"]["contact_retreat"]["pass"]),
                "every_reset_camera_committed": receipt["camera_map_commits"]["pass"],
                "reset_physics": receipt["reset_physics"]["pass"],
                "retained_episode_cleared": getattr(runtime, "_contact_episode", None) is None,
                "contact_exemption_cleared": runtime.arm.harness._grasp_exempt is None,
                "fresh_map": occ._grid is not None and not occ.scene_reset_pending and not occ.is_stale()
                             and occ._body_error is None and occ.last_error is None,
                **fresh_camera_checks(reset["result"], {"cam0", "side", "proof"}),
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
            source_unchanged=all(sha(ROOT / p) == h for p, h in receipt["source_sha256"].items()),
            historical_inputs_unchanged=all(sha(ROOT / p) == h for p, h in historical["inputs_sha256"].items()),
            no_diagnostic_errors=not receipt["errors"] and not receipt.get("cleanup_errors"),
            observer_complete=bool(observer and observer.complete and not observer.errors),
            cleanup_sent_no_commands=not receipt.get("cleanup_calls"))
        receipt["pass"] = all(receipt["checks"].values())
        receipt["artifacts_sha256"] = {str(p.relative_to(out)): sha(p)
            for p in out.rglob("*") if p.is_file() and p.name != "receipt.json"}
        write_json(out / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--historical-ref", default="81076d1")
    parser.add_argument("--occupancy-port", type=int, default=25559)
    parser.add_argument("--grasp-port", type=int, default=25556)
    parser.add_argument("--scene-config", type=Path, default=ROOT / "demo/scene/kitchen_config.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    for name in ("campaign", "case", "capture", "scene_config", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if any(not getattr(args, name).is_relative_to(ROOT)
           for name in ("campaign", "case", "capture", "scene_config", "output")) or args.output.exists():
        parser.error("all paths must be inside this checkout; output must be new")
    port = json.loads(args.campaign.read_text())["port"]
    with open(f"/tmp/cascade-acceptance-{port}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True)
        sys.path.insert(0, str(ROOT / "src"))
        os.environ.update(CASCADE_BRIDGE_PORT=str(port), CASCADE_OCCUPANCY="1", CASCADE_INSTALL_PROFILE="spark",
            CASCADE_BELIEFS="0", CASCADE_ISAAC_PIXEL_MASK="1", CASCADE_PROOF_CAMERA="1", CASCADE_REQUIRE_CUDA="1",
            YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True", CASCADE_GRASP_MEMORY_PATH=str(args.output / "grasp-memory.json"))
        os.chdir(ROOT / "models")
        receipt = run(args, args.output)
    print(args.output, "PASS" if receipt["pass"] else "FAIL", flush=True)
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
