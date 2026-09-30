#!/usr/bin/env python3
"""Run real kitchen skills and adjudicate them with the existing passive GPU proof.

Example (from the checkout):
  .venv/bin/python benchmark/diagnostics/kitchen_acceptance.py --port 8681 \
      --engine newton --rounds 1 --output runs/acceptance-nw5

This is a skill-path diagnostic, not the launch/LLM acceptance. Occupancy is
disabled unless --occupancy nvblox is selected; belief persistence is disabled. The
configured grasp planner is preserved, including its logged OBB fallback.
The passive witness does not feed the controller. The production runtime's
existing truth-assisted held-object XY/slip compensation is retained; this
is not a perception-only evaluation. Release height uses FK and calibration.
Output must be a new directory inside this checkout (the passive proof's
existing provenance restriction). No acceptance tolerance is changed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "green_cube": "green square", "orange": "open box",
    "pink_cube": "green square", "lemon": "open box",
    "tomato_can": "green square",
}


def write_json(path, value):
    def encode(obj):
        if hasattr(obj, "as_dict"):
            return obj.as_dict()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=encode) + "\n")


def load_proof():
    spec = importlib.util.spec_from_file_location(
        "independent_kitchen_proof", ROOT / "demo/kitchen/physics/spark_proof.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retarget_ports(node, port):
    count = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "bridge_port":
                node[key] = port
                count += 1
            else:
                count += retarget_ports(value, port)
    elif isinstance(node, list):
        count += sum(retarget_ports(value, port) for value in node)
    return count


def phase(observer, runtime, skill, arguments, phase_name, receipt):
    observer.mark(phase_name + "_begin")
    started = time.monotonic()
    try:
        receipt[phase_name + "_result"] = runtime.execute(skill, arguments)
        occupancy = runtime.arm.harness.occupancy
        if occupancy is not None:
            receipt[phase_name + "_payload_query"] = occupancy.last_payload_query
            receipt[phase_name + "_payload_query_scope"] = (
                "Last map query, including hypothetical route preflight; not necessarily the last "
                "executed pose. commands.jsonl and the passive witness establish actual motion.")
        observer.settle(simulation_seconds=.65, wall_timeout=40)
        observer.capture_frames("placed" if phase_name == "pick" else "reset")
    except Exception:
        receipt["errors"].append(traceback.format_exc())
    finally:
        receipt[phase_name + "_wall_seconds"] = time.monotonic() - started
        observer.mark(phase_name + "_end")


def install_command_trace(runtime, path):
    """Keep every first failure/recovery call; no extra reads or commands.

    The skill-level trace retains only the final retry error. This diagnostic
    records the existing actuator calls and contact-exemption transitions so
    a later recovery refusal cannot hide the original failed stage.
    """
    lock = threading.Lock()
    def record(event):
        event.update(monotonic=time.monotonic(), held=runtime.held_object,
                     provisional=getattr(runtime, "_held_provisional", None),
                     exemption=runtime.arm.harness._grasp_exempt)
        def encode(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            raise TypeError(type(value).__name__)
        with lock, path.open("a") as output:
            output.write(json.dumps(event, default=encode, allow_nan=False) + "\n")
    def wrap(owner, name):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            call_id = time.monotonic_ns()
            public_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            record({"event": "begin", "call": name, "id": call_id,
                    "args": args, "kwargs": public_kwargs})
            try:
                result = original(*args, **kwargs)
            except Exception:
                record({"event": "exception", "call": name, "id": call_id,
                        "traceback": traceback.format_exc()})
                raise
            record({"event": "end", "call": name, "id": call_id, "result": result})
            return result
        setattr(owner, name, call)
    for name in ("move_joints", "set_gripper"):
        wrap(runtime.arm, name)
    for name in ("allow_grasp_descent", "clear_grasp_exemption"):
        wrap(runtime.arm.harness, name)
    original_add = runtime.memory.add
    def add(kind, text, *args, **kwargs):
        record({"event": "memory", "kind": kind, "text": text})
        return original_add(kind, text, *args, **kwargs)
    runtime.memory.add = add


def run_case(args, proof, case_dir, object_name):
    from cascade.apps.demo import build_runtime, shutdown_runtime
    from cascade.config import load_demo_config

    destination = CASES[object_name]
    receipt = {"object": object_name, "destination": destination,
               "requested_engine": args.engine, "port": args.port,
               "pass": False, "errors": [], "diagnostic_profile": {
                   "llm": "mock", "occupancy": args.occupancy, "belief_persistence": False,
                   "fresh_grasp_memory": True,
                   "held_xy_slip_feedback": "production truth lookup with perception fallback",
                   "release_height_feedback": "FK and configured support plane",
                   "cuda_perception_required": os.environ.get("CASCADE_REQUIRE_CUDA") == "1",
                   "acceptance": "independent passive physics proof; skill ok alone is insufficient"}}
    os.environ["CASCADE_GRASP_MEMORY_PATH"] = str(case_dir / "grasp-memory.json")
    runtime = None
    observer = None
    try:
        cfg = load_demo_config(cameras=["isaac", "isaac_side", "isaac_proof"],
                               arm="isaac_kitchen_gpu", llm="mock")
        if not retarget_ports(cfg._data, args.port):
            raise RuntimeError("No bridge_port found in resolved runtime configuration")
        if args.occupancy == "nvblox":
            cfg._data["occupancy"]["port"] = args.occupancy_port
            cfg._data["occupancy"]["track_payload"] = True
            cfg._data["occupancy"]["allowed_contact_paths"] = ["/World_Props/" + name for name in CASES]
            for index, camera_cfg in enumerate(cfg._data["cameras"]):
                camera_cfg["map_depth"] = index < args.map_cameras
            receipt["occupancy_camera_count"] = args.map_cameras
        if args.pregrasp_offset is not None:
            cfg._data["grasp"]["pregrasp_offset_m"] = args.pregrasp_offset
        if args.place_support_clearance is not None:
            ceiling = float(cfg.grasp.get("topdown_carry_z_max", cfg.grasp.get("topdown_z_max", .15))) - .005
            if args.place_support_clearance > ceiling - float(cfg.safety.get("table_z", 0.)):
                raise ValueError("requested support clearance exceeds the configured TCP release ceiling")
            cfg._data["grasp"]["place_support_clearance_m"] = args.place_support_clearance
            receipt["release_gap_scope"] = (
                "place_support_clearance_m is requested controller configuration, not measured release height. "
                "The held support offset and TCP ceiling can limit the effective gap; placed_at records "
                "the commanded TCP pose, and the independent physical audit judges release and settling.")
        receipt["placement_configuration"] = {
            key: cfg.grasp.get(key) for key in (
                "pregrasp_offset_m", "release_height_m", "release_clearance_m", "place_support_clearance_m",
                "source_support_top_z_m",
                "carry_height_m", "pre_carry_lift", "open_box",
                "release_open_timeout_s", "close_feedback_timeout_s")}
        runtime, _ = build_runtime(cfg, case_dir / "runtime", lazy_arm=True)
        install_command_trace(runtime, case_dir / "commands.jsonl")
        if args.occupancy == "nvblox":
            occ = runtime.arm.harness.occupancy
            status = occ.probe(timeout_ms=2000) if occ is not None else None
            if not status or status.get("backend") != "nvblox" or not str(status.get("device", "")).startswith("cuda:"):
                raise RuntimeError(f"Expected a real CUDA nvblox bridge, got {status}")
            receipt["occupancy_probe"] = status

        # Generalize only the witness's two-order constructor. Its snapshot,
        # camera, contact, geometry, release, reset and audit code are inherited
        # unchanged and the shared auditor already supports all five props.
        class Witness(proof.SparkKitchenWitness):
            def __init__(self):
                proof.shared.GpuProofObserver.__init__(
                    self, case_dir / "witness", port=args.port, interval_s=.15,
                    budget_s=900, object_name=object_name,
                    destination_name=destination,
                    expected_scene_geometry=proof.shared.load_expected_scene_geometry(args.scene_config))

        observer = Witness()
        with observer:
            engines = {r["physics"].get("engine") for r in observer.records}
            if engines != {args.engine}:
                raise RuntimeError(f"Requested {args.engine}; observed {sorted(engines)}")
            observer.settle(simulation_seconds=.65, wall_timeout=40)
            try:
                phase(observer, runtime, "pick_and_place",
                      {"object": object_name.replace("_", " "), "destination": destination},
                      "pick", receipt)
            finally:
                # Reset is an explicit real skill, including after a failed pick.
                phase(observer, runtime, "reset_scene", {}, "reset", receipt)
        receipt["physical_audit"] = observer.audit()
        checks = receipt["physical_audit"]["checks"]
        checks["requested_engine_matches_every_sample"] = bool(observer.records) and all(
            r["physics"].get("engine") == args.engine for r in observer.records)
        checks["pick_skill_completed"] = receipt.get("pick_result", {}).get("ok") is True
        checks["reset_skill_completed"] = receipt.get("reset_result", {}).get("ok") is True
        checks["harness_has_no_errors"] = not receipt["errors"]
        if args.occupancy == "nvblox":
            occ = runtime.arm.harness.occupancy
            checks["nvblox_has_fresh_distance_grid"] = occ._grid is not None and not occ.is_stale()
            checks["nvblox_has_no_mapping_error"] = occ.last_error is None and occ._body_error is None
        receipt["pass"] = bool(receipt["physical_audit"]["pass"] and all(checks.values()))
        receipt["physical_audit"]["pass"] = receipt["pass"]
        receipt["failed_checks"] = [key for key, passed in checks.items() if not passed]
        # Store final placement and reset states separately, with separate SI
        # linear/angular velocities. Never use the post-reset pose as placement.
        receipt["phase_final_states"] = {}
        for name in ("pick", "reset"):
            begin, end = [observer.marks[name + suffix]["monotonic"] for suffix in ("_begin", "_end")]
            rows = [r for r in observer.records
                    if r["client_started_monotonic"] >= begin
                    and r["client_finished_monotonic"] <= end]
            if rows:
                receipt["phase_final_states"][name] = rows[-1]["physics"]["props"]
        write_json(observer.out / "gpu-physical-audit.json", receipt["physical_audit"])
    except Exception:
        receipt["errors"].append(traceback.format_exc())
        if observer is not None:
            receipt["observer_errors"] = list(observer.errors)
    finally:
        if runtime is not None:
            receipt["backends"] = runtime.backends()
            occ = runtime.arm.harness.occupancy
            if args.occupancy == "nvblox" and occ is not None:
                receipt["occupancy_final"] = {
                    "stale": occ.is_stale(), "last_error": occ.last_error,
                    "last_masked_pixels": occ.last_masked_px,
                    "has_distance_grid": occ._grid is not None,
                    "replayed_frames_on_last_transition": occ.last_replayed_frames,
                }
            planner = getattr(runtime, "_graspgenx", None)
            receipt["graspgenx"] = {
                "required": bool(cfg.grasp.graspgenx.get("required", False)),
                "status": getattr(planner, "status", None),
                "branch_counts": getattr(planner, "last_branch_counts", {}),
                "last_latency_s": getattr(planner, "last_latency_s", None),
            }
            shutdown_runtime(runtime, getattr(runtime, "arm", None))
        write_json(case_dir / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--engine", choices=("newton", "physx"), required=True)
    parser.add_argument("--occupancy", choices=("none", "nvblox"), default="none")
    parser.add_argument("--occupancy-port", type=int, default=5557)
    parser.add_argument("--pregrasp-offset", type=float, help="Diagnostic lift/approach distance, recorded in every receipt")
    parser.add_argument("--place-support-clearance", type=float,
                        help="Requested diagnostic support gap in metres; runtime TCP ceiling still applies")
    parser.add_argument("--map-cameras", type=int, choices=(2, 3), default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed case and its reset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--scene-config", type=Path, default=ROOT / "demo/scene/kitchen_config.json")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.scene_config = args.scene_config.resolve()
    if not 1 <= args.rounds <= 100 or not 1 <= args.port <= 65535:
        parser.error("rounds must be 1..100 and port 1..65535")
    if args.place_support_clearance is not None and (
            not math.isfinite(args.place_support_clearance) or args.place_support_clearance < 0):
        parser.error("place-support-clearance must be finite and nonnegative")
    if not args.output.is_relative_to(ROOT) or args.output.exists():
        parser.error("output must be a NEW directory inside this checkout")
    if len(set(args.objects)) != len(args.objects):
        parser.error("objects must not contain duplicates")
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.update(CASCADE_BRIDGE_PORT=str(args.port), CASCADE_OCCUPANCY="1" if args.occupancy == "nvblox" else "0",
                      CASCADE_ISAAC_PIXEL_MASK="1", CASCADE_PROOF_CAMERA="1",
                      CASCADE_INSTALL_PROFILE="spark", CASCADE_BELIEFS="0",
                      CASCADE_REQUIRE_CUDA="1",
                      YOLO_OFFLINE="True", ULTRALYTICS_OFFLINE="True")
    # Never race a second copy of this harness on the same bridge.
    with open(f"/tmp/cascade-acceptance-{args.port}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another acceptance harness owns bridge {args.port}")
        proof = load_proof()
        proof.shared.load_expected_scene_geometry(args.scene_config)
        args.output.mkdir(parents=True)
        os.chdir(ROOT / "models")
        sources = ("benchmark/diagnostics/kitchen_acceptance.py",
                   "scripts/isaac_bridge.py", "scripts/isaac_materials.py", "scripts/isaac_runtime.py",
                   "scripts/isaac_self_mask.py", "src/cascade/sim/bridge_client.py",
                   "src/cascade/safety/harness.py", "src/cascade/types.py",
                   "src/cascade/safety/trajectory.py",
                   "src/cascade/control/arm_base.py", "src/cascade/control/mock_arm.py",
                   "src/cascade/skills/runtime.py", "src/cascade/grasping/obb_grasp.py",
                   "src/cascade/grasping/graspgenx_backend.py", "src/cascade/config.py",
                   "src/cascade/apps/demo.py",
                   "src/cascade/perception/grounding.py", "src/cascade/perception/cuda_math.py",
                   "src/cascade/perception/occupancy.py", "src/cascade/perception/occupancy_backends.py",
                   "src/cascade/perception/world.py",
                   "src/cascade/agent/effects.py", "src/cascade/sim/truth.py",
                   "demo/kitchen/physics/gpu_proof_audit.py", "demo/kitchen/physics/convex_geometry.py",
                   "demo/kitchen/physics/destination_entry.py", "demo/kitchen/physics/spark_proof.py",
                   "assets/newton/rebot_gripper_hulls.usda",
                   "configs/arms/isaac_kitchen.yaml", "configs/arms/isaac_kitchen_gpu.yaml")
        source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        campaign = {"pass": False, "source_sha256": source_hashes, "engine": args.engine, "port": args.port,
                    "rounds": args.rounds, "objects": args.objects,
                    "scene_config": str(args.scene_config), "cases": []}
        for round_number in range(1, args.rounds + 1):
            for obj in args.objects:
                case_dir = args.output / f"{obj}_r{round_number}"
                case_dir.mkdir()
                print(f"BEGIN {obj} round={round_number} engine={args.engine}", flush=True)
                with (case_dir / "execution.log").open("w", buffering=1) as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        receipt = run_case(args, proof, case_dir, obj)
                campaign["cases"].append({"object": obj, "round": round_number,
                    "pass": receipt["pass"], "receipt": str(case_dir / "receipt.json"),
                    "failed_checks": receipt.get("failed_checks", []), "errors": receipt["errors"]})
                write_json(args.output / "campaign.json", campaign)
                print(f"{'PASS' if receipt['pass'] else 'FAIL'} {obj} round={round_number} "
                      f"checks={receipt.get('failed_checks', [])} errors={len(receipt['errors'])}", flush=True)
                if args.fail_fast and not receipt["pass"]:
                    break
            if args.fail_fast and not campaign["cases"][-1]["pass"]:
                break
        campaign["source_unchanged"] = all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in source_hashes.items())
        campaign["pass"] = (campaign["source_unchanged"] and len(campaign["cases"]) == args.rounds * len(args.objects)
                             and all(case["pass"] for case in campaign["cases"]))
        write_json(args.output / "campaign.json", campaign)
        return 0 if campaign["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
